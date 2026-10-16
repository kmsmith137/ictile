"""Pure-JAX implementation of the Gaussian random field modes, and of the fused
(generate + regrid) step. This is the reference for the C++/CUDA kernels in csrc/, and must
agree with csrc/grf_common.h. See "Random numbers" in Appendix A of the notes.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import _tables

_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF


def _mulhilo32(a, b):
    """(hi, lo) 32-bit words of a*b, where a is a Python int and b is a uint32 array."""
    lo = jnp.uint32(a) * b
    a0, a1 = a & 0xFFFF, a >> 16
    b0, b1 = b & 0xFFFF, b >> 16
    p00, p01 = jnp.uint32(a0) * b0, jnp.uint32(a0) * b1
    p10, p11 = jnp.uint32(a1) * b0, jnp.uint32(a1) * b1
    mid = (p00 >> 16) + (p01 & 0xFFFF) + (p10 & 0xFFFF)
    hi = p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16)
    return hi, lo


def philox4x32_10(c0, c1, c2, c3, key):
    """Philox4x32-10 on uint32 arrays (c0,c1,c2,c3), with 64-bit integer key. Returns 4 uint32 arrays."""
    k0, k1 = key & _MASK, (key >> 32) & _MASK
    for r in range(10):
        if r > 0:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo32(_M0, c0)
        hi1, lo1 = _mulhilo32(_M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ jnp.uint32(k0), lo1, hi0 ^ c3 ^ jnp.uint32(k1), lo0
    return c0, c1, c2, c3


def fixed_point_dtype(real_dtype):
    """Phases are fixed-point fractions of a turn: uint32 for float32, uint64 for float64."""
    return jnp.uint32 if np.dtype(real_dtype) == np.float32 else jnp.uint64


def phase_table(nside, rc, real_dtype):
    """Fixed-point phase exp(2 pi i k rc / nside) for unsigned 0 <= k < nside, where rc is a
    (traced) integer scalar with 0 <= rc < nside. Requires nside <= 2^16.
    """
    udt = fixed_point_dtype(real_dtype)
    nbits = 32 if udt == jnp.uint32 else 64
    q, r = divmod(2**nbits, nside)
    k = jnp.arange(nside, dtype=jnp.uint32)
    m = ((k * jnp.asarray(rc).astype(jnp.uint32)) % jnp.uint32(nside)).astype(udt)
    return m * udt(q) + (m * udt(r) + udt(nside//2)) // udt(nside)


def _u32(x):
    return lax.bitcast_convert_type(jnp.asarray(x).astype(jnp.int32), jnp.uint32)


def mode_values(nx, ny, nz, *, nside, seed, stream, sqrtp, tab_scale, fixed_amplitude, real_dtype, phase=None, deriv=None):
    """Fourier modes a(k) of the Gaussian random field (delta(r) = sum_k a(k) e^{ik.r}).

    Arguments nx, ny, nz are signed int32 wavenumbers (broadcastable), with -nside/2 <= n_i < nside/2
    and 0 <= nz < nside/2. If 'phase' is specified, it is a fixed-point phase (see phase_table())
    which multiplies the result. If deriv = j (0 <= j < 3), the result is multiplied by i n_j / n^2,
    which gives the linear displacement psi_j = -d_j del^{-2} delta, up to a factor box_size/(2 pi).
    Returns a complex array.
    """
    rdt = np.dtype(real_dtype)
    udt = fixed_point_dtype(rdt)
    cdt = jnp.complex64 if rdt == np.float32 else jnp.complex128
    nx, ny, nz = (jnp.asarray(n, dtype=jnp.int32) for n in (nx, ny, nz))
    h = nside // 2

    canonical = (nz > 0) | (nx > 0) | ((nx == 0) & (ny > 0))
    cx = jnp.where(canonical, nx, -nx)
    cy = jnp.where(canonical, ny, -ny)
    x0, x1, x2, x3 = philox4x32_10(_u32(cx), _u32(cy >> 1), _u32(nz), jnp.uint32(stream), seed)
    odd = (cy & 1) == 1
    xa = jnp.where(odd, x2, x0)
    xb = jnp.where(odd, x3, x1)

    # Amplitude: sqrt(P/V) from the table (linear in t = log2(n^2)), times sqrt(-ln u1).
    n2 = (_u32(nx*nx) + _u32(ny*ny) + _u32(nz*nz)).astype(rdt)
    t = jnp.log2(jnp.maximum(n2, 1)) * rdt.type(tab_scale)
    i = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, sqrtp.shape[0] - 2)
    f = t - i.astype(rdt)
    amp = sqrtp[i] + f * (sqrtp[i+1] - sqrtp[i])
    if not fixed_amplitude:
        if rdt == np.float32:
            u1 = xa.astype(rdt) * rdt.type(2.**-32) + rdt.type(2.**-33)
        else:
            u1 = (xa.astype(rdt) + 0.5) * 2.**-32
        amp = amp * jnp.sqrt(-jnp.log(u1))
    if deriv is not None:
        amp = amp * ((nx, ny, nz)[deriv].astype(rdt) / jnp.maximum(n2, 1))

    # Phase: fixed-point turns u2 = (xb + 1/2)/2^32, conjugated if non-canonical, plus extra phase.
    if udt == jnp.uint32:
        tu = xb
        nbits, sdt = 32, jnp.int32
    else:
        tu = (xb.astype(jnp.uint64) << 32) | jnp.uint64(0x80000000)
        nbits, sdt = 64, jnp.int64
    tu = jnp.where(canonical, tu, udt(0) - tu)
    if phase is not None:
        tu = tu + phase
    if deriv is not None:
        tu = tu + udt(1 << (nbits - 2))    # factor i
    turns = lax.bitcast_convert_type(tu, sdt).astype(rdt) * rdt.type(2.**-nbits)
    theta = rdt.type(2*np.pi) * turns

    zero = (nx == -h) | (ny == -h) | (n2 == 0)
    return jnp.where(zero, 0, lax.complex(amp * jnp.cos(theta), amp * jnp.sin(theta))).astype(cdt)


def regrid_axis(x, axis, M, dmin, ws):
    """Regrid along one axis: ambient (length nside, FFT order) -> bounding box (length M).

    out[n'] = sum_{n} W(n,n') x[n], where n*M/nside = Q + p/P contributes to n' = (Q + dmin + k) mod M
    with weight ws[p,k] (see _tables.py).
    """
    nside = x.shape[axis]
    K = ws.shape[1]
    Q, p = _tables.axis_index(np.arange(nside), nside, M)
    x = jnp.moveaxis(x, axis, -1)
    out = jnp.zeros(x.shape[:-1] + (M,), dtype=x.dtype)
    for k in range(K):
        # Elementwise products and scatter-add (not einsum), to avoid TF32 on GPU.
        out = out.at[..., (Q + dmin + k) % M].add(x * ws[p, k])
    return jnp.moveaxis(out, -1, axis)


def h2_jax(kz0, *, kz_batch, nside, Mx, My, dmin_x, dmin_y, ws_x, ws_y, phx, phy, phz, **gen):
    """Pure-JAX version of the fused kernel: returns H2[b, k'x, k'y] for kz = kz0 + b.

    Each kz plane is generated in full (shape (nside, nside)), so this is only practical for
    small nside. Keyword arguments 'gen' are passed to mode_values().
    """
    kz = kz0 + jnp.arange(kz_batch, dtype=jnp.int32)
    valid = kz < nside//2
    kzc = jnp.where(valid, kz, 0)
    k = jnp.arange(nside, dtype=jnp.int32)
    n = jnp.where(k >= nside//2, k - nside, k)
    phase = phx[None, :, None] + phy[None, None, :] + phz[kzc][:, None, None]
    a = mode_values(n[None, :, None], n[None, None, :], kzc[:, None, None], nside=nside, phase=phase, **gen)
    a = jnp.where(valid[:, None, None], a, 0)
    a = regrid_axis(a, -1, My, dmin_y, ws_y)
    return regrid_axis(a, -2, Mx, dmin_x, ws_x)
