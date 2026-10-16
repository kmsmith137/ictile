"""Loads the C++/CUDA kernels in csrc/ and registers them as XLA FFI targets."""

import ctypes
import os

import numpy as np
import jax
import jax.numpy as jnp

_here = os.path.dirname(os.path.abspath(__file__))
_targets = {np.dtype(np.float32): 'ictile_h2_f32', np.dtype(np.float64): 'ictile_h2_f64'}
_libs = {
    # jax platform name -> (library filename, FFI platform name, symbol suffix)
    'cpu': ('libictile_cpu.so', 'cpu', 'Cpu'),
    'gpu': ('libictile_cuda.so', 'CUDA', 'Cuda'),
}

_available = set()
_load_errors = {}


def _load():
    for platform, (libname, ffi_platform, suffix) in _libs.items():
        path = os.path.join(_here, libname)
        if not os.path.exists(path):
            _load_errors[platform] = f'{path} not found (was ictile built with this backend?)'
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            for prec, name in (('F32', _targets[np.dtype(np.float32)]), ('F64', _targets[np.dtype(np.float64)])):
                fn = getattr(lib, f'IctileH2{prec}{suffix}')
                jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(fn), platform=ffi_platform)
            _available.add(platform)
        except Exception as e:     # e.g. missing CUDA driver
            _load_errors[platform] = repr(e)


_load()


def available(platform):
    """True if the custom kernel is available on the given jax platform ('cpu' or 'gpu')."""
    return platform in _available


def load_error(platform):
    return _load_errors.get(platform, None)


def cpu_scratch_size(nside, My, Ky, so_y, Py, Ly, Kx):
    """Per-thread scratch size (in units of the real dtype) for the CPU kernel, where so_y is a numpy
    array (see _tables.gather_tables). Must agree with CpuProblem::scratch_size() in csrc/grf_cpu.cc.
    """
    Ay = len(so_y)
    ws = lambda o: (o // Ay) * Py + int(so_y[o % Ay])
    ys = ws(0) & ~1
    ncols = ws(My - 1 + Ky) - ys
    ncols += (ncols & 1)
    return 2 * (ncols + Kx * (ncols + Ly))


def h2_ffi(kz0, sqrtp, ws_x, so_x, pf_x, so_y, gw_y, phx, phy, phz, *, platform, kz_batch, Mx, My, nside,
           dmin_x, dmin_y, Py, Ky, Ly, cpu_scratch, seed, stream, fixed_amplitude, tab_scale, real_dtype,
           deriv=None):
    """Fused kernel: returns H2[b, k'x, k'y] (shape (kz_batch, Mx, My)) for kz = kz0 + b (see csrc/grf_cuda.cu).

    Tables ws_x, so_x, pf_x, so_y, gw_y are described in _tables.py. On the CPU, cpu_scratch is the
    value of cpu_scratch_size(). If deriv = j (0 <= j < 3), the modes are multiplied by i n_j / n^2
    (see _modes.mode_values()).
    """
    rdt = np.dtype(real_dtype)
    cdt = jnp.complex64 if rdt == np.float32 else jnp.complex128

    # The CPU kernel needs per-thread scratch memory, which we allocate through JAX as an extra
    # output (XLA's CPU runtime doesn't implement the FFI scratch allocator).
    nwork = ((os.cpu_count() or 1) * cpu_scratch) if (platform == 'cpu') else 1

    out_types = (jax.ShapeDtypeStruct((kz_batch, Mx, My), cdt), jax.ShapeDtypeStruct((nwork,), rdt))
    h2, _ = jax.ffi.ffi_call(_targets[rdt], out_types)(
        jnp.asarray(kz0, dtype=jnp.int32), sqrtp, ws_x, so_x, pf_x, so_y, gw_y, phx, phy, phz,
        nside=np.int64(nside), dmin_x=np.int64(dmin_x), dmin_y=np.int64(dmin_y),
        py=np.int64(Py), ky=np.int64(Ky), ly=np.int64(Ly),
        key0=np.int64(seed & 0xFFFFFFFF), key1=np.int64((seed >> 32) & 0xFFFFFFFF),
        stream=np.int64(stream), fixed_amplitude=np.int64(bool(fixed_amplitude)),
        tab_scale=np.float64(tab_scale), deriv=np.int64(-1 if (deriv is None) else deriv))
    return h2
