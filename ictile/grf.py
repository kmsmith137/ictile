"""Tiled simulation of Gaussian random fields (tiled-FFT algorithm 2 in the notes)."""

import functools
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from . import _ffi
from . import _modes
from . import _tables

_HIGHEST = lax.Precision.HIGHEST


class TiledGRF:
    """Gaussian random field in a periodic box, which can be simulated one tile at a time.

    The field is defined by its Fourier modes (see Appendix A of the notes, "Random numbers"):

        delta(r) = sum_k a(k) e^{i k.r},     <|a(k)|^2> = P(k) / V,

    where r is a pixel position, k = 2*pi*n/box_size, and V = box_size^3. Each mode a(k) is a
    deterministic function of (n, seed), computed with the Philox4x32-10 counter-based RNG, so any
    tile can be simulated independently. Nyquist modes are zero, so modes with |n_i| < nside/2
    do not depend on nside.

    The linear (Zel'dovich) displacement psi = -grad(del^{-2} delta), i.e. psi(k) = i k a(k) / k^2,
    is available through displacement_tile().

    Each tile is a box of (ntile_x, ntile_y, ntile_z) pixels with lower corner 'start' (wrapping
    around the periodic box). The tile is computed using ES regridding with kernel width w, so it
    agrees with the full box to ~1e-6 in float32 (w=8), or ~1e-11 in float64 (w=13), relative to
    rms(delta).

    Constructor arguments:

      nside: number of pixels per side of the box (even, <= 65536).
      box_size: box side length (same length units as 1/k in the power spectrum).
      pk: power spectrum, a vectorized function of k (a float64 numpy array), which returns P(k).
      seed: integer, 0 <= seed < 2^64.
      ntile: tile shape, either an integer (for a cubical tile) or a tuple (ntile_x, ntile_y, ntile_z),
         with 1 <= ntile_i <= nside/2.
      dtype: np.float32 or np.float64 (float64 requires jax_enable_x64).
      w: ES kernel width (default 8 in float32, 13 in float64).
      kz_batch: number of kz values processed per iteration (memory ~ kz_batch * 4 * ntile_x * ntile_y).
      backend: 'custom' (C++/CUDA kernel), 'jax' (pure JAX, only practical for small nside), or 'auto'
         (default): 'custom' if available. Otherwise, on the CPU, 'jax' with a warning; on a GPU, an error
         (the pure-JAX backend is much slower there, so pass backend='jax' explicitly to use it).
      device: jax device (default: jax.devices()[0]).
      fixed_amplitude: if True, |a(k)| = sqrt(P(k)/V) exactly ("fixed" initial conditions).
      stream: integer in [0, 2^32), for generating independent fields with the same seed.
      ntab: size of the sqrt(P(k)) interpolation table.
    """

    def __init__(self, nside, box_size, pk, seed, ntile, *, dtype=np.float32, w=None, beta=None,
                 kz_batch=16, backend='auto', device=None, fixed_amplitude=False, stream=0, ntab=4096):

        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float64):
            raise ValueError("ictile.TiledGRF: dtype must be float32 or float64")
        if (self.dtype == np.float64) and (jnp.zeros(1, dtype=jnp.float64).dtype != jnp.float64):
            raise RuntimeError("ictile.TiledGRF: float64 requires jax.config.update('jax_enable_x64', True)")
        if (nside % 2) or (nside < 4) or (nside > 2**16):
            raise ValueError(f"ictile.TiledGRF: nside={nside} must be even, and 4 <= nside <= 65536")
        nt = np.asarray(ntile, dtype=int)
        nt = np.full(3, int(nt)) if (nt.ndim == 0) else nt
        if (nt.shape != (3,)) or np.any(nt < 1) or np.any(2*nt > nside):
            raise ValueError(f"ictile.TiledGRF: ntile={ntile} must be an integer or 3 integers, with 1 <= ntile <= nside/2 (nside={nside})")
        if not (0 <= seed < 2**64) or not (0 <= stream < 2**32):
            raise ValueError("ictile.TiledGRF: need 0 <= seed < 2^64, and 0 <= stream < 2^32")

        self.nside = int(nside)
        self.box_size = float(box_size)
        self.seed = int(seed)
        self.stream = int(stream)
        self.ntile = tuple(int(n) for n in nt)
        self.M = tuple(2*n for n in self.ntile)
        self.P = tuple(_tables.axis_phases(self.nside, m)[1] for m in self.M)    # number of fractional offsets
        self.w = int(w) if (w is not None) else (8 if (self.dtype == np.float32) else 13)
        self.beta = float(beta) if (beta is not None) else (2.30 * self.w)
        self.kz_batch = int(kz_batch)
        self.fixed_amplitude = bool(fixed_amplitude)
        self.cdtype = np.dtype(np.complex64 if (self.dtype == np.float32) else np.complex128)
        self.device = device if (device is not None) else jax.devices()[0]

        # Per-axis regridding weights and de-apodization factors (axes may have different P).
        dmin, ws, psi = [], [], []
        for i, (nt_i, M_i, P_i) in enumerate(zip(self.ntile, self.M, self.P)):
            d, wt = _tables.polyphase_weights(P_i, self.w, self.beta)
            if (i == 0) and (wt.shape[1] > 16):
                raise ValueError(f"ictile.TiledGRF: kernel width w={self.w} is too large (number of taps must be <= 16)")
            h = nt_i // 2
            dmin.append(d)
            ws.append(wt)
            psi.append(_tables.deapodization(np.arange(-h, nt_i - h), M_i, d, wt))
        self.dmin = tuple(dmin)
        self.K = tuple(wt.shape[1] for wt in ws)

        so_x, pf_x, _, _ = _tables.gather_tables(self.nside, self.M[0], self.dmin[0], ws[0])
        so_y, _, self.Ly, gw_y = _tables.gather_tables(self.nside, self.M[1], self.dmin[1], ws[1])
        self._cpu_scratch = _ffi.cpu_scratch_size(self.nside, self.M[1], self.K[1], so_y, self.P[1], self.Ly, self.K[0])
        sqrtp, self.tab_scale = _tables.sqrtp_table(pk, self.nside, self.box_size, ntab)
        k0, Ap, Am = _tables.zregrid_windows(self.nside, self.ntile[2], self.dmin[2], ws[2], self.kz_batch)
        self.nbatch = len(k0)

        platform = self.device.platform
        if backend == 'auto':
            if _ffi.available(platform):
                backend = 'custom'
            elif platform == 'gpu':
                # On a GPU the pure-JAX backend is far slower, so silently falling back to it is almost never wanted.
                raise RuntimeError(_ffi.unavailable_message(platform) + " To use the (much slower) pure-JAX "
                                   "implementation on the GPU anyway, pass backend='jax'.")
            else:
                backend = 'jax'
                warnings.warn(f"ictile: custom kernel not available on platform '{platform}' "
                              f"({_ffi.load_error(platform)}), falling back to the (slow) pure-JAX backend")
        elif (backend == 'custom') and not _ffi.available(platform):
            raise RuntimeError(_ffi.unavailable_message(platform))
        elif backend not in ('custom', 'jax'):
            raise ValueError(f"ictile.TiledGRF: backend='{backend}' must be 'auto', 'custom', or 'jax'")
        self.backend = backend

        put = lambda x, dt: jax.device_put(np.asarray(x, dtype=dt), self.device)
        self._tables = dict(
            ws_x=put(ws[0], self.dtype),
            ws_y=put(ws[1], self.dtype),
            so_x=put(so_x, np.int32),
            pf_x=put(pf_x, np.int32),
            so_y=put(so_y, np.int32),
            gw_y=put(gw_y, self.dtype),
            sqrtp=put(sqrtp, self.dtype),
            inv_psi_x=put(1.0 / psi[0], self.dtype),
            inv_psi_y=put(1.0 / psi[1], self.dtype),
            inv_psi_z=put(1.0 / psi[2], self.dtype),
            Ap=put(Ap, self.dtype),
            Am=put(Am, self.dtype),
            k0=put(k0, np.int32),
        )

        self._tile_jit = jax.jit(self._tile_impl, static_argnames=('displacement',))
        self._h2_jit = jax.jit(self._h2_impl, static_argnames=('backend', 'deriv'))
        self._full_jit = jax.jit(self._full_box_impl, static_argnames=('deriv',))

    # ------------------------------------------------------------------------------------------

    def tile(self, start):
        """Returns the tile with lower corner 'start' (3 integers), as an array with shape ntile.

        Element [i,j,k] of the tile is delta at pixel (start + (i,j,k)) mod nside.
        """
        return self._tile_jit(self._start(start), self._tables)

    def displacement_tile(self, start):
        """Returns the linear displacement psi = -grad(del^{-2} delta) on the tile with lower corner 'start',
        as an array with shape (3,) + ntile, in the same length units as box_size.

        Element [j,i,k,l] is psi_j at pixel (start + (i,k,l)) mod nside. The three components are
        computed one at a time, so this takes 3 times as long as tile(), and needs memory for the
        output plus the scratch space of one call to tile().
        """
        return self._tile_jit(self._start(start), self._tables, displacement=True)

    def full_box(self):
        """Returns the full box (shape (nside,)*3), using an ordinary FFT. For testing at small nside."""
        return self._full_jit(self._tables)

    def full_box_displacement(self):
        """Returns the displacement psi in the full box (shape (3,) + (nside,)*3). For testing at small nside."""
        return jnp.stack([self._full_jit(self._tables, deriv=j) for j in range(3)])

    def h2(self, kz0, start=(0, 0, 0), backend=None, deriv=None):
        """Returns the regridded array H2[b, k'x, k'y] (shape (kz_batch, Mx, My)) for kz = kz0 + b (for testing).

        If deriv = j (0 <= j < 3), the modes are multiplied by i n_j / n^2 (see _modes.mode_values()).
        """
        kz0 = jax.device_put(np.int32(kz0), self.device)
        return self._h2_jit(kz0, self._start(start), self._tables, backend=(backend or self.backend), deriv=deriv)

    # ------------------------------------------------------------------------------------------

    def _start(self, start):
        start = np.asarray(start, dtype=np.int64).reshape(3) % self.nside
        return jax.device_put(start.astype(np.int32), self.device)

    def _gen_kwargs(self, tables, deriv=None):
        return dict(seed=self.seed, stream=self.stream, sqrtp=tables['sqrtp'], tab_scale=self.tab_scale,
                    fixed_amplitude=self.fixed_amplitude, real_dtype=self.dtype, deriv=deriv)

    def _phases(self, start):
        return [_modes.phase_table(self.nside, (start[i] + self.ntile[i] // 2) % self.nside, self.dtype) for i in range(3)]

    def _h2(self, kz0, phases, tables, backend, deriv):
        phx, phy, phz = phases
        geom = dict(kz_batch=self.kz_batch, nside=self.nside, Mx=self.M[0], My=self.M[1],
                    dmin_x=self.dmin[0], dmin_y=self.dmin[1])
        if backend == 'custom':
            return _ffi.h2_ffi(kz0, tables['sqrtp'], tables['ws_x'], tables['so_x'], tables['pf_x'],
                               tables['so_y'], tables['gw_y'], phx, phy, phz,
                               platform=self.device.platform, Py=self.P[1], Ky=self.K[1], Ly=self.Ly,
                               cpu_scratch=self._cpu_scratch, seed=self.seed, stream=self.stream,
                               fixed_amplitude=self.fixed_amplitude, tab_scale=self.tab_scale,
                               real_dtype=self.dtype, deriv=deriv, **geom)
        return _modes.h2_jax(kz0, ws_x=tables['ws_x'], ws_y=tables['ws_y'], phx=phx, phy=phy, phz=phz,
                             **geom, **self._gen_kwargs(tables, deriv))

    def _h2_impl(self, kz0, start, tables, backend, deriv):
        return self._h2(kz0, self._phases(start), tables, backend, deriv)

    def _restrict(self, x, axis, i):
        """Restrict bounding-box index (mod M_i) along 'axis' to tile offsets t in [-ntile_i/2, ntile_i/2)."""
        nt, M = self.ntile[i], self.M[i]
        h = nt // 2
        return jnp.concatenate([lax.slice_in_dim(x, M - h, M, axis=axis),
                                lax.slice_in_dim(x, 0, nt - h, axis=axis)], axis=axis)

    def _tile_impl(self, start, tables, displacement=False):
        """Returns delta on the tile (shape ntile), or the displacement psi (shape (3,) + ntile).

        The displacement components are computed one at a time, in a loop (so that they share
        scratch memory), and written directly into the output.
        """
        ntx, nty, ntz = self.ntile
        Mz = self.M[2]
        phases = self._phases(start)
        ncomp = 3 if displacement else 1
        z = jnp.int32(0)

        def h2(ib, j):
            kz0 = ib * self.kz_batch
            if not displacement:
                return self._h2(kz0, phases, tables, self.backend, None)
            # Displacement component psi_j: the modes are multiplied by i n_j/n^2 in the kernel.
            return lax.switch(j, [functools.partial(self._h2, kz0, phases, tables, self.backend, d) for d in range(3)])

        # De-apodization in x,y,z (and for the displacement, the factor box_size/(2 pi) from i k_j/k^2).
        ipy, ipz = tables['inv_psi_y'], tables['inv_psi_z']
        if displacement:
            ipz = ipz * (self.box_size / (2*np.pi))
        cx = _chunk_size(ntx, nty * Mz * self.dtype.itemsize, 2**28)

        def component(j, out):
            def body(ib, G):
                # Step 2(a), first part: generate modes, regrid kx -> k'x and ky -> k'y.
                H2 = h2(ib, j)
                # Step 2(a), second part: 2-d FFT, restrict to tile (de-apodization in x,y is done at the end).
                F = jnp.fft.ifft2(H2, norm='forward')
                F = self._restrict(self._restrict(F, 1, 0), 2, 1)
                # Step 2(b): regrid kz -> k'z, accumulating into G[x, y, k'z].
                Ap = lax.dynamic_index_in_dim(tables['Ap'], ib, keepdims=False)
                Am = lax.dynamic_index_in_dim(tables['Am'], ib, keepdims=False)
                k0 = tables['k0'][ib]
                Wn = Ap.shape[-1]
                Gr = jnp.einsum('bxy,bk->xyk', F.real, Ap, precision=_HIGHEST)
                Gi = jnp.einsum('bxy,bk->xyk', F.imag, Am, precision=_HIGHEST)
                win = lax.dynamic_slice(G, (z, z, k0), (ntx, nty, Wn))
                return lax.dynamic_update_slice(G, win + lax.complex(Gr, Gi), (z, z, k0))

            G = jnp.zeros((ntx, nty, ntz + 1), dtype=self.cdtype)
            G = lax.fori_loop(0, self.nbatch, body, G)

            # Step 3: z-FFT, restrict to tile, de-apodize, write to out[j]. Done in chunks of x, to save memory.
            def zfinal(ic, out):
                x0 = (ic * cx).astype(jnp.int32)
                g = lax.dynamic_slice_in_dim(G, x0, cx, axis=0)
                ipx = lax.dynamic_slice_in_dim(tables['inv_psi_x'], x0, cx)
                f = jnp.fft.irfft(g, n=Mz, axis=-1, norm='forward')
                f = self._restrict(f, 2, 2) * ipx[:, None, None] * ipy[None, :, None] * ipz[None, None, :]
                return lax.dynamic_update_slice(out, f[None], (j.astype(jnp.int32), x0, z, z))

            return lax.fori_loop(0, ntx // cx, zfinal, out)

        out = jnp.zeros((ncomp,) + self.ntile, dtype=self.dtype)
        out = lax.fori_loop(0, ncomp, component, out)
        return out if displacement else out[0]

    def _full_box_impl(self, tables, deriv=None):
        N = self.nside
        k = jnp.arange(N, dtype=jnp.int32)
        n = jnp.where(k >= N//2, k - N, k)
        nz = jnp.arange(N//2, dtype=jnp.int32)
        a = _modes.mode_values(n[:, None, None], n[None, :, None], nz[None, None, :], nside=N, **self._gen_kwargs(tables, deriv))
        a = jnp.pad(a, ((0, 0), (0, 0), (0, 1)))
        f = jnp.fft.irfftn(a, s=(N, N, N), norm='forward')
        return f if (deriv is None) else (f * (self.box_size / (2*np.pi)))


def _chunk_size(n, bytes_per_row, max_bytes):
    """Largest divisor c of n such that c * bytes_per_row <= max_bytes (or 1)."""
    best = 1
    for c in range(1, n + 1):
        if (n % c == 0) and (c * bytes_per_row <= max_bytes):
            best = c
    return best
