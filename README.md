# ictile

Tiled simulation of Gaussian random fields in JAX (e.g. initial conditions for large-scale
structure). The field is generated one tile at a time, so that boxes much larger than GPU memory
(e.g. 8192^3) can be simulated. Every tile is a piece of the same realization, which is a
deterministic function of the seed.

The algorithm is "tiled-FFT algorithm 2" in the notes *NUFFT + ES + Cosmology = Fun!* (see
Appendix A for implementation details). Code comments refer to these as "the notes".

## Install

    pip install --no-build-isolation -e .

This compiles a C++ kernel (CPU) and a CUDA kernel (GPU), which are called from JAX through the
XLA FFI. It needs jax, jaxlib (which supplies the XLA FFI headers), and a C++17 compiler. The CUDA
kernel needs nvcc, and is skipped if no working nvcc is found. See `setup.py` for environment
variables (`ICTILE_NVCC`, `ICTILE_CUDA_ARCH`, ...).

## Example

```python
import numpy as np
import ictile

pk = lambda k: ...    # power spectrum P(k), vectorized over a numpy array of k
grf = ictile.TiledGRF(nside=8192, box_size=16384.0, pk=pk, seed=1, ntile=1024, dtype=np.float32)
delta = grf.tile((0, 1024, 2048))    # shape (1024, 1024, 1024), lower corner at pixel (0, 1024, 2048)
psi = grf.displacement_tile((0, 1024, 2048))    # shape (3, 1024, 1024, 1024)
```

`displacement_tile()` returns the linear (Zel'dovich) displacement psi = -grad(del^{-2} delta), in
the same length units as `box_size`. The three components are computed one at a time, so it takes
3 times as long as `tile()`, and needs memory for the output plus the scratch space of one `tile()`.

Tiles can have any shape `ntile=(nx, ny, nz)` with `1 <= n_i <= nside/2`. For speed, prefer tiles
which are longer along z (the per-kz work scales with nx*ny).

Float32 tiles agree with an ordinary FFT of the full box to ~4e-7 rms (relative to rms(delta)),
and float64 tiles to ~3e-12.

On one NVIDIA L40S in float32, a 1024^3 tile of an 8192^3 box takes 4.2 seconds, or 0.6 hours for
the full box. Every tile has to generate all nside^3 Fourier modes (2.4 of the 4.2 seconds here),
so large tiles are more efficient.

## Examples

[`examples/01_halo_catalog.ipynb`](examples/01_halo_catalog.ipynb) chains ictile with an N-body emulator and a
friends-of-friends halo finder to make a halo catalog. See [`examples/README.md`](examples/README.md) for how to install
the extra packages.

## Tests

    pytest

GPU tests are skipped if there is no GPU.
