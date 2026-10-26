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
kernel also needs nvcc.

**On a GPU machine**, install with `ICTILE_CUDA=1`, so that a missing or broken nvcc is a build error:

    ICTILE_CUDA=1 pip install --no-build-isolation -e .

By default (`ICTILE_CUDA=auto`), the CUDA kernel is skipped, with a warning that pip only shows with `-v`, if no
working nvcc is found. `ICTILE_CUDA=0` builds for the CPU only.

Other build options (environment variables, see `setup.py`):
- `ICTILE_NVCC`: the path to nvcc, if it is not found automatically.
- `ICTILE_CUDA_ARCH`: the GPU compute capabilities to build for. The default is `"80;86;89;90;100;120"`, minus any
  that nvcc doesn't support, plus PTX for newer GPUs.

`ictile.build_info()` shows how the kernels were built, and whether they loaded.

**On Perimeter's Symmetry cluster** (H200 GPUs), see [`examples/perimeter_h200.md`](examples/perimeter_h200.md).

**At run time**, `TiledGRF(..., backend='auto')` (the default) uses the custom kernel for the device. If the CUDA
kernel is missing, it raises an error on a GPU, rather than falling back to the much slower pure-JAX implementation
(pass `backend='jax'` to use that). On the CPU it falls back with a warning.

To make any problem with the CUDA kernel fail immediately, set `ICTILE_REQUIRE_CUDA=1`, or call
`ictile.require_cuda()`. Importing ictile then fails unless the kernel is available and passes a quick self-test
on every GPU that jax sees. The self-test catches, e.g., a GPU that the kernel wasn't built for.

For tuning the CUDA kernel, `ICTILE_DEBUG=1` prints its launch plan: the thread block shape, which is chosen by a cost
model in `make_plan()` (in `csrc/grf_cuda.cu`). The plan is printed once for each problem size, tile shape, dtype and
GPU, when it is first used.

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

`ntile=nside` is also allowed: every tile is then the whole box, rolled so that `start` is at index 0, computed
with an ordinary FFT of all the modes (in pure JAX). This is for boxes that fit in memory: `tile()` needs memory for
~3 nside^3 floats, and `displacement_tile()` for ~8 nside^3 floats. It's useful when a calculation needs tiles larger
than `nside/2`: compute the whole box once, and cut the tiles from it.

Float32 tiles agree with an ordinary FFT of the full box to ~4e-7 rms (relative to rms(delta)),
and float64 tiles to ~3e-12.

On one NVIDIA L40S in float32, a 1024^3 tile of an 8192^3 box takes 4.2 seconds, or 0.6 hours for
the full box. Every tile has to generate all nside^3 Fourier modes (2.4 of the 4.2 seconds here),
so large tiles are more efficient. On an H200, the same tile takes 3.8 seconds (0.54 hours for the full box).

## Examples

[`examples/01_halo_catalog.ipynb`](examples/01_halo_catalog.ipynb) chains ictile with an N-body emulator and a
friends-of-friends halo finder to make a halo catalog. See [`examples/README.md`](examples/README.md) for how to install
the extra packages.

## Tests

    pytest

GPU tests are skipped if there is no GPU.
