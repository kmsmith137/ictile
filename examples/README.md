# ictile examples

| | what it is |
|---|---|
| [`01_halo_catalog.ipynb`](01_halo_catalog.ipynb) | Initial conditions (ictile) → nonlinear displacements (N-body emulator) → friends-of-friends halos (jz-tree) → a halo catalog and its mass function. 1024³ particles in a 2 Gpc/*h* box, on one GPU. |
| [`quijote_hmf/`](quijote_hmf) | Halo counts from the Quijote N-body simulations, which the notebook's mass function is compared with, and the script that made them from the Quijote catalogs. |

The example uses two other packages besides ictile:

- **[jax_nbody_emulator](https://github.com/dsjamieson/jax_nbody_emulator)**: the field-level N-body emulator of
  Jamieson et al. ([arXiv:2408.07699](https://arxiv.org/abs/2408.07699)).
- **[jz-tree](https://github.com/jstuecker/jztree)** (`jztree`): GPU friends-of-friends and tree algorithms in JAX
  ([arXiv:2604.05885](https://arxiv.org/abs/2604.05885)).

## Requirements

- **Hardware and drivers:** Linux, and an NVIDIA GPU with a recent driver.
- **GPU memory:** with the notebook's default `TILE = 128`, ~40 GB, mostly while the emulator is being compiled.
  With `TILE = 64`, ~16 GB.
- **Python:** 3.11 or newer.
- **Compilers:** a C++17 compiler (e.g. g++), and the CUDA compiler `nvcc` (CUDA 12.8 or newer for the newest GPUs),
  both to build ictile's kernels.
  - If you don't have a CUDA toolkit, the `nvcc` that pip installs with `jax[cuda13]` works (step 3).

## Installation

The commands below set up a fresh environment with CUDA 13. They were tested from scratch with Python 3.12.

**CUDA 12:** replace `cuda13` with `cuda12` throughout, and use a CUDA 12 toolkit's `nvcc`. The pip packages for
CUDA 12 don't include `nvcc`.

```bash
# 1. A new Python environment (a venv here; conda works too)
python3 -m venv ~/venvs/ictile
source ~/venvs/ictile/bin/activate
pip install --upgrade pip setuptools

# 2. JAX with GPU support (see https://docs.jax.dev/en/latest/installation.html)
pip install "jax[cuda13]"
python -c "import jax; print(jax.devices())"      # should list CudaDevice(id=0), ...

# 3. ictile, which compiles a C++ kernel and a CUDA kernel
git clone https://github.com/kmsmith137/ictile
cd ictile
# export ICTILE_NVCC=...          # only if nvcc is not found (see below)
ICTILE_CUDA=1 pip install --no-build-isolation -e .     # ICTILE_CUDA=1: fail if the CUDA kernel can't be built
cd ..

# 4. The N-body emulator (pinned to the version this example was tested with)
pip install "git+https://github.com/dsjamieson/jax_nbody_emulator.git@55a95dc"

# 5. jz-tree: prebuilt wheels, no compilation needed
pip install "jztree[cuda13]"

# 6. For the notebook
pip install camb matplotlib jupyterlab
```

**Notes on step 3:**

- **GPU architectures.** By default the CUDA kernel is compiled for compute capabilities 8.0, 8.6, 8.9, 9.0, 10.0 and
  12.0, which covers A100, A40/A6000, RTX 30xx/40xx/50xx, L40S, H100/H200 and B200 GPUs. Newer GPUs are covered
  through PTX.
  - An nvcc older than CUDA 12.8 doesn't know 10.0 and 12.0, so those are left out, with a message.
  - To build for other GPUs, set `ICTILE_CUDA_ARCH`, e.g. `"75"` for T4, or `"native"` for the GPU in this machine.
- **Finding nvcc.** ictile looks for `nvcc` in `$ICTILE_NVCC`, then `$CUDA_HOME/bin`, then your `PATH`, then
  `/usr/local/cuda/bin`. On a cluster, `module load cuda` usually puts it on your `PATH`. Otherwise, use the `nvcc`
  that came with `jax[cuda13]`:

  ```bash
  export ICTILE_NVCC=$(python -c "import nvidia, os; print(os.path.join(nvidia.__path__[0], 'cu13', 'bin', 'nvcc'))")
  ```
- **`ICTILE_CUDA=1`** makes the install fail if no working nvcc is found. Without it, the build would skip the CUDA
  kernel with only a warning that pip hides, and ictile would then refuse to run on the GPU.

## Check the installation and run the example

```bash
cd ictile/examples
python -c "import ictile, jax_nbody_emulator, jztree, camb; print('ictile CUDA kernel:', ictile.custom_kernel_available('gpu'))"
```

This should print `ictile CUDA kernel: True`. Then open the notebook:

```bash
jupyter lab 01_halo_catalog.ipynb
```

Or run it without a browser, e.g. in a batch job. This saves the outputs into the notebook and writes `halos.npz`:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_halo_catalog.ipynb
```

The parameters are in the notebook's "Global parameters" cell: `NSIDE` (box size), `SEED`, `REDSHIFT`, `TILE`, the
linking length and the cosmology.

## Troubleshooting

- **`ictile CUDA kernel: False`, or "ictile: the CUDA kernel is not available".** The message says why;
  `python -c "import ictile; print(ictile.build_info())"` shows how ictile was built. Usually nvcc was not found:
  set `ICTILE_NVCC` (see above) and re-run `ICTILE_CUDA=1 pip install --no-build-isolation -e .` in the `ictile`
  directory.
- **An error about "no kernel image is available".** The CUDA kernel wasn't built for this GPU. The message gives the
  GPU's compute capability and the `ICTILE_CUDA_ARCH` setting to rebuild with; changing it triggers a rebuild.
- **`AttributeError: module 'ictile' has no attribute ...`.** Python found the `ictile` *repository* directory
  instead of the installed package. This happens when Python runs in the directory that contains the clone. Run
  from somewhere else, e.g. `ictile/examples`.
- **Out of memory while the emulator compiles** (the first tile). Use a smaller `TILE` in the notebook.
  - Compiling needs ~1.7× the memory of running, because XLA tries several convolution algorithms.
  - The notebook lets JAX use 90% of the GPU's memory (`XLA_PYTHON_CLIENT_MEM_FRACTION`). Lower that if you share the
    GPU.
- **"Could not get kernel mode driver version" and similar log messages** from jax are harmless. Set
  `TF_CPP_MIN_LOG_LEVEL=3` to silence them.
