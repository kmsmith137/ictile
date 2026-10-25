# Running ictile on Perimeter's H200 GPUs

How to install ictile and run the example notebook on the H200 GPUs of Perimeter's Symmetry cluster (head node
`mn003`, also called `symmetry.pitp.io`; GPU nodes `cn111` and `cn113`, with 8 × H200 each). Tested on 2026-10-25 with
jax 0.12.0.1 and the GPU nodes' driver 595.45.04 (CUDA 13.2).

The head node has no GPU. Install and build there, and run on a GPU node through Slurm (`module load slurm` first).

## Install (on the head node)

```bash
conda create -n ictile python=3.12 pip
conda activate ictile
pip install "jax[cuda13]"

# ictile, built with the cluster's CUDA 13.2 nvcc. The env remembers these variables, for rebuilds.
conda env config vars set ICTILE_NVCC=/usr/local/cuda-13.2/bin/nvcc ICTILE_CUDA=1
conda activate ictile
git clone https://github.com/kmsmith137/ictile
cd ictile
pip install --no-build-isolation -e .

# For the example notebook, and its Jupyter kernel (see "Interactive Jupyter" below)
pip install "git+https://github.com/dsjamieson/jax_nbody_emulator.git@55a95dc" "jztree[cuda13]" camb matplotlib jupyterlab
python -m ipykernel install --user --name ictile --display-name "Python (ictile)" --env PYTHONPATH ""
```

- **nvcc:** use CUDA 13.2 by its full path, since it matches the GPU nodes' driver. `/usr/local/cuda` on the head node
  is a symlink that has pointed to newer toolkits.
- **GPU architectures:** the default build includes H200 code (compute capability 9.0). Don't set
  `ICTILE_CUDA_ARCH=native` on the head node: it has no GPU, so the build stops with an error.
- **CUDA libraries:** `jax[cuda13]` installs CUDA 13.4 libraries, newer than the driver's 13.2. They work (CUDA's
  minor version compatibility).
- **On the head node,** jax prints `cuInit(0) failed: CUDA_ERROR_NO_DEVICE` and runs on the CPU. That's expected.
- **The Jupyter kernel** has `PYTHONPATH` cleared, because the cluster's Jupyter puts its own Python 3.9 packages on
  `PYTHONPATH`, which breaks a Python 3.12 kernel.

To check the CUDA kernel on a GPU (this should print `OK`):

```bash
srun -p h200debugq --gpus 1 --cpus-per-gpu=4 -t 00:05:00 python -c "import ictile; ictile.require_cuda(); print('OK')"
```

## Run the example notebook as a batch job

With the default parameters, the notebook takes ~3.5 minutes on an H200 (0.25 s per tile, 3× faster than an L40S),
so the debug queue is enough. From `ictile/examples`, with the `ictile` env active (the job inherits it):

```bash
sbatch -p h200debugq --gpus 1 --cpus-per-gpu=4 -t 00:20:00 -J halo-catalog -o %x-%j.out --wrap \
  "jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 01_halo_catalog.ipynb --output 01_halo_catalog.out.ipynb"
```

This writes the executed notebook, with its plots, to `01_halo_catalog.out.ipynb`, and the halo catalog to
`halos.npz`. The run needs ~7 GB of host memory, well within the ~83 GB that comes with `--cpus-per-gpu=4`. For a
bigger box (`NSIDE`), or anything else longer than an hour, use `-p h200q` and a longer `-t`.

## Interactive Jupyter

**JupyterHub** (https://symmetry.pitp.io). Under "Server Options", choose "GPU node - H200 GPU - 1 GPU, 1 hour". Your
JupyterLab then runs on a GPU node with one H200. Open the notebook and pick the kernel "Python (ictile)".
- This uses the `h200q` queue, which is often full for hours. There is no debug-queue option.
- The server holds the GPU for the full hour unless you stop it: File → Hub Control Panel → Stop My Server.

**Your own JupyterLab in a Slurm job**, for the debug queue, or for more than an hour:

```bash
module load slurm
conda activate ictile
salloc --no-shell -p h200debugq --gpus 1 --cpus-per-gpu=4 -t 00:55:00 -J $USER-lab
J=$(squeue -h -u $USER -n $USER-lab -o %i)
srun --jobid=$J --overlap bash -c 'jupyter lab --no-browser --ip=$(hostname) --port=8888'
```

This prints a URL like `http://cn111:8888/lab?token=...` (the port may differ). To open it, forward the port from your
computer through the head node, with `ssh -N -L 8888:cn111:8888 symmetry.pitp.io`, and open the URL with `localhost`
in place of `cn111`. From VS Code connected to the head node (Remote-SSH), you can use the URL as it is: "Select
Kernel" → "Existing Jupyter Server". Run `scancel $J` when you're done.

## Etiquette

The debug queue (`h200debugq`, up to 1 hour) has 2 GPUs for all users together: hold at most one at a time. Ask for
`--cpus-per-gpu=4` (the default of 12 CPUs per GPU can leave `h200q` jobs waiting for CPUs while GPUs are free), and
`scancel` your jobs as soon as you're done.
