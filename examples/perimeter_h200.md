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

# For the examples, and the Jupyter kernel (see "Interactive Jupyter" below)
pip install "git+https://github.com/dsjamieson/jax_nbody_emulator.git@55a95dc" "jztree[cuda13]" camb matplotlib jupyterlab h5py
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

## Make many halo catalogs

[`02_make_halo_catalog.py`](02_make_halo_catalog.py) (see `README.md` in this directory) makes one catalog per seed. A
512³ catalog with velocities takes 37 s on an H200, so 100 catalogs take about an hour of GPU time. The preemptible
partition `h200preq` suits this well: it has no caps, a preempted job is requeued automatically, and the script then
skips the catalogs that already exist. For example, two jobs of 50 catalogs each, from `ictile/examples` with the
`ictile` env active, after `mkdir logs` (Slurm doesn't create the log directory):

```bash
#!/bin/bash
#SBATCH -p h200preq
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu=4
#SBATCH -t 01:30:00
#SBATCH --array=0-1
#SBATCH -J halos
#SBATCH -o logs/%x-%A_%a.out
S=$((SLURM_ARRAY_TASK_ID * 50))
python 02_make_halo_catalog.py $S $((S + 50)) --outdir catalogs/
```

`--open-mode=append` keeps the log of a run that was preempted.

## Compare with the Quijote simulations (CPU)

[`03_halo_power_spectra.py`](03_halo_power_spectra.py) computes the power spectra for
[`04_monte_carlo_comparison.ipynb`](04_monte_carlo_comparison.ipynb). It needs [kszx](https://github.com/kmsmith137/kszx)
and h5py but no GPU, so install it in its own env (see kszx's README) and run it on the CPU nodes. The CPU partitions
`amdq` and `defq`, and the preemptible `amdpreq` and `preq`, can all be listed in one job: it runs on whichever
starts it first. Short time limits (`-t`) help a job start early, in gaps between bigger jobs. Keep the files in
`$HOME` or `/mnt/beegfs`: `/gpfs` isn't mounted on the AMD nodes.

**Quijote catalogs.** kszx downloads them with Globus, through a Globus Connect Personal endpoint that you run on the
head node. With the kszx env active (it needs `globus-cli`):

```bash
module load globusconnectpersonal/3.3.0
globus login
globus gcp create mapped "mn003 ($USER)"       # prints a setup key
globusconnectpersonal -setup SETUP_KEY
nohup globusconnectpersonal -start -restrict-paths rw~/ > ~/globus_gcp.log 2>&1 &
python -c "import kszx; kszx.quijote.download('fiducial', 1000, products=('halos',))"     # 34 GB
globusconnectpersonal -stop                     # when the transfer is done
```

`-restrict-paths` limits what the endpoint can write to: add `,rw/mnt/beegfs/$USER` if kszx's data directory
(`$KSZX_DATA_DIR`, or `~/kszx_data`) is there.

**Spectra.** With the kszx env active (the jobs inherit it), from `ictile/examples`, after `mkdir logs`. First the
matching of the two halo samples (~10 minutes):

```bash
sbatch -p amdq,defq,preq,amdpreq --requeue -c 16 --mem=32G -t 00:40:00 -J match -o logs/%x-%j.out --wrap \
  "python 03_halo_power_spectra.py match --emulator-dir catalogs/ --out spectra/matching.json"
```

Then the spectra of 1000 catalogs of each kind, as 40 tasks of 50 catalogs. A catalog takes ~17 s on the AMD nodes
(`amdq`, `amdpreq`) and ~28 s on the Intel nodes (`defq`, `preq`), so a task takes 14 to 23 minutes:

```bash
#!/bin/bash
#SBATCH -p amdq,defq,preq,amdpreq
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH --array=0-39
#SBATCH -J pk
#SBATCH -o logs/%x-%A_%a.out
T=$SLURM_ARRAY_TASK_ID
KIND=$( (( T < 20 )) && echo emulator || echo quijote )
S=$(( (T % 20) * 50 ))
python 03_halo_power_spectra.py compute $KIND $S $((S + 50)) --outdir spectra/ --emulator-dir catalogs/ \
    --matching spectra/matching.json
```

A task that was preempted, or that ran out of time, finishes when resubmitted: the files that exist are skipped. Then
run `04_monte_carlo_comparison.ipynb` with `SPECTRA_DIR = 'spectra'` (a minute, on the head node or in JupyterHub).

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
