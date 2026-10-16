Use the `tile` conda env for Python and compilers. Agent Bash calls don't trigger direnv, so prefix commands with `source .envrc &&` (from the repo root).

- Build the C++/CUDA kernels in place: `python setup.py build_ext --inplace` (the user installs with `pip install --no-build-isolation -e .`; the conda env is read-only inside the agent sandbox).
- Tests: `pytest` from the repo root (about 3.5 minutes; GPU tests are skipped if there is no GPU).
- `ictile/csrc/grf_common.h` and `ictile/_modes.py` both define the random field realization (Philox counters, Box-Muller, Hermitian symmetry); they must stay in sync, and `tests/test_ictile.py::test_h2_custom_vs_jax` checks this.
- Nsight Compute works in the sandbox: `/usr/local/cuda/bin/ncu --kernel-name regex:h2_kernel ...`.
- `ICTILE_DEBUG=1` prints the CUDA kernel's launch plan (thread block shape, chosen by a cost model in `make_plan()`).
- "The notes" in code comments are the Overleaf notes cloned in `../tiling_experiments/extern/extern/overleaf_nufft_ics` (Appendix A is `tiled_impl.tex`). Code comments cite them by algorithm and step number, and by equation label.
- `../tiling_experiments/chain/` uses ictile (tiled ICs -> N-body emulator -> FoF), so keep the `TiledGRF` API compatible or update it there.
