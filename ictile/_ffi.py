"""Loads the C++/CUDA kernels in csrc/ and registers them as XLA FFI targets."""

import ctypes
import json
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
_build_keys = {'cpu': 'cpu', 'gpu': 'cuda'}     # jax platform -> key in _build_info.json (written by setup.py)


def _read_build_info():
    try:
        with open(os.path.join(_here, '_build_info.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


_build_info = _read_build_info()


def _load():
    for platform, (libname, ffi_platform, suffix) in _libs.items():
        path = os.path.join(_here, libname)
        if not os.path.exists(path):
            rec = _build_info.get(_build_keys[platform], {})
            _load_errors[platform] = rec.get('reason') or f'{path} not found (was ictile built?)'
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


def unavailable_message(platform):
    """Explains why the custom kernel isn't available on a jax platform, and what to do."""
    if platform == 'gpu':
        return (f'ictile: the CUDA kernel is not available: {load_error(platform)}. To build it, make nvcc available '
                f'(see ICTILE_NVCC in setup.py), then reinstall ictile with ICTILE_CUDA=1, which makes a missing or broken '
                f'nvcc a build error. See ictile.build_info().')
    return f"ictile: the custom kernel is not available on platform '{platform}': {load_error(platform)}. See ictile.build_info()."


def build_info():
    """How ictile's kernels were built (from _build_info.json, written by setup.py), and whether they loaded.

    Returns a dict with keys 'cpu' and 'gpu' (jax platform names). Each value is a dict with 'available'
    (bool), 'error' (why it isn't available, or None), and the recorded build configuration, e.g. 'archs'
    (compute capabilities) and 'nvcc' for the CUDA kernel."""
    return {platform: dict(available=available(platform), error=load_error(platform),
                           **_build_info.get(_build_keys[platform], {}))
            for platform in _libs}


_checked_devices = set()


def require_cuda(devices=None):
    """Raises RuntimeError unless the CUDA kernel is available and runs correctly on the given jax devices
    (default: all GPUs that jax sees; it is an error if there are none).

    Runs a tiny problem on each device (once per device per process), so that problems such as a GPU
    architecture that the kernel wasn't built for show up immediately, with an explanation.
    Called on import if the environment variable ICTILE_REQUIRE_CUDA=1."""
    if not available('gpu'):
        raise RuntimeError(unavailable_message('gpu'))
    if devices is None:
        try:
            devices = jax.devices('gpu')
        except RuntimeError as e:
            raise RuntimeError(f'ictile: CUDA required, but jax sees no GPU (jax.devices() = {jax.devices()}): {e}') from e
    from .grf import TiledGRF
    for d in devices:
        if d in _checked_devices:
            continue
        if d.platform != 'gpu':
            raise RuntimeError(f'ictile: CUDA required, but {d} is not a GPU')
        try:
            g = TiledGRF(nside=16, box_size=16.0, pk=lambda k: np.ones_like(k), seed=1, ntile=8, backend='custom', device=d)
            x = np.asarray(g.tile((0, 0, 0)))
        except Exception as e:
            cc = getattr(d, 'compute_capability', '?')
            raise RuntimeError(f'ictile: the CUDA kernel failed a self-test on {d} ({d.device_kind}, compute capability {cc}), '
                               f'built for compute capabilities {build_info()["gpu"].get("archs")}: {e}') from e
        if not np.all(np.isfinite(x)) or not np.any(x != 0):
            raise RuntimeError(f'ictile: the CUDA kernel failed a self-test on {d}: the output is all zeros or not finite')
        _checked_devices.add(d)


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
