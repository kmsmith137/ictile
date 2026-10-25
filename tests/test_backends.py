"""Backend selection, build info, and failing loudly when the CUDA kernel is required but missing."""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import ictile
from ictile import _ffi
from conftest import HAS_GPU, device

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pk = lambda k: np.ones_like(k)
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason='no GPU')


@pytest.fixture
def no_kernel(monkeypatch):
    """Pretends that the custom kernel for a platform failed to load."""
    def hide(platform):
        monkeypatch.setattr(_ffi, '_available', _ffi._available - {platform})
        monkeypatch.setitem(_ffi._load_errors, platform, 'hidden by test')
    return hide


def test_build_info():
    info = ictile.build_info()
    assert set(info) == {'cpu', 'gpu'}
    for platform, rec in info.items():
        assert rec['available'] == ictile.custom_kernel_available(platform)
        assert (rec['error'] is None) == rec['available']
    if info['gpu']['available']:
        assert info['gpu']['archs']


@needs_gpu
def test_auto_backend_on_gpu_requires_kernel(no_kernel):
    no_kernel('gpu')
    with pytest.raises(RuntimeError, match="backend='jax'"):
        ictile.TiledGRF(nside=16, box_size=16.0, pk=pk, seed=1, ntile=8, device=device('gpu'))
    with pytest.raises(RuntimeError, match='hidden by test'):
        ictile.TiledGRF(nside=16, box_size=16.0, pk=pk, seed=1, ntile=8, device=device('gpu'), backend='custom')
    g = ictile.TiledGRF(nside=16, box_size=16.0, pk=pk, seed=1, ntile=8, device=device('gpu'), backend='jax')
    assert g.backend == 'jax'


def test_auto_backend_on_cpu_falls_back(no_kernel):
    no_kernel('cpu')
    with pytest.warns(UserWarning, match='falling back'):
        g = ictile.TiledGRF(nside=16, box_size=16.0, pk=pk, seed=1, ntile=8, device=device('cpu'))
    assert g.backend == 'jax'


def test_auto_backend_with_kernel():
    for platform in ['cpu'] + (['gpu'] if HAS_GPU else []):
        if ictile.custom_kernel_available(platform):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                g = ictile.TiledGRF(nside=16, box_size=16.0, pk=pk, seed=1, ntile=8, device=device(platform))
            assert g.backend == 'custom'


@needs_gpu
def test_require_cuda(no_kernel):
    if not ictile.custom_kernel_available('gpu'):
        pytest.skip('CUDA kernel not built')
    ictile.require_cuda()
    no_kernel('gpu')
    with pytest.raises(RuntimeError, match='CUDA kernel is not available'):
        ictile.require_cuda()


def _import_ictile(**env):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO, os.environ.get('PYTHONPATH', '')]), **env)
    return subprocess.run([sys.executable, '-c', 'import ictile'], cwd=REPO, env=e, capture_output=True, text=True)


def test_require_cuda_env_without_gpu():
    r = _import_ictile(ICTILE_REQUIRE_CUDA='1', CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
    assert r.returncode != 0
    assert ('jax sees no GPU' in r.stderr) or ('CUDA kernel is not available' in r.stderr), r.stderr


@needs_gpu
def test_require_cuda_env_with_gpu():
    if not ictile.custom_kernel_available('gpu'):
        pytest.skip('CUDA kernel not built')
    r = _import_ictile(ICTILE_REQUIRE_CUDA='1')
    assert r.returncode == 0, r.stderr
