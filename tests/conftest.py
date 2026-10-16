import jax
import pytest

# float64 tests need x64; this must be set before any jax arrays are created.
jax.config.update('jax_enable_x64', True)


def _has_gpu():
    try:
        return len(jax.devices('gpu')) > 0
    except RuntimeError:
        return False


HAS_GPU = _has_gpu()

# Parametrization over jax platforms, with GPU tests skipped (not silently dropped) if no GPU.
PLATFORMS = ['cpu', pytest.param('gpu', marks=pytest.mark.skipif(not HAS_GPU, reason='no GPU'))]


def device(platform):
    return jax.devices(platform)[0]
