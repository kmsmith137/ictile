"""ictile: tiled simulation of large-scale structure initial conditions.

Main entry point is TiledGRF (Gaussian random field, simulated one tile at a time).
"""

__version__ = '0.0.1'

import os as _os

from .grf import TiledGRF
from ._modes import philox4x32_10, mode_values
from ._ffi import available as custom_kernel_available, build_info, require_cuda

# ICTILE_REQUIRE_CUDA=1: fail on import unless the CUDA kernel is available and works on every GPU jax sees.
if _os.environ.get('ICTILE_REQUIRE_CUDA', '0').strip() not in ('', '0'):
    require_cuda()
