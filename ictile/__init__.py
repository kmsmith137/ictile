"""ictile: tiled simulation of large-scale structure initial conditions.

Main entry point is TiledGRF (Gaussian random field, simulated one tile at a time).
"""

__version__ = '0.0.1'

from .grf import TiledGRF
from ._modes import philox4x32_10, mode_values
from ._ffi import available as custom_kernel_available
