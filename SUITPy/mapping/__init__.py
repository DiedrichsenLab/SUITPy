
"""
Functions for mapping and plotting cerebellar data
"""

from .flatmap import (vol_to_surf, make_func_gifti, make_label_gifti, plot)

__all__ = ['vol_to_surf', 'make_func_gifti', 'make_label_gifti', 'plot']