"""
Helper functions to download NeuroImaging datasets
"""

from warnings import warn
from .atlas import (fetch_king_2019, fetch_buckner_2011,
                    fetch_diedrichsen_2009, fetch_ji_2019,
                    fetch_xue_2021)


from .utils import get_data_dirs

__all__ = [fetch_king_2019, fetch_buckner_2011,
           fetch_diedrichsen_2009, fetch_ji_2019,
           fetch_xue_2021
           ]

warn("This module is experimental", FutureWarning)
