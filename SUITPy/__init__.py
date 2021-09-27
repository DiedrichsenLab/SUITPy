"""
Helper functions to download NeuroImaging datasets
"""

from warnings import warn
from .atlas import (fetch_king_2019,
                    fetch_buckner_2011,
                    fetch_diedrichsen_2009,
                    fetch_ji_2019,
                    fetch_xue_2021
                    )

from .flatmap import (vol_to_surf,
                    make_func_gifti,
                    make_label_gifti,
                    get_gifti_column_names,
                    get_gifti_colortable,
                    get_gifti_anatomical_struct,
                    plot)

__all__ = [fetch_king_2019, fetch_buckner_2011, fetch_diedrichsen_2009,
        fetch_ji_2019, fetch_xue_2021, vol_to_surf, make_func_gifti,
        make_label_gifti, get_gifti_column_names, get_gifti_colortable,
        get_gifti_anatomical_struct, plot]

warn("Fetchers from the SUITPy.atlas module will be "
     "updated in later versions as new atlases become available", FutureWarning)