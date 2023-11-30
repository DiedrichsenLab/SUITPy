"""
Helper functions to download NeuroImaging datasets
"""
import gzip

__version__ = '1.3.2'

from .atlas import (fetch_king_2019,
                    fetch_buckner_2011,
                    fetch_diedrichsen_2009,
                    fetch_ji_2019,
                    fetch_xue_2021
                    )

from .flatmap import (vol_to_surf,
                    save_colorbar,
                    plot)

from .reslice import (reslice_image,
                      reslice_img)

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

__all__ = [fetch_king_2019, fetch_buckner_2011, fetch_diedrichsen_2009,
        fetch_ji_2019, fetch_xue_2021, vol_to_surf, plot]
