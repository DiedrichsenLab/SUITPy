"""
Helper functions to download NeuroImaging datasets
"""
import gzip

__version__ = '2.0.1'

from .atlas import (fetch_atlas,summarize_data)

from .flatmap import (vol_to_surf,
                    save_colorbar,
                    plot)

from .reslice import (reslice_image,
                      reslice_img)

from .isolation import (isolate, registration)

from .normalization import (normalize, bounding_box, deformation_from_displacement)

# Monkey-patch gzip to have faster reads on large gzip files
if hasattr(gzip.GzipFile, 'max_read_chunk'):
    gzip.GzipFile.max_read_chunk = 100 * 1024 * 1024  # 100Mb

__all__ = [fetch_atlas, summarize_data,vol_to_surf, plot,isolate,normalize,
        reslice_image, reslice_img, save_colorbar, bounding_box,deformation_from_displacement]
