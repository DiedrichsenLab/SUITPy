"""
Downloading NeuroImaging datasets: atlas datasets
"""
import os
import warnings
import xml.etree.ElementTree
from tempfile import mkdtemp
import json
import shutil

import nibabel as nb
import numpy as np
from numpy.lib import recfunctions
from sklearn.utils import Bunch

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr
from .._utils import check_niimg, fill_doc

_TALAIRACH_LEVELS = ['hemisphere', 'lobe', 'gyrus', 'tissue', 'ba']

@fill_doc
def fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1,
                              data_dir=None, base_url=None, resume=True,
                              verbose=1):
    """Download and return file names for the Schaefer 2018 parcellation
    .. versionadded:: 0.5.1
    The provided images are in MNI152 space.
    For more information on this dataset, see :footcite:`schaefer_atlas`,
    :footcite:`Schaefer2017parcellation`,
    and :footcite:`Yeo2011organization`.
    Parameters
    ----------
    n_rois : int, optional
        Number of regions of interest {100, 200, 300, 400, 500, 600,
        700, 800, 900, 1000}.
        Default=400.
    yeo_networks : int, optional
        ROI annotation according to yeo networks {7, 17}.
        Default=7.
    resolution_mm : int, optional
        Spatial resolution of atlas image in mm {1, 2}.
        Default=1mm.
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:
        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: ROI labels including Yeo-network annotation,list of strings.
        - description: A short description of the atlas and some references.
    References
    ----------
    .. footbibliography::
    Notes
    -----
    Release v0.14.3 of the Schaefer 2018 parcellation is used by
    default. Versions prior to v0.14.3 are known to contain erroneous region
    label names. For more details, see
    https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/Updates/Update_20190916_README.md
    Licence: MIT.
    """
    valid_n_rois = list(range(100, 1100, 100))
    valid_yeo_networks = [7, 17]
    valid_resolution_mm = [1, 2]
    if n_rois not in valid_n_rois:
        raise ValueError("Requested n_rois={} not available. Valid "
                         "options: {}".format(n_rois, valid_n_rois))
    if yeo_networks not in valid_yeo_networks:
        raise ValueError("Requested yeo_networks={} not available. Valid "
                         "options: {}".format(yeo_networks,valid_yeo_networks))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm)
                         )

    if base_url is None:
        base_url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/'
                    'v0.14.3-Update_Yeo2011_Schaefer2018_labelname/'
                    'stable_projects/brain_parcellation/'
                    'Schaefer2018_LocalGlobal/Parcellations/MNI/'
                    )

    files = []
    labels_file_template = 'Schaefer2018_{}Parcels_{}Networks_order.txt'
    img_file_template = ('Schaefer2018_{}Parcels_'
                         '{}Networks_order_FSLMNI152_{}mm.nii.gz')
    for f in [labels_file_template.format(n_rois, yeo_networks),
              img_file_template.format(n_rois, yeo_networks, resolution_mm)]:
        files.append((f, base_url + f, {}))

    dataset_name = 'schaefer_2018'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    labels_file, atlas_file = _fetch_files(data_dir, files, resume=resume,
                                           verbose=verbose)

    labels = np.genfromtxt(labels_file, usecols=1, dtype="S", delimiter="\t")
    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(maps=atlas_file,
                 labels=labels,
                 description=fdescr)