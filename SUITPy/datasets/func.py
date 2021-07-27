"""
Downloading NeuroImaging datasets: functional datasets (task + resting-state)
"""
import fnmatch
import glob
import warnings
import os
import re
import json

import nibabel as nib
import numpy as np
import numbers

from io import BytesIO

import nibabel
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab.miobase import MatReadError
from sklearn.utils import Bunch, deprecated

from .utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr,
                    _read_md5_sum_file, _tree, _filter_columns, _fetch_file, _uncompress_file)
from .._utils import check_niimg, fill_doc
from .._utils.numpy_conversions import csv_to_array
from nilearn.image import get_data

@fill_doc
def fetch_openneuro_dataset_index(data_dir=None,
                                  dataset_version='ds000030_R1.0.4',
                                  verbose=1):
    """Download a file with OpenNeuro :term:`BIDS` dataset index.
    Downloading the index allows to explore the dataset directories
    to select specific files to download. The index is a sorted list of urls.
    Parameters
    ----------
    %(data_dir)s
    dataset_version : string, optional
        Dataset version name. Assumes it is of the form [name]_[version].
        Default='ds000030_R1.0.4'.
    %(verbose)s
    Returns
    -------
    urls_path : string
        Path to downloaded dataset index.
    urls : list of string
        Sorted list of dataset directories.
    """
    data_prefix = '{}/{}/uncompressed'.format(dataset_version.split('_')[0],
                                              dataset_version,
                                              )
    data_dir = _get_dataset_dir(data_prefix, data_dir=data_dir,
                                verbose=verbose)

    file_url = 'https://osf.io/86xj7/download'
    final_download_path = os.path.join(data_dir, 'urls.json')
    downloaded_file_path = _fetch_files(data_dir=data_dir,
                                        files=[(final_download_path,
                                                file_url,
                                                {'move': final_download_path}
                                                )],
                                        resume=True
                                        )
    urls_path = downloaded_file_path[0]
    with open(urls_path, 'r') as json_file:
        urls = json.load(json_file)
    return urls_path, urls