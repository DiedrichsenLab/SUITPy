"""
Downloading Cerebellum NeuroImaging datasets: atlas datasets
"""
import os
import warnings
import xml.etree.ElementTree
from tempfile import mkdtemp
import json
import shutil
import requests

import nibabel as nb
import numpy as np
from numpy.lib import recfunctions

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr, _fetch_file
from .._utils import fill_doc

@fill_doc
def fetch_king_2019(data='con', space='SUIT', data_dir=None, 
                    base_url=None, resume=True, verbose=1,
                    ):
    
    """"Download and return file names for the King et al. (2019) atlas
    or contrast images set by `data`

    NOT CONFIGURED YET FOR data='atl'. Naming convention is not final in `cerebellar_atlases`

    The provided images are in `space` (SUIT or MNI)

    Parameters
    ----------
    data : str, optional
        Options are 'atl', 'con'
        Default='atl'
    space : str, optional
        Options are 'SUIT', 'MNI'
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - files: list of string. 
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.
    References
    ----------
    .. footbibliography::
    Notes
    -----
     For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/King_2019
    Licence: MIT.
    """
    valid_spaces = ['SUIT', 'MNI']
    valid_data = ['atl', 'con']

    if space not in valid_spaces:
        raise ValueError(f'Requested {space} not available. Valid options: {valid_spaces}')

    if data not in valid_data:
        raise ValueError(f'Requested {data} not available. Valid options: {valid_data}')

    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/King_2019')

    dataset_name = 'king_2019'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)
    
    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    'dseg.label.gii'
    'dseg.nii'
    '.func.gii'

    # filter map names (and add suffixes)
    maps_full = [f'{m}_space-{space}.nii' for m in maps if data in m]
    maps_gii = [f'{m}_space-{space}.func.gii' for m in maps if data in m]
    maps_full.extend(maps_gii)

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'files': fpaths, 
            'description': fdescr})

@fill_doc
def fetch_buckner_2011(space='SUIT', data_dir=None, base_url=None, 
                    resume=True, verbose=1,
                    ):
    
    """"Download and return file names for the Buckner et al. (2011) atlas

    The provided images are in space SUIT and MNI and in nifti and gifti formats

    Parameters
    ----------
    space : str, optional
        Options are 'SUIT', 'MNI'
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - files: list of string. 
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.
    References
    ----------
    .. footbibliography::
    Notes
    -----
     For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Buckner_2011
    Licence: MIT.
    """
    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/Buckner_2011')

    dataset_name = 'buckner_2011'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)
    
    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    # build maps
    maps_full = [f'{m}_space-{space}_dseg.nii' for m in maps]

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'files': fpaths, 
            'description': fdescr})

@fill_doc
def fetch_diedrichsen_2009():
    pass

@fill_doc
def fetch_ji_2019():
    pass

@fill_doc
def fetch_xue_2021():
    pass