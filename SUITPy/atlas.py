"""
Downloading Cerebellum neuroImaging datasets: atlas datasets

@author: maedbhking

A lot of the functionality was based on `nilearn.datasets.atlas`
https://github.com/nilearn/nilearn/blob/main/nilearn/datasets/atlas.py`
"""
import json
import requests

import nibabel as nb
import numpy as np

from SUITPy.utils import _get_dataset_dir, _fetch_files
from SUITPy._utils import fill_doc

@fill_doc
def fetch_king_2019(data='con', data_dir=None,
                    base_url=None, resume=True, verbose=1,
                    ):
    """Download and return file names for the King et al. (2019) atlas or contrast images set by `data`.
    The provided images are in SUIT and MNI spaces

    Parameters
    ----------
    data : str, optional
        Options are 'atl', 'con'
        Default='atl'
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - data_dir: Absolute path of downloaded folder
        - files: list of string.
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.

    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/King_2019
    """

    valid_data = ['atl', 'con']

    if data=='atl':
        suffixes = ['_dseg.label.gii', '_space-SUIT_dseg.nii'] # '_space-MNI_dseg.nii'
    elif data=='con':
        suffixes = ['.func.gii', '_space-SUIT.nii'] # '_space-MNI.nii'

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

    # get filename for maps
    maps_filter = [m for m in maps if data in m]
    maps_full = []
    for map in maps_filter:
        for suffix in suffixes:
            maps_full.append(f'{map}{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})

@fill_doc
def fetch_buckner_2011(data_dir=None, base_url=None,
                    resume=True, verbose=1,
                    ):
    """Download and return file names for the Buckner et al. (2011) atlas
    The provided images are in SUIT and MNI spaces

    Parameters
    ----------
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - data_dir: Absolute path of downloaded folder
        - files: list of string.
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.

    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Buckner_2011
    """

    suffixes = ['desc-confid_space-SUIT.nii', 'dseg.label.gii', 'space-MNI_dseg.nii', 'space-SUIT_dseg.nii']

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

    # get filename for maps
    maps_full = []
    for map in maps:
        for suffix in suffixes:
            maps_full.append(f'{map}_{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})

@fill_doc
def fetch_diedrichsen_2009(data_dir=None, base_url=None,
                    resume=True, verbose=1):
    """Download and return file names for the Diedrichsen et al. (2009) atlas

    The provided images are in SUIT and MNI spaces

    Parameters
    ----------
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - data_dir: Absolute path of downloaded folder
        - files: list of string.
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.

    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Diedrichsen_2009
    """
    suffixes = ['dseg.label.gii', 'space-MNI_dseg.nii', 'space-SUIT_dseg.nii']

    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/Diedrichsen_2009')

    dataset_name = 'diedrichsen_2009'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)

    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    # get filename for maps
    maps_full = []
    for map in maps:
        if 'desc-confid' in map:
                suffixes = ['space-SUIT.nii', 'space-MNI.nii']
        else:
            suffixes = ['dseg.label.gii', 'space-MNI_dseg.nii', 'space-SUIT_dseg.nii']
        for suffix in suffixes:
            maps_full.append(f'{map}_{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})

@fill_doc
def fetch_ji_2019(data_dir=None, base_url=None,
                    resume=True, verbose=1):
    """Download and return file names for the Ji et al. (2019) atlas
    The provided images are in SUIT and MNI spaces

    Parameters
    ----------
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s
    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - data_dir: Absolute path of downloaded folder
        - files: list of string.
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.

    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Ji_2019
    """

    suffixes = ['dseg.label.gii', 'space-MNI_dseg.nii', 'space-SUIT_dseg.nii']

    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/Ji_2019')

    dataset_name = 'ji_2019'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)

    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    # get filename for maps
    maps_full = []
    for map in maps:
        for suffix in suffixes:
            maps_full.append(f'{map}_{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})

@fill_doc
def fetch_xue_2021(data_dir=None, base_url=None,
                    resume=True, verbose=1):
    """"Download and return file names for the Xue et al. (2021) atlas

    The provided images are in SUIT and MNI spaces

    Parameters
    ----------
    %(data_dir)s
    base_url : string, optional
        base_url of files to download (None results in default base_url).
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : data dict
        Dictionary, contains keys:
        - data_dir: Absolute path of downloaded folder
        - files: list of string.
            Absolute paths of downloaded files on disk.
        - description: A short description of `data` and some references.

    Notes
    -----
     For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases/tree/master/Xue_2021
    """

    suffixes = ['dseg.label.gii', 'space-MNI_dseg.nii', 'space-SUIT_dseg.nii']

    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/Xue_2021')

    dataset_name = 'xue_2021'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # get maps from `atlas_description.json`
    url = base_url + '/atlas_description.json'
    resp = requests.get(url)
    data_dict = json.loads(resp.text)

    # get map names and description
    maps = data_dict['Maps']
    fdescr = data_dict['LongDesc']

    # get filename for maps
    maps_full = []
    for map in maps:
        for suffix in suffixes:
            maps_full.append(f'{map}_{suffix}')

    files = []
    for f in maps_full:
        files.append((f, base_url + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': data_dir,
                'files': fpaths,
                'description': fdescr})
