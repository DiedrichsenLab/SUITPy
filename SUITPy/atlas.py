"""
Importing Cerebellar atlases and templates from the cerebellar atlas
repository
https://github.com/DiedrichsenLab/cerebellar_atlases

"""
import json
import requests
import os

from SUITPy.utils import _get_dataset_dir, _fetch_files

def fetch_atlas(atlas, atlas_dir=None, maps = 'all', space='all',
                    base_url=None, resume=True, verbose=1):
    """Download and install cerebellar atlas maps from github.com/DiedrichsenLab/cerebellar_atlases

    Args:
        atlas (str): Name of the atlas (Diedrichsen_2009, King_2019, Nettekoven_2024, etc. )
        atlas_dir (str): Base directory of Cerebellar atlases, files will be in atlas_dir/atlas_name/..
        maps (list or str): Which maps to download within the altas (i.e. atl-Buckner7)
        space (str): Volumetric files should be in 'SUIT', 'MNI', or 'MNISym' space (default 'all')
        base_url : string, optional
            base_url of files to download (None results in default base_url).
        resume (bool): REsume download after fail
        verbose (int): Default 1

    Returns:
        data (data dict):
            Dictionary, contains keys:
                - data_dir: Absolute path of downloaded folder
                - files: list of string. Absolute paths of downloaded files on disk.
                - description: A short description of `data` and some references.

    Notes
    -----
    For more details, see
    https://github.com/DiedrichsenLab/cerebellar_atlases
    """
    if base_url is None:
        base_url = ('https://github.com/DiedrichsenLab/cerebellar_atlases/raw/master/')


    # get information from `package_description.json`
    url = base_url + '/package_description.json'
    resp = requests.get(url)
    package_dict = json.loads(resp.text)

    # Check if requested atlas is in package
    atlases = list(package_dict.keys())
    if atlas not in atlases:
        raise(NameError(f'{atlas} is found: Available atlases are {atlases}'))

    # Determine the download directory
    atlas_dir = _get_dataset_dir(atlas,atlas_dir)


    # get map names and description
    atlas_dict = package_dict[atlas]
    fdescr = atlas_dict['ShortDesc']

    # get space for volumes
    if space=='all':
        space =atlas_dict['Spaces']
    elif isinstance(space,str):
        space = [space]
    for s in space:
        if s not in atlas_dict['Spaces']:
            raise(NameError(f'{s} is found: Available spaces for {atlas} are {atlas_dict["Spaces"]}'))

    # Get names of different maps
    if maps=='all':
        maps = atlas_dict['Maps']
    elif isinstance(maps,str):
        maps = [maps]
    for m in maps:
        if m not in atlas_dict['Maps']:
            raise(NameError(f'{m} is found: Available maps for {atlas} are {atlas_dict["Maps"]}'))

    # Generale the list of all possible files
    at_ex = ['_dseg.label.gii','.lut']
    con_ex = ['.func.gii']
    file_names = ['atlas_description.json']
    for m in maps:
        if m[:3]=='atl':
            extensions = at_ex
        elif m[:3]=='con':
            extensions = con_ex
        for ex in extensions:
            file_names.append(m+ex)
        for s in space:
            if m[:3]=='atl':
                file_names.append(f'{m}_space-{s}_dseg.nii')
            elif m[:3]=='con':
                file_names.append(f'{m}_space-{s}.nii')

    files = []
    for f in file_names:
        files.append((f, base_url + '/' + atlas + '/' + f, {}))

    # get local fullpath(s) of downloaded file(s)
    fpaths = _fetch_files(atlas_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': atlas_dir,
                'files': fpaths,
                'description': fdescr})

