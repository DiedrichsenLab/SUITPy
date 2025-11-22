"""
Importing Cerebellar atlases and templates from the cerebellar atlas
repository
https://github.com/DiedrichsenLab/cerebellar_atlases

"""

import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
import requests
import SUITPy.utils as utils

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
    atlas_dir = utils._get_atlas_dir(atlas,atlas_dir)


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
    fpaths = utils._fetch_files(atlas_dir, files, resume=resume, verbose=verbose)

    return dict({'data_dir': atlas_dir,
                'files': fpaths,
                'description': fdescr})


def _read_lut(lut_path):
    """
    Read a simple LUT file: each non-empty, non-comment line
    starts with an integer label followed by the region name.
    """
    lut = {}
    with open(lut_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                lab = int(parts[0])
            except ValueError:
                continue
            name = " ".join(parts[1:]) if len(parts) > 1 else f"region {lab}"
            lut[lab] = name
    return lut


def summarize_data(
    images,
    label_image = None,
    atlas=None,
    maps=None,
    space="SUIT",
    atlas_dir=None,
    stats=("mean", "std", "abs"),
    region_names=None,
    outfilename=None,
    verbose=1):

    """ Summarize the data from the images by the ROIs defined in the atlas map. Works optimally together with the files provided in the cerebellar atlas repository

    Args:
        images (list): List of str. Absolute paths of images to summarize
        label_image (str or Nift1Image): Image file with with ROI/Atlas definition
        atlas (str): Name of the atlas (Diedrichsen_2009, King_2019, Nettekoven_2024, etc. )
        maps (str): Name of the map within the atlas (atl-Buckner7, atl-Anatomical)
        space (str): Default 'SUIT', space for the volumetric atlas file: 'SUIT', 'MNI', 'MNISym', etc.
        atlas_dir (str): Base directory of Cerebellar atlases, files will be in atlas_dir/atlas_name/..
        stats (sequence of str): Default ('nanmean'). Which statistics to compute inside each ROI. Supported keys: 'mean', 'nanmean', 'std', 'nanstd', 'max', 'min', 'median', 'abs'.
        region_names (sequence of str or None): Optional list of region names. If provided and length >= number of non-zero labels, it overrides names from the LUT.
        outfilename (str or None): If not None, write the resulting table as a tab-delimited text file.
    Returns:
        summary (df : pandas.DataFrame):
            Dictionary, contains keys:
                - image: Name of the image files summarized
                - region_id: list of int. IDs of the regions in the atlas map
                - region_name: list of str. Names of the regions in the atlas map, defined in the .lut file
                - values: statistics in each region for each image
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    def _load_img(img_obj):
        if isinstance(img_obj, str):
            return nib.load(img_obj), os.path.basename(img_obj)
        elif hasattr(img_obj, "get_fdata"):
            return img_obj, "<nibabel_image>"
        else:
            raise TypeError("Images must be file paths or nibabel NIfTI images.")

    if label_image is None:
        my_atlas_dir = utils._get_atlas_dir(atlas,atlas_dir,)
        if my_atlas_dir is None:
            raise(NameError(f'{atlas} not found. Set atlas_dir correctly or call suit.fetch_atlas {atlas}.'))
        map_image_name  = os.path.join(my_atlas_dir,f"{maps}_space-{space}_dseg.nii")
        lut_file_name =  os.path.join(my_atlas_dir,f"{maps}.lut")

    if not os.path.isfile(map_image_name):
        raise FileNotFoundError(
            f"Could not find label image for map '{maps}' in space '{space}'. "
            "Make sure this combination exists in cerebellar_atlases."
        )

    # Use read_lut in neuroimaging tools
    if os.path.isfile(lut_file_name):
        lut = _read_lut(lut_file_name)

    # Load atlas label image
    atlas_img = nib.load(map_image_name)
    atlas_data = np.asarray(atlas_img.get_fdata()).astype(int)
    voxel_vol_mm3 = np.abs(np.linalg.det(atlas_img.affine[:3, :3]))

    # Region labels (ignore 0 = background)
    region_labels = np.sort(np.unique(atlas_data))
    region_labels = region_labels[region_labels != 0]


    # Optionally override with user-supplied names
    def _region_name(label):
        if region_names is not None and label > 0:
            idx = label - 1  # 1-based labels
            if idx < len(region_names):
                return str(region_names[idx])
        if label in lut:
            return lut[label]
        return f"region {label}"

    # ----------------------------
    # Stats functions
    # ----------------------------
    def _nanmean(x):
        return np.nanmean(x) if x.size > 0 else np.nan

    def _nanstd(x):
        return np.nanstd(x) if x.size > 0 else np.nan

    stat_fns = {
        "mean": _nanmean,
        "nanmean": _nanmean,
        "std": _nanstd,
        "nanstd": _nanstd,
        "max": lambda x: np.nanmax(x) if x.size > 0 else np.nan,
        "min": lambda x: np.nanmin(x) if x.size > 0 else np.nan,
        "median": lambda x: np.nanmedian(x) if x.size > 0 else np.nan,
        "abs": lambda x: _nanmean(np.abs(x)),  # mean absolute value
    }

    stats = list(stats)
    for s in stats:
        if s not in stat_fns:
            raise ValueError(f"Unsupported stat '{s}'. Supported: {list(stat_fns.keys())}")

    # ----------------------------
    # Loop over images and regions
    # ----------------------------
    rows = []

    for img_idx, img_spec in enumerate(images, start=1):
        img, img_name = _load_img(img_spec)
        data = np.asarray(img.get_fdata())

        if data.shape != atlas_data.shape:
            raise ValueError(
                "Atlas and image have different shapes. "
                "Please reslice your image to the SUIT atlas grid first."
            )

        for r in region_labels:
            mask = atlas_data == r
            if not np.any(mask):
                continue

            roi_vals = data[mask]
            size_mm3 = float(mask.sum() * voxel_vol_mm3)

            row = {
                "image": img_idx,
                "image_name": img_name,
                "atlas": atlas,
                "map": maps,
                "space": space,
                "region": int(r),
                "regionname": _region_name(int(r)),
                "size": size_mm3,
            }

            for s in stats:
                row[s] = float(stat_fns[s](roi_vals))

            rows.append(row)

    df = pd.DataFrame(rows)
    if outfilename is not None:
        # Save TXT (tab-delimited)
        df.to_csv(outfilename, sep="\t", index=False)

    return df

