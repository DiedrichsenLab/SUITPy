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
from nitools.volume import sample_image, affine_transform


def fetch_atlas(atlas, atlas_dir=None, maps = 'all', space='all',
                    base_url=None, resume=True, verbose=1):
    """Download and install cerebellar atlas maps from github.com/DiedrichsenLab/cerebellar_atlases

    Args:
        atlas (str): Name of the atlas (Diedrichsen_2009, King_2019, etc. )
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


def _read_lut(fname):
    """Reads a Lookuptable file

    Args:
        fname (str): Filename

    Returns:
        index (ndarray): Numerical keys
        colors (ndarray): N x 3 ndarray of colors
        labels (list): List of labels
    """
    L = pd.read_csv(fname,header=None,sep=' ',names=['ind','R','G','B','label'])
    index = L.ind.to_numpy()
    colors = np.c_[L.R.to_numpy(),L.G.to_numpy(),L.B.to_numpy()]
    labels = list(L.label)
    return index,colors,labels


# Helper functions
def _load_img(img_obj):
    """Load file path or nibabel image into nibabel image"""
    if isinstance(img_obj, str):
        return nib.load(img_obj), os.path.basename(img_obj)
    elif hasattr(img_obj, "get_fdata"):
        return img_obj, "<nibabel_image>"
    else:
        raise TypeError("Images must be a file path or nibabel NIfTI image.")


def _nanmean(x):
    return np.nanmean(x) if x.size > 0 else np.nan


def _nanstd(x):
    return np.nanstd(x) if x.size > 0 else np.nan


def _region_name(label, lut, region_names):
    """Resolve region name from LUT or region_names or fallback."""
    if region_names is not None:
        idx = label - 1
        if idx < len(region_names):
            return str(region_names[idx])
    if lut is not None and label in lut:
        return lut[label]
    return f"region {label}"

# summarize_data
def summarize_data(
    images,
    label_image=None,
    atlas=None,
    maps=None,
    space="SUIT",
    atlas_dir=None,
    stats=("nanmean", "nanstd"),
    region_names=None,
    outfilename=None,
    verbose=1,):
    
    
    """Summarize the data from the images by ROIs defined in a label image.

    This works optimally with the files provided in the cerebellar atlas
    repository, but can also operate on completely custom label images.

    Important:
        - The atlas is **not** downloaded automatically. You must call
          `fetch_atlas(...)` yourself beforehand if you want to use the
          cerebellar-atlases repository.
        - If you provide `label_image`, the function will use that image
          directly and ignore the atlas / maps / space for determining
          ROIs.
        - If images are not in the same voxel grid as the label image / atlas,
          they are resampled into atlas voxel space using the affine transform.

    Args:
        images (list or str or nib image):
            One or multiple image(s) (3D or 4D NIfTI) to summarize.
        atlas (str or None):
            Name of the atlas (Diedrichsen_2009, King_2019, etc.).
            For custom label images this can be None or a free-form name.
        maps (str or None):
            Name of the map within the atlas (atl-Buckner7, atl-Anatomical).
            Ignored if `label_image` is provided.
        space (str):
            Space for the volumetric atlas file: 'SUIT', 'MNI', 'MNISym', etc.
            Used only when `label_image` is None.
        atlas_dir (str or None):
            Base directory of cerebellar atlases. If None, the default atlas
            directory used by SUITPy is used.
        stats (sequence of str):
            Which statistics to compute inside each ROI. Supported keys:
            'mean', 'nanmean', 'std', 'nanstd'.
        region_names (sequence of str or None):
            Optional list of region names. If provided and length >= number of
            non-zero labels, it overrides names from the LUT.
        outfilename (str or None):
            If not None, write the resulting table as a tab-delimited text
            file AND an .xlsx file with the same basename.
        resume (bool):
            Unused here, kept for API compatibility.
        verbose (int):
            Verbosity level.
        label_image (str or nib image or None):
            Custom label image independent of cerebellar atlases. If provided,
            this is used as the ROI definition and no atlas download or lookup
            is performed.
        lut_file (str or None):
            Optional LUT file for mapping label indices to names. If None and
            using cerebellar atlases, it defaults to `<maps>.lut` in the atlas
            directory. For custom label images, if lut_file is None, generic
            names ('region <id>') are used.
        interp_order (int):
            Reserved for future use (currently ignored).

    Returns:
        df (pandas.DataFrame):
            One row per image / frame / region with columns:
                - image: integer index of input image
                - image_name: basename or placeholder for the image
                - atlas: atlas name (or custom label name)
                - map: map name (if applicable)
                - space: space name (if applicable)
                - frame: frame index (0 for 3D images)
                - region: integer label value
                - regionname: region label name
                - size: ROI volume in mm^3
                - plus one column per requested statistic.
    """

    if not isinstance(images, (list, tuple)):
        images = [images]

    lut = None

    # CASE A — USER PROVIDED CUSTOM LABEL IMAGE (NO ATLAS REQUIRED)
    if label_image is not None:
        atlas_img, _ = _load_img(label_image)
    
    # CASE B — USE CEREBELLAR ATLAS 
    else:
        if atlas is None or maps is None:
            raise ValueError("If no 'label_image' is provided, both 'atlas' and 'maps' must be specified.")

        atlas_path = utils._get_atlas_dir(atlas, atlas_dir)

        if atlas_path is None or not os.path.isdir(atlas_path):
            raise FileNotFoundError(f"Atlas '{atlas}' not found in: {atlas_dir}\n"
                "Please download it manually using fetch_atlas(atlas=..., atlas_dir=...).")

        map_file = os.path.join(atlas_path, f"{maps}_space-{space}_dseg.nii")
        if not os.path.isfile(map_file):
            raise FileNotFoundError(
                "Atlas label image not found on disk.\n"
                f"  Expected file:\n"
                f" img_path\n\n"
                "You need to download this atlas first from "
                "https://github.com/DiedrichsenLab/cerebellar_atlases. "
                "In Python, run:\n\n"
                "    from SUITPy.atlas import fetch_atlas\n"
                f"    fetch_atlas(atlas='{atlas}', "
                f"atlas_dir=r'{atlas_dir}', "
                f"maps='{maps}', space='{space}', "
                "base_url=None)\n\n"
                "After this has finished successfully, rerun roi_summarize().")

        atlas_img = nib.load(map_file)

        # Read atlas LUT using YOUR exact function
        lut_file = os.path.join(atlas_path, f"{maps}.lut")
        if os.path.isfile(lut_file):
            index, colors, labels = _read_lut(lut_file)
            # FIX: map index -> label (do not try to unpack colors here)
            lut = {int(i): lbl for i, lbl in zip(index, labels)}

    # Extract region IDs
    atlas_data = np.asarray(atlas_img.get_fdata()).astype(int)
    voxel_vol = np.abs(np.linalg.det(atlas_img.affine[:3, :3]))

    region_labels = np.unique(atlas_data)
    region_labels = region_labels[region_labels != 0]

    # Precompute atlas-voxel world coordinates for resampling
    nx, ny, nz = atlas_data.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    xm, ym, zm = affine_transform(ii, jj, kk, atlas_img.affine)

    # Stats functions
    stat_fns = {"mean": _nanmean,
        "nanmean": _nanmean,
        "std": _nanstd,
        "nanstd": _nanstd,}

    stats = list(stats)
    for s in stats:
        if s not in stat_fns:
            raise ValueError(f"Unsupported stat: {s}")

    rows = []

    # Loop through images and ROIs
    for idx, img_path in enumerate(images, start=1):
        img, img_name = _load_img(img_path)
        data = img.get_fdata()

        # If image is not in the same voxel grid, resample into atlas space
        # For 4D images we check only the spatial part (first 3 dims)
        if data.ndim < 3:
            raise ValueError("Input image must be at least 3D.")
        if data.shape[:3] != atlas_data.shape:
            if verbose:
                print(f"Resampling image '{img_name}' from shape {data.shape} "
                    f"to atlas spatial shape {atlas_data.shape}.")
            if data.ndim == 3:
                # 3D case: same as before
                data = sample_image(img, xm, ym, zm, interpolation=1)
            elif data.ndim == 4:
                # 4D case: resample each frame separately into atlas space
                n_frames = data.shape[3]
                resampled = np.zeros(atlas_data.shape + (n_frames,), dtype=float)
                for t in range(n_frames):
                    frame_img = nib.Nifti1Image(data[..., t], img.affine)
                    resampled[..., t] = sample_image(frame_img, xm, ym, zm, interpolation=1)
                data = resampled
            else:
                raise ValueError("Images with more than 4 dimensions are not supported.")

        # Now handle 3D vs 4D summarization
        if data.ndim == 3:
            # Single 3D volume -> treat as frame 0
            frame_indices = [0]
        elif data.ndim == 4:
            # 4D NIfTI: iterate over each frame sequentially
            frame_indices = range(data.shape[3])
        else:
            raise ValueError("Images with more than 4 dimensions are not supported.")

        is_4d = (data.ndim == 4)

        for frame in frame_indices:
            # Select the appropriate 3D volume
            if data.ndim == 3:
                frame_data = data
            else:
                frame_data = data[..., frame]

            # Decide how to label image / image_name
            if is_4d:
                image_id = int(frame)
                image_name_frame = f"{img_name}_frame{frame:04d}"
            else:
                image_id = idx
                image_name_frame = img_name

            for r in region_labels:
                mask = (atlas_data == r)
                if not np.any(mask):
                    continue

                roi_vals = frame_data[mask]
                vol = mask.sum() * voxel_vol

                row = {
                    "image": image_id,
                    "image_name": image_name_frame,
                    "atlas": atlas,
                    "map": maps,
                    "space": space,
                    "frame": int(frame),  # 0 for 3D, 0..T-1 for 4D
                    "region": int(r),
                    "regionname": _region_name(int(r), lut, region_names),
                    "size": float(vol),}

                for s in stats:
                    row[s] = float(stat_fns[s](roi_vals))

                rows.append(row)

    df = pd.DataFrame(rows)
    
    # File output
    if outfilename is not None:
        # Save TXT (tab-delimited)
        df.to_csv(outfilename, sep="\t", index=False)

        # Save Excel (.xlsx)
        if outfilename.lower().endswith(".txt"):
            xlsx_filename = outfilename[:-4] + ".xlsx"
        else:
            xlsx_filename = outfilename + ".xlsx"
        df.to_excel(xlsx_filename, index=False)

        if verbose:
            print(f"Saved TXT → {outfilename}")
            print(f"Saved XLSX → {xlsx_filename}")
            
    
    if outfilename is not None:
        df.to_csv(outfilename, sep="\t", index=False)

    return df
