"""
Cerebellar normalization and isolation using ANTsPy
"""

import sys
import argparse
import os
import nibabel as nib
import ants
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine



def normalize(source_file, mask_file, space='SUIT', template_file=None, write_deformation_field=True, write_normalized_image=True, result_folder=None, verbose=True):
    """
    Normalizes a T1w image to the SUIT template using ANTsPy

    Args:
        source_file (str): File name (wuth path) to the source T1w image
        mask_file (str): File name (with path) to the cerebellar mask image
        space (str): Cerebellar-only template (`SUIT`, `MNI152NLin6AsymC` / 'MNI', `MNI152NLin2009cSymC` / 'MNISym')
        template_file (str): Optional path to a custom template file
        write_deformation_file (bool): Whether to write the deformation file to disk
        write_normalized_image (bool): Whether to write the normalized image to disk
        verbose (bool): Whether to print out status information during processing

    Returns:
        mytx (dict): A dictionary returned by `ants.registration`, containing:
            warpedmovout (ants.ANTsImage): The moving image warped into template space
            warpedfixout (ants.ANTsImage): The fixed image warped into moving space (rarely used)
            fwdtransforms (str): Paths to the forward transforms
            invtransforms (str): Paths to the inverse transforms
            deformation_file (str): Paths to the deformation file for reslice
            Other registration metadata such as metric values, iterations, and composite transforms

    """
    # Get result folder and base name
    result_folder = os.path.dirname(os.path.abspath(source_file)) if result_folder is None else result_folder
    basename = os.path.splitext(os.path.basename(source_file))
    if basename[1] == '.gz':
        basename = os.path.splitext(basename[0])
    basename = basename[0]
    # load source image
    source_img = ants.image_read(source_file)
    mask_img = ants.image_read(mask_file)

    # Determine name of template file
    if template_file is None:
        template_dir = os.path.join(os.path.dirname(__file__),'templates')
        if space == 'MNI':
            template = 'MNI152NLin6AsymC'
        elif space == 'MNISym':
            template = 'MNI152NLin2009cSymC'
        else:
            template = space
        template_file = os.path.join(template_dir,f'tpl-{template}_T1w.nii.gz')
    if os.path.exists(template_file)==False:
        raise(NameError(f'Unknown template {template}: Set space to `SUIT`, `MNI` /`MNI152NLin6AsymC`, `MNISym`/`MNI152NLin2009cSymC` or provide custom template_file'))
    template_img = ants.image_read(template_file)

    # mask the source image and normalize 
    prefix = f'{result_folder}/{basename}_xfm-{space}_'
    masked_source_img = source_img * mask_img
    if verbose:
        print(f"registering to template of {template_file}")
    mytx = ants.registration(fixed=template_img,moving=masked_source_img,type_of_transform='SyN',outprefix=prefix,write_composite_transform=True)
    if verbose:
        print(f"Saving the forward transforms into {mytx['fwdtransforms']}")

    # Write the normalized image in template space
    if write_normalized_image:
        normalized_img = ants.apply_transforms(fixed=template_img,moving=masked_source_img,transformlist=mytx['fwdtransforms'],interpolator='linear')
        ants.image_write(normalized_img, f'{result_folder}/{basename}_space-{space}.nii.gz')
        if verbose:
            print(f"Saving the normalized image into {result_folder}")
    if write_deformation_field:
        deformation_file = f"{result_folder}/{basename}_from-SUIT_mode-point_xfm.nii.gz"
        built_xfm_from_ants(
            template_file=template_file,
            fwdtransforms=mytx["fwdtransforms"],
            deformation_file=deformation_file
        )
        if verbose:
            print(f"Saving the deformation field into {deformation_file}")
        mytx["deformation_file"] = deformation_file
    return mytx



def built_xfm_from_ants(template_file, fwdtransforms, deformation_file):
    """
    Generate a SUIT-compatible point-to-point deformation field y(x) that maps 
    voxel coordinates in the template space into world coordinates of the 
    subject space.

    Args:
        template_file (str): Path to the template image from which voxel coordinates are defined.
        fwdtransforms (str): The forward (template→subject) transform specification produced by ANTs.
        deformation_file (str): Output filename for the resulting deformation field NIfTI image.

    Returns:
        deformation_file (str): The path to the saved deformation field.

    """
    # Load template
    tpl_img = nib.load(template_file)
    A_tpl   = tpl_img.affine
    nx, ny, nz = tpl_img.shape

    # Build voxel grid (template index space)
    I, J, K = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    vox = np.vstack([I.ravel(), J.ravel(), K.ravel()]).T

    # Index → Template RAS
    pts_ras_tpl = apply_affine(A_tpl, vox)

    # RAS → LPS (ANTs convention)
    pts_lps_tpl = pts_ras_tpl.copy()
    pts_lps_tpl[:,0] *= -1
    pts_lps_tpl[:,1] *= -1
    df_pts = pd.DataFrame(pts_lps_tpl, columns=["x", "y", "z"])

    # Apply ANTs transform (template → subject)
    pts_lps_subj = ants.apply_transforms_to_points(
        dim=3,
        points=df_pts,
        transformlist=fwdtransforms if isinstance(fwdtransforms, list) else [fwdtransforms]
    )
    pts_lps_subj = pts_lps_subj[["x","y","z"]].values

    # LPS → RAS
    pts_ras_subj = pts_lps_subj.copy()
    pts_ras_subj[:,0] *= -1
    pts_ras_subj[:,1] *= -1

    # Reshape into SUIT deformation format
    deformation = pts_ras_subj.reshape(nx, ny, nz, 1, 3).astype(np.float32)

    out_img = nib.Nifti1Image(deformation, A_tpl)
    nib.save(out_img, deformation_file)

    return deformation_file


# Use main to make function callable from command line (see isolate.py)
if __name__ == '__main__':
    pass 
