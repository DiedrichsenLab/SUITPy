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



def normalize(source_file, mask_file, space='SUIT', template_file=None, 
              type_of_transform='antsRegistrationSyN[s]', write_normalized_image=True, 
              write_deformation_field=True, write_jacobian_determinant=True,
              result_folder=None, verbose=False):
    """
    Normalizes a T1w image to the SUIT template using ANTsPy

    Args:
        source_file (str): Path to the source T1w image
        mask_file (str): Path to the cerebellar mask image
        space (str): Cerebellar-only template (`SUIT`, `MNI152NLin6AsymC` / 'MNI', `MNI152NLin2009cSymC` / 'MNISym')
        template_file (str): Optional path to a custom template file
        type_of_transform (str): ANTs registration type (e.g., 'antsRegistrationSyN[s]')
        write_normalized_image (bool): Save normalized (template-space) T1 image
        write_deformation_file (bool): Save deformation field y(x) for reslice other images
        write_jacobian_determinant (bool): Compute & save log-Jacobian determinant
        result_folder (str): Output folder. If None, uses same folder as source file
        verbose (bool): Print detailed logs

    Returns:
        dict: A dictionary containing:
            fwdtransforms (str): Path to forward transforms
            invtransforms (str): Path to inverse transforms
            displacement_file (str): Path to composite displacement field
            deformation_file (str): Path to deformation field (or None)
            normalized_file (str): Path to normalized image in template space (or None)
            jacobian_file (str): Path to log-Jacobian determinant map (or None)
    """
    # Get result folder and base name
    result_folder = os.path.dirname(os.path.abspath(source_file)) if result_folder is None else result_folder
    basename = os.path.splitext(os.path.basename(source_file))
    if basename[1] == '.gz':
        basename = os.path.splitext(basename[0])
    basename = basename[0]
    source_img = ants.image_read(source_file)
    mask_img = ants.image_read(mask_file)
    # mask the source image and normalize 
    masked_source_img = source_img * mask_img

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

    # ANTs Registration
    if verbose:
        print(f"Registering to template of {template_file}")
    prefix = f'{result_folder}/{basename}_xfm-{space}_'
    mytx = ants.registration(fixed=template_img,moving=masked_source_img,
                             type_of_transform=type_of_transform,
                             outprefix=prefix,write_composite_transform=False,
                             verbose=verbose)

    # Compose taffine + warp → displacement field
    displacement_file = f"{result_folder}/{basename}_to-SUIT_mode-image_disp.nii.gz"
    ants.apply_transforms(
        fixed=template_img,
        moving=masked_source_img.clone('float'),
        transformlist=[f'{prefix}1Warp.nii.gz', f'{prefix}0GenericAffine.mat'],
        whichtoinvert=[False, False],
        compose=f'{prefix}')
    os.rename(f'{prefix}comptx.nii.gz', displacement_file)
    if verbose:
        print(f"Saving the displacement field into {displacement_file}")

    # Write the normalized image in template space
    if write_normalized_image:
        normalized_file = f'{result_folder}/{basename}_space-{space}.nii.gz'
        if verbose:
            print(f"Saving the normalized image into {normalized_file}")
        ants.image_write(mytx['warpedmovout'], normalized_file)

    # Write the deformation field for reslice images from subject to template space
    if write_deformation_field:
        deformation_file = f"{result_folder}/{basename}_to-SUIT_mode-image_xfm.nii.gz"
        if verbose:
            print(f"Saving the deformation field into {deformation_file}")
        deformation_from_displacement(
            template_file=template_file,
            displacement_file=displacement_file,
            deformation_file=deformation_file,
            verbose=verbose
        )

    # Write the Jacobian determinant image for vbm analysis
    if write_jacobian_determinant:
        jacobian_file = f"{result_folder}/{basename}_to-SUIT_mode-image_detJ.nii.gz"
        # Jacobian settings
        use_log = True      # log-Jacobian
        use_geom = True     # geometric Jacobian
        jac_img = ants.create_jacobian_determinant_image(
            domain_image=template_img,
            tx=displacement_file,
            do_log=use_log,
            geom=use_geom
        )
        jac_img.to_filename(jacobian_file)
        if verbose:
            print(f"Computing Jacobian: geom={use_geom}, log={use_log},\
                    Saving the Jacobian determinant into {jacobian_file}")
    
    # Lightweight return dictionary (no ANTsImages)
    return {
        "fwdtransforms": mytx["fwdtransforms"],
        "invtransforms": mytx["invtransforms"],
        "displacement_file": displacement_file,
        "deformation_file": deformation_file if write_deformation_field else None,
        "normalized_file": normalized_file if write_normalized_image else None,
        "jacobian_file": jacobian_file if write_jacobian_determinant else None,
        }   



def deformation_from_displacement(template_file, displacement_file, deformation_file, verbose=False):
    """
    Convert an ANTs composite displacement field into a deformation field y(x) that maps 
    voxel coordinates in the template space into world coordinates of the subject space.
    Importantly, point mapping goes the opposite direction of image mapping, 
    for both reasons of convention and engineering.

    Args:
        template_file (str): Path to the template image (defines grid, affine, spacing)
        displacement_file (str): Path to ANTs displacement field.
        deformation_file (str): Output filename for the resulting deformation field.

    Returns:
        deformation_file (str): Path to the saved deformation field.

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
        transformlist=displacement_file if isinstance(displacement_file, list) else [displacement_file]
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
