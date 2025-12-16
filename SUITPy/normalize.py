"""
Cerebellar normalization and isolation using ANTsPy
@authors: Yaping Wang, Jorn Diedrichsen
"""

import sys
import argparse
import os
import nibabel as nib
import ants
import numpy as np
import pandas as pd
from nibabel.affines import apply_affine
import nitools as nt


def normalize(source_file, mask_file, space='SUIT', template_file=None,
              type_of_transform='antsRegistrationSyN[s]',
              write_normalized=True,
              write_ants_transform=False,
              write_deformation=True,
              write_inv_deformation=False,
              write_jacobian_determinant=False,
              result_folder=None,
              verbose=1):
    """
    Normalizes a T1w image to the SUIT template using ANTsPy

    Args:
        source_file (str): Path to the source T1w image
        mask_file (str): Path to the cerebellar mask image
        space (str): Cerebellar-only template (`SUIT`, `MNI152NLin6AsymC` / 'MNI', `MNI152NLin2009cSymC` / 'MNISym')
        template_file (str): Optional path to a custom template file
        type_of_transform (str): ANTs registration type (e.g., 'antsRegistrationSyN[s]')
        write_normalized (bool): Save normalized (template-space) T1 image
        write_deformation (bool): Save deformation field y(x) for reslice other images
        write_inv_deformation (bool): Save deformation field y(x) for reslice other images
        write_jacobian_determinant (bool): Computes & save log-Jacobian determinant
        result_folder (str): Output folder. If None, uses same folder as source file
        verbose (int): 0: silent, 1:Progress log, 2:detailed log

    Returns:
        dict: A dictionary containing output images (if selected to be written)
            fwdtransforms (str): Path to forward transforms (if write_ant_transform)
            invtransforms (str): Path to inverse transforms (if write_ant_transform)
            fwd_deformation (str): Path to composite displacement field
            inv_deformation (str): Path to inverse deformation field
            normalized_image (str): Path to normalized image in template space
            jacobian_determinant (str): Path to log-Jacobian determinant map
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
    if verbose>0:
        print(f"Normalizing {basename} to {template_file}")
    prefix = f'{result_folder}/{basename}_xfm-{space}_'
    mytx = ants.registration(fixed=template_img,moving=masked_source_img,
                             type_of_transform=type_of_transform,
                             outprefix=prefix,
                             write_composite_transform=False,
                             verbose=(verbose>1))

    # Write the normalized image in template space
    if write_normalized:
        normalized_file = f'{result_folder}/{basename}_space-{space}.nii.gz'
        if verbose:
            print(f"Saving the normalized image into {normalized_file}")
        ants.image_write(mytx['warpedmovout'], normalized_file)

    # Write the deformation field for reslicing images from subject to template space
    if write_deformation:
        deformation_file = f"{result_folder}/{basename}_to-SUIT_mode-image_xfm.nii.gz"
        if verbose:
            print(f"Saving deformation field into {deformation_file}")
        deformation_from_displacement(
            template_file=template_file,
            displacement_file=mytx['fwdtransforms'],
            deformation_file=deformation_file,
            verbose=verbose
        )

    # Write inverse deformation if requested
    if write_inv_deformation:
        raise(NotImplementedError('inverse deformation not implemented yet'))
        """
        Below code is not correct - the affine martix needs to be invertedd
        inv_deformation_file = f"{result_folder}/{basename}_from-SUIT_mode-image_xfm.nii.gz"
        if verbose:
            print(f"Saving inverse deformation field into {inv_deformation_file}")
        deformation_from_displacement(
            template_file=source_file,
            displacement_file=mytx['invtransforms'],
            deformation_file=inv_deformation_file,
            verbose=verbose
        )
        """

    # Write the Jacobian determinant image for vbm analysis
    if write_jacobian_determinant:
        # Produce displacement map
        # Compose taffine + warp → displacement field
        displacement_file = f"{result_folder}/{basename}_to-SUIT_mode-image_disp.nii.gz"
        ants.apply_transforms(
            fixed=template_img,
            moving=masked_source_img,
            transformlist=[f'{prefix}1Warp.nii.gz', f'{prefix}0GenericAffine.mat'],
            whichtoinvert=[False, False],
            compose=f'{prefix}')
        os.rename(f'{prefix}comptx.nii.gz', displacement_file)


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
        os.remove(displacement_file)

    # Lightweight return dictionary
    return_dict={}
    if not write_ants_transform:
        os.remove(mytx["fwdtransforms"][0])
        os.remove(mytx["fwdtransforms"][1])
        os.remove(mytx["invtransforms"][1])
    else:
        return_dict["fwd_transforms"]= mytx["fwdtransforms"]
        return_dict["inv_transforms"]= mytx["invtransforms"]

    if write_deformation:
        return_dict["fwd_deformation"] = deformation_file
    if write_normalized: 
        return_dict["normalized_image"] = normalized_file

    if write_inv_deformation:
        return_dict["inv_deformation"] = None
    if write_jacobian_determinant:
        return_dict["jacobian_determinant"] = jacobian_file

    return return_dict



def deformation_from_displacement(template_file, displacement_file, deformation_file, verbose=False):
    """
    Convert an ANTs composite displacement field into a deformation field y(x) that maps
    voxel coordinates in the template space into world coordinates of moveable image.
    This deformation map can be used to convert points from template space into the moveable space
    or to resample the moveable image into the template space.

    Args:
        template_file (str): Path to the template image (defines grid, affine, spacing)
        displacement_file (str): Path to ANTs displacement field.
        deformation_file (str): Output filename for the resulting deformation field.
    """
    # Load template
    tpl_img = nib.load(template_file)
    A_tpl   = tpl_img.affine
    nx, ny, nz = tpl_img.shape

    # Build voxel grid (template index space)
    I, J, K = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    vox = np.vstack([I.ravel(), J.ravel(), K.ravel()])

    # Voxel indices → World coordinates
    pts_tmp = nt.affine_transform_mat(vox,A_tpl)

    # LPI → RAI (ANTs convention)
    pts_tmp_ants = pts_tmp.copy()
    pts_tmp_ants[0,:] *= -1
    pts_tmp_ants[1,:] *= -1
    df_pts = pd.DataFrame(pts_tmp_ants.T, columns=["x", "y", "z"])

    # Apply ANTs transform (template → subject)
    df_subj_ants = ants.apply_transforms_to_points(
        dim=3,
        points=df_pts,
        transformlist=displacement_file if isinstance(displacement_file, list) else [displacement_file]
    )
    pts_subj_ants = df_subj_ants[["x","y","z"]].values

    # Move the points back to LPI coordinates
    pts_subj = pts_subj_ants.copy()
    pts_subj[:,0] *= -1
    pts_subj[:,1] *= -1

    # Reshape into SUIT deformation format
    deformation = pts_subj.reshape(nx, ny, nz, 1, 3).astype(np.float32)

    out_img = nib.Nifti1Image(deformation, A_tpl)
    nib.save(out_img, deformation_file)

    return


# Use main to make function callable from command line (see isolate.py)
if __name__ == '__main__':
    pass
