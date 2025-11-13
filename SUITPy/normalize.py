"""
Cerebellar normalization and isolation using ANTsPy
"""

import sys
import argparse
import os
import nibabel as nib
import ants
import numpy as np



def normalize(source_file, mask_file,space='SUIT',template_file=None, write_normalized_image=True,result_folder=None):
    """
    Normalizes a T1w image to the SUIT template using ANTsPy

    Args:
        source_file (str): File name (wuth path) to the source T1w image
        mask_file (str): File name (with path) to the cerebellar mask image
        space (str): Cerebellar-only template (`SUIT`, `MNI152NLin6AsymC` / 'MNI', `MNI152NLin2009cSymC` / 'MNISym')
        template_file (str): Optional path to a custom template file
        write_normalized_image (bool): Whether to write the normalized image to disk

    Returns:

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
    mytx = ants.registration(fixed=template_img , moving=masked_source_img,type_of_transform='SyN',outprefix=prefix,write_composite_transform=True)
    
    # Write the normalized image in template space
    if write_normalized_image:
        normalized_img = ants.apply_transforms(fixed=template_img, moving=masked_source_img,transformlist=mytx['fwdtransforms'],interpolator='linear')
        ants.image_write(normalized_img, f'{result_folder}/{basename}_space-{space}.nii.gz')
    return mytx


# Use main to make function callable from command line (see isolate.py)
if __name__ == '__main__':
    pass 
