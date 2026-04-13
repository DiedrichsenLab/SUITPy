#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to resample image into atlas space 
@authors: Jorn Diedrichsen
"""
import nibabel as nib
import numpy as np
from numpy.linalg import inv
import nitools.volume as ntv

def reslice_image(
                source_image,
                deformation,
                mask = None,
                interp = 1,
                voxelsize = None,
                imagedim = None,
                affine = None,
                replace_nan = True,
                mask_thr = 1.0):
    """Reslices images into atlas space using a deformation map and mask 

    Args:
        source_image: (Nifti1Image, str, or iterable of NIFTI)
            Images to reslice
        deformation: (Nifti1Image, str):
            Nonlinear deformation file (y_xxx.nii)
        mask (NIFTI, str):
            Optional masking image (defaults to None)
        interp (int):
            0: nearest neighbor, 1:trilinear
        voxelsize (tuple):
            Desired voxel size - defaults to deformation image
        imagedim (tuple):
            desired image dimensions: Defaults to deformation image
        affine (ndaray):
            affine transformation matrix of target image
        replace_nan (bool):
            if true, replaces Nan values with 0
        mask_thr (float):
            If given, binarizes the mask at this threshold
            Defaults to 1.0
            
    Returns:
        image (NIFTI image or list of NIFTI Images )
    """

    if type(deformation) == str:
        deformation = nib.load(deformation)

    if mask != None:
        if type(mask) == str:
            mask = nib.load(mask)

    # Deal with voxelsize: This works only for
    # image in LPI / RPI format
    if voxelsize is not None:
        if (imagedim is not None) | (affine is not None):
            raise(NameError('give either voxelsize or (imagedim / affine), but not both'))
        fac = voxelsize / np.abs(np.diag(deformation.affine[0:3,0:3]))
        aff_scale = np.diag(np.append(fac,[1]))
        affine = deformation.affine @ aff_scale
        imagedim = np.ceil(deformation.shape[0:3] / fac).astype(int)

    if affine is None:
        affine = deformation.affine
    if imagedim is None:
        imagedim = deformation.shape[0:3]

    # Now iterate over images
    if type(source_image) == list:
        output_list = []
        for img in source_image:
            if type(img) == str:
                img = nib.load(img)
            output_img = reslice_img(img, deformation, mask, interp, imagedim, affine, replace_nan, mask_thr)
            output_list.append(output_img)
        return output_list
    else:
        if type(source_image) == str:
            source_image = nib.load(source_image)
        output_img = reslice_img(source_image, deformation, mask, interp, imagedim, affine, replace_nan, mask_thr)
        return output_img

def reslice_img(img,
                deformation,
                mask,
                interp,
                imagedim,
                affine,
                replace_nan,
                mask_thr):
    """
    Resample image

    Args:
        img: (NIFTI Image)
            Images to reslice
        deformation: (NIFTI):
            Nonlinear deformation file (y_xxx.nii)
        mask (NIFTI):
            Optional masking image (defaults to None)
        interp (int):
            0: nearest neighbor, 1:trilinear
        imagedim (tuple):
            desired image size
        affine (ndarray):
            Affine transformation matrix of desired target image
        replace_nan (bool):
            if true, replaces Nan values with 0
        mask_thr (float):
            If given, binarizes the mask at this threshold
    Returns:
        image (NIFTI image or list of NIFTI Images )
    """
    I,J,K = np.meshgrid(np.arange(imagedim[0]),
                        np.arange(imagedim[1]),
                        np.arange(imagedim[2]),
                        indexing='ij')
    X,Y,Z = ntv.affine_transform(I,J,K, affine)
    coord_def = ntv.sample_image(deformation,X,Y,Z,1).squeeze()
    xm = coord_def[:,:,:,0]
    ym = coord_def[:,:,:,1]
    zm = coord_def[:,:,:,2]
    data = ntv.sample_image(img, xm, ym, zm, interp)
    if mask != None:
        maskData = ntv.sample_image(mask, xm, ym, zm, interp)
        if mask_thr is not None:
            maskData = (maskData >= mask_thr).astype(np.float32)
        data = np.multiply(data,maskData)

    # if replace_nan, replace nan with zero
    if replace_nan:
        np.nan_to_num(data,copy=False)
    # Create new image
    output_img = nib.Nifti1Image(data, affine=affine)
    output_img.set_qform(output_img.get_qform())
    output_img.header.set_xyzt_units('mm', 'sec')
    output_img.update_header()
    return output_img

