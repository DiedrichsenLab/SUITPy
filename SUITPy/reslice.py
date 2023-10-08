#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUIT toolbox reslice module

Basic functionality for resample image into atlas

"""
import nibabel as nib
from numpy import *
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
                affine = None
):
    """[summary]
        source_image: (NIFTI Image, str, or iterable of NIFTI)
            Images to reslice
        deformation: (NIFTI, str):
            Nonlinear deformation file (y_xxx.nii)
        mask (NIFTI, str):
            Optional masking image (defaults to None)
        interp (int):
            0: nearest neighbor, 1:trilinear
        voxelsize (tuple):
            Desired voxel size - defaults to deformation image
            [THROW A WARNING IF BOTH VOXEL SIZE AND AFFINE MAT ARE SPECIFIC]
        imagedim (tuple):
            desired image dimensions: Defaults to deformation image
        affine (ndaray)"
            affine transformation matrix of target image
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
            output_img = reslice_img(img, deformation, mask, interp, imagedim,affine)
            output_list.append(output_img)
        return output_list
    else:
        if type(source_image) == str:
            source_image = nib.load(source_image)
        output_img = reslice_img(source_image, deformation, mask, interp, imagedim, affine)
        return output_img

def reslice_img(img,
                deformation,
                mask,
                interp,
                imagedim,
                affine
):
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
        data = np.multiply(data,maskData)

    # Create new image
    output_img = nib.Nifti1Image(data, affine=affine)
    output_img.set_qform(output_img.get_qform())
    output_img.header.set_xyzt_units('mm', 'sec')
    output_img.update_header()
    return output_img

