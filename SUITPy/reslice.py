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
from SUITPy.flatmap import affine_transform

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
            desired image size: Defaults to deformation image
        affine (ndarray): 
            Affine transformation matrix of desired target image 
    Returns: 
        image (NIFTI image or list of NIFTI Images )
    """
    I,J,K = np.meshgrid(np.arange(imagedim[0]),
                        np.arange(imagedim[1]),
                        np.arange(imagedim[2]),
                        indexing='ij')
    X,Y,Z = affine_transform(I,J,K, affine)
    coord_def = sample_image(deformation,X,Y,Z,1).squeeze()
    xm = coord_def[:,:,:,0]
    ym = coord_def[:,:,:,1]
    zm = coord_def[:,:,:,2]
    data = sample_image(img, xm, ym, zm, interp)
    if mask != None:
        maskData = sample_image(mask, xm, ym, zm, interp)
        data = np.multiply(data,maskData)
    
    # Create new image 
    output_img = nib.Nifti1Image(data, affine=affine)
    output_img.set_qform(output_img.get_qform())
    output_img.header.set_xyzt_units('mm', 'sec')
    output_img.update_header()
    return output_img

def check_range(img,im,jm,km):
    """
    Returns xm, ym, zm which are in their ranges

    Args:
        img (NIFTI image):
            Nifti image
        im (np.array):
            all x-coordinates
        jm (np.array):
            all y-coordinates
        km (np.array):
            all z-coordinates

    Returns:
        im (np.array):
        jm (np.array):
        km (np.array):
            voxel coordinates - set to zero if invalid
        invalid (nd.array)
    """
    invalid = np.logical_not((im>=0) & (im<img.shape[0]) & (jm>=0) & (jm<img.shape[1]) & (km>=0) & (km<img.shape[2]))
    im[invalid] = 0
    jm[invalid] = 0
    km[invalid] = 0 
    return im,jm,km,invalid

def sample_image(img,xm,ym,zm,interpolation):
    """
    Return values after resample image
    
    Args:
        img (Nifti image)
        xm (np-array)
            X-coordinate in world coordinates 
        ym (np-array)
            Y-coordinate in world coordinates
        zm (np-array)
            Z-coordinate in world coordinates 
        interpolation (int)
            0: Nearest neighbor
            1: Trilinear interpolation 
    Returns:
        value (np-array)
            Array contains all values in the image
    """
    im,jm,km = affine_transform(xm,ym,zm,inv(img.affine))

    if interpolation == 1:
        ir = np.floor(im).astype('int')
        jr = np.floor(jm).astype('int')
        kr = np.floor(km).astype('int')

        invalid = np.logical_not((im>=0) & (im<img.shape[0]-1) & (jm>=0) & (jm<img.shape[1]-1) & (km>=0) & (km<img.shape[2]-1))
        ir[invalid] = 0
        jr[invalid] = 0
        kr[invalid] = 0 
                
        id = im - ir
        jd = jm - jr
        kd = km - kr

        D = img.get_fdata()
        if D.ndim == 4:
            ns = id.shape + (1,)
        if D.ndim ==5: 
            ns = id.shape + (1,1)
        else:
            ns = id.shape
        
        id = id.reshape(ns)
        jd = jd.reshape(ns)
        kd = kd.reshape(ns)

        c000 = D[ir, jr, kr]
        c100 = D[ir+1, jr, kr]
        c110 = D[ir+1, jr+1, kr]
        c101 = D[ir+1, jr, kr+1]
        c111 = D[ir+1, jr+1, kr+1]
        c010 = D[ir, jr+1, kr]
        c011 = D[ir, jr+1, kr+1]
        c001 = D[ir, jr, kr+1]

        c00 = c000*(1-id)+c100*id
        c01 = c001*(1-id)+c101*id
        c10 = c010*(1-id)+c110*id
        c11 = c011*(1-id)+c111*id
        
        c0 = c00*(1-jd)+c10*jd
        c1 = c01*(1-jd)+c11*jd
        
        value = c0*(1-kd)+c1*kd
    elif interpolation == 0:
        ir = np.rint(im).astype('int')
        jr = np.rint(jm).astype('int')
        kr = np.rint(km).astype('int')

        ir, jr, kr, invalid = check_range(img, ir, jr, kr)
        value = img.get_fdata()[ir, jr, kr]
    
    # Kill the invalid elements
    if value.dtype is float:
        value[invalid]=np.nan
    else: 
        value[invalid]=0
    return value



