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
import nilearn.image as image

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

    # Deal with voxelsize and imagedim and affine
    if voxelsize is not None: 
        if (imagedim is not None) | (affine is not None):
            raise(NameError('give either voxelsize or (imagedim / affine), but not both'))
        fac = np.diag(deformation.affine[0:3,0:3])/voxelsize
        pass

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

def reslice_img(
                img,
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
                        np.arange(imagedim[2]))
    X,Y,Z = affine_transform(I,J,K, affine)
    coord_def = sample_image(deformation,X,Y,Z)
    xm = coord_def[0]
    ym = coord_def[1]
    zm = coord_def[2]
    data = sample_image(img, xm, ym, zm, interp)
    if mask != None:
        maskData = sample_image(mask, xm, ym, zm, interp))
        data = np.multiply(data,maskData)
    
    img = create_img(data, affine)
    return img

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
        xm (np.array):
            X-coordinates which are in the range
        ym (np.array):
            Y-coordinates which are in the range
        zm (np.array):
            Z-coordinates which are in the range
    """
    is_in_image = np.zeros(xm.shape[0])
    check_x = np.zeros(xm.shape[0])
    check_y = np.zeros(ym.shape[0])
    check_z = np.zeros(zm.shape[0])

    check_x = np.where(np.logical_and(xm >= 0, xm <= img.shape[0]-1), True, False)
    check_y = np.where(np.logical_and(ym >= 0, ym <= img.shape[1]-1), True, False)
    check_z = np.where(np.logical_and(zm >= 0, zm <= img.shape[2]-1), True, False)
    
    is_in_image = np.where(np.logical_and(np.logical_and(check_x, check_y), check_z), 1, 0)
        
    xm = np.multiply(xm, is_in_image)
    ym = np.multiply(ym, is_in_image)
    zm = np.multiply(zm, is_in_image)
    return xm, ym, zm


def trilinear(
            img,
            xm,
            ym,
            zm
            ):
    """
    Return values after trilinear interpolation
    
    Args:
        img (Nifti image)
        xm (np.array)
            X-coordinate in voxel system 
        ym (np.array)
            Y-coordinate in voxel system 
        zm (np.array)
            Z-coordinate in voxel system
    Returns:
        c (np.array)
            Array contains all values in the image
    """
    xm, ym, zm = check_range(img, xm, ym, zm)
    
    xcoord = np.floor(xm).astype('int')
    ycoord = np.floor(ym).astype('int')
    zcoord = np.floor(zm).astype('int')

    xcoord = np.where(xcoord > img.shape[0]-2, img.shape[0]-2, xcoord)
    xcoord = np.where(xcoord < 0, 0, xcoord)
    ycoord = np.where(ycoord > img.shape[1]-2, img.shape[1]-2, ycoord)
    ycoord = np.where(ycoord < 0, 0, ycoord)
    zcoord = np.where(zcoord > img.shape[2]-2, img.shape[2]-2, zcoord)
    zcoord = np.where(zcoord < 0, 0, zcoord)
    
    xd = xm - xcoord
    yd = ym - ycoord
    zd = zm - zcoord

    
    c000 = img.get_fdata()[xcoord, ycoord, zcoord]
    c100 = img.get_fdata()[xcoord+1, ycoord, zcoord]
    c110 = img.get_fdata()[xcoord+1, ycoord+1, zcoord]
    c101 = img.get_fdata()[xcoord+1, ycoord, zcoord+1]
    c111 = img.get_fdata()[xcoord+1, ycoord+1, zcoord+1]
    c010 = img.get_fdata()[xcoord, ycoord+1, zcoord]
    c011 = img.get_fdata()[xcoord, ycoord+1, zcoord+1]
    c001 = img.get_fdata()[xcoord, ycoord, zcoord+1]

    c00 = c000*(1-xd)+c100*xd
    c01 = c001*(1-xd)+c101*xd
    c10 = c010*(1-xd)+c110*xd
    c11 = c011*(1-xd)+c111*xd
    
    c0 = c00*(1-yd)+c10*yd
    c1 = c01*(1-yd)+c11*yd
    
    c = c0*(1-zd)+c1*zd
    
    return c


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
    im,jm,jk = affine_transform(xm,ym,zm,inv(img.affine)
    value = np.zeros(xm.shape, dtype=int)
    if interpolation == 1:
        value = trilinear(img, xm, ym, zm)
    elif interpolation == 0:
        xm = np.round(xm).astype('int')
        ym = np.round(ym).astype('int')
        zm = np.round(zm).astype('int')
        xm, ym, zm = check_range(img, im, jm, km)
        value = img.get_fdata()[xm, ym, zm]
    return value


def create_img(
            value,
            affine
            ):
    """
    Saving an array as an NIFTI image

    Args:
        value (np.array):
            An array contains values which should be stored into an NIFTI image
        affine (np.array):
            Affine matrix for the output image
    
    Returns:
        output_img (NIFTI image):
            The output image which contains values
    """
    output_img = nib.Nifti1Image(value.astype('int16'), affine=affine)
    output_img.set_qform(output_img.get_qform())
    output_img.header.set_xyzt_units('mm', 'sec')
    output_img.update_header()

    return output_img

