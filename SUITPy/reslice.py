#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUIT toolbox reslice module

Basic functionality for resample image into atlas

"""

from tkinter import N
import nibabel as nib
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import warnings
from nilearn import image
from SUITPy.flatmap import affine_transform

def reslice_image(
                source_image,
                deformation,
                mask = None,
                interp = 1,
                voxelsize = None,
                imagesize = None,
                affinemat = None
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
        imagesize (tuple): 
            desired image size: Defaults to deformation image 
        affinemat (ndarray): 
            The affine transformation matrix for the linear part of the normalization
    Returns: 
        image (NIFTI image or list of NIFTI Images )
    """
    
    if type(affinemat) == str:
        affinemat = sio.loadmat(affinemat).get('Affine')
        
    if type(deformation) == str:
        deformation = nib.load(deformation)
    
    if mask != None:
        if type(mask) == str:
            mask = nib.load(mask)


    list_after_def_img = []

    for i in range(len(source_image)):
        if type(source_image[i]) == str:
            source_image[i] = nib.load(source_image[i])
        img = source_image[i]
    
        xm_data, ym_data, zm_data = mesh_data(affinemat,img,deformation)
        data = np.zeros((deformation.shape[0], deformation.shape[1], deformation.shape[2]))
        data = sample_image(img, xm_data, ym_data, zm_data, interp).reshape((deformation.shape[0], deformation.shape[1], deformation.shape[2]))
        if mask != None:
            xm_mask, ym_mask, zm_mask = mesh_data(affinemat, mask, deformation)
            maskData = np.zeros((deformation.shape[0], deformation.shape[1], deformation.shape[2]))
            maskData = sample_image(mask, xm_mask, ym_mask, zm_mask, interp).reshape((deformation.shape[0], deformation.shape[1], deformation.shape[2]))
            masked = np.multiply(data,maskData)
            img = create_img(masked, deformation.affine)
        else:
            img = create_img(data, deformation.affine)
        after_def = non_linear_deformation(deformation, img, affinemat)
        aff = np.copy(deformation.affine)
        
        after_def_img = create_img(after_def, aff)
        
        if imagesize != None:
            initial_x, initial_y, initial_z = after_def_img.shape
            new_x, new_y, new_z = imagesize[0], imagesize[1], imagesize[2]
            delta_x = initial_x/new_x
            delta_y = initial_y/new_y
            delta_z = initial_z/new_z
            
            xx = np.linspace(0, new_x-1, new_x, dtype=int)
            yy = np.linspace(0, new_y-1, new_y, dtype=int)
            zz = np.linspace(0, new_z-1, new_z, dtype=int)
            
            x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')
            
            affine = np.eye(4)
            affine[0][0] = delta_x
            affine[1][1] = delta_y
            affine[2][2] = delta_z
            
            x_data, y_data, z_data = affine_transform(x, y, z, affine)
            
            after_def_img = sample_image(after_def_img, x_data, y_data, z_data, 1).reshape(new_x,new_y,new_z)
            if voxelsize == None:
                after_def_img = create_img(after_def_img, np.eye(4))
            else:
                warnings.warn('Both affine matrix and voxel size are specified!')
                affine = np.eye(4)
                affine[0][0] = voxelsize[0]
                affine[1][1] = voxelsize[1]
                affine[2][2] = voxelsize[2]
                after_def_img = create_img(after_def_img, affine)
                
        elif voxelsize != None and imagesize == None:
            warnings.warn('Both affine matrix and voxel size are specified!')
            aff = np.copy(after_def_img.affine)
            aff[0][0] = voxelsize[0] if aff[0][0] > 0 else -voxelsize[0]
            aff[1][1] = voxelsize[1] if aff[1][1] > 0 else -voxelsize[1]
            aff[2][2] = voxelsize[2] if aff[2][2] > 0 else -voxelsize[2]
            after_def_img = image.resample_img(after_def_img, aff)
        
        
        
        list_after_def_img.append(after_def_img)
        
            
    return list_after_def_img
    



def mesh_data(
            affineTr,
            img,
            deformation
            ):
    """
    Meshgrid x, y, z coordinates and then applying affine matrix to them to get x, y, z coordinates in voxel space.

    Args:
        affineTr (np.array):
            The affine transformation matrix for the linear part of the normalization
        img (NIFTI image):
            The target image
        deformation (NIFTI image):
            Non-linear deformation
    
    Returns:
        x_data (np.array):
            The array contains x coordinates in voxel space
        y_data (np.array):
            The array contains y coordinates in voxel space
        z_data (np.array):
            The array contains z coordinates in voxel space
    """
    xaxis = deformation.shape[0]
    yaxis = deformation.shape[1]
    zaxis = deformation.shape[2]

    xx = np.linspace(0, xaxis-1, xaxis, dtype=int)
    yy = np.linspace(0, yaxis-1, yaxis, dtype=int)
    zz = np.linspace(0, zaxis-1, zaxis, dtype=int)

    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')

    affine = np.linalg.inv(affineTr@img.affine)@deformation.affine

    x_data, y_data, z_data = affine_transform(x, y, z, affine)

    return x_data, y_data, z_data


def check_range(
                img,
                xm,
                ym,
                zm
                ):
    """
    Returns xm, ym, zm which are in their ranges

    Args:
        img (NIFTI image):
            Nifti image
        xm (np.array):
            all x-coordinates
        ym (np.array):
            all y-coordinates
        zm (np.array):
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


def sample_image(
                img,
                xm,
                ym,
                zm,
                interpolation
                ):
    """
    Return values after resample image
    
    Args:
        img (Nifti image)
        xm (np-array)
            X-coordinate in voxel system 
        ym (np-array)
            Y-coordinate in voxel system 
        zm (np-array)
            Z-coordinate in voxel system
    Returns:
        value (np-array)
            Array contains all values in the image
    """
    xm = xm.reshape(-1)
    ym = ym.reshape(-1)
    zm = zm.reshape(-1)
    value = np.zeros(xm.shape, dtype=int)
    if interpolation == 1:
        value = trilinear(img, xm, ym, zm)
    elif interpolation == 0:
        xm = np.round(xm).astype('int')
        ym = np.round(ym).astype('int')
        zm = np.round(zm).astype('int')
        xm, ym, zm = check_range(img, xm, ym, zm)
        value = img.get_fdata()[xm, ym, zm]
    return value



def non_linear_deformation(
                        deformation_img,
                        img,
                        affineTr
                        ):
    """
    Applying non-linear deformation to image and return image.

    Args:
        deformation_img (NIFTI image):
            A image which contains x, y, z coordinates in the native image that correspond to that voxel in atlas space.
        img (NIFTI image):
            The target image
        affineTr (np.array):
            The affine transformation matrix for the linear part of the normalization
        
    Returns:
        value (np.array):
            An array which contains values of the image after applying non-linear deformation on this image
        
    """
    deformation_data = deformation_img.get_fdata()
    deformation_data = np.squeeze(deformation_data)
    
    x = deformation_data[:,:,:,0]
    y = deformation_data[:,:,:,1]
    z = deformation_data[:,:,:,2]

    deformation_aff = deformation_img.affine

    mm_to_voxel = affine_transform(x, y, z, np.linalg.inv(deformation_aff)@affineTr)
    mm_to_voxel = np.array(mm_to_voxel)

    x = mm_to_voxel[0,:,:,:]
    y = mm_to_voxel[1,:,:,:]
    z = mm_to_voxel[2,:,:,:]
    value = sample_image(img, x, y, z, 1).reshape(img.shape[0], img.shape[1], img.shape[2])

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
