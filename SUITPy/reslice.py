#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUIT toolbox reslice module

Basic functionality for resample image into atlas

"""

import nibabel as nib
from numpy import *
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from SUITPy._utils import fill_doc
from SUITPy.flatmap import affine_transform

def reslice_image(

)
    """[summary]
        image: (NIFTI Image, str, or iterable of NIFTI)
            Images to reslice
        deformation: (NIFTI, str): 
            Nonlinear dEformation file (y_xxx.nii) 
        mask (NIFTI, str): 
            Optional masking image (defaults to None)
        interp (int):
            0: nearest neighbor, 1:trilinear        
        voxelsize (tuple): Desired voxel size - defaults to deformation image
            [THROW A WARNING IF BOTH VOXEL SIZE AND AFFINE MAT ARE SPECIFIC] 
        imagesize (tuple): 
            desired image size: Defaults to deformation image 
        affinemat (ndarray): Desired affine transformation matrix 
            of the output image
    Returns: 
        image (NIFTI image or list of NIFTI Images )
    """


def mesh_data(
            affineTr,
            img,
            flowfield
            ):
    """
    Meshgrid x, y, z coordinates and then applying affine matrix to them to get x, y, z coordinates in voxel space.

    Args:
        affineTr (np.array):
            The affine transformation matrix for the linear part of the normalization
        img (NIFTI image):
            The target image
        flowfield (NIFTI image):
            Non-linear flowfield
    
    Returns:
        x_data (np.array):
            The array contains x coordinates in voxel space
        y_data (np.array):
            The array contains y coordinates in voxel space
        z_data (np.array):
            The array contains z coordinates in voxel space
    """
    xaxis = flowfield.shape[0]
    yaxis = flowfield.shape[1]
    zaxis = flowfield.shape[2]

    xx = np.linspace(0, xaxis-1, xaxis, dtype=int)
    yy = np.linspace(0, yaxis-1, yaxis, dtype=int)
    zz = np.linspace(0, zaxis-1, zaxis, dtype=int)

    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')

    affine = np.linalg.inv(affineTr@img.affine)@flowfield.affine

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
    if interpolation == 'trilinear':
        value = trilinear(img, xm, ym, zm)
    elif interpolation == 'nearest':
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
    value = sample_image(img, x, y, z, 'trilinear').reshape(img.shape[0], img.shape[1], img.shape[2])

    return value


def create_img(
            value,
            affine,
            save
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

    if save == True:
        nib.save(output_img, filename="reslice_img.nii")

    return output_img


def img_in_LPI(
                img,
                flowfield
                ):
    """
    Put output image into LPI

    Args:
        img (NIFTI image):
            The target image
        flowfield (NIFTI image):
            Non-linear flowfield
    
    Returns:
        data (np.array):
            An array which contains data for the output image.
        vff (np.array):
            An affine matrix for image in LPI
    """
    xaxis = flowfield.shape[0]
    yaxis = flowfield.shape[1]
    zaxis = flowfield.shape[2]

    bbx_min = -flowfield.affine[0,3]
    bbx_max = xaxis + bbx_min-1

    bby_min = flowfield.affine[1,3]
    bby_max = yaxis + bby_min-1

    bbz_min = flowfield.affine[2,3]
    bbz_max = zaxis + bbz_min-1

    xx = np.linspace(bbx_min, bbx_max, xaxis)
    yy = np.linspace(bby_min, bby_max, yaxis)
    zz = np.linspace(bbz_min, bbz_max, zaxis)
    x, y, z = np.meshgrid(xx, yy, zz, indexing='ij')
    xm, ym, zm = affine_transform(x, y, z, np.linalg.inv(flowfield.affine))
    sample_data = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    sample_data = sample_image(img, xm, ym, zm, 'trilinear').reshape((img.shape[0], img.shape[1], img.shape[2]))

    i1 = img.shape[0]-1
    i2 = img.shape[1]-1
    i3 = img.shape[2]-1

    x_affine = np.array([[x[0,0,0], x[i1,0,0], x[i1,i2,0], x[i1,i2,i3]], 
                [y[0,0,0], y[i1,0,0], y[i1,i2,0], y[i1,i2,i3]], 
                [z[0,0,0], z[i1,0,0], z[i1,i2,0], z[i1,i2,i3]], 
                [1,1,1,1]])

    v = np.array([[0,i1+1,i1+1,i1+1],
                [0,0,i2+1,i2+1],
                [0,0,0,i3+1],
                [1,1,1,1]])

    v = np.linalg.pinv(v)

    vff = x_affine@v
    vff = np.round(vff)
    vff[vff==0]=0

    return sample_data, vff
