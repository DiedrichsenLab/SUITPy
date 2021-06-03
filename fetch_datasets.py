#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob

def atlases():
    """fetch SUIT atlases (*.nii and *.nii.gz)

    Returns:   
        dictionary containing keys `img_dir` (fullpath to image dir) 
        and `images` (list of str)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    atlas_dir = os.path.join(base_dir, 'atlasesSUIT')
    os.chdir(atlas_dir)

    # get niftis
    fpaths = glob.glob('*.nii')
    fpaths.extend(glob.glob('*.nii.gz'))

    return {'img_dir': atlas_dir, 'images': fpaths}

def surfaces():
    """fetch SUIT surfaces (*.label.gii and *.func.gii)

    Returns:   
        dictionary containing keys `img_dir` (fullpath to image dir) 
        and `images` (list of str)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    surf_dir = os.path.join(base_dir, 'surfaces')
    os.chdir(surf_dir)

    # get surfacs
    fpaths = glob.glob('*.gii')

    return {'img_dir': surf_dir, 'images': fpaths}

def contrasts():
    """fetch SUIT contrasts (*.nii) from MDTB dataset

    Returns:   
        dictionary containing keys `img_dir` (fullpath to image dir) 
        and `images` (list of str)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    con_dir = os.path.join(base_dir, 'functionalMapsSUIT')
    os.chdir(con_dir)
    
    # get surfacs
    fpaths = glob.glob('*.nii')

    return {'img_dir': con_dir, 'images': fpaths}
