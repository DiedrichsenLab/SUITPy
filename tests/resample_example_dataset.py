#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: joerndiedrichsen
"""

import SUITPy as suit
import nilearn.image as ni
import nibabel as nb 


if __name__ == "__main__":
    # test other functions if needed
    A = nb.load('avg_t1_raw.nii')

    
    aff = np.eye(4)