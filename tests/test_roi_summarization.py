#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:40 2025

@author: vashkani, jdiedrichsen
"""

from SUITPy.atlas import summarize_data
import os
base_dir = '/Users/jdiedrichsen/Python/SUITPy'

def test_roi_summarize():
    image_file = os.path.join(base_dir,'docs','source','tutorials','MDTB08_Math.nii')

    df = summarize_data(
        images=[image_file],
        atlas="Buckner_2011",
        maps="atl-Buckner7",
        space="SUIT",
        stats=("mean", "std", "abs"),
        outfilename=None
    )
    print(df.head())
    pass


if __name__ == "__main__":
    # test other functions if needed
    test_roi_summarize()
    pass
