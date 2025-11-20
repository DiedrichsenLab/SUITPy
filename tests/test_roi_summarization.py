#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:40 2025

@author: vashkani
"""

from SUITPy.atlas import summarize_data

def test_roi_summarize():
    image_file = "/home/UWO/vashkani/Github/anatomical_sess-01_crop_space-SUIT_resliced_test.nii.gz"

    df = summarize_data(
        images=[image_file],
        atlas="Buckner_2011",
        maps="atl-Buckner7",
        space="SUIT",
        stats=("mean", "std", "abs"),
        outfilename="buckner7_ROI_summary.txt"
    )

    print(df.head())
    pass


if __name__ == "__main__":
    # test other functions if needed
    test_roi_summarize()
    pass
