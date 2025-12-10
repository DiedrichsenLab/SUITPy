# test reslice
"""
test for reslice module 
Also test correctness of ANTS deformation file
"""

import SUITPy as suit
import nibabel as nb
def test_reslice_anatomical():
    T1_file = 'docs/source/tutorials/sub-ex_T1w.nii.gz'
    def_file = 'docs/source/tutorials/sub-ex_T1w_to-SUIT_mode-image_xfm.nii.gz'
    mask_file = 'docs/source/tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    img = suit.reslice_image(T1_file,def_file,mask=mask_file)
    nb.save(img,'docs/source/tutorials/sub-ex_T1w_resliced.nii.gz')
    pass

if __name__ == '__main__':
    # make_shapes()
    test_reslice_anatomical()
    pass

