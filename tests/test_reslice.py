# test reslice
"""
test for reslice module
Also test correctness of ANTS deformation file
"""

import SUITPy as suit
import nibabel as nb
def test_reslice_anatomical():
    """This uses the forward deformation"""
    T1_file = 'docs/source/tutorials/sub-ex_T1w.nii.gz'
    def_file = 'docs/source/tutorials/sub-ex_T1w_to-SUIT_mode-image_xfm.nii.gz'
    mask_file = 'docs/source/tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    img = suit.reslice_image(T1_file,def_file,mask=mask_file)
    nb.save(img,'docs/source/tutorials/sub-ex_T1w_resliced.nii.gz')
    pass

def test_reslice_functional():
    """This uses the forward deformation"""
    func_file = 'docs/source/tutorials/sub-ex_task-fingerseq.nii.gz'
    def_file = 'docs/source/tutorials/sub-ex_T1w_to-SUIT_mode-image_xfm.nii.gz'
    mask_file = 'docs/source/tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    img = suit.reslice_image(func_file,def_file,mask=mask_file,voxelsize=2)
    nb.save(img,'docs/source/tutorials/sub-ex_task-fingerseq_space-SUIT.nii.gz')
    pass

def test_reslice_suit():
    """ This is a test for the inverse deformation"""
    T1_file = 'SUITPy/templates/tpl-SUIT_T1w.nii.gz'
    def_file = 'docs/source/tutorials/sub-ex_T1w_from-SUIT_mode-image_xfm.nii.gz'
    img = suit.reslice_image(T1_file,def_file)
    nb.save(img,'docs/source/tutorials/tpl-SUIT_space-ex_resliced.nii.gz')
    pass


if __name__ == '__main__':
    # make_shapes()
    test_reslice_anatomical()
    test_reslice_functional()

    # This is for testing the inverse transformation:
    # test_reslice_suit()
    pass

