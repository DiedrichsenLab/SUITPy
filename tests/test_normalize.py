# test flatmap
"""
test for normalization module
@author: jdiedrichsen
"""

import SUITPy as suit
import nibabel as nb

def test_normalize():
    T1_file = 'docs/source/tutorials/sub-ex_T1w.nii.gz'
    mask_file = 'docs/source/tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    suit.normalize(T1_file,mask_file,space='MNISym',verbose=1,write_jacobian_determinant=True,write_ants_transform=False,write_inv_deformation=True)
    pass

def test_bounding_box():
    T1_file = 'docs/source/tutorials/sub-ex_T1w.nii.gz'
    mask_file = 'docs/source/tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    img = suit.normalization.bounding_box(T1_file,mask_file)
    nb.save(img,'docs/source/tutorials/sub-ex_T1w_cropped.nii.gz')
    pass        

if __name__ == '__main__':
    # make_shapes()
    # test_bounding_box()
    test_normalize()
    # test_plot_label(render='matplotlib')
    # test_plot_rgba(render='matplotlib')
    pass