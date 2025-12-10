# test flatmap
"""
test for isolation module 
@author: jdiedrichsen
"""

import SUITPy as suit

def test_normalize():
    T1_file = 'tutorials/sub-ex_T1w.nii.gz'
    mask_file = 'tutorials/sub-ex_T1w_cerebellum_dseg.nii.gz'
    suit.normalize(T1_file,mask_file,space='SUIT')
    pass

if __name__ == '__main__':
    # make_shapes()
    test_normalize()
    # test_plot_label(render='matplotlib')
    # test_plot_rgba(render='matplotlib')
    pass