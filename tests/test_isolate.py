# test flatmap
"""
test for isolation module 
@author: jdiedrichsen
"""

import SUITPy as suit

def test_isolate():
    T1_path = '/Users/jdiedrichsen/Dropbox/projects/SUIT_test/subj_01/S11_T1w.nii'
    suit.isolate(T1_path)
    pass

if __name__ == '__main__':
    # make_shapes()
    test_isolate()
    # test_plot_label(render='matplotlib')
    # test_plot_rgba(render='matplotlib')
    pass