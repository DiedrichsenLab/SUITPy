# test flatmap
"""
test for isolation module 
@author: jdiedrichsen
"""

import SUITPy as suit

def test_isolate():
    T1_file = 'docs/source/notebooks/anatomical_sess-01.nii'
    suit.isolate(T1_file,save_cropped_files=True)
    pass

if __name__ == '__main__':
    # make_shapes()
    test_isolate()
    # test_plot_label(render='matplotlib')
    # test_plot_rgba(render='matplotlib')
    pass