# test flatmap
"""
Unit test for flatmap unit
@author: jdiedrichsen
"""
import unittest 
import SUITPy.flatmap as flatmap
import numpy as np 

def test_flatmap_plot(): 
    ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    pass
    

if __name__ == '__main__':
    test_flatmap_plot()