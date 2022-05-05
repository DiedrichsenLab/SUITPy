# test flatmap
"""
Unit test for flatmap unit
@author: jdiedrichsen
"""
import unittest
import SUITPy.flatmap as flatmap
import numpy as np
import plotly.graph_objs as go


def test_flatmap_plot():
    ax = flatmap.plot('docs/source/notebooks/MDTB08_Math.func.gii',threshold=[0.01, 0.12],
    bordersize=3,bordercolor='k',backgroundcolor='w',render='plotly')
    ax.show()
    # ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    # ax.show()
    pass


if __name__ == '__main__':
    # make_shapes()
    test_flatmap_plot()