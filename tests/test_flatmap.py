# test flatmap
"""
Unit test for flatmap unit
@author: jdiedrichsen
"""
import unittest
import SUITPy.flatmap as flatmap
import numpy as np
import plotly.graph_objs as go
import nibabel as nb


def test_flatmap_plot():
    ax = flatmap.plot('docs/source/notebooks/MDTB08_Math.func.gii',threshold=[0.01, 0.12],
    bordersize=3,bordercolor='k',backgroundcolor='w',render='plotly')
    ax.show()
    # ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    # ax.show()
    pass

def test_plot_label():
    fname = 'docs/source/notebooks/Buckner_17Networks.label.gii'
    fig=flatmap.plot(fname, overlay_type='label',new_figure=True, colorbar=True,
    cscale=[1,5],render='plotly')
    fig.show()

def test_plot_rgba():
    A = nb.load('docs/source/notebooks/MDTB08_Math.func.gii')
    nvert = A.darrays[0].data.shape[0]
    data = np.zeros((nvert,4))
    data[:,0]=A.darrays[0].data
    data[:,0]=data[:,0]/np.max(data[:,0])
    data[data[:,0]<0.3,0]=0
    data[:,1]=-A.darrays[0].data
    data[:,1]=data[:,1]/np.max(data[:,1])
    data[data[:,1]<0.3,1]=0
    data[data[:,0:3].sum(axis=1)==0]=np.nan
    data[:,2]=data[:,0]
    data[:,3]=1
    ax = flatmap.plot(data,
                      overlay_type='rgb',
                        bordersize=3,bordercolor='k',backgroundcolor='w',render='plotly')
    ax.show()
    # ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    # ax.show()
    pass


if __name__ == '__main__':
    # make_shapes()
    # test_flatmap_plot()
    # test_plot_label()
    test_plot_rgba()
    pass