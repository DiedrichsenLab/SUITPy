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


def test_flatmap_plot(render='plotly'):
    ax = flatmap.plot('docs/source/notebooks/MDTB08_Math.func.gii',
                      threshold=[-0.05,0.05],bordersize=3,bordercolor='k',backgroundcolor='w',render=render,colorbar=True)
    #  ax.show()
    # ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    # ax.show()
    pass

def test_plot_label(render='plotly'):
    fname = 'docs/source/notebooks/Buckner_17Networks.label.gii'
    fig=flatmap.plot(fname, overlay_type='label',new_figure=True,
    cscale=[1,5],render=render,colorbar=True)
    fig.show()

def test_plot_rgba(render='plotly'):
    fname = ['MDTB08_Math.func.gii','MDTB04_Action_Observation.func.gii','MDTB16_Finger_Sequence.func.gii']
    nvert = 28935
    data = np.zeros((nvert,4))
    for i,f in enumerate(fname):
        A = nb.load('docs/source/notebooks/'+f)
        data[:,i]=np.nan_to_num(A.darrays[0].data)
        data[:,i]=data[:,i]/np.max(data[:,i])
        data[data[:,i]<0.3,i]=0
    data[data[:,0:3].sum(axis=1)==0]=np.nan
    data[:,3]=1
    ax = flatmap.plot(data,
                      overlay_type='rgb',
                        bordersize=3,bordercolor='k',backgroundcolor='w',render=render)
    ax.show()
    # ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    # ax.show()
    pass


if __name__ == '__main__':
    # make_shapes()
    test_flatmap_plot(render='matplotlib')
    # test_plot_label(render='matplotlib')
    # test_plot_rgba(render='matplotlib')
    pass