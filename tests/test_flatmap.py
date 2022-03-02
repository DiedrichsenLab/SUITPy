# test flatmap
"""
Unit test for flatmap unit
@author: jdiedrichsen
"""
import unittest
import SUITPy.flatmap as flatmap
import numpy as np
import plotly.graph_objs as go

def make_shapes():
    sc=[]
    sc.append(go.Scatter(x=[0,1,2,0], y=[0,2,0,0],
            fill="toself",
            fillcolor='rgba(0,0,255,1)',
            line = dict(width=0),
            mode='lines'
            ))
    sc.append(go.Scatter(x=[1,2,3,1], y=[2,3,3,2],
            fill="toself",
            fillcolor="#000A00",
            line = dict(width=0),
            mode='lines'
            ))
    fig = go.Figure(sc)
    fig.show()
    pass

def test_flatmap_plot():
    ax = flatmap.plot('docs/source/notebooks/Buckner_17Networks.label.gii',overlay_type='label',new_figure=True, colorbar=False,render='plotly')
    pass


if __name__ == '__main__':
    make_shapes()
    # test_flatmap_plot()