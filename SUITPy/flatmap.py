#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUIT toolbox flatmap module

Basic functionality for mapping and plotting functional
Data for the cerebellum

@authors jdiedrichsen, maedbhking, eliu72, dzhi1993, switt
"""

import numpy as np
import os
import sys
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
import scipy.stats as ss
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colorbar import make_axes
from matplotlib.colors import Normalize, LinearSegmentedColormap
import warnings

_base_dir = os.path.dirname(os.path.abspath(__file__))
_surf_dir = os.path.join(_base_dir, 'surfaces')


def vol_to_surf(
    volumes,
    space='SUIT',
    ignore_zeros=False,
    depths=[0,0.2,0.4,0.6,0.8,1.0],
    stats='nanmean',
    outer_surf_gifti=None,
    inner_surf_gifti=None
    ):
    """Maps volume data onto a surface, defined by inner and outer surface.

    Function enables mapping of volume-based data onto the vertices of a
    surface. For each vertex, the function samples the volume along the line
    connecting the two surfaces. The points along the line
    are specified in the variable 'depths'. default is to sample at 5
    locations between white an gray matter surface. Set 'depths' to 0 to
    sample only along the white matter surface, and to 0.5 to sample along
    the mid-gray surface.
    The averaging across the sampled points for each vertex is dictated by
    the variable 'stats'. For functional activation, use 'mean' or
    'nanmean'. For discrete label data, use 'mode'.

    Args:
        volumes (list or nib obj):
            List of filenames/nib objs, or nib obj to be mapped
        space (string):
            Normalization space: 'SUIT' (default), 'FSL', 'SPM'
        ignore_zeros (bool):
            Should zeros be ignored in mapping? default: False
        depths (array-like):
            Depths of points along line at which to map (0=white/gray, 1=pial).
            DEFAULT: [0.0,0.2,0.4,0.6,0.8,1.0]
        stats (lambda function):
            function that calculates the Statistics to be evaluated.
            'nanmean': default and used for activation data
            'mode': used when discrete labels are sampled. The most frequent label is assigned.
        outer_surf_gifti (string or nibabel.GiftiImage):
            optional pial surface, filename or loaded gifti object, overwrites space
        inner_surf_gifti (string or nibabel.GiftiImage):
            White surface, filename or loaded gifti object, overwrites space

    Returns:
        mapped_data (numpy.array):
            A Data array for the mapped data
    """
    # Get the surface files
    if inner_surf_gifti is None:
        inner_surf_gifti = f'PIAL_{space}.surf.gii'
    if outer_surf_gifti is None:
        outer_surf_gifti = f'WHITE_{space}.surf.gii'

    inner = os.path.join(_base_dir,'surfaces',inner_surf_gifti)
    inner_surf_giftiImage = nb.load(inner)
    outer = os.path.join(_base_dir,'surfaces',outer_surf_gifti)
    outer_surf_giftiImage = nb.load(outer)

    # Get the vertices and check that the numbers are the same
    c1 = inner_surf_giftiImage.darrays[0].data
    c2 = outer_surf_giftiImage.darrays[0].data
    num_verts = c1.shape[0]
    if c2.shape[0] != num_verts:
        sys.exit('Error: White and pial surfaces should have same number of vertices.')

    # Prepare the mapping
    firstGood = None
    depths = np.array(depths)
    numPoints = len(depths)

    # Make a list of the files to be mapped
    Vols = []
    if type(volumes) is not list:
        volumes = [volumes]

    # Make a list of the files to be mapped
    volsize=np.zeros((len(volumes),),dtype=int)
    for i in range(len(volumes)):
        if (type(volumes[i]) is nb.Nifti2Image) or (type(volumes[i]) is nb.Nifti1Image):
            Vols.append(volumes[i])
            firstGood = i
        else:
            try:
                a = nb.load(volumes[i])
                Vols.append(a)
                firstGood = i
            except:
                print(f'File {volumes[i]} could not be opened')
                Vols.append(None)
        if Vols[-1] is None:
            volsize[i]=0
        elif Vols[-1].ndim == 3:
            volsize[i]=1
        else:
            volsize[i]=Vols[-1].shape[3]


    if firstGood is None:
        sys.exit('Error: None of the images could be opened.')

    # Get the indices for all the points being sampled
    indices = np.zeros((numPoints,num_verts,3),dtype=int)
    for i in range(numPoints):
        c = (1-depths[i])*c1.T+depths[i]*c2.T
        ijk = nt.coords_to_voxelidxs(c,Vols[firstGood])
        indices[i] = ijk.T

    # Read the data and map it
    mapped_data = np.zeros((num_verts,volsize.sum()))
    index=volsize.cumsum()
    index=np.insert(index,0,0)
    for v,vol in enumerate(Vols):
        if vol is None:
            pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                X = vol.get_fdata()
                if ignore_zeros:
                    X[X==0] = np.nan
                data = X[indices[:,:,0],indices[:,:,1],indices[:,:,2]]
                if data.ndim == 2:
                    data = data.reshape(data.shape + (1,))
                # Determine the right statistics - if function - call it
                if stats=='nanmean':
                    mapped_data[:,index[v]:index[v+1]] = np.nanmean(data,axis=0)
                elif stats=='mode':
                    mapped_data[:,index[v]:index[v+1]],_ = ss.mode(data,axis=0,keepdims=False)
                elif callable(stats):
                    mapped_data[:,index[v]:index[v+1]] = stats(data)

    return mapped_data

def save_colorbar(
    gifti,
    outpath
    ):
    """plots colorbar for gifti object (*.label.gii)
    and saves it to outpath
    Args:
        gifti (gifti image):
            Nibabel Gifti image
        outpath (str):
            outpath for colorbar image

    """
    _, ax = plt.subplots(figsize=(1,10)) # figsize=(1, 10)

    _, cmap = nt.get_gifti_colortable(gifti)
    labels = nt.get_gifti_labels(gifti)

    bounds = np.arange(cmap.N + 1)

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb3 = mpl.colorbar.ColorbarBase(ax, cmap=cmap.reversed(cmap),
                                    norm=norm,
                                    ticks=bounds,
                                    format='%s',
                                    orientation='vertical',
                                    )
    cb3.set_ticklabels(labels[::-1])
    cb3.ax.tick_params(size=0)
    cb3.set_ticks(bounds+.5)
    cb3.ax.tick_params(axis='y', which='major', labelsize=30)

    plt.savefig(outpath, bbox_inches='tight', dpi=150)

def map_to_rgb(data,scale=None,threshold=[0,0,0]):
    """Maps data to RGB

    Args:
        data (_type_):
            List of vectors or 3xP ndarray.
            use [data,None,None] to skip color channels
        scale (list): maximum brightness
        threshold (list): Threshold [0,0,0].
    returns:
        rgba (ndarray): Nx4 array of RGBA values
    """
    if isinstance(data,np.ndarray):
        data = [data[:,0],data[:,1],data[:,2]]
    nvert = data[0].shape[0]
    rgba = np.zeros((nvert,4))
    for i,d in enumerate(data):
        if d is not None:
            rgba[:,i]=np.nan_to_num(d)
            rgba[rgba[:,i]<threshold[i],i]=0
            if scale is None:
                s = rgba[:,i].max()
            else:
                s=scale[i]
            rgba[:,i]=rgba[:,i]/s
            rgba[rgba[:,i]>1,i]=1.0
            # Remove data below threshold
    # Set below-threshold areas to nan (transparent)
    rgba[rgba[:,0:3].sum(axis=1)==0]=np.nan
    rgba[:,3]=1
    return rgba


def plot(data,
        surf=None,
        underlay='SUIT.shape.gii',
        undermap='Greys',
        underscale=[-0.5, 0.5],
        overlay_type='func',
        threshold=None,
        cmap=None,
        cscale=None,
        label_names=None,
        borders='borders.txt',
        bordercolor = 'k',
        bordersize = 2,
        alpha=1.0,
        render='matplotlib',
        hover = 'auto',
        new_figure=True,
        colorbar=False,
        cbar_tick_format="%.2g",
        backgroundcolor = 'w',
        frame = [-110,110,-110,110]):
    """Visualize cerebellar activity on a flatmap

    Args:
        data (np.array, giftiImage, or name of gifti file):
            Data to be plotted, should be a 28935x1 vector
        surf (str or giftiImage):
            surface file for flatmap (default: FLAT.surf.gii in SUIT pkg)
        underlay (str, giftiImage, or np-array):
            Full filepath of the file determining underlay coloring (default: SUIT.shape.gii in SUIT pkg)
        undermap (str)
            Matplotlib colormap used for underlay (default: gray)
        underscale (array-like)
            Colorscale [min, max] for the underlay (default: [-1, 0.5])
        overlay_type (str)
            'func': functional activation (default)
            'label': categories
            'rgb': RGB(A) values (0-1) directly specified. Alpha is optional
        threshold (scalar or 2-element array)
            Threshold for functional overlay. If one value is given, only values above are shown
            If two values are given, values below lower threshold or above upper threshold are shown
        cscale (ndarray or list)
            Colorscale [min, max] for the overlay (default: [data.min, data.max])
        cmap (str)
            A Matplotlib colormap or an equivalent Nx3 or Nx4 floating point array (N rgb or rgba values). (defaults to 'jet' if none given)
        label_names (list)
            labelnames (default is None - extracts from .label.gii )
        borders (str)
            Full filepath of the borders txt file or workbench border file (default: borders.txt in SUIT pkg)
        bordercolor (char or matplotlib.color)
            Color of border - defaults to 'k'
        bordersize (int)
            Size of the border points - defaults to 2
        alpha (float)
            Opacity of the overlay (default: 1)
        render (str)
            Renderer for graphic display 'matplot' / 'plotly'. Dafault is matplotlib
        hover (str)
            When renderer is plotly, it determines what is displayed in the hover label: 'auto', 'value', or None
        new_figure (bool)
            If False, plot renders into matplotlib's current axis. If True, it creates a new figure (default=True)
        colorbar (bool)
            By default, colorbar is not plotted into matplotlib's current axis (or new figure if new_figure is set to True)
        cbar_tick_format : str, optional
            Controls how to format the tick labels of the colorbar, and for the hover label.
            Ex: use "%i" to display as integers.
            Default='%.2g' for scientific notation.
        backgroundcolor (str or matplotlib.color):
            Axis background color (default: 'w')
        frame (ndarray): [L,R,T,B] of the area of flatmap that is rendered
            Defaults to entire flatmap

    Returns:
        ax (matplotlib.axis)
            If render is matplotlib, the function returns the axis
        fig (plotly.go.Figure)
            If render is plotly, it returns Figure object

    """
    # default directory
    if surf is None:
        surf = os.path.join(_surf_dir,'FLAT.surf.gii')

    # load topology
    if isinstance(surf,str):
        flatsurf = nb.load(surf)
    elif isinstance(surf,nb.gifti.gifti.GiftiImage):
        flatsurf = surf
    else:
        raise ValueError('surf should be a string or giftiImage')
    vertices = flatsurf.darrays[0].data
    faces    = flatsurf.darrays[1].data
    num_vert = vertices.shape[0]

    # Load the overlay if it's a string
    if type(data) is str:
        data = nb.load(data)

    # If it is a giftiImage, figure out colormap
    if type(data) is nb.gifti.gifti.GiftiImage:
        if overlay_type == 'label':
            _, cmap = nt.get_gifti_colortable(data)
            if label_names is None:
                labels = data.labeltable
                label_names = list(labels.get_labels_as_dict().values())
        data_arr = data.darrays[0].data

    # If it's a nd array, copy data arr
    if type(data) is np.ndarray:
        data_arr = np.copy(data)

    # decide whether to map to faces
    if (render=='plotly'):
        mapfac=None    # Don't map to faces
    else:
        mapfac = faces # Map to faces

    # Determine foreground color depending on type
    if overlay_type=='label':
        # If 2d-array, take the first column only
        if data_arr.ndim>1:
            data_arr = data_arr[:,0]
        i = np.isnan(data_arr)
        data_arr = data_arr.astype(int)
        data_arr[i]=0

        # create label names if they don't exist
        if label_names is None:
            label_names = [f"L-{i:02d}" for i in range(data_arr.max()+1)]
        # map the overlay to colors:
        overlay_color, cmap, cscale = _map_color(data=data_arr,
            faces = mapfac, cscale=cscale,
            cmap=cmap, threshold=threshold)
    elif overlay_type=='func':
        if data_arr.ndim>1:
            data_arr = data_arr[:,0]
        # map the overlay to colors:
        overlay_color, cmap, cscale = _map_color(data=data_arr,
            faces = mapfac, cscale=cscale,
            cmap=cmap, threshold=threshold)
    elif overlay_type=='rgb':
        if mapfac is not None:
            data  = _map_to_face(data,mapfac)
        if data.shape[1]==3:
            overlay_color = np.c_[data,np.ones(data.shape[0],1)]

        elif data.shape[1]==4:
            overlay_color = data[:,0:4]
            alpha = data[:,3:4]
        else:
            raise(NameError('for RGB(A), the data needs to have 3 or 4 columns'))

    # Load underlay and assign color
    if underlay is None:
        underlay = np.zeros((num_vert,))
    if type(underlay) is not np.ndarray:
        if not os.path.isfile(underlay):
            underlay = os.path.join(os.path.join(_surf_dir, underlay))
        underlay = nb.load(underlay).darrays[0].data
    underlay_color,_,_ = _map_color(data=underlay,
                    faces = mapfac,
                    cscale=underscale, cmap=undermap)

    # Combine underlay and overlay: For Nan overlay, let underlay shine through
    color = underlay_color * (1-alpha) + overlay_color * alpha
    i = np.isnan(color.sum(axis=1))
    color[i,:]=underlay_color[i,:]
    color[i,3]=1.0

    # If present, get the borders
    if borders is not None:
        if not os.path.isfile(borders):
            borders = os.path.join(_surf_dir, borders)
        if borders.endswith('.txt'):
            borders = np.genfromtxt(borders, delimiter=',')
        elif borders.endswith('.border'):
            border_list,_ = nt.read_borders(borders)
            borders = [b.get_coords(flatsurf) for b in border_list]
            borders = np.vstack(borders)
        else:
            raise ValueError('borders should be a txt or border file')
    # Render with Matplotlib
    if render == 'matplotlib':
        ax = _render_matplotlib(vertices, faces, color, borders,
                                bordercolor, bordersize, new_figure,backgroundcolor,frame)
        # set up colorbar
        if colorbar:
            if overlay_type=='label':
                cbar = _colorbar_label(ax, cmap, cbar_tick_format, label_names)
            elif overlay_type=='func':
                cbar = _colorbar_func(ax, cmap, cscale, cbar_tick_format)
    elif render == 'plotly':
        if hover == 'auto':
            if overlay_type=='func':
                textlabel = _make_labels(data_arr,'{:' + cbar_tick_format[1:] + '}')
            if overlay_type=='label':
                textlabel = _make_labels(data_arr,label_names)
            if overlay_type=='rgb':
                textlabel = _make_labels(data_arr,'({:.2f},{:.2f},{:.2f})')
        if hover == 'value':
            textlabel = _make_labels(data_arr,'{:' + cbar_tick_format[1:] + '}')
        elif hover is None:
            textlabel=None

        ax = _render_plotly(vertices,faces,color,borders,
                                bordercolor,bordersize, new_figure,
                                textlabel,backgroundcolor,frame)

    else:
        raise(NameError('render needs to be matplotlib or plotly'))
    return ax

def _map_to_face(data,faces):
    numFaces = faces.shape[0]
    face_value = np.zeros((3,numFaces,4),dtype = data.dtype)
    for i in range(3):
        face_value[i,:,:] = data[faces[:,i],:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        face_value = np.nanmean(face_value, axis=0)
    return face_value


def _map_color(
    data,
    faces=None,
    cscale=None,
    cmap=None,
    threshold=None
    ):
    """Scales the values, and
    then looks up the RGB values in the color map
    If faces are provided, maps the data to faces

    Args:
        data (1d-np-array)
            Numpy Array of values to scale. If integer, if it is not scaled
        faces (nd.array)
            Array of Faces, if provided, it maps to faces
        cscale (array like)
            (min,max) of the scaling of the data
        cmap (str, or matplotlib.colors.Colormap)
            The Matplotlib colormap an equivalent Nx3 or Nx4 floating point array (N rgb or rgba values).
        threshold (scalae or array like)
            threshold for data display - only values above threshold are displayed
    Returns:
        color_data(ndarray): N x 4 ndarray
        cmap
        cscale
    """
    # if scale not given, find it
    if cscale is None:
        cscale = np.array([np.nanmin(data), np.nanmax(data)])

    # When continuous data, scale and threshold
    if data.dtype.kind == 'f':
        # if threshold is given, threshold the data
        if threshold is not None:
            if np.isscalar(threshold):
                data[data<threshold]=np.nan
            elif len(threshold)==2:
                data[(data>threshold[0]) & (data<threshold[1])]=np.nan
            else:
                raise(NameError('Threshold needs to be scalar or 2-element array'))

        # scale the data
        data = ((data - cscale[0]) / (cscale[1] - cscale[0]))

    # Map the values from vertices to faces and integrate
    if faces is not None:
        numFaces = faces.shape[0]
        face_value = np.zeros((3,numFaces),dtype = data.dtype)
        for i in range(3):
            face_value[i,:] = data[faces[:,i]]

        if data.dtype.kind == 'i':
            value,_ = ss.mode(face_value,axis=0,keepdims=False)
            value = value.reshape((numFaces,))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                value = np.nanmean(face_value, axis=0)
    else:
        value = data

    # Get the color map
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    elif type(cmap) is np.ndarray:
        cmap = ListedColormap(cmap)
    elif cmap is None:
        cmap = plt.get_cmap('jet')

    # Map the color
    color_data = cmap(value)

    # Set missing data 0 for int or NaN for float to NaN
    if data.dtype.kind == 'f':
        color_data[np.isnan(value),:]=np.nan
    elif data.dtype.kind == 'i':
        color_data[value==0,:]=np.nan

    return color_data, cmap, cscale


def _render_matplotlib(vertices,faces,face_color,borders,
                    bordercolor, bordersize,
                    new_figure,backgroundcolor,
                    frame):
    """
    Render the data in matplotlib

    Args:
        vertices (np.ndarray)
            Array of vertices
        faces (nd.array)
            Array of Faces
        face_color (nd.array)
            RGBA array of color and alpha of all vertices
        borders (np.ndarray)
            default is None
        bordercolor (char or matplotlib.color)
            Color of border
        bordersize (int)
            Size of the border points
        new_figure (bool)
            Create new Figure or render in currrent axis
        frame (ndarray)
            [L,R,B,T] of the plotted area
    Returns:
        ax (matplotlib.axes)
            Axis that was used to render the axis
    """
    if frame is None:
        frame = [np.nanmin(vertices[:,0]),
                 np.nanmax(vertices[:,0]),
                 np.nanmin(vertices[:,1]),
                 np.nanmax(vertices[:,1])]
    vertex_in = (vertices[:,0]>=frame[0]) & \
                (vertices[:,0]<=frame[1]) & \
                (vertices[:,1]>=frame[2]) & \
                (vertices[:,1]<=frame[3])
    face_in  = np.any(vertex_in[faces],axis=1)
    patches = []

    for i,f in enumerate(faces[face_in]):
        polygon = Polygon(vertices[f,0:2])
        patches.append(polygon)
    p = PatchCollection(patches)
    p.set_facecolor(face_color[face_in])
    p.set_edgecolor(face_color[face_in])
    p.set_linewidth(0.5)

    # Get the current axis and plot it
    if new_figure:
        fig = plt.figure(figsize=(7,7))
    ax = plt.gca()
    ax.add_collection(p)
    ax.set_xlim(frame[0],frame[1])
    ax.set_ylim(frame[2],frame[3])
    ax.axis('equal')
    ax.axis('off')
    fig=plt.gcf()
    fig.set_facecolor(backgroundcolor)

    if borders is not None:
        ax.plot(borders[:,0],borders[:,1],color=bordercolor,
                marker='.', linestyle=None,
                markersize=bordersize,linewidth=0)

    return ax

def _render_plotly(vertices,faces,color,borders,
            bordercolor, bordersize,
            new_figure, hovertext=None,
            backgroundcolor='#ffffff',
            frame=None):
    """
    Render the data in plotly
    Args:
        vertices (np.ndarray)
            Array of vertices
        faces (nd.array)
            Array of Faces
        face_color (nd.array)
            RGBA array of color and alpha of all vertices
        borders (np.ndarray)
            default is None
        bordercolor (char or matplotlib.color)
            Color of border
        bordersize (int)
            Size of the border points
        new_figure (bool)
            Create new Figure or render in currrent axis
        hovertext (list of str)
            Text for hovering for each vertex
        frame (ndarray)
            [L,R,B,T] of the plotted area

    Returns:
        ax (matplotlib.axes)
            Axis that was used to render the axis
    """
    # Check whether to color faces or vertices:
    if color.shape[0]==vertices.shape[0]:
        vertcolor=color
        facecolor=None
    elif color.shape[0]==faces.shape[0]:
        vertcolor=None
        facecolor=color
    else:
        raise(NameError('Color data not the correct shape'))

    if hovertext is None:
        hi = 'skip'
    else:
        hi = 'text'
    traces = []
    colorbar = dict(exponentformat='none',tickformat='%.2f')

    if frame is None:
        frame = [np.nanmin(vertices[:,0]),
                 np.nanmax(vertices[:,0]),
                 np.nanmin(vertices[:,1]),
                 np.nanmax(vertices[:,1])]
    aspect_ratio = (frame[3]-frame[2])/(frame[1]-frame[0])

    traces.append(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor = facecolor,
        vertexcolor = vertcolor,
        lightposition=dict(x=0, y=0, z=2.5),
        text = hovertext,
        hoverinfo=hi))
    if borders is not None:
        bordercolor=_color_matplotlib_to_plotly(bordercolor)
        traces.append(go.Scatter3d(
                x=borders[:,0],
                y=borders[:,1],
                z=np.ones((borders.shape[0],))*0.01,
                marker = dict(
                    color=bordercolor,
                    size=bordersize,
                    symbol='circle'),
                mode='markers',
                hoverinfo=None))


    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=1.1)
    )
    xaxis_dict= dict(visible=False,
        showbackground=False,
        showline=False,
        showgrid=False,
        showspikes=False,
        showticklabels=False,
        title=None,
        range=frame[:2])
    yaxis_dict= xaxis_dict.copy()
    yaxis_dict['range']=frame[2:]
    zaxis_dict= xaxis_dict.copy()
    scene = dict(xaxis=xaxis_dict,
                yaxis=yaxis_dict,
                zaxis=zaxis_dict,
                aspectmode= 'manual')
                # aspectratio=dict(x=1, y=1, z=0.1))
    backgroundcolor=_color_matplotlib_to_plotly(backgroundcolor)
    fig = go.Figure(data=traces,)
    fig.update_layout(scene_camera=camera,
                dragmode=False,
                margin=dict(r=0, l=0, b=0, t=0),
                scene = scene,
                width=400,
                height=400*aspect_ratio,
                paper_bgcolor=backgroundcolor
                )
    return fig

def _make_labels(data,labelstr):
    numvert=data.shape[0]
    labels = np.empty((data.shape[0],),dtype=object)
    if type(labelstr) is str:
        for i in range(numvert):
            if data.ndim==1:
                labels[i]=labelstr.format(data[i])
            else:
                labels[i]=labelstr.format(*data[i])
    else:
        for i in range(numvert):
            labels[i]=labelstr[data[i]]
    return labels

def _colorbar_label(
    ax,
    cmap,
    cbar_tick_format,
    label_names
    ):
    """adds colorbar to figure

    Args:
        ax (matplotlib.axes.Axes)
            Pre-existing axes for the plot.
        cmap (str, or matplotlib.colors.Colormap)
            The Matplotlib colormap
        cbar_tick_format : str, optional
            Controls how to format the tick labels of the colorbar.
            Ex: use "%i" to display as integers.
            Default='%.2g' for scientific notation.
        label_names (list)
            List of strings for label names

    Returns:
        cbar (matplotlib.colorbar)
            Colorbar object
    """
    N = len(label_names) # Length of the colormap
    cax, _ = make_axes(ax, location='right', fraction=.15,
                        shrink=.5, pad=.0, aspect=10.)
    norm = Normalize(vmin=0, vmax=cmap.N)
    ticks = np.arange(0,N)+0.5
    bounds = np.arange(0,N+1)
    proxy_mappable = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        proxy_mappable, cax=cax, ticks=ticks,
        boundaries=bounds, spacing='proportional',
        format=cbar_tick_format, orientation='vertical')

    cbar.ax.set_yticklabels(label_names)

    return cbar

def _colorbar_func(
    ax,
    cmap,
    cscale,
    cbar_tick_format
    ):
    """adds colorbar to figure

    Args:
        ax (matplotlib.axes.Axes)
            Pre-existing axes for the plot.
        cmap (str, or matplotlib.colors.Colormap)
            The Matplotlib colormap
        cscale (array like)
            (min,max) of the scaling of the data
        cbar_tick_format : str, optional
            Controls how to format the tick labels of the colorbar.
            Ex: use "%i" to display as integers.
            Default='%.2g' for scientific notation.

    @author: maedbhking
    """
    nb_ticks = 5
    # ...unless we are dealing with integers with a small range
    # in this case, we reduce the number of ticks
    if cbar_tick_format == "%i" and cscale[1] - cscale[0] < nb_ticks:
        ticks = np.arange(cscale[0], cscale[1] + 1)
    else:
        ticks = np.linspace(cscale[0], cscale[1], nb_ticks)

    # set up colorbar
    cax, kw = make_axes(ax, location='right', fraction=.15,
                        shrink=.5, pad=.0, aspect=10.)
    bounds = np.linspace(cscale[0], cscale[1], cmap.N)
    norm = Normalize(vmin=cscale[0], vmax=cscale[1])

    proxy_mappable = ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(
        proxy_mappable, cax=cax, ticks=ticks,
        boundaries=bounds, spacing='proportional',
        format=cbar_tick_format, orientation='vertical')

    return cbar


def _color_matplotlib_to_plotly(color):
    """Transforms Matplotlib color string to
    plotly/html hexadecimal representation

    Args:
        color (str): color string
    Returns:
        colorhex (str): web-based color, e.g., '#000000'
    """
    if color=='k':
        color='#000000'
    if color=='w':
        color='#ffffff'
    return color