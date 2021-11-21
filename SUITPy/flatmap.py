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
import matplotlib.pyplot as plt
import matplotlib as mpl
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

def affine_transform(
    x1, 
    x2, 
    x3, 
    M
    ):
    """
    Returns affine transform of x

    Args:
        x1 (np-array):
            X-coordinate of original
        x2 (np-array):
            Y-coordinate of original
        x3 (np-array):
            Z-coordinate of original
        M (2d-array):
            4x4 transformation matrix

    Returns:
        x1 (np-array):
            X-coordinate of transform
        x2 (np-array):
            Y-coordinate of transform
        x3 (np-array):
            Z-coordinate of transform

    """
    y1 = np.multiply(M[0,0],x1) + np.multiply(M[0,1],x2) + np.multiply(M[0,2],x3) + M[0,3]
    y2 = np.multiply(M[1,0],x1) + np.multiply(M[1,1],x2) + np.multiply(M[1,2],x3) + M[1,3]
    y3 = np.multiply(M[2,0],x1) + np.multiply(M[2,1],x2) + np.multiply(M[2,2],x3) + M[2,3]
    return (y1,y2,y3)

def coords_to_voxelidxs(
    coords,
    vol_def
    ):
    """
    Maps coordinates to linear voxel indices

    Args:
        coords (3*N matrix or 3xPxQ array):
            (x,y,z) coordinates
        vol_def (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)

    Returns:
        linidxsrs (np.ndarray):
            N-array or PxQ matrix of Linear voxel indices
    """
    mat = np.array(vol_def.affine)

    # Check that coordinate transformation matrix is 4x4
    if (mat.shape != (4,4)):
        sys.exit('Error: Matrix should be 4x4')

    rs = coords.shape
    if (rs[0] != 3):
        sys.exit('Error: First dimension of coords should be 3')

    if (np.size(rs) == 2):
        nCoordsPerNode = 1
        nVerts = rs[1]
    elif (np.size(rs) == 3):
        nCoordsPerNode = rs[1]
        nVerts = rs[2]
    else:
        sys.exit('Error: Coordindates have %d dimensions, not supported'.format(np.size(rs)))

    # map to 3xP matrix (P coordinates)
    coords = np.reshape(coords,[3,-1])
    coords = np.vstack([coords,np.ones((1,rs[1]))])

    ijk = np.linalg.solve(mat,coords)
    ijk = np.rint(ijk)[0:3,:]
    # Now set the indices out of range to -1
    for i in range(3):
        ijk[i,ijk[i,:]>=vol_def.shape[i]]=-1
    return ijk

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

    if firstGood is None:
        sys.exit('Error: None of the images could be opened.')

    # Get the indices for all the points being sampled
    indices = np.zeros((numPoints,num_verts,3),dtype=int)
    for i in range(numPoints):
        c = (1-depths[i])*c1.T+depths[i]*c2.T
        ijk = coords_to_voxelidxs(c,Vols[firstGood])
        indices[i] = ijk.T

    # Read the data and map it
    data = np.zeros((numPoints,num_verts))
    mapped_data = np.zeros((num_verts,len(Vols)))
    for v,vol in enumerate(Vols):
        if vol is None:
            pass
        else:
            X = vol.get_data()
            if ignore_zeros:
                X[X==0] = np.nan
            for p in range(numPoints):
                data[p,:] = X[indices[p,:,0],indices[p,:,1],indices[p,:,2]]
                outside = (indices[p,:,:]<0).any(axis=1) # These are vertices outside the volume
                data[p,outside] = np.nan

            # Determine the right statistics - if function - call it
            if stats=='nanmean':
                mapped_data[:,v] = np.nanmean(data,axis=0)
            elif stats=='mode':
                mapped_data[:,v],_ = ss.mode(data,axis=0)
            elif callable(stats):
                mapped_data[:,v] = stats(data)

    return mapped_data

def make_func_gifti(
    data,
    anatomical_struct='Cerebellum',
    column_names=[]
    ):
    """Generates a function GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        column_names (list):
            List of strings for names for columns

    Returns:
        FuncGifti (GiftiImage): functional Gifti Image
    """
    num_verts, num_cols = data.shape
    #
    # Make columnNames if empty
    if len(column_names)==0:
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    C = nb.gifti.GiftiMetaData.from_dict({
    'AnatomicalStructurePrimary': anatomical_struct,
    'encoding': 'XML_BASE64_GZIP'})

    E = nb.gifti.gifti.GiftiLabel()
    E.key = 0
    E.label= '???'
    E.red = 1.0
    E.green = 1.0
    E.blue = 1.0
    E.alpha = 0.0

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_NONE',
            datatype='NIFTI_TYPE_FLOAT32',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)

    return gifti

def make_label_gifti(
                    data,
                    anatomical_struct='Cerebellum',
                    label_names=[],
                    column_names=[],
                    label_RGBA=[]
                    ):
    """Generates a label GiftiImage from a numpy array

    Args:
        data (np.array):
             num_vert x num_col data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        label_names (list):
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors

    Returns:
        gifti (GiftiImage): Label gifti image

    """
    num_verts, num_cols = data.shape
    num_labels = len(np.unique(data))

    # check for 0 labels
    zero_label = 0 in data

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if len(column_names) == 0:
        for i in range(num_cols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if len(label_RGBA) == 0:
        hsv = plt.cm.get_cmap('hsv',num_labels)
        color = hsv(np.linspace(0,1,num_labels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(num_labels)]
        label_RGBA = np.zeros([num_labels,4])
        for i in range(num_labels):
            label_RGBA[i] = color[i]
        if zero_label:
            label_RGBA = np.vstack([[0,0,0,1], label_RGBA[1:,]])

    # Create label names
    if len(label_names) == 0:
        idx = 0
        if not zero_label:
            idx = 1
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i + idx))

    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    num_labels = np.arange(num_labels)
    E_all = []
    for (label, rgba, name) in zip(num_labels, label_RGBA, label_names):
        E = nb.gifti.gifti.GiftiLabel()
        E.key = label
        E.label= name
        E.red = rgba[0]
        E.green = rgba[1]
        E.blue = rgba[2]
        E.alpha = rgba[3]
        E.rgba = rgba[:]
        E_all.append(E)

    D = list()
    for i in range(num_cols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_UINT8',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti

def get_gifti_column_names(gifti):
    """
    Returns the column names from a gifti file (*.label.gii or *.func.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image 

    Returns:
        names (list):
            List of column names from gifti object attribute data arrays

    """
    N = len(gifti.darrays)
    names = []
    for n in range(N):
        for i in range(len(gifti.darrays[n].meta.data)):
            if 'Name' in gifti.darrays[n].meta.data[i].name:
                names.append(gifti.darrays[n].meta.data[i].value)
    return names

def get_gifti_colortable(gifti,ignore_0=True):
    """Returns the RGBA color table and matplotlib cmap from gifti object (*.label.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image 

    Returns:
        rgba (np.ndarray):
            N x 4 of RGB values
        
        cmap (mpl obj):
            matplotlib colormap

    """
    labels = gifti.labeltable.labels

    rgba = np.zeros((len(labels),4))
    for i,label in enumerate(labels):
        rgba[i,] = labels[i].rgba
    
    if ignore_0:
        rgba = rgba[1:]
        labels = labels[1:]

    cmap = LinearSegmentedColormap.from_list('mylist', rgba, N=len(rgba))
    mpl.cm.register_cmap("mycolormap", cmap)

    return rgba, cmap

def get_gifti_anatomical_struct(gifti):
    """
    Returns the primary anatomical structure for a gifti object (*.label.gii or *.func.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image 

    Returns:
        anatStruct (string):
            AnatomicalStructurePrimary attribute from gifti object

    """
    N = len(gifti._meta.data)
    anatStruct = []
    for i in range(N):
        if 'AnatomicalStructurePrimary' in gifti._meta.data[i].name:
            anatStruct.append(gifti._meta.data[i].value)
    return anatStruct

def get_gifti_labels(gifti):
    """Returns labels from gifti object (*.label.gii)

    Args:
        gifti (gifti image):
            Nibabel Gifti image 

    Returns:
        labels (list):
            labels from gifti object
    """
    # labels = img.labeltable.get_labels_as_dict().values()
    label_dict = gifti.labeltable.get_labels_as_dict()
    labels = list(label_dict.values())
    return labels

def save_colorbar(
    gifti, 
    outpath
    ):
    """plots colorbar for gifti object (*.label.gii)
        
    Args:
        gifti (gifti image):
            Nibabel Gifti image 
        outpath (str):
            outpath for colorbar
    
    """
    _, ax = plt.subplots(figsize=(1,10)) # figsize=(1, 10)

    _, cmap = get_gifti_colortable(gifti)
    labels = get_gifti_labels(gifti)

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

def plot(
        data, surf=None, underlay='SUIT.shape.gii',
        undermap='Greys', underscale=[-1, 0.5], overlay_type='func', threshold=None,
        cmap=None, label_names=None, cscale=None, borders='borders.txt', alpha=1.0,
        outputfile=None, render='matplotlib', new_figure=False, colorbar=False, cbar_tick_format="%.2g"
        ):
    """
    Visualize cerebellar activity on a flatmap

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
            'func': functional activation 'label': categories 'rgb': RGB values (default: func)
        threshold (scalar or array-like)
            Threshold for functional overlay. If one value is given, it is used as a positive threshold.
            If two values are given, an positive and negative threshold is used.
        cmap (str)
            Matplotlib colormap used for overlay (defaults to 'jet' if none given)
        label_names (list)
            labelnames for .label.gii (default is None)
        borders (str)
            Full filepath of the borders txt file (default: borders.txt in SUIT pkg)
        cscale (int array)
            Colorscale [min, max] for the overlay, valid input values from -1 to 1 (default: [overlay.max, overlay.min])
        alpha (float)
            Opacity of the overlay (default: 1)
        outputfile (str)
            Name / path to file to save figure (default None)
        render (str)
            Renderer for graphic display 'matplot' / 'opengl'. Dafault is matplotlib
        new_figure (bool)
            By default, flatmap.plot renders the color map into matplotlib's current axis. It new_figure is set to True is will create a new figure
        colorbar (bool)
            By default, colorbar is not plotted into matplotlib's current axis (or new figure if new_figure is set to True)
        cbar_tick_format : str, optional
            Controls how to format the tick labels of the colorbar.
            Ex: use "%i" to display as integers.
            Default='%.2g' for scientific notation.

    Returns:
        ax (matplotlib.axis)
            If render is matplotlib, the function returns the axis

    """
    # default directory
    if surf is None:
        surf = os.path.join(_surf_dir,'FLAT.surf.gii')

    # load topology
    flatsurf = nb.load(surf)
    vertices = flatsurf.darrays[0].data
    faces    = flatsurf.darrays[1].data

    # Load the overlay if it's a string
    if type(data) is str:
        data = nb.load(data)

    # If it is a giftiImage, figure out colormap
    if type(data) is nb.gifti.gifti.GiftiImage:
        if overlay_type == 'label':
            _, cmap = get_gifti_colortable(data)
            if label_names is None:
                labels = data.labeltable
                label_names = list(labels.get_labels_as_dict().values())
        data_arr = data.darrays[0].data

    # If it's a nd array, copy `data` arr
    if type(data) is np.ndarray:
        data_arr = np.copy(data)

    # If 2d-array, take the first column only
    if data_arr.ndim>1:
        data_arr = data_arr[:,0]
    # depending on data type - type cast into int
    if overlay_type=='label':
        i = np.isnan(data_arr)
        data_arr = data_arr.astype(int)
        data_arr[i]=0

    # create label names if they don't exist
    if overlay_type=='label' and label_names is None:
        num_labels = len(np.unique(data_arr))
        label_names = []
        # check for 0 labels
        zero_label = 0 in data_arr
        idx = 1
        if zero_label:
            idx = 0
        for i in range(num_labels):
            label_names.append("label-{:02d}".format(i + idx))

    # map the overlay to the faces
    overlay_color, cmap, cscale = _map_color(faces=faces, data=data_arr, cscale=cscale, cmap=cmap, threshold=threshold)

    # Load underlay and assign color
    if type(underlay) is not np.ndarray:
        if not os.path.isfile(underlay):
            underlay = os.path.join(os.path.join(_surf_dir, underlay))
        underlay = nb.load(underlay).darrays[0].data
    underlay_color,_,_ = _map_color(faces=faces, data=underlay, cscale=underscale, cmap=undermap)

    # Combine underlay and overlay: For Nan overlay, let underlay shine through
    face_color = underlay_color * (1-alpha) + overlay_color * alpha
    i = np.isnan(face_color.sum(axis=1))
    face_color[i,:]=underlay_color[i,:]
    face_color[i,3]=1.0

    # If present, get the borders
    if borders is not None:
        if not os.path.isfile(borders):
            borders = os.path.join(os.path.join(_surf_dir, borders))
        borders = np.genfromtxt(borders, delimiter=',')

    # Render with Matplotlib
    ax = _render_matplotlib(vertices, faces, face_color, borders, new_figure)

    # set up colorbar
    if colorbar:
        if overlay_type=='label':
            cbar = _colorbar_label(ax, cmap, cscale, cbar_tick_format, label_names)
        elif overlay_type=='func':
            cbar = _colorbar_func(ax, cmap, cscale, cbar_tick_format)

    return ax

def _map_color(
    faces,
    data,
    cscale=None,
    cmap=None,
    threshold=None
    ):
    """
    Maps data from vertices to faces, scales the values, and
    then looks up the RGB values in the color map

    Args:
        faces (nd.array)
            Array of Faces
        data (1d-np-array)
            Numpy Array of values to scale. If integer, if it is not scaled
        cscale (array like)
            (min,max) of the scaling of the data
        cmap (str, or matplotlib.colors.Colormap)
            The Matplotlib colormap
        threshold (array like)
            (lower, upper) threshold for data display -
             only data x<lower and x>upper will be plotted
            if one value is given (-inf) is assumed for the lower

    """

    # When continuous data, scale and threshold
    if data.dtype.kind == 'f':
        # if threshold is given, threshold the data
        if threshold is not None:
            if np.isscalar(threshold):
                threshold=np.array([-np.inf,threshold])
            data[~np.logical_and(data>threshold[0], data<threshold[1])]=np.nan

        # if scale not given, find it
        if cscale is None:
            cscale = np.array([np.nanmin(data), np.nanmax(data)])

        # scale the data
        data = ((data - cscale[0]) / (cscale[1] - cscale[0]))

    elif data.dtype.kind == 'i':
        if cscale is None:
            cscale = np.array([np.nanmin(data), np.nanmax(data)])

    # Map the values from vertices to faces and integrate
    numFaces = faces.shape[0]
    face_value = np.zeros((3,numFaces),dtype = data.dtype)
    for i in range(3):
        face_value[i,:] = data[faces[:,i]]

    if data.dtype.kind == 'i':
        face_value,_ = ss.mode(face_value,axis=0)
        face_value = face_value.reshape((numFaces,))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            face_value = np.nanmean(face_value, axis=0)

    # Get the color map
    if type(cmap) is str:
        cmap = plt.get_cmap(cmap)
    elif type(cmap) is np.ndarray:
        cmap = ListedColormap(cmap)
    elif cmap is None:
        cmap = plt.get_cmap('jet')

    # Map the color
    color_data = cmap(face_value)

    # Set missing data 0 for int or NaN for float to NaN
    if data.dtype.kind == 'f':
        color_data[np.isnan(face_value),:]=np.nan
    elif data.dtype.kind == 'i':
        color_data[face_value==0,:]=np.nan

    return color_data, cmap, cscale

def _colorbar_label(
    ax,
    cmap,
    cscale,
    cbar_tick_format,
    label_names
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
        label_names (list)
            List of strings for label names

    Returns:
        cbar (matplotlib.colorbar)
            Colorbar object
    """
    # check if there is a 0 label and adjust colorbar accordingly
    if cscale[0]==0:
        cmap.N = cmap.N-1
        label_names = label_names[1:]
    # set up colorbar
    cax, _ = make_axes(ax, location='right', fraction=.15,
                        shrink=.5, pad=.0, aspect=10.)
    norm = Normalize(vmin=cscale[0], vmax=cscale[1])
    ticks = np.arange(1,len(label_names)+1)+0.5
    bounds = np.arange(1,len(label_names)+2)
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

def _render_matplotlib(vertices,faces,face_color,borders,new_figure):
    """
    Render the data in matplotlib: This is segmented to allow for openGL renderer

    Args:
        vertices (np.ndarray)
            Array of vertices
        faces (nd.array)
            Array of Faces
        face_color (nd.array)
            RGBA array of color and alpha of all vertices
        borders (np.ndarray)
            default is None
        new_figure (bool)
            Create new Figure or render in currrent axis

    Returns:
        ax (matplotlib.axes)
            Axis that was used to render the axis
    """
    patches = []
    for i in range(faces.shape[0]):
        polygon = Polygon(vertices[faces[i],0:2], True)
        patches.append(polygon)
    p = PatchCollection(patches)
    p.set_facecolor(face_color)
    p.set_linewidth(0.0)

    # Get the current axis and plot it
    if new_figure:
        fig = plt.figure(figsize=(7,7))
    ax = plt.gca()
    ax.add_collection(p)
    xrang = [np.nanmin(vertices[:,0]),np.nanmax(vertices[:,0])]
    yrang = [np.nanmin(vertices[:,1]),np.nanmax(vertices[:,1])]

    ax.set_xlim(xrang[0],xrang[1])
    ax.set_ylim(yrang[0],yrang[1])
    ax.axis('equal')
    ax.axis('off')

    if borders is not None:
        ax.plot(borders[:,0],borders[:,1],color='k',
                marker='.', linestyle=None,
                markersize=2,linewidth=0)
    return ax

