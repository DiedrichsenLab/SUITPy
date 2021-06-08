#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUIT toolbox flatmap module

Basic functionality for mapping and plotting functional
Data for the cerebellum

@authors jdiedrichsen, eliu72, dzhi1993, switt
"""

import numpy as np
import os
import sys
import nibabel as nb
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import warnings

_base_dir = os.path.dirname(os.path.abspath(__file__))
_surf_dir = os.path.join(_base_dir, 'surfaces')


def affine_transform(x1,x2,x3,M):
    """
    Returns affine transform of x
    INPUT:
        x1 (np-array): X-coordinate of original
        x2 (np-array): Y-coordinate of original
        x3 (np-array): Z-coordinate of original
        M (2d-array): 4x4 transformation matrix

    OUTPUT:
        x1 (np-array): X-coordinate of transform
        x2 (np-array): Y-coordinate of transform
        x3 (np-array): Z-coordinate of transform
        transformed coordinates: same form as x1,x2,x3
    """
    y1 = np.multiply(M[0,0],x1) + np.multiply(M[0,1],x2) + np.multiply(M[0,2],x3) + M[0,3]
    y2 = np.multiply(M[1,0],x1) + np.multiply(M[1,1],x2) + np.multiply(M[1,2],x3) + M[1,3]
    y3 = np.multiply(M[2,0],x1) + np.multiply(M[2,1],x2) + np.multiply(M[2,2],x3) + M[2,3]
    return (y1,y2,y3)

def coords_to_voxelidxs(coords,volDef):
    """
    Maps coordinates to linear voxel indices

    INPUT:
        coords (3*N matrix or 3xPxQ array):
            (x,y,z) coordinates
        voldef (nibabel object):
            Nibabel object with attributes .affine (4x4 voxel to coordinate transformation matrix from the images to be sampled (1-based)) and shape (1x3 volume dimension in voxels)

    OUTPUT:
        linidxsrs (N-array or PxQ matrix):
            Linear voxel indices
    """
    mat = np.array(volDef.affine)

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
        ijk[i,ijk[i,:]>=volDef.shape[i]]=-1
    return ijk


def vol_to_surf(volumes, space = 'SUIT', ignoreZeros=0, 
                depths=[0,0.2,0.4,0.6,0.8,1.0],stats='nanmean',outerSurfGifti=None, innerSurfGifti=None):
    """
    Maps volume data onto a surface, defined by inner and outer surface.
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

    @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        volumes (list or nib obj):
            List of filenames/nib objs, or nib obj to be mapped
        space (string):
            Normalization space: 'SUIT' (default), 'FSL', 'SPM'
    OPTIONAL:
        ignoreZeros (bool):
            Should zeros be ignored in mapping? DEFAULT:  False
        depths (array-like):
            Depths of points along line at which to map (0=white/gray, 1=pial).
            DEFAULT: [0.0,0.2,0.4,0.6,0.8,1.0]
        stats (lambda function):
            function that calculates the Statistics to be evaluated.
            'nanmean': default and used for activation data
            'mode': used when discrete labels are sampled. The most frequent label is assigned.
        outerSurfGifti (string or nibabel.GiftiImage):
            optional pial surface, filename or loaded gifti object, overwrites space
        innerSurfGifti (string or nibabel.GiftiImage):
            White surface, filename or loaded gifti object, overwrites space
    OUTPUT:
        mapped_data (numpy.array):
            A Data array for the mapped data
    """
    # Get the surface files
    if innerSurfGifti is None:
        innerSurfGifti = f'PIAL_{space}.surf.gii'
    if outerSurfGifti is None:
        outerSurfGifti = f'WHITE_{space}.surf.gii'

    inner = os.path.join(_base_dir,'surfaces',innerSurfGifti)
    innerSurfGiftiImage = nb.load(inner)
    outer = os.path.join(_base_dir,'surfaces',outerSurfGifti)
    outerSurfGiftiImage = nb.load(outer)

    # Get the vertices and check that the numbers are the same
    c1 = innerSurfGiftiImage.darrays[0].data
    c2 = outerSurfGiftiImage.darrays[0].data
    numVerts = c1.shape[0]
    if c2.shape[0] != numVerts:
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
    indices = np.zeros((numPoints,numVerts,3),dtype=int)
    for i in range(numPoints):
        c = (1-depths[i])*c1.T+depths[i]*c2.T
        ijk = coords_to_voxelidxs(c,Vols[firstGood])
        indices[i] = ijk.T

    # Read the data and map it
    data = np.zeros((numPoints,numVerts))
    mapped_data = np.zeros((numVerts,len(Vols)))
    for v,vol in enumerate(Vols):
        if vol is None:
            pass
        else:
            X = vol.get_data()
            if (ignoreZeros>0):
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

def make_func_gifti(data,anatomical_struct='Cerebellum',column_names=[]):
    """
    Generates a function GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        data (np.array):
             numVert x numCol data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'CortexLeft'
        column_names (list):
            List of strings for names for columns
    OUTPUTS:
        FuncGifti (functional GiftiImage)
    """
    numVerts, numCols = data.shape
    #
    # Make columnNames if empty
    if len(column_names)==0:
        for i in range(numCols):
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
    for i in range(numCols):
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

def make_label_gifti(data,anatomical_struct='Cerebellum',label_names=[],column_names=[],label_RGBA=[]):
    """
    Generates a label GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        data (np.array):
             numVert x numCol data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'Cerebellum'
        label_names (list): 
            List of strings for label names
        column_names (list):
            List of strings for names for columns
        label_RGBA (list):
            List of rgba vectors
    OUTPUTS:
        gifti (label GiftiImage)

    """
    numVerts, numCols = data.shape
    numLabels = len(np.unique(data))

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if len(column_names) == 0:
        for i in range(numCols):
            column_names.append("col_{:02d}".format(i+1))

    # Determine color scale if empty
    if len(label_RGBA) == 0:
        hsv = plt.cm.get_cmap('hsv',numLabels)
        color = hsv(np.linspace(0,1,numLabels))
        # Shuffle the order so that colors are more visible
        color = color[np.random.permutation(numLabels)]
        label_RGBA = np.zeros([numLabels,4])
        for i in range(numLabels):
            label_RGBA[i] = color[i]

    # Create label names
    if len(label_names) == 0:
        for i in range(numLabels):
            label_names.append("label-{:02d}".format(i+1))

    # Create label.gii structure
    C = nb.gifti.GiftiMetaData.from_dict({
        'AnatomicalStructurePrimary': anatomical_struct,
        'encoding': 'XML_BASE64_GZIP'})

    num_labels = np.arange(numLabels)
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
    for i in range(numCols):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_INT16',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': column_names[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.extend(E_all)
    return gifti


def get_gifti_column_names(G):
    """
    Returns the column names from a functional gifti file.

    INPUT:
    G:				Nibabel gifti object

    OUTPUT:
    names:			List of column names from gifti object attribute data arrays

    @author: jdiedrichsen (Python conversion: switt)
    """
    N = len(G.darrays)
    names = []
    for n in range(N):
        for i in range(len(G.darrays[n].meta.data)):
            if 'Name' in G.darrays[n].meta.data[i].name:
                names.append(G.darrays[n].meta.data[i].value)
    return names

def get_gifti_colortable(G):
    """
    Returns the RGBA color table from a label gifti.
    INPUT:
    G:				Nibabel gifti object

    OUTPUT:
    A:			     np.ndarray N x 4 of RGB values

    @author: jdiedrichsen
    """
    labels = G.labeltable.labels
    N = len(labels)
    colors = np.zeros((N,4))
    for i in range(N):
        colors[i,:]=labels[i].rgba
    return colors

def get_gifti_anatomical_struct(G):
    """
    Returns the primary anatomical structure for a gifti object.

    INPUT:
    G:				Nibabel gifti object

    OUTPUT:
    anatStruct:		AnatomicalStructurePrimary attribute from gifti object

    @author: jdiedrichsen (Python conversion: switt)
    """
    N = len(G._meta.data)
    anatStruct = []
    for i in range(N):
        if 'AnatomicalStructurePrimary' in G._meta.data[i].name:
            anatStruct.append(G._meta.data[i].value)
    return anatStruct

def plot(data, surf = None, underlay = os.path.join(_surf_dir,'SUIT.shape.gii'),
        undermap = 'Greys', underscale = [-1, 0.5], overlay_type = 'func', threshold = None,
        cmap = None, cscale = None, borders = os.path.join(_surf_dir,'borders.txt'), alpha = 1.0,
        outputfile = None, render='matplotlib'):
    """
    Visualised cerebellar cortical acitivty on a flatmap in a matlab window
    INPUT:
        data (np.array, giftiImage, or name of gifti file)
            Data to be plotted, should be a 28935x1 vector
        surf (str or giftiImage)
            surface file for flatmap (default: FLAT.surf.gii in SUIT pkg)
        underlay (str, giftiImage, or np-array)
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
    OUTPUT:
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

    # Load underlay and assign color
    if type(underlay) is not np.ndarray:
        underlay = nb.load(underlay).darrays[0].data
    underlay_color = _map_color(faces, underlay, underscale, undermap)

    # Load the overlay if it's a string
    if type(data) is str:
        data = nb.load(data)

    # If it is a giftiImage, figure out colormap
    if type(data) is nb.gifti.gifti.GiftiImage:
        if overlay_type == 'label':
            cmap = get_gifti_colortable(data)
            data = data.darrays[0].data

    # If 2d-array, take the first column only
    if data.ndim>1:
        data = data[:,0]
    # depending on data type - type cast into int
    if overlay_type=='label':
        i = np.isnan(data)
        data = data.astype(int)
        data[i]=0

    # map the overlay to the faces
    overlay_color  = _map_color(faces, data, cscale, cmap, threshold)

    # Combine underlay and overlay: For Nan overlay, let underlay shine through
    face_color = underlay_color * (1-alpha) + overlay_color * alpha
    i = np.isnan(face_color.sum(axis=1))
    face_color[i,:]=underlay_color[i,:]
    face_color[i,3]=1.0

    # If present, get the borders
    if borders is not None:
        borders = np.genfromtxt(borders, delimiter=',')

    # Render with Matplotlib
    ax = _render_matplotlib(vertices, faces, face_color, borders)
    return ax

def _map_color(faces, data, scale, cmap=None, threshold = None):
    """
    Maps data from vertices to faces, scales the values, and
    then looks up the RGB values in the color map

    Input:
        data (1d-np-array)
            Numpy Array of values to scale. If integer, if it is not scaled
        scale (array like)
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
            data[np.logical_and(data>threshold[0], data<threshold[1])]=np.nan

        # if scale not given, find it
        if scale is None:
            scale = np.array([np.nanmin(data), np.nanmax(data)])

        # Scale the data
        data = ((data - scale[0]) / (scale[1] - scale[0]))

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
    return color_data

def _render_matplotlib(vertices,faces,face_color, borders):
    """
    Render the data in matplotlib: This is segmented to allow for openGL renderer

    Input:
        vertices (np.ndarray)
            Array of vertices
        faces (nd.array)
            Array of Faces
        face_color (nd.array)
            RGBA array of color and alpha of all vertices
    """
    patches = []
    for i in range(faces.shape[0]):
        polygon = Polygon(vertices[faces[i],0:2], True)
        patches.append(polygon)
    p = PatchCollection(patches)
    p.set_facecolor(face_color)
    p.set_linewidth(0.0)

    # Get the current axis and plot it
    plt.figure(figsize=(8,8))
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

