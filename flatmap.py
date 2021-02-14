#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:15:59 2019

@author: switt, jdiedrichsen
"""

import numpy as np
import os
import sys
import nibabel as nb
from scipy import sparse
#from casadi import sparsify

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

def subs_to_inds(siz,pos):
    """
    Linear indices from multiple subscripts; a generalization of sub2ind
    Function returns the linear indices for each row of subscripts in  pos based on a matrix with dimensions siz.
    If siz=[s1, ..., sn] refers to a matrix of size s1 x ... x sN, and pos is
    an M x N where each row contains the subindices referring to a single
    element, then ids is a M x 1 vector with linear indices.
    For indices in pos that are out of bounds, the corresponding element in ids
    is NaN. If pos is of type int32, however, the corresponding element is set
    to zero.

    INPUT:
        siz (1x3 array like):
            Dimension of image
        pos (np-array):
            Matrix of subindices referring to a single element

    OUTPUT:
        ids:
            Vector of linear indices
    """

    dimCount = np.shape(siz)[0]
    [posCount,dimCount2] = np.shape(pos)

    if (dimCount != dimCount2):
        sys.exit('Error: Number of dimensions do not match.')

    # Make sure pos and siz have same data type class
    posClass = pos.dtype
    sizClass = siz.dtype
    if (posClass != sizClass):
        siz = siz.astype(posClass)

    # Multiplication factors for the different positions
    mply = np.zeros([1,dimCount],dtype=clpos)
    mply[0,0] = 1
    for k in range(dimCount-1):
        mply[0][k+1] = mply[0][k]*siz[k]

    alot = 1e6
    if posCount>alot:
        ids = np.zeros([posCount,1],dtype=clpos)
        for k in range(1,posCount,alot):
            idxs = range(k,np.min([k+alot-1,posCount]))
            ids[idxs] = subs2inds(siz,pos[idxs,:])
        return

    ids = np.sum(np.multiply((pos-1),np.tile(mply,(posCount,1))),axis=1)+1

    beyond = np.tile(siz,(posCount,1))-pos
    outOfBounds = np.sum(np.logical_or(pos<1,beyond<0),axis=1)>0
    if len(np.where(outOfBounds)[0]):
        ids[np.where(outOfBounds)]=np.nan

    return ids

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
    # allinidxs = subs_to_inds(dim,np.transpose(ijk))
    # linidxs = allinidxs

    # linidxsrs = np.transpose(np.reshape(linidxs,[nCoordsPerNode,nVerts]))

    return ijk


def vol_to_surf(whiteSurfGifti, pialSurfGifti, volumes,
            ignoreZeros=0, excludeThres=0, depths=[0,0.2,0.4,0.6,0.8,1.0],
            stats=lambda X:np.nanmean(X,axis=0)):
    """
    Maps volume data onto a surface, defined by white and pial surface.
    Function enables mapping of volume-based data onto the vertices of a
    surface. For each vertex, the function samples the volume along the line
    connecting the white and gray matter surfaces. The points along the line
    are specified in the variable 'depths'. default is to sample at 5
    locations between white an gray matter surface. Set 'depths' to 0 to
    sample only along the white matter surface, and to 0.5 to sample along
    the mid-gray surface.

    The averaging across the sampled points for each vertex is dictated by
    the variable 'stats'. For functional activation, use 'mean' or
    'nanmean'. For discrete label data, use 'mode'.

    If 'exclude_thres' is set to a value >0, the function will exclude voxels that
    touch the surface at multiple locations - i.e. voxels within a sulcus
    that touch both banks. Set this option, if you strongly want to prevent
    spill-over of activation across sulci. Not recommended for voxels sizes
    larger than 3mm, as it leads to exclusion of much data.

    For alternative functionality see wb_command volumne-to-surface-mapping
    https://www.humanconnectome.org/software/workbench-command/-volume-to-surface-mapping

    @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        whiteSurfGifti (string or nibabel.GiftiImage):
            White surface, filename or loaded gifti object
        pialSurfGifti (string or nibabel.GiftiImage):
            Pial surface, filename or loaded gifti object
        volumes (list):
            List of filenames, or nibable.NiftiImage  to be mapped
    OPTIONAL:
        ignoreZeros (bool):
            Should zeros be ignored in mapping? DEFAULT:  False
        depths (array-like):
            Depths of points along line at which to map (0=white/gray, 1=pial).
            DEFAULT: [0.0,0.2,0.4,0.6,0.8,1.0]
        stats (lambda function):
            function that calculates the Statistics to be evaluated.
            lambda X: np.nanmean(X,axis=0) default and used for activation data
            lambda X: np.mode(X,axis=0) used when discrete labels are sampled.
            The most frequent label is assigned.
        excludeThres (float):
            Threshold enables the exclusion of voxels that touch the surface
            in two distinct places
            (e.g., voxels that lie in the middle of a sulcus). If a voxel projects to two separate place
            on the surface, the algorithm excludes it, if the proportion of the bigger cluster
            is smaller than the threshold. (i.e. threshold = 0.9 means that the voxel has to
            lie at least to 90% on one side of the sulcus).
            **** Currently not supported.  excludeThres is automatically reset to 0. ****
            DEFAULT: 0

    OUTPUT:
        mapped_data (numpy.array):
            A Data array for the mapped data
    """

    Vols = []
    firstGood = None
    depths = np.array(depths)

    if excludeThres != 0:
        print('Warning: excludeThres option currently not supported. Resetting excludeThres to 0.')
        excludeThres = 0

    numPoints = len(depths)

    whiteSurfGiftiImage = nb.load(whiteSurfGifti)
    pialSurfGiftiImage = nb.load(pialSurfGifti)

    c1 = whiteSurfGiftiImage.darrays[0].data
    c2 = pialSurfGiftiImage.darrays[0].data
    faces = whiteSurfGiftiImage.darrays[1].data

    numVerts = c1.shape[0]

    if c2.shape[0] != numVerts:
        sys.exit('Error: White and pial surfaces should have same number of vertices.')

    for i in range(len(volumes)):
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

# indices = np.transpose(np.squeeze(np.asarray(indices)))
    # indices = indices.astype(int)


    # Case: excludeThres > 0
    # If necessary, now ensure that voxels are mapped on to continuous location
    # only on the flat map - exclude voxels that are assigned to two sides of
    # the sulcus
#    if (excludeThres>0):
#        exclude = np.zeros([np.prod(V[1].shape),1])
#        if not faces:
#            sys.exit('provide faces (topology data), so that projections should be avoided')
#        S.Tiles = faces

        # Precalculate edges for fast cluster finding
#        print('Calculating edges.')
#        S.numNodes = np.max(np.max(S.Tiles))
#        for i in range (3):
#            i1 = S.Tiles[:,i]
#            i2 = S.Tiles[:,np.remainder(i,3)+1]
#            S.Edges.append(i1,i2)
#        S.Edges = np.unique(S.Edges,axis=0)

        # Generate connectivity matrix
        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#        data = np.ones(len(S.Edges))
#        rowInd = S.Edges[:,0]
#        colInd = S.Edges[:,1]
#        M = S.numNodes
#        G = sparse.csr_matrix((data,(rowInd,colInd)),shape=(M,M))

        # Cluster the projections to the surface and exclude all voxels
        # in which the weight of the biggest cluster is not > thres
#        print('Checking projections.')
#        I = np.unique(indices[np.isfinite(indices)])
#        for i in I:

            # Calculate the weight of voxel on node
#            weight = np.sum(indices==1,1)
#            indx = np.where(weight>0)[0]

            # Check whether the nodes cluster
#            H = G[indx,indx]
#            H = H+np.transpose(H)+sparse.identity(len(H))
            # Matlab vairable translation: p=rowPerm,q=colPerm,r=rowBlock,s=colBlock
#            nb,rowPerm,colPerm,rowBlock,colBlock,coarseRowBlock,coarseColBlock = H.sparsify().btf()
#            CL = np.zeros(np.shape(indx))
#            for c in range(len(rowBlock)-1):
#                CL[rowPerm[rowBlock[c]]:rowBlock[(c+1)]-1,0]=c

#            if (np.max(CL)>1):
#                weight_cl=np.zeros([np.max(CL),1])
#                for cl in range(np.max(CL)):
#                    weight_cl[cl,0] = np.sum(weight[indx[CL==cl]])
#                [m,cl] = np.max(weight_cl)
#                if (m/np.sum(weight_cl)>excludeThres):
#                    A = indices[indx[CL!=cl],:]
#                    A[A==i] = np.nan
#                    indices[indx[CL!=cl],:] = A
#                    exclude[i] = 1
#                    print('assigned: %2.3f'.format(m/np.sum(weight_cl)))
#                else:
#                    A[A==i] = np.nan
#                    indices[indx,:] = A
#                    exclude[i] = 2
#                    print('excluded: %2.3f %d'.format(m/np.sum(weight_cl),np.max(CL)))

        # For debugging: save the volume showing the exluded voxels in current
        # directory
#        Vexcl = V[1]
#        Vexcl.set_filename = 'excl.nii'
#        nb.save(np.reshape(exclude,np.array(Vexcl.shape)),'excl.nii')

    # Case: excludeThres = 0
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
                mapped_data[:,v] = stats(data)

    return mapped_data

def make_func_gifti(data,anatomical_struct='CortexLeft',column_names=[]):
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
    'AnatomicalStructurePrimary': anatomicalStruct,
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
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': columnNames[i]})
        )
        D.append(d)

    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)

    return gifti

def make_label_gifti(data,anatomical_struct='CortexLeft',label_names=[],column_names=[],label_RGBA=[]):
    """
    Generates a label GiftiImage from a numpy array
       @author joern.diedrichsen@googlemail.com, Feb 2019 (Python conversion: switt)

    INPUTS:
        data (np.array):
             numVert x numCol data
        anatomical_struct (string):
            Anatomical Structure for the Meta-data default= 'CortexLeft'
        column_names (list):
            List of strings for names for columns
    OUTPUTS:
        gifti (label GiftiImage)

    """
    numVerts, numCols = data.shape
    numLabels = len(np.unique(data))

    # Create naming and coloring if not specified in varargin
    # Make columnNames if empty
    if len(column_names) == 0:
        for i in range(numLabels):
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

    E = nb.gifti.gifti.GiftiLabel()
    E.key = np.arange(label_names)
    E.label= label_names
    E.red = label_RGBA[:,0]
    E.green = label_RGBA[:,1]
    E.blue = label_RGBA[:,2]
    E.alpha = label_RGBA[:,3]

    D = list()
    for i in range(Q):
        d = nb.gifti.GiftiDataArray(
            data=np.float32(data[:, i]),
            intent='NIFTI_INTENT_LABEL',
            datatype='NIFTI_TYPE_INT32',
            meta=nb.gifti.GiftiMetaData.from_dict({'Name': columnNames[i]})
        )
        D.append(d)

    # Make and return the gifti file
    gifti = nb.gifti.GiftiImage(meta=C, darrays=D)
    gifti.labeltable.labels.append(E)
    return gifti


def get_gifti_column_names(G):
    """
    Created on Mon Mar 25 21:11:31 2019

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
