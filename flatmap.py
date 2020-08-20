"""
Created on Wed May  1 09:20:41 2019
 
@author joern.diedrichsen@googlemail.com, Aug 2020 (Python conversion: eliu72, dzhi1993)
"""

import math
import nibabel as nb
import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from matplotlib import cm  # for colormaps
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import pathlib
from PIL import Image


"""
Initialize vertex shader and fragment shader program
    Vertex Shader calculates vertex position
    Fragment Shader determines color for each "fragment" / pixel
"""
vertex_shader = """
    #version 330
    in vec3 position;
    in vec4 color;

    out vec4 newColor;
    void main()
    {
        gl_Position = vec4(position, 1.0f);
        newColor = color;
    }
    """

fragment_shader = """
    #version 330
    in vec4 newColor;

    out vec4 outColor;
    void main()
    {
        outColor = newColor;
    }
    """


def plot(data=[], surf="", underlay="", undermap="", underscale=[-1, 0.5], threshold=[], cmap="", borders="", cscale=[], alpha=1, overlay_type="func", output_file="output/file.jpg"):
    """
    Plots the flatmap using PyOpenGL
    
    Input:
        data (str)
            Full filepath of the surface data gifti file (required)
        surf (str)
            Full filepath of the surface file for flatmap (default: FLAT.surf.gii in SUIT pkg)
        underlay (str)
            Full filepath of the file determining underlay coloring (default: SUIT.shape.gii in SUIT pkg)
        undermap (str)
            Matplotlib colormap used for underlay (default: gray)
        underscale (int array)
            Colorscale [min, max] for the underlay (default: [-1, 0.5])
        threshold (int array)
            Threshold for functional overlay, valid input values from -1 to 1  (default: [-1])
        cmap (str)
            Matplotlib colormap used for overlay (default: parula in SUIT pkg)
        borders (str)
            Full filepath of the borders txt file (default: borders.txt in SUIT pkg)
        cscale (int array)
            Colorscale [min, max] for the overlay, valid input values from -1 to 1 (default: [overlay.max, overlay.min])
        alpha (int)
            Opacity of the overlay (default: 1)
        overlay_type (str)
            'func': funtional activation 'label': categories 'rgb': RGB values (default: func)
        output_file (str)
            Full filepath of the location to store outputted screenshot. Writes to JPG, PNG, PDF, GIF types. 
            See https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html for more formats. (default: 'output/file.jpg') 
    """

    # default directory 
    if not surf:
        surf = str(pathlib.Path(__file__).parent.absolute()) + "/data/FLAT.surf.gii"
    if not underlay:
        underlay = str(pathlib.Path(__file__).parent.absolute()) + "/data/SUIT.shape.gii"
    if not borders:
        borders = str(pathlib.Path(__file__).parent.absolute()) + "/data/borders.txt"

    # load topology
    surf_vertices, surf_faces = load_topo(surf)
    numVert = surf_vertices.shape[0]

    # Determine underlay and assign color
    underlay_color = load_underlay(underlay, numVert, underscale, undermap)

    # Determine overlay and assign color
    overlay_color, indx = load_overlay(data, numVert, cmap, cscale, threshold, overlay_type)
    
    # Combine underlay and overlay
    layer = blend_underlay_overlay(underlay_color, overlay_color, indx, alpha)

    # prepare layer for rendering
    layer_render = np.concatenate((surf_vertices, layer), axis=1)
    layer_render = np.concatenate((layer_render, np.reshape(np.repeat(alpha, layer_render.shape[0]), (layer_render.shape[0], 1))), axis=1)
    layer_render = np.array(layer_render.flatten(), dtype=np.float32)

    # Determine borders
    borders_render = load_borders(borders, surf_vertices)

    # Render with PyOpenGL
    render(surf_faces, layer_render, borders_render, output_file)


def load_topo(filename):
    """
    Load data using nibabel. Returns the topology of the flatmap.
    
    Input:
        filename (str)
            The full filepath of the surface file (eg. "FLAT.surf.gii")
    """
    gifti_image = nb.load(filename)  
    vertices = gifti_image.darrays[0].data / 100
    faces = gifti_image.darrays[1].data.flatten()

    return vertices, faces


def load_underlay(underlay_file, numVert, underscale, undermap):
    """
    Load vertices coordinates and combine with underlay color to make underlay data. Returns the underlay_color data.
    
    Input:
        underlay_file (str)
            The full filepath of the underlay file (eg. "SUIT.shape.gii")
        numVert (int)
            Number of vertices of the flatmap
        underscale (int array)
            The scaling of the underlay color. [min, max] between -1 and 1.
        undermap (str)
            The Matplotlib colormap for the underlay
    """

    # load underlay 
    underlay = nb.load(underlay_file)
    underlay_data = underlay.darrays[0].data # underlay.cdata, (25935, 1)

    # scale underlay color
    underlay_data = ((underlay_data - underscale[0]) / (underscale[1] - underscale[0]))
    underlay_data[underlay_data < (-1)] = -1
    underlay_data[underlay_data > 1] = 1

    # load default cmap
    if undermap == "":
        underlay_color = np.repeat(underlay_data, 3)
        underlay_color = np.reshape(underlay_color, (len(underlay_data), 3))
    # else load given cmap
    else:
        try:
            underlay_cmap = cm.get_cmap(undermap, numVert)
            underlay_color = underlay_cmap(underlay_data)
        except:
            raise Exception("Please enter a valid cmap or leave blank for default cmap. See matplotlib for more cmaps.");

    # underlay color
    underlay_color = underlay_color[:, 0:3] 
    
    return underlay_color


def load_overlay(data, numVert, cmap, cscale, threshold, overlay_type):
    """
    Returns overlay data for rendering. Returns list of indices in overlay_color where the positive overlay will be rendered.
    
    Input:
        data (str)
            Full filepath for the surface data gifti file
        numVert (int)
            Number of vertices of the flatmap.
        cmap (str)
            The Matplotlib colormap for the overlay
        cscale (int array)
            The scaling of the overlay cmap. [min, max] between -1 and 1
        threshold (int array)
            Threshold for functional overlay, valid input values from -1 to 1  (default: [-1])        
        overlay_type (str)
            'func': funtional activation 'label': categories 'rgb': RGB values (default: func)
    """

    # if input data is empty, return an array of NaNs
    if data == []:
        data = np.empty([202545, 1]);  # shape of final overlay_render array
        data[:] = np.NaN
        return data

    # load surface data
    overlay = nb.load(data)
    overlay_data = overlay.darrays[0].data

    # check that overlay data is the right np.array shape
    if len(overlay_data) != numVert:
        raise Exception('Input data must be an array of size numVertices * 1')

    # load default cmap
    if not cmap:
        overlay_cmap = default_cmap()
    # else, load given cmap
    else: 
        # Txt file with colors for type=labels
        if cmap.endswith(".txt"):
            overlay_cmap = default_cmap()
            label_colors = np.loadtxt(cmap)  # Here, we always use 0-1 scale RGB colors
        else:
            try:
                overlay_cmap = cm.get_cmap(cmap, numVert)
            except:
                raise Exception("Please enter a valid cmap or param blank for default cmap. See matplotlib for more cmaps.")

    indx = []

    # determine overlay type
    if overlay_type == 'label':
        overlay_data = overlay_data /100
        indx = np.nonzero((overlay_data > -1) & (overlay_data <= 1))

        overlay_color = overlay_cmap(overlay_data)
        overlay_color = overlay_color[:, 0:3]

        if cmap.endswith(".txt"):
            unique_val = np.unique(overlay_color, axis=0)
            unique_val = unique_val[1:]

            if len(label_colors) < len(unique_val):
                count = len(label_colors)
            else:
                count = len(unique_val)
            for i in range(count):
                overlay_color = np.where(overlay_color==unique_val[i], label_colors[i], overlay_color)

    elif overlay_type == 'func':
        if (not cscale) or (np.any(np.isnan(cscale))):
            cscale = [np.nanmin(overlay_data), np.nanmax(overlay_data)]

        # scale overlay color
        overlay_data = (overlay_data - cscale[0]) / (cscale[1] - cscale[0])
        overlay_data[overlay_data < -1] = 0
        overlay_data[overlay_data > 1] = 1

        # check that threshold value is valid
        if threshold:
            if (threshold[0] > 1) | (threshold[0] < -1):
                raise Exception('Threshold value must be between 0 and 1.')
            else:
                # find indices where overlay_data is greater than threshold (returns an array)
                indx = np.nonzero((overlay_data > threshold[0]))
        else:
            indx = np.nonzero((overlay_data > -1) & (overlay_data <= 1))
        
        # apply overlay color
        overlay_color = overlay_cmap(overlay_data)  
        overlay_color = overlay_color[:, 0:3]

    elif overlay_type == 'rgb':
        overlay_data = overlay_data
        # apply overlay color
        overlay_color = overlay_cmap(overlay_data)  
        overlay_color = overlay_color[:, 0:3]
    else:
        raise Exception("Unknown overlay type. Must be 'label', 'func'm or 'rgb'.")

    return overlay_color, indx


def blend_underlay_overlay(underlay_color, overlay_color, indx, alpha):
    """
    Blends the underlay and overlay using alpha blending.
    
    Input:
        underlay_color (int array)
            Array holding the rgb values for each underlay vertice
        overlay_color (int array)
            Array holding the rgb values for each overlay vertice
        indx (int array)
            Array of indices in overlay_color that we want as the positive overlay (areas greater than threshold)
        alpha (int)
            Opacity of the overlay (default: 1)
    """

    if not indx:
        return underlay_color
    else:
        # alpha blending using porter and duff eqns
        # out_alpha = src_alpha + dst_alpha(1-src_alpha)
        # out_rgb = (src_rgb*src_alpha + dst_rgb*dst_alpha(1-src_alpha))/out_alpha
        # if out_alpha = 0 then out_rgb = 0
        out_rgb = (overlay_color[indx]*alpha + underlay_color[indx] * (1-alpha))
        underlay_color[indx] = out_rgb

    return underlay_color


def default_cmap():

    """
    Returns: 
        A matplotlib registered cmap based on the default matlab cmap colors (parula cmap)
    """

    # define parula colormap (default matlab cmap)
    cm_data = [[0.2422, 0.1504, 0.6603], [0.2444, 0.1534, 0.6728], [0.2464, 0.1569, 0.6847], [0.2484, 0.1607, 0.6961],
    [0.2503, 0.1648, 0.7071],[0.2522, 0.1689, 0.7179],[0.254, 0.1732, 0.7286],[0.2558, 0.1773, 0.7393],
    [0.2576, 0.1814, 0.7501],[0.2594, 0.1854, 0.761],[0.2611, 0.1893, 0.7719],[0.2628, 0.1932, 0.7828],
    [0.2645, 0.1972, 0.7937],[0.2661, 0.2011, 0.8043],[0.2676, 0.2052, 0.8148],[0.2691, 0.2094, 0.8249],
    [0.2704, 0.2138, 0.8346],[0.2717, 0.2184, 0.8439],[0.2729, 0.2231, 0.8528],[0.274, 0.228, 0.8612],
    [0.2749, 0.233, 0.8692],[0.2758, 0.2382, 0.8767],[0.2766, 0.2435, 0.884],[0.2774, 0.2489, 0.8908],
    [0.2781, 0.2543, 0.8973],[0.2788, 0.2598, 0.9035],[0.2794, 0.2653, 0.9094],[0.2798, 0.2708, 0.915],
    [0.2802, 0.2764, 0.9204],[0.2806, 0.2819, 0.9255],[0.2809, 0.2875, 0.9305],[0.2811, 0.293, 0.9352],
    [0.2813, 0.2985, 0.9397],[0.2814, 0.304, 0.9441],[0.2814, 0.3095, 0.9483],[0.2813, 0.315, 0.9524],
    [0.2811, 0.3204, 0.9563],[0.2809, 0.3259, 0.96],[0.2807, 0.3313, 0.9636],[0.2803, 0.3367, 0.967],
    [0.2798, 0.3421, 0.9702],[0.2791, 0.3475, 0.9733],[0.2784, 0.3529, 0.9763],[0.2776, 0.3583, 0.9791],
    [0.2766, 0.3638, 0.9817],[0.2754, 0.3693, 0.984],[0.2741, 0.3748, 0.9862],[0.2726, 0.3804, 0.9881],
    [0.271, 0.386, 0.9898],[0.2691, 0.3916, 0.9912],[0.267, 0.3973, 0.9924],[0.2647, 0.403, 0.9935],
    [0.2621, 0.4088, 0.9946],[0.2591, 0.4145, 0.9955],[0.2556, 0.4203, 0.9965],[0.2517, 0.4261, 0.9974],
    [0.2473, 0.4319, 0.9983],[0.2424, 0.4378, 0.9991],[0.2369, 0.4437, 0.9996],[0.2311, 0.4497, 0.9995],
    [0.225, 0.4559, 0.9985],[0.2189, 0.462, 0.9968],[0.2128, 0.4682, 0.9948],[0.2066, 0.4743, 0.9926],
    [0.2006, 0.4803, 0.9906],[0.195, 0.4861, 0.9887],[0.1903, 0.4919, 0.9867],[0.1869, 0.4975, 0.9844],
    [0.1847, 0.503, 0.9819],[0.1831, 0.5084, 0.9793],[0.1818, 0.5138, 0.9766],[0.1806, 0.5191, 0.9738],
    [0.1795, 0.5244, 0.9709],[0.1785, 0.5296, 0.9677],[0.1778, 0.5349, 0.9641],[0.1773, 0.5401, 0.9602],
    [0.1768, 0.5452, 0.956],[0.1764, 0.5504, 0.9516],[0.1755, 0.5554, 0.9473],[0.174, 0.5605, 0.9432],
    [0.1716, 0.5655, 0.9393],[0.1686, 0.5705, 0.9357],[0.1649, 0.5755, 0.9323],[0.161, 0.5805, 0.9289],
    [0.1573, 0.5854, 0.9254],[0.154, 0.5902, 0.9218],[0.1513, 0.595, 0.9182],[0.1492, 0.5997, 0.9147],
    [0.1475, 0.6043, 0.9113],[0.1461, 0.6089, 0.908],[0.1446, 0.6135, 0.905],[0.1429, 0.618, 0.9022],
    [0.1408, 0.6226, 0.8998],[0.1383, 0.6272, 0.8975],[0.1354, 0.6317, 0.8953],[0.1321, 0.6363, 0.8932],
    [0.1288, 0.6408, 0.891],[0.1253, 0.6453, 0.8887],[0.1219, 0.6497, 0.8862],[0.1185, 0.6541, 0.8834],
    [0.1152, 0.6584, 0.8804],[0.1119, 0.6627, 0.877],[0.1085, 0.6669, 0.8734],[0.1048, 0.671, 0.8695],
    [0.1009, 0.675, 0.8653],[0.0964, 0.6789, 0.8609],[0.0914, 0.6828, 0.8562],[0.0855, 0.6865, 0.8513],
    [0.0789, 0.6902, 0.8462],[0.0713, 0.6938, 0.8409],[0.0628, 0.6972, 0.8355],[0.0535, 0.7006, 0.8299],
    [0.0433, 0.7039, 0.8242],[0.0328, 0.7071, 0.8183],[0.0234, 0.7103, 0.8124],[0.0155, 0.7133, 0.8064],
    [0.0091, 0.7163, 0.8003],[0.0046, 0.7192, 0.7941],[0.0019, 0.722, 0.7878],[0.0009, 0.7248, 0.7815],
    [0.0018, 0.7275, 0.7752],[0.0046, 0.7301, 0.7688],[0.0094, 0.7327, 0.7623],[0.0162, 0.7352, 0.7558],
    [0.0253, 0.7376, 0.7492],[0.0369, 0.74, 0.7426],[0.0504, 0.7423, 0.7359],[0.0638, 0.7446, 0.7292],
    [0.077, 0.7468, 0.7224],[0.0899, 0.7489, 0.7156],[0.1023, 0.751, 0.7088],[0.1141, 0.7531, 0.7019],
    [0.1252, 0.7552, 0.695],[0.1354, 0.7572, 0.6881],[0.1448, 0.7593, 0.6812],[0.1532, 0.7614, 0.6741],
    [0.1609, 0.7635, 0.6671],[0.1678, 0.7656, 0.6599],[0.1741, 0.7678, 0.6527],[0.1799, 0.7699, 0.6454],
    [0.1853, 0.7721, 0.6379],[0.1905, 0.7743, 0.6303],[0.1954, 0.7765, 0.6225],[0.2003, 0.7787, 0.6146],
    [0.2061, 0.7808, 0.6065],[0.2118, 0.7828, 0.5983],[0.2178, 0.7849, 0.5899],[0.2244, 0.7869, 0.5813],
    [0.2318, 0.7887, 0.5725],[0.2401, 0.7905, 0.5636],[0.2491, 0.7922, 0.5546],[0.2589, 0.7937, 0.5454],
    [0.2695, 0.7951, 0.536],[0.2809, 0.7964, 0.5266],[0.2929, 0.7975, 0.517],[0.3052, 0.7985, 0.5074],
    [0.3176, 0.7994, 0.4975],[0.3301, 0.8002, 0.4876],[0.3424, 0.8009, 0.4774],[0.3548, 0.8016, 0.4669],
    [0.3671, 0.8021, 0.4563],[0.3795, 0.8026, 0.4454],[0.3921, 0.8029, 0.4344],[0.405, 0.8031, 0.4233],
    [0.4184, 0.803, 0.4122],[0.4322, 0.8028, 0.4013],[0.4463, 0.8024, 0.3904],[0.4608, 0.8018, 0.3797],
    [0.4753, 0.8011, 0.3691],[0.4899, 0.8002, 0.3586],[0.5044, 0.7993, 0.348],[0.5187, 0.7982, 0.3374],
    [0.5329, 0.797, 0.3267],[0.547, 0.7957, 0.3159],[0.5609, 0.7943, 0.305],[0.5748, 0.7929, 0.2941],
    [0.5886, 0.7913, 0.2833],[0.6024, 0.7896, 0.2726],[0.6161, 0.7878, 0.2622],[0.6297, 0.7859, 0.2521],
    [0.6433, 0.7839, 0.2423],[0.6567, 0.7818, 0.2329],[0.6701, 0.7796, 0.2239],[0.6833, 0.7773, 0.2155],
    [0.6963, 0.775, 0.2075],[0.7091, 0.7727, 0.1998],[0.7218, 0.7703, 0.1924],[0.7344, 0.7679, 0.1852],
    [0.7468, 0.7654, 0.1782],[0.759, 0.7629, 0.1717],[0.771, 0.7604, 0.1658],[0.7829, 0.7579, 0.1608],
    [0.7945, 0.7554, 0.157],[0.806, 0.7529, 0.1546],[0.8172, 0.7505, 0.1535],[0.8281, 0.7481, 0.1536],
    [0.8389, 0.7457, 0.1546],[0.8495, 0.7435, 0.1564],[0.86, 0.7413, 0.1587],[0.8703, 0.7392, 0.1615],
    [0.8804, 0.7372, 0.165],[0.8903, 0.7353, 0.1695],[0.9, 0.7336, 0.1749],[0.9093, 0.7321, 0.1815],
    [0.9184, 0.7308, 0.189],[0.9272, 0.7298, 0.1973],[0.9357, 0.729, 0.2061],[0.944, 0.7285, 0.2151],
    [0.9523, 0.7284, 0.2237],[0.9606, 0.7285, 0.2312],[0.9689, 0.7292, 0.2373],[0.977, 0.7304, 0.2418],
    [0.9842, 0.733, 0.2446],[0.99, 0.7365, 0.2429],[0.9946, 0.7407, 0.2394],[0.9966, 0.7458, 0.2351],
    [0.9971, 0.7513, 0.2309],[0.9972, 0.7569, 0.2267],[0.9971, 0.7626, 0.2224],[0.9969, 0.7683, 0.2181],
    [0.9966, 0.774, 0.2138],[0.9962, 0.7798, 0.2095],[0.9957, 0.7856, 0.2053],[0.9949, 0.7915, 0.2012],
    [0.9938, 0.7974, 0.1974],[0.9923, 0.8034, 0.1939],[0.9906, 0.8095, 0.1906],[0.9885, 0.8156, 0.1875],
    [0.9861, 0.8218, 0.1846],[0.9835, 0.828, 0.1817],[0.9807, 0.8342, 0.1787],[0.9778, 0.8404, 0.1757],
    [0.9748, 0.8467, 0.1726],[0.972, 0.8529, 0.1695],[0.9694, 0.8591, 0.1665],[0.9671, 0.8654, 0.1636],
    [0.9651, 0.8716, 0.1608],[0.9634, 0.8778, 0.1582],[0.9619, 0.884, 0.1557],[0.9608, 0.8902, 0.1532],
    [0.9601, 0.8963, 0.1507],[0.9596, 0.9023, 0.148],[0.9595, 0.9084, 0.145],[0.9597, 0.9143, 0.1418],
    [0.9601, 0.9203, 0.1382],[0.9608, 0.9262, 0.1344],[0.9618, 0.932, 0.1304],[0.9629, 0.9379, 0.1261],
    [0.9642, 0.9437, 0.1216],[0.9657, 0.9494, 0.1168],[0.9674, 0.9552, 0.1116],[0.9692, 0.9609, 0.1061],
    [0.9711, 0.9667, 0.1001],[0.973, 0.9724, 0.0938],[0.9749, 0.9782, 0.0872],[0.9769, 0.9839, 0.0805]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    return parula_map


def load_borders(border_file, surf_vertices):
    """
    Look up the vertices coord to find which node is the border and make it black color for rendering. Returns the border_data array.

    Input:
        border_file (str)
            The border filepath. (eg. "data/borders.txt")
        surf_vertices (int array)
            Array of the vertices coordinates of the flatmap. shape (N, 3)
    """

    borders = []
    from numpy import genfromtxt
    border = genfromtxt(border_file, delimiter=',')/100
    bcolor = np.array([[0.0, 0.0, 0.0], ] * border.shape[0]) # set border default color (black)
    border_render = np.concatenate((border, bcolor), axis = 1)
    border_render = np.concatenate((border_render, np.reshape(np.repeat(1, border_render.shape[0]), (border_render.shape[0], 1))), axis=1)
    border_render = np.array(border_render.flatten(), dtype=np.float32)

    return border_render


def render_underlay(underlay, faces, shader):
    """
    Renders the underlay

    Input:
        underlay (int array)
            The vertices coordinates buffer of the flatmap for the underlay
        vertices_index (int array)
            The flatmap vertices drawing order (faces)
        shader (shader)
            The compiled shader program by opengl pipeline
    """
    
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, underlay.shape[0]*4, underlay, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.shape[0]*4, faces, GL_STATIC_DRAW)
    
    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    glDrawElements(GL_TRIANGLES, faces.shape[0], GL_UNSIGNED_INT, None)


def render_borders(borders_buffer, shader):
    """
    Renders borders

    Input:
        borders_buffer (int array)
            The border buffers objects (flatten)
        shader (shader)
            The compiled shader program by opengl pipeline
    """
    #for border_info in borders_buffer:
    BBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, BBO)
    glBufferData(GL_ARRAY_BUFFER, borders_buffer.shape[0] * 4, borders_buffer, GL_STATIC_DRAW)

    b_position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(b_position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(b_position)

    b_color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(b_color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(b_color)
    glPointSize(3)
    glDrawArrays(GL_POINTS, 0, int(borders_buffer.shape[0] / 7))


def render(vertices_index, layer, borders, output_file):
    """
    OpenGL rendering

    Input:
        vertices_index (int array)
            The flatmap vertices drawing order (faces)
        borders (int array)
            The border buffers objects (flatten) shape list(N, )
        layer (int array)
            The blended overlay and underlay buffers object
        output_file (str)
            The filepath for the output image
    """

    # initialize glfw
    if not glfw.init():
        return

    w_width, w_height = 600,600
    glfw.window_hint(glfw.RESIZABLE, GL_TRUE)
    window = glfw.create_window(w_width, w_height, "Cerebellum Flatmap", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, window_resize)

    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(1, 1, 1, 1.0)  # Background color, default white
    glClear(GL_COLOR_BUFFER_BIT) # clear background to this color

    glUseProgram(shader)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)

    # rendering objects
    render_underlay(layer, vertices_index, shader)  # -- the underlay rendering
    render_borders(borders, shader)  # -- border buffer object

    # save window to jpg
    render_to_jpg(output_file)

    glDisable(GL_BLEND)  # Disable gl blending from this point
    glDepthMask(GL_TRUE)

    glfw.swap_buffers(window)


    # FILE   *out = fopen(tga_file, "w");
    # short  TGAhead[] = {0, 2, 0, 0, 0, 0, WIDTH, HEIGHT, 24};
    # fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    # fwrite(buffer, 3 * WIDTH * HEIGHT, 1, out);
    # fclose(out);

    while not glfw.window_should_close(window):
        glfw.poll_events()

    glfw.terminate()


def render_to_jpg(output_file):
    """
    Save PyOpenGL render as image

    Input:
        output_file (str)
            The filepath for the output image
    """
    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width), int(height)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    image.save(output_file)


def window_resize(window, width, height):
    glViewport(0, 0, width, height)


if __name__ == "__main__":
    #plot("data/Wiestler_2011_motor_z.gii", cscale=[0, 2])
    #suit_plotflatmap("data/HCP_WM_BODY_vs_REST.gii", cmap='hot', threshold=[0.25], cscale=[0,2])
    plot("data/Buckner_7Networks.gii", overlay_type="label", cmap="data/Buckner_7Networks_cmap.txt")
