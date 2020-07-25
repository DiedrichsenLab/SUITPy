import math
import nibabel as nb
import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from matplotlib import cm  # for colormaps
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#import sys
#sys.path.insert(1, 'C:/Users/Elaine Liu/Documents/Elaine/Jobs/WUSRI/surfAnalysis_project/surfAnalysisPy')

#import vol2surf as vs


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

def suit_plotflatmap(data=[], flat_dir="", surf="FLAT.surf.gii", under="SUIT.shape.gii", undermap="", \
    underscale = [-1, 0.5], threshold=0.53, cmap="", borders='borders.txt', cscale = [], alpha=1, overlay_type = "func"):
    """
    # ---- Plot Flatmap ---- 
    Input:
        data: surface data gifti file
        flat_dir: path of the directory where surface, underlay, border files are stored (default = os.getcwd() + /data/)
        surf: surface gifti file
        under: underlay gifti file
        undermap: matplotlib colormap used for underlay - see matplotlib for more
        underscale: the scaling of the underlay color. [min, max] between -1 and 1
        threshold: intensities of positive overlay that is displayed (0 to 1 where 1 is all intensities shown)
        cmap: matplotlib colormap for overlay. see matplotlib for more colormaps.
        borders: txt file containing coordinates (x,y,z) of the border pixels 
        alpha: transparency of the overlay
        overlay_type: label, func, rgb
    Output:
        Renders the flatmap using pyopengl
    """

    # Determine the directory
    if flat_dir == "":
        flat_dir = os.getcwd() + "/data/"
    
    surface_file = flat_dir + surf + "";
    underlay_file = flat_dir + under + "";
    borders_file = flat_dir + borders + "";

    # load topology
    surf_vertices, surf_faces = load_topo(surface_file)
    numVert = surf_vertices.shape[0]

    # Determine underlay and assign color
    underlay_color = load_underlay(underlay_file, numVert, underscale, undermap)

    # Determine overlay and assign color
    overlay_color, indx = load_overlay(data, numVert, cmap, cscale, threshold, overlay_type)
    
    # Combine underlay and overlay
    layer = blend_underlay_overlay(underlay_color, overlay_color, indx, alpha)

    # prepare layer for rendering
    layer_render = np.concatenate((surf_vertices, layer), axis=1)
    layer_render = np.concatenate((layer_render, np.reshape(np.repeat(alpha, layer_render.shape[0]), (layer_render.shape[0], 1))), axis=1)
    layer_render = np.array(layer_render.flatten(), dtype=np.float32)

    # Determine borders
    borders_render = load_borders(borders_file, surf_vertices)

    # Render with PyOpenGL
    render(surf_faces, layer_render, borders_render)


def load_topo(filename):
    """
    # ---- Topology ---- Load data using nibabel
    Input:
        filename: the filename with path. (eg. "FLAT.surf.gii")
    Returns: the topology of the flatmap
        vertices: the coordinates of the vertices, shape: (N, 3)
        faces: the vertices connectivity info used for indexed drawing. shape: flatten
    """
    gifti_image = nb.load(filename)  
    vertices = gifti_image.darrays[0].data / 100
    faces = gifti_image.darrays[1].data.flatten()

    return vertices, faces

def load_underlay(underlay_file, numVert, underscale, undermap):
    """
    # ---- Load Underlay ----
        Load vertices coordinates and combine with underlay color to make underlay data
    Input:
        underlay_file: the underlay file name with path. (eg. "SUIT.shape.gii")
        numVert: number of vertices of the flatmap.
        underscale: the scaling of the underlay color. [min, max] between -1 and 1
        undermap: the colormap for the underlay
    Returns: the flatten buffer data for underlay
        underlay_color: the underlay color itself, shape (N, 3)
    """

    # load underlay 
    underlay = nb.load(underlay_file)
    underlay_data = underlay.darrays[0].data # underlay.cdata, (25935, 1)

    # scale underlay color
    underlay_data[:] = ((underlay_data[:] - underscale[0]) / (underscale[1] - underscale[0]))
    underlay_data[underlay_data<(-1)]=-1
    underlay_data[underlay_data>1]=1

    # load default cmap
    if (undermap == ""):
        underlay_color = np.repeat(underlay_data, 3)
        underlay_color = np.reshape(underlay_color, (len(underlay_data), 3))
    # else load given cmap
    else:
        try:
            underlay_cmap = cm.get_cmap(undermap, numVert);
            underlay_color = underlay_cmap(underlay_data)
        except:
            raise Exception("Please enter a valid cmap or leave blank for default cmap. See matplotlib for more cmaps.");

    # underlay color
    underlay_color = underlay_color[:, 0:3] 
    
    return underlay_color

def load_overlay(data, numVert, cmap, cscale, threshold, overlay_type):

    """
    # ---- Load Overlay ----
        Load vertices coordinates and combine with overlay color (converted to selected cmap value)
        to make overlays data for rendering
    Input:
        data: surface data gifti file
        numVert: number of vertices of the flatmap.
        cmap: the colormap for the overlay
        cscale: the scaling of the overlay cmap. [min, max] between -1 and 1
        threshold: intensities of positive overlay that is displayed (0 to 1 where 1 is all intensities shown)
        overlay_type: label, func, rgb
    Returns: 
        overlay_color: the overlay data for rendering. shape: flatten
        indx: list of indices in overlay_color where the positive overlay will be rendered
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
    if (len(overlay_data) != numVert):
        raise Exception('Input data must be an array of size numVertices * 1')

    # load default cmap
    if (not cmap):
        overlay_cmap = default_cmap()
    # else, load given cmap
    else: 
        try:
            overlay_cmap = cm.get_cmap(cmap, numVert);
        except:
            raise Exception("Please enter a valid cmap or param blank for default cmap. See matplotlib for more cmaps.");

    indx = []

    # determine overlay type
    if (overlay_type == 'label'):
        overlay_data = overlay_data / 10
    elif (overlay_type == 'func'):
        if (not cscale) or (np.any(np.isnan(cscale))):
            cscale = [overlay_data.min(), overlay_data.max()]

        # scale overlay color
        overlay_data[:] = (overlay_data[:] - cscale[0]) / (cscale[1] - cscale[0])
        overlay_data[overlay_data<0] = 0
        overlay_data[overlay_data>1] = 1

        # check that threshold value is valid
        if (threshold):
            if ((threshold > 1) | (threshold < 0)):
                raise Exception('Threshold value must be between 0 and 1.')
            else:
                # find indices where overlay_data is greater than threshold (returns an array)
                indx = np.nonzero(overlay_data[:] > threshold)
    elif (overlay_type == 'rgb'):
        overlay_data = overlay_data
    else:
        raise Exception("Unknown overlay type. Must be 'label', 'func'm or 'rgb'.")

    # apply overlay color
    overlay_color = overlay_cmap(overlay_data)  
    overlay_color = overlay_color[:, 0:3]

    return overlay_color, indx

def blend_underlay_overlay(underlay_color, overlay_color, indx, alpha):

    """
    # ---- Blend Underlay with Overlay ----
        Blends the underlay and overlay using alpha blending.
    Input:
        underlay_color: array holding the rgb values for each underlay vertice
        overlay_color: array holding the rgb values for each overlay vertice
        indx: array of indices in overlay_color that we want as the positive overlay (areas greater than threshold)
        alpha: transparency value for the positive overlay
    Returns: 
        An array with the blended colors.
    """

    if not indx:
        return underlay_render;
    else:
        # alpha blending using porter and duff eqns
        # out_alpha = src_alpha + dst_alpha(1-src_alpha)
        # out_rgb = (src_rgb*src_alpha + dst_rgb*dst_alpha(1-src_alpha))/out_alpha
        # if out_alpha = 0 then out_rgb = 0
        out_rgb = (overlay_color[indx]*alpha + underlay_color[indx] * (1-alpha))
        underlay_color[indx] = out_rgb

    return underlay_color;

def default_cmap():

    """
    Returns: A matplotlib registered cmap based on the default matlab cmap colors (parula cmap)
    """

    # define parula colormap (default matlab cmap)
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
    0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
    0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
    0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
    0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
    0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
    0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
    0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
    0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
    0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
    0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
    0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
    0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
    0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
    0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
    0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

    return parula_map

def load_borders(border_file, surf_vertices):
    """
    # ---- Making borders buffer ----
        Look up the vertices coord to find which node is the border and make it black color for rendering
    Input:
        border_file: the border files name with path. (eg. "data/borders.txt")
        surf_vertices: the vertices coordinates of the flatmap. shape (N, 3)
    Returns: the borders data for rendering
        borders: the borders data for rendering. shape: list(borders#, )
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
    # ---- Rendering underlay contrast ----
    Input:
        underlay: the vertices coordinates buffer of the flatmap for the underlay
        vertices_index: the flatmap vertices drawing order (faces)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing underlay to the scene
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
    # ---- Rendering borders ----
    Input:
        borders_buffer: the border buffers objects (flatten)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing selected borders to the scene
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

def render(vertices_index, underlay, borders):
    """
    # ---- The main entry of the OpenGL rendering ----
    Input:
        vertices_index: the flatmap vertices drawing order (faces)
        borders: the border buffers objects (flatten) shape list(N, )
        underlay: the underlay buffers object
        overlays: the overlays buffer object (flatten), shape list(N, )
    Returns:
        Draw the scene
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
    render_underlay(underlay, vertices_index, shader)  # -- the underlay rendering
    render_borders(borders, shader)  # -- border buffer object

    glDisable(GL_BLEND)  # Disable gl blending from this point
    glDepthMask(GL_TRUE)

    glfw.swap_buffers(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()

    glfw.terminate()

def window_resize(window, width, height):
    glViewport(0, 0, width, height)

if __name__ == "__main__":
    suit_plotflatmap("data/Wiestler_2011_motor_z.gii")

   #suit_plotflatmap("data/HCP_WM_BODY_vs_REST.gii", threshold = 0.01, cmap="jet")
    #suit_plotflatmap("data/Buckner_17Networks.label.gii", cmap="jet", overlay_type = "label")


    #vs.vol2surf("data/WHITE_SUIT.surf.gii","data/PIAL_SUIT.surf.gii","data/nifti.txt")


# def test(data=[], flat_dir="", surf="FLAT.surf.gii", under="SUIT.shape.gii",\
#     undermap="gray", undermap_norm = [-0.2, 0.3], underscale = [0, 0.5], threshold=[], cmap='viridis', 
#     borders='borders.txt', cscale = [], alpha=1, overlay_type = "func", xlims = [-100,100], ylims = [-100,100]):

#     # Determine the directory
#     if flat_dir == "":
#         flat_dir = os.getcwd() + "/data/"
    
#     surface_file = flat_dir + surf + "";
#     underlay_file = flat_dir + under + "";
#     borders_file = flat_dir + borders + "";

#     # Load surface and determine X,Y coordinates for all tiles
#     gifti_image = nb.load(surface_file) 
#     vertices = gifti_image.darrays[0].data / 100
#     faces = gifti_image.darrays[1].data.flatten()

    
#     P = len(vertices[0])

#     # reshape surf_faces into shape: 56588 x 3 (len(surf_faces/3) == 56588)
#     C_faces = np.reshape(faces, (56588, 3));
#     C_first_col = C_faces[:,0];
#     C_second_col = C_faces[:,1];
#     C_third_col = C_faces[:,2];

#     X = np.empty([3,56588])
#     Y = np.empty([3,56588])
#     X[0,:] = vertices[C_faces[:,0],0]
#     X[1,:] = vertices[C_faces[:,1],0]
#     X[2,:] = vertices[C_faces[:,2],0]
#     Y[0,:] = vertices[C_faces[:,0],1]
#     Y[1,:] = vertices[C_faces[:,1],1]
#     Y[2,:] = vertices[C_faces[:,2],1]

#     k = np.nonzero(np.any((X>xlims[0]) & (X<xlims[1]), 0) & np.any((Y>ylims[0]) & (Y<ylims[1]), 0))
#     X = X[:,k]
#     Y = Y[:,k] #openGL plots based on vertices between -1 and 1

#     X = np.reshape(X, (3, 56588))
#     Y = np.reshape(Y, (3, 56588))

#     # load underlay 
#     underlay = nb.load(underlay_file)
#     underlay_cdata = underlay.darrays[0].data # underlay.cdata, (25935, 1)

#     d_first = np.empty([56588]);
#     d_second = np.empty([56588]);
#     d_third = np.empty([56588]);

#     d_first[:] = underlay_cdata[C_first_col[:]]
#     d_second[:] = underlay_cdata[C_second_col[:]]
#     d_third[:] = underlay_cdata[C_third_col[:]]

#     d = np.concatenate((d_first, d_second, d_third))
#     d = np.reshape(d, (3,56588))

#     M=64

#     # load cmap
#     try:
#         colors = cm.get_cmap(undermap, M);
#     except:
#         raise Exception("Please enter a valid cmap. See matplotlib for more cmaps.");

#     # register new colormap
#     newcolors = colors(np.linspace(0,1,M)) # 0 to 1 is taking the entire cmap (0.25 to 0.75 reduces the cmap range)
#     newcmap = ListedColormap(newcolors)
#     cm.register_cmap(name = "newcmap", cmap=newcmap)

#     dindx = np.empty([3,56588])
#     dindx[:] = np.ceil((d[:] - underscale[0]) / (underscale[1] - underscale[0])*M)
#     dindx[dindx<1]=1
#     dindx[dindx>M]=M


#     # assign colors
#     COL = np.empty([3,56588,3])
#     for i in range(3):
#         for j in range(dindx.shape[0]):
#             # convert dindx into int array
#             COL[j,:,i] = newcmap.colors[dindx[j,:].astype(int)-1,i]

#     COL = np.reshape(COL, (int(len(COL[0])*3),3))


#     zeros = np.zeros((len(X[0])))
#     coord1 = np.vstack((X[0,:],Y[0,:], zeros)).ravel('F')
#     coord2 = np.vstack((X[1,:],Y[1,:], zeros)).ravel('F')
#     coord3 = np.vstack((X[2,:],Y[2,:], zeros)).ravel('F')
#     coord = np.concatenate((coord1, coord2, coord3))
#     coord = np.reshape(coord, (int(len(coord)/3), 3))
    
#     underlay_render = np.empty((len(coord),7))
#     underlay_render[:,0] = coord[:,0]
#     underlay_render[:,1] = coord[:,1]
#     underlay_render[:,2] = coord[:,2]
#     underlay_render[:,3] = COL[:,0]
#     underlay_render[:,4] = COL[:,1]
#     underlay_render[:,5] = COL[:,2]
#     underlay_render[:,6] = 1

#     underlay_render = underlay_render.flatten()



#     #other function
#     underlay_render2, underlay_color, underlay_data = load_underlay(data, underlay_file, vertices, faces, undermap, undermap_norm, 1, underscale, cscale, threshold, xlims, ylims)
#     # for i in range(int(len(underlay_render)/7)):
#     #     x = underlay_render[i]
#     #     y = underlay_render[i+1]
#     #     for j in range(int(len(underlay_render2)/7)):
#     #         if (x == underlay_render2[j]) & (y == underlay_render2[j+1]):
#     #             underlay_render2[j+3] = underlay_render[i+3]
#     #             underlay_render2[j+4] = underlay_render[i+4]
#     #             underlay_render2[j+5] = underlay_render[i+5]
#     #             break;

#     print(underlay_render.shape)
#     print(underlay_render2.shape)
#     #render(faces, underlay_render2, [], [])

# load_overlay()
    # # create custom colormap with threshold
    # colors = cm.get_cmap(cmap, numVert)
    # newcolors = colors(np.linspace(0,1,numVert)) # 0 to 1 is taking the entire cmap (0.25 to 0.75 reduces the cmap range)
    # white = np.array([1,1,1,0])

    # # find the threshold point and color everything before that value white
    # point = (int) (threshold * numVert)
    # newcolors[:point, :] = white

    # # register new cmap
    # newcmap = ListedColormap(newcolors)
    # cm.register_cmap(name = "newcmap", cmap=newcmap)

    # # apply new cmap
    # cmap = cm.get_cmap("newcmap", numVert)
    # overlay_color = cmap(overlay_data)    

        # # create custom colormap with threshold
    # colors = cm.get_cmap(cmap, point)
    # newcolors = colors(np.linspace(0,1,point)) # 0 to 1 is taking the entire cmap (0.25 to 0.75 reduces the cmap range)
    # white = np.ones([numVert-point,4])
    # newcolors = np.concatenate((white,newcolors))

    # # register new cmap
    # newcmap = ListedColormap(newcolors) 
    # cm.register_cmap(name = "newcmap", cmap=newcmap)

    # apply new cmap
    #cmap = cm.get_cmap("newcmap", numVert)

#def render_overlays(overlays_buffer, faces, shader):
#     """
#     # ---- Rendering overlays contrast ----
#     Input:
#         overlays_buffer: the vertices coordinates buffers (flatten)
#         faces: the flatmap vertices drawing order (faces)
#         shader: the compiled shader program by opengl pipeline
#     Returns:
#         Indexed drawing overlay to the scene
#     """

#     for overlay in overlays_buffer:
#         VBO = glGenBuffers(1)
#         glBindBuffer(GL_ARRAY_BUFFER, VBO)
#         glBufferData(GL_ARRAY_BUFFER, overlay.shape[0] * 4, overlay, GL_STATIC_DRAW)

#         EBO = glGenBuffers(1)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
#         glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.shape[0] * 4, faces, GL_STATIC_DRAW)

#         position = glGetAttribLocation(shader, "position")
#         glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
#         glEnableVertexAttribArray(position)

#         color = glGetAttribLocation(shader, "color")
#         glVertexAttribPointer(color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
#         glEnableVertexAttribArray(color)
#         glDrawElements(GL_TRIANGLES, faces.shape[0], GL_UNSIGNED_INT, None)