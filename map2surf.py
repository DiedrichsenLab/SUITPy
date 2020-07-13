import math
import nibabel as nb
import numpy as np
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
from matplotlib import cm  # for colormaps
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as cl

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


def suit_plotflatmap(data=[], flat_dir="", surf="FLAT.surf.gii", under="SUIT.shape.gii",\
    undermap="gray", undermap_norm = [-0.2, 0.3], threshold=0.8, cmap='viridis', borders='borders.txt', cscale = 0, alpha=1, overlay_type = "func"):
    """
    # ---- Plot Flatmap ---- 
    Input:
        data: surface data gifti file
        flat_dir: path of the directory where surface, underlay, border files are stored (default = os.getcwd() + /data/)
        surf: surface gifti file
        under: underlay gifti file
        undermap: matplotlib colormap used for underlay - see matplotlib for more
        undermap_norm: default normalization values for "gray" underlay cmap
        threshold: intensities of positive overlay that is displayed (0 to 1 where 1 is all intensities shown)
        cmap: matplotlib colormap for overlay
        borders: txt file containing coordinates (x,y,z) of the border pixels 
        alpha: transparency of the overlay
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

    # Determine underlay and assign color
    underlay_render, underlay_color = load_underlay(underlay_file, surf_vertices, undermap, undermap_norm, 1)

    # Determine overlay and assign color
    overlay_render = load_overlay(data, surf_vertices, underlay_color, cmap, threshold, cscale, overlay_type)

    # Determine borders
    borders_render = load_borders(borders_file, surf_vertices)

    # Render with PyOpenGL
    render(surf_faces, underlay_render, [overlay_render], borders_render)


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


def load_underlay(underlay_file, surf_vertices, undermap, undermap_norm, alpha):
    """
    # ---- Load Underlay ----
        Load vertices coordinates and combine with underlay color to make underlay data
    Input:
        underlay_file: the underlay file name with path. (eg. "SUIT.shape.gii")
        surf_vertices: the vertices coordinates of the flatmap. shape (N, 3)
        undermap: the colormap for the underlay
        alpha: opacity of the underlay
    Returns: the flatten buffer data for underlay
        underlay_render: the underlay buffer data for rendering. shape: flatten
        underlay_color: the underlay color itself, shape (N, 3)
    """

    numVert = surf_vertices.shape[0]

    # load underlay 
    underlay = nb.load(underlay_file)
    underlay_data = underlay.darrays[0].data # underlay.cdata

    # load cmap
    try:
        colors = cm.get_cmap(undermap, numVert)
        underlay_color = colors(underlay_data)
    except:
        raise Exception("Please enter a valid cmap. See matplotlib for more cmaps.")

    # Normalize the underlay cmap
    if undermap_norm != []:
        norm = cm.colors.Normalize(undermap_norm[0], undermap_norm[1])
        underlay_color = norm(underlay_color)

    # Reshape numpy array
    underlay_color = underlay_color[:, 0:3]
    underlay_render = np.concatenate((surf_vertices, underlay_color), axis=1)
    underlay_render = np.concatenate((underlay_render, np.reshape(np.repeat(alpha, underlay_render.shape[0]), (underlay_render.shape[0], 1))), axis=1)
    underlay_render = np.array(underlay_render.flatten(), dtype=np.float32)

    return underlay_render, underlay_color

def load_overlay(data, surf_vertices, underlay_color, cmap, threshold, cscale, overlay_type, alpha=0.3):

    """
    # ---- Load Overlay ----
        Load vertices coordinates and combine with overlay color (converted to selected cmap value)
        to make overlays data for rendering
    Input:
        data: surface data gifti file
        surf_vertices: the vertices coordinates of the flatmap
        underlay_color: the underlay color itself, shape (N, 3)
        cmap: the colormap for the overlay
        threshold: intensities of positive overlay that is displayed (0 to 1 where 1 is all intensities shown)
        alpha: opacity of the overlay
    Returns: 
        overlay_render: the overlay data for rendering. shape: flatten
    """

    numVert = surf_vertices.shape[0]

    # if no data argument, return an array of NaNs
    if data == []:
        data = np.empty([numVert, 1]); 
        data[:] = np.NaN
        return data
    
    # load surface data
    overlay = nb.load(data)
    overlay_data = overlay.darrays[0].data

    if (overlay_type == 'label'):
        overlay_data = overlay_data / 10
        alpha = 1

    # check that overlay data is the right np.array shape
    if (len(overlay_data) != numVert):
        raise Exception('Input data must be an array of size numVertices * 1')

    # check that threshold value is valid
    if ((threshold > 1) | (threshold < 0)):
        raise Exception('Threshold value must be between 0 and 1.')

    """
    # create custom colormap with threshold
    colors = cm.get_cmap(cmap, numVert)
    newcolors = colors(np.linspace(0,1,numVert)) # 0 to 1 is taking the entire cmap (0.25 to 0.75 reduces the cmap range)
    white = np.array([1,1,1,0])

    # find the threshold point and color everything before that value white
    point = (int) (threshold * numVert)
    newcolors[:point, :] = white

    # register new cmap
    newcmap = ListedColormap(newcolors)
    cm.register_cmap(name = "newcmap", cmap=newcmap)

    # apply new cmap
    cmap = cm.get_cmap("newcmap", numVert)
    overlay_color = cmap(overlay_data)    
    """

    # find the threshold point and color everything before that value white
    point = (int) ((threshold * numVert))

    # create custom colormap with threshold
    colors = cm.get_cmap(cmap, point)
    newcolors = colors(np.linspace(0,1,point)) # 0 to 1 is taking the entire cmap (0.25 to 0.75 reduces the cmap range)
    white = np.ones([numVert-point,4])
    newcolors = np.concatenate((white,newcolors))

    # register new cmap
    newcmap = ListedColormap(newcolors) 
    cm.register_cmap(name = "newcmap", cmap=newcmap)

    # apply new cmap
    cmap = cm.get_cmap("newcmap", numVert)
    overlay_color = cmap(overlay_data)  
    overlay_color = overlay_color[:, 0:3]
    overlay_render = np.concatenate((surf_vertices, overlay_color), axis=1)
    overlay_render = np.concatenate((overlay_render, np.reshape(np.repeat(alpha, overlay_render.shape[0]), (overlay_render.shape[0], 1))), axis=1)
    overlay_render = np.array(overlay_render.flatten(), dtype=np.float32)

    return overlay_render

def load_borders(border_file, surf_vertices, alpha=1):
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
    border_render = np.concatenate((border_render, np.reshape(np.repeat(alpha, border_render.shape[0]), (border_render.shape[0], 1))), axis=1)
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
    glBufferData(GL_ARRAY_BUFFER, underlay.shape[0] * 4, underlay, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.shape[0] * 4, faces, GL_STATIC_DRAW)

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 4, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    glDrawElements(GL_TRIANGLES, faces.shape[0], GL_UNSIGNED_INT, None)

def render_overlays(overlays_buffer, faces, shader):
    """
    # ---- Rendering overlays contrast ----
    Input:
        overlays_buffer: the vertices coordinates buffers (flatten)
        faces: the flatmap vertices drawing order (faces)
        shader: the compiled shader program by opengl pipeline
    Returns:
        Indexed drawing overlay to the scene
    """

    for overlay in overlays_buffer:
        VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, overlay.shape[0] * 4, overlay, GL_STATIC_DRAW)

        EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.shape[0] * 4, faces, GL_STATIC_DRAW)

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

def render(vertices_index, underlay, overlays, borders):
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
    # glfw.window_hint(glfw.RESIZABLE, GL_FALSE)
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
    render_overlays(overlays, vertices_index, shader)  # -- the overlays rendering
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
    suit_plotflatmap("data/Wiestler_2011_motor_z.gii", threshold = 0.02, cscale = 0.9)
    suit_plotflatmap("data/HCP_WM_BODY_vs_REST.gii", threshold = 0.01, cmap="jet")
    suit_plotflatmap("data/Buckner_17Networks.label.gii", cmap="jet", overlay_type = "label")


    #vs.vol2surf("data/WHITE_SUIT.surf.gii","data/PIAL_SUIT.surf.gii","data/nifti.txt")