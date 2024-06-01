#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time
import imgui
import glm
import magic
# We import the 'lab_utils' module as 'lu' to save a bit of typing while still clearly marking where the code came from.
import lab_utils as lu
from ObjModel import ObjModel
import glob
import os
#endregion
#--- Globals ---#
#region
g_lightYaw = 25.0
g_lightYawSpeed = 0.0 #145.0
g_lightPitch = -75.0
g_lightPitchSpeed = 0.0 #30.0
g_lightDistance = 250.0
g_lightColourAndIntensity = lu.vec3(0.9, 0.9, 0.6)
g_ambientLightColourAndIntensity = lu.vec3(0.1)
SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024
g_camera = lu.OrbitCamera([0,0,0], 8, -25.0, -35.0)
g_yFovDeg = 45.0

g_currentModelName = "the_dome.obj"
g_shadowCreators = ["low_poly_man1.obj"]
g_shadowScreen = None
g_vertexShaderSource = ObjModel.defaultVertexShader
g_fragmentShaderSource = ObjModel.defaultFragmentShader
g_currentFragmentShaderName = 'fragmentShader.glsl'

g_currentEnvMapName = "None"

g_environmentCubeMap = None

g_reloadTimeout = 1.0

g_currentMaterial = 0

"""
    Set the texture unit to use for the cube map to the next 
    free one (free as in not used by the ObjModel)
"""
TU_EnvMap = ObjModel.TU_Max
#endregion
#--- Callbacks ---#
#region
def update(dt: float, keys: dict[str, bool], 
           mouse_delta: list[float]) -> None:
    """
        Update the state of the world.

        Parameters:

            dt: frametime

            keys: current state of all keys

            mouse_delta: mouse movement since the last frame
    """
    global g_camera
    global g_reloadTimeout
    global g_lightYaw
    global g_lightYawSpeed
    global g_lightPitch
    global g_lightPitchSpeed

    g_lightYaw += g_lightYawSpeed * dt
    g_lightPitch += g_lightPitchSpeed * dt

    g_reloadTimeout -= dt
    if g_reloadTimeout <= 0.0:
        reLoad_shader()
        g_reloadTimeout = 1.0

    g_camera.update(dt, keys, mouse_delta)

def render_frame(x_offset: int, width: int, height: int) -> None:
    global g_camera
    global g_yFovDeg
    global g_shadowScreen
    global g_shadowCreators
    
    
    #object transforms and clip transforms and camera transforms

    world_to_view = g_camera.get_world_to_view_matrix(lu.vec3(0,1,0)) 
    man_transform = lu.make_scale(0.2,0.2,0.2) * lu.make_rotation_x(math.pi/2)
    view_to_clip = lu.make_perspective(g_yFovDeg, width/height, 0.1, 1500.0)
    model_to_view_man = world_to_view * man_transform 
    model_to_view = world_to_view

    """
        This is a special transform that ensures that normal vectors 
        remain orthogonal to the surface they are supposed to be even
        in the prescence of non-uniform scaling. It is a 3x3 matrix 
        as vectors don't need translation anyway and this transform 
        is only for vectors, not points. If there is no non-uniform 
        scaling this is just the same as Mat3(modelToViewTransform)
    """
    model_to_view_normal = lu.inverse(
        lu.transpose(lu.Mat3(model_to_view)))
    
    model_to_view_normal_man = lu.inverse(
        lu.transpose(lu.Mat3(model_to_view_man))) 
    
    """
        This dictionary contains a few transforms that are needed to 
        render the ObjModel using the default shader. It would be 
        possible to just set the modelToWorld transform, as this is 
        the only thing that changes between the objects, and compute 
        the other matrices in the vertex shader. However, this would 
        push a lot of redundant computation to the vertex shader and 
        makes the code less self contained, in this way we set all 
        the required parameters explicitly.
    """
    transforms = {
        "modelToClipTransform" : view_to_clip * world_to_view, 
        "modelToViewTransform" : model_to_view,
        "modelToViewNormalTransform" : model_to_view_normal,
    }

    transforms_man = {
        "modelToClipTransform" : view_to_clip * world_to_view * man_transform,
        "modelToViewTransform" : model_to_view_man,
        "modelToViewNormalTransform" : model_to_view_normal_man,
    }
    #light calculations
    light_rotation = lu.Mat3(lu.make_rotation_y(math.radians(g_lightYaw))) \
        * lu.Mat3(lu.make_rotation_x(math.radians(g_lightPitch))) 
    light_position = g_shadowScreen.centre \
        + light_rotation * lu.vec3(0,0,g_lightDistance)
    light_projection = lu.make_orthographic_projection(-10,10,-10,10,1.0,20.0)
    light_view = lu.make_lookAt(light_position, g_shadowScreen.centre, [0,1,0])
    light_space_matrix = light_view * light_projection

    
    depth_map_fbo, depth_map = create_depth_map_framebuffer(SHADOW_WIDTH, SHADOW_HEIGHT)



def draw_ui(width: int, height: int) -> None:
    """
        Draws the UI overlay

        Parameters:
        
            width, height: the size of the frame buffer, or window
    """

    global g_yFovDeg
    global g_currentMaterial
    global g_lightYaw
    global g_lightYawSpeed
    global g_lightPitch
    global g_lightPitchSpeed
    global g_lightDistance
    global g_lightColourAndIntensity
    global g_ambientLightColourAndIntensity
    global g_environmentCubeMap
    global g_currentEnvMapName
    global g_currentModelName
    global g_currentFragmentShaderName
    global g_shadowScreen

    #global g_cameraYawDeg
    #global g_cameraPitchDeg

    models = sorted([os.path.basename(p) for p in glob.glob("lab4/lab4/data/*.obj", recursive = False)]) + [""]
    ind = models.index(g_currentModelName)
    _,ind = imgui.combo("Model", ind, models)
    if models[ind] != g_currentModelName:
        g_currentModelName = models[ind]
        load_model_centre(g_currentModelName)   

    fragmentShaders = sorted([os.path.basename(p) for p in glob.glob("lab4/lab4/frag*.glsl", recursive = False)]) + [""]
    ind = fragmentShaders.index(g_currentFragmentShaderName)
    _,ind = imgui.combo("Fragment Shader", ind, fragmentShaders)
    if fragmentShaders[ind] != g_currentFragmentShaderName:
        g_currentFragmentShaderName = fragmentShaders[ind]
        reLoad_shader()

    if imgui.tree_node("Light", imgui.TREE_NODE_DEFAULT_OPEN):
        imgui.columns(2)
        _,g_lightYaw = imgui.slider_float("Yaw (Deg)", g_lightYaw, -360.00, 360.0)
        imgui.next_column()
        _,g_lightYawSpeed = imgui.slider_float("YSpeed", g_lightYawSpeed, -180.00, 180.0)
        imgui.next_column()
        _,g_lightPitch = imgui.slider_float("Pitch (Deg)", g_lightPitch, -360.00, 360.0)
        imgui.next_column()
        _,g_lightPitchSpeed = imgui.slider_float("PSpeed", g_lightPitchSpeed, -180.00, 180.0)
        imgui.next_column()
        _,g_lightDistance = imgui.slider_float("Distance", g_lightDistance, 1.00, 500.0)
        _,g_lightColourAndIntensity = lu.imguiX_color_edit3_list("ColourAndIntensity",  g_lightColourAndIntensity)
        imgui.columns(1)
        imgui.tree_pop()
    if imgui.tree_node("Environment", imgui.TREE_NODE_DEFAULT_OPEN):
        _,g_ambientLightColourAndIntensity = lu.imguiX_color_edit3_list("AmbientLight",  g_ambientLightColourAndIntensity)
        cubeMaps = sorted([os.path.basename(p) for p in glob.glob("lab4/lab4/data/cube_maps/*", recursive = False)]) + [""]
        ind = cubeMaps.index(g_currentEnvMapName)
        _,ind = imgui.combo("EnvironmentTexture", ind, cubeMaps)
        if cubeMaps[ind] != g_currentEnvMapName:
            glDeleteTextures([g_environmentCubeMap])
            g_currentEnvMapName = cubeMaps[ind]
            g_environmentCubeMap = lu.load_cubemap("lab4/lab4/data/cube_maps/" + g_currentEnvMapName + "/%s.jpg", True)   
        imgui.tree_pop()

    #_,g_yFovDeg = imgui.slider_float("Y-Fov (Degrees)", g_yFovDeg, 1.00, 90.0)
    g_camera.draw_ui()
    if imgui.tree_node("Materials", imgui.TREE_NODE_DEFAULT_OPEN):
        names = [str(s) for s in g_shadowScreen.materials.keys()]
        _,g_currentMaterial = imgui.combo("Material Name", g_currentMaterial, names + [''])
        m = g_shadowScreen.materials[names[g_currentMaterial]]
        cs = m["color"]
        _,cs["diffuse"] = lu.imguiX_color_edit3_list("diffuse",  cs["diffuse"])
        _,cs["specular"] = lu.imguiX_color_edit3_list("specular", cs["specular"])
        _,cs["emissive"] = lu.imguiX_color_edit3_list("emissive", cs["emissive"])
        imgui.columns(2)
        for n,v in m["texture"].items():
            imgui.image(v if v >= 0 else g_shadowScreen.defaultTextureOne, 32, 32, (0,1), (1,0))
            imageHovered = imgui.is_item_hovered()
            imgui.next_column()
            imgui.label_text("###"+n, n)
            imgui.next_column()
            if (imageHovered or imgui.is_item_hovered()) and v >= 0:
                imgui.begin_tooltip()
                w,h,name = g_shadowScreen.texturesById[v]
                imgui.image(v, w / 2, h / 2, (0,1), (1,0))
                imgui.end_tooltip()
        imgui.columns(1)
        _,m["alpha"] = imgui.slider_float("alpha", m["alpha"], 0.0, 1.0)
        _,m["specularExponent"] = imgui.slider_float("specularExponent", m["specularExponent"], 1.0, 2000.0)
        imgui.tree_pop()

    g_shadowScreen.updateMaterials()

def init_resources() -> None:
    """
        Load any required resources.
    """
    global g_camera
    global g_lightDistance
    global g_shader
    global g_environmentCubeMap
    global g_depthShader

    g_environmentCubeMap = lu.load_cubemap(
        "lab4/lab4/data/cube_maps/" + g_currentEnvMapName + "/%s.jpg", True)   
    load_model_centre(g_currentModelName)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glEnable(GL_FRAMEBUFFER_SRGB)

    # Build with default first since that really should work, so then we have some fallback
    g_shader = build_shader(g_vertexShaderSource, g_fragmentShaderSource)
    g_depthShader = create_depth_shader_program()
    reLoad_shader()    
#endregion
#--- Functions ---#
#region
def build_shader(vertex_src: str, fragment_src: str) -> int:
    """
        Build a shader.

        Parameters:

            vertex_src, fragment_src: source code for modules

        Returns:

            integer handle to the new shader program
    """
    shader = lu.build_shader(vertex_src, fragment_src, 
                             ObjModel.getDefaultAttributeBindings())
    if shader:
        glUseProgram(shader)
        ObjModel.setDefaultUniformBindings(shader)
        glUseProgram(0)
    return shader

def itemListCombo(currentItem, items, name):
    ind = items.index(currentItem)
    _,ind = imgui.combo(name, ind, items)
    return items[ind]

def reLoad_shader():
    global g_vertexShaderSource
    global g_fragmentShaderSource
    global g_shader
    
    vertexShader = ""
    with open('lab4/lab4/vertexShader.glsl') as f:
        vertexShader = f.read()
    fragmentShader = ""
    with open('lab4/lab4/'+g_currentFragmentShaderName) as f:
        fragmentShader = f.read()

    if g_vertexShaderSource != vertexShader \
        or fragmentShader != g_fragmentShaderSource:
        newShader = build_shader(vertexShader, fragmentShader)
        if newShader:
            if g_shader:
                glDeleteProgram(g_shader)
            g_shader = newShader
            print("Reloaded shader, ok!")
        g_vertexShaderSource = vertexShader
        g_fragmentShaderSource = fragmentShader


def load_shadowModels(modelNames):
    global g_shadowCreators
    g_lightDistance

def load_model(modelName) ->ObjModel:
    model = ObjModel("lab4/lab4/data/" + modelName)
    return model
    
def load_model_centre(modelName):
    global g_shadowScreen
    global g_lightDistance
    g_shadowScreen = ObjModel("lab4/lab4/data/" + modelName)
    #g_shadowScreen = ObjModel("data/house.obj");

    g_camera.target = g_shadowScreen.centre
    g_camera.distance = lu.length(g_shadowScreen.centre - g_shadowScreen.aabbMin) * 3.1
    g_lightDistance = lu.length(g_shadowScreen.centre - g_shadowScreen.aabbMin) * 1.3

# Create the depth shader program
def create_depth_shader_program():
    with open('lab4/lab4/depth_vertex.glsl', 'r') as f:
        vertex_shader_source = f.read()
    with open('lab4/lab4/depth_fragment.glsl', 'r') as f:
        fragment_shader_source = f.read()   
    shader = build_shader(vertex_shader_source, fragment_shader_source)
    return shader


def create_depth_map_framebuffer(shadow_width, shadow_height):
    depth_map_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
    
    depth_map = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_map)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_width, shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    borderColor = [1.0, 1.0, 1.0, 1.0]
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    return depth_map_fbo, depth_map


def render_objects(objects:list[ObjModel], shaderProgram, transforms):
    for obj in objects:
        obj.render


magic.run_program(
    "Graphics Shadow Dome Project", 
    960, 640, render_frame, init_resources, draw_ui, update)