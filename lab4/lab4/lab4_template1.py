#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time
import imgui

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
g_lightYawSpeed = 0.0#145.0
g_lightPitch = -75.0
g_lightPitchSpeed = 0.0#30.0
g_lightDistance = 250.0
g_lightColourAndIntensity = lu.vec3(0.9, 0.9, 0.6)
g_ambientLightColourAndIntensity = lu.vec3(0.1)

g_camera = lu.OrbitCamera([0,0,0], 10.0, -25.0, -35.0)
g_yFovDeg = 45.0

g_currentModelName = "shaderBall1.obj"
g_model = None
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
    """
        Draws a frame

        Parameters:

            x_offset: offset to render within the window
        
            width, height: the size of the frame buffer, or window
    """
    global g_camera
    global g_yFovDeg
    global g_model

    light_rotation = lu.Mat3(lu.make_rotation_y(math.radians(g_lightYaw))) \
        * lu.Mat3(lu.make_rotation_x(math.radians(g_lightPitch))) 
    light_position = g_model.centre \
        + light_rotation * lu.vec3(0,0,g_lightDistance)

    """
        This configures the fixed-function transformation from 
        Normalized Device Coordinates (NDC) to the screen 
        (pixels - called 'window coordinates' in OpenGL documentation).
        See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    """
    glViewport(x_offset, 0, width, height)
    # Set the colour we want the frame buffer cleared to, 
    glClearColor(0.05, 0.1, 0.05, 1.0)
    """
        Tell OpenGL to clear the render target to the clear values 
        for both depth and colour buffers (depth uses the default)
    """
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

    world_to_view = g_camera.get_world_to_view_matrix(lu.vec3(0,1,0))
    view_to_clip = lu.make_perspective(g_yFovDeg, width/height, 0.1, 1500.0)

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

    """
        Bind the shader program such that we can set the uniforms 
        (model.render sets it again)
    """
    glUseProgram(g_shader)

    lu.set_uniform(g_shader, "viewSpaceLightPosition", 
                  lu.transform_point(world_to_view, light_position))
    lu.set_uniform(g_shader, "lightColourAndIntensity", 
                  g_lightColourAndIntensity)
    lu.set_uniform(g_shader, "ambientLightColourAndIntensity", 
                   g_ambientLightColourAndIntensity)
    """
        transform (rotate) light direction into view space 
        (as this is what the ObjModel shader wants)
    """
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
    
    g_model.render(g_shader, ObjModel.RF_Opaque, transforms)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    flags = ObjModel.RF_Transparent| ObjModel.RF_AlphaTested
    g_model.render(g_shader, flags, transforms)
    glDisable(GL_BLEND)

    colour = np.array([1,1,0,1], np.float32)
    lu.draw_sphere(light_position, 10.0, colour, view_to_clip, world_to_view)

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
    global g_model

    #global g_cameraYawDeg
    #global g_cameraPitchDeg

    models = sorted([os.path.basename(p) for p in glob.glob("lab4/lab4/data/*.obj", recursive = False)]) + [""]
    ind = models.index(g_currentModelName)
    _,ind = imgui.combo("Model", ind, models)
    if models[ind] != g_currentModelName:
        g_currentModelName = models[ind]
        load_model(g_currentModelName)   

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
        _,g_lightDistance = imgui.slider_float("Distance", g_lightDistance, 1.00, 1000.0)
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
        names = [str(s) for s in g_model.materials.keys()]
        _,g_currentMaterial = imgui.combo("Material Name", g_currentMaterial, names + [''])
        m = g_model.materials[names[g_currentMaterial]]
        cs = m["color"]
        _,cs["diffuse"] = lu.imguiX_color_edit3_list("diffuse",  cs["diffuse"])
        _,cs["specular"] = lu.imguiX_color_edit3_list("specular", cs["specular"])
        _,cs["emissive"] = lu.imguiX_color_edit3_list("emissive", cs["emissive"])
        imgui.columns(2)
        for n,v in m["texture"].items():
            imgui.image(v if v >= 0 else g_model.defaultTextureOne, 32, 32, (0,1), (1,0))
            imageHovered = imgui.is_item_hovered()
            imgui.next_column()
            imgui.label_text("###"+n, n)
            imgui.next_column()
            if (imageHovered or imgui.is_item_hovered()) and v >= 0:
                imgui.begin_tooltip()
                w,h,name = g_model.texturesById[v]
                imgui.image(v, w / 2, h / 2, (0,1), (1,0))
                imgui.end_tooltip()
        imgui.columns(1)
        _,m["alpha"] = imgui.slider_float("alpha", m["alpha"], 0.0, 1.0)
        _,m["specularExponent"] = imgui.slider_float("specularExponent", m["specularExponent"], 1.0, 2000.0)
        imgui.tree_pop()

    g_model.updateMaterials()

def init_resources() -> None:
    """
        Load any required resources.
    """
    global g_camera
    global g_lightDistance
    global g_shader
    global g_environmentCubeMap

    g_environmentCubeMap = lu.load_cubemap(
        "lab4/lab4/data/cube_maps/" + g_currentEnvMapName + "/%s.jpg", True)   
    load_model(g_currentModelName)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glEnable(GL_FRAMEBUFFER_SRGB)

    # Build with default first since that really should work, so then we have some fallback
    g_shader = build_shader(g_vertexShaderSource, g_fragmentShaderSource)

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

def load_model(modelName):
    global g_model
    global g_lightDistance
    g_model = ObjModel("lab4/lab4/data/" + modelName)
    #g_model = ObjModel("data/house.obj");

    g_camera.target = g_model.centre
    g_camera.distance = lu.length(g_model.centre - g_model.aabbMin) * 3.1
    g_lightDistance = lu.length(g_model.centre - g_model.aabbMin) * 1.3
#endregion
# This does all the openGL setup and window creation needed
# it hides a lot of things that we will want to get a handle on as time goes by.
magic.run_program(
    "COSC3000 - Computer Graphics Lab 4, part 1", 
    960, 640, render_frame, init_resources, draw_ui, update)
