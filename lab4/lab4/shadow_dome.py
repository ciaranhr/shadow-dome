


#--- Imports ---#
#region
from OpenGL.GL import *
import math
import numpy as np
import time
import imgui
import glm
from PIL import Image
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
    """
        Draws a frame

        Parameters:

            x_offset: offset to render within the window
        
            width, height: the size of the frame buffer, or window
    """
    global g_camera
    global g_yFovDeg
    global g_shadowScreen
    global g_shadowCreators #list for rendering the shadow creators in the frame
    man = load_model(g_shadowCreators[0])
    
    #object transforms and clip transforms and camera transforms

    world_to_view = g_camera.get_world_to_view_matrix(lu.vec3(0,1,0)) 
    man_transform = lu.make_scale(0.8,0.8,0.8) * lu.make_rotation_x(math.pi/2)
    man2_transform = lu.make_rotation_x(math.pi/2) * lu.make_translation(1, 1, 1)
    dome_transform = lu.make_scale(15,15,15)
    view_to_clip = lu.make_perspective(g_yFovDeg, width/height, 0.1, 1500.0)
    model_to_view_man = world_to_view * man_transform 
    model_to_view = world_to_view * dome_transform

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
        "modelToClipTransform" : view_to_clip * world_to_view * dome_transform, 
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
    light_projection = lu.make_orthographic_projection(-1,1,-1,1,1.0,15)
    light_view = lu.make_lookAt(light_position, g_shadowScreen.centre, [0,1,0])
    light_space_matrix = light_projection * light_view 

    
   

    transformsLightMan = {
        "modelToClipTransform" : light_space_matrix * man_transform, 
    }

    transformsLightMan2 = {
        "modelToClipTransform" : light_space_matrix * man2_transform, 
    }

    transformsLightDome = {
        "modelToClipTransform" : light_space_matrix * dome_transform,
    }

    glClearColor(0.05, 0.1, 0.05, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #create the depth map texture bind to relevant buffer
    SHADOW_WIDTH, SHADOW_HEIGHT = 1024, 1024
    depthMapFBO, depthMap = create_depth_map_fbo(SHADOW_WIDTH, SHADOW_HEIGHT)
 
    glEnable(GL_DEPTH_TEST)
    # First pass: render scene from light's perspective
    glUseProgram(g_depthShader)
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO)
    glClear(GL_DEPTH_BUFFER_BIT)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    # Implement your scene rendering function 
    glEnable(GL_DEPTH_TEST)
    man.render(g_depthShader, transforms=transformsLightMan)
    g_shadowScreen.render(g_depthShader, transforms=transformsLightDome)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    #reset viewport
    glViewport(x_offset, 0, width, height)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    #render depth map to quad
    #tex = load_texture("lab4/lab4/data/grass2.png")
    glUseProgram(g_debugShader)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    render_quad()
    
    """
        This configures the fixed-function transformation from 
        Normalized Device Coordinates (NDC) to the screen 
        (pixels - called 'window coordinates' in OpenGL documentation).
        See: https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glViewport.xhtml
    """
    
    glViewport(x_offset, 0, width, height)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    # Set the colour we want the frame buffer cleared to, 
    glClearColor(0.05, 0.1, 0.05, 1.0)
    """
        #Tell OpenGL to clear the render target to the clear values 
        #for both depth and colour buffers (depth uses the default)
    """
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)
    



    """
        #Bind the shader program such that we can set the uniforms 
        #(model.render sets it again)
    #"""
    glUseProgram(g_shader)
    lu.set_uniform(g_shader, "viewSpaceLightPosition", 
                  lu.transform_point(world_to_view, light_position))
    lu.set_uniform(g_shader, "lightColourAndIntensity", 
                  g_lightColourAndIntensity)
    lu.set_uniform(g_shader, "ambientLightColourAndIntensity", 
                   g_ambientLightColourAndIntensity)
    lu.set_uniform(g_shader, "lightSpaceMatrix", light_space_matrix)
    
    #bind shadow map
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, depthMap)
    glUniform1i(glGetUniformLocation(g_shader, "shadowMap"), 1)

    man.render(g_shader, transforms=transforms_man)
    g_shadowScreen.render(g_shader, ObjModel.RF_Opaque, transforms)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    flags = ObjModel.RF_Transparent| ObjModel.RF_AlphaTested
    g_shadowScreen.render(g_shader, flags, transforms)
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
    global g_shadowScreen

    global g_cameraYawDeg
    global g_cameraPitchDeg

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
    global g_debugShader

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
    g_debugShader = create_debug_shader()
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
    g_camera.distance = lu.length(g_shadowScreen.centre - g_shadowScreen.aabbMin) * 15
    g_lightDistance = lu.length(g_shadowScreen.centre - g_shadowScreen.aabbMin) * 15

def load_texture(image_path):
    image = Image.open(image_path)
    image_data = image.convert("RGB").tobytes()
    width, height = image.size

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    return texture

# Create the depth shader program
def create_depth_shader_program():
    with open('lab4/lab4/depth_vertex.glsl', 'r') as f:
        vertex_shader_source = f.read()
    with open('lab4/lab4/depth_fragment.glsl', 'r') as f:
        fragment_shader_source = f.read()   
    shader = build_shader(vertex_shader_source, fragment_shader_source)
    return shader

def create_debug_shader():
    with open('lab4/lab4/depthMapVertex.glsl', 'r') as f:
        vertex_shader_source = f.read()
    with open('lab4/lab4/depthMapFragment.glsl', 'r') as f:
        fragment_shader_source = f.read()   
    shader = build_shader(vertex_shader_source, fragment_shader_source)
    return shader

def render_quad():
    quad_vertices = np.array([
    # positions   # texCoords
        -1.0,  1.0, 0.0, 0.0, 1.0,
        -1.0, -1.0, 0.0, 0.0, 0.0,
        1.0,  1.0, 0.0, 1.0, 1.0,
        1.0, -1.0, 0.0, 1.0, 0.0,
    ], dtype=np.float32)

    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    # Position attribute
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(0))
    
    # Texture coordinate attribute
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(3 * quad_vertices.itemsize))
    
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glBindVertexArray(0)


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        print(f"Shader compile failed with error: {error}")
        raise RuntimeError(f"Shader compile failed with error: {error}")
    return shader


def create_depth_map_fbo(shadow_width, shadow_height):
    depth_map_fbo = glGenFramebuffers(1) 
    depth_map = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depth_map)
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, shadow_width, shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    border_color = [1.0, 1.0, 1.0, 1.0]
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
    
    glBindFramebuffer(GL_FRAMEBUFFER, depth_map_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer not complete")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    return depth_map_fbo, depth_map
    
magic.run_program(
    "Graphics Shadow Dome Project", 
    960, 640, render_frame, init_resources, draw_ui, update)

