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
import glfw
# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can not be initialized!")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "OpenGL Window", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can not be created!")

# Make the window's context current
glfw.make_context_current(window)

# Check OpenGL version
print(glGetString(GL_VERSION))

vertex_shader = "lab4/lab4/depth_vertex.glsl"
fragment_shader = "lab4/lab4/depth_fragment.glsl"

with open(vertex_shader) as f:
    vertex_shader_source = f.read()
with open(fragment_shader) as f:
    fragment_shader_source = f.read()


def check_shader_compile_status(shader):
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not result:
        error_log = glGetShaderInfoLog(shader)
        raise RuntimeError(f"Shader compilation failed: {error_log}")

# Example usage
vertex_shader = glCreateShader(GL_VERTEX_SHADER)
glShaderSource(vertex_shader, vertex_shader_source)
glCompileShader(vertex_shader)
check_shader_compile_status(vertex_shader)

fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
glShaderSource(fragment_shader, fragment_shader_source)
glCompileShader(fragment_shader)
check_shader_compile_status(fragment_shader)