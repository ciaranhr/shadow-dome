#--- Imports ---#
#region
from OpenGL.GL import *
import numpy as np
import math
import imgui
from PIL import Image
#endregion
#--- Globals ---#
#region
g_sphereVertexArrayObject = None
g_sphereShader = None
g_numSphereVerts = 0
#endregion
#--- Vectors ---#
#region
Vec2 = np.ndarray
Vec3 = np.ndarray
def vec2(x: float, y: float | None = None) -> Vec2:
    """
        Constructs and returns a 2D Vector.

        Parameters:

            x: x component

            y: optional y component. If none is provided, the x
                component is used instead.
        
        Returns:

            The new vector.
    """
    if y is None:
        return np.array([x,x], dtype=np.float32)
    return np.array([x,y], dtype=np.float32)

def vec3(
    x: float, y: float | None = None, z: float | None = None) -> Vec3:
    """
        Constructs and returns a 2D Vector.

        Parameters:

            x: x component

            y: optional y component. If none is provided, the x
                component is used for all components.
            
            z: optional z component. If none is provided, the y
                component will be used instead.
        
        Returns:

            The new vector.
    """
    
    if y is None:
        return np.array([x,x,x], dtype=np.float32)
    if z is None:
        return np.array([x,y,y], dtype=np.float32)
    vector = np.array([x, y, z], dtype=np.float32)
    return vector

def normalize(v: Vec2 | Vec3) -> Vec2 | Vec3:
    """
        Returns a normalized copy of the given vector.
    """
    norm = np.linalg.norm(v)
    return v / norm

def length(v: Vec2 | Vec3) -> float:
    """
        Returns the magnitude of the given vector.
    """
    return np.linalg.norm(v)

def cross(a: Vec3, b: Vec3) -> Vec3:
    """
        Returns the cross product of the vectors a & b
    """
    return np.cross(a,b)

def mix(v0: Vec2 | Vec3, v1: Vec2 | Vec3, t: float) -> Vec2 | Vec3:
    """
        Linearly interpolate from v0 to v1, t in [0,1].
        Named to match GLSL.
    """
    return v0 * (1.0 - t) + v1 * t

def dot(a: Vec2 | Vec3, b: Vec2 | Vec3) -> float:
    """
        Returns the dot product of the vectors a & b.
    """
    return np.dot(a, b)
#endregion
#--- Matrices ---#
#region
class Mat4:

    def __init__(self, 
        p : "Mat4 | Mat3 | np.ndarray | list[list[float]]" = None):
        """
            Construct a Mat4 from a description.

            Parameters:

                p: description, can be an iterable python object
                    like list or tuple, a numpy array, or another
                    Mat3 or Mat4.
        """
        if p is None:
            self.mat_data = np.identity(4, dtype=np.float32)
        elif isinstance(p, Mat3):
            self.mat_data = np.identity(4, dtype=np.float32)
            self.mat_data[0:3, 0:3] = p.mat_data
        elif isinstance(p, Mat4):
            self.mat_data = p.mat_data.copy()
        else:
            self.mat_data = np.array(p, dtype = np.float32)

    def __mul__(self, 
        other: "np.ndarray | list[float] | Mat4") \
        -> "np.ndarray | list[float] | Mat4":
        """
            overload the multiplication operator to enable sane 
            looking transformation expressions!
        """

        """
            if it is a list, we let numpy attempt to convert the data
            we then return it as a list also (the typical use case is 
            for transforming a vector). Could be made more robust...
        """
        other_data = other
        if isinstance(other, Mat4):
            other_data = other.mat_data
        result = np.dot(other_data, self.mat_data)
        if isinstance(other, list):
            #python list
            return list(result)
        if isinstance(other, Mat4):
            #Mat4
            return Mat4(result)
        #numpy array
        return result
    
    def get_data(self) -> np.ndarray:
        """
            Returns the matrix's data as a contiguous array for
            upload to OpenGL
        """
        return self.mat_data

    def _inverse(self) -> "Mat4":
        """
            Returns an inverted copy, does not change the object 
            (for clarity use the global function instead)
            only implemented as a member to make it easy to overload
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat4(np.linalg.inv(self.mat_data))
    
    def _affine_inverse(self) -> "Mat4":
        """
            Returns an inverted copy, does not change the object.
            
            Matrices which represent affine transformations have
            closed form inverses. This is actually how the lookat
            transform is calculated.
        """
        A = self.mat_data
        data = (
            (A[0][0], A[1][0], A[2][0], 0.0),
            (A[0][1], A[1][1], A[2][1], 0.0),
            (A[0][2], A[1][2], A[2][2], 0.0),
            (-np.dot(A[0], A[3]), -np.dot(A[1], A[3]), -np.dot(A[2], A[3]), 1.0)
        )
        return Mat4(data)

    def _transpose(self) -> "Mat4":
        """
            Returns a matrix representing the transpose of
            this matrix. This matrix is not altered.
        """
        return Mat4(self.mat_data.T)

    def _set_open_gl_uniform(self, location: int) -> None:
        """
            Uploads the matrix to the given location.
        """
        glUniformMatrix4fv(location, 1, GL_FALSE, self.mat_data)

class Mat3:
    
    def __init__(self, 
        p : "Mat4 | Mat3 | np.ndarray | list[list[float]]" = None):
        """
            Construct a Mat3 from a description.

            Parameters:

                p: description, can be an iterable python object
                    like list or tuple, a numpy array, or another
                    Mat3 or Mat4.
        """
        if p is None:
            self.mat_data = np.identity(3, dtype=np.float32)
        elif isinstance(p, Mat3):
            self.mat_data = p.mat_data.copy()
        elif isinstance(p, Mat4):
            self.mat_data = np.identity(3, dtype = np.float32)
            self.mat_data[0:3, 0:3] = p.mat_data[0:3, 0:3]
        else:
            self.mat_data = np.array(p, dtype = np.float32)

    def __mul__(self, 
        other: "np.ndarray | list[float] | Mat3") \
        -> "np.ndarray | list[float] | Mat3":
        """
            overload the multiplication operator to enable sane 
            looking transformation expressions!
        """

        """
            if it is a list, we let numpy attempt to convert the data
            we then return it as a list also (the typical use case is 
            for transforming a vector). Could be made more robust...
        """
        other_data = other
        if isinstance(other, Mat3):
            other_data = other.mat_data
        result = np.dot(other_data, self.mat_data)
        if isinstance(other, list):
            #python list
            return list(result)
        if isinstance(other, Mat3):
            #Mat4
            return Mat3(result)
        #numpy array
        return result
    
    def get_data(self) -> np.ndarray:
        """
            Returns the matrix's data as a contiguous array for
            upload to OpenGL
        """
        return self.mat_data

    def _inverse(self) -> "Mat3":
        """
            Returns an inverted copy, does not change the object 
            (for clarity use the global function instead) only 
            implemented as a member to make it easy to overload 
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat3(np.linalg.inv(self.mat_data))

    def _transpose(self) -> "Mat3":
        """
            Returns a transposed copy, does not change the object 
            (for clarity use the global function instead) only 
            implemented as a member to make it easy to overload 
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat3(self.mat_data.T)

    def _set_open_gl_uniform(self, location: int) -> None:
        """
            Uploads the matrix to the given location.
        """

        glUniformMatrix3fv(location, 1, GL_FALSE, self.mat_data)

def make_translation(x: float, y: float, z: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a translation by the
        given amounts in the x,y,z axes.
    """

    return Mat4([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [x,y,z,1]])

def make_scale(x: float, y: float, z: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a scale transform
        of the given amount in the x,y,z axes.
    """
    return Mat4([[x,0,0,0],
                 [0,y,0,0],
                 [0,0,z,0],
                 [0,0,0,1]])

def make_rotation_y(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the y axis by the given angle (in radians).
    """

    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[ c, 0, s, 0],
                 [ 0, 1, 0, 0],
                 [-s, 0, c, 0],
                 [ 0, 0, 0, 1]])

def make_rotation_x(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the x axis by the given angle (in radians).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[1,  0, 0, 0],
                 [0,  c, s, 0],
                 [0, -s, c, 0],
                 [0,  0, 0, 1]])

def make_rotation_z(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the z axis by the given angle (in radians).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[ c, s, 0, 0],
                 [-s, c, 0, 0],
                 [ 0, 0, 1, 0],
                 [ 0, 0, 0, 1]])

def make_lookFrom(eye: Vec3, direction: Vec3, up: Vec3) -> Mat4:
    """
        The reason we need a 'look from', and don't just use 
        lookAt(pos, pos+dir, up) is because if pos is large 
        (i.e., far from the origin) and 'dir' is a unit vector 
        (common case) then the precision loss in the addition 
        followed by subtraction in lookAt to get the direction 
        back is _significant_, and leads to jerky camera movements.

        Parameters:

            eye: camera position

            direction: camera direction

            up: camera's up vector
    """
    f = normalize(direction)
    U = np.array(up[:3], dtype=np.float32)
    s = normalize(np.cross(f, U))
    u = normalize(np.cross(s, f))

    """
        {side, up, forwards} now form an orthonormal
        basis for R3, being unit vectors in the camera's
        local {x,y,z} axes respectively.

        Now we use this to produce the affine inverse of
        the camera's frame of reference.
    """

    M = np.array((
        (           s[0],            u[0],          -f[0], 0.0),
        (           s[1],            u[1],          -f[1], 0.0),
        (           s[2],            u[2],          -f[2], 0.0),
        (-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0)), 
        dtype=np.float32)
    return Mat4(M)

def make_lookAt(eye: Vec3, target: Vec3, up: Vec3) -> Mat4:
    """
        Constructs and returns a 4x4 matrix representing a 
        view transform, i.e., from world to view space.
        this is basically the same as what we saw in 
        Lecture #2 for placing the car in the world, 
        except the inverse! (and also view-space 'forwards' is the 
        negative z-axis)

        Parameters:

            eye: The camera's position

            target: Point being looked at

            up: The camera's up vector (helps to orient the camera).
        
        Returns:

            An appropriate view transform.
    """
    direction = np.array(target[:3], dtype = np.float32) - np.array(eye[:3], dtype = np.float32)
    return make_lookFrom(eye, direction, up)

def make_perspective(
    fovy: float, aspect: float, n: float, f: float) -> Mat4:
    """
        Make a perspective projection matrix.

        Parameters:

            fovy: field of view (in degrees)

            aspect: aspect ratio of the screen (w/h)

            n: near distance

            f: far distance
        
        Returns:

            The perspective projection matrix.
    """
    radFovY = math.radians(fovy)
    tanHalfFovY = math.tan(radFovY / 2.0)
    sx = 1.0 / (tanHalfFovY * aspect)
    sy = 1.0 / tanHalfFovY
    zz = -(f + n) / (f - n)
    zw = -(2.0 * f * n) / (f - n)

    return Mat4([[sx,  0,  0,  0],
                 [ 0, sy,  0,  0],
                 [ 0,  0, zz, -1],
                 [ 0,  0, zw,  0]])

def inverse(mat: Mat3 | Mat4) -> Mat3 | Mat4:
    """
        returns an inverted copy, does not change the object.
    """
    return mat._inverse()

def transpose(mat: Mat3 | Mat4) -> Mat3 | Mat4:
    """
        returns a transposed copy, does not change the object.
    """
    return mat._transpose()
#endregion
#--- Camera ---#
#region
class FreeCamera:
    """
        Very simple free-moving camera class
    """
    position = vec3(0.0,0.0,0.0)
    yawDeg = 0.0
    pitchDeg = 0.0
    maxSpeed = 10
    angSpeed = 90

    def __init__(self, pos: list[float], 
        yaw_deg: float, pitch_deg: float):
        """
            Initialize the camera.

            Parameters:

                pos: camera's (x,y,z) position

                yaw_deg, pitch_deg: yaw and pitch of the
                    camera, in degrees.

        """
        #star operator unpacks a list in-place
        self.position = vec3(*pos)
        self.yawDeg = yaw_deg
        self.pitchDeg = pitch_deg

    def update(self, dt: float, 
        keys: dict[str, bool], mouse_delta: list[float]) -> None:
        """
            Update the camera's state each frame.

            Parameters:

                dt: approximate frametime

                keys: current state of all keys

                mouse_delta: mouse movement since the last update
        """
        move_speed = 0.0
        turn_speed = 0.0
        pitch_speed = 0.0
        strafe_speed = 0.0

        if keys["UP"] or keys["W"]:
            move_speed += self.maxSpeed
        if keys["DOWN"] or keys["S"]:
            move_speed -= self.maxSpeed
        if keys["LEFT"]:
            turn_speed -= self.angSpeed
        if keys["RIGHT"]:
            turn_speed += self.angSpeed
        if keys["A"]:
            strafe_speed += self.maxSpeed
        if keys["D"]:
            strafe_speed -= self.maxSpeed

        # Mouse look is enabled with right mouse button
        if keys["MOUSE_BUTTON_LEFT"]:
            turn_speed = mouse_delta[0] * self.angSpeed
            pitch_speed = mouse_delta[1] * self.angSpeed

        self.yawDeg += turn_speed * dt
        self.pitchDeg = min(89.0, max(-89.0, self.pitchDeg + pitch_speed * dt))

        cameraRotation = Mat3(make_rotation_y(math.radians(self.yawDeg))) \
            * Mat3(make_rotation_x(math.radians(self.pitchDeg))) 
        forwards = cameraRotation * vec3(0,0,1)
        self.position += forwards * move_speed * dt

        """
            strafe measns perpendicular left-right movement, 
            so rotate the X unit vector and go
        """
        self.position += cameraRotation * vec3(1,0,0) * strafe_speed * dt

    def draw_ui(self) -> None:
        """
            Draw the camera's state on the UI.
        """
        if imgui.tree_node("FreeCamera", imgui.TREE_NODE_DEFAULT_OPEN):
            _,self.yawDeg = imgui.slider_float("Yaw (Deg)", self.yawDeg, -180.00, 180.0)
            _,self.pitchDeg = imgui.slider_float("Pitch (Deg)", self.pitchDeg, -89.00, 89.0)
            imgui.tree_pop()
    
    def get_world_to_view_matrix(self, up: Vec3) -> Mat4:
        """
            Get the camera's view transform.

            Parameters:

                up: The camera's up vector.

            Returns:

                A 4x4 Matrix which maps world space into
                the camera's view space
        """
        forwards = Mat3(make_rotation_y(math.radians(self.yawDeg))) \
            * Mat3(make_rotation_x(math.radians(self.pitchDeg))) \
            * np.array([0, 0, 1], dtype=np.float32)
        return make_lookFrom(self.position, forwards, up)

class OrbitCamera:
    """
        Very simple target-orbiting camera class
    """
    target = vec3(0.0,0.0,0.0)
    distance = 1.0
    yawDeg = 0.0
    pitchDeg = 0.0
    maxSpeed = 10
    angSpeed = 90
    position = vec3(0.0,1.0,0.0)

    def __init__(self, target: list[float], distance: float, 
                 yawDeg: float, pitchDeg:float):
        """
            Initialize a new orbit camera.

            Parameters:

                target: center of orbit

                distance: radius of orbit

                yawDeg: initial yaw, in degrees

                pitchDeg: initial pitch, in degrees
        """
        self.target = vec3(*target)
        self.yawDeg = yawDeg
        self.pitchDeg = pitchDeg
        self.distance = distance

    def update(self, dt: float, keys: dict[str, bool], 
               mouse_delta: list[float]) -> None:
        """
            Update the state of the camera.

            Parameters:

                dt: frametime

                keys: the current state of all keys

                mouse_delta: mouse movement since last frame
        """
        yaw_speed = 0.0
        pitch_speed = 0.0

        # Mouse look is enabled with right mouse button
        if keys["MOUSE_BUTTON_LEFT"]:
            yaw_speed = mouse_delta[0] * self.angSpeed
            pitch_speed = mouse_delta[1] * self.angSpeed

        if keys["MOUSE_BUTTON_RIGHT"]:
            self.distance = max(1.0, self.distance + mouse_delta[1])

        self.yawDeg += yaw_speed * dt
        self.pitchDeg = min(89.0, 
                        max(-89.0, self.pitchDeg + pitch_speed * dt))

        rotation = Mat3(make_rotation_y(math.radians(self.yawDeg))) \
            * Mat3(make_rotation_x(math.radians(-self.pitchDeg))) 
        self.position = rotation * vec3(0,0,self.distance)

    def draw_ui(self) -> None:
        """
            Draw the camera's UI.
        """
        if imgui.tree_node("OrbitCamera", imgui.TREE_NODE_DEFAULT_OPEN):
            _,self.yawDeg = imgui.slider_float("Yaw (Deg)", self.yawDeg, -180.00, 180.0)
            _,self.pitchDeg = imgui.slider_float("Pitch (Deg)", self.pitchDeg, -89.00, 89.0)
            _,self.distance = imgui.slider_float("Distance", self.distance, 1.00, 1000.0)
            imgui.tree_pop()
    
    def get_world_to_view_matrix(self, up: Vec3) -> Mat4:
        """
            Get the camera's view transform.

            Parameters:

                up: The camera's up vector.

            Returns:

                A 4x4 Matrix which maps world space into
                the camera's view space
        """
        return make_lookAt(self.position, self.target, up)
#endregion
#--- ImGui ---#
#region
def imguiX_color_edit3_list(label: str, v: Vec3) \
    -> tuple[bool, list[float]]:
    """
        Displays an RGB color edit widget that consists of 
        3 inputs (RGB) along with a color picker.

        Parameters:

            label: label for the widget

            v: the color to edit
        
        Returns:

            whether the color was changed, along with its value.
    """
    #edit_flags = imgui.GuiColorEditFlags_Float \
    #    | imgui.GuiColorEditFlags_HSV
    edit_flags = 0
    a,b = imgui.color_edit3(label, *v, flags = edit_flags)
    return a,list(b)
#endregion
#--- Geometry ---#
#region
def subdivide(dest: list[Vec3], 
              v0: Vec3, v1: Vec3, v2: Vec3, level: int) -> None:
    """
        Recursively subdivide a triangle with its vertices on the 
        surface of the unit sphere such that the new vertices also 
        are on part of the unit sphere.

        Parameters:

            dest: list into which new vertices are appended

            v0, v1, v2: vertices of the triangle

            level: subdivision level, subdivision terminates at
                level zero
    """
    
    if level == 0:
        """
            If we have reached the terminating level, 
            just output the vertex position
        """
        dest.append(v0)
        dest.append(v1)
        dest.append(v2)
        return
    """
        ...we subdivide the input triangle into four equal 
        sub-triangles. The mid points are the half way between 
        two vertices, which is really (v0 + v2) / 2, but instead 
        we normalize the vertex to 'push' it out to the surface 
        of the unit sphere.
    """
    v3 = normalize(v0 + v1)
    v4 = normalize(v1 + v2)
    v5 = normalize(v2 + v0)
    
    """
        ...and then recursively call this function for each of 
        those (with the level decreased by one)
    """
    subdivide(dest, v0, v3, v5, level - 1)
    subdivide(dest, v3, v4, v5, level - 1)
    subdivide(dest, v3, v1, v4, level - 1)
    subdivide(dest, v5, v4, v2, level - 1)

def create_sphere(detail: int) -> list[Vec3]:
    """
        Create a sphere.

        Parameters:

            detail: number of subdivisions to perform

        Returns:

            A list of positions on the sphere.
    """
    vertices = []

    """
        The root level sphere is formed from 8 triangles in a 
        diamond shape (two pyramids)
    """
    a = [ 0,  1,  0]
    b = [ 0, -1,  0]
    c = [ 0,  0,  1]
    d = [ 1,  0,  0]
    e = [ 0,  0, -1]
    f = [-1,  0,  0]
    subdivide(vertices, vec3(*a), vec3(*c), vec3(*d), detail)
    subdivide(vertices, vec3(*a), vec3(*d), vec3(*e), detail)
    subdivide(vertices, vec3(*a), vec3(*e), vec3(*f), detail)
    subdivide(vertices, vec3(*a), vec3(*f), vec3(*c), detail)

    subdivide(vertices, vec3(*b), vec3(*d), vec3(*c), detail)
    subdivide(vertices, vec3(*b), vec3(*c), vec3(*f), detail)
    subdivide(vertices, vec3(*b), vec3(*f), vec3(*e), detail)
    subdivide(vertices, vec3(*b), vec3(*e), vec3(*d), detail)

    return vertices

def transform_point(transform: Mat4, point: Vec3) -> Vec3:
    """
        Helper function to extend a 3D point to homogeneous, 
        transform it and back again. (For practically all cases 
        except projection, the W component is still 1 after, but 
        this covers the correct implementation). Note that it does 
        not work for (direction) vectors! For vectors we're usually 
        better off just using the 3x3 part of the matrix.
    """
    x,y,z,w = transform * [point[0], point[1], point[2], 1.0]
    return vec3(x,y,z) / w
#endregion
#--- Rendering ---#
#region
def draw_sphere(position: Vec3, radius: float, colour: Vec3, 
    viewToClipTransform: Mat4, worldToViewTransform: Mat4) -> None:
    """
        Draw a sphere.

        Parameters:

            position, radius, colour: describe the sphere's
                appearance.
            
            viewToClipTransform, worldToViewTransform: associated
                transform matrices
    """
    global g_sphereVertexArrayObject
    global g_sphereShader
    global g_numSphereVerts

    modelToWorldTransform = make_translation(*position) \
        * make_scale(radius, radius, radius)

    if not g_sphereVertexArrayObject:
        sphereVerts = create_sphere(3)
        g_numSphereVerts = len(sphereVerts)
        g_sphereVertexArrayObject = create_vertex_array_object()
        create_buffer_from_data(g_sphereVertexArrayObject, sphereVerts, 0)
        # redundantly add as normals...
        create_buffer_from_data(g_sphereVertexArrayObject, sphereVerts, 1)

        vertex_src = """
            #version 330
            in vec3 positionIn;
            in vec3 normalIn;

            uniform mat4 modelToClipTransform;
            uniform mat4 modelToViewTransform;
            uniform mat3 modelToViewNormalTransform;

            // 'out' variables declared in a vertex shader can be accessed in the subsequent stages.
            // For a fragment shader the variable is interpolated (the type of interpolation can be modified, try placing 'flat' in front here and in the fragment shader!).
            out VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            void main() 
            {
                v2f_viewSpacePosition = (modelToViewTransform * vec4(positionIn, 1.0)).xyz;
                v2f_viewSpaceNormal = normalize(modelToViewNormalTransform * normalIn);

                // gl_Position is a buit-in 'out'-variable that gets passed on to the clipping and rasterization stages (hardware fixed function).
                // it must be written by the vertex shader in order to produce any drawn geometry. 
                // We transform the position using one matrix multiply from model to clip space. Note the added 1 at the end of the position to make the 3D
                // coordinate homogeneous.
                gl_Position = modelToClipTransform * vec4(positionIn, 1.0);
            }
"""

        fragment_src = """
            #version 330
            // Input from the vertex shader, will contain the interpolated (i.e., area weighted average) vaule out put for each of the three vertex shaders that 
            // produced the vertex data for the triangle this fragmet is part of.
            in VertexData
            {
                vec3 v2f_viewSpacePosition;
                vec3 v2f_viewSpaceNormal;
            };

            uniform vec4 sphereColour;

            out vec4 fragmentColor;

            void main() 
            {
                float shading = max(0.0, dot(normalize(-v2f_viewSpacePosition), v2f_viewSpaceNormal));
                fragmentColor = vec4(sphereColour.xyz * shading, sphereColour.w);

            }
"""
        attributes = {"positionIn" : 0, "normalIn" : 1}
        g_sphereShader = build_shader(vertex_src, fragment_src, 
                                     attributes)


    glUseProgram(g_sphereShader)
    set_uniform(g_sphereShader, "sphereColour", colour)

    modelToClipTransform = viewToClipTransform * worldToViewTransform * modelToWorldTransform
    modelToViewTransform = worldToViewTransform * modelToWorldTransform
    modelToViewNormalTransform = inverse(transpose(Mat3(modelToViewTransform)))
    set_uniform(g_sphereShader, "modelToClipTransform", modelToClipTransform);
    set_uniform(g_sphereShader, "modelToViewTransform", modelToViewTransform);
    set_uniform(g_sphereShader, "modelToViewNormalTransform", modelToViewNormalTransform);

    glBindVertexArray(g_sphereVertexArrayObject)
    glDrawArrays(GL_TRIANGLES, 0, g_numSphereVerts)

def flatten(array: list[any], data_type: np.dtype) -> np.ndarray:
    """
        Turns a multidimensional array into a 1D array

        Parameters:

            array: the array to be flattened

            data_type: the format of the data.

        Returns:

            a one dimensional numpy array holding the
            original data.
    """
    data_array = np.array(array, dtype = data_type)
    length = data_array.nbytes // data_array.itemsize
    return data_array.reshape(length)

def upload_data(
    buffer: int, data: list[any], data_type: np.dtype) -> None:
    """
        Uploads the given set of numbers to the given buffer object.

        Parameters:

            buffer: integer handle to the buffer which
                will be written to.
            
            data: the data to be uploaded.

            data_type: the format of the data.
    """
    flat_data = flatten(data, data_type)
    """
        Upload data to the currently bound GL_ARRAY_BUFFER, 
        note that this is completely anonymous binary data, 
        no type information is retained (we'll supply that 
        later in glVertexAttribPointer)
    """
    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, 
                 flat_data, GL_STATIC_DRAW)

def create_vertex_array_object() -> int:
    """
        Creates and returns a new Vertex Array Object
    """
    return glGenVertexArrays(1)

def create_buffer_from_data(vao: int, data: list[Vec3], 
                            attribute_index: int) -> int:
    """
        Create a buffer, fill it with data, and bind it to the given
        vertex array object.

        Parameters:

            vao: Vertex Array Object for the current mesh

            data: set of data to upload to the new buffer

            attribute_index: index of the attribute represented
                by the data.
        
        Returns:

            An integer handle to the new buffer.
    """
    glBindVertexArray(vao)
    buffer = glGenBuffers(1)
    upload_data(buffer, data, np.float32)

    glBindBuffer(GL_ARRAY_BUFFER, buffer)
    element_count = len(data[0])
    data_format = GL_FLOAT
    normalize_data = GL_FALSE
    stride = 0 #indicates tightly packed
    offset = ctypes.c_void_p(0)
    glVertexAttribPointer(attribute_index, element_count, 
        data_format, normalize_data, stride, offset)
    glEnableVertexAttribArray(attribute_index)

    """
        Unbind the buffers again to avoid unintentional 
        GL state corruption (this is something that can be rather 
        inconvenient to debug)
    """
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return buffer

def create_and_add_index_buffer(vao: int, indices: list[int]) -> None:
    """
        Create and add an index buffer to a mesh.

        Parameters:

            vao: Vertex Array Object for the current mesh

            indices: set of indices to upload to the new buffer.
                These will be interpreted as 16 bit unsigned,
                which saves some space and is enough for most 
                models.
        
        Returns:

            An integer handle to the new buffer.
    """

    glBindVertexArray(vao)
    index_buffer = glGenBuffers(1)
    upload_data(index_buffer, indices, np.uint16)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    """
        Bind the index buffer as the element array buffer of the VAO.
        This causes it to stay bound to this VAO - fairly unobvious.
    """
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

    """
        Unbind the buffers again to avoid unintentional 
        GL state corruption (this is something that can be rather 
        inconvenient to debug)
    """
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return index_buffer

def get_shader_info_log(obj: int) -> str:
    """
        Get the current error message from the shader.

        Parameters:

            obj: integer handle to the shader object being
            compiled or linked.
        
        Returns:

            The current error message, or an empty string
            if the operation failed silently (GOOD LUCK)
    """
    logLength = glGetShaderiv(obj, GL_INFO_LOG_LENGTH)

    if logLength > 0:
        return glGetShaderInfoLog(obj).decode()

    return ""

def compile_and_attach_shader_module(shader_program: int, 
    shader_stage: int, source_code: str) -> bool:
    """
        Compile the source code for a shader module 
        (e.g., vertex / fragment) and attaches it to the
        given shader program.

        Parameters:

            shader_program: the program to attach the compiled
            module to.

            shader_stage: the stage which the module is meant for

            source_code: the source code to be compiled.
        
        Returns:

            Whether the source code was successfully compiled.
    """

    # Create the opengl shader module object
    module = glCreateShader(shader_stage)
    # upload the source code for the shader
    # Note the function takes an array of source strings and lengths.
    glShaderSource(module, [source_code])
    glCompileShader(module)

    """
        If there is a syntax or other compiler error during shader 
        compilation, we'd like to know
    """
    compile_ok = glGetShaderiv(module, GL_COMPILE_STATUS)

    if not compile_ok:
        err = get_shader_info_log(module)
        print("SHADER COMPILE ERROR: '%s'" % err);
        return False

    glAttachShader(shader_program, module)
    glDeleteShader(module)
    return True

def build_shader(
    vertex_shader_source: str, fragment_shader_source: str, 
    attrib_locations: dict[str, int], 
    frag_data_locations: dict[str, int] = {}) -> int:
    """
        Creates a more general shader that binds a map of attribute 
        streams to the shader and the also any number of output 
        shader variables.
        The fragDataLocs can be left out for programs that don't 
        use multiple render targets as the default for any 
        output variable is zero.

        Parameters:

            vertex_shader_source, fragment_shader_source: source code
                for the vertex and fragment modules.
            
            attrib_locations: location of each attribute.
                eg. {"position": 0, "colour": 1, ...}
            
            frag_data_locations: optional, describes each colour
                attachment.

                eg. {"albedo": 0, "normal": 1, ...}
    """
    shader = glCreateProgram()

    if compile_and_attach_shader_module(
        shader, GL_VERTEX_SHADER, vertex_shader_source) \
        and compile_and_attach_shader_module(
            shader, GL_FRAGMENT_SHADER, fragment_shader_source):
	    # Link the attribute names we used in the vertex shader to the integer index
        for name, location in attrib_locations.items():
            glBindAttribLocation(shader, location, name)

        """
	        If we have multiple images bound as render targets, 
            we need to specify which 'out' variable in the fragment 
            shader goes where in this case it is totally redundant 
            as we only have one (the default render target, 
            or frame buffer) and the default binding is always zero.
        """
        for name, location in frag_data_locations.items():
            glBindFragDataLocation(shader, location, name)

        """
            once the bindings are done we can link the program stages 
            to get a complete shader pipeline. This can yield errors,
            for example if the vertex and fragment shaders don't have 
            compatible out and in variables (e.g., the fragment 
            shader expects some data that the vertex shader is not 
            outputting).
        """
        glLinkProgram(shader)
        linkStatus = glGetProgramiv(shader, GL_LINK_STATUS)
        if not linkStatus:
            err = glGetProgramInfoLog(shader)
            print(f"SHADER LINKER ERROR: {err}")
            return None
    return shader

def get_uniform_location_debug(shader_program: int, name: str) -> int:
    """
        Attempt to fetch the location of the given uniform within the
        given shader program. Prints out a helpful message upon failure.
    """
    loc = glGetUniformLocation(shader_program, name)
    # Useful point for debugging, replace with silencable logging 
    #if loc == -1:
    #    print(f"Uniform \'{name}\' was not found")
    return loc

def set_uniform(shader: int, name: str, value: any) -> None:
    """
        Sets uniforms of different types, looks the way it does 
        since Python does not have support for function overloading
        (as C++ has for example). This function covers the types 
        used in the code here, but makes no claim of completeness. 
        
        Parameters:

            shader: integer handle to the shader program

            name: name of the uniform to set

            value: value with which to set the uniform
    """
    loc = get_uniform_location_debug(shader, name)
    if isinstance(value, float):
        glUniform1f(loc, value)
    elif isinstance(value, int):
        glUniform1i(loc, value)
    elif isinstance(value, (np.ndarray, list)):
        if len(value) == 2:
            glUniform2fv(loc, 1, value)
        if len(value) == 3:
            glUniform3fv(loc, 1, value)
        if len(value) == 4:
            glUniform4fv(loc, 1, value)
    elif isinstance(value, (Mat3, Mat4)):
        value._set_open_gl_uniform(loc)
    else:
        """
            If this happens the type was not supported, check your 
            argument types and either add a new else case above or 
            change the type.
        """
        assert False
#endregion
#--- Textures ---#
#region
def load_cubemap(base_name: str, srgb: bool) -> int:
    """
        Load images into a cubemap.

        Parameters:

            base_name: the filename stem. 
                eg. "data/cube_maps/Colosseum/"
            
            srgb: whether to store the texture in srgb colourspace
        
        Returns:

            Integer handle to the new cubemap
    """
    tex_id = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex_id)

    texSuffixFaceMap = {
        "posx" : GL_TEXTURE_CUBE_MAP_POSITIVE_X,
        "negx" : GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
        "posy" : GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
        "negy" : GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
        "posz" : GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
        "negz" : GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
    }
         
    try:
        for suf,faceId in texSuffixFaceMap.items():
            with Image.open(base_name%suf).transpose(Image.FLIP_TOP_BOTTOM) as im:
                mode = "RGBX" if im.mode == 'RGB' else "RGBA"
                data = im.tobytes("raw", mode, 0, -1)
                texture_format = GL_SRGB_ALPHA if srgb else GL_RGBA
                glTexImage2D(faceId, 0, texture_format, im.size[0], im.size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    except:
        print("WARNING: FAILED to load texture '%s'"%base_name)
        return -1

    glGenerateMipmap(GL_TEXTURE_CUBE_MAP)

    """
        Sets the type of mipmap interpolation to be used on 
        magnifying and minifying the active texture. For cube maps, 
        filtering across faces causes artifacts - so disable 
        filtering.
    """
    #glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    #glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # In case you want filtering anyway, try this below instead
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    # Use tri-linear mip map filtering
    #glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, 16)
    #or replace trilinear mipmap filtering with nicest anisotropic filtering
    return tex_id

def bind_texture(unit: int, tex_id: int, target = GL_TEXTURE_2D) -> None:
    """
        Bind the texture.

        Parameters:

            unit: the texture unit we're binding

            tex_id: handle to the texture being bound

            target: type of texture being bound
                eg. 2D, cubemap, 2D_array, ...
    """
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(target, tex_id)

def make_orthographic_projection(left, right, bottom, top, near, far) -> Mat4:
    return Mat4(np.array([
        [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
        [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
        [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0.0, 0.0, 0.0, 1.0]
    ]))
#endregion