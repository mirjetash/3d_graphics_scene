#!/usr/bin/env python3


# Python built-in modules
import os                           # os function, i.e. checking file status
from itertools import cycle
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np                  # all matrix manipulations & OpenGL args
import pyassimp                     # 3D resource loader
import pyassimp.errors              # Assimp error management + exceptions
from PIL import Image               # load images for textures


from bisect import bisect_left      # search sorted keyframe lists
from transform import Trackball, identity, translate, rotate, scale, lerp, vec, quaternion_slerp, quaternion_matrix, quaternion, quaternion_from_euler


# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object


class VertexArray:
    """ helper class to create and self destroy OpenGL vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            Attributes should be list of arrays with one row per vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load buffer per vertex attribute (in list with index = shader layout)
        for loc, data in enumerate(attributes):
            if data is not None:
                # bind a new vbo, upload its data to GPU, declare size and type
                self.buffers += [GL.glGenBuffers(1)]
                data = np.array(data, np.float32, copy=False)  # ensure format
                nb_primitives, size = data.shape
                GL.glEnableVertexAttribArray(loc)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
                GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
                GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)  # good format
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def execute(self, primitive):
        """ draw a vertex array, either as direct array or indexed array """
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)


class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


class TextureSkyBox:
    """ Creation of the Skybox Texture """

    def __init__(self, files, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            texture_target = [GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
            for i in range(len(files)):
                tex = np.array(Image.open(files[i]))
                GL.glTexImage2D(texture_target[i], 0, GL.GL_RGBA, tex.shape[1], tex.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
            GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE); 
            GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            print(message % (files[1], tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)


# -------------- SkyBox Texture ----------------------------------
TEXTURE_VERT = """#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 view;
out vec3 fragTexCoord;

void main() {
    fragTexCoord = position;
    vec4 pos = projection * view * vec4(position, 1.0);
    gl_Position = pos.xyww;
}"""


TEXTURE_FRAG = """#version 330 core
uniform samplerCube cubeMap;
in vec3 fragTexCoord;
out vec4 outColor;

void main() {
    outColor = texture(cubeMap, fragTexCoord);
}"""


class Skybox_Textured:
    """ Textured Skybox """

    def __init__(self, files):
        self.shader = Shader(TEXTURE_VERT, TEXTURE_FRAG)

        # triangle and face buffers        
        vertices = np.array(((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1), 
                             (-1,  -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)), np.float32)

        faces = np.array(((4,5,6), (4,6,7), (0,4,7), (0,7,3), (5,1,2), (5,2,6), (1, 0, 3), (1, 3, 2), (7,6,2), (7,2,3), (0,1,5), (0,5,4)), np.uint32)
        
        self.vertex_array = VertexArray([vertices], faces)

        self.wrap_mode, self.filter_mode = GL.GL_REPEAT, (GL.GL_NEAREST, GL.GL_NEAREST)
        self.files = files
        self.texture = TextureSkyBox(files, self.wrap_mode, *self.filter_mode)

    def draw(self, projection, view, model, win=None, **_kwargs):
        # GL.glDepthMask(GL.GL_FALSE)
        GL.glDepthFunc(GL.GL_LEQUAL) # change depth function so depth test passes when values are equal to depth buffer's content
        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'view')
        mul = np.array([[1,1,1,0],[1,1,1,0],[1,1,1,0],[1,1,1,1]])
        no_translation = view * mul  # to remove translation of the camera
        
        # rotate the camera around y axis over time stops when time>12 
        # when time is reset to 0 (with the Space bar) it starts rotating again
        time = glfw.get_time()
        if time<=12:
            rotation_matrix = rotate(axis=(0,1,0), angle=30*time)
            no_translation = no_translation @ rotation_matrix

        GL.glUniformMatrix4fv(loc, 1, True, no_translation)
        
        loc1 = GL.glGetUniformLocation(self.shader.glid, 'projection')
        GL.glUniformMatrix4fv(loc1, 1, True, projection)

        # texture access setups
        loc2 = GL.glGetUniformLocation(self.shader.glid, 'cubeMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture.glid)
        GL.glUniform1i(loc2, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)
        GL.glDepthFunc(GL.GL_LESS); # set depth function back to default
        # GL.glDepthMask(GL.GL_TRUE);

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
        GL.glUseProgram(0)

# -------------- Color and Vertex Shaders ----------------------------------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragNormal;
out vec3 fragView;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragNormal = transpose(inverse(mat3(view * model))) * normal;
    fragView = normalize((view * model * vec4(position, 1)).xyz);
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragNormal;
in vec3 fragView;
vec3 lightDir;
vec3 v;
vec3 n;
vec3 r;
vec3 ka;
vec3 kd;
vec3 ks;

vec3 lightning;
out vec4 outColor;

void main() {
    lightDir = normalize(vec3(0, 0.5, 0.6));   // light direction
    v = normalize(fragView);
    n = normalize(fragNormal);
    r = reflect(lightDir, n);

    ka = vec3(0.42, 0.23, 0);
    kd = vec3(1, 0.5, 0.5);
    ks = vec3(1, 0, 0);

    lightning = ka + kd*max(dot(n, lightDir), 0) + ks*pow(max(dot(r,v),0),32);

    outColor = vec4(lightning, 1);
}"""


# -------------- Object classes -----------------------------------------
class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = model @ self.transform
        for child in self.children:
            child.draw(projection, view, model, **param)


class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader, **param):

        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)
        
        translation_matrix = translate(x=1.5, y=0.0, z = 1.5)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        self.vertex_array.execute(GL.GL_TRIANGLES)


class Surface(ColorMesh):
    """ Modelling of a non-flat surface """

    def __init__(self):
        # random positions in a surface generated then the height values were changed to make it non-flat
        position = np.array(((-0.5, -0.3, -0.5),(0.5, -0.3, -0.5), 
                             (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5), 
                             (0.7, -0.2, -0.5), (0.9, -0.2, -0.5), 
                             (0.9, -0.5, 0.2), (1, -0.3, -0.5), 
                             (-0.7, -0.2, 0) , (-0.8, -0.5, 0.7), 
                             (-1.6, -0.4, -0.7), (-1.5, -0.3,0.4),
                             (0.1, -0.4, 0.7), (-0.1, -0.3, 0.9),
                             (0.3, -0.5, 1.1), (0.6, -0.6, 0.8),
                             (0.8, -0.3, 0.5), (-0.4,-0.3,0.9),
                             (-0.4, -0.3, 1), (-0.1,-0.3, 1.1),
                             (0.1, -0.5, 1.1), (0.5, -0.4, 1.4), 
                             (0.8, -0.5, 1.3), (0.9,-0.3,0.9),
                             (0, -0.4, 1.3), (-0.7,-0.5,1.4),
                             (-1.1, -0.4, 0.9), (0.1, -0.25,-1),
                             (1.8,-0.7,1), (0.3,-0.7,2.5),
                             (-2.2, -0.7,0.6), (0.5, -0.8, 0.5)  
                             ), 'f')       
        
        index = np.array(((3,2,1), (3,1,0), (4,1,2), (5,4,6), (6,4,2),(7,5,6), (3,0,8), (8,9,3), (8,10,9),(8,0,10), (9,10,11),
                          (2,3,12),(3,13,12),(13,14,12), (2,12,15), (15,12,14), (15,16,2), (6,2,16),(3,9,17), (3, 17,13), (18,13,17),
                          (18,19,13), (13,19,20), (13,20,14), (15,23,16), (16,23,6), (15,22,23),(15,21,22), (15,14,21), (14,20,21),
                          (24,21,20), (19,24,20), (25,24,19), (25,19,18), (9,25,18), (9,18,17), (25,9,26), (9,11,26), (0,1,27),
                          (10,0,27), (1,4,27), (4,5,27), (6,23,28), (7,6,28), (22,28,23), (29,22,21), (21,24,29), (24,25,29),
                          (11,30,26), (30,25,26), (11,10,30), (29,28,22), (30,29,25), (29,31,28), (30,31,29), (27,31,30), (28,31,27)), np.uint32)

        # calculate the normals for each triangle in the surface
        normals_list=[]
        for p1,p2,p3 in index:
            v = position[p2] - position[p1]
            w = position[p3] - position[p1]
            nx = v[1]*w[2] - v[2]*w[1]
            ny = v[2]*w[2] - v[0]*w[2]
            nz = v[0]*w[1] - v[1]*w[0]
            normals_list.append([nx, ny, nz])

        normals = np.array(normals_list) 
        super().__init__([position, normals], index)


class Shape(Node):
    """ Load object from file <filename.obj> with not texture(just shape) """
    def __init__(self, filename):
        super().__init__()
        self.add(*load(filename))  # just load the cylinder from file


class PhongMesh(Node):
    "Phong Mesh object"
    def __init__(self, filename):
        super().__init__()
        self.add(*load_textured(filename))
  

# -------------- Texture mesh class ----------------------------------
TEXTURE_VERT1 = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 coordsTex;
layout(location = 2) in vec3 normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragNormal;
out vec3 fragView;

out vec2 fragTexCoord;
void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragNormal = transpose(inverse(mat3(view * model))) * normal;
    fragView = normalize((view * model * vec4(position, 1)).xyz);
    fragTexCoord = coordsTex;
}"""

TEXTURE_FRAG1 = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;

in vec3 fragNormal;
in vec3 fragView;
vec3 lightDir;
vec3 v;
vec3 n;
vec3 r;
vec3 ka;
vec3 kd;
vec3 ks;

uniform float step; // to control the blur if 1 there is no blur

// Gaussian mask of size 9 used for bluring effect
float[9] gaussian = float[9](
    0.0162162162, 0.0540540541,
    0.1216216216, 0.1945945946,
    0.2270270270, 0.1945945946,
    0.1216216216, 0.0540540541,
    0.0162162162 );

vec4 lightning;
out vec4 outColor;
void main() {
    lightDir = normalize(vec3(0, 0.5, 0.6));   // light direction
    v = normalize(fragView);
    n = normalize(fragNormal);
    r = reflect(lightDir, n);

    ka = vec3(0.6, 0.6, 0.6);
    kd = vec3(1, 0.5, 0.5);
    ks = vec3(1, 0, 0);

    lightning = vec4(ka + kd*max(dot(n, lightDir), 0) + ks*pow(max(dot(r,v),0),32), 10);

    vec4 sum = vec4(0.0);
    float blur = 2.0; // how many pixels to take into account

    if (step ==1){  // 
        outColor = texture(diffuseMap, fragTexCoord) * lightning;
    }else{
        sum += texture(diffuseMap, vec2(fragTexCoord.x - 4.0*blur*step, fragTexCoord.y - 4.0*blur*step)) * gaussian[0];
        sum += texture(diffuseMap, vec2(fragTexCoord.x - 3.0*blur*step, fragTexCoord.y - 3.0*blur*step)) * gaussian[1];
        sum += texture(diffuseMap, vec2(fragTexCoord.x - 2.0*blur*step, fragTexCoord.y - 2.0*blur*step)) * gaussian[2];
        sum += texture(diffuseMap, vec2(fragTexCoord.x - 1.0*blur*step, fragTexCoord.y - 1.0*blur*step)) * gaussian[3];
        sum += texture(diffuseMap, vec2(fragTexCoord.x, fragTexCoord.y)) * gaussian[4];
        sum += texture(diffuseMap, vec2(fragTexCoord.x + 1.0*blur*step, fragTexCoord.y + 1.0*blur*step)) * gaussian[5];
        sum += texture(diffuseMap, vec2(fragTexCoord.x + 2.0*blur*step, fragTexCoord.y + 2.0*blur*step)) * gaussian[6];
        sum += texture(diffuseMap, vec2(fragTexCoord.x + 3.0*blur*step, fragTexCoord.y + 3.0*blur*step)) * gaussian[7];
        sum += texture(diffuseMap, vec2(fragTexCoord.x + 4.0*blur*step, fragTexCoord.y + 4.0*blur*step)) * gaussian[8];
        outColor = vec4(sum.rgb, 1.0) * lightning;
    }

}"""

# -------------- 3D textured mesh loader ---------------------------------------
class TexturedMesh:


    def __init__(self, texture, attributes, indexes):
        self.shader = Shader(TEXTURE_VERT1, TEXTURE_FRAG1)
        self.vertex_array = VertexArray(attributes, indexes)
        self.texture = Texture(texture);

    def draw(self, projection, view, model, win=None, **_kwargs):
        GL.glUseProgram(self.shader.glid)
        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'model')
        GL.glUniformMatrix4fv(loc, 1, True, model)

        loc = GL.glGetUniformLocation(self.shader.glid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)

        loc = GL.glGetUniformLocation(self.shader.glid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)

        step = 1;
        if glfw.get_key(win, glfw.KEY_B) == glfw.PRESS: # to blur the objects
            step = 0.0008

        loc = GL.glGetUniformLocation(self.shader.glid, 'step')
        GL.glUniform1f(loc, step)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.execute(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


def load_textured(file):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    path = os.path.join('.', '') if path == '' else path
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture

        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # print(mesh.normals)
        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv, mesh.normals], mesh.faces))

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes


# -------------- Interpolator class -----------------------------------------
class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if time <= self.times[0]:
            return self.values[0]
        if time >= self.times[-1]:
            return self.values[-1]

        # 2. search for closest index entry in self.times, using bisect_left function
        index_entry = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        fraction = (time - self.times[index_entry - 1]) / (self.times[index_entry] - self.times[index_entry-1])
        return self.interpolate(self.values[index_entry - 1], self.values[index_entry], fraction)

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        self.translate_keys = KeyFrames(translate_keys)
        self.rotate_keys = KeyFrames(rotate_keys, quaternion_slerp)
        self.scale_keys = KeyFrames(scale_keys)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        translate_mat = translate(self.translate_keys.value(time))
        rotate_mat = quaternion_matrix(self.rotate_keys.value(time))
        scale_mat = scale(self.scale_keys.value(time))
        return translate_mat @ rotate_mat @ scale_mat

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)



# -------------- Object Control Classes ---------------------------------------
class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])


class RotationControlNode(Node):
    def __init__(self, key_up, key_down, axis, angle=0, **param):
        super().__init__(**param)  
        self.angle, self.axis = angle, axis
        self.key_up, self.key_down = key_up, key_down

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.angle += 2 * int(glfw.get_key(win, self.key_up) == glfw.PRESS)
        self.angle -= 2 * int(glfw.get_key(win, self.key_down) == glfw.PRESS)
        self.transform = rotate(self.axis, self.angle)

        super().draw(projection, view, model, win=win, **param)


class RotatePlanet(Node):
    def __init__(self, axis, speed, angle=0, **param):
        super().__init__(**param)
        self.axis = axis
        self.speed = speed
        self.angle = angle

    def draw(self, projection, view, model, win=None,**param):    
        self.angle += self.speed
        if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS: # set angle to 0 to restart the animation
            self.angle = 0
        if self.angle < 361: # stop on a full rotation
            self.transform = rotate(self.axis, self.angle)

        super().draw(projection, view, model, win=win, **param)


class TranslationControlNode(Node):
    def __init__(self, key_w, key_s, key_a, key_d ,x_pos=0, z_pos=0, angle=0, **param):
        super().__init__(**param) 
        self.angle = angle
        self.key_forward, self.key_backward = key_w, key_s
        self.key_right, self.key_left = key_d, key_a
        self.z_pos, self.x_pos = z_pos, x_pos
        # self.key_rotate_l, self.key_rotate_r = key_q, key_e

    def draw(self, projection, view, model, win=None, **param):
        assert win is not None
        self.z_pos -= 0.005 * int(glfw.get_key(win, self.key_forward) == glfw.PRESS)
        self.z_pos += 0.005 * int(glfw.get_key(win, self.key_backward) == glfw.PRESS)
        self.x_pos -= 0.005 * int(glfw.get_key(win, self.key_left) == glfw.PRESS)
        self.x_pos += 0.005 * int(glfw.get_key(win, self.key_right) == glfw.PRESS)
        
        self.transform = translate(x=self.x_pos, y=0, z=self.z_pos)

        super().draw(projection, view, model, win=win, **param)


# ------------  Viewer class & window management ------------------------------
class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=1540, height=880):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)         
        GL.glEnable(GL.GL_CULL_FACE)         

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        # initially empty list of object to draw
        self.drawables = []

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])
        
    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(),
                              color_shader=self.color_shader, win=self.win)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_SPACE:  # space resets the time
                glfw.set_time(0)
            if key == glfw.KEY_M:   # change view mode with M
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))


# ------------- Drawing -------------
def add_hierarchical_model():
    # ----------SHAPES---------------
    sphere_mesh = Shape('resources/sphere.obj')
    head_shape = Node(transform=translate(x=1, y=-0.2, z=0.03)@scale(0.2, 0.18, 0.1))
    head_shape.add(sphere_mesh)

    cylinder = Shape('resources/cylinder.obj')
    leg_shape1 = Node(transform=translate(x=1, y=-0.25, z=0.05)@scale(0.01, 0.05, 0.01))
    leg_shape1.add(cylinder)

    leg_shape2 = Node(transform=translate(x=0.92, y=-0.25, z=0.05)@scale(0.01, 0.05, 0.01))
    leg_shape2.add(cylinder)

    foot_shape1 = Node(transform=translate(x=0.92, y=0.025, z=0.291)@scale(0.009, 0.025, 0.01))
    foot_shape1.add(cylinder)
    
    foot_shape2 = Node(transform=translate(x=1, y=0.025, z=0.291)@scale(0.009, 0.025, 0.01))
    foot_shape2.add(cylinder)

    # ------- Node Connections---------
    transform_foot2 = Node(transform=rotate(axis=(1,0,0),angle=89))
    transform_foot2.add(foot_shape2)

    transform_foot1 = Node(transform=rotate(axis=(1,0,0),angle=89))
    transform_foot1.add(foot_shape1)

    transform_leg2 = Node(transform=translate(x=0,y=0,z=0))
    transform_leg2.add(leg_shape2, transform_foot2)
    
    transform_leg1 = Node(transform=translate(x=0,y=0,z=0))
    transform_leg1.add(leg_shape1, transform_foot1)

    transform_body = Node(transform=translate(x=-0.4,y=0,z=0.8))
    transform_body.add(transform_leg1, transform_leg2, head_shape)

    trans_node = TranslationControlNode(glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D)
    trans_node.add(transform_body)

    return trans_node
def add_hierarchical_banner():
    # ----------SHAPES---------------
    cube = Shape('resources/cube.obj')

    left_h = Node(transform=translate(x=0.5, y=0, z=1.5)@scale(0.02, 0.2, 0.02))
    left_h.add(cube)

    right_h = Node(transform=translate(x=0.78, y=0, z=1.5)@scale(0.02, 0.2, 0.02))
    right_h.add(cube)

    main_branch = Node(transform=translate(x=0.64, y=0.05, z=1.5)@scale(0.3, 0.1, 0.02))
    main_branch.add(cube)

    transform_left_h = Node()
    transform_left_h.add(left_h)
    transform_right_h = Node()
    transform_right_h.add(right_h)

    transform_main_branch = Node(transform=translate(x=-0.4, y=-0.32, z=-0.2)@rotate(axis=(0,1,0),angle=30))
    transform_main_branch.add(main_branch, transform_left_h, transform_right_h)
    return transform_main_branch

def add_ufo_ship():
    ufo_mesh = PhongMesh('resources/ufo/ufo.obj')
    ufo_base = Node(transform=translate(x=-0.6,y=0,z=0)@scale(x=0.04))   
    ufo_base.add(ufo_mesh)
 
    translate_keys = {0: vec(-8, 0.8, -10), 
                      1: vec(-7, 0.7, -9), 
                      2: vec(-6, 0.6, -8), 
                      3: vec(-5, 0.5, -7), 
                      4: vec(-4, 0.45, -6), 
                      5: vec(-3, 0.38, -5), 
                      6: vec(-2, 0.35, -4), 
                      7: vec(-1, 0.25, -3), 
                      8: vec(-0.3, 0.15, -2.5),
                      9: vec(-0.1, -0.1, -1),
                      10: vec(0.03, -0.25, -0.5),
                      11: vec(0.09, -0.44, 0.1)
                      }

    rotate_keys = {0: quaternion_from_euler(0, 0, 20), 
                   1: quaternion_from_euler(0, 0, 20), 
                   2: quaternion_from_euler(0, 0, 25),
                   3: quaternion_from_euler(0, 0, 28), 
                   4: quaternion_from_euler(0, 0, 33),
                   5: quaternion_from_euler(0, 0, 35),
                   6: quaternion_from_euler(0, 0, 37),
                   7: quaternion_from_euler(0, 0, 30),
                   8: quaternion_from_euler(0, 0, 20),
                   9: quaternion_from_euler(0, 0, 15),
                   10: quaternion_from_euler(0, 0, 25),
                   11: quaternion_from_euler(0, 0, 10)
                   }

    scale_keys = {0: 0.2, 
                  1: 0.25, 
                  2: 0.3, 
                  3: 0.3, 
                  4: 0.3, 
                  5: 0.3, 
                  6: 0.3,
                  7: 0.3,
                  8: 0.3,
                  9: 0.3,
                  10: 0.3,
                  11: 0.3
                  }

   
    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    keynode.add(ufo_base)
    return keynode
def add_asteroid():
    a_mesh = PhongMesh('resources/Asteroid/10464_Asteroid_v1_Iterations-2.obj')
    asteroid_base = Node(transform=translate(x=2.2,y=4,z=-10)@scale(x=0.02))
    asteroid_base.add(a_mesh)

    translate_keys = {0: vec(2, 5, -11), 
                      2: vec(1.5, 3.5, -12), 
                      4: vec(1, 2, -13), 
                      6: vec(0.5, 0.5, -14), 
                      8: vec(0, -1, -15),
                      10: vec(-0.2, -2.5, -16), 
                      }

    rotate_keys = {0: quaternion_from_euler(30, 0, 150), 
                   2: quaternion_from_euler(45, 0, 200), 
                   4: quaternion_from_euler(50, 0, 250), 
                   6: quaternion_from_euler(55, 0, 300), 
                   8: quaternion_from_euler(60, 0, 350),
                   10: quaternion_from_euler(65, 0, 400)
                   }

    scale_keys = {0: 0.1, 
                  2: 0.08, 
                  4: 0.06, 
                  6: 0.04, 
                  8: 0.01,
                  10: 0.001
                  }
   
    keynode = KeyFrameControlNode(translate_keys, rotate_keys, scale_keys)
    keynode.add(asteroid_base)
    return keynode


# -------------- main program and scene setup --------------------------------
def main():
    viewer = Viewer()

    # SkyBox
    files = ['resources/skybox/sky_right1.png', 'resources/skybox/sky_left2.png', 'resources/skybox/sky_top3.png', 'resources/skybox/sky_bottom4.png', 'resources/skybox/sky_back6.png', 'resources/skybox/sky_front5.png']
    txt1 = Skybox_Textured(files)
    viewer.add(txt1)

    surface = Surface()
    surface_base = Node(transform=scale(x=0.7))   
    surface_base.add(surface)
    viewer.add(surface_base)

    viewer.add(add_hierarchical_model())
    viewer.add(add_hierarchical_banner())
    viewer.add(add_ufo_ship())
    viewer.add(add_asteroid())

    earth_mesh = PhongMesh('resources/Earth1/earth.obj')
    earth_base = Node(transform=translate(x=4,y=1.5,z=-7)@scale(x=0.0033))   
    earth_base.add(earth_mesh)
    rotate_earth = RotatePlanet(axis=(0, 1, 0), speed=0.8)
    rotate_earth.add(earth_base)
    viewer.add(rotate_earth)

    jupiter_mesh = PhongMesh('resources/Jupiter/Jupiter.obj')
    jupiter_base = Node(transform=translate(x=-0.6,y=2,z=-10)@rotate(axis=(1,0,0), angle=90)@scale(x=0.001))   
    jupiter_base.add(jupiter_mesh)
    rotate_jupiter = RotatePlanet(axis=(0, 1, 0), speed=0.6)
    rotate_jupiter.add(jupiter_base)
    viewer.add(rotate_jupiter)
    
    sun_mesh = PhongMesh('resources/Sun/Sun.obj')
    sun_base = Node(transform=translate(x=-4,y=2,z=-9)@scale(x=0.0013))   
    sun_base.add(sun_mesh)
    rotate_sun = RotatePlanet(axis=(0, 1, 0), speed=0.4)
    rotate_sun.add(sun_base)
    viewer.add(rotate_sun)

    # Rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts



