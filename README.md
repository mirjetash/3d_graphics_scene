# 3D Graphics Project

## Project utilizing Python and OpenGL 
Implementation of different techinques:
  * Modeling: geometrical representations, hierarchical modeling
  * Rendering: illumination, shading, textures
  * Animation: Keyframing

## The represented scene includes the following:
#### Modeling
  * Different objects represented using meshes and hierarchical modeling (Planets, Spaceship, etc,.)
#### Rendering
  * A Textured Skybox: six 2D images mapped to a cube representing the space with stars.
  * Modeling of light using the Phong Model.
  * The Blur effect (A Gaussian maske of size 9) applied to the texture fragment. shader
#### Animation
  * Keyframe animation of objects; specifying the positions of the animated objects on specific frames and then interpolating them.
  * Keyboard controls to move specific object in the scene.

 
# To run the project: 
```
python3 Project3D.py
```

**Note** Resources folder must be in the same directory as scene_implementation.py file.


## Keyboard controls:
* Space_bar: to restart time, therefore restart the animations.
* Key_B: hold this key to render the meshes with blur effect.
* Key W, A, S, D: to control the transform of the modeled figure.



