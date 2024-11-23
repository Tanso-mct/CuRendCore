# Architecture
This file describes the overview of the entire project.

## Components
### CUDA
CUDA provides GPU calculate and CuRendCore uses CUDA in the rendering process. It makes more high performance in rendering.

### Direct3D
Direct3D is used to share buffer data for drawing on the screen between Direct2D and CUDA. Does not use rendering function。

### Direct2D
Direct2D is used to draw on the screen by using a buffer from CUDA and Direct3D. Windows GDI can't draw GPU buffer, but Direct2D can draw from a GPU buffer by using a Direct3D texture.

### Windows API
To create a Windows app need Windows API. CuRendCore uses it too.

## Data flow
I want to edit the GPU's buffer data in CUDA and use Direct2D to draw it on the screen without sending it to the CPU, so I create a Direct3D texture, map it to CUDA, process it, and then transfer that texture data to Direct2D. It used to draw on the screen. The buffer shared between Direct2D and CUDA is only the screen drawing buffer, so other vertex buffers etc are created and used on the CUDA side.

