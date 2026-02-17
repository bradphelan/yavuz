# Mesh Boolean Playground

Interactively explore boolean operations on 3D manifold meshes using VisPy.
This demo requires a VisPy backend (pyglet is included by default).

## Features
- Cube, sphere, cylinder, and tetrahedron operands
- Union, intersection, and subtraction booleans
- Keyboard controls for distance and shear
- Wireframe/solid toggle

## Usage
Run directly:
```bash
python mesh_booleans.py
```

Or launch from the main launcher:
```bash
cd ../..
python launcher.py
```

## Controls
- **A / B**: Cycle shapes for operand A/B
- **O**: Cycle boolean operation
- **Arrow Left/Right**: Adjust distance
- **Arrow Up/Down**: Adjust shear
- **W**: Toggle wireframe vs solid
- **R**: Reset defaults
- **N**: Randomize shapes and offsets
