# Cutter Tool Sweep

Visualize a spherical cutting tool sweeping through a 3D triangulated geometry. This demo shows how a tool moves along a path and the volume it removes from the stock geometry.

## Features

- **Interactive tool path** - Drag the tool widget to move it along the predefined sweep path
- **Sweep progress control** - Slider to control the tool's position along the path
- **Adjustable tool radius** - Change the tool diameter in real-time
- **Swept volume visualization** - See the accumulated volume removed by the tool
- **Visibility toggles** - Show/hide the swept volume and tool for clarity
- **3D mesh representation** - Work with triangulated geometry (cutter format)

## Usage

Run from the launcher or directly:
```bash
uv run python -m demos.cutter.cutter
```

## Controls

- **Sweep Progress**: Slider (0-100%) to move the tool along the path
- **Tool Radius**: Adjust the cutting tool diameter (0.1-1.0 units)
- **Show Swept Volume**: Toggle visibility of the accumulated swept volume
- **Show Tool**: Toggle visibility of the current tool position
- **Drag red sphere**: Interactively move the tool along the sweep path

## Algorithm

1. Defines a 3D sweep path (curved trajectory through space)
2. Creates a triangulated box geometry (stock/workpiece)
3. As the tool moves along the path, it sweeps out a volume
4. Real-time visualization shows the swept volume and current tool position
5. Useful for CAM (Computer-Aided Manufacturing) simulation and verification
