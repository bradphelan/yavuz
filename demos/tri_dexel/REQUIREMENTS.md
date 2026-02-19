# Tri-Dexel CNC Machining Simulation — Requirements

## Overview
Interactive demonstration of the Tri-dexel volumetric representation and Boolean
subtraction used in CNC machining simulation. The workpiece is represented as three
orthogonal sets of dexel rays (X, Y, Z). A tool swept volume is subtracted from the
workpiece to simulate material removal.

---

## R01 — Workpiece Initialisation
**GIVEN** the demo is launched
**WHEN** the scene is first rendered
**THEN** a rectangular block workpiece shall be displayed as a solid mesh derived from
a tri-dexel grid at a configurable resolution (default 50 x 50 x 50)

## R02 — Dexel Resolution Control
**GIVEN** the workpiece is displayed
**WHEN** the user adjusts the "Resolution" slider in the control panel
**THEN** the tri-dexel grid shall be regenerated at the new resolution and the
workpiece mesh shall update without flicker

## R03 — Tool Geometry Selection
**GIVEN** the control panel is visible
**WHEN** the user selects a tool type from a dropdown (Flat End Mill, Ball Nose, Bull Nose)
**THEN** a translucent tool mesh shall appear in the scene representing the selected
cutter geometry with a default diameter of 10 mm

## R04 — Tool Diameter Control
**GIVEN** a tool type has been selected
**WHEN** the user adjusts the "Tool Diameter" slider
**THEN** the tool mesh shall resize accordingly and any pending cut preview shall update

## R05 — Tool Position via 3D Widget
**GIVEN** the tool mesh is displayed
**WHEN** the user drags the tool position handle (sphere widget constrained to XY plane at current Z)
**THEN** the tool mesh shall follow the handle position in real time

## R06 — Tool Depth (Z) Control
**GIVEN** the tool is positioned on the workpiece
**WHEN** the user adjusts the "Depth of Cut" slider
**THEN** the tool mesh shall translate along Z to reflect the new cutting depth

## R07 — Single Cut Operation
**GIVEN** the tool is positioned at a desired location and depth
**WHEN** the user presses the "Cut" button
**THEN** the swept volume of the tool at its current pose shall be Boolean-subtracted
from the tri-dexel model and the workpiece mesh shall update to show the removed material

## R08 — Linear Toolpath Cut
**GIVEN** the user has defined a start point and an end point (via two sphere widgets or coordinate entry)
**WHEN** the user presses the "Cut Path" button
**THEN** the tool swept volume along the linear path shall be subtracted from the
workpiece, with the mesh updating to show the groove/slot

## R09 — Swept Volume Visualisation
**GIVEN** the user has positioned the tool and defined a path
**WHEN** the "Show Swept Volume" checkbox is enabled
**THEN** a translucent mesh of the tool swept volume shall be overlaid on the scene
before the cut is committed, serving as a preview

## R10 — Dexel Ray Visualisation
**GIVEN** the workpiece is displayed
**WHEN** the user toggles the "Show Dexels" checkbox and selects an axis (X, Y, or Z)
**THEN** the individual dexel segments for the chosen axis shall be rendered as coloured
line segments overlaid on the workpiece (X=red, Y=green, Z=blue)

## R11 — Cross-Section View
**GIVEN** the workpiece has undergone one or more cuts
**WHEN** the user enables "Cross Section" and adjusts the section plane slider
**THEN** a clipping plane shall slice the workpiece along the chosen axis, revealing
the internal dexel structure and cut geometry

## R12 — Undo Last Cut
**GIVEN** one or more cuts have been performed
**WHEN** the user presses the "Undo" button
**THEN** the last Boolean subtraction shall be reverted by restoring the previous
tri-dexel state, and the workpiece mesh shall update accordingly

## R13 — Reset Workpiece
**GIVEN** the workpiece has been modified by cuts
**WHEN** the user presses the "Reset" button
**THEN** the tri-dexel model shall be restored to the original uncut block and all
tool positions and paths shall be cleared

## R14 — Surface Mesh Reconstruction
**GIVEN** the tri-dexel model has been updated after a cut
**WHEN** the workpiece mesh is regenerated
**THEN** the dexel endpoints shall be converted to a watertight surface mesh using
marching cubes (or equivalent), with smooth shading and pastel colouring

## R15 — Performance Indicator
**GIVEN** a cut operation is triggered
**WHEN** the Boolean subtraction completes
**THEN** the elapsed computation time (in ms) shall be displayed in a status label
on the control panel, allowing the user to observe how resolution affects performance

---

## Non-Functional Requirements

- **NF01**: All mesh updates shall reuse actors (no remove/re-add) per project guidelines
- **NF02**: Controls shall use Qt dock widgets only (no PyVista widgets for UI)
- **NF03**: 3D lighting shall use three camera lights for depth perception
- **NF04**: Colours shall use pastel palette; no axis grid unless explicitly useful
- **NF05**: The demo shall follow the `DEMO_TEMPLATE.md` structure and register via `DEMO_MANIFEST`
