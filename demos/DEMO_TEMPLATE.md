# Demo Template

This template shows how to create a new demo for the Yavuz project.

## Steps to Create a New Demo

1. **Create a new folder** in the `demos/` directory:
   ```bash
   mkdir demos/my_new_demo
   ```

2. **Create your main script** (e.g., `my_new_demo.py`):
   ```python
   """
   My New Demo
   Brief description of what this demo does.
   """

    import numpy as np
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter


   class MyNewDemo:
       def __init__(self):
           self.plotter = BackgroundPlotter(
               window_size=(1000, 800),
               title="My New Demo",
           )
           self.plotter.set_background("white")
           self.plotter.add_axes()
           # Setup your visualization

       def setup_controls(self):
           """Setup GUI controls."""
           # Add sliders, buttons, etc.
           pass

       def update_plot(self):
           """Update the visualization."""
           # Your update logic here
           pass

       def show(self):
           """Display the demo."""
           self.plotter.show()
           self.plotter.app.exec_()


   def main():
       """Run the demo."""
       demo = MyNewDemo()
       demo.show()


   if __name__ == "__main__":
       main()
   ```

3. **Create a README.md** in your demo folder:
   ```markdown
   # My New Demo

   Brief description of the demo.

   ## Features
   - Feature 1
   - Feature 2

   ## Usage
   ```bash
   python my_new_demo.py
   ```

   ## Controls
   - Control 1: Description
   - Control 2: Description
   ```

4. **Register your demo** in `launcher.py`:

   Open `launcher.py` and add your demo to the `demo_info` dictionary in the `discover_demos()` method:

   ```python
   "my_new_demo": {
       "name": "My New Demo Display Name",
       "description": "Full description of what this demo does.\n\n"
                    "Features:\n"
                    "• Feature 1\n"
                    "• Feature 2\n"
                    "• Feature 3",
       "script": "my_new_demo.py"
   }
   ```

5. **Test your demo**:
   ```bash
   # Test directly
   python demos/my_new_demo/my_new_demo.py

   # Test from launcher
   python launcher.py
   ```

## Best Practices

1. **Use descriptive names** for functions and classes
2. **Add docstrings** to explain functionality
3. **Include error handling** for user inputs
4. **Make controls intuitive** with clear labels
5. **Add comments** for complex algorithms
6. **Test edge cases** before adding to launcher

## Common GUI Widgets

### PyVista Widgets
```python
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

plotter = BackgroundPlotter(window_size=(1000, 800), title="Example")
plotter.set_background("white")
plotter.add_axes()

# Slider
plotter.add_slider_widget(
    callback_function,
    [min_val, max_val],
    value=initial,
    title="Label",
    pointa=(0.1, 0.1),
    pointb=(0.4, 0.1),
)

# Checkbox button
plotter.add_checkbox_button_widget(
    callback_function,
    value=True,
    position=(10, 10),
    size=30,
)

# Text label
plotter.add_text("Example", position="upper_left", font_size=12)

# 3D mesh
mesh = pv.Sphere()
plotter.add_mesh(mesh, color="skyblue", smooth_shading=True)
```

## Example Callback Functions

```python
def update_parameter(self, val):
    """Update parameter from slider."""
    self.parameter = val
    self.update_plot()

def on_toggle(self, state):
    """Handle checkbox toggle."""
    self.enabled = bool(state)
    self.update_plot()
```

## Tips for Algorithm Visualization

1. **Start simple** - Get basic visualization working first
2. **Add interactivity gradually** - One control at a time
3. **Use colors effectively** - Highlight important elements
4. **Provide feedback** - Show what the algorithm is doing
5. **Control speed** - Allow users to slow down fast algorithms
6. **Add descriptions** - Explain what's happening

## Resources

- [PyVista Documentation](https://docs.pyvista.org/)
- [PyVistaQt Widgets](https://github.com/pyvista/pyvistaqt)
- [NumPy Documentation](https://numpy.org/doc/)
