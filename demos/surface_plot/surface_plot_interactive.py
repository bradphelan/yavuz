"""
Interactive 3D surface plot with parameter controls.
Demonstrates numpy computations and matplotlib 3D plotting with GUI widgets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D


class InteractiveSurface:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        
        # Create 3D axis
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(bottom=0.25)
        
        # Initial parameters
        self.frequency = 1.0
        self.amplitude = 1.0
        self.phase = 0.0
        
        # Create initial surface
        self.X, self.Y, self.Z = self.compute_surface()
        self.surf = self.ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.8)
        
        # Set labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Interactive 3D Surface: Z = A*sin(f*√(X²+Y²) + φ)')
        
        # Create sliders
        self.setup_controls()
        
    def compute_surface(self):
        """Compute the surface based on current parameters."""
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = self.amplitude * np.sin(self.frequency * R + self.phase)
        return X, Y, Z
    
    def setup_controls(self):
        """Setup GUI controls (sliders and text boxes)."""
        # Frequency slider
        ax_freq = plt.axes([0.2, 0.15, 0.6, 0.03])
        self.slider_freq = Slider(
            ax_freq, 'Frequency', 0.1, 5.0, 
            valinit=self.frequency, valstep=0.1
        )
        self.slider_freq.on_changed(self.update_frequency)
        
        # Amplitude slider
        ax_amp = plt.axes([0.2, 0.10, 0.6, 0.03])
        self.slider_amp = Slider(
            ax_amp, 'Amplitude', 0.1, 3.0, 
            valinit=self.amplitude, valstep=0.1
        )
        self.slider_amp.on_changed(self.update_amplitude)
        
        # Phase slider
        ax_phase = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider_phase = Slider(
            ax_phase, 'Phase', 0.0, 2*np.pi, 
            valinit=self.phase, valstep=0.1
        )
        self.slider_phase.on_changed(self.update_phase)
    
    def update_frequency(self, val):
        """Update frequency parameter."""
        self.frequency = val
        self.update_plot()
    
    def update_amplitude(self, val):
        """Update amplitude parameter."""
        self.amplitude = val
        self.update_plot()
    
    def update_phase(self, val):
        """Update phase parameter."""
        self.phase = val
        self.update_plot()
    
    def update_plot(self):
        """Redraw the surface with updated parameters."""
        self.ax.clear()
        self.X, self.Y, self.Z = self.compute_surface()
        self.surf = self.ax.plot_surface(self.X, self.Y, self.Z, cmap='viridis', alpha=0.8)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Interactive 3D Surface: Z = A*sin(f*√(X²+Y²) + φ)')
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive plot."""
        plt.show()


def main():
    """Run the interactive surface plot."""
    app = InteractiveSurface()
    app.show()


if __name__ == "__main__":
    main()
