import os
import sys

import numpy as np
import pyvista as pv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from boolean_model_base import BooleanModelBase

class BooleanSphereExclusionModel(BooleanModelBase):
    def __init__(self, box_size=1.0, intensity=1, radius=0.15, seed=None):
        """
        Initialize Boolean Model of spherical exclusions.
        
        Args:
            box_size (float): Size of the cubic domain
            intensity (float): Intensity of the Poisson point process (lambda)
            radius (float): Radius of the spherical particles
            seed (int): Random seed for reproducibility
        """
        super().__init__(box_size=box_size, intensity=intensity, seed=seed)
        self.radius = radius

    def create_particles(self, points):
        """Create spherical particles at the given points."""
        particles = []
        for center in points:
            sphere = pv.Sphere(radius=self.radius, center=center)
            particles.append(sphere)
        return particles, points
    
    def calculate_porosity(self, points):
        """
        Calculate theoretical porosity of the structure.
        
        Returns:
            float: Theoretical porosity (void fraction)
        """
        # Volume of one sphere
        sphere_vol = (4/3) * np.pi * self.radius**3
        
        # Theoretical porosity (using Boolean model formula)
        porosity = np.exp(-self.intensity * sphere_vol)
        return porosity

def main():
    # Create Boolean model with specified parameters
    model1 = BooleanSphereExclusionModel(
        box_size=3.0,
        intensity=1.8,  # Poisson point process intensity
        radius=0.4,   # Sphere radius
        seed=42        # For reproducibility
    )

    model2 = BooleanSphereExclusionModel(
        box_size=3.0,
        intensity=60,  # Poisson point process intensity
        radius=0.1,   # Sphere radius
        seed=42        # For reproducibility
    )
    
    # Generate and visualize structure
    points, particles, porosity, _ = model1.generate_structure()

    # Generate and visualize structure
    # points2, particles2, porosity2, _ = model2.generate_structure()
    
    print(f"Structure generated with:")
    print(f"Number of particles: {len(points)}")
    print(f"Theoretical porosity: {porosity:.3f}")
    print(f"Volume fraction: {1 - porosity:.3f}")

if __name__ == "__main__":
    main() 
