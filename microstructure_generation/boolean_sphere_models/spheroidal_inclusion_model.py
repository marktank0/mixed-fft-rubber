import os
import sys

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from boolean_model_base import BooleanModelBase

class BooleanSpheroidalInclusionModel(BooleanModelBase):
    def __init__(self, box_size=1.0, intensity=50, min_R1=1.0, max_R1=4.0, seed=None):
        """
        Initialize Boolean Model of spheroidal exclusions with linear R1-R2 relationship
        and R3=R1 for circular cross-section in R3 plane.
        
        Args:
            box_size (float): Size of the cubic domain
            intensity (float): Intensity of the Poisson point process (lambda)
            min_R1 (float): Minimum value for R1 (shorter radius)
            max_R1 (float): Maximum value for R1 (shorter radius)
            seed (int): Random seed for reproducibility
        """
        super().__init__(box_size=box_size, intensity=intensity, seed=seed)
        self.min_R1 = min_R1
        self.max_R1 = max_R1
        
        # Linear relationship coefficients: R2 = m*R1 + b
        scale = 0.025 *0.6
        self.m = 2.5535 * scale  # slope
        self.b = 12.0916 * scale # intercept
            
    def calculate_R2(self, R1):
        """Calculate R2 based on the linear relationship."""
        return self.m * R1 + self.b
    
    def generate_particle_dimensions(self):
        """
        Generate R1, R2, and R3 dimensions where:
        - R2 follows linear relationship with R1
        - R3 equals R1 for circular cross-section
        """
        R1 = np.random.uniform(self.min_R1, self.max_R1)
        R2 = self.calculate_R2(R1)
        R3 = R1  # Make it circular in R3 plane
        return R1, R2, R3
        
    def generate_points_and_dimensions(self):
        """Generate Poisson point process, random orientations, and particle dimensions."""
        # Generate points
        points = self.generate_points()
        
        # Generate random orientations using uniform rotation matrices
        orientations = [Rotation.random() for _ in range(len(points))]
        
        # Generate dimensions for each particle
        dimensions = [self.generate_particle_dimensions() for _ in range(len(points))]
        
        return points, orientations, dimensions
    
    def create_spheroid_mesh(self, center, rotation, dimensions):
        """Create a spheroid mesh with given center, orientation, and dimensions."""
        R1, R2, R3 = dimensions
        # Create base sphere
        sphere = pv.Sphere(radius=1.0, phi_resolution=20, theta_resolution=20)
        
        # Get points and scale them to create ellipsoid
        points = np.array(sphere.points)
        scaling = np.array([R1, R2, R3])  # Scale to actual dimensions
        points = points * scaling
        
        # Apply rotation
        points = rotation.apply(points)
        
        # Apply translation
        points = points + center
        
        # Create mesh
        spheroid = pv.PolyData(points, sphere.faces)
        return spheroid

    def create_particles(self, points, orientations=None, dimensions=None):
        """Create spheroidal particles at the given points.

        If ``orientations``/``dimensions`` are not supplied they are generated to
        match ``points`` (one per point). Previously this method called
        ``generate_points_and_dimensions()`` internally, which drew a *fresh*
        Poisson number of points and then relied on ``zip`` truncation -- so the
        particle count silently mismatched ``points`` and some centers received
        no spheroid. Callers that later test containment against these meshes must
        pass the same orientations/dimensions used to build them.
        """
        if orientations is None:
            orientations = [Rotation.random() for _ in range(len(points))]
        if dimensions is None:
            dimensions = [self.generate_particle_dimensions() for _ in range(len(points))]

        # Create particles
        particles = []
        for center, rotation, dims in zip(points, orientations, dimensions):
            spheroid = self.create_spheroid_mesh(center, rotation, dims)
            particles.append(spheroid)
        return particles, points
    
    def calculate_porosity(self, points):
        """
        Calculate theoretical porosity of the structure.
        
        Returns:
            float: Theoretical porosity (void fraction)
        """
        # Generate dimensions for porosity calculation
        _, _, dimensions = self.generate_points_and_dimensions()
        
        # Calculate total volume
        total_volume = 0
        for R1, R2, R3 in dimensions:
            # Volume of ellipsoid: V = (4/3)πR1·R2·R3
            volume = (4/3) * np.pi * R1 * R2 * R3
            total_volume += volume
        avg_volume = total_volume / len(dimensions) if dimensions else 0
        
        # Theoretical porosity (using Boolean model formula)
        porosity = np.exp(-self.intensity * avg_volume)
        return porosity

    def generate_structure(self, show_plot=True):
        """
        Generate and visualize the Boolean model structure.
        
        Args:
            show_plot (bool): Whether to show the visualization
        
        Returns:
            tuple: (points, particles, porosity, plotter)
        """
        # Generate points, orientations, and dimensions
        points, orientations, dimensions = self.generate_points_and_dimensions()

        # Calculate porosity
        porosity = self.calculate_porosity(points)

        # Create particles from the exact points/orientations/dimensions above
        particles, points = self.create_particles(points, orientations, dimensions)
        
        # Create visualization
        plotter = self.visualize(particles)
        
        if show_plot:
            plotter.show()
        
        return points, particles, porosity, plotter

def main():
    # Create Boolean model with specified parameters
    model = BooleanSpheroidalInclusionModel(
        box_size=3.0,    # Box size
        intensity=2.5,  # Intensity for particle generation
        min_R1=0.5,    # Minimum R1 value
        max_R1=0.5,     # Maximum R1 value
        seed=44         # For reproducibility
    )
    
    # Generate and visualize structure
    points, particles, porosity, _ = model.generate_structure()
    
    print(f"Structure generated with:")
    print(f"Number of particles: {len(points)}")
    print(f"R1 range: {model.min_R1:.1f} - {model.max_R1:.1f}")
    print(f"R2 relationship: R2 = {model.m:.4f}*R1 + {model.b:.4f}")
    print(f"R3 relationship: R3 = R1 (circular in R3 plane)")
    print(f"Theoretical porosity: {porosity:.3f}")
    print(f"Volume fraction: {1 - porosity:.3f}")

if __name__ == "__main__":
    main() 
