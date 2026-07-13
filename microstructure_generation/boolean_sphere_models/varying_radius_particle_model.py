'''
This model is currently not in use.
For the model generation, only the constant radius model is used
currently.
'''
import os
import sys

import numpy as np
import pyvista as pv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from boolean_model_base import BooleanModelBase

class BooleanParticleModel(BooleanModelBase):
    def __init__(self, box_size=1.0, intensity=50, min_radius=0.1, max_radius=0.3, 
                 distribution='uniform', lambda_poisson=5, seed=None):
        """
        Initialize Boolean Model of polydisperse spherical exclusions.
        
        Args:
            box_size (float): Size of the cubic domain
            intensity (float): Intensity of the Poisson point process (lambda)
            min_radius (float): Minimum radius of spherical particles
            max_radius (float): Maximum radius of spherical particles
            distribution (str): Distribution type for particle radii ('uniform', 'lognormal', or 'poisson')
            lambda_poisson (float): Lambda parameter for Poisson distribution (mean number of occurrences)
            seed (int): Random seed for reproducibility
        """
        super().__init__(box_size=box_size, intensity=intensity, seed=seed)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.distribution = distribution
        self.lambda_poisson = lambda_poisson
        
    def generate_radius(self):
        """Generate random radius based on specified distribution."""
        if self.distribution == 'uniform':
            return np.random.uniform(self.min_radius, self.max_radius)
        elif self.distribution == 'lognormal':
            # Parameters for lognormal distribution to match min/max radius range
            mu = np.log((self.min_radius + self.max_radius) / 2)
            sigma = 0.4  # Adjustable spread parameter
            radius = np.random.lognormal(mu, sigma)
            # Clip to ensure radius stays within bounds
            return np.clip(radius, self.min_radius, self.max_radius)
        elif self.distribution == 'poisson':
            # Generate Poisson random number and scale it to fit within radius range
            poisson_value = np.random.poisson(self.lambda_poisson)
            # Scale the Poisson value to fit within the radius range
            radius_range = self.max_radius - self.min_radius
            scaled_radius = self.min_radius + (poisson_value / (2 * self.lambda_poisson)) * radius_range
            # Clip to ensure radius stays within bounds
            return np.clip(scaled_radius, self.min_radius, self.max_radius)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def generate_points_and_radii(self):
        """Generate Poisson point process and particle radii."""
        points = self.generate_points()
        radii = np.array([self.generate_radius() for _ in range(len(points))])
        return points, radii

    def create_particles(self, points):
        """Create spherical particles with varying radii at the given points."""
        _, radii = self.generate_points_and_radii()  # Generate new radii for the points
        particles = []
        for center, radius in zip(points, radii):
            sphere = pv.Sphere(radius=radius, center=center)
            particles.append(sphere)
        return particles
    
    def calculate_porosity(self, points):
        """
        Calculate theoretical porosity of the structure.
        
        Returns:
            float: Theoretical porosity (void fraction)
        """
        # Generate radii for porosity calculation
        _, radii = self.generate_points_and_radii()
        
        # Calculate sphere volumes
        sphere_volumes = (4/3) * np.pi * radii**3
        
        # Average sphere volume for porosity calculation
        avg_sphere_vol = np.mean(sphere_volumes)
        
        # Theoretical porosity (using Boolean model formula)
        porosity = np.exp(-self.intensity * avg_sphere_vol)
        return porosity

    def generate_structure(self, show_plot=True):
        """
        Generate and visualize the Boolean model structure.
        
        Args:
            show_plot (bool): Whether to show the visualization
        
        Returns:
            tuple: (points, radii, porosity, plotter)
        """
        # Generate points and radii
        points, radii = self.generate_points_and_radii()
        
        # Calculate porosity
        porosity = self.calculate_porosity(points)
        
        # Create particles
        particles = self.create_particles(points)
        
        # Create visualization
        plotter = self.visualize(particles)
        
        if show_plot:
            plotter.show()
        
        return points, radii, porosity, plotter

def main():
    # Create Boolean model with specified parameters
    model = BooleanParticleModel(
        box_size=3,          # Box size
        intensity=50,        # Poisson point process intensity
        min_radius=0.1,      # Minimum sphere radius
        max_radius=0.15,     # Maximum sphere radius
        distribution='poisson',  # Distribution type for radii
        lambda_poisson=5,    # Lambda parameter for Poisson distribution
        seed=42             # For reproducibility
    )
    
    # Generate and visualize structure
    points, radii, porosity, _ = model.generate_structure()
    
    print(f"Structure generated with:")
    print(f"Number of particles: {len(points)}")
    print(f"Radius range: {model.min_radius:.2f} - {model.max_radius:.2f}")
    print(f"Radius distribution: {model.distribution}")
    print(f"Poisson lambda: {model.lambda_poisson}")
    print(f"Theoretical porosity: {porosity:.3f}")
    print(f"Volume fraction: {1 - porosity:.3f}")

if __name__ == "__main__":
    main() 
