import numpy as np
import pyvista as pv
from scipy.spatial import distance_matrix

class BooleanModelBase:
    def __init__(self, box_size=1.0, intensity=50, seed=None):
        """
        Initialize base Boolean Model class.
        
        Args:
            box_size (float): Size of the cubic domain
            intensity (float): Intensity of the Poisson point process (lambda)
            seed (int): Random seed for reproducibility
        """
        self.box_size = box_size
        self.intensity = intensity
        
        if seed is not None:
            np.random.seed(seed)
            
        # Calculate expected number of points
        self.volume = box_size**3
        self.mean_points = self.intensity * self.volume


    def generate_points(self):
        """Generate Poisson point process in the domain."""
        # Number of points follows Poisson distribution
        n_points = np.random.poisson(self.mean_points)
        
        # Generate uniform random points in the box
        points = np.random.uniform(0, self.box_size, (n_points, 3))
        return points

    def create_particles(self, points):
        """
        Create particles from points. To be implemented by child classes.
        
        Args:
            points: Array of particle center coordinates
            
        Returns:
            tuple: (list of PyVista PolyData objects representing particles, points)
        """
        raise NotImplementedError("This method should be implemented by child classes")

    def calculate_porosity(self, points):
        """
        Calculate theoretical porosity. To be implemented by child classes.
        
        Returns:
            float: Theoretical porosity (void fraction)
        """
        raise NotImplementedError("This method should be implemented by child classes")

    def generate_structure(self, show_plot=True):
        """
        Generate and visualize the Boolean model structure.
        
        Args:
            show_plot (bool): Whether to show the visualization
        
        Returns:
            tuple: (points, particles, porosity, plotter)
        """
        # Generate points
        points = self.generate_points()
        
        # Calculate porosity
        porosity = self.calculate_porosity(points)
        
        # Create particles
        particles, points = self.create_particles(points)
        
        # Create visualization
        plotter = self.visualize(particles)
        
        if show_plot:
            plotter.show()
        
        return points, particles, porosity, plotter 
    
    def clip_particles_to_box(self, particles_list):
        """
        Clip particles to stay within the box boundaries.
        
        Args:
            particles_list: List of PyVista PolyData objects representing particles
            
        Returns:
            list: Clipped particle meshes
        """
        clipped_particles = []
        for particle in particles_list:
            # Clip the particle with the box
            clipped = particle.clip_box(
                bounds=[0, self.box_size, 0, self.box_size, 0, self.box_size],
                invert=False
            )
            
            # Only add if the clipped particle is not empty
            if clipped.n_points > 0:
                clipped_particles.append(clipped)
                
        return clipped_particles

    def visualize(self, particles, opacity=1.0):
        """
        Visualize the Boolean model structure.
        
        Args:
            particles: List of PyVista PolyData objects representing particles
            opacity (float): Opacity of particles for visualization
        """
        plotter = pv.Plotter()
        
        # Clip particles to box boundaries
        clipped_particles = self.clip_particles_to_box(particles)
        
        # Add clipped particles to visualization
        for particle in clipped_particles:
            plotter.add_mesh(particle, color='red', opacity=opacity)
        
        # Add bounding box
        box = pv.Box([0, self.box_size, 0, self.box_size, 0, self.box_size])
        plotter.add_mesh(box, style='wireframe', color='black')
        
        # Set camera position
        plotter.camera_position = 'iso'
        return plotter