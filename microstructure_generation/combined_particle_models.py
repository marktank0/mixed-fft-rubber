import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pyvista as pv
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from boolean_sphere_models.constant_particle_model import BooleanSphereExclusionModel
from boolean_sphere_models.spheroidal_inclusion_model import BooleanSpheroidalInclusionModel
from file_viewers.view_npz_3d import view_npz_file

# Default run settings
# Edit these values to change the defaults used by the CLI and helper functions.
DEFAULT_FILENAME_BASE = "temp_test"
DEFAULT_SAVE_DIR = "microstructure_generation/3D_samples"
DEFAULT_BOX_SIZE = 3.0

'''
Settings for seed 4 and 5
DEFAULT_MODEL1_INTENSITY = 240.0
DEFAULT_MODEL1_RADIUS = 0.10
DEFAULT_MODEL2_INTENSITY = 2
DEFAULT_MODEL2_MIN_R1 = 0.25
DEFAULT_MODEL2_MAX_R1 = 0.5
DEFAULT_MODEL3_INTENSITY = 1.8
DEFAULT_MODEL3_RADIUS = 0.4
DEFAULT_SEED = 5
'''

DEFAULT_MODEL1_INTENSITY = 240.0
DEFAULT_MODEL1_RADIUS = 0.10
DEFAULT_MODEL2_INTENSITY = 3
DEFAULT_MODEL2_MIN_R1 = 0.25
DEFAULT_MODEL2_MAX_R1 = 0.5
DEFAULT_MODEL3_INTENSITY = 1.8
DEFAULT_MODEL3_RADIUS = 0.4
DEFAULT_SEED = 5

DEFAULT_SAVE_PARTICLE_VTP = True
DEFAULT_SAVE_UNION_VTP = True
DEFAULT_SAVE_SPHERES_NPZ = True
DEFAULT_SAVE_VOXEL_NPZ = True
# Sub-folders (under the output dir) that each output type is written into.
SPHERES_SUBDIR = "sphere_structures"
VOXEL_SUBDIR = "voxel_structures"
# Densities used for the PHR value baked into filenames / reported by the CLI.
DEFAULT_FILLER_DENSITY = 1.8
DEFAULT_RUBBER_DENSITY = 0.92
# VOXEL_COARSENESS = 17 # the courseness of the voxel grid per 1 unit of box size
# DEFAULT_VOXEL_GRID_SHAPE = (VOXEL_COARSENESS*DEFAULT_BOX_SIZE, VOXEL_COARSENESS*DEFAULT_BOX_SIZE, VOXEL_COARSENESS*DEFAULT_BOX_SIZE)
DEFAULT_VOXEL_GRID_SHAPE = (63, 63, 63)
DEFAULT_NOTES = ""

# Cleanup defaults (both steps are opt-in via the 'cleanup' params block).
DEFAULT_REMOVE_FLOATING_CLUSTERS = True
DEFAULT_MIN_CLUSTER_SIZE = 4
DEFAULT_FILL_OCCLUSIONS = True
DEFAULT_MAX_OCCLUSION_VOXELS = 9
DEFAULT_OCCLUSION_PERIODIC = True
# Radius (in voxels) of the morphological closing that seals thin matrix
# channels through aggregates; gaps up to ~2*radius voxels wide are closed.
# 0 disables the step.
DEFAULT_CLOSING_RADIUS = 2

DEFAULT_VISUALIZE = True
DEFAULT_SHOW_UNIONS_IN_VIEWER = True
DEFAULT_VISUALIZATION_TARGET = "both" # "particles" "voxels" or "both"
DEFAULT_VOXEL_VIEW_BACKEND = "auto"
DEFAULT_VOXEL_VIEW_MODE = "surface"


def create_union_of_bodies(bodies):
    """
    Create a union of multiple PyVista meshes by merging them.
    This is optimized for non-intersecting meshes.
    
    Args:
        bodies (list): List of PyVista meshes to combine
        
    Returns:
        pv.PolyData: The combined mesh of all input meshes
    """
    if not bodies:
        return None
    
    # Start with the first body
    combined = bodies[0]
    for body in bodies[1:]:
        combined = combined.merge(body)
    return combined.triangulate()

def points_inside_spheres(points, centers, radius):
    """Boolean mask marking points inside ANY sphere of the given radius.

    Uses the exact analytic test (center-to-center distance <= radius). Unlike
    mesh ray-casting, this is unaffected by how many particles overlap a point --
    a point is "inside" as long as at least one sphere contains it. This is what
    fixes the failure where a cube fully covered by overlapping particle-3 spheres
    still left particle-1 spheres in the output.

    Args:
        points (np.ndarray): (N, 3) query points (particle-1 centers).
        centers (np.ndarray): (M, 3) sphere centers.
        radius (float): Common sphere radius.

    Returns:
        np.ndarray: Boolean mask of shape (N,), True where the point is inside.
    """
    points = np.asarray(points, dtype=float)
    inside = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0:
        return inside
    r_sq = float(radius) * float(radius)
    for center in np.asarray(centers, dtype=float):
        rel = points - center
        inside |= np.einsum("ij,ij->i", rel, rel) <= r_sq
    return inside


def points_inside_spheroids(points, centers, orientations, dimensions):
    """Boolean mask marking points inside ANY spheroid.

    Each point is transformed into the spheroid's local (unrotated) frame and
    tested against the unit sphere after dividing by the spheroid radii. This is
    the exact analytic containment test and, like ``points_inside_spheres``, is
    immune to particle overlap.

    Args:
        points (np.ndarray): (N, 3) query points (particle-1 centers).
        centers (np.ndarray): (M, 3) spheroid centers.
        orientations (list): M scipy ``Rotation`` objects (local -> world), the
            same ones used to build the spheroid meshes.
        dimensions (list): M tuples ``(R1, R2, R3)`` of spheroid radii.

    Returns:
        np.ndarray: Boolean mask of shape (N,), True where the point is inside.
    """
    points = np.asarray(points, dtype=float)
    inside = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0:
        return inside
    for center, rotation, dims in zip(centers, orientations, dimensions):
        rot_matrix = rotation.as_matrix()  # local -> world; its transpose maps world -> local
        rel = points - np.asarray(center, dtype=float)
        local = rel @ rot_matrix  # equals (rot_matrix.T @ rel_i) per row
        scaled = local / np.asarray(dims, dtype=float)
        inside |= np.einsum("ij,ij->i", scaled, scaled) <= 1.0
    return inside


def floating_cluster_mask(centers, radius, min_cluster_size):
    """Boolean mask keeping only spheres in clusters of >= min_cluster_size.

    Two equal-radius spheres are connected when their centers are within
    2 * radius (touching or overlapping). Connected components of that graph
    are the physical aggregates; isolated spheres or tiny groups floating in
    the matrix are not realistic and can be dropped.

    Args:
        centers (np.ndarray): (N, 3) sphere centers.
        radius (float): Common sphere radius.
        min_cluster_size (int): Minimum number of spheres a cluster needs to
            be kept.

    Returns:
        np.ndarray: Boolean mask of shape (N,), True for spheres to keep.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    if n == 0 or int(min_cluster_size) <= 1:
        return np.ones(n, dtype=bool)

    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0 * float(radius), output_type="ndarray")
    if pairs.size:
        adjacency = coo_matrix(
            (np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
            shape=(n, n),
        )
    else:
        adjacency = coo_matrix((n, n))
    n_components, labels = connected_components(adjacency, directed=False)
    cluster_sizes = np.bincount(labels, minlength=n_components)
    return cluster_sizes[labels] >= int(min_cluster_size)


def close_thin_channels_3d(phase, closing_radius=DEFAULT_CLOSING_RADIUS, periodic=DEFAULT_OCCLUSION_PERIODIC):
    """Seal thin matrix channels through filler with a morphological closing.

    3D counterpart of the closing step in smooth_2d_occlusions: the filler
    phase is dilated by ``closing_radius`` voxels and eroded back, which fills
    matrix gaps up to ~2 * closing_radius voxels wide (through-channels
    included, unlike the enclosed-pocket filler) while leaving the outer
    aggregate shape unchanged. Run this BEFORE fill_enclosed_occlusions_3d:
    closing a channel's mouths can turn its interior into a sealed pocket the
    pocket filler then removes.

    Args:
        phase (np.ndarray): 3D uint8 phase tensor, 1 = filler, 0 = matrix.
        closing_radius (int): Half-width of the closing structuring element in
            voxels; 0 disables the step.
        periodic (bool): Wrap across domain faces so gaps spanning the
            boundary close the same way as interior ones.

    Returns:
        tuple: (phase, n_closed) where phase is a new array if anything was
        closed (the input otherwise) and n_closed counts the added voxels.
    """
    phase = np.asarray(phase, dtype=np.uint8)
    radius = int(closing_radius)
    if radius <= 0:
        return phase, 0

    filler = phase > 0
    structure = np.ones((2 * radius + 1,) * 3, dtype=bool)
    if periodic:
        # A closing's region of influence extends 2*radius past a voxel, so a
        # wrap-pad of that width makes the boundary behave like the interior.
        pad = 2 * radius
        padded = np.pad(filler, pad, mode="wrap")
        closed = ndimage.binary_closing(padded, structure=structure)
        closed = closed[pad:-pad, pad:-pad, pad:-pad]
    else:
        closed = ndimage.binary_closing(filler, structure=structure)

    added = closed & ~filler
    n_closed = int(np.count_nonzero(added))
    if n_closed:
        phase = phase.copy()
        phase[added] = 1
    return phase, n_closed


def fill_enclosed_occlusions_3d(phase, max_voxels=DEFAULT_MAX_OCCLUSION_VOXELS, periodic=DEFAULT_OCCLUSION_PERIODIC):
    """Fill small enclosed matrix pockets inside filler in a voxel phase tensor.

    3D counterpart of smooth_2d_occlusions.fill_enclosed_polymer_occlusions:
    matrix (phase == 0) components with at most ``max_voxels`` voxels are set
    to filler. With ``periodic=True`` components are merged across opposite
    domain faces first (FFT solvers treat the domain as periodic, so a pocket
    split across the boundary is still one enclosed pocket). With
    ``periodic=False`` components touching the domain boundary are never
    filled, mirroring the 2D utility.

    Args:
        phase (np.ndarray): 3D uint8 phase tensor, 1 = filler, 0 = matrix.
        max_voxels (int): Largest matrix component (in voxels) to fill.
        periodic (bool): Treat the domain as periodic when finding components.

    Returns:
        tuple: (phase, n_filled) where phase is a new array if anything was
        filled (the input otherwise) and n_filled counts the filled voxels.
    """
    phase = np.asarray(phase, dtype=np.uint8)
    matrix = phase == 0
    labels, n_labels = ndimage.label(matrix)
    if n_labels == 0:
        return phase, 0

    if periodic:
        # Merge labels that touch each other across opposite faces. Node 0 is
        # the filler background; no edges reference it, so it stays alone in
        # its own component and keeps a unique id after relabeling.
        edges = []
        for axis in range(3):
            lo = np.take(labels, 0, axis=axis).ravel()
            hi = np.take(labels, -1, axis=axis).ravel()
            both = (lo > 0) & (hi > 0)
            if both.any():
                edges.append(np.stack([lo[both], hi[both]], axis=1))
        if edges:
            edges = np.unique(np.concatenate(edges), axis=0)
            adjacency = coo_matrix(
                (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                shape=(n_labels + 1, n_labels + 1),
            )
        else:
            adjacency = coo_matrix((n_labels + 1, n_labels + 1))
        n_ids, remap = connected_components(adjacency, directed=False)
        labels = remap[labels]
        background_id = int(remap[0])
    else:
        n_ids = n_labels + 1
        background_id = 0

    sizes = np.bincount(labels.ravel(), minlength=n_ids)
    fillable = sizes <= int(max_voxels)
    fillable[background_id] = False
    if not periodic:
        # Components reaching any domain face are open to the outside, not
        # enclosed pockets.
        for axis in range(3):
            fillable[np.unique(np.take(labels, 0, axis=axis))] = False
            fillable[np.unique(np.take(labels, -1, axis=axis))] = False

    fill_mask = fillable[labels]
    n_filled = int(np.count_nonzero(fill_mask))
    if n_filled:
        phase = phase.copy()
        phase[fill_mask] = 1
    return phase, n_filled


def _resolve_save_dir(save_dir):
    if os.path.isabs(save_dir):
        return save_dir
    return os.path.join(REPO_ROOT, save_dir)


def _sanitize_grid_shape(voxel_grid_shape):
    if len(voxel_grid_shape) != 3:
        raise ValueError("voxel_grid_shape must have 3 entries (Nx, Ny, Nz).")
    nx, ny, nz = [int(v) for v in voxel_grid_shape]
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("voxel_grid_shape values must all be positive.")
    return nx, ny, nz


def _prepare_sphere_arrays(sphere_centers, sphere_radii):
    centers = np.asarray(sphere_centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("sphere_centers must have shape (N, 3).")

    radii = np.asarray(sphere_radii, dtype=float).reshape(-1)
    if radii.size == 1 and centers.shape[0] > 1:
        radii = np.full(centers.shape[0], radii.item(), dtype=float)
    if radii.size != centers.shape[0]:
        raise ValueError("sphere_radii must have length N (or be a scalar).")
    if np.any(radii <= 0.0):
        raise ValueError("All sphere radii must be positive.")

    return centers, radii


def _infer_sphere_params_from_particles(particles):
    if not particles:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    centers = []
    radii = []
    for particle in particles:
        centers.append(np.asarray(particle.center, dtype=float))
        bounds = np.asarray(particle.bounds, dtype=float)
        extents = np.array(
            [
                bounds[1] - bounds[0],
                bounds[3] - bounds[2],
                bounds[5] - bounds[4],
            ],
            dtype=float,
        )
        radii.append(0.5 * np.mean(extents))
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def save_sphere_parameters_npz(file_path, sphere_centers, sphere_radii, box_size, notes=""):
    centers, radii = _prepare_sphere_arrays(sphere_centers, sphere_radii)
    np.savez_compressed(
        file_path,
        centers=centers.astype(np.float32),
        radii=radii.astype(np.float32),
        origin=np.zeros(3, dtype=np.float32),
        box_size=np.array([float(box_size)], dtype=np.float32),
        notes=np.array([notes], dtype=np.str_),
        format=np.array(["sphere_parameters_v1"], dtype=np.str_),
    )


def voxelize_spheres_to_phase(sphere_centers, sphere_radii, box_size, voxel_grid_shape):
    """
    Rasterize spheres into a 3D binary phase tensor.

    Returns:
        np.ndarray: phase tensor with shape (Nx, Ny, Nz), dtype uint8, values in {0,1}.
    """
    centers, radii = _prepare_sphere_arrays(sphere_centers, sphere_radii)
    nx, ny, nz = _sanitize_grid_shape(voxel_grid_shape)
    box_size = float(box_size)
    if box_size <= 0.0:
        raise ValueError("box_size must be positive.")

    phase = np.zeros((nx, ny, nz), dtype=np.uint8)
    if centers.shape[0] == 0:
        return phase

    dx = box_size / float(nx)
    dy = box_size / float(ny)
    dz = box_size / float(nz)
    x_coords = (np.arange(nx, dtype=float) + 0.5) * dx
    y_coords = (np.arange(ny, dtype=float) + 0.5) * dy
    z_coords = (np.arange(nz, dtype=float) + 0.5) * dz

    for center, radius in zip(centers, radii):
        cx, cy, cz = center
        radius_sq = float(radius * radius)

        ix0 = max(0, int(np.floor((cx - radius) / dx)))
        iy0 = max(0, int(np.floor((cy - radius) / dy)))
        iz0 = max(0, int(np.floor((cz - radius) / dz)))
        ix1 = min(nx - 1, int(np.floor((cx + radius) / dx)))
        iy1 = min(ny - 1, int(np.floor((cy + radius) / dy)))
        iz1 = min(nz - 1, int(np.floor((cz + radius) / dz)))

        if ix0 > ix1 or iy0 > iy1 or iz0 > iz1:
            continue

        local_x = x_coords[ix0 : ix1 + 1] - cx
        local_y = y_coords[iy0 : iy1 + 1] - cy
        local_z = z_coords[iz0 : iz1 + 1] - cz
        dist_sq = (
            local_x[:, None, None] ** 2
            + local_y[None, :, None] ** 2
            + local_z[None, None, :] ** 2
        )
        phase[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] |= (
            dist_sq <= radius_sq
        ).astype(np.uint8)

    return phase


def save_voxel_phase_npz(file_path, phase, box_size, notes=""):
    phase = np.asarray(phase, dtype=np.uint8)
    if phase.ndim != 3:
        raise ValueError("phase must be a 3D array.")
    nx, ny, nz = phase.shape
    box_size = float(box_size)
    voxel_size = np.array(
        [box_size / float(nx), box_size / float(ny), box_size / float(nz)],
        dtype=np.float32,
    )
    np.savez_compressed(
        file_path,
        phase=phase,
        voxel_size=voxel_size,
        origin=np.zeros(3, dtype=np.float32),
        box_size=np.array([box_size], dtype=np.float32),
        notes=np.array([notes], dtype=np.str_),
        format=np.array(["voxel_phase_v1"], dtype=np.str_),
    )


def visualize_combined_models(
    model1_particles,
    model2_union=None,
    model3_union=None,
    box_size=DEFAULT_BOX_SIZE,
    show_unions=DEFAULT_SHOW_UNIONS_IN_VIEWER,
):
    """
    Visualize both models together.
    
    Args:
        model1_particles: List of triangulated particles from Model 1
        model2_union: Union mesh of Model 2 particles
        model3_union: Union mesh of Model 3 particles
        box_size: Size of the cubic domain
        show_unions: Whether to show the union meshes
    """
    plotter = pv.Plotter()
    
    # Add Model 1 particles (spheres)
    for particle in model1_particles:
        plotter.add_mesh(particle, color='red', opacity=1.0)
    
    if show_unions:
        # Add Model 2 union (spheroids) if provided
        if model2_union is not None:
            plotter.add_mesh(model2_union, color='blue', opacity=1.0)#0.3)
        
        # Add Model 3 union if provided
        if model3_union is not None:
            plotter.add_mesh(model3_union, color='green', opacity=1.0)#0.15)
    
    # Add bounding box
    box = pv.Box([0, box_size, 0, box_size, 0, box_size])
    plotter.add_mesh(box, style='wireframe', color='black')
    
    # Set camera position
    plotter.camera_position = "iso"
    plotter.show()


def visualize_voxel_structure(
    voxel_npz_path,
    backend=DEFAULT_VOXEL_VIEW_BACKEND,
    mode=DEFAULT_VOXEL_VIEW_MODE,
):
    """
    Visualize a saved voxel NPZ using the shared viewer module.

    Args:
        voxel_npz_path (str): Path to the saved voxel NPZ file.
        backend (str): Viewer backend ('auto', 'pyvista', or 'matplotlib').
        mode (str): Viewer mode ('scatter', 'surface', or 'voxels').
    """
    view_npz_file(
        input_path=voxel_npz_path,
        backend=backend,
        mode=mode,
        no_show=False,
    )


def save_particle_structure(
    particles,
    unions,
    filename_base,
    save_dir=DEFAULT_SAVE_DIR,
    sphere_centers=None,
    sphere_radii=None,
    box_size=DEFAULT_BOX_SIZE,
    save_particle_vtp=DEFAULT_SAVE_PARTICLE_VTP,
    save_union_vtp=DEFAULT_SAVE_UNION_VTP,
    save_spheres_npz=DEFAULT_SAVE_SPHERES_NPZ,
    save_voxel_npz=DEFAULT_SAVE_VOXEL_NPZ,
    voxel_grid_shape=DEFAULT_VOXEL_GRID_SHAPE,
    notes=DEFAULT_NOTES,
    precomputed_phase=None,
):
    """
    Save the generated structure in one or more formats.

    Sphere (.npz) and voxel (.npz) outputs are written into dedicated
    sub-folders of ``save_dir`` (``sphere_structures/`` and
    ``voxel_structures/``) so the two representations stay separated.

    Args:
        particles (list): List of particle meshes to save.
        unions (dict): Dictionary containing union meshes with keys 'union2' and 'union3'
        filename_base (str): Base name for saved files.
        save_dir (str): Relative (or absolute) output directory.
        sphere_centers (np.ndarray): Optional sphere centers with shape (N, 3).
        sphere_radii (np.ndarray|float): Optional radii with length N (or scalar).
        box_size (float): Side length of the full cubic domain.
        save_particle_vtp (bool): Save each particle as individual .vtp.
        save_union_vtp (bool): Save union2 and union3 meshes as .vtp.
        save_spheres_npz (bool): Save compact sphere parameter file.
        save_voxel_npz (bool): Save solver-ready voxel phase tensor.
        voxel_grid_shape (tuple): Grid size for voxel export (Nx, Ny, Nz).
        notes (str): Optional metadata text saved into npz files.
        precomputed_phase (np.ndarray|None): Reuse an already-voxelized phase
            tensor instead of rasterizing again (used when the caller has
            already voxelized to compute PHR).
    """
    full_save_dir = _resolve_save_dir(save_dir)
    # Create save directory if it doesn't exist
    os.makedirs(full_save_dir, exist_ok=True)

    saved_paths = {}

    # Save particles individually (.vtp)
    if save_particle_vtp and particles:
        particles_dir = os.path.join(full_save_dir, f"{filename_base}_particles")
        os.makedirs(particles_dir, exist_ok=True)

        for i, particle in enumerate(particles):
            particle.save(os.path.join(particles_dir, f"particle_{i:06d}.vtp"), binary=True)
        saved_paths["particles_vtp_dir"] = particles_dir

    # Save union meshes (.vtp)
    if save_union_vtp and unions.get("union2") is not None:
        union2_path = os.path.join(full_save_dir, f"{filename_base}_union2.vtp")
        unions["union2"].save(
            union2_path,
            binary=True
        )
        saved_paths["union2_vtp"] = union2_path
    if save_union_vtp and unions.get("union3") is not None:
        union3_path = os.path.join(full_save_dir, f"{filename_base}_union3.vtp")
        unions["union3"].save(
            union3_path,
            binary=True
        )
        saved_paths["union3_vtp"] = union3_path

    need_sphere_data = save_spheres_npz or save_voxel_npz
    if need_sphere_data:
        if sphere_centers is None or sphere_radii is None:
            inferred_centers, inferred_radii = _infer_sphere_params_from_particles(particles)
            if sphere_centers is None:
                sphere_centers = inferred_centers
            if sphere_radii is None:
                sphere_radii = inferred_radii

        sphere_centers, sphere_radii = _prepare_sphere_arrays(sphere_centers, sphere_radii)

    if box_size is None:
        raise ValueError(
            "box_size must be provided when saving a structure. "
            "The save routine no longer infers a smaller box from particle extents."
        )

    # Save compact sphere parameter representation (.npz)
    if save_spheres_npz:
        spheres_dir = os.path.join(full_save_dir, SPHERES_SUBDIR)
        os.makedirs(spheres_dir, exist_ok=True)
        sphere_npz_path = os.path.join(spheres_dir, f"{filename_base}_spheres.npz")
        save_sphere_parameters_npz(
            sphere_npz_path,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            box_size=box_size,
            notes=notes,
        )
        saved_paths["spheres_npz"] = sphere_npz_path

    # Save solver-ready voxel representation (.npz)
    if save_voxel_npz:
        if precomputed_phase is not None:
            phase = precomputed_phase
        else:
            phase = voxelize_spheres_to_phase(
                sphere_centers=sphere_centers,
                sphere_radii=sphere_radii,
                box_size=box_size,
                voxel_grid_shape=voxel_grid_shape,
            )
        voxel_dir = os.path.join(full_save_dir, VOXEL_SUBDIR)
        os.makedirs(voxel_dir, exist_ok=True)
        voxel_npz_path = os.path.join(voxel_dir, f"{filename_base}_voxel.npz")
        save_voxel_phase_npz(
            voxel_npz_path,
            phase=phase,
            box_size=box_size,
            notes=notes,
        )
        saved_paths["voxel_npz"] = voxel_npz_path

    return saved_paths


def load_particle_structure(filename_base, save_dir=DEFAULT_SAVE_DIR):
    """
    Load a previously saved particle structure.

    Args:
        filename_base (str): Base name of the saved files
        save_dir (str): Directory where files are saved (relative to script location)

    Returns:
        tuple: (particles_mesh, union2_mesh, union3_mesh)
    """
    import glob

    # Create the full path for the save directory
    full_save_dir = _resolve_save_dir(save_dir)

    # Initialize return values
    particles = []
    union2_mesh = None
    union3_mesh = None

    # Load particles
    particles_dir = os.path.join(full_save_dir, f"{filename_base}_particles")
    if os.path.exists(particles_dir):
        # Get all particle files
        particle_files = sorted(glob.glob(os.path.join(particles_dir, "particle_*.vtp")))
        for file in particle_files:
            particles.append(pv.read(file))

    # Load union2
    union2_file = os.path.join(full_save_dir, f"{filename_base}_union2.vtp")
    if os.path.exists(union2_file):
        union2_mesh = pv.read(union2_file)

    # Load union3
    union3_file = os.path.join(full_save_dir, f"{filename_base}_union3.vtp")
    if os.path.exists(union3_file):
        union3_mesh = pv.read(union3_file)

    return particles, union2_mesh, union3_mesh


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate boolean-sphere structures and save outputs in mesh (.vtp), "
            "sphere-parameter (.npz), and optional voxel-phase (.npz) formats."
        )
    )
    parser.add_argument("--filename-base", default=DEFAULT_FILENAME_BASE, help="Base name for saved files.")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR, help="Output folder (relative or absolute).")
    parser.add_argument("--box-size", type=float, default=DEFAULT_BOX_SIZE, help="Cubic domain size.")

    parser.add_argument("--model1-intensity", type=float, default=DEFAULT_MODEL1_INTENSITY, help="PPP intensity for filler spheres.")
    parser.add_argument("--model1-radius", type=float, default=DEFAULT_MODEL1_RADIUS, help="Radius for filler spheres.")
    parser.add_argument("--model2-intensity", type=float, default=DEFAULT_MODEL2_INTENSITY, help="Intensity for spheroidal inclusion model.")
    parser.add_argument("--model2-min-r1", type=float, default=DEFAULT_MODEL2_MIN_R1, help="Minimum major radius for model2 spheroids.")
    parser.add_argument("--model2-max-r1", type=float, default=DEFAULT_MODEL2_MAX_R1, help="Maximum major radius for model2 spheroids.")
    parser.add_argument("--model3-intensity", type=float, default=DEFAULT_MODEL3_INTENSITY, help="Intensity for exclusion spheres.")
    parser.add_argument("--model3-radius", type=float, default=DEFAULT_MODEL3_RADIUS, help="Radius for exclusion spheres.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility.")

    parser.add_argument(
        "--no-particle-vtp",
        dest="save_particle_vtp",
        action="store_false",
        default=DEFAULT_SAVE_PARTICLE_VTP,
        help="Disable per-particle .vtp export.",
    )
    parser.add_argument(
        "--no-union-vtp",
        dest="save_union_vtp",
        action="store_false",
        default=DEFAULT_SAVE_UNION_VTP,
        help="Disable union .vtp export.",
    )
    parser.add_argument(
        "--no-spheres-npz",
        dest="save_spheres_npz",
        action="store_false",
        default=DEFAULT_SAVE_SPHERES_NPZ,
        help="Disable compact sphere .npz export.",
    )
    parser.add_argument(
        "--save-voxel-npz",
        dest="save_voxel_npz",
        action="store_true",
        default=DEFAULT_SAVE_VOXEL_NPZ,
        help="Also export solver-ready voxel .npz.",
    )
    parser.add_argument(
        "--voxel-grid",
        nargs=3,
        type=int,
        default=list(DEFAULT_VOXEL_GRID_SHAPE),
        metavar=("NX", "NY", "NZ"),
        help="Grid size for voxel export.",
    )
    parser.add_argument(
        "--remove-floating-clusters",
        action="store_true",
        default=DEFAULT_REMOVE_FLOATING_CLUSTERS,
        help="Drop spheres in clusters smaller than --min-cluster-size before saving/voxelization.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="Minimum sphere count for a cluster to be kept.",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=DEFAULT_CLOSING_RADIUS,
        help="Morphological closing radius (voxels) sealing thin matrix channels; 0 disables.",
    )
    parser.add_argument(
        "--fill-occlusions",
        action="store_true",
        default=DEFAULT_FILL_OCCLUSIONS,
        help="Fill small enclosed matrix pockets in the voxelized structure.",
    )
    parser.add_argument(
        "--max-occlusion-voxels",
        type=int,
        default=DEFAULT_MAX_OCCLUSION_VOXELS,
        help="Largest matrix pocket (in voxels) that gets filled.",
    )
    parser.add_argument(
        "--occlusion-non-periodic",
        dest="occlusion_periodic",
        action="store_false",
        default=DEFAULT_OCCLUSION_PERIODIC,
        help="Treat the domain as non-periodic when detecting enclosed pockets.",
    )
    parser.add_argument("--notes", default=DEFAULT_NOTES, help="Optional notes metadata stored in npz outputs.")
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=DEFAULT_VISUALIZE,
        help="Disable final PyVista visualization.",
    )
    parser.add_argument(
        "--hide-unions-in-viewer",
        dest="show_unions_in_viewer",
        action="store_false",
        default=DEFAULT_SHOW_UNIONS_IN_VIEWER,
        help="When visualizing, hide union2/union3 and show only final spheres.",
    )
    parser.add_argument(
        "--visualization-target",
        choices=["particles", "voxels", "both"],
        default=DEFAULT_VISUALIZATION_TARGET,
        help="Choose whether final visualization shows particle meshes, voxel view, or both.",
    )
    parser.add_argument(
        "--voxel-view-backend",
        choices=["auto", "pyvista", "matplotlib"],
        default=DEFAULT_VOXEL_VIEW_BACKEND,
        help="Backend used when visualizing voxel output.",
    )
    parser.add_argument(
        "--voxel-view-mode",
        choices=["scatter", "surface", "voxels"],
        default=DEFAULT_VOXEL_VIEW_MODE,
        help="Rendering mode used when visualizing voxel output.",
    )
    return parser


def _phr_from_phase(phase, box_size, filler_density, rubber_density):
    """Return (filler_fraction, phr) for a voxel phase tensor using calculate_phr."""
    import calculate_phr as cp

    filler_fraction = float(np.count_nonzero(phase > 0.5)) / float(phase.size)
    total_volume = float(box_size) ** 3
    filler_volume = filler_fraction * total_volume
    phr = cp.calculate_phr(
        filler_volume=filler_volume,
        total_volume=total_volume,
        filler_density=filler_density,
        rubber_density=rubber_density,
    )
    return filler_fraction, phr


def _resolve_structure_name(params, phr):
    """Resolve the final filename base, filling {phr} (2 decimals) when templated.

    The CLI passes a literal ``name``; the batch runner passes ``name_template``
    plus ``name_context`` so the PHR value can be injected after generation.
    """
    template = params.get("name_template")
    if template is None:
        return params.get("name", DEFAULT_FILENAME_BASE)

    context = dict(params.get("name_context", {}))
    context["phr"] = "{:.2f}".format(phr) if phr is not None else "NA"
    return template.format(**context)


def generate_and_save(
    params,
    return_meshes=False,
    verbose=True,
    filler_density=DEFAULT_FILLER_DENSITY,
    rubber_density=DEFAULT_RUBBER_DENSITY,
    on_existing="overwrite",
):
    """Generate one combined-model structure from a parameter dict and save it.

    This wraps the exact same pipeline the CLI uses so that single CLI runs and
    batch runs stay identical. No visualization is performed here.

    The structure is voxelized once; its PHR is computed from that phase (via
    calculate_phr and the given densities), can be injected into the filename,
    and the same phase is reused for the voxel export.

    Args:
        params (dict): Per-structure parameters. Recognized keys:
            box_size (float), model1/model2/model3 (dict), voxel_grid (tuple),
            outputs (dict of save_* flags), notes (str), seed (int|None),
            output_dir (str), and either name (str) or
            name_template (str) + name_context (dict).
        return_meshes (bool): If True, include the particle/union meshes in the
            result so the CLI can visualize them afterwards.
        verbose (bool): Print a one-line progress summary.
        filler_density, rubber_density (float): Densities for the PHR value.
        on_existing (str): 'overwrite' | 'skip' | 'error' when the primary
            output file already exists (checked after the name is resolved).

    Returns:
        dict: name, seed, box_size, saved_paths, n_points, n_final, phr,
              filler_fraction, status, elapsed and (optionally)
              final_particles/union2/union3.
    """
    box_size = float(params.get("box_size", DEFAULT_BOX_SIZE))
    model1_cfg = params.get("model1", {}) or {}
    model2_cfg = params.get("model2", {}) or {}
    model3_cfg = params.get("model3", {}) or {}
    outputs = params.get("outputs", {}) or {}
    cleanup_cfg = params.get("cleanup", {}) or {}
    seed = params.get("seed", DEFAULT_SEED)
    save_dir = params.get("output_dir", DEFAULT_SAVE_DIR)
    voxel_grid = tuple(params.get("voxel_grid", DEFAULT_VOXEL_GRID_SHAPE))
    notes = params.get("notes", DEFAULT_NOTES)

    start_time = time.time()

    # Create models and get their particles and points
    model1 = BooleanSphereExclusionModel(
        box_size=box_size,
        intensity=model1_cfg.get("intensity", DEFAULT_MODEL1_INTENSITY),
        radius=model1_cfg.get("radius", DEFAULT_MODEL1_RADIUS),
        seed=seed,
    )

    # Create Model 2 (Spheroidal inclusion model)
    model2 = BooleanSpheroidalInclusionModel(
        box_size=box_size,
        intensity=model2_cfg.get("intensity", DEFAULT_MODEL2_INTENSITY),
        min_R1=model2_cfg.get("min_r1", DEFAULT_MODEL2_MIN_R1),
        max_R1=model2_cfg.get("max_r1", DEFAULT_MODEL2_MAX_R1),
        seed=seed,
    )

    model3 = BooleanSphereExclusionModel(
        box_size=box_size,
        intensity=model3_cfg.get("intensity", DEFAULT_MODEL3_INTENSITY),
        radius=model3_cfg.get("radius", DEFAULT_MODEL3_RADIUS),
        seed=seed,
    )

    # First generate only the points for model1 (no particles yet)
    model1_points = model1.generate_points()

    # Generate model2 (spheroids) keeping the exact centers/orientations/dimensions
    # so containment is tested against the true particle volumes, and the meshes
    # (if built for visualization) match those same parameters.
    model2_points, model2_orientations, model2_dimensions = model2.generate_points_and_dimensions()
    model2_particles, _ = model2.create_particles(
        model2_points, model2_orientations, model2_dimensions
    )

    # Model3 spheres have a constant radius; keep their centers for the analytic test.
    model3_points = model3.generate_points()
    model3_particles, _ = model3.create_particles(model3_points)

    # Keep only model1 centers that lie inside the volume of at least one model2
    # particle AND outside every model3 particle. These analytic tests use the
    # particle centers themselves (not a merged mesh), so they are exact and are
    # not fooled by overlapping particles -- covering the box with model3 spheres
    # now correctly removes all model1 spheres in that region.
    inside_model2 = points_inside_spheroids(
        model1_points, model2_points, model2_orientations, model2_dimensions
    )
    inside_model3 = points_inside_spheres(model1_points, model3_points, model3.radius)
    final_mask = inside_model2 & ~inside_model3

    # Get the filtered points
    filtered_points = model1_points[final_mask]

    # Optionally drop floating spheres / tiny groups not attached to a larger
    # aggregate. Done before meshing, voxelization and PHR so every saved
    # output reflects the cleaned structure.
    n_removed_floaters = 0
    if cleanup_cfg.get("remove_floating_clusters", DEFAULT_REMOVE_FLOATING_CLUSTERS):
        keep = floating_cluster_mask(
            filtered_points,
            model1.radius,
            cleanup_cfg.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE),
        )
        n_removed_floaters = int(np.count_nonzero(~keep))
        filtered_points = filtered_points[keep]

    # Union meshes are only needed for optional visualization / .vtp export.
    need_union_meshes = return_meshes or outputs.get("save_union_vtp", DEFAULT_SAVE_UNION_VTP)
    union2 = create_union_of_bodies(model2_particles) if need_union_meshes else None
    union3 = create_union_of_bodies(model3_particles) if need_union_meshes else None

    # Only now create particles for the filtered points
    final_particles, _ = model1.create_particles(filtered_points)

    # Voxelize once, then reuse the phase for both the PHR value and the export.
    sphere_radii = np.full(len(filtered_points), model1.radius, dtype=float)
    phase = voxelize_spheres_to_phase(
        sphere_centers=filtered_points,
        sphere_radii=sphere_radii,
        box_size=box_size,
        voxel_grid_shape=voxel_grid,
    )

    # Optionally seal thin matrix channels through aggregates (morphological
    # closing), then fill tiny enclosed matrix pockets, before PHR is computed
    # so the filename/manifest match. Closing runs first: sealing a channel's
    # mouths can turn its interior into a pocket the filler then removes.
    n_closed_voxels = 0
    closing_radius = int(cleanup_cfg.get("closing_radius", DEFAULT_CLOSING_RADIUS))
    if closing_radius > 0:
        phase, n_closed_voxels = close_thin_channels_3d(
            phase,
            closing_radius=closing_radius,
            periodic=cleanup_cfg.get("periodic", DEFAULT_OCCLUSION_PERIODIC),
        )

    n_filled_voxels = 0
    if cleanup_cfg.get("fill_occlusions", DEFAULT_FILL_OCCLUSIONS):
        phase, n_filled_voxels = fill_enclosed_occlusions_3d(
            phase,
            max_voxels=cleanup_cfg.get("max_occlusion_voxels", DEFAULT_MAX_OCCLUSION_VOXELS),
            periodic=cleanup_cfg.get("periodic", DEFAULT_OCCLUSION_PERIODIC),
        )

    filler_fraction, phr = _phr_from_phase(phase, box_size, filler_density, rubber_density)

    # Resolve the final filename (may embed the PHR value) now that PHR is known.
    name = _resolve_structure_name(params, phr)

    # Honor the on_existing policy against the resolved primary output.
    full_save_dir = _resolve_save_dir(save_dir)
    if outputs.get("save_spheres_npz", DEFAULT_SAVE_SPHERES_NPZ):
        primary = os.path.join(full_save_dir, SPHERES_SUBDIR, f"{name}_spheres.npz")
    else:
        primary = os.path.join(full_save_dir, VOXEL_SUBDIR, f"{name}_voxel.npz")

    status = "generated"
    if os.path.exists(primary):
        if on_existing == "error":
            raise RuntimeError(f"Output already exists for {name!r}: {primary}")
        if on_existing == "skip":
            status = "skipped"

    unions = {"union2": union2, "union3": union3}
    if status == "skipped":
        saved_paths = {}
        if verbose:
            print(f"[{name}] exists, skipping (phr={phr:.2f})")
    else:
        saved_paths = save_particle_structure(
            final_particles,
            unions,
            name,
            save_dir=save_dir,
            sphere_centers=filtered_points,
            sphere_radii=sphere_radii,
            box_size=box_size,
            save_particle_vtp=outputs.get("save_particle_vtp", DEFAULT_SAVE_PARTICLE_VTP),
            save_union_vtp=outputs.get("save_union_vtp", DEFAULT_SAVE_UNION_VTP),
            save_spheres_npz=outputs.get("save_spheres_npz", DEFAULT_SAVE_SPHERES_NPZ),
            save_voxel_npz=outputs.get("save_voxel_npz", DEFAULT_SAVE_VOXEL_NPZ),
            voxel_grid_shape=voxel_grid,
            notes=notes,
            precomputed_phase=phase,
        )

    result = {
        "name": name,
        "seed": seed,
        "box_size": box_size,
        "saved_paths": saved_paths,
        "n_points": int(len(model1_points)),
        "n_final": int(len(filtered_points)),
        "n_removed_floaters": n_removed_floaters,
        "n_closed_voxels": n_closed_voxels,
        "n_filled_voxels": n_filled_voxels,
        "filler_fraction": filler_fraction,
        "phr": phr,
        "status": status,
        "elapsed": time.time() - start_time,
    }

    if verbose and status == "generated":
        cleanup_note = ""
        if n_removed_floaters or n_closed_voxels or n_filled_voxels:
            cleanup_note = ", removed {} floaters, closed {} + filled {} voxels".format(
                n_removed_floaters, n_closed_voxels, n_filled_voxels
            )
        print(
            "[{}] final {} spheres, phr={:.2f}{} ({:.1f}s)".format(
                name, result["n_final"], phr, cleanup_note, result["elapsed"]
            )
        )

    if return_meshes:
        result["final_particles"] = final_particles
        result["union2"] = union2
        result["union3"] = union3

    return result


def _params_from_args(args):
    """Translate parsed CLI arguments into a generate_and_save params dict."""
    return {
        "box_size": args.box_size,
        "model1": {"intensity": args.model1_intensity, "radius": args.model1_radius},
        "model2": {
            "intensity": args.model2_intensity,
            "min_r1": args.model2_min_r1,
            "max_r1": args.model2_max_r1,
        },
        "model3": {"intensity": args.model3_intensity, "radius": args.model3_radius},
        "voxel_grid": tuple(args.voxel_grid),
        "outputs": {
            "save_particle_vtp": args.save_particle_vtp,
            "save_union_vtp": args.save_union_vtp,
            "save_spheres_npz": args.save_spheres_npz,
            "save_voxel_npz": args.save_voxel_npz,
        },
        "cleanup": {
            "remove_floating_clusters": args.remove_floating_clusters,
            "min_cluster_size": args.min_cluster_size,
            "closing_radius": args.closing_radius,
            "fill_occlusions": args.fill_occlusions,
            "max_occlusion_voxels": args.max_occlusion_voxels,
            "periodic": args.occlusion_periodic,
        },
        "notes": args.notes,
        "seed": args.seed,
        "name": args.filename_base,
        "output_dir": args.save_dir,
    }


def main(cli_args=None):
    args = _build_parser().parse_args(cli_args)

    params = _params_from_args(args)
    result = generate_and_save(params, return_meshes=args.visualize)
    saved_paths = result["saved_paths"]
    box_size = args.box_size

    print("\nSaved files:")
    for key, value in saved_paths.items():
        print(f"  {key}: {value}")

    if args.visualize and args.visualization_target in {"particles", "both"}:
        visualize_combined_models(
            result["final_particles"],
            result["union2"],
            result["union3"],
            box_size=box_size,
            show_unions=args.show_unions_in_viewer,
        )

    if args.visualize and args.visualization_target in {"voxels", "both"}:
        voxel_npz_path = saved_paths.get("voxel_npz")
        if voxel_npz_path is None:
            print(
                "\nVoxel visualization requested, but no voxel NPZ was saved. "
                "Enable voxel export with the current defaults or pass --save-voxel-npz."
            )
        else:
            visualize_voxel_structure(
                voxel_npz_path,
                backend=args.voxel_view_backend,
                mode=args.voxel_view_mode,
            )

    print(f"Total time taken: {result['elapsed']:.2f} seconds")

if __name__ == "__main__":
    main()
