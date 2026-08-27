"""TEMPORARY figure script: show each Boolean model of the combined pipeline on its own.

Reproduces the exact generation sequence of ``combined_particle_models.generate_and_save``
(same construction order, same global-RNG draws, so with the same --seed the geometry is
identical to a real run) but renders the intermediate stages one at a time and writes a
PNG per stage, for use as paper figures.

Stages
------
  model1     Boolean sphere model 1: all filler spheres from the PPP (red)
  model2     Boolean spheroidal inclusion model: aggregate domains (blue)
  model3     Boolean sphere exclusion model: exclusion spheres (green)
  overlay    model2 + model3 together, no filler
  inside2    model-1 spheres kept inside a model-2 spheroid
  final      after the model-3 exclusion and the floating-cluster cleanup
  voxels     the final structure voxelized (what the solver actually sees)

Usage
-----
  # interactive: a window per stage; orbit to the view you want, close it, PNG is saved
  .venv/Scripts/python.exe microstructure_generation/temp_view_boolean_models.py

  # same camera for every figure (taken from the first window you close)
  ... --link-camera --camera-file paper_cpos.json

  # no windows at all, just write the PNGs from a fixed iso view
  ... --off-screen

  # only some stages, with a non-zero model 3 (the pipeline default is 0 -> empty stage)
  ... --stages model2 model3 final --model3-intensity 1.8
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pyvista as pv

import combined_particle_models as cpm
from boolean_sphere_models.constant_particle_model import BooleanSphereExclusionModel
from boolean_sphere_models.spheroidal_inclusion_model import BooleanSpheroidalInclusionModel

STAGES = ["model1", "model2", "model3", "overlay", "inside2", "final", "voxels"]

COLOR_MODEL1 = "red"
COLOR_MODEL2 = "blue"
COLOR_MODEL3 = "green"

DEFAULT_OUT_DIR = "microstructure_generation/3D_samples/model_figures"


def build_stage_geometry(args):
    """Run the combined pipeline and return the meshes/points for every stage.

    The order of model construction and of the generate_* calls is kept identical to
    ``combined_particle_models.generate_and_save`` -- each model constructor reseeds the
    global numpy RNG, so any reordering would change the geometry.
    """
    box_size = args.box_size

    model1 = BooleanSphereExclusionModel(
        box_size=box_size,
        intensity=args.model1_intensity,
        radius=args.model1_radius,
        seed=args.seed,
    )
    model2 = BooleanSpheroidalInclusionModel(
        box_size=box_size,
        intensity=args.model2_intensity,
        min_R1=args.model2_min_r1,
        max_R1=args.model2_max_r1,
        seed=args.seed,
    )
    model3 = BooleanSphereExclusionModel(
        box_size=box_size,
        intensity=args.model3_intensity,
        radius=args.model3_radius,
        seed=args.seed,
    )

    model1_points = model1.generate_points()

    model2_points, model2_orientations, model2_dimensions = model2.generate_points_and_dimensions()
    model2_particles, _ = model2.create_particles(
        model2_points, model2_orientations, model2_dimensions
    )

    model3_points = model3.generate_points()
    model3_particles, _ = model3.create_particles(model3_points)

    inside_model2 = cpm.points_inside_spheroids(
        model1_points, model2_points, model2_orientations, model2_dimensions
    )
    inside_model3 = cpm.points_inside_spheres(model1_points, model3_points, model3.radius)

    inside2_points = model1_points[inside_model2]
    filtered_points = model1_points[inside_model2 & ~inside_model3]

    n_removed_floaters = 0
    if args.remove_floating_clusters:
        keep = cpm.floating_cluster_mask(filtered_points, model1.radius, args.min_cluster_size)
        n_removed_floaters = int(np.count_nonzero(~keep))
        filtered_points = filtered_points[keep]

    # Voxelize + clean exactly as the pipeline does, so the 'voxels' figure matches the
    # tensor that is written to disk and fed to the solver.
    phase = cpm.voxelize_spheres_to_phase(
        sphere_centers=filtered_points,
        sphere_radii=np.full(len(filtered_points), model1.radius, dtype=float),
        box_size=box_size,
        voxel_grid_shape=tuple(args.voxel_grid),
    )
    if args.closing_radius > 0:
        phase, _ = cpm.close_thin_channels_3d(
            phase, closing_radius=args.closing_radius, periodic=args.occlusion_periodic
        )
    if args.fill_occlusions:
        phase, _ = cpm.fill_enclosed_occlusions_3d(
            phase, max_voxels=args.max_occlusion_voxels, periodic=args.occlusion_periodic
        )

    return {
        "box_size": box_size,
        "model1_points": model1_points,
        "model2_particles": model2_particles,
        "model3_particles": model3_particles,
        "inside2_points": inside2_points,
        "final_points": filtered_points,
        "sphere_radius": model1.radius,
        "phase": phase,
        "n_removed_floaters": n_removed_floaters,
    }


def spheres_from_points(points, radius):
    """One pv.Sphere per center (same meshing the pipeline uses for its particles)."""
    return [pv.Sphere(radius=radius, center=center) for center in points]


def phase_to_mesh(phase, box_size):
    """Surface mesh of the filler voxels of a phase tensor."""
    nx, ny, nz = phase.shape
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(box_size / nx, box_size / ny, box_size / nz),
        origin=(0.0, 0.0, 0.0),
    )
    grid.cell_data["phase"] = phase.ravel(order="F")
    return grid.threshold(0.5, scalars="phase")


def clip_to_box(meshes, box_size):
    clipped = []
    for mesh in meshes:
        out = mesh.clip_box(bounds=[0, box_size, 0, box_size, 0, box_size], invert=False)
        if out.n_points > 0:
            clipped.append(out)
    return clipped


def stage_layers(stage, geom, args):
    """Return (list of (mesh, color, opacity), caption) for one stage."""
    radius = geom["sphere_radius"]
    box_size = geom["box_size"]

    if stage == "model1":
        meshes = spheres_from_points(geom["model1_points"], radius)
        return (
            [(meshes, COLOR_MODEL1, 1.0)],
            "model 1: {} filler spheres, r = {:g}".format(len(meshes), radius),
        )
    if stage == "model2":
        meshes = geom["model2_particles"]
        return (
            [(meshes, COLOR_MODEL2, args.union_opacity)],
            "model 2: {} inclusion spheroids".format(len(meshes)),
        )
    if stage == "model3":
        meshes = geom["model3_particles"]
        return (
            [(meshes, COLOR_MODEL3, args.union_opacity)],
            "model 3: {} exclusion spheres, r = {:g}".format(len(meshes), args.model3_radius),
        )
    if stage == "overlay":
        return (
            [
                (geom["model2_particles"], COLOR_MODEL2, args.union_opacity),
                (geom["model3_particles"], COLOR_MODEL3, args.union_opacity),
            ],
            "model 2 ({}) + model 3 ({})".format(
                len(geom["model2_particles"]), len(geom["model3_particles"])
            ),
        )
    if stage == "inside2":
        meshes = spheres_from_points(geom["inside2_points"], radius)
        layers = [(meshes, COLOR_MODEL1, 1.0)]
        if args.show_context:
            layers.append((geom["model2_particles"], COLOR_MODEL2, args.context_opacity))
        return (
            layers,
            "model 1 inside model 2: {} of {} spheres".format(
                len(meshes), len(geom["model1_points"])
            ),
        )
    if stage == "final":
        meshes = spheres_from_points(geom["final_points"], radius)
        layers = [(meshes, COLOR_MODEL1, 1.0)]
        if args.show_context:
            layers.append((geom["model2_particles"], COLOR_MODEL2, args.context_opacity))
            layers.append((geom["model3_particles"], COLOR_MODEL3, args.context_opacity))
        return (
            layers,
            "final: {} spheres ({} floaters removed)".format(
                len(meshes), geom["n_removed_floaters"]
            ),
        )
    if stage == "voxels":
        mesh = phase_to_mesh(geom["phase"], box_size)
        fraction = float(np.count_nonzero(geom["phase"])) / geom["phase"].size
        return (
            [([mesh], COLOR_MODEL1, 1.0)],
            "voxelized {}: filler fraction {:.3f}".format(
                "x".join(str(n) for n in geom["phase"].shape), fraction
            ),
        )
    raise ValueError("unknown stage: {}".format(stage))


def render_stage(stage, geom, args, cpos):
    """Render one stage, save its PNG, and return the camera position used."""
    layers, caption = stage_layers(stage, geom, args)
    box_size = geom["box_size"]

    n_meshes = sum(len(meshes) for meshes, _, _ in layers)
    print("[{}] {}".format(stage, caption))
    if n_meshes == 0:
        print("[{}] nothing to draw -- skipped (intensity 0?)".format(stage))
        return cpos

    pv.global_theme.anti_aliasing = "ssaa"
    plotter = pv.Plotter(off_screen=args.off_screen, window_size=list(args.window_size))
    plotter.set_background(args.background)

    for meshes, color, opacity in layers:
        if args.clip:
            meshes = clip_to_box(meshes, box_size)
        for mesh in meshes:
            plotter.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True)

    if args.show_box:
        box = pv.Box([0, box_size, 0, box_size, 0, box_size])
        plotter.add_mesh(box, style="wireframe", color="black", line_width=2)

    if args.show_caption:
        plotter.add_text(caption, font_size=10, color="black")

    plotter.camera_position = cpos if cpos is not None else "iso"

    out_path = os.path.join(args.out_dir, "{}_{}.png".format(args.figure_prefix, stage))
    if args.off_screen:
        plotter.screenshot(out_path, transparent_background=args.transparent)
        used_cpos = plotter.camera_position
        plotter.close()
    else:
        # auto_close=False keeps the render window alive after interaction so the
        # screenshot uses the camera the user left it at.
        used_cpos = plotter.show(auto_close=False, return_cpos=True)
        plotter.screenshot(out_path, transparent_background=args.transparent)
        plotter.close()

    print("[{}] saved {}".format(stage, out_path))
    return used_cpos


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="Which stages to render, in order.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Folder for the PNGs (relative to repo root or absolute).")
    p.add_argument("--figure-prefix", default="boolean_model", help="Filename prefix: <prefix>_<stage>.png")

    # Geometry: same defaults as the combined pipeline.
    p.add_argument("--box-size", type=float, default=cpm.DEFAULT_BOX_SIZE)
    p.add_argument("--model1-intensity", type=float, default=cpm.DEFAULT_MODEL1_INTENSITY)
    p.add_argument("--model1-radius", type=float, default=cpm.DEFAULT_MODEL1_RADIUS)
    p.add_argument("--model2-intensity", type=float, default=cpm.DEFAULT_MODEL2_INTENSITY)
    p.add_argument("--model2-min-r1", type=float, default=cpm.DEFAULT_MODEL2_MIN_R1)
    p.add_argument("--model2-max-r1", type=float, default=cpm.DEFAULT_MODEL2_MAX_R1)
    p.add_argument("--model3-intensity", type=float, default=cpm.DEFAULT_MODEL3_INTENSITY)
    p.add_argument("--model3-radius", type=float, default=cpm.DEFAULT_MODEL3_RADIUS)
    p.add_argument("--seed", type=int, default=cpm.DEFAULT_SEED)
    p.add_argument("--voxel-grid", nargs=3, type=int, default=list(cpm.DEFAULT_VOXEL_GRID_SHAPE), metavar=("NX", "NY", "NZ"))

    # Cleanup: same defaults as the pipeline, so 'final'/'voxels' match a real run.
    p.add_argument("--no-remove-floating-clusters", dest="remove_floating_clusters", action="store_false", default=cpm.DEFAULT_REMOVE_FLOATING_CLUSTERS)
    p.add_argument("--min-cluster-size", type=int, default=cpm.DEFAULT_MIN_CLUSTER_SIZE)
    p.add_argument("--closing-radius", type=int, default=cpm.DEFAULT_CLOSING_RADIUS)
    p.add_argument("--no-fill-occlusions", dest="fill_occlusions", action="store_false", default=cpm.DEFAULT_FILL_OCCLUSIONS)
    p.add_argument("--max-occlusion-voxels", type=int, default=cpm.DEFAULT_MAX_OCCLUSION_VOXELS)
    p.add_argument("--occlusion-non-periodic", dest="occlusion_periodic", action="store_false", default=cpm.DEFAULT_OCCLUSION_PERIODIC)

    # Rendering / figure style.
    p.add_argument("--off-screen", action="store_true", help="Render without opening windows (fixed iso view unless --camera-file is given).")
    p.add_argument("--link-camera", action="store_true", help="Reuse the camera of the first rendered stage for all later stages.")
    p.add_argument("--camera-file", default=None, help="JSON file to load the camera from (if it exists) and save it to after the first stage.")
    p.add_argument("--window-size", nargs=2, type=int, default=[1600, 1600], metavar=("W", "H"))
    p.add_argument("--background", default="white")
    p.add_argument("--transparent", action="store_true", help="Save PNGs with a transparent background.")
    p.add_argument("--no-box", dest="show_box", action="store_false", default=True, help="Hide the domain wireframe.")
    p.add_argument("--caption", dest="show_caption", action="store_true", default=False, help="Burn a caption into the image (off by default for paper figures).")
    p.add_argument("--union-opacity", type=float, default=1.0, help="Opacity of the model-2/model-3 bodies when they are the subject.")
    p.add_argument("--show-context", action="store_true", help="In the 'inside2'/'final' stages, also draw the model-2/3 bodies faintly.")
    p.add_argument("--context-opacity", type=float, default=0.15)
    p.add_argument("--clip", action="store_true", help="Clip every body to the domain box.")
    return p


def main(cli_args=None):
    args = _build_parser().parse_args(cli_args)

    args.out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(cpm.REPO_ROOT, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    cpos = None
    if args.camera_file and os.path.exists(args.camera_file):
        with open(args.camera_file, "r") as fh:
            cpos = json.load(fh)
        print("Camera loaded from {}".format(args.camera_file))

    geom = build_stage_geometry(args)
    print(
        "Generated: {} model-1 points, {} spheroids, {} exclusion spheres, {} final spheres".format(
            len(geom["model1_points"]),
            len(geom["model2_particles"]),
            len(geom["model3_particles"]),
            len(geom["final_points"]),
        )
    )
    if args.model3_intensity == 0:
        print("Note: model3 intensity is 0, so the 'model3' stage is empty. Pass --model3-intensity to show it.")

    for stage in args.stages:
        used_cpos = render_stage(stage, geom, args, cpos)
        if (args.link_camera or args.camera_file) and cpos is None and used_cpos is not None:
            cpos = [list(v) for v in used_cpos]
            if args.camera_file:
                with open(args.camera_file, "w") as fh:
                    json.dump(cpos, fh, indent=2)
                print("Camera saved to {}".format(args.camera_file))

    print("\nFigures written to {}".format(args.out_dir))


if __name__ == "__main__":
    main()
