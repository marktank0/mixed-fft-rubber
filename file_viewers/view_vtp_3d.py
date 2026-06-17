#!/usr/bin/env python
"""
View VTP meshes in 3D.

Supports:
- single .vtp file
- folder with many .vtp files
- saved structure format from microstructure_generation:
  <base>_particles/particle_*.vtp + <base>_union2.vtp + <base>_union3.vtp

Fast defaults:
- auto backend (pyvista)
- particle autosampling for large folders
- optional decimation for heavy meshes

RUN:
python view_vtl_3d.py --input path_of_input_file.vtp
"""

import argparse
import glob
import os
import numpy as np


def categorize_vtp(path):
	name = os.path.basename(path).lower()
	parent = os.path.basename(os.path.dirname(path)).lower()
	if "union2" in name:
		return "union2"
	if "union3" in name:
		return "union3"
	if name.startswith("particle_") or parent.endswith("_particles"):
		return "particle"
	return "other"


def discover_structure_files(root_dir, base_name):
	root_dir = os.path.abspath(root_dir)
	files = []
	union2 = os.path.join(root_dir, "%s_union2.vtp" % base_name)
	union3 = os.path.join(root_dir, "%s_union3.vtp" % base_name)
	particles_dir = os.path.join(root_dir, "%s_particles" % base_name)

	if os.path.isfile(union2):
		files.append(union2)
	if os.path.isfile(union3):
		files.append(union3)
	if os.path.isdir(particles_dir):
		files.extend(sorted(glob.glob(os.path.join(particles_dir, "particle_*.vtp"))))
	return files


def discover_generic_files(input_path, pattern, recursive):
	input_path = os.path.abspath(input_path)
	if os.path.isfile(input_path):
		if input_path.lower().endswith(".vtp"):
			return [input_path]
		return []
	if not os.path.isdir(input_path):
		return []
	search = os.path.join(input_path, "**", pattern) if recursive else os.path.join(input_path, pattern)
	files = glob.glob(search, recursive=recursive)
	return sorted([os.path.abspath(p) for p in files if os.path.isfile(p) and p.lower().endswith(".vtp")])


def split_files_by_type(files):
	typed = {"particle": [], "union2": [], "union3": [], "other": []}
	for path in files:
		typed[categorize_vtp(path)].append(path)
	return typed


def choose_particle_subset(particles, max_particles, seed):
	if max_particles <= 0 or len(particles) <= max_particles:
		return particles, 1.0
	rng = np.random.default_rng(seed)
	ix = rng.choice(len(particles), size=max_particles, replace=False)
	ix = np.sort(ix)
	subset = [particles[i] for i in ix]
	ratio = float(max_particles) / float(len(particles))
	return subset, ratio


def select_files_for_view(typed_files, args):
	selected = []

	if not args.hide_unions:
		selected.extend(typed_files["union2"])
		selected.extend(typed_files["union3"])

	if not args.hide_other:
		selected.extend(typed_files["other"])

	particle_files = [] if args.hide_particles else typed_files["particle"]
	if args.all_particles:
		selected.extend(particle_files)
		kept_ratio = 1.0
	else:
		subset, kept_ratio = choose_particle_subset(
			particle_files,
			max_particles=args.max_particles,
			seed=args.seed,
		)
		selected.extend(subset)

	return selected, len(particle_files), kept_ratio


def load_mesh(path, decimate):
	import pyvista as pv

	mesh = pv.read(path)
	if mesh is None:
		return None

	if isinstance(mesh, pv.MultiBlock):
		mesh = mesh.combine()
		if mesh is None:
			return None
	mesh = mesh.triangulate()

	if decimate > 0.0 and mesh.n_cells > 0:
		mesh = mesh.decimate(float(decimate))
	return mesh


def add_mesh_to_plot(plotter, mesh, mesh_type, args):
	colors = {
		"particle": [0.92, 0.92, 0.92],
		"union2": [0.15, 0.45, 0.95],
		"union3": [0.15, 0.75, 0.25],
		"other": [0.75, 0.40, 0.80],
	}
	color = colors.get(mesh_type, [0.7, 0.7, 0.7])

	if args.mode == "points":
		plotter.add_mesh(
			mesh.points,
			color=color,
			point_size=max(1.0, args.point_size),
			render_points_as_spheres=True,
		)
	elif args.mode == "wireframe":
		plotter.add_mesh(mesh, color=color, style="wireframe", line_width=max(0.1, args.linewidth))
	else:
		plotter.add_mesh(
			mesh,
			color=color,
			opacity=args.alpha,
			smooth_shading=not args.no_smooth,
			show_edges=False,
		)


def render_with_pyvista(selected_files, args):
	import pyvista as pv

	plotter = pv.Plotter(window_size=(1200, 860), off_screen=bool(args.no_show or args.save))
	plotter.set_background("white")

	total_points = 0
	total_cells = 0
	type_counts = {"particle": 0, "union2": 0, "union3": 0, "other": 0}
	centers = []

	for i, path in enumerate(selected_files):
		mesh = load_mesh(path, decimate=args.decimate)
		if mesh is None or mesh.n_points == 0:
			continue

		mesh_type = categorize_vtp(path)
		type_counts[mesh_type] += 1
		total_points += int(mesh.n_points)
		total_cells += int(mesh.n_cells)

		if args.mode == "centers" and mesh_type == "particle":
			centers.append(np.asarray(mesh.center, dtype=float))
		else:
			add_mesh_to_plot(plotter, mesh, mesh_type, args)

		if (i + 1) % 200 == 0:
			print("Loaded %d/%d files..." % (i + 1, len(selected_files)))

	if args.mode == "centers" and centers:
		pts = pv.PolyData(np.asarray(centers, dtype=float))
		plotter.add_mesh(
			pts,
			color=[0.1, 0.1, 0.1],
			point_size=max(2.0, args.point_size),
			render_points_as_spheres=True,
		)

	if args.show_bounds:
		plotter.add_bounding_box()
	if args.show_axes:
		plotter.add_axes()

	title = (
		"VTP viewer: files=%d | points=%d | cells=%d | "
		"particles=%d union2=%d union3=%d other=%d"
	) % (
		len(selected_files),
		total_points,
		total_cells,
		type_counts["particle"],
		type_counts["union2"],
		type_counts["union3"],
		type_counts["other"],
	)
	plotter.add_text(title, position="upper_left", font_size=10)

	if args.zoom > 0:
		plotter.camera.zoom(args.zoom)

	if args.save:
		save_path = os.path.abspath(args.save)
		save_dir = os.path.dirname(save_path)
		if save_dir and not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		plotter.show(auto_close=False)
		plotter.screenshot(save_path)
		plotter.close()
		print("Saved figure: %s" % save_path)
	elif args.no_show:
		plotter.close()
	else:
		plotter.show()


def run(args):
	if args.backend not in ("auto", "pyvista"):
		raise ValueError("Only pyvista backend is supported for VTP viewing.")

	if args.structure_base:
		files = discover_structure_files(args.input, args.structure_base)
	else:
		files = discover_generic_files(args.input, pattern=args.pattern, recursive=args.recursive)

	if not files:
		raise RuntimeError("No VTP files found. Check --input, --pattern, or --structure-base.")

	typed = split_files_by_type(files)
	selected_files, n_particles, kept_ratio = select_files_for_view(typed, args)
	if not selected_files:
		raise RuntimeError("No VTP files selected after filtering flags.")

	print("Found VTP files: %d" % len(files))
	print(
		"Selected for view: %d (particle source=%d, kept_ratio=%.3f)"
		% (len(selected_files), n_particles, kept_ratio)
	)
	print("Viewer backend: pyvista")

	render_with_pyvista(selected_files, args)


def build_parser():
	p = argparse.ArgumentParser(description="3D viewer for VTP mesh files.")
	p.add_argument(
		"--input",
		required=True,
		help="Input VTP file, VTP folder, or structure root folder.",
	)
	p.add_argument(
		"--structure-base",
		default="",
		help="Optional structure basename. Loads <base>_particles + <base>_union2/3 from --input folder.",
	)
	p.add_argument(
		"--pattern",
		default="*.vtp",
		help="Glob pattern when --input is a folder and --structure-base is not used.",
	)
	p.add_argument(
		"--recursive",
		action="store_true",
		help="Recursively search folder for pattern matches.",
	)
	p.add_argument(
		"--backend",
		choices=["auto", "pyvista"],
		default="auto",
		help="auto currently selects pyvista.",
	)
	p.add_argument(
		"--mode",
		choices=["surface", "wireframe", "points", "centers"],
		default="surface",
		help="centers draws only particle centers (fastest for huge particle sets).",
	)
	p.add_argument("--alpha", type=float, default=0.9, help="Opacity in surface mode.")
	p.add_argument("--linewidth", type=float, default=0.2, help="Line width in wireframe mode.")
	p.add_argument("--point-size", type=float, default=3.0, help="Point size for points/centers mode.")
	p.add_argument(
		"--decimate",
		type=float,
		default=0.0,
		help="Optional mesh reduction fraction in [0,1). Example: 0.7 keeps ~30%% geometry.",
	)
	p.add_argument(
		"--max-particles",
		type=int,
		default=600,
		help="Max particle files to load unless --all-particles is set.",
	)
	p.add_argument(
		"--all-particles",
		action="store_true",
		help="Load all particle VTP files (can be slow on large structures).",
	)
	p.add_argument("--seed", type=int, default=0, help="Seed for random particle sampling.")
	p.add_argument("--hide-particles", action="store_true", help="Do not show particle VTP files.")
	p.add_argument("--hide-unions", action="store_true", help="Do not show union2/union3 files.")
	p.add_argument("--hide-other", action="store_true", help="Do not show uncategorized VTP files.")
	p.add_argument("--zoom", type=float, default=1.0, help="Camera zoom factor.")
	p.add_argument("--no-smooth", action="store_true", help="Disable smooth shading in surface mode.")
	p.add_argument("--show-axes", action="store_true", help="Show orientation axes widget.")
	p.add_argument("--show-bounds", action="store_true", help="Show bounding box.")
	p.add_argument("--save", default=None, help="Optional screenshot output path (.png).")
	p.add_argument("--no-show", action="store_true", help="Do not open interactive window.")
	return p


if __name__ == "__main__":
	parser = build_parser()
	run(parser.parse_args())
