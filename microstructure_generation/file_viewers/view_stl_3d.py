#!/usr/bin/env python
"""
View STL meshes in 3D with optional save-to-image.

Fast default behavior:
- backend auto-selects pyvista (VTK/GPU) when available
- optional decimation for very large meshes
- matplotlib fallback when pyvista is unavailable


RUN:
python view_stl_3d.py --input path_of_input_file.stl
"""

import argparse
import os
import numpy as np


def set_equal_axes(ax, vertices):
	xs = vertices[:, 0]
	ys = vertices[:, 1]
	zs = vertices[:, 2]
	xmin, xmax = np.min(xs), np.max(xs)
	ymin, ymax = np.min(ys), np.max(ys)
	zmin, zmax = np.min(zs), np.max(zs)
	max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
	x_mid = 0.5 * (xmin + xmax)
	y_mid = 0.5 * (ymin + ymax)
	z_mid = 0.5 * (zmin + zmax)
	ax.set_xlim(x_mid - 0.5 * max_range, x_mid + 0.5 * max_range)
	ax.set_ylim(y_mid - 0.5 * max_range, y_mid + 0.5 * max_range)
	ax.set_zlim(z_mid - 0.5 * max_range, z_mid + 0.5 * max_range)


def render_with_pyvista(path, args):
	try:
		import pyvista as pv
	except Exception as exc:
		raise RuntimeError("pyvista backend requested but unavailable: %s" % exc)

	mesh = pv.read(path)
	if mesh is None or mesh.n_points == 0:
		raise RuntimeError("Input STL is empty or unreadable: %s" % path)

	if args.decimate and args.decimate > 0.0:
		if args.decimate >= 1.0:
			raise ValueError("--decimate must be in [0,1).")
		mesh = mesh.decimate(float(args.decimate))

	plotter = pv.Plotter(window_size=(1100, 820), off_screen=bool(args.no_show or args.save))
	plotter.set_background("white")

	if args.mode == 'points':
		plotter.add_mesh(mesh.points, color='royalblue', point_size=max(1.0, args.point_size), render_points_as_spheres=True)
	elif args.mode == 'wireframe':
		plotter.add_mesh(mesh, color='royalblue', style='wireframe', line_width=max(0.1, args.linewidth))
	else:
		plotter.add_mesh(mesh, color=[0.2, 0.4, 0.9], opacity=args.alpha, smooth_shading=not args.no_smooth, show_edges=False)

	plotter.add_axes()
	plotter.add_bounding_box()
	title = "STL (pyvista): %s | points=%d cells=%d" % (os.path.basename(path), mesh.n_points, mesh.n_cells)
	plotter.add_text(title, position='upper_left', font_size=10)
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

	print("Input STL: %s" % path)
	print("points: %d" % mesh.n_points)
	print("cells: %d" % mesh.n_cells)


def render_with_matplotlib(path, args):
	try:
		import trimesh
	except Exception as exc:
		raise RuntimeError(
			"matplotlib fallback requires trimesh for STL loading. Install with `pip install trimesh`. Error: %s" % exc
		)
	try:
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		import matplotlib.pyplot as plt
	except Exception as exc:
		raise RuntimeError(
			"matplotlib backend unavailable. Install with `pip install matplotlib`. Error: %s" % exc
		)

	mesh = trimesh.load_mesh(path, process=False)
	if mesh is None or mesh.is_empty:
		raise RuntimeError("Input STL is empty or unreadable: %s" % path)
	vertices = np.asarray(mesh.vertices, dtype=float)
	faces = np.asarray(mesh.faces, dtype=int)
	if faces.size == 0:
		raise RuntimeError("No faces found in STL: %s" % path)

	if args.decimate and args.decimate > 0.0:
		# Coarse simplification fallback for matplotlib path by random face subset.
		# Keeps speed reasonable when trimesh simplifier backend is not available.
		keep = int(max(1, (1.0 - args.decimate) * faces.shape[0]))
		rng = np.random.default_rng(args.seed)
		ix = rng.choice(faces.shape[0], size=keep, replace=False)
		faces = faces[ix]

	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')

	if args.mode == 'points':
		ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=args.point_size, c='royalblue', alpha=args.alpha, linewidths=0.0)
	elif args.mode == 'wireframe':
		triangles = vertices[faces]
		collection = Poly3DCollection(triangles, facecolors=(0, 0, 0, 0), edgecolors='royalblue', linewidths=args.linewidth)
		ax.add_collection3d(collection)
	else:
		triangles = vertices[faces]
		collection = Poly3DCollection(
			triangles,
			facecolors=(0.2, 0.4, 0.9, args.alpha),
			edgecolors=(0.1, 0.1, 0.1, min(1.0, args.alpha + 0.05)),
			linewidths=args.linewidth
		)
		ax.add_collection3d(collection)

	set_equal_axes(ax, vertices)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.set_title("STL (mpl): %s | vertices=%d faces=%d" % (os.path.basename(path), len(vertices), len(faces)))

	if args.save:
		save_path = os.path.abspath(args.save)
		save_dir = os.path.dirname(save_path)
		if save_dir and not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		plt.tight_layout()
		plt.savefig(save_path, dpi=args.dpi)
		print("Saved figure: %s" % save_path)

	if not args.no_show:
		plt.show()
	else:
		plt.close(fig)

	print("Input STL: %s" % path)
	print("vertices: %d" % len(vertices))
	print("faces used: %d" % len(faces))


def run(args):
	path = os.path.abspath(args.input)
	if not os.path.isfile(path):
		raise IOError("STL not found: %s" % path)

	backend = args.backend
	if backend == 'auto':
		try:
			import pyvista  # noqa: F401
			backend = 'pyvista'
		except Exception:
			backend = 'matplotlib'
	print("Viewer backend: %s" % backend)

	if backend == 'pyvista':
		render_with_pyvista(path, args)
	else:
		render_with_matplotlib(path, args)


def build_parser():
	p = argparse.ArgumentParser(description="3D viewer for STL files.")
	p.add_argument('--input', required=True, help='Input STL file path.')
	p.add_argument('--backend', choices=['auto', 'pyvista', 'matplotlib'], default='auto',
				   help='auto chooses pyvista if available, otherwise matplotlib fallback.')
	p.add_argument('--mode', choices=['surface', 'wireframe', 'points'], default='surface',
				   help='surface=filled mesh, wireframe=edges only, points=vertex cloud.')
	p.add_argument('--alpha', type=float, default=0.9, help='Surface/points opacity (0..1).')
	p.add_argument('--linewidth', type=float, default=0.2, help='Edge line width for surface/wireframe.')
	p.add_argument('--point-size', type=float, default=1.5, help='Point size in points mode.')
	p.add_argument('--decimate', type=float, default=0.0,
				   help='Optional mesh reduction fraction in [0,1). Example: 0.6 keeps ~40%% geometry.')
	p.add_argument('--zoom', type=float, default=1.0, help='Camera zoom factor for pyvista backend.')
	p.add_argument('--no-smooth', action='store_true', help='Disable smooth shading in surface mode (pyvista).')
	p.add_argument('--seed', type=int, default=0, help='Seed for matplotlib fallback decimation sampling.')
	p.add_argument('--save', default=None, help='Optional path to save rendered figure.')
	p.add_argument('--dpi', type=int, default=180, help='DPI for saved figure.')
	p.add_argument('--no-show', action='store_true', help='Do not open interactive window.')
	return p


if __name__ == '__main__':
	parser = build_parser()
	run(parser.parse_args())
