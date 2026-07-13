#!/usr/bin/env python
"""
View voxel NPZ files in 3D.

Fast default behavior:
- backend auto-selects pyvista (VTK/GPU) when available
- mode defaults to surface for smooth interaction
- auto-downsampling keeps view responsive on large volumes

RUN:
python view_npz_3d.py --input path_of_input_file.npz
"""

import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)


def load_phase_from_npz(path, phase_key='phase', invert_phase=False):
	data = np.load(path, allow_pickle=True)
	if phase_key not in data.files:
		raise KeyError("Missing phase key '%s' in %s" % (phase_key, path))
	phase = np.asarray(data[phase_key])
	if invert_phase:
		phase = np.logical_not(phase.astype(bool)).astype(np.uint8)
	meta = {}
	for key in data.files:
		if key != phase_key:
			meta[key] = data[key]
	return phase, meta


def load_npz_content(path, phase_key='phase'):
	data = np.load(path, allow_pickle=True)
	keys = set(data.files)

	if phase_key in keys:
		phase, meta = load_phase_from_npz(path, phase_key=phase_key, invert_phase=False)
		return "phase", phase, meta

	if "centers" in keys and "radii" in keys:
		centers = np.asarray(data["centers"], dtype=float)
		radii = np.asarray(data["radii"], dtype=float).reshape(-1)
		if centers.ndim != 2 or centers.shape[1] != 3:
			raise ValueError("centers must have shape (N, 3), got %s" % (centers.shape,))
		if centers.shape[0] != radii.shape[0]:
			raise ValueError("centers and radii length mismatch: %d vs %d" % (centers.shape[0], radii.shape[0]))
		meta = {}
		for key in data.files:
			if key in ("centers", "radii"):
				continue
			meta[key] = data[key]
		return "spheres", (centers, radii), meta

	raise KeyError(
		"Unsupported NPZ format for viewer. Need either '%s' key or ['centers','radii']. Available keys: %s"
		% (phase_key, sorted(list(data.files)))
	)


def downsample_nearest(arr, step):
	if step <= 1:
		return arr
	return arr[::step, ::step, ::step]


def choose_auto_downsample(phase, max_filler_cells):
	if max_filler_cells <= 0:
		return 1
	n_fill = int(np.count_nonzero(phase))
	if n_fill <= max_filler_cells:
		return 1
	ratio = float(n_fill) / float(max_filler_cells)
	step = int(np.ceil(ratio ** (1.0 / 3.0)))
	return max(1, step)


def _meta_vector(meta, key, length, default):
	value = meta.get(key, default)
	arr = np.asarray(value, dtype=float).reshape(-1)
	if arr.size == 1 and length > 1:
		arr = np.full(length, arr.item(), dtype=float)
	if arr.size != length:
		return np.asarray(default, dtype=float)
	return arr


def phase_domain_from_meta(phase, meta, step=1):
	origin = _meta_vector(meta, 'origin', 3, np.zeros(3, dtype=float))
	voxel_size = _meta_vector(meta, 'voxel_size', 3, np.ones(3, dtype=float))
	step = max(1, int(step))
	view_voxel_size = voxel_size * float(step)
	view_shape = np.asarray(phase.shape, dtype=int)
	max_corner = origin + view_voxel_size * view_shape.astype(float)
	return {
		'origin': origin,
		'voxel_size': voxel_size,
		'view_voxel_size': view_voxel_size,
		'bounds': (
			float(origin[0]), float(max_corner[0]),
			float(origin[1]), float(max_corner[1]),
			float(origin[2]), float(max_corner[2]),
		),
	}


def sphere_domain_from_meta(centers, radii, meta):
	origin = _meta_vector(meta, 'origin', 3, np.zeros(3, dtype=float))
	box_size_raw = np.asarray(meta.get('box_size', [0.0]), dtype=float).reshape(-1)
	if box_size_raw.size >= 1 and box_size_raw[0] > 0.0:
		box_size = float(box_size_raw[0])
		max_corner = origin + box_size
		return {
			'origin': origin,
			'box_size': box_size,
			'bounds': (
				float(origin[0]), float(max_corner[0]),
				float(origin[1]), float(max_corner[1]),
				float(origin[2]), float(max_corner[2]),
			),
		}

	if centers.shape[0] == 0:
		return None
	mins = np.min(centers - radii[:, None], axis=0)
	maxs = np.max(centers + radii[:, None], axis=0)
	return {
		'origin': mins,
		'box_size': float(np.max(maxs - mins)),
		'bounds': (
			float(mins[0]), float(maxs[0]),
			float(mins[1]), float(maxs[1]),
			float(mins[2]), float(maxs[2]),
		),
	}


def set_axes_bounds(ax, bounds):
	xmin, xmax, ymin, ymax, zmin, zmax = bounds
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_zlim(zmin, zmax)
	ax.set_box_aspect((max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12), max(zmax - zmin, 1e-12)))


def set_equal_axes(ax, xs, ys, zs):
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


def render_scatter_mpl(ax, phase, domain, max_points, point_size, seed):
	coords = np.argwhere(phase > 0)
	n_all = coords.shape[0]
	if n_all == 0:
		set_axes_bounds(ax, domain['bounds'])
		return 0, 0
	if n_all > max_points:
		rng = np.random.default_rng(seed)
		ix = rng.choice(n_all, size=max_points, replace=False)
		coords = coords[ix]
	origin = domain['origin']
	spacing = domain['view_voxel_size']
	points = origin + (coords.astype(float) + 0.5) * spacing
	xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
	ax.scatter(xs, ys, zs, s=point_size, alpha=0.8, c='royalblue', marker='o', linewidths=0.0)
	set_axes_bounds(ax, domain['bounds'])
	return n_all, coords.shape[0]


def render_voxels_mpl(ax, phase, domain, alpha=0.95):
	mask = phase.astype(bool)
	if np.count_nonzero(mask) == 0:
		set_axes_bounds(ax, domain['bounds'])
		return 0
	facecolors = np.zeros(mask.shape + (4,), dtype=float)
	facecolors[mask] = np.array([0.2, 0.4, 0.9, alpha], dtype=float)
	origin = domain['origin']
	spacing = domain['view_voxel_size']
	x = origin[0] + np.arange(mask.shape[0] + 1, dtype=float) * spacing[0]
	y = origin[1] + np.arange(mask.shape[1] + 1, dtype=float) * spacing[1]
	z = origin[2] + np.arange(mask.shape[2] + 1, dtype=float) * spacing[2]
	X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
	ax.voxels(X, Y, Z, mask, facecolors=facecolors, edgecolor='none')
	set_axes_bounds(ax, domain['bounds'])
	return int(np.count_nonzero(mask))


def render_with_matplotlib(args, phase, meta, step):
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')
	domain = phase_domain_from_meta(phase, meta, step=step)

	if args.mode == 'scatter':
		n_all, n_used = render_scatter_mpl(ax, phase, domain, max_points=args.max_points, point_size=args.point_size, seed=args.seed)
		title = "3D NPZ scatter (mpl): all=%d used=%d shape=%s" % (n_all, n_used, phase.shape)
	elif args.mode == 'surface':
		n_used = render_voxels_mpl(ax, phase, domain, alpha=args.voxel_alpha)
		title = "3D NPZ surface fallback (mpl voxels): filled=%d shape=%s" % (n_used, phase.shape)
	else:
		n_used = render_voxels_mpl(ax, phase, domain, alpha=args.voxel_alpha)
		title = "3D NPZ voxels (mpl): filled=%d shape=%s" % (n_used, phase.shape)

	ax.set_title(title)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

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


def render_with_pyvista(args, phase, meta, step):
	try:
		import pyvista as pv
	except Exception as exc:
		raise RuntimeError("pyvista backend requested but unavailable: %s" % exc)

	filler = np.ascontiguousarray(phase.astype(np.uint8))
	domain = phase_domain_from_meta(phase, meta, step=step)
	dims = np.array(filler.shape, dtype=int) + 1
	grid = pv.ImageData(
		dimensions=tuple(dims),
		spacing=tuple(domain['view_voxel_size']),
		origin=tuple(domain['origin']),
	)
	grid.cell_data['phase'] = filler.ravel(order='F')

	n_fill = int(np.count_nonzero(filler))
	plotter = pv.Plotter(window_size=(1100, 820), off_screen=bool(args.no_show or args.save))
	plotter.set_background("white")

	if args.mode == 'scatter':
		coords = np.argwhere(filler > 0)
		n_all = coords.shape[0]
		n_used = n_all
		if n_all > args.max_points:
			rng = np.random.default_rng(args.seed)
			ix = rng.choice(n_all, size=args.max_points, replace=False)
			coords = coords[ix]
			n_used = coords.shape[0]
		points = domain['origin'] + (coords.astype(float) + 0.5) * domain['view_voxel_size']
		pts = pv.PolyData(points)
		plotter.add_mesh(pts, color='royalblue', point_size=max(1.0, args.point_size), render_points_as_spheres=True)
		title = "NPZ scatter (pyvista): all=%d used=%d shape=%s" % (n_all, n_used, phase.shape)
	elif args.mode == 'surface':
		cells = grid.threshold([0.5, 1.5], scalars='phase')
		surf = cells.extract_surface()
		plotter.add_mesh(surf, color=[0.2, 0.4, 0.9], opacity=args.voxel_alpha, smooth_shading=True)
		title = "NPZ surface (pyvista): filled=%d shape=%s" % (n_fill, phase.shape)
	else:
		cells = grid.threshold([0.5, 1.5], scalars='phase')
		plotter.add_mesh(cells, color=[0.2, 0.4, 0.9], opacity=args.voxel_alpha, show_edges=False)
		title = "NPZ cells (pyvista): filled=%d shape=%s" % (n_fill, phase.shape)

	full_box = pv.Box(bounds=domain['bounds'])
	plotter.add_mesh(full_box, style='wireframe', color='black', line_width=1.0)
	plotter.add_axes()
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


def render_spheres_with_matplotlib(args, centers, radii, meta):
	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(8, 7))
	ax = fig.add_subplot(111, projection='3d')
	domain = sphere_domain_from_meta(centers, radii, meta)

	n_all = centers.shape[0]
	n_used = n_all
	if n_all > args.max_points:
		rng = np.random.default_rng(args.seed)
		ix = rng.choice(n_all, size=args.max_points, replace=False)
		centers = centers[ix]
		radii = radii[ix]
		n_used = centers.shape[0]

	if n_used > 0:
		r_norm = radii / max(1e-12, np.max(radii))
		sizes = np.maximum(2.0, args.point_size * 20.0 * (r_norm ** 2))
		ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=sizes, c='royalblue', alpha=0.85, linewidths=0.0)
	if domain is not None:
		set_axes_bounds(ax, domain['bounds'])

	ax.set_title("3D NPZ spheres (mpl): all=%d used=%d" % (n_all, n_used))
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

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


def render_spheres_with_pyvista(args, centers, radii, meta):
	try:
		import pyvista as pv
	except Exception as exc:
		raise RuntimeError("pyvista backend requested but unavailable: %s" % exc)

	n_all = centers.shape[0]
	domain = sphere_domain_from_meta(centers, radii, meta)
	if n_all > args.max_points:
		rng = np.random.default_rng(args.seed)
		ix = rng.choice(n_all, size=args.max_points, replace=False)
		centers = centers[ix]
		radii = radii[ix]
	n_used = centers.shape[0]

	plotter = pv.Plotter(window_size=(1100, 820), off_screen=bool(args.no_show or args.save))
	plotter.set_background("white")

	if args.mode == 'scatter' or args.mode == 'voxels':
		pts = pv.PolyData(centers.astype(float))
		plotter.add_mesh(pts, color='royalblue', point_size=max(1.0, args.point_size), render_points_as_spheres=True)
		title = "NPZ spheres centers (pyvista): all=%d used=%d" % (n_all, n_used)
	else:
		pts = pv.PolyData(centers.astype(float))
		pts['radius'] = radii.astype(float)
		res = max(8, int(args.sphere_resolution))
		base_sphere = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res)
		glyphs = pts.glyph(scale='radius', geom=base_sphere, orient=False)
		plotter.add_mesh(glyphs, color=[0.2, 0.4, 0.9], opacity=args.voxel_alpha, smooth_shading=True, show_edges=False)
		title = "NPZ spheres (pyvista): all=%d used=%d" % (n_all, n_used)

	if domain is not None:
		full_box = pv.Box(bounds=domain['bounds'])
		plotter.add_mesh(full_box, style='wireframe', color='black', line_width=1.0)
	plotter.add_axes()
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


def run(args):
	input_path = os.path.abspath(args.input)
	content_kind, content, meta = load_npz_content(input_path, phase_key=args.phase_key)

	backend = args.backend
	if backend == 'auto':
		try:
			import pyvista  # noqa: F401
			backend = 'pyvista'
		except Exception:
			backend = 'matplotlib'
	print("Viewer backend: %s" % backend)

	if content_kind == "phase":
		phase = content
		user_step = max(1, args.downsample)
		auto_step = choose_auto_downsample(phase, max_filler_cells=args.max_filler_cells)
		step = max(user_step, auto_step)
		phase = downsample_nearest(phase, step)
		if step > 1:
			print("Applied downsample step %d for display speed." % step)

		if backend == 'pyvista':
			render_with_pyvista(args, phase, meta, step)
		else:
			render_with_matplotlib(args, phase, meta, step)

		print("Input: %s" % input_path)
		print("NPZ type: voxel phase")
		print("phase shape used for view: %s" % (phase.shape,))
		print("filler fraction: %.6f" % float(np.mean(phase)))
		if 'voxel_size' in meta:
			print("voxel_size metadata: %s" % np.asarray(meta['voxel_size']))
		return

	centers, radii = content
	if backend == 'pyvista':
		render_spheres_with_pyvista(args, centers, radii, meta)
	else:
		render_spheres_with_matplotlib(args, centers, radii, meta)

	print("Input: %s" % input_path)
	print("NPZ type: sphere parameters")
	print("num spheres: %d" % centers.shape[0])
	print("radius range: [%.6g, %.6g]" % (float(np.min(radii)), float(np.max(radii))))


def view_npz_file(
	input_path,
	phase_key='phase',
	backend='auto',
	mode='surface',
	downsample=1,
	max_filler_cells=250000,
	max_points=120000,
	point_size=2.0,
	sphere_resolution=16,
	voxel_alpha=1.00,
	seed=0,
	zoom=1.0,
	save=None,
	dpi=180,
	no_show=False,
):
	args = argparse.Namespace(
		input=input_path,
		phase_key=phase_key,
		backend=backend,
		mode=mode,
		downsample=downsample,
		max_filler_cells=max_filler_cells,
		max_points=max_points,
		point_size=point_size,
		sphere_resolution=sphere_resolution,
		voxel_alpha=voxel_alpha,
		seed=seed,
		zoom=zoom,
		save=save,
		dpi=dpi,
		no_show=no_show,
	)
	run(args)


def build_parser():
	p = argparse.ArgumentParser(description="3D viewer for NPZ files: voxel phase or sphere parameters.")
	p.add_argument('--input', required=True,
				   help='Input NPZ file, typically from Structures/3D_structures/Spheres or Structures/3D_structures/Voxolized.')
	p.add_argument('--phase-key', default='phase',
				   help='Key for phase tensor in npz (default: phase).')
	p.add_argument('--backend', choices=['auto', 'pyvista', 'matplotlib'], default='auto',
				   help='auto chooses pyvista if available, otherwise matplotlib.')
	p.add_argument('--mode', choices=['scatter', 'surface', 'voxels'], default='surface',
				   help='surface is fast/smooth with pyvista; voxels enforce touching cells; scatter is preview.')
	p.add_argument('--downsample', type=int, default=1,
				   help='Manual downsample factor (>=1).')
	p.add_argument('--max-filler-cells', type=int, default=250000,
				   help='Auto-downsample target for filled cells to keep interaction responsive.')
	p.add_argument('--max-points', type=int, default=120000,
				   help='Max points used in scatter mode.')
	p.add_argument('--point-size', type=float, default=2.0,
				   help='Point size in scatter mode.')
	p.add_argument('--sphere-resolution', type=int, default=16,
				   help='Sphere mesh resolution for sphere-parameter NPZ (pyvista, surface mode).')
	p.add_argument('--voxel-alpha', type=float, default=0.95,
				   help='Opacity in surface/voxel mode (0..1).')
	p.add_argument('--seed', type=int, default=0,
				   help='Random seed for scatter downsampling.')
	p.add_argument('--zoom', type=float, default=1.0,
				   help='Camera zoom factor for pyvista backend.')
	p.add_argument('--save', default=None,
				   help='Optional path to save rendered figure.')
	p.add_argument('--dpi', type=int, default=180,
				   help='DPI for saved figure (matplotlib backend).')
	p.add_argument('--no-show', action='store_true',
				   help='Do not open interactive window.')
	return p


if __name__ == '__main__':
	parser = build_parser()
	run(parser.parse_args())
