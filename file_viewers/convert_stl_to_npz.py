#!/usr/bin/env python
"""
Convenience converter for STL -> NPZ in the structured 3D workspace.
"""

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
	sys.path.insert(0, REPO_ROOT)

def ensure_dir(path):
	if path and not os.path.isdir(path):
		os.makedirs(path)


def load_converter():
	try:
		from microstructure3D.convert_stl_to_voxel_npz import convert as convert_single_stl
	except ImportError as exc:
		raise RuntimeError(
			"STL conversion backend is not available in the current microstructure3D package."
		) from exc
	return convert_single_stl


def list_stl_files(input_path, pattern='*.stl', recursive=False):
	if os.path.isfile(input_path):
		if input_path.lower().endswith('.stl'):
			return [os.path.abspath(input_path)]
		return []
	if not os.path.isdir(input_path):
		return []
	if recursive:
		search = os.path.join(input_path, '**', pattern)
		paths = glob.glob(search, recursive=True)
	else:
		search = os.path.join(input_path, pattern)
		paths = glob.glob(search)
	return sorted([os.path.abspath(p) for p in paths if os.path.isfile(p)])


def build_output_path(stl_path, output_dir):
	stem = os.path.splitext(os.path.basename(stl_path))[0]
	return os.path.abspath(os.path.join(output_dir, stem + '.npz'))


def run(args):
	input_path = os.path.abspath(args.input)
	output_dir = os.path.abspath(args.output_dir)
	ensure_dir(output_dir)
	convert_single_stl = load_converter()

	stl_files = list_stl_files(input_path, pattern=args.pattern, recursive=args.recursive)
	if not stl_files:
		raise RuntimeError("No STL files found at input path: %s" % input_path)

	print("Found %d STL file(s)." % len(stl_files))
	ok = 0
	skipped = 0
	for stl in stl_files:
		out_npz = build_output_path(stl, output_dir)
		if os.path.exists(out_npz) and not args.overwrite:
			print("skip (exists): %s" % out_npz)
			skipped += 1
			continue
		print("convert: %s" % stl)
		ns = argparse.Namespace(
			input=stl,
			output=out_npz,
			pitch=args.pitch,
			grid=args.grid,
			fill_holes=args.fill_holes,
			close_radius=args.close_radius,
			keep_largest_component=args.keep_largest_component,
			notes=args.notes,
		)
		convert_single_stl(ns)
		ok += 1

	print("\nDone. converted=%d, skipped=%d, total=%d" % (ok, skipped, len(stl_files)))
	print("Output folder: %s" % output_dir)


def build_parser():
	default_input = os.path.join('Structures', '3D_structures', 'STL_input')
	default_output = os.path.join('Structures', '3D_structures', 'Voxolized')
	p = argparse.ArgumentParser(description="Convert STL files to NPZ volumes in the structured 3D workspace.")
	p.add_argument('--input', default=default_input,
				   help='STL file or folder with STL files (default: Structures/3D_structures/STL_input).')
	p.add_argument('--output-dir', default=default_output,
				   help='Folder for converted NPZ files (default: Structures/3D_structures/Voxolized).')
	p.add_argument('--pattern', default='*.stl',
				   help='Glob pattern when input is a directory (default: *.stl).')
	p.add_argument('--recursive', action='store_true',
				   help='Search input directory recursively.')
	p.add_argument('--pitch', type=float, required=True,
				   help='Voxelization pitch in STL length units.')
	p.add_argument('--grid', nargs=3, type=int, default=None, metavar=('NX', 'NY', 'NZ'),
				   help='Optional final resampling grid.')
	p.add_argument('--fill-holes', action='store_true',
				   help='Apply binary hole filling.')
	p.add_argument('--close-radius', type=int, default=0,
				   help='Binary closing radius (0 disables).')
	p.add_argument('--keep-largest-component', action='store_true',
				   help='Keep only largest connected component.')
	p.add_argument('--notes', default='',
				   help='Optional notes saved in output npz.')
	p.add_argument('--overwrite', action='store_true',
				   help='Overwrite existing NPZ files.')
	return p


if __name__ == '__main__':
	parser = build_parser()
	run(parser.parse_args())
