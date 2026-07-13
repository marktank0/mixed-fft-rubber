# 3D Samples Workspace

This folder is a dedicated workspace for viewing 3D structure files:

- `convert_stl_to_npz.py`: conversion CLI
- `view_npz_3d.py`: 3D visualization CLI for `.npz` volumes
- `view_stl_3d.py`: 3D visualization CLI for `.stl` meshes
- `view_vtp_3d.py`: 3D visualization CLI for `.vtp` meshes/structure folders


## Requirements
- For STL conversion: `pip install trimesh`
- Optional cleanup operations (`--fill-holes`, `--close-radius`, `--keep-largest-component`) need `scipy`
- For fast interactive viewing (recommended): `pip install pyvista`
- Matplotlib fallback viewer: `pip install matplotlib`

## 1) Convert STL -> NPZ

Single file:

```bash
python 3D_samples/convert_stl_to_npz.py --input 3D_samples/stl_input/my_part.stl --pitch 0.5
```

Directory (batch):

```bash
python 3D_samples/convert_stl_to_npz.py --input 3D_samples/stl_input --pitch 0.5 --pattern *.stl
```

Optional cleanup:

```bash
--fill-holes --close-radius 1 --keep-largest-component
```

Optional explicit FFT target grid:

```bash
--grid 65 65 65
```

## 2) View NPZ in 3D

Fast default view (auto backend, prefers PyVista):

```bash
python 3D_samples/view_npz_3d.py --input 3D_samples/npz_output/my_part.npz
```

Default mode is `surface` for smoother interaction.

For strict touching-cell visualization:

```bash
python 3D_samples/view_npz_3d.py --input 3D_samples/npz_output/my_part.npz --mode voxels
```

Explicit scatter preview:

```bash
python 3D_samples/view_npz_3d.py --input 3D_samples/npz_output/my_part.npz --mode scatter --max-points 120000
```

Voxel rendering + saved image:

```bash
python 3D_samples/view_npz_3d.py --input 3D_samples/npz_output/my_part.npz --mode voxels --save 3D_samples/npz_output/my_part_view.png --no-show
```

## 3) View STL in 3D

Fast default STL view (auto backend, prefers PyVista):

```bash
python 3D_samples/view_stl_3d.py --input 3D_samples/stl_input/my_part.stl
```

Large mesh speedup by decimation (example keeps ~40% geometry):

```bash
python 3D_samples/view_stl_3d.py --input 3D_samples/stl_input/my_part.stl --decimate 0.6
```

Wireframe + save image:

```bash
python 3D_samples/view_stl_3d.py --input 3D_samples/stl_input/my_part.stl --mode wireframe --save 3D_samples/npz_output/my_part_stl_view.png --no-show
```

## 4) View VTP in 3D

Single VTP:

```bash
python 3D_samples/view_vtp_3d.py --input path/to/file.vtp
```

Saved structure format (`<base>_particles`, `<base>_union2.vtp`, `<base>_union3.vtp`):

```bash
python 3D_samples/view_vtp_3d.py --input 3D_structure_generation/saved_structures --structure-base particle_structure_7
```

Fast preview on very large particle sets:

```bash
python 3D_samples/view_vtp_3d.py --input 3D_structure_generation/saved_structures --structure-base particle_structure_7 --mode centers --max-particles 1000
```

Save screenshot without opening a window:

```bash
python 3D_samples/view_vtp_3d.py --input path/to/folder --recursive --save 3D_samples/npz_output/vtp_view.png --no-show
```
