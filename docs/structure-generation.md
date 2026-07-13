# Structure Generation Notes

Use this file for detailed, durable notes about 3D structure generation, voxelization, slicing, and PHR naming. Keep `project-memory.md` as the concise overview.

## Core Files

- `microstructure_generation/combined_particle_models.py`: generates sphere structures and saves sphere/voxel NPZ outputs.
- `microstructure_generation/cross_section_generator.py`: creates 2D cross-section images from particle meshes, voxel NPZ files, or sphere-parameter NPZ files.
- `microstructure_generation/calculate_phr.py`: calculates 3D PHR for voxel or sphere NPZ files.
- `microstructure_generation/create_2d_testset_images_by_phr.py`: batch-generates 2D slice images from TestSet 3D sphere structures, calculates each slice's own 2D PHR, and prefixes filenames by PHR.
- `microstructure_generation/smooth_2d_occlusions.py`: fills small black polymer components fully enclosed by white filler in 2D masks and writes a manifest of area changes.

## Path Behavior

`combined_particle_models.py` resolves relative save directories from the repository root, not from the script folder.

The TestSet source folder currently exists as:

```text
Structures/3D_structures/Spheres
```

Voxelized 3D structures are stored in:

```text
Structures/3D_structures/Voxolized
```

Preserve the existing `Voxolized` spelling unless the repository is explicitly renamed.

## PHR Naming For 2D Images

The 2D TestSet image generation workflow computes PHR per slice from the 2D slice pixels. Slices from the same 3D structure can have different PHR.

Generated images are flat-saved into:

```text
Structures/2D_images
```

Filename format:

```text
{zero_padded_2_sig_digit_phr}phr_{structure_name}_section_{index}.png
```

Example:

```text
008.00phr_1_spheres_section_001.png
```

An exact manifest is written to:

```text
Structures/2D_images/image_manifest.csv
```

The filename prefix is rounded to two significant digits for sorting and readability; use the manifest for exact PHR.

## 2D Occlusion Smoothing

`Structures/2D_test_set_smoothed` contained selected 2D masks where some small black polymer pockets were fully surrounded by white filler, which can create artificial stress concentrations in 2D simulations.

The post-processed selected set is stored in:

```text
Structures/2D_test_set_occlusion_smoothed
```

It was produced with:

```text
python -B microstructure_generation/smooth_2d_occlusions.py --input-dir Structures/2D_test_set_smoothed --output-dir Structures/2D_test_set_occlusion_smoothed --manifest Structures/2D_test_set_occlusion_smoothed/occlusion_smoothing_manifest.csv --max-hole-area 512
```

That pass filled 12 enclosed polymer pockets totaling 433 pixels across `023.00phr.png`, `029.00phr.png`, `037.00phr.png`, `040.00phr.png`, and `046.00phr.png`; validation found zero enclosed polymer pockets remaining in the output folder.

## Cross-Section Generation

For sphere NPZ files, `cross_section_generator.py` can voxelize spheres internally without importing PyVista. The particle-mesh path still imports PyVista lazily only when needed.

Default section settings currently used by TestSet generation:

- axis: `z`
- sections per structure: `10`
- image resolution: `256`
- sphere NPZ voxel grid: `(256, 256, 256)` for slicing

## 3D Structure PHR

`calculate_phr.py` supports:

- voxel NPZ files with `phase`
- sphere parameter NPZ files with `centers` and `radii`

For sphere NPZ files, PHR is based on summed sphere volumes and `box_size`. For voxel NPZ files, PHR is based on filled voxel fraction and `box_size`/`voxel_size`.
