# -*- coding: utf-8 -*-
"""VTK ImageData export for voxel fields."""

import os
from xml.sax.saxutils import escape

import numpy as np


COMPONENTS = (
    ("11", 0, 0), ("12", 0, 1), ("13", 0, 2),
    ("21", 1, 0), ("22", 1, 1), ("23", 1, 2),
    ("31", 2, 0), ("32", 2, 1), ("33", 2, 2),
)


def _array_to_ascii(array):
    flat = np.asarray(array, dtype=float).ravel(order="F")
    return " ".join("{:.16e}".format(value) for value in flat)


def _check_field_shapes(fields):
    shapes = {np.asarray(value).shape for value in fields.values()}
    if len(shapes) != 1:
        raise ValueError("All VTK fields must have the same shape; got {}".format(sorted(shapes)))
    shape = shapes.pop()
    if len(shape) != 3:
        raise ValueError("VTK fields must be 3D arrays; got shape {}".format(shape))
    return shape


def green_lagrange_strain_fields(F):
    N = F.shape[-1]
    strain = np.zeros((3, 3, N, N, N))
    eye = np.eye(3)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                C = F[:, :, x, y, z].T @ F[:, :, x, y, z]
                strain[:, :, x, y, z] = 0.5*(C - eye)
    return strain


def solution_fields(F, P, phase, pressure=None):
    """Build named scalar fields from local solver arrays."""
    fields = {}
    for suffix, i, j in COMPONENTS:
        fields["F{}".format(suffix)] = F[i, j, :, :, :]
        fields["P{}".format(suffix)] = P[i, j, :, :, :]

    strain = green_lagrange_strain_fields(F)
    for suffix, i, j in COMPONENTS:
        fields["E{}".format(suffix)] = strain[i, j, :, :, :]

    fields["J"] = np.linalg.det(np.moveaxis(F, (0, 1), (-2, -1)))
    fields["phase"] = phase
    if pressure is not None:
        fields["pressure"] = pressure
    return fields


def save_vti_cell_fields(output_path, fields, filename="fields.vti"):
    """Save scalar cell fields as an ASCII .vti file readable by ParaView."""
    os.makedirs(output_path, exist_ok=True)
    shape = _check_field_shapes(fields)
    nx, ny, nz = shape
    outfile = os.path.join(output_path, filename)

    with open(outfile, "w") as file:
        file.write('<?xml version="1.0"?>\n')
        file.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        file.write(
            '  <ImageData WholeExtent="0 {0} 0 {1} 0 {2}" Origin="0 0 0" Spacing="1 1 1">\n'
            .format(nx, ny, nz)
        )
        file.write('    <Piece Extent="0 {0} 0 {1} 0 {2}">\n'.format(nx, ny, nz))
        file.write('      <CellData>\n')
        for name in sorted(fields):
            file.write(
                '        <DataArray type="Float64" Name="{}" format="ascii">\n'
                .format(escape(name))
            )
            file.write("          {}\n".format(_array_to_ascii(fields[name])))
            file.write("        </DataArray>\n")
        file.write('      </CellData>\n')
        file.write('      <PointData/>\n')
        file.write('    </Piece>\n')
        file.write('  </ImageData>\n')
        file.write('</VTKFile>\n')

    return outfile
