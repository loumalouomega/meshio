"""Affine transform of point coordinates (translate / scale / rotate / matrix).

A dependency-free mesh *operation* (not a file format): it applies a geometric
transform to a mesh's point coordinates while carrying connectivity and data
through unchanged. Vector/tensor ``point_data`` can optionally be rotated by the
transform's linear part. The C++ core (``_core.transform``) does the work; this
module is the thin shim (try C++, fall back to a pure-numpy reference) that also
builds the 4x4 matrix from high-level parameters and carries the shim-only
``point_sets``/``cell_sets`` (indices are unchanged, so they pass through as-is).

Public API:

* :func:`transform` — transform a mesh by translate / scale / rotate / matrix /
  unit-scale (compose several; ``translate`` is applied last).
"""

from __future__ import annotations

import numpy as np

from ._mesh import Mesh


# --------------------------------------------------------------------------- #
# 4x4 matrix builders                                                         #
# --------------------------------------------------------------------------- #
def _identity():
    return np.eye(4, dtype=np.float64)


def _translation(dx, dy, dz):
    m = _identity()
    m[0, 3], m[1, 3], m[2, 3] = dx, dy, dz
    return m


def _scale(sx, sy, sz):
    m = _identity()
    m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
    return m


def _rotation(axis, angle_rad):
    """Rodrigues rotation matrix about ``axis`` (a 3-vector or 'x'/'y'/'z')."""
    if isinstance(axis, str):
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[
            axis.lower()
        ]
    v = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-300:
        raise ValueError("transform: rotation axis is zero-length")
    x, y, z = v / n
    c, s, t = np.cos(angle_rad), np.sin(angle_rad), 1.0 - np.cos(angle_rad)
    m = _identity()
    m[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    return m


def _as_scale3(scale):
    arr = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    if arr.size == 3:
        return float(arr[0]), float(arr[1]), float(arr[2])
    raise ValueError("transform: scale must be a scalar or a 3-vector")


def _build_matrix(translate, scale, rotate, matrix, scale_units):
    if matrix is not None:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (16,):
            m = m.reshape(4, 4)
        if m.shape != (4, 4):
            raise ValueError("transform: matrix must be 4x4 (or a flat 16-vector)")
        return m
    m = _identity()
    if scale_units is not None:
        f = float(scale_units)
        m = _scale(f, f, f) @ m
    if scale is not None:
        m = _scale(*_as_scale3(scale)) @ m
    if rotate is not None:
        axis, angle_deg = rotate
        m = _rotation(axis, np.deg2rad(float(angle_deg))) @ m
    if translate is not None:
        dx, dy, dz = (
            float(v) for v in np.asarray(translate, dtype=np.float64).reshape(3)
        )
        m = _translation(dx, dy, dz) @ m
    return m


# --------------------------------------------------------------------------- #
# pure-numpy reference (used when the C++ core is unavailable)                 #
# --------------------------------------------------------------------------- #
def _rotate_point_data(arr, R, n):
    cols = arr.size // n if n else 0
    if arr.dtype.kind != "f" or n == 0 or cols not in (3, 9):
        return arr.copy()
    out = np.array(arr, dtype=arr.dtype)
    flat = out.reshape(n, cols)
    if cols == 3:
        rotated = flat @ R.T
    else:  # 3x3 tensor: A' = R A R^T
        A = flat.reshape(n, 3, 3).astype(np.float64)
        rotated = (R @ A @ R.T).reshape(n, 9)
    out.reshape(n, cols)[...] = rotated.astype(arr.dtype)
    return out


def _transform_py(mesh, matrix, rotate_vector_data):
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(mesh.points)
    n = pts.shape[0]
    dim = pts.shape[1] if pts.ndim == 2 else 0
    homog = np.zeros((n, 4), dtype=np.float64)
    homog[:, :dim] = pts[:, :dim] if n else pts
    homog[:, 3] = 1.0
    new = (homog @ m.T)[:, :dim].astype(pts.dtype)

    R = m[:3, :3]
    point_data = {}
    for key, value in mesh.point_data.items():
        value = np.asarray(value)
        if rotate_vector_data and value.shape[:1] == (n,):
            point_data[key] = _rotate_point_data(value, R, n)
        else:
            point_data[key] = value.copy()

    cells = [(cb.type, np.array(cb.data)) for cb in mesh.cells]
    counts = [len(cb.data) for cb in mesh.cells]
    cell_data = {}
    for key, blocks in mesh.cell_data.items():
        out_blocks = []
        for i, block in enumerate(blocks):
            if block is None:
                out_blocks.append(None)
                continue
            block = np.asarray(block)
            rows = counts[i] if i < len(counts) else 0
            if rotate_vector_data and rows and block.shape[:1] == (rows,):
                out_blocks.append(_rotate_point_data(block, R, rows))
            else:
                out_blocks.append(np.array(block))
        cell_data[key] = out_blocks
    field_data = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in mesh.field_data.items()
    }
    return Mesh(
        new,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )


# --------------------------------------------------------------------------- #
# set carry-through (indices are unchanged by an affine transform)            #
# --------------------------------------------------------------------------- #
def _carry_sets(src, out):
    point_sets = getattr(src, "point_sets", None)
    if point_sets:
        out.point_sets = {k: np.array(v) for k, v in point_sets.items()}
    cell_sets = getattr(src, "cell_sets", None)
    if cell_sets:
        out.cell_sets = {
            k: [None if b is None else np.array(b) for b in blocks]
            for k, blocks in cell_sets.items()
        }


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def transform(
    mesh,
    *,
    translate=None,
    scale=None,
    rotate=None,
    matrix=None,
    scale_units=None,
    rotate_vector_data: bool = False,
) -> Mesh:
    """Apply an affine transform to a mesh's point coordinates.

    Parameters
    ----------
    mesh :
        the mesh to transform (unmodified).
    translate :
        a 3-vector ``(dx, dy, dz)`` translation (applied last when composed).
    scale :
        a scalar (uniform) or 3-vector ``(sx, sy, sz)`` scale about the origin.
    rotate :
        a tuple ``(axis, angle_deg)`` where ``axis`` is ``'x'``/``'y'``/``'z'``
        or a 3-vector; the angle is in **degrees**.
    matrix :
        a full 4x4 (or flat 16-element) row-major affine matrix. When given, the
        other parameters are ignored.
    scale_units :
        a scalar unit-conversion factor (e.g. ``0.001`` for mm -> m).
    rotate_vector_data :
        when true, rotate vector (trailing dim 3) and tensor (trailing dim 9)
        ``point_data`` by the transform's linear part. Off by default.

    Returns
    -------
    Mesh
        a new mesh with transformed points; connectivity and data preserved.
        ``point_sets``/``cell_sets`` pass through unchanged (indices are stable).
    """
    m = _build_matrix(translate, scale, rotate, matrix, scale_units)

    out = None
    try:
        from . import _core

        out = _core.transform(mesh, m.reshape(-1).tolist(), bool(rotate_vector_data))
    except Exception:
        out = None
    if out is None:
        out = _transform_py(mesh, m, rotate_vector_data)

    _carry_sets(mesh, out)
    return out
