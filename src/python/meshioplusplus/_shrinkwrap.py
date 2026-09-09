"""Shrinkwrap: project a mesh's points onto a target triangle surface,
optionally offset along the surface normal.

A dependency-free mesh *operation* (not a file format). Adapted from NVIDIA
PhysicsNeMo's ``physicsnemo.mesh.shrinkwrap`` (2.2): ``x' = x + w * (p +
offset * n - x)`` with ``p`` the closest point on the target, ``n`` a unit
normal there and ``w`` a per-point weight -- ONE projection, not an iteration,
with no self-intersection or inversion guard, exactly as upstream. Rewritten
over meshio++'s own nearest-triangle search (the one ``sample_distance``
uses), so the point a query is projected to is the point ``sample_distance``
measures to.

**One deliberate divergence from upstream.** Upstream offsets along the normal
of the SELECTED triangle. At an edge or vertex hit that is the wrong
direction: the offset surface of a creased mesh is the rounded one, whose
normal at the crease is the bisector of the incident faces, and offsetting
along one face's normal from the crease lands off that surface -- and makes
the result depend on which equidistant face won the tie-break. meshio++
offsets along the pseudonormal of the hit FEATURE, the same tables the signed
distance's sign already reads.

### Where byte-parity stops

The numpy reference in this module reuses ``_sdf``'s brute-force
``(distance^2, triangle id)`` search and its normal tables, and the projection
itself is ``+ - * /`` and one correctly rounded ``sqrt``, so the moved points
are bit-identical to the compiled core's -- with ONE exclusion. The
angle-weighted vertex pseudonormal (``normal_weight="angle"``, the default)
uses ``acos``, which is not correctly rounded; a sign does not care about its
last bits, but an OFFSET enters the coordinates linearly. So the reference
refuses ``offset != 0`` under ``"angle"`` weighting by name and points at
``normal_weight="area"``, which is bit-exactly twinned. The raise only fires
when the compiled core is genuinely unavailable.

Public API:

* :func:`shrinkwrap` -- project the points.
"""

from __future__ import annotations

import numpy as np

from ._mesh import Mesh
from ._sdf import _closest_points, _normal_tables, _soup, _watertight_py

__all__ = ["shrinkwrap", "DISTANCE_NAME", "CLOSEST_CELL_NAME"]

_PREFIX = "meshio++: shrinkwrap: "

#: Point data (opt-in): each point's distance to the target before the move.
DISTANCE_NAME = "shrinkwrap:distance"
#: Point data (opt-in): the target cell each point was projected onto.
CLOSEST_CELL_NAME = "shrinkwrap:closest_cell"

_WEIGHTS_TEMP = "shrinkwrap:weights"


def _target_soup(target, region):
    """The target's triangle soup, with the kernel's errors re-prefixed."""
    if region:
        from ._curvature import _region_mask

        mask = _region_mask(target, region)
    else:
        mask = None
    try:
        points, verts, corners, source = _soup(target)
    except ValueError as e:
        msg = str(e)
        kernel = "meshio++: surface distance: "
        if msg.startswith(kernel):
            msg = msg[len(kernel) :]
        raise ValueError(f"{_PREFIX}target: {msg}") from None
    if mask is not None and len(verts):
        keep = mask[source]
        verts, corners, source = verts[keep], corners[keep], source[keep]
    if len(verts) == 0:
        raise ValueError(
            f"{_PREFIX}target: the surface has no triangles to project onto"
        )
    return points, verts, corners, source


def _shrinkwrap_py(
    mesh,
    target,
    offset,
    max_distance,
    weights,
    target_region,
    normal_weight,
    record_distance,
    record_closest_cell,
):
    """The numpy twin of ``shrinkwrap()`` (C++ ``shrinkwrap.cpp``)."""
    if offset != 0.0 and normal_weight == "angle":
        raise NotImplementedError(
            f"{_PREFIX}the numpy reference does not implement a nonzero offset "
            "under normal_weight='angle': the angle-weighted vertex pseudonormal "
            "goes through acos, which is not correctly rounded, and an offset "
            "puts its last bits straight into the coordinates. Use the "
            "compiled core for it, or normal_weight='area', which is "
            "bit-exactly twinned."
        )
    if normal_weight not in ("angle", "area"):
        raise ValueError(
            f"meshio++: surface distance: unknown weight '{normal_weight}' "
            "(expected one of: angle, area)"
        )
    xyz = np.asarray(mesh.points, dtype=np.float64)
    n = len(xyz)
    dim = xyz.shape[1] if xyz.ndim == 2 else 0
    if dim < 2 or dim > 3:
        raise ValueError(f"{_PREFIX}points must be 2-D or 3-D")
    if dim == 2:
        xyz = np.column_stack([xyz, np.zeros(n)])
    xyz = xyz.copy()

    tpoints, verts, corners, source = _target_soup(target, target_region)
    quality = _watertight_py(target)
    face, vertex, edge = _normal_tables(tpoints, verts, corners, normal_weight)

    w = np.ones(n, dtype=np.float64)
    if weights:
        if weights not in mesh.point_data:
            raise ValueError(
                f"{_PREFIX}no point_data array named '{weights}' "
                f"(available: {', '.join(sorted(mesh.point_data)) or 'none'})"
            )
        a = np.asarray(mesh.point_data[weights])
        if a.shape[0] != n or a.size != n:
            raise ValueError(
                f"{_PREFIX}weights array '{weights}' must be a scalar per point "
                f"({n} rows, 1 component)"
            )
        a = a.reshape(n)
        if np.issubdtype(a.dtype, np.floating):
            w = a.astype(np.float64)
        else:
            w = (a != 0).astype(np.float64)

    distance = np.full(n, np.nan, dtype=np.float64)
    closest = np.full(n, -1, dtype=np.int64)
    projected = missed = 0
    max_disp = 0.0
    for i in range(n):
        if w[i] == 0.0:
            continue
        x = xyz[i].copy()
        point, dist2, feature = _closest_points(x, corners)
        t = int(np.argmin(dist2))
        d = float(np.sqrt(dist2[t]))
        distance[i] = d
        closest[i] = source[t]
        if max_distance > 0.0 and d > max_distance:
            missed += 1
            continue
        tgt = point[t].copy()
        if offset != 0.0:
            f = int(feature[t])
            if f in (0, 1, 2):
                normal = vertex[verts[t, f]]
            elif f in (3, 4, 5):
                e = {3: 0, 4: 1, 5: 2}[f]
                p, r = int(verts[t, e]), int(verts[t, (e + 1) % 3])
                normal = edge.get((p, r) if p < r else (r, p), face[t])
            else:
                normal = face[t]
            nn = float(
                np.sqrt(
                    normal[0] * normal[0]
                    + normal[1] * normal[1]
                    + normal[2] * normal[2]
                )
            )
            if not nn > 0.0:
                missed += 1
                continue
            # The C++ order: offset / |n| first, then scale, then add.
            tgt = tgt + normal * (offset / nn)
        moved = x + (tgt - x) * w[i]
        xyz[i] = moved
        projected += 1
        diff = moved - x
        disp = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
        if disp > max_disp:
            max_disp = disp

    out = Mesh(
        np.ascontiguousarray(xyz[:, :dim]).astype(
            np.asarray(mesh.points).dtype, copy=False
        ),
        [(b.type, b.data) for b in mesh.cells],
    )
    out.point_data = {k: v for k, v in mesh.point_data.items()}
    out.cell_data = {k: list(v) for k, v in mesh.cell_data.items()}
    out.field_data = {k: v for k, v in mesh.field_data.items()}
    if record_distance:
        out.point_data[DISTANCE_NAME] = distance
    if record_closest_cell:
        out.point_data[CLOSEST_CELL_NAME] = closest
    report = {
        "num_projected": projected,
        "num_missed": missed,
        "num_skipped": int(np.sum(w == 0.0)),
        "max_displacement": max_disp,
        "quality": quality,
    }
    return out, report


def shrinkwrap(
    mesh,
    target,
    offset: float = 0.0,
    max_distance: float = 0.0,
    weights=None,
    target_region: str = "",
    normal_weight: str = "angle",
    record_distance: bool = False,
    record_closest_cell: bool = False,
    return_report: bool = False,
):
    """Project a mesh's points onto the surface of ``target``.

    Every point of ``mesh`` moves, whatever cells it carries (a volume mesh's
    interior points are projected too -- only the target must be a surface);
    ``weights`` selects or blends. A point farther than ``max_distance`` from
    the target is left where it is and counted, as is one whose hit feature
    has no direction to offset along. Connectivity, data, regions and property
    sets pass through verbatim, since this is a pure coordinate move.

    :param mesh: the mesh whose points move (never modified).
    :param target: the surface to project onto; quads and polygons are fanned,
        a volume or higher-order block is refused by name.
    :param offset: signed offset along the hit feature's unit pseudonormal
        (positive is the target's outward side when it is consistently wound).
    :param max_distance: leave points farther than this alone; ``<= 0`` means
        unlimited.
    :param weights: per-point weights: the name of a ``point_data`` array
        (float: unclamped blend factor; integer/bool: a selection), or an
        array of that shape (attached as a temporary ``shrinkwrap:weights``
        array for the call). ``None`` moves every point with ``w = 1``.
    :param target_region: restrict the target to this named Cell region.
    :param normal_weight: ``"angle"`` (default) or ``"area"`` weighting of the
        target's vertex pseudonormals.
    :param record_distance: attach ``shrinkwrap:distance`` (NaN where not
        queried).
    :param record_closest_cell: attach ``shrinkwrap:closest_cell`` (-1 where
        not queried).
    :param return_report: also return the run summary dict (counters plus the
        target's ``quality``).
    :returns: the moved mesh, or ``(mesh, report)`` if ``return_report``.
    :raises ValueError: on an unusable target, an unknown region or weights
        array, or a weights array of the wrong shape.
    :raises NotImplementedError: from the numpy reference only, for a nonzero
        offset under ``normal_weight="angle"`` (see the module docstring).
    """
    src = mesh
    weights_name = weights if isinstance(weights, str) else ""
    if weights is not None and not isinstance(weights, str):
        import copy as _copy

        src = _copy.copy(mesh)
        src.point_data = dict(mesh.point_data)
        src.point_data[_WEIGHTS_TEMP] = np.asarray(weights)
        weights_name = _WEIGHTS_TEMP

    out = None
    report = None
    try:
        from . import _core

        res = _core.shrinkwrap(
            src,
            target,
            float(offset),
            float(max_distance),
            weights_name,
            str(target_region),
            str(normal_weight),
            bool(record_distance),
            bool(record_closest_cell),
            0.0,
        )
        out = res.pop("mesh")
        report = res
    except (ValueError, TypeError):
        raise
    except Exception:
        out = None
    if out is None:
        out, report = _shrinkwrap_py(
            src,
            target,
            float(offset),
            float(max_distance),
            weights_name,
            str(target_region),
            str(normal_weight),
            bool(record_distance),
            bool(record_closest_cell),
        )
    if weights_name == _WEIGHTS_TEMP:
        out.point_data.pop(_WEIGHTS_TEMP, None)

    # Sets are carried through untouched: nothing is renumbered.
    if getattr(mesh, "point_sets", None):
        out.point_sets = {k: np.asarray(v).copy() for k, v in mesh.point_sets.items()}
    if getattr(mesh, "cell_sets", None):
        out.cell_sets = {
            k: [np.asarray(a).copy() for a in v] for k, v in mesh.cell_sets.items()
        }
    if not report["quality"]["watertight"]:
        from ._common import warn

        q = report["quality"]
        warn(
            f"shrinkwrap: the target is not watertight: {q['boundary_edges']} "
            f"boundary edge(s), {q['non_manifold_edges']} non-manifold edge(s), "
            f"{q['inconsistent_pairs']} inconsistently wound pair(s), "
            f"{q['degenerate_triangles']} degenerate triangle(s) -- a nonzero "
            "offset may point to different sides near the defects"
        )
    return (out, report) if return_report else out
