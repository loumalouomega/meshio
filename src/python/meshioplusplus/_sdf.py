"""Signed distance to a surface (``sample_distance``, ``distance_to_surface``).

How far a point is from a triangle skin, and which side of it the point is on --
the primitive collision detection, offsetting, inside/outside queries and
voxel-style preprocessing all reduce to.

The C++ core (``_core.sample_distance`` and friends) does the work; this module is
the thin shim plus a pure-numpy reference.

### Why the reference can be brute force

The C++ kernel accelerates its nearest-triangle search with a bucket grid, and
this module has none. That is not a shortcut: the accelerator is *provably*
unable to change the answer, because every candidate comparison is totally
ordered on ``(squared distance, triangle id)``. A brute-force scan in ascending
triangle order therefore produces the identical result -- which is what
``tests/python/test_sdf.py::test_cpp_matches_python`` asserts, and what
``SurfaceDistance.TheBucketSizeDoesNotChangeTheAnswer`` asserts from the other
side.

### Where byte-parity stops

``sign="winding-number"`` sums ``n_triangles`` calls to ``atan2`` -- which is not
correctly rounded, and whose last-ulp behaviour differs between libm and numpy --
and compares the total against a threshold. Two implementations can genuinely
disagree there, so this module **raises** :class:`NotImplementedError` rather than
hiding the difference behind a tolerance, exactly as ``_smooth.py`` does for the
inversion guard and ``_partition.py`` for ghost layers. The C++ path is still
tested, against analytic fields and against the pseudonormal answer.

Everything else here is ``+ - * /``, comparisons and one correctly-rounded
``sqrt``, so it is bit-identical by construction.

Public API:

* :func:`sample_distance` -- distances at arbitrary query points.
* :func:`distance_to_surface` -- the same, attached to a mesh as data.
* :func:`surface_watertight_check` -- what is wrong with a skin, in numbers.
* :func:`compute_sdf` -- generate a grid over a surface and fill it.
"""

from __future__ import annotations

import warnings

import numpy as np

from ._mesh import Mesh
from ._regions import block_bases

__all__ = [
    "sample_distance",
    "distance_to_surface",
    "surface_watertight_check",
    "compute_sdf",
]

_PREFIX = "meshio++: sdf: "
_SD_PREFIX = "meshio++: surface distance: "

_SIGNS = ("unsigned", "pseudonormal", "winding-number")
_WEIGHTS = ("angle", "area")
_LOCATIONS = ("corner", "center")


def _validate(sign, weight, location, watertight_check):
    if sign not in _SIGNS:
        raise ValueError(
            f"{_PREFIX}unknown sign '{sign}' (expected one of: {', '.join(_SIGNS)})"
        )
    if weight not in _WEIGHTS:
        raise ValueError(
            f"{_PREFIX}unknown weight '{weight}' (expected one of: {', '.join(_WEIGHTS)})"
        )
    if location not in ("corner", "point", "center", "centre", "cell"):
        raise ValueError(
            f"{_PREFIX}unknown location '{location}' (expected one of: corner, center)"
        )
    if watertight_check not in ("off", "warn", "error"):
        raise ValueError(
            f"{_PREFIX}unknown watertight check '{watertight_check}' "
            "(expected one of: off, warn, error)"
        )


# --------------------------------------------------------------------------- #
# the triangle soup                                                            #
# --------------------------------------------------------------------------- #
def _soup(mesh):
    """A surface reduced to triangles, remembering which input cell each came from.

    The fan is corner 0 to every non-adjacent edge -- the same one
    ``convert_cells(simplexify)`` uses, so the two cannot disagree about which
    diagonal a quad is split on.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    bases = block_bases(mesh.cells)

    verts = []
    source = []
    for b, cb in enumerate(mesh.cells):
        data = cb.data
        ragged = isinstance(data, list)
        name = cb.type
        if name.startswith("polyhedron") or (
            not name.startswith("polygon") and name not in ("triangle", "quad")
        ):
            from ._common import topological_dimension

            try:
                dim = topological_dimension(name)
            except Exception:
                dim = 2
            if dim == 3 or name.startswith("polyhedron"):
                raise ValueError(
                    f"{_SD_PREFIX}cell block '{name}' is a volume; distance is measured to a "
                    "surface (run extract_surface first)"
                )
            if dim < 2:
                continue
            raise ValueError(
                f"{_SD_PREFIX}cell block '{name}' is not a linear surface cell (linearize the "
                "mesh first, then run extract_surface if needed)"
            )
        rows = data if ragged else np.asarray(data)
        for c in range(len(rows)):
            ids = [int(x) for x in (rows[c] if ragged else rows[c])]
            if len(ids) < 3:
                continue
            for k in range(1, len(ids) - 1):
                verts.append((ids[0], ids[k], ids[k + 1]))
                source.append(int(bases[b]) + c)

    verts = np.asarray(verts, dtype=np.int64).reshape(-1, 3)
    source = np.asarray(source, dtype=np.int64)
    corners = points[verts] if len(verts) else np.zeros((0, 3, 3))
    return points, verts, corners, source


def surface_watertight_check(surface):
    """The four edge defect counts of a surface, and the resulting verdict."""
    try:
        from . import _core

        return _core.surface_watertight_check(surface)
    except (ValueError, TypeError):
        raise
    except Exception:
        pass
    return _watertight_py(surface)


def _watertight_py(surface):
    _points, verts, corners, _source = _soup(surface)
    degenerate = 0
    if len(verts):
        n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        degenerate = int(np.sum(np.sum(n * n, axis=1) <= 0.0))

    counts = {}
    for tri in verts:
        for e in range(3):
            u, w = int(tri[e]), int(tri[(e + 1) % 3])
            key = (u, w) if u < w else (w, u)
            used, forward = counts.get(key, (0, 0))
            counts[key] = (used + 1, forward + (1 if u < w else 0))

    boundary = non_manifold = inconsistent = 0
    for used, forward in counts.values():
        if used == 1:
            boundary += 1
        elif used > 2:
            non_manifold += 1
        elif used == 2 and forward != 1:
            inconsistent += 1
    return {
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "inconsistent_pairs": inconsistent,
        "degenerate_triangles": degenerate,
        "watertight": boundary == 0
        and non_manifold == 0
        and inconsistent == 0
        and degenerate == 0,
    }


# --------------------------------------------------------------------------- #
# the closest-point primitive, vectorized over triangles                       #
# --------------------------------------------------------------------------- #
def _closest_points(query, corners):
    """Closest point on every triangle to one query point.

    Ericson's region classification, transcribed branch for branch from
    ``detail/point_triangle.hpp``. The regions are applied in the same order and
    resolved by ``np.where`` so that an earlier region wins, which is what the
    C++ early returns do.
    """
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    ab, ac, ap = b - a, c - a, query - a
    d1 = np.sum(ab * ap, axis=1)
    d2 = np.sum(ac * ap, axis=1)
    bp = query - b
    d3 = np.sum(ab * bp, axis=1)
    d4 = np.sum(ac * bp, axis=1)
    cp = query - c
    d5 = np.sum(ab * cp, axis=1)
    d6 = np.sum(ac * cp, axis=1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    n = len(corners)
    point = np.empty((n, 3), dtype=np.float64)
    feature = np.empty(n, dtype=np.int64)
    done = np.zeros(n, dtype=bool)

    def take(mask, pts, feat):
        m = mask & ~done
        if not np.any(m):
            return
        point[m] = pts[m] if pts.ndim == 2 else pts
        feature[m] = feat
        done[m] = True

    take((d1 <= 0.0) & (d2 <= 0.0), a, 0)  # VertexA
    take((d3 >= 0.0) & (d4 <= d3), b, 1)  # VertexB
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d1 / (d1 - d3)
    take((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), a + ab * v[:, None], 3)  # EdgeAB
    take((d6 >= 0.0) & (d5 <= d6), c, 2)  # VertexC
    with np.errstate(divide="ignore", invalid="ignore"):
        w = d2 / (d2 - d6)
    take((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), a + ac * w[:, None], 5)  # EdgeCA
    with np.errstate(divide="ignore", invalid="ignore"):
        w2 = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    take(
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
        b + (c - b) * w2[:, None],
        4,
    )  # EdgeBC

    den = va + vb + vc
    degenerate = ~done & ~(den > 0.0)
    if np.any(degenerate):
        # The same explicit fallback the C++ takes: the face branch would divide
        # by zero and a NaN then wins every comparison in a min-reduction.
        best = np.full(n, np.inf)
        for e, (p0, p1, feat) in enumerate(((a, b, 3), (b, c, 4), (c, a, 5))):
            d = p1 - p0
            len2 = np.sum(d * d, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(len2 > 0.0, np.sum((query - p0) * d, axis=1) / len2, 0.0)
            t = np.clip(t, 0.0, 1.0)
            q = p0 + d * t[:, None]
            d2q = np.sum((query - q) ** 2, axis=1)
            better = degenerate & (d2q < best)
            best = np.where(better, d2q, best)
            point[better] = q[better]
            feature[better] = feat
        done |= degenerate

    rest = ~done
    if np.any(rest):
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / den
        pf = a + ab * (vb * inv)[:, None] + ac * (vc * inv)[:, None]
        point[rest] = pf[rest]
        feature[rest] = 6  # Face

    dist2 = np.sum((query - point) ** 2, axis=1)
    return point, dist2, feature


# --------------------------------------------------------------------------- #
# the pseudonormal tables                                                      #
# --------------------------------------------------------------------------- #
def _normal_tables(points, verts, corners, weight):
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lens = np.sqrt(np.sum(face * face, axis=1))
    vertex = np.zeros((len(points), 3), dtype=np.float64)
    edge = {}
    # SERIAL and in ascending (triangle, corner) order, matching the C++: summing
    # unit normals in a different order changes the last bits, and a last-bit
    # change can flip the sign of a point sitting almost exactly on the surface.
    for t in range(len(verts)):
        length = lens[t]
        if not length > 0.0:
            continue
        # Multiply by the reciprocal, as the C++ `vec3_scale(n, 1.0 / len)`
        # does: `face / length` differs in the last ulp, which a sign never
        # notices but an offset (shrinkwrap) puts straight into a coordinate.
        unit = face[t] * (1.0 / length)
        for i in range(3):
            if weight == "angle":
                u = corners[t, (i + 1) % 3] - corners[t, i]
                v = corners[t, (i + 2) % 3] - corners[t, i]
                nu = np.sqrt(np.sum(u * u))
                nv = np.sqrt(np.sum(v * v))
                if not nu > 0.0 or not nv > 0.0:
                    w = 0.0
                else:
                    cosang = np.sum(u * v) / (nu * nv)
                    w = float(np.arccos(min(1.0, max(-1.0, cosang))))
            else:
                w = length
            vertex[verts[t, i]] += unit * w
            p, r = int(verts[t, i]), int(verts[t, (i + 1) % 3])
            key = (p, r) if p < r else (r, p)
            edge[key] = edge.get(key, np.zeros(3)) + unit
    return face, vertex, edge


def _sample_py(surface, queries, sign, weight, band):
    if sign == "winding-number":
        raise NotImplementedError(
            f"{_PREFIX}the numpy reference does not implement sign='winding-number': it sums "
            "one atan2 per triangle and compares the total against a threshold, and atan2 is "
            "not correctly rounded, so the two implementations could genuinely disagree. Use "
            "the compiled core for it."
        )
    points, verts, corners, source = _soup(surface)
    if len(verts) == 0:
        raise ValueError(f"{_SD_PREFIX}the surface has no triangles to measure against")
    face, vertex, edge = _normal_tables(points, verts, corners, weight)

    out = np.empty(len(queries), dtype=np.float64)
    cells = np.full(len(queries), -1, dtype=np.int64)
    inband = np.ones(len(queries), dtype=bool)
    for q in range(len(queries)):
        point, dist2, feature = _closest_points(queries[q], corners)
        # argmin returns the FIRST minimum, i.e. the lowest triangle id -- the
        # same total order the C++ applies.
        t = int(np.argmin(dist2))
        d = float(np.sqrt(dist2[t]))
        if band > 0.0 and dist2[t] > band * band:
            out[q] = band
            inband[q] = False
            continue
        cells[q] = source[t]
        if sign == "unsigned":
            out[q] = d
            continue
        f = int(feature[t])
        if f in (0, 1, 2):
            normal = vertex[verts[t, f]]
        elif f in (3, 4, 5):
            e = {3: 0, 4: 1, 5: 2}[f]
            p, r = int(verts[t, e]), int(verts[t, (e + 1) % 3])
            normal = edge.get((p, r) if p < r else (r, p), face[t])
        else:
            normal = face[t]
        side = float(np.sum((queries[q] - point[t]) * normal))
        out[q] = -d if side < 0.0 else d
    return out, cells, inband


def _query_points_py(mesh, location):
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    if location in ("corner", "point"):
        return points
    out = []
    for cb in mesh.cells:
        data = cb.data
        if isinstance(data, list):
            for row in data:
                ids = np.asarray(row, dtype=np.int64).reshape(-1)
                out.append(points[ids].sum(axis=0) / len(ids))
            continue
        rows = np.asarray(data)
        npc = rows.shape[1]
        acc = np.zeros((len(rows), 3), dtype=np.float64)
        for i in range(npc):
            acc += points[rows[:, i]]
        out.extend(acc / npc)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def sample_distance(
    surface,
    points,
    sign="pseudonormal",
    weight="angle",
    band=0.0,
    watertight_check="warn",
    surface_region="",
    grid_cell_size=0.0,
    max_winding_work=2.0e9,
):
    """Signed distances from arbitrary points to ``surface``.

    Returns a Float64 ``(n,)`` array. Negative is inside, by the usual convention.
    """
    _validate(sign, weight, "corner", watertight_check)
    queries = np.asarray(points, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] not in (2, 3):
        raise ValueError(f"{_PREFIX}query points must be a 2-D (n, 2) or (n, 3) array")
    if queries.shape[1] == 2:
        queries = np.column_stack([queries, np.zeros(len(queries))])

    try:
        from . import _core

        return _core.sample_distance(
            surface,
            queries,
            sign,
            weight,
            float(band),
            watertight_check,
            surface_region,
            float(grid_cell_size),
            float(max_winding_work),
        )
    except (ValueError, TypeError):
        raise
    except Exception:
        pass
    return _sample_py(surface, queries, sign, weight, float(band))[0]


def distance_to_surface(
    query,
    surface,
    sign="pseudonormal",
    weight="angle",
    location="corner",
    band=0.0,
    record_closest_cell=False,
    record_inside=False,
    watertight_check="warn",
    surface_region="",
    grid_cell_size=0.0,
    max_winding_work=2.0e9,
    return_report=False,
):
    """Attach distances from ``query``'s points (or cell centres) to ``surface``."""
    _validate(sign, weight, location, watertight_check)

    out = None
    report = None
    try:
        from . import _core

        res = _core.distance_to_surface(
            query,
            surface,
            sign,
            weight,
            location,
            float(band),
            bool(record_closest_cell),
            bool(record_inside),
            watertight_check,
            surface_region,
            float(grid_cell_size),
            float(max_winding_work),
        )
        out = res["mesh"]
        report = {"num_banded": res["num_banded"], "quality": res["quality"]}
    except (ValueError, TypeError):
        raise
    except Exception:
        out = None

    if out is None:
        quality = _watertight_py(surface)
        if not quality["watertight"] and watertight_check != "off":
            what = (
                f"the surface is not watertight: {quality['boundary_edges']} boundary edge(s), "
                f"{quality['non_manifold_edges']} non-manifold edge(s), "
                f"{quality['inconsistent_pairs']} inconsistently wound pair(s), "
                f"{quality['degenerate_triangles']} degenerate triangle(s)"
            )
            if watertight_check == "error":
                raise ValueError(_PREFIX + what)
            warnings.warn(_PREFIX + what, stacklevel=2)

        queries = _query_points_py(query, location)
        dist, cells, inband = _sample_py(surface, queries, sign, weight, float(band))
        out = Mesh(
            np.asarray(query.points).copy(),
            [(cb.type, cb.data) for cb in query.cells],
            point_data={k: np.asarray(v).copy() for k, v in query.point_data.items()},
            cell_data={
                k: [np.asarray(b).copy() for b in v] for k, v in query.cell_data.items()
            },
            field_data={k: np.asarray(v).copy() for k, v in query.field_data.items()},
        )
        arrays = {"sdf:distance": dist}
        if band > 0.0:
            arrays["sdf:band"] = inband.astype(np.int64)
        if record_inside:
            arrays["sdf:inside"] = (dist < 0.0).astype(np.int64)
        if record_closest_cell:
            arrays["sdf:closest_cell"] = cells
        if location in ("corner", "point"):
            out.point_data.update(arrays)
        else:
            bases = block_bases(query.cells)
            for name, flat in arrays.items():
                out.cell_data[name] = [
                    flat[int(bases[b]) : int(bases[b + 1])]
                    for b in range(len(query.cells))
                ]
        report = {"num_banded": int(np.sum(~inband)), "quality": quality}

    return (out, report) if return_report else out


# --------------------------------------------------------------------------- #
# the umbrella: generate a grid, then fill it                                  #
# --------------------------------------------------------------------------- #
_STRUCTURES = ("voxel", "octree")


def _cell_diagonals_py(mesh):
    """Every cell's own corner-bbox diagonal.

    The numpy twin of ``sdfop_cell_diagonals``. Derived from the geometry rather
    than from ``refine:level``, so it is right for a mesh this module did not
    build, and exact for the axis-aligned boxes an octree contains.
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    out = []
    for cb in mesh.cells:
        data = cb.data
        if isinstance(data, list):
            out.extend([0.0] * len(data))
            continue
        rows = np.asarray(data, dtype=np.int64)
        corners = points[rows]
        e = corners.max(axis=1) - corners.min(axis=1)
        # Written out rather than np.sum: three terms, left to right, matching
        # the C++ accumulation exactly.
        out.extend(np.sqrt(e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1] + e[:, 2] * e[:, 2]))
    return np.asarray(out, dtype=np.float64)


def _compute_sdf_py(surface, structure, kwargs, distance_kwargs):
    from ._grid import _grid_py
    from ._voxelize import _resolve_lattice

    if structure == "octree":
        root = int(kwargs["root_resolution"])
        resolution, cell_size = (root, root, root), None
    else:
        resolution, cell_size = kwargs["resolution"], kwargs["cell_size"]

    lo, spacing, dims = _resolve_lattice(
        surface,
        resolution,
        cell_size,
        kwargs["bounds"],
        kwargs["padding"],
        kwargs["padding_relative"],
        int(kwargs["max_cells"]),
        _PREFIX,
    )
    mesh = _grid_py(dims, lo, spacing)
    depth = 0
    final_spacing = np.asarray(spacing, dtype=np.float64).copy()

    if structure == "octree":
        from ._refine import refine

        band_cells = float(kwargs["band_cells"])
        for _ in range(int(kwargs["max_depth"])):
            # Recomputed from the CURRENT mesh every pass: these are global
            # block-major indices, so a selection carried over would name cells
            # of a mesh that no longer exists.
            centres = _query_points_py(mesh, "center")
            diag = _cell_diagonals_py(mesh)
            # Unsigned: only the magnitude matters for selection, and an
            # unsigned query neither needs a closed surface nor pays for a sign
            # it would discard.
            dist = _sample_py(
                surface, centres, "unsigned", distance_kwargs["weight"], 0.0
            )[0]
            selected = np.nonzero(np.abs(dist) <= band_cells * diag)[0]
            if selected.size == 0:
                break
            mesh = refine(
                mesh,
                cells=[int(c) for c in selected],
                closure="balanced",
                record_levels=bool(kwargs["record_levels"]),
            )
            depth += 1
            final_spacing = final_spacing * 0.5
            total = sum(len(cb.data) for cb in mesh.cells)
            max_cells = int(kwargs["max_cells"])
            if max_cells > 0 and total > max_cells:
                raise ValueError(
                    f"{_PREFIX}the octree reached {total} cells after {depth} pass(es), "
                    f"above the limit of {max_cells} "
                    "(raise max_cells, lower max_depth or band_cells)"
                )

    # The field is computed ONCE, on the final mesh: `refine` interpolates
    # point_data, so a field attached mid-way would come out a smooth,
    # plausible interpolation of the coarse values rather than the distance.
    filled, report = distance_to_surface(
        mesh, surface, return_report=True, **distance_kwargs
    )
    hi = np.asarray(lo, dtype=np.float64) + np.asarray(
        dims, dtype=np.float64
    ) * np.asarray(spacing, dtype=np.float64)
    filled.field_data["sdf:origin"] = np.asarray(lo, dtype=np.float64).copy()
    filled.field_data["sdf:spacing"] = final_spacing
    filled.field_data["sdf:dims"] = np.asarray(dims, dtype=np.int64).copy()
    filled.field_data["sdf:bounds"] = np.concatenate(
        [np.asarray(lo, dtype=np.float64), hi]
    )
    filled.field_data["sdf:max_depth"] = np.array([depth], dtype=np.int64)
    filled.field_data["sdf:structure"] = np.array(
        [1 if structure == "octree" else 0], dtype=np.int64
    )
    out = {
        "mesh": filled,
        "dims": [int(v) for v in dims],
        "origin": [float(v) for v in lo],
        "spacing": [float(v) for v in final_spacing],
        "max_depth": depth,
        "num_banded": report["num_banded"],
        "quality": report["quality"],
    }
    return out


def compute_sdf(
    surface,
    structure="voxel",
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.1,
    root_resolution=8,
    max_depth=4,
    band_cells=1.0,
    record_levels=True,
    max_cells=20000000,
    sign="pseudonormal",
    weight="angle",
    location="corner",
    band=0.0,
    record_closest_cell=False,
    record_inside=False,
    watertight_check="warn",
    surface_region="",
    grid_cell_size=0.0,
    max_winding_work=2.0e9,
    return_report=False,
):
    """Generate a grid over ``surface`` and fill it with signed distances.

    The one call that turns a surface into a field.

    Parameters
    ----------
    surface :
        The surface to measure against.
    structure :
        ``"voxel"`` for a dense lattice, ``"octree"`` for one refined near the
        surface. The octree's output is **1-irregular** -- it has hanging nodes;
        see :func:`meshioplusplus.refine`'s ``closure="balanced"``.
    resolution, cell_size :
        Exactly one, and ``"voxel"`` only. An octree's finest cell is
        ``root cell / 2**depth`` and is therefore already determined, so passing
        either with ``structure="octree"`` is an error rather than a silent
        preference.
    bounds, padding, padding_relative :
        The box to cover. The relative padding defaults to 0.1 because a field
        that stops at the surface is not much use.
    root_resolution, max_depth, band_cells, record_levels :
        Octree only. A cell is refined while
        ``abs(distance) <= band_cells * its own diagonal``, so the band narrows
        as the tree deepens.
    max_cells :
        Refuse by name above this many cells, re-checked after every octree pass.
    sign, weight, location, band, record_closest_cell, record_inside,
    watertight_check, surface_region, grid_cell_size, max_winding_work :
        Passed through to :func:`distance_to_surface`.
    return_report :
        Also return the grid geometry, the surface verdict and the banded count.

    Returns
    -------
    Mesh, or (Mesh, dict)
        The grid carrying ``sdf:distance``, plus the ``sdf:*`` ``field_data``
        header describing itself. **No format persists arbitrary ``field_data``**,
        so that header survives in memory only -- ``.vti`` round-trips the
        geometry instead. See ``doc/sdf.md``.
    """
    _validate(sign, weight, location, watertight_check)
    if structure not in _STRUCTURES:
        raise ValueError(
            f"{_PREFIX}unknown structure '{structure}' "
            f"(expected one of: {', '.join(_STRUCTURES)})"
        )
    if structure == "octree":
        if root_resolution <= 0:
            raise ValueError(
                f"{_PREFIX}root_resolution must be positive, got {int(root_resolution)}"
            )
        if max_depth < 0:
            raise ValueError(
                f"{_PREFIX}max_depth must not be negative, got {int(max_depth)}"
            )
        if not band_cells > 0.0:
            raise ValueError(
                f"{_PREFIX}band_cells must be positive, got {float(band_cells)}"
            )
        if resolution is not None or cell_size is not None:
            raise ValueError(
                f"{_PREFIX}structure 'octree' sizes itself from root_resolution and "
                "max_depth; resolution and cell_size apply to structure 'voxel' only"
            )

    res = None
    try:
        from . import _core

        res = _core.compute_sdf(
            surface,
            structure,
            None if resolution is None else [int(v) for v in resolution],
            None if cell_size is None else float(cell_size),
            (
                None
                if bounds is None
                else [float(v) for v in np.asarray(bounds).reshape(-1)]
            ),
            float(padding),
            float(padding_relative),
            int(root_resolution),
            int(max_depth),
            float(band_cells),
            bool(record_levels),
            int(max_cells),
            sign,
            weight,
            location,
            float(band),
            bool(record_closest_cell),
            bool(record_inside),
            watertight_check,
            surface_region,
            float(grid_cell_size),
            float(max_winding_work),
        )
    except (ValueError, TypeError):
        raise
    except Exception:
        res = None

    if res is None:
        res = _compute_sdf_py(
            surface,
            structure,
            {
                "resolution": resolution,
                "cell_size": cell_size,
                "bounds": bounds,
                "padding": padding,
                "padding_relative": padding_relative,
                "root_resolution": root_resolution,
                "max_depth": max_depth,
                "band_cells": band_cells,
                "record_levels": record_levels,
                "max_cells": max_cells,
            },
            {
                "sign": sign,
                "weight": weight,
                "location": location,
                "band": band,
                "record_closest_cell": record_closest_cell,
                "record_inside": record_inside,
                "watertight_check": watertight_check,
                "surface_region": surface_region,
                "grid_cell_size": grid_cell_size,
                "max_winding_work": max_winding_work,
            },
        )

    mesh = res["mesh"]
    if not return_report:
        return mesh
    report = {k: v for k, v in res.items() if k != "mesh"}
    return mesh, report
