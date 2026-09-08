"""Per-vertex mean and Gaussian curvature of a surface, by the standard
discrete estimators: the angle defect for ``K`` and the cotangent
Laplace-Beltrami operator for ``H``.

These are the estimators the discrete-differential-geometry literature's
convergence results are about, and the ones NVIDIA PhysicsNeMo's own
``gaussian_curvature_vertices``/``mean_curvature_vertices`` use, so the numbers
here are comparable with a model's. Curvature is the signed distance's natural
companion as a node feature: :func:`meshioplusplus.sample_distance` says how
far a point is from the surface, this says how the surface bends there.

**Deliberately NOT** :func:`meshioplusplus.remesh`'s estimator, which fits an
osculating paraboloid over each 1-ring to drive its own gradation and
anisotropic metric. That one returns only the larger-magnitude principal
curvature as a *magnitude*, runs on remesh's subdivided working copy rather
than the caller's mesh, and has no structural invariant to test against. It is
left exactly as it is; this is a second, independent estimator.

**The oracle is Gauss-Bonnet.** For a closed surface the angle defects sum to
``2*pi*chi`` -- ``4*pi`` for a sphere -- whatever the tessellation and
whichever dual area. The result carries that sum as ``total_angle_defect``
precisely so the invariant is checkable from every binding and not only from a
gtest.

**``H`` is orientation-dependent and ``K`` is not.** The mean curvature's sign
comes from the surface's own winding, so a mesh whose facets disagree about
which side is out yields sign-flipped patches with no error raised. That is why
the report carries the input's ``quality``: check ``quality["inconsistent_pairs"]``
before trusting a sign. This operation never silently repairs its input.

### Where byte-parity stops

``dual_area="barycentric"`` is branch-free -- every vertex gets a third of each
incident triangle's area -- so this module implements it as a genuine numpy
twin of the C++ core and ``tests/python/test_curvature.py`` pins the two
against each other with exact equality.

``dual_area="mixed-voronoi"`` (the default) is not twinned and **raises**
:class:`NotImplementedError`: its obtuse/non-obtuse test is a discrete branch
on a sign, so two implementations could land on opposite sides for a
right-angled triangle and then disagree macroscopically rather than in the last
ulp -- exactly the case ``_smooth.py`` refuses for its inversion guard and
``_sdf.py`` for ``sign="winding-number"``. The raise only fires when the
compiled core is genuinely unavailable.

Public API:

* :func:`compute_curvature` -- the estimator.
"""

from __future__ import annotations

import numpy as np

from ._regions import block_bases

_PREFIX = "meshio++: curvature: "

#: Point data written by :func:`compute_curvature`. Twins of ``curvature.hpp``'s
#: ``kCurvature*Name`` constants -- keep the two in step.
MEAN_NAME = "curvature:mean"
GAUSSIAN_NAME = "curvature:gaussian"
AREA_NAME = "curvature:area"
PRINCIPAL_NAME = "curvature:principal"

_DUAL_AREAS = ("mixed-voronoi", "barycentric")

_TWO_PI = 6.283185307179586476925286766559
_PI = 3.141592653589793238462643383279


def _validate(dual_area):
    if dual_area not in _DUAL_AREAS:
        raise ValueError(
            f"{_PREFIX}unknown dual area '{dual_area}' "
            "(expected 'mixed-voronoi' or 'barycentric')"
        )


def _region_mask(mesh, region):
    """Global block-major cell mask for a named ``Cell`` region.

    The numpy twin of ``sd_region_mask``: same lookup, same message shape, and
    the same "out-of-range entries are ignored" rule.
    """
    if not region:
        return None
    for r in getattr(mesh, "regions", ()) or ():
        if r.name == region and r.kind == "cell":
            bases = block_bases(mesh.cells)
            total = int(bases[-1]) + (
                len(mesh.cells[-1].data) if len(mesh.cells) else 0
            )
            mask = np.zeros(total, dtype=bool)
            e = np.asarray(r.entries, dtype=np.int64).ravel()
            e = e[(e >= 0) & (e < total)]
            mask[e] = True
            return mask
    have = ", ".join(sorted({r.name for r in (getattr(mesh, "regions", ()) or ())}))
    raise ValueError(
        f"{_PREFIX}no cell region named '{region}' "
        f"(available: {have if have else 'none'})"
    )


def _dot(u, v):
    """``vec3_dot``'s own left-to-right order, never ``np.sum``/``einsum``.

    ``detail/geometry.hpp`` spells it ``a[0]*b[0] + a[1]*b[1] + a[2]*b[2]``;
    numpy's reductions are free to pair-sum, which differs in the last bit.
    """
    return u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1] + u[:, 2] * v[:, 2]


def _cross(u, v):
    """``vec3_cross``'s own component order."""
    return np.column_stack(
        [
            u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1],
            u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
            u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0],
        ]
    )


def _corner(apex, b, c):
    """``curv_corner``'s angle and cotangent, from one ``(cross, dot)`` pair.

    The angle is ``atan2(|cross|, dot)`` and deliberately **not** ``acos`` of a
    clamped ratio: an angle defect is a sum of angles minus ``2*pi``, so the
    digits are exactly what survives. The cotangent is ``dot / |cross|``,
    trig-free, correctly negative at an obtuse corner, and **never clamped** --
    clamping destroys the operator's linear precision, which is the whole
    reason for cotangent weights over a uniform graph Laplacian.
    """
    u = b - apex
    v = c - apex
    cr = _cross(u, v)
    cross_norm = np.sqrt(_dot(cr, cr))
    dot = _dot(u, v)
    angle = np.arctan2(cross_norm, dot)
    cot = np.where(
        cross_norm > 0.0, dot / np.where(cross_norm > 0.0, cross_norm, 1.0), 0.0
    )
    return angle, cot


def _soup(mesh, region):
    """The triangle soup ``build_triangle_soup`` builds, region filter included.

    Reuses ``_sdf._soup`` rather than transcribing the fan a second time, so
    the two cannot disagree about which diagonal a quad is split on. The region
    filter is applied to the *triangles* by their source cell, which is
    equivalent to filtering cells before fanning because the fan is per-cell.
    """
    from ._sdf import _soup as sdf_soup

    points, verts, corners, source = sdf_soup(mesh)
    mask = _region_mask(mesh, region)
    if mask is not None and len(verts):
        keep = mask[source]
        verts = verts[keep]
        corners = corners[keep]
        source = source[keep]
    return points, verts, corners


def _boundary_vertices(npts, verts):
    """Vertices on an edge used by exactly one triangle.

    From the same edge set ``build_surface_edges`` builds -- **every** triangle,
    degenerate ones included, so the two cannot disagree about what a boundary
    is. Self-loops are filtered out for the reason ``curv_boundary_vertices``
    states: they come only from a repeated corner, whose triangle contributed
    nothing, and letting one mark a boundary would NaN out a good vertex.
    """
    out = np.zeros(npts, dtype=bool)
    if not len(verts):
        return out
    lo = np.minimum(verts, np.roll(verts, -1, axis=1)).ravel()
    hi = np.maximum(verts, np.roll(verts, -1, axis=1)).ravel()
    keys = lo * (np.int64(npts) + 1) + hi
    uniq, counts = np.unique(keys, return_counts=True)
    once = uniq[counts == 1]
    if not len(once):
        return out
    u_lo = once // (np.int64(npts) + 1)
    u_hi = once % (np.int64(npts) + 1)
    real = u_lo != u_hi
    out[u_lo[real]] = True
    out[u_hi[real]] = True
    return out


def _curvature_py(
    mesh,
    mean,
    gaussian,
    dual_area,
    include_boundary,
    record_area,
    record_principal,
    region,
):
    """The numpy twin. ``barycentric`` only -- see the module docstring."""
    if dual_area != "barycentric":
        raise NotImplementedError(
            f"{_PREFIX}the numpy reference does not implement "
            f"dual_area='{dual_area}': its obtuse/non-obtuse test is a discrete "
            "branch on a sign, so two implementations could land on opposite "
            "sides for a right-angled triangle and then disagree "
            "macroscopically rather than in the last ulp. Use the compiled "
            "core for it, or dual_area='barycentric', which is branch-free and "
            "bit-exactly twinned."
        )

    points, verts, corners = _soup(mesh, region)
    npts = len(points)
    nan = float("nan")

    angle_sum = np.zeros(npts, dtype=np.float64)
    lap = np.zeros((npts, 3), dtype=np.float64)
    area = np.zeros(npts, dtype=np.float64)
    normal = np.zeros((npts, 3), dtype=np.float64)
    touched = np.zeros(npts, dtype=bool)
    num_degenerate = 0

    is_boundary = _boundary_vertices(npts, verts)

    if len(verts):
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        cr = _cross(b - a, c - a)
        cr_sq = _dot(cr, cr)
        # Exactly `soup_quality`'s predicate, so the two cannot disagree about
        # which triangles are degenerate. Written as `not (x > 0)` so a NaN
        # counts as degenerate, matching the C++ `!(... > 0.0)`.
        alive = cr_sq > 0.0
        num_degenerate = int(np.count_nonzero(~alive))

        a, b, c = a[alive], b[alive], c[alive]
        cr, cr_sq = cr[alive], cr_sq[alive]
        v = verts[alive]
        tri_area = 0.5 * np.sqrt(cr_sq)

        ang_a, cot_a = _corner(a, b, c)
        ang_b, cot_b = _corner(b, c, a)
        ang_c, cot_c = _corner(c, a, b)

        # Branch-free barycentric dual area: a third of the triangle, each.
        third = tri_area / 3.0

        # `mAngleSum`/`mArea`/`mNormal` take their contributions in (triangle,
        # corner) order. `np.add.at` is unbuffered and applies them in
        # index-array order, which is what reproduces the C++ serial scatter
        # bit for bit -- never `np.bincount`/`reduceat`, which reassociate.
        idx = v.ravel()
        touched[idx] = True
        np.add.at(angle_sum, idx, np.column_stack([ang_a, ang_b, ang_c]).ravel())
        np.add.at(area, idx, np.repeat(third, 3))
        np.add.at(normal, idx, np.repeat(cr, 3, axis=0))

        # The cotangent Laplacian takes its contributions in a DIFFERENT order:
        # per triangle e = 0, 1, 2 with a +/- pair each, so the index sequence
        # is v0, v1, v1, v2, v2, v0. Edge (i, j)'s weight is the cotangent at
        # the corner opposite it. Build the interleaving explicitly; do not
        # "simplify" it to the (triangle, corner) order above.
        w0 = 0.5 * cot_c
        w1 = 0.5 * cot_a
        w2 = 0.5 * cot_b
        d0, d1, d2 = a - b, b - c, c - a
        lidx = np.column_stack(
            [v[:, 0], v[:, 1], v[:, 1], v[:, 2], v[:, 2], v[:, 0]]
        ).ravel()
        lval = np.empty((len(v), 6, 3), dtype=np.float64)
        lval[:, 0] = d0 * w0[:, None]
        lval[:, 1] = d0 * (-w0)[:, None]
        lval[:, 2] = d1 * w1[:, None]
        lval[:, 3] = d1 * (-w1)[:, None]
        lval[:, 4] = d2 * w2[:, None]
        lval[:, 5] = d2 * (-w2)[:, None]
        np.add.at(lap, lidx, lval.reshape(-1, 3))

    # The parallel, per-vertex, order-independent half of the phase split.
    out_area = np.where(touched, area, nan)
    turn = np.where(is_boundary, _PI, _TWO_PI)
    bad = (~touched) | ~(area > 0.0) | (is_boundary & (not include_boundary))
    safe = np.where(area > 0.0, area, 1.0)

    k = (turn - angle_sum) / safe
    h_mag = 0.5 * np.sqrt(_dot(lap, lap)) / safe
    # `lap` accumulates sum(w * (p_i - p_j)), the NEGATIVE of the usual
    # Laplacian, so the mean-curvature normal is -lap; on a convex
    # outward-oriented surface that points along the vertex normal and H must
    # come out positive there.
    sign = np.where(_dot(lap, normal) > 0.0, 1.0, -1.0)
    h = h_mag * sign
    # k1, k2 = H +- sqrt(H^2 - K). The radicand is negative only through
    # discretization error, so clamp rather than produce NaN.
    disc = h * h - k
    root = np.sqrt(np.where(disc > 0.0, disc, 0.0))

    k = np.where(bad, nan, k)
    h = np.where(bad, nan, h)
    principal = np.column_stack(
        [np.where(bad, nan, h + root), np.where(bad, nan, h - root)]
    )

    num_isolated = int(np.count_nonzero(~touched))
    num_boundary = int(np.count_nonzero(touched & is_boundary))
    # A plain left-to-right serial sum, as the C++ finalize loop does.
    # `np.sum` pair-sums and would differ; `cumsum` is sequential by
    # construction, since it must produce every prefix.
    contrib = np.where(touched, _TWO_PI - angle_sum, 0.0)
    total = float(np.cumsum(contrib)[-1]) if npts else 0.0

    out = mesh.copy() if hasattr(mesh, "copy") else mesh
    if mean:
        out.point_data[MEAN_NAME] = h
    if gaussian:
        out.point_data[GAUSSIAN_NAME] = k
    if record_area:
        out.point_data[AREA_NAME] = out_area
    if record_principal:
        out.point_data[PRINCIPAL_NAME] = principal

    from ._sdf import _watertight_py

    return (
        out,
        {
            "num_boundary": num_boundary,
            "num_isolated": num_isolated,
            "num_degenerate": num_degenerate,
            "total_angle_defect": total,
            "quality": _watertight_py(mesh),
        },
    )


def compute_curvature(
    mesh,
    mean=True,
    gaussian=True,
    dual_area="mixed-voronoi",
    include_boundary=False,
    record_area=False,
    record_principal=False,
    region="",
    return_report=False,
):
    """Per-vertex mean and Gaussian curvature of a surface mesh.

    Triangles come from the same fan :func:`meshioplusplus.convert_cells`'s
    ``simplexify`` uses, so a quad mesh's curvature is the curvature of its
    canonical triangulation. A volume or polyhedron block is refused by name
    pointing at ``extract_surface``, a higher-order one pointing at
    ``linearize``.

    :param mesh: a surface mesh (never modified).
    :param mean: attach ``curvature:mean``.
    :param gaussian: attach ``curvature:gaussian``.
    :param dual_area: ``"mixed-voronoi"`` (default) or ``"barycentric"``.
    :param include_boundary: compute a (biased) value at boundary vertices
        instead of leaving them NaN. Isolated vertices are NaN either way.
    :param record_area: also attach ``curvature:area``, the dual area each
        curvature was divided by.
    :param record_principal: also attach ``curvature:principal``, the ``(n, 2)``
        pair ``k1 >= k2`` recovered as ``H +- sqrt(H^2 - K)``.
    :param region: restrict to this named ``Cell`` region; ``""`` takes every
        surface cell.
    :param return_report: also return the counters and the input's surface
        ``quality``.
    :returns: the mesh carrying the requested ``point_data``, or
        ``(mesh, report)`` when ``return_report`` is set.
    :raises ValueError: on a non-surface input (naming the fix) or an unknown
        region name.
    :raises NotImplementedError: when the compiled core is unavailable and
        ``dual_area`` is not ``"barycentric"`` -- see this module's docstring.

    For a closed surface ``report["total_angle_defect"]`` is ``2*pi*chi``
    exactly, whatever the tessellation and whichever dual area: the
    Gauss-Bonnet invariant, and the cheapest check that a result is sane.
    """
    _validate(dual_area)

    out = None
    report = None
    try:
        from . import _core

        res = _core.compute_curvature(
            mesh,
            bool(mean),
            bool(gaussian),
            dual_area,
            bool(include_boundary),
            bool(record_area),
            bool(record_principal),
            str(region),
        )
        out = res["mesh"]
        report = {
            "num_boundary": res["num_boundary"],
            "num_isolated": res["num_isolated"],
            "num_degenerate": res["num_degenerate"],
            "total_angle_defect": res["total_angle_defect"],
            "quality": res["quality"],
        }
    except (ValueError, TypeError):
        # A genuine user error must not fall through to the numpy path, which
        # would either raise something less helpful or silently differ.
        raise
    except Exception:
        out = None

    if out is None:
        out, report = _curvature_py(
            mesh,
            bool(mean),
            bool(gaussian),
            dual_area,
            bool(include_boundary),
            bool(record_area),
            bool(record_principal),
            str(region),
        )

    if report["quality"]["inconsistent_pairs"]:
        from ._common import warn

        warn(
            f"curvature: {report['quality']['inconsistent_pairs']} edge pair(s) "
            f"wind the same way, so the sign of '{MEAN_NAME}' is not "
            "trustworthy; run repair(mesh, fix_orientation=True) first"
        )

    return (out, report) if return_report else out


__all__ = ["compute_curvature"]
