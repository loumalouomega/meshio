"""Tests for per-vertex mean/Gaussian curvature (the ``compute_curvature``
operation). Fixtures mirror ``tests/cpp/test_curvature.cpp``'s
``icosphere``/``open_cylinder``/``plane_grid``, same vertex order, so a
failure in one suite is diagnosable against the other."""

import math

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import compute_curvature
from meshioplusplus._curvature import _curvature_py

try:
    from meshioplusplus import _core
except ImportError:  # pragma: no cover - a pure-Python build
    _core = None

needs_core = pytest.mark.skipif(
    _core is None, reason="dual_area='mixed-voronoi' has no pure-Python fallback"
)

_TWO_PI = 6.283185307179586476925286766559
_PI = 3.141592653589793238462643383279


def icosphere(subdiv, radius=1.0):
    t = (1 + 5**0.5) / 2
    verts = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    verts = [list(v) for v in verts]
    for _ in range(subdiv):
        mid = {}
        new_faces = []

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in mid:
                p = [(verts[i][k] + verts[j][k]) / 2 for k in range(3)]
                verts.append(p)
                mid[key] = len(verts) - 1
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    pts = np.array(verts, dtype=np.float64)
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    pts *= radius
    return mio.Mesh(pts, [("triangle", np.array(faces, dtype=np.int64))])


def open_cylinder(around=12, along=4, radius=1.0, height=2.0):
    pts = []
    for j in range(along + 1):
        for i in range(around):
            a = 2 * math.pi * i / around
            pts.append((radius * math.cos(a), radius * math.sin(a), height * j / along))
    faces = []
    for j in range(along):
        for i in range(around):
            a = j * around + i
            b = j * around + (i + 1) % around
            c = (j + 1) * around + i
            d = (j + 1) * around + (i + 1) % around
            faces += [(a, b, d), (a, d, c)]
    return mio.Mesh(
        np.array(pts, dtype=np.float64), [("triangle", np.array(faces, dtype=np.int64))]
    )


def plane_grid(n=4):
    pts = []
    for j in range(n + 1):
        for i in range(n + 1):
            pts.append((float(i), float(j), 0.0))
    faces = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces += [(a, b, d), (a, d, c)]
    return mio.Mesh(
        np.array(pts, dtype=np.float64), [("triangle", np.array(faces, dtype=np.int64))]
    )


# ---------------------------------------------------------------------------
# Gauss-Bonnet: the tessellation-independent oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subdiv", [0, 1, 2, 3])
@pytest.mark.parametrize("dual_area", ["mixed-voronoi", "barycentric"])
def test_gauss_bonnet_on_closed_surfaces(subdiv, dual_area):
    if dual_area == "mixed-voronoi" and _core is None:
        pytest.skip("mixed-voronoi has no pure-Python fallback")
    m = icosphere(subdiv, radius=7.0)
    out, report = compute_curvature(m, dual_area=dual_area, return_report=True)
    assert abs(report["total_angle_defect"] - 4.0 * math.pi) < 1e-9


def test_gauss_bonnet_is_independent_of_the_dual_area():
    m = icosphere(2, radius=1.0)
    _, mv = compute_curvature(m, dual_area="mixed-voronoi", return_report=True)
    _, bary = compute_curvature(m, dual_area="barycentric", return_report=True)
    assert mv["total_angle_defect"] == bary["total_angle_defect"]


# ---------------------------------------------------------------------------
# Analytic answers: a sphere has constant H = 1/R, K = 1/R^2.
# ---------------------------------------------------------------------------


@needs_core
def test_sphere_has_constant_mean_and_gaussian_curvature():
    m = icosphere(3, radius=1.0)
    out = compute_curvature(m, dual_area="mixed-voronoi")
    h = np.asarray(out.point_data["curvature:mean"])
    k = np.asarray(out.point_data["curvature:gaussian"])
    assert np.all(h > 0)  # outward-oriented convex surface
    assert np.max(np.abs(h - 1.0)) < 1e-4
    assert np.max(np.abs(k - 1.0)) < 0.01


def test_sphere_barycentric_is_cruder_but_correctly_signed():
    m = icosphere(3, radius=1.0)
    out = compute_curvature(m, dual_area="barycentric")
    h = np.asarray(out.point_data["curvature:mean"])
    k = np.asarray(out.point_data["curvature:gaussian"])
    assert np.all(h > 0)
    assert np.max(np.abs(h - 1.0)) < 0.16
    assert np.max(np.abs(k - 1.0)) < 0.16


# ---------------------------------------------------------------------------
# A developable surface (cylinder) separates K (~0) from H (~1/(2R)).
# ---------------------------------------------------------------------------


def test_cylinder_separates_mean_from_gaussian():
    m = open_cylinder(around=24, along=6, radius=1.0, height=2.0)
    out, report = compute_curvature(
        m, dual_area="barycentric", include_boundary=False, return_report=True
    )
    k = np.asarray(out.point_data["curvature:gaussian"])
    h = np.asarray(out.point_data["curvature:mean"])
    interior = ~np.isnan(k)
    assert interior.any()
    assert np.max(np.abs(k[interior])) < 1e-9
    assert np.max(np.abs(np.abs(h[interior]) - 0.5)) < 0.02
    assert report["num_boundary"] > 0


def test_plane_is_flat():
    m = plane_grid(6)
    out = compute_curvature(m, dual_area="barycentric")
    h = np.asarray(out.point_data["curvature:mean"])
    k = np.asarray(out.point_data["curvature:gaussian"])
    interior = ~np.isnan(h)
    assert interior.sum() > 0
    assert np.max(np.abs(h[interior])) < 1e-12
    assert np.max(np.abs(k[interior])) < 1e-12


# ---------------------------------------------------------------------------
# Boundary / isolated vertices.
# ---------------------------------------------------------------------------


def test_boundary_is_nan_unless_opted_in():
    m = open_cylinder(around=8, along=2)
    out, report = compute_curvature(m, dual_area="barycentric", return_report=True)
    h = np.asarray(out.point_data["curvature:mean"])
    assert report["num_boundary"] > 0
    assert np.isnan(h).sum() >= report["num_boundary"]

    out2, report2 = compute_curvature(
        m, dual_area="barycentric", include_boundary=True, return_report=True
    )
    h2 = np.asarray(out2.point_data["curvature:mean"])
    assert not np.isnan(h2).any()
    assert report2["num_boundary"] == report["num_boundary"]


def test_a_closed_surface_has_no_boundary_vertices():
    m = icosphere(1)
    _, report = compute_curvature(m, dual_area="barycentric", return_report=True)
    assert report["num_boundary"] == 0
    assert report["quality"]["watertight"]


def test_isolated_vertex_is_nan_and_counted():
    m = icosphere(0)
    pts = np.vstack([m.points, [[100.0, 100.0, 100.0]]])
    m2 = mio.Mesh(pts, m.cells)
    out, report = compute_curvature(m2, dual_area="barycentric", return_report=True)
    assert report["num_isolated"] == 1
    h = np.asarray(out.point_data["curvature:mean"])
    assert np.isnan(h[-1])


def test_degenerate_triangles_are_skipped_and_counted():
    m = icosphere(0)
    pts = np.vstack([m.points, [m.points[0]]])
    bad_cell = np.array([[0, 0, 1]], dtype=np.int64)
    conn = np.vstack([m.cells[0].data, bad_cell])
    m2 = mio.Mesh(pts, [("triangle", conn)])
    _, report = compute_curvature(m2, dual_area="barycentric", return_report=True)
    assert report["num_degenerate"] == 1


# ---------------------------------------------------------------------------
# Opt-in arrays and principal curvatures.
# ---------------------------------------------------------------------------


def test_optional_arrays_are_opt_in():
    m = icosphere(1)
    out = compute_curvature(m, dual_area="barycentric")
    assert "curvature:area" not in out.point_data
    assert "curvature:principal" not in out.point_data
    out2 = compute_curvature(
        m, dual_area="barycentric", record_area=True, record_principal=True
    )
    assert "curvature:area" in out2.point_data
    assert "curvature:principal" in out2.point_data
    assert np.asarray(out2.point_data["curvature:principal"]).shape == (
        len(m.points),
        2,
    )


def test_principal_curvatures_agree_with_mean_and_gaussian():
    m = icosphere(2)
    out = compute_curvature(m, dual_area="barycentric", record_principal=True)
    h = np.asarray(out.point_data["curvature:mean"])
    k = np.asarray(out.point_data["curvature:gaussian"])
    p = np.asarray(out.point_data["curvature:principal"])
    finite = ~np.isnan(h)
    np.testing.assert_allclose(p[finite, 0] + p[finite, 1], 2 * h[finite], atol=1e-9)
    # k1*k2 == K only where the discriminant H^2-K is non-negative; where it
    # is clamped to zero (discretization noise on a coarse mesh), k1==k2==H
    # instead -- matching curvature.cpp's own clamp rather than a NaN.
    disc = h[finite] ** 2 - k[finite]
    unclamped = disc >= 0
    np.testing.assert_allclose(
        (p[finite, 0] * p[finite, 1])[unclamped], k[finite][unclamped], atol=1e-6
    )
    assert np.all(p[finite, 0] >= p[finite, 1] - 1e-12)


def test_the_dual_areas_partition_the_surface():
    m = icosphere(2, radius=1.0)
    from meshioplusplus import compute_stats

    total_area = compute_stats(m)["total_area"]
    for dual_area in ("mixed-voronoi", "barycentric"):
        if dual_area == "mixed-voronoi" and _core is None:
            continue
        out = compute_curvature(m, dual_area=dual_area, record_area=True)
        a = np.asarray(out.point_data["curvature:area"])
        assert abs(float(np.sum(a)) - total_area) < 1e-9


def test_geometry_and_existing_data_survive():
    m = icosphere(1)
    m.point_data["existing"] = np.arange(len(m.points), dtype=np.float64)
    out = compute_curvature(m, dual_area="barycentric")
    np.testing.assert_array_equal(out.points, m.points)
    np.testing.assert_array_equal(out.cells[0].data, m.cells[0].data)
    np.testing.assert_array_equal(out.point_data["existing"], m.point_data["existing"])


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


def test_refuses_a_volume_mesh_by_name():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    m = mio.Mesh(pts, [("tetra", np.array([[0, 1, 2, 3]], dtype=np.int64))])
    with pytest.raises(ValueError, match="extract_surface"):
        compute_curvature(m)


def test_unknown_region_raises():
    m = icosphere(0)
    with pytest.raises(ValueError):
        compute_curvature(m, region="nope")


def test_unknown_dual_area_raises():
    m = icosphere(0)
    with pytest.raises(ValueError, match="dual area"):
        compute_curvature(m, dual_area="nope")


def test_numpy_twin_refuses_mixed_voronoi():
    m = icosphere(0)
    with pytest.raises(NotImplementedError, match="barycentric"):
        _curvature_py(m, True, True, "mixed-voronoi", False, False, False, "")


# ---------------------------------------------------------------------------
# Orientation: H's sign is not trustworthy when facets disagree.
# ---------------------------------------------------------------------------


def test_inconsistent_orientation_is_reported_not_repaired():
    m = icosphere(1)
    conn = m.cells[0].data.copy()
    conn[0] = conn[0][::-1]  # flip one facet's winding
    m2 = mio.Mesh(m.points.copy(), [("triangle", conn)])
    _, report = compute_curvature(m2, dual_area="barycentric", return_report=True)
    assert report["quality"]["inconsistent_pairs"] > 0
    # Geometry is unchanged -- this reports, it does not repair.
    np.testing.assert_array_equal(m2.cells[0].data, conn)


# ---------------------------------------------------------------------------
# C++ / numpy parity, barycentric only.
# ---------------------------------------------------------------------------


@needs_core
@pytest.mark.parametrize(
    "fixture",
    [icosphere(2), icosphere(3, radius=7.0), open_cylinder()],
    ids=["icosphere2", "icosphere3", "cylinder"],
)
def test_cpp_matches_python(fixture):
    """Exact equality, not a tolerance: `barycentric` is branch-free, so the
    twin reproduces the core bit for bit -- including the atan2-derived
    arrays (curvature:gaussian, total_angle_defect), measured to agree with
    numpy's arctan2 on this platform. See ``_curvature.py``'s module
    docstring for the parity policy and the fallback if a platform disagrees.
    """
    cpp, cpp_report = compute_curvature(
        fixture,
        dual_area="barycentric",
        record_area=True,
        record_principal=True,
        return_report=True,
    )
    py, py_report = _curvature_py(
        fixture, True, True, "barycentric", False, True, True, ""
    )
    for name in (
        "curvature:mean",
        "curvature:gaussian",
        "curvature:area",
        "curvature:principal",
    ):
        np.testing.assert_array_equal(
            np.asarray(cpp.point_data[name]), np.asarray(py.point_data[name])
        )
    assert cpp_report["num_boundary"] == py_report["num_boundary"]
    assert cpp_report["num_isolated"] == py_report["num_isolated"]
    assert cpp_report["num_degenerate"] == py_report["num_degenerate"]
    assert cpp_report["total_angle_defect"] == py_report["total_angle_defect"]
    assert cpp_report["quality"] == py_report["quality"]
