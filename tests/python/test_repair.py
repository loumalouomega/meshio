"""Tests for surface repair (the ``repair`` operation). Fixtures mirror
``tests/cpp/test_repair.cpp``; the operation is C++-core only, so every
behavioural test is gated on ``_core`` and one test pins the named refusal
when it is absent."""

import math
import sys

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import repair
from meshioplusplus._repair import HOLE_NAME, PARENT_POINT_NAME

from .test_curvature import icosphere, open_cylinder, plane_grid

try:
    from meshioplusplus import _core
except ImportError:  # pragma: no cover - a pure-Python build
    _core = None

needs_core = pytest.mark.skipif(
    _core is None, reason="repair has no pure-Python fallback"
)


def flipped(mesh, which):
    conn = np.asarray(mesh.cells[0].data).copy()
    for w in which:
        conn[w, 1], conn[w, 2] = conn[w, 2], conn[w, 1]
    return mio.Mesh(mesh.points.copy(), [("triangle", conn)])


def signed_volume(mesh):
    """The volume a triangle surface encloses (divergence theorem)."""
    p = np.asarray(mesh.points, dtype=np.float64)
    vol = 0.0
    for cb in mesh.cells:
        if cb.type != "triangle":
            continue
        t = p[np.asarray(cb.data)]
        vol += (
            float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))))
            / 6.0
        )
    return vol


def num_cells(mesh):
    return sum(len(cb.data) for cb in mesh.cells)


def num_edges(mesh):
    edges = set()
    for cb in mesh.cells:
        if cb.type != "triangle":
            continue
        for a, b, c in np.asarray(cb.data):
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add((min(u, v), max(u, v)))
    return len(edges)


def folded_strip():
    th = math.radians(150.0)
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [1 + math.cos(th), 0, math.sin(th)],
            [1 + math.cos(th), 1, math.sin(th)],
        ],
        dtype=np.float64,
    )
    f = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 4], [2, 4, 5]], dtype=np.int64)
    return mio.Mesh(pts, [("triangle", f)])


def two_tets_pinched():
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2], [1, 0, 2], [0, 1, 2]],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [0, 3, 2],
            [3, 5, 4],
            [3, 4, 6],
            [4, 5, 6],
            [3, 6, 5],
        ],
        dtype=np.int64,
    )
    return mio.Mesh(pts, [("triangle", f)])


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


@needs_core
def test_rewinds_a_randomly_flipped_sphere():
    sphere = icosphere(2)
    rng = np.random.default_rng(7)
    which = [c for c in range(len(sphere.cells[0].data)) if rng.integers(10) == 0]
    assert len(which) > 5
    out, rep = repair(flipped(sphere, which), return_report=True)
    assert rep["quality_before"]["inconsistent_pairs"] > 0
    assert rep["quality_after"]["inconsistent_pairs"] == 0
    assert rep["quality_after"]["watertight"]
    assert rep["num_flipped"] == len(which)
    assert rep["num_components"] == 1
    assert rep["num_unorientable"] == 0
    assert signed_volume(out) > 0.0
    assert len(out.points) == len(sphere.points)
    assert len(out.cells) == 1


@needs_core
def test_orients_an_inward_sphere_outward_unless_told_not_to():
    sphere = icosphere(1)
    inward = flipped(sphere, range(len(sphere.cells[0].data)))
    assert signed_volume(inward) < 0.0
    out, rep = repair(inward, return_report=True)
    assert rep["num_flipped"] == len(sphere.cells[0].data)
    assert rep["num_oriented_outward"] == 1
    assert signed_volume(out) > 0.0
    out2, rep2 = repair(inward, orient_outward=False, return_report=True)
    assert rep2["num_flipped"] == 0
    assert signed_volume(out2) < 0.0
    assert rep2["quality_after"]["inconsistent_pairs"] == 0


@needs_core
def test_does_not_touch_a_consistently_wound_crease():
    """The topological half-edge rule, not upstream's normal-dot test: the
    strip's normals disagree by 150 degrees yet it is consistently wound."""
    strip = folded_strip()
    _, rep = repair(strip, fill_holes=False, return_report=True)
    assert rep["quality_before"]["inconsistent_pairs"] == 0
    assert rep["num_flipped"] == 0
    assert rep["num_components"] == 1
    _, rep2 = repair(flipped(strip, [2, 3]), fill_holes=False, return_report=True)
    assert rep2["num_flipped"] == 2
    assert rep2["quality_after"]["inconsistent_pairs"] == 0


@needs_core
def test_fix_orientation_off_leaves_the_winding_alone():
    broken = flipped(icosphere(1), [0, 5, 9])
    _, rep = repair(broken, fix_orientation=False, return_report=True)
    assert rep["num_flipped"] == 0
    assert (
        rep["quality_after"]["inconsistent_pairs"]
        == rep["quality_before"]["inconsistent_pairs"]
    )


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------


@needs_core
def test_fills_an_open_cylinders_two_rims_and_closes_it():
    cyl = open_cylinder(12, 4, 1.0, 2.0)
    out, rep = repair(cyl, max_hole_edges=0, return_report=True)
    assert rep["quality_before"]["boundary_edges"] == 24
    assert rep["num_holes_detected"] == 2
    assert rep["num_holes_filled"] == 2
    assert rep["num_faces_added"] == 24
    assert rep["num_points_added"] == 2
    assert rep["quality_after"]["watertight"]
    assert len(out.cells) == 2
    assert len(out.cells[1].data) == 24
    V, E, F = len(out.points), num_edges(out), num_cells(out)
    assert V - E + F == 2
    assert signed_volume(out) == pytest.approx(12 * 0.5 * math.sin(math.pi / 6) * 2.0)
    closed, rep2 = repair(out, max_hole_edges=0, return_report=True)
    assert rep2["num_holes_detected"] == 0
    assert len(closed.cells) == 2


@needs_core
def test_the_hole_size_limit_is_honoured():
    cyl = open_cylinder(12, 4, 1.0, 2.0)
    out, rep = repair(cyl, max_hole_edges=3, return_report=True)
    assert rep["num_holes_skipped"] == 2
    assert rep["num_faces_added"] == 0
    assert len(out.cells) == 1
    _, rep2 = repair(cyl, fill_holes=False, return_report=True)
    assert rep2["num_holes_detected"] == 0
    assert rep2["quality_after"]["boundary_edges"] == 24


@needs_core
def test_fills_a_single_missing_triangle_consistently():
    sphere = icosphere(2)
    conn = np.asarray(sphere.cells[0].data)[1:]
    holed = mio.Mesh(sphere.points.copy(), [("triangle", conn)])
    out, rep = repair(holed, return_report=True)
    assert rep["num_holes_filled"] == 1
    assert rep["num_faces_added"] == 3
    assert rep["num_flipped"] == 0
    assert rep["quality_after"]["watertight"]
    assert len(out.points) - num_edges(out) + num_cells(out) == 2
    assert signed_volume(out) > 0.0


# ---------------------------------------------------------------------------
# Bowties
# ---------------------------------------------------------------------------


@needs_core
def test_splits_a_pinched_vertex():
    pinched = two_tets_pinched()
    out, rep = repair(pinched, return_report=True)
    assert rep["num_vertices_split"] == 1
    assert rep["num_points_added"] == 1
    assert rep["num_components"] == 2
    assert rep["quality_after"]["watertight"]
    assert len(out.points) == 8
    np.testing.assert_array_equal(out.points[7], out.points[3])
    _, rep2 = repair(pinched, split_non_manifold=False, return_report=True)
    assert rep2["num_vertices_split"] == 0


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


@needs_core
def test_quads_are_triangulated_and_data_follows():
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]], float
    )
    m = mio.Mesh(
        pts,
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
            ("quad", np.array([[1, 4, 5, 2]])),
        ],
        point_data={"T": np.arange(6, dtype=float)},
        cell_data={"tag": [np.array([1, 1]), np.array([2])]},
    )
    out, rep = repair(m, fill_holes=False, return_report=True)
    assert [cb.type for cb in out.cells] == ["triangle", "triangle"]
    assert len(out.cells[1].data) == 2
    assert len(rep["cell_maps"]) == 2
    assert list(rep["cell_maps"][1]) == [0]
    assert len(rep["point_map"]) == 6
    np.testing.assert_array_equal(out.cell_data["tag"][1], [2, 2])
    np.testing.assert_array_equal(out.point_data["T"], m.point_data["T"])


@needs_core
def test_fill_data_provenance_and_regions():
    cyl = open_cylinder(6, 2, 1.0, 2.0)
    n = len(cyl.points)
    cyl.point_data["T"] = cyl.points[:, 2].copy()
    cyl.cell_data["tag"] = [np.full(len(cyl.cells[0].data), 7, dtype=np.int32)]
    cyl.point_sets["top"] = np.arange(12, 18)
    out, rep = repair(cyl, max_hole_edges=0, record_provenance=True, return_report=True)
    assert rep["num_holes_filled"] == 2
    assert len(out.points) == n + 2
    c = sorted(out.point_data["T"][n:])
    assert c[0] == pytest.approx(0.0) and c[1] == pytest.approx(2.0)
    assert out.cell_data["tag"][1].dtype == np.int32
    assert not out.cell_data["tag"][1].any()
    pp = out.point_data[PARENT_POINT_NAME]
    assert pp[0] == 0 and pp[n] == -1 and pp[n + 1] == -1
    assert out.cell_data[HOLE_NAME][0][0] == -1
    assert set(out.cell_data[HOLE_NAME][1]) == {0, 1}
    assert sorted(out.point_sets["top"]) == list(range(12, 18))
    # A float cell array's fill rows are NaN.
    cyl.cell_data["w"] = [np.ones(len(cyl.cells[0].data))]
    out2 = repair(cyl, max_hole_edges=0)
    assert np.isnan(out2.cell_data["w"][1]).all()


@needs_core
def test_a_split_copy_joins_its_sources_point_region():
    pinched = two_tets_pinched()
    pinched.point_sets["apex"] = np.array([3, 5])
    out = repair(pinched)
    assert sorted(out.point_sets["apex"]) == [3, 5, 7]


@needs_core
def test_lower_dimensional_blocks_ride_along():
    m = plane_grid(2)
    m.cells.append(mio.CellBlock("line", np.array([[0, 1], [1, 2]])))
    out, rep = repair(m, return_report=True)
    assert [cb.type for cb in out.cells] == ["triangle", "line", "triangle"]
    assert len(out.cells[1].data) == 2
    assert rep["num_holes_filled"] == 1


@needs_core
def test_welds_first_when_asked():
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 1, 0], [0, 1, 0]], float
    )
    split = mio.Mesh(pts, [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))])
    _, plain = repair(split, fill_holes=False, return_report=True)
    assert plain["num_components"] == 2
    out, welded = repair(
        split, fill_holes=False, weld_tolerance=1e-9, return_report=True
    )
    assert welded["points_welded"] == 2
    assert welded["num_components"] == 1
    assert len(out.points) == 4
    assert len(welded["point_map"]) == 6


@needs_core
def test_refuses_out_of_scope_input_by_name():
    tet = mio.Mesh(np.eye(4, 3), [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(ValueError, match="extract_surface"):
        repair(tet)
    line = mio.Mesh(np.eye(2, 3), [("line", np.array([[0, 1]]))])
    with pytest.raises(ValueError, match="no surface"):
        repair(line)


def test_without_core_the_refusal_is_named(monkeypatch):
    """The operation has no numpy fallback: with the core absent it says so."""
    monkeypatch.setitem(sys.modules, "meshioplusplus._core", None)
    monkeypatch.delattr(mio, "_core", raising=False)
    with pytest.raises(NotImplementedError, match="C\\+\\+-core only"):
        repair(icosphere(0))
