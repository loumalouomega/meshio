"""Tests for ``shrinkwrap``. Fixtures mirror ``tests/cpp/test_shrinkwrap.cpp``;
the numpy twin is pinned bit-exact against the core wherever the module
docstring promises it, and refuses by name where it does not."""

import math

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import sample_distance, shrinkwrap
from meshioplusplus._shrinkwrap import CLOSEST_CELL_NAME, DISTANCE_NAME, _shrinkwrap_py

from .test_curvature import icosphere, plane_grid

try:
    from meshioplusplus import _core
except ImportError:  # pragma: no cover - a pure-Python build
    _core = None

needs_core = pytest.mark.skipif(_core is None, reason="parity needs the compiled core")


def cloud(pts):
    pts = np.asarray(pts, dtype=np.float64)
    return mio.Mesh(pts, [("vertex", np.arange(len(pts)).reshape(-1, 1))])


def book():
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1]], float
    )
    f = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1], [0, 5, 4]], dtype=np.int64)
    return mio.Mesh(pts, [("triangle", f)])


def test_projects_onto_a_sphere():
    target = icosphere(3)
    source = icosphere(2, radius=1.3)
    out, rep = shrinkwrap(source, target, return_report=True)
    assert rep["num_projected"] == len(source.points)
    assert rep["num_missed"] == 0 and rep["num_skipped"] == 0
    assert rep["max_displacement"] == pytest.approx(0.3, abs=0.02)
    assert rep["quality"]["watertight"]
    d = sample_distance(target, out.points)
    np.testing.assert_allclose(d, 0.0, atol=1e-12)
    assert [cb.type for cb in out.cells] == ["triangle"]
    np.testing.assert_array_equal(out.cells[0].data, source.cells[0].data)


def test_offset_on_a_plane_puts_every_point_at_that_height():
    target = plane_grid(4)
    pts = [
        (0.7 + 0.2 * i, 1.3 + 0.1 * i, (-1 if i % 3 == 0 else 1) * (0.1 + 0.05 * i))
        for i in range(12)
    ]
    out = shrinkwrap(cloud(pts), target, offset=0.25)
    np.testing.assert_allclose(out.points[:, 2], 0.25, atol=1e-15)
    np.testing.assert_array_equal(out.points[:, :2], np.asarray(pts)[:, :2])


def test_max_distance_and_recorded_arrays():
    target = plane_grid(4)
    source = cloud([[1, 1, 0.05], [2, 2, 0.5], [1.5, 2.5, -0.02], [3, 1, -2.0]])
    out, rep = shrinkwrap(
        source,
        target,
        max_distance=0.1,
        record_distance=True,
        record_closest_cell=True,
        return_report=True,
    )
    assert rep["num_projected"] == 2 and rep["num_missed"] == 2
    np.testing.assert_array_equal(out.points[:, 2], [0.0, 0.5, 0.0, -2.0])
    d = out.point_data[DISTANCE_NAME]
    assert d[1] == pytest.approx(0.5) and d[3] == pytest.approx(2.0)
    assert (out.point_data[CLOSEST_CELL_NAME] >= 0).all()


def test_weights_blend_or_select_by_name_or_array():
    target = plane_grid(4)
    source = cloud([[1, 1, 1.0], [2, 2, 1.0], [3, 3, 1.0]])
    source.point_data["w"] = np.array([0.5, 0.0, 1.0])
    out, rep = shrinkwrap(source, target, weights="w", return_report=True)
    np.testing.assert_array_equal(out.points[:, 2], [0.5, 1.0, 0.0])
    assert rep["num_skipped"] == 1
    # An array is attached temporarily and never leaks into the output.
    out2 = shrinkwrap(source, target, weights=np.array([0, 1, 1]))
    np.testing.assert_array_equal(out2.points[:, 2], [1.0, 0.0, 0.0])
    assert "shrinkwrap:weights" not in out2.point_data
    with pytest.raises(ValueError, match="nope"):
        shrinkwrap(source, target, weights="nope")


def test_a_crease_offsets_along_the_bisector():
    """The deliberate divergence from upstream: an edge hit offsets along the
    feature pseudonormal (the bisector), not along one face's normal."""
    s = 0.5 / math.sqrt(2.0)
    out = shrinkwrap(cloud([[0.5, -s, -s]]), book(), offset=0.1)
    b = 0.1 / math.sqrt(2.0)
    np.testing.assert_allclose(out.points[0], [0.5, b, b], atol=1e-15)


def test_a_volume_source_moves_interior_points_too():
    target = icosphere(3)
    tet = mio.Mesh(
        np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]], float),
        [("tetra", np.array([[0, 1, 2, 3]]))],
    )
    out, rep = shrinkwrap(tet, target, return_report=True)
    assert rep["num_projected"] == 4
    np.testing.assert_allclose(sample_distance(target, out.points), 0.0, atol=1e-12)
    assert out.cells[0].type == "tetra"


def test_data_and_sets_survive_and_the_target_is_checked():
    target = plane_grid(3)
    m = icosphere(0)
    m.point_data["T"] = np.arange(12, dtype=float)
    m.point_sets["s"] = np.array([1, 2])
    out = shrinkwrap(m, target)
    np.testing.assert_array_equal(out.point_data["T"], m.point_data["T"])
    assert sorted(out.point_sets["s"]) == [1, 2]
    assert out.points.dtype == m.points.dtype
    tet = mio.Mesh(np.eye(4, 3), [("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(ValueError, match="shrinkwrap: target"):
        shrinkwrap(m, tet)


def test_numpy_twin_refuses_an_angle_weighted_offset():
    with pytest.raises(NotImplementedError, match="area"):
        _shrinkwrap_py(
            cloud([[0, 0, 1]]), plane_grid(2), 0.1, 0.0, "", "", "angle", False, False
        )
    # Offset 0 under "angle", and any offset under "area", are fine.
    _shrinkwrap_py(
        cloud([[0.5, 0.5, 1]]), plane_grid(2), 0.0, 0.0, "", "", "angle", False, False
    )
    _shrinkwrap_py(
        cloud([[0.5, 0.5, 1]]), plane_grid(2), 0.1, 0.0, "", "", "area", False, False
    )


@needs_core
@pytest.mark.parametrize(
    "offset, weight", [(0.0, "angle"), (0.0, "area"), (0.07, "area"), (-0.2, "area")]
)
def test_cpp_matches_python(offset, weight):
    """Exact equality: the projection is + - * / and one sqrt over the same
    totally ordered search the SDF twin already pins."""
    target = icosphere(2)
    source = icosphere(1, radius=1.35)
    source.point_data["w"] = np.linspace(0.2, 1.0, len(source.points))
    for weights in ("", "w"):
        cpp, cpp_rep = shrinkwrap(
            source,
            target,
            offset=offset,
            max_distance=0.5,
            weights=weights or None,
            normal_weight=weight,
            record_distance=True,
            record_closest_cell=True,
            return_report=True,
        )
        py, py_rep = _shrinkwrap_py(
            source, target, offset, 0.5, weights, "", weight, True, True
        )
        np.testing.assert_array_equal(cpp.points, py.points)
        np.testing.assert_array_equal(
            cpp.point_data[DISTANCE_NAME], py.point_data[DISTANCE_NAME]
        )
        np.testing.assert_array_equal(
            cpp.point_data[CLOSEST_CELL_NAME], py.point_data[CLOSEST_CELL_NAME]
        )
        for k in (
            "num_projected",
            "num_missed",
            "num_skipped",
            "max_displacement",
            "quality",
        ):
            assert cpp_rep[k] == py_rep[k], k


@needs_core
def test_cpp_matches_python_on_the_crease():
    src = cloud([[0.5, -0.3, -0.3], [0.2, -0.1, 0.4], [0.9, 0.4, -0.05]])
    cpp = shrinkwrap(src, book(), offset=0.1, normal_weight="area")
    py, _ = _shrinkwrap_py(src, book(), 0.1, 0.0, "", "", "area", False, False)
    np.testing.assert_array_equal(cpp.points, py.points)
