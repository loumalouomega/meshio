"""Tests for ``sobolev_deform``. The operation is C++-core only; the oracles
here mirror ``tests/cpp/test_sobolev_deform.cpp`` and add the gated parity
test against PhysicsNeMo's own ``sobolev_deform``, which is what the uniform
mean mass and the single global solve were matched for."""

import sys

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import sobolev_deform
from meshioplusplus._sobolev_deform import DISPLACEMENT_NAME

from .test_curvature import plane_grid

try:
    from meshioplusplus import _core
except ImportError:  # pragma: no cover - a pure-Python build
    _core = None

needs_core = pytest.mark.skipif(
    _core is None, reason="sobolev_deform has no pure-Python fallback"
)


def tet_cube(n=3):
    hexes = mio.grid([n, n, n], spacing=[1.0 / n] * 3)
    return mio.convert_cells(hexes, mode="simplexify")


@needs_core
def test_a_constant_displacement_is_preserved_exactly():
    m = plane_grid(6)
    m.point_data["d"] = np.tile([0.3, -0.2, 0.7], (len(m.points), 1))
    out, rep = sobolev_deform(m, "d", 2.0, return_report=True)
    assert rep["converged"] and rep["num_iterations"] == 0 and rep["num_fixed"] == 0
    np.testing.assert_array_equal(out.points, m.points + m.point_data["d"])


@needs_core
def test_length_scale_zero_applies_the_raw_displacement():
    m = plane_grid(4)
    m.point_data["d"] = np.random.default_rng(1).normal(size=(len(m.points), 3))
    out, rep = sobolev_deform(m, "d", 0.0, return_report=True)
    assert rep["num_iterations"] == 0
    np.testing.assert_array_equal(out.points, m.points + m.point_data["d"])
    assert out.points.dtype == m.points.dtype


@needs_core
def test_fixed_points_by_ids_name_array_and_boundary():
    m = plane_grid(5)
    n = len(m.points)
    m.point_data["d"] = np.column_stack(
        [np.zeros(n), np.zeros(n), 0.2 * m.points[:, 0]]
    )
    m.point_data["pin"] = np.zeros(n, dtype=np.int32)
    m.point_data["pin"][7] = 1
    m.point_sets["three"] = np.array([3])
    out, rep = sobolev_deform(
        m, "d", 1.0, fixed_points=[3], fix_boundary=True, return_report=True
    )
    assert rep["converged"]
    assert rep["num_fixed"] == 20  # the rim; 3 is on it
    rim = (
        (m.points[:, 0] == 0)
        | (m.points[:, 0] == 5)
        | (m.points[:, 1] == 0)
        | (m.points[:, 1] == 5)
    )
    np.testing.assert_array_equal(out.points[rim], m.points[rim])
    _, rep2 = sobolev_deform(m, "d", 1.0, fixed_points="pin", return_report=True)
    assert rep2["num_fixed"] == 1
    _, rep3 = sobolev_deform(m, "d", 1.0, fixed_points="three", return_report=True)
    assert rep3["num_fixed"] == 1
    with pytest.raises(ValueError, match="no point_set or point_data"):
        sobolev_deform(m, "d", 1.0, fixed_points="missing")


@needs_core
def test_damps_high_frequencies_more_than_low_ones():
    def rel_change(field):
        m = plane_grid(10)
        m.point_data["d"] = field(m.points)
        out = sobolev_deform(m, "d", 2.0)
        u = out.points - m.points
        d = m.point_data["d"]
        return float(np.linalg.norm(u - d) / np.linalg.norm(d))

    def checker(p):
        s = ((p[:, 0].astype(int) + p[:, 1].astype(int)) % 2) * 0.2 - 0.1
        return np.column_stack([np.zeros(len(p)), np.zeros(len(p)), s])

    def linear(p):
        return np.column_stack([np.zeros(len(p)), np.zeros(len(p)), 0.02 * p[:, 0]])

    c, lin = rel_change(checker), rel_change(linear)
    assert c > 0.9 and lin < 0.2 and c > 4 * lin


@needs_core
def test_runs_on_every_dimension_and_records_the_filtered_field():
    vol = tet_cube(3)
    n = len(vol.points)
    vol.point_data["d"] = np.column_stack(
        [
            0.02 * np.where(np.arange(n) % 2, 1.0, -1.0),
            np.zeros(n),
            0.01 * vol.points[:, 2],
        ]
    )
    out, rep = sobolev_deform(vol, "d", 0.5, record_filtered=True, return_report=True)
    assert rep["converged"] and rep["num_iterations"] > 0 and rep["num_isolated"] == 0
    np.testing.assert_allclose(
        out.point_data[DISPLACEMENT_NAME], out.points - vol.points, atol=1e-14
    )
    assert out.cells[0].type == "tetra"
    line = mio.Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        [("line", np.array([[0, 1], [1, 2]]))],
    )
    line.point_data["d"] = np.array([[0.0, 0.1], [0.0, -0.1], [0.0, 0.1]])
    out2 = sobolev_deform(line, "d", 0.5)
    assert out2.points.shape == (3, 2)


@needs_core
def test_reports_non_convergence_instead_of_throwing(capsys):
    m = plane_grid(10)
    p = m.points
    s = ((p[:, 0].astype(int) + p[:, 1].astype(int)) % 2) * 0.2 - 0.1
    m.point_data["d"] = np.column_stack([np.zeros(len(p)), np.zeros(len(p)), s])
    out, rep = sobolev_deform(m, "d", 3.0, max_iterations=1, return_report=True)
    assert not rep["converged"] and rep["num_iterations"] == 1 and rep["residual"] > 0
    assert np.isfinite(out.points).all()
    assert "did not converge" in capsys.readouterr().err


@needs_core
def test_refuses_non_simplices_and_bad_arrays_by_name():
    quad = mio.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
        [("quad", np.array([[0, 1, 2, 3]]))],
    )
    quad.point_data["d"] = np.zeros((4, 3))
    with pytest.raises(ValueError, match="simplexify"):
        sobolev_deform(quad, "d", 1.0)
    tri = plane_grid(1)
    with pytest.raises(ValueError, match="no point_data"):
        sobolev_deform(tri, "missing", 1.0)
    tri.point_data["bad"] = np.zeros(len(tri.points))
    with pytest.raises(ValueError, match="must be"):
        sobolev_deform(tri, "bad", 1.0)


@needs_core
def test_data_and_sets_survive():
    m = plane_grid(3)
    m.point_data["d"] = np.full((len(m.points), 3), 0.05)
    m.point_data["T"] = np.arange(len(m.points), dtype=float)
    m.point_sets["s"] = np.array([0, 1])
    out = sobolev_deform(m, "d", 0.3)
    np.testing.assert_array_equal(out.point_data["T"], m.point_data["T"])
    assert sorted(out.point_sets["s"]) == [0, 1]
    assert out.points.dtype == m.points.dtype


def test_without_core_the_refusal_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "meshioplusplus._core", None)
    monkeypatch.delattr(mio, "_core", raising=False)
    m = plane_grid(1)
    m.point_data["d"] = np.zeros((4, 3))
    with pytest.raises(NotImplementedError, match="C\\+\\+-core only"):
        sobolev_deform(m, "d", 1.0)


@needs_core
def test_matches_physicsnemo_reference():
    """Gated parity against physicsnemo.mesh.sobolev_deform, on a triangle
    surface and a tetrahedral volume, with and without fixed points."""
    torch = pytest.importorskip("torch")
    pnm = pytest.importorskip("physicsnemo.mesh")
    if not hasattr(pnm.Mesh, "sobolev_deform"):
        pytest.skip("physicsnemo.mesh.Mesh.sobolev_deform not available (needs 2.2+)")

    def run(m, l, fixed):
        cpp, rep = sobolev_deform(m, "d", l, fixed_points=fixed, return_report=True)
        assert rep["converged"]
        pts = torch.tensor(np.asarray(m.points, dtype=np.float64))
        cells = torch.tensor(np.asarray(m.cells[0].data, dtype=np.int64))
        disp = torch.tensor(np.asarray(m.point_data["d"], dtype=np.float64))
        pm = pnm.Mesh(points=pts, cells=cells)
        fp = None
        if fixed is not None:
            fp = torch.zeros(len(m.points), dtype=torch.bool)
            fp[np.asarray(fixed)] = True
        ref = pm.sobolev_deform(
            disp, length_scale=l, fixed_points=fp, max_iterations=512
        )
        diag = float(np.linalg.norm(np.ptp(np.asarray(m.points), axis=0)))
        np.testing.assert_allclose(
            cpp.points, ref.points.numpy(), rtol=1e-6, atol=1e-9 * diag
        )

    rng = np.random.default_rng(3)
    tri = plane_grid(6)
    tri.point_data["d"] = 0.1 * rng.normal(size=(len(tri.points), 3))
    run(tri, 1.5, None)
    run(tri, 2.5, [0, 7, 20])
    vol = tet_cube(3)
    vol.point_data["d"] = 0.05 * rng.normal(size=(len(vol.points), 3))
    run(vol, 0.6, None)
    run(vol, 0.6, list(range(0, len(vol.points), 5)))
