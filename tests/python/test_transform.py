"""Tests for the affine transform operation."""

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import transform
from meshioplusplus._transform import _build_matrix, _transform_py


def _cube():
    pts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    return meshioplusplus.Mesh(
        pts, [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    )


def test_translate_moves_points_only():
    mesh = _cube()
    out = transform(mesh, translate=[10, 20, 30])
    assert np.allclose(out.points, mesh.points + [10, 20, 30])
    # connectivity unchanged
    assert np.array_equal(out.cells[0].data, mesh.cells[0].data)


def test_scale_per_axis_and_uniform():
    mesh = _cube()
    out = transform(mesh, scale=[2, 3, 4])
    assert np.allclose(out.points[6], [2, 3, 4])
    out2 = transform(mesh, scale=5)
    assert np.allclose(out2.points[6], [5, 5, 5])


def test_rotate_90_about_z_maps_known_point():
    mesh = _cube()
    out = transform(mesh, rotate=("z", 90))
    # (1,0,0) -> (0,1,0)
    assert np.allclose(out.points[1], [0, 1, 0], atol=1e-12)
    # (0,1,0) -> (-1,0,0)
    assert np.allclose(out.points[3], [-1, 0, 0], atol=1e-12)


def test_scale_units_multiplies_coordinates():
    mesh = _cube()
    out = transform(mesh, scale_units=0.001)
    assert np.allclose(out.points, mesh.points * 0.001)


def test_matrix_equivalent_to_translate():
    mesh = _cube()
    mat = np.eye(4)
    mat[:3, 3] = [1, 2, 3]
    out = transform(mesh, matrix=mat)
    assert np.allclose(out.points, mesh.points + [1, 2, 3])


def test_rotate_vector_point_data():
    mesh = _cube()
    mesh.point_data["v"] = np.tile([1.0, 0.0, 0.0], (8, 1))
    out = transform(mesh, rotate=("z", 90), rotate_vector_data=True)
    # each vector (1,0,0) rotates to (0,1,0)
    assert np.allclose(
        out.point_data["v"], np.tile([0.0, 1.0, 0.0], (8, 1)), atol=1e-12
    )
    # without the flag, vector data is untouched
    out2 = transform(mesh, rotate=("z", 90))
    assert np.allclose(out2.point_data["v"], np.tile([1.0, 0.0, 0.0], (8, 1)))


def test_rotate_vector_cell_data():
    # A vector living on a CELL rotates exactly as one living on a point does.
    # Before v10.33.0 the flag reached point data only and cell data rode
    # through in the old frame -- silently, since nothing about an array says
    # which frame it is in.
    mesh = _cube()
    mesh.cell_data["vel"] = [np.array([[1.0, 0.0, 0.0]])]
    mesh.cell_data["stress"] = [np.diag([1.0, 2.0, 3.0]).reshape(1, 9)]
    mesh.cell_data["mat"] = [np.array([7], dtype=np.int64)]
    out = transform(mesh, rotate=("z", 90), rotate_vector_data=True)
    assert np.allclose(out.cell_data["vel"][0][0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(
        out.cell_data["stress"][0][0].reshape(3, 3),
        np.diag([2.0, 1.0, 3.0]),
        atol=1e-12,
    )
    # an integer tag is never a vector
    assert np.array_equal(out.cell_data["mat"][0], [7])
    # and without the flag cell data is untouched
    plain = transform(mesh, rotate=("z", 90))
    assert np.allclose(plain.cell_data["vel"][0][0], [1.0, 0.0, 0.0])


def test_cpp_matches_python():
    core = pytest.importorskip("meshioplusplus._core")
    mesh = _cube()
    mesh.point_data["v"] = np.tile([1.0, 2.0, 3.0], (8, 1))
    mesh.cell_data["cv"] = [np.array([[1.0, 2.0, 3.0]])]
    mesh.cell_data["ct"] = [np.arange(9, dtype=np.float64).reshape(1, 9)]
    mat = _build_matrix(None, None, ("z", 37.0), None, None)
    got = core.transform(mesh, mat.reshape(-1).tolist(), True)
    ref = _transform_py(mesh, mat, True)
    assert np.allclose(got.points, ref.points, atol=1e-12)
    assert np.allclose(got.point_data["v"], ref.point_data["v"], atol=1e-12)
    # The two engines must agree on cell data as well, or a Windows CI run
    # (which takes the numpy path) would disagree with a Linux one.
    for name in ("cv", "ct"):
        assert np.allclose(got.cell_data[name][0], ref.cell_data[name][0], atol=1e-12)


def test_roundtrip_write_read(tmp_path):
    mesh = _cube()
    out = transform(mesh, translate=[1, 2, 3])
    p = tmp_path / "t.vtu"
    meshioplusplus.write(p, out)
    back = meshioplusplus.read(p)
    assert np.allclose(back.points, out.points)


def test_sets_pass_through():
    mesh = _cube()
    mesh.point_sets = {"a": np.array([0, 1, 2])}
    out = transform(mesh, translate=[1, 0, 0])
    assert np.array_equal(out.point_sets["a"], [0, 1, 2])
