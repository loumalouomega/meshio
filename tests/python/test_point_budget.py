"""Point-cloud budgets: `select_points` / `subsample_points` / `PointBudget`.

The coverage oracle is the **fill distance** -- the largest distance from any
source point to its nearest selected point -- computed by brute force here so
the test depends on nothing but numpy. It is what a token budget is for: a
transformer sees only the selected points, and the fill distance is the size of
the largest region it never sees.
"""

from __future__ import annotations

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import PointBudget, select_points, subsample_points
from meshioplusplus._point_budget import BUDGET_ID_NAME


def _sphere_cloud(n=4000, seed=0):
    """A surface-like cloud: points on the unit sphere, deterministic."""
    rng = np.random.RandomState(seed)
    p = rng.normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _surface_mesh():
    """The skin of a small tet grid, carrying a scalar, a vector and regions."""
    mesh = mio.grid((4, 4, 4), spacing=[0.25, 0.5, 1.0])
    mesh = mio.extract_surface(mio.convert_cells(mesh, mode="simplexify"))
    p = mesh.points
    mesh.point_data["u"] = 1.0 + 2.0 * p[:, 0] - p[:, 2]
    mesh.point_data["v"] = np.column_stack([p[:, 0], p[:, 1], -p[:, 2]])
    mesh.cell_data["tag"] = [np.arange(len(cb)) for cb in mesh.cells]
    lo_x = np.flatnonzero(p[:, 0] == 0.0)
    mesh.regions.append(mio.Region("left", "point", lo_x, dim=2, tag=7))
    mesh.regions.append(mio.Region("faces", "cell", np.arange(3), dim=2, tag=1))
    return mesh


def _fill_distance(points, selected):
    d = ((points[:, None, :] - selected[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min(axis=1)).max())


# --------------------------------------------------------------------------- #
# the value                                                                   #
# --------------------------------------------------------------------------- #
def test_budget_validates_itself():
    b = PointBudget([3, 1, 2], "farthest", 3)
    assert b.indices.dtype == np.int64 and len(b) == 3
    assert list(b.sorted_indices) == [1, 2, 3]
    assert list(b.indices) == [3, 1, 2], "selection order is kept, not sorted"
    with pytest.raises(ValueError, match="unique"):
        PointBudget([1, 1], "farthest", 2)
    with pytest.raises(ValueError, match="count is 2 but there are 3"):
        PointBudget([1, 2, 3], "farthest", 2)
    with pytest.raises(ValueError, match="unknown method"):
        PointBudget([1], "nearest", 1)
    with pytest.raises(ValueError, match="non-negative"):
        PointBudget([-1], "random", 1)


def test_budget_round_trips_through_a_dict():
    b = select_points(_sphere_cloud(200), 16, method="grid", seed=3)
    doc = b.to_dict()
    assert doc["version"] == 1 and doc["method"] == "grid" and doc["count"] == 16
    assert "grid" in doc and doc["grid"]["layout"] == "channels_first_zyx"
    back = PointBudget.from_dict(doc)
    assert back == b
    with pytest.raises(ValueError, match="missing key 'indices'"):
        PointBudget.from_dict({"method": "random"})


def test_take_gathers_in_selection_order():
    cloud = _sphere_cloud(300)
    b = select_points(cloud, 20)
    values = np.arange(300, dtype=float)
    assert np.array_equal(b.take(values), values[b.indices])
    with pytest.raises(ValueError, match="expected a per-point array with 300"):
        b.take(values[:10])


# --------------------------------------------------------------------------- #
# the three selectors                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["farthest", "grid", "random"])
def test_selection_is_unique_exactly_count_and_deterministic(method):
    cloud = _sphere_cloud()
    a = select_points(cloud, 128, method=method, seed=5)
    b = select_points(cloud, 128, method=method, seed=5)
    assert a.count == 128 and len(np.unique(a.indices)) == 128
    assert np.array_equal(a.indices, b.indices)
    assert a.schema["num_points"] == 4000 and a.schema["num_candidates"] == 4000


def test_random_changes_with_the_seed_and_farthest_does_not():
    cloud = _sphere_cloud()
    r0 = select_points(cloud, 64, method="random", seed=0)
    r1 = select_points(cloud, 64, method="random", seed=1)
    assert not np.array_equal(r0.indices, r1.indices)
    f0 = select_points(cloud, 64, seed=0)
    f1 = select_points(cloud, 64, seed=1)
    assert np.array_equal(f0.indices, f1.indices), "start=0 makes FPS seed-free"
    # `start=None` is where the seed reaches FPS.
    s0 = select_points(cloud, 64, seed=0, start=None)
    s1 = select_points(cloud, 64, seed=1, start=None)
    assert s0.schema["start"] != s1.schema["start"]
    assert s0.indices[0] == s0.schema["start"]


def test_farthest_is_greedy_and_prefix_closed():
    """Every prefix of an FPS selection IS the FPS selection at that budget.

    That is the property that makes selection order worth keeping: one
    selection at 8192 serves 4096 and 2048 by slicing.
    """
    cloud = _sphere_cloud(1500)
    big = select_points(cloud, 256)
    small = select_points(cloud, 64)
    assert np.array_equal(big.indices[:64], small.indices)
    assert np.array_equal(big.prefix(64).indices, small.indices)
    assert big.prefix(64).schema["prefix_of"] == 256
    # And each new point really is the farthest from everything before it.
    for i in (1, 5, 40, 255):
        chosen = cloud[big.indices[:i]]
        d2 = ((cloud[:, None, :] - chosen[None, :, :]) ** 2).sum(-1).min(axis=1)
        assert d2[big.indices[i]] == d2.max()


def test_farthest_start_is_honoured_and_validated():
    cloud = _sphere_cloud(500)
    b = select_points(cloud, 10, start=123)
    assert b.indices[0] == 123 and b.schema["start"] == 123
    with pytest.raises(ValueError, match="start=500 is not a point index"):
        select_points(cloud, 10, start=500)


def test_grid_is_prefix_closed_too_and_records_its_lattice():
    cloud = _sphere_cloud()
    big = select_points(cloud, 256, method="grid")
    small = select_points(cloud, 64, method="grid")
    # The lattice depends on the count, so the two selections need not share a
    # prefix -- but a prefix of ONE selection is a valid smaller budget.
    assert big.prefix(64).count == 64
    assert small.schema["num_representatives"] >= 2 * 64
    lattice = mio.GridSpec.from_dict(big.schema["grid"])
    lo, hi = lattice.bounds
    assert np.all(lo <= cloud.min(axis=0) + 1e-12)
    assert np.all(hi >= cloud.max(axis=0) - 1e-12)


def test_farthest_and_grid_cover_the_cloud_far_better_than_random():
    """The measured claim behind the method table.

    Fill distance -- the largest gap a token budget leaves unseen -- on a
    surface cloud: FPS is the reference, `grid` is close to it, and a uniform
    random draw is several times worse at the same budget.
    """
    cloud = _sphere_cloud(6000, seed=1)
    fd = {
        m: _fill_distance(cloud, cloud[select_points(cloud, 200, method=m).indices])
        for m in ("farthest", "grid", "random")
    }
    assert fd["farthest"] < fd["grid"] < fd["random"]
    assert fd["grid"] < 1.5 * fd["farthest"]
    assert fd["random"] > 1.5 * fd["farthest"]


def test_bounds_restrict_the_candidates_but_index_the_whole_mesh():
    cloud = _sphere_cloud()
    box = [0.0, -1.0, -1.0, 1.0, 1.0, 1.0]
    b = select_points(cloud, 50, bounds=box)
    assert np.all(cloud[b.indices, 0] >= 0.0)
    assert b.schema["num_candidates"] < 4000 and b.schema["bounds"] == box
    assert b.indices.max() >= 2000, "indices are into the whole cloud"
    with pytest.raises(ValueError, match="inside the bounds"):
        select_points(cloud, 4000, bounds=box)
    with pytest.raises(ValueError, match="not a candidate inside the bounds"):
        start = int(np.flatnonzero(cloud[:, 0] < 0.0)[0])
        select_points(cloud, 5, bounds=box, start=start)
    with pytest.raises(ValueError, match="six numbers"):
        select_points(cloud, 5, bounds=[0, 1])


def test_count_and_method_are_validated_by_name():
    cloud = _sphere_cloud(100)
    with pytest.raises(ValueError, match="count is 101 but only 100"):
        select_points(cloud, 101)
    with pytest.raises(ValueError, match="at least 1"):
        select_points(cloud, 0)
    with pytest.raises(ValueError, match="unknown method 'nearest'"):
        select_points(cloud, 5, method="nearest")


def test_duplicate_positions_are_refused_not_selected_twice():
    cloud = np.repeat(_sphere_cloud(10), 3, axis=0)  # 30 points, 10 positions
    assert select_points(cloud, 10).count == 10
    with pytest.raises(ValueError, match="only 10 distinct positions"):
        select_points(cloud, 11)
    with pytest.raises(ValueError, match="fewer than the requested 11"):
        select_points(cloud, 11, method="grid")


def test_two_dimensional_points_are_accepted():
    pts = _sphere_cloud(300)[:, :2]
    b = select_points(pts, 30)
    assert b.count == 30
    g = select_points(pts, 30, method="grid")
    assert g.count == 30 and g.schema["grid"]["dims"][2] == 1


# --------------------------------------------------------------------------- #
# subsample_points                                                            #
# --------------------------------------------------------------------------- #
def test_subsample_keeps_points_data_and_point_regions(capsys):
    mesh = _surface_mesh()
    out = subsample_points(mesh, 40, record_ids=True)
    err = capsys.readouterr().err
    assert "dropped" in err and "cells" in err and "faces" in err
    assert len(out.points) == 40
    assert [cb.type for cb in out.cells] == ["vertex"]
    assert np.array_equal(out.cells[0].data.reshape(-1), np.arange(40))
    ids = out.point_data[BUDGET_ID_NAME]
    assert np.array_equal(out.points, mesh.points[ids])
    assert np.array_equal(out.point_data["u"], mesh.point_data["u"][ids])
    assert np.array_equal(out.point_data["v"], mesh.point_data["v"][ids])
    assert not out.cell_data
    names = {(r.name, r.kind) for r in out.regions}
    assert names == {("left", "point")}
    left = out.regions[0]
    assert left.dim == 2 and left.tag == 7
    # every kept "left" point is one whose original index was in the region
    original_left = set(mesh.regions[0].entries.tolist())
    assert all(int(ids[i]) in original_left for i in left.entries)
    assert len(left) == sum(int(i) in original_left for i in ids)


def test_subsample_accepts_a_prebuilt_budget_and_refuses_a_foreign_one():
    mesh = _surface_mesh()
    budget = select_points(mesh, 25, method="grid")
    out = subsample_points(mesh, budget)
    assert np.array_equal(out.points, mesh.points[budget.indices])
    with pytest.raises(ValueError, match="alongside an already-made budget"):
        subsample_points(mesh, budget, method="random")
    small = mio.Mesh(mesh.points[:5], [])
    with pytest.raises(ValueError, match="made for a different mesh"):
        subsample_points(small, budget)


def test_feature_matrix_composes_with_a_budget():
    mesh = _surface_mesh()
    budget = select_points(mesh, 16)
    fm = mio.feature_matrix(mesh)
    tokens = budget.take(fm.matrix)
    assert tokens.shape == (16, fm.matrix.shape[1])
    # The same table, built from the subsampled mesh, agrees column for column
    # (coords + fields + the one surviving point region).
    sub = subsample_points(mesh, budget)
    fm2 = mio.feature_matrix(sub)
    assert fm2.columns == fm.columns
    assert np.allclose(fm2.matrix, tokens)


@pytest.mark.parametrize("ext", ["vtu", "vtp", "ply"])
def test_a_point_cloud_writes_and_reads_back(tmp_path, ext):
    mesh = _surface_mesh()
    out = subsample_points(mesh, 30, record_ids=True)
    path = tmp_path / f"cloud.{ext}"
    mio.write(path, out)
    back = mio.read(path)
    assert len(back.points) == 30
    assert np.allclose(back.points, out.points)
