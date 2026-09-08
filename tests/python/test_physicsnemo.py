"""Tests for the PhysicsNeMo adapter (`meshioplusplus.physicsnemo`).

Two halves, the `test_gpu.py` pattern: `graph_sample`, the stats accumulators
and the manifest iteration are pure numpy and run in the default CI matrix
with nothing optional installed; the framework-facing halves gate with
`pytest.importorskip` (torch / torch_geometric / physicsnemo) and are
exercised on a real GPU machine, not by public CI.
"""

from __future__ import annotations

import numpy as np
import pytest

import meshioplusplus
import meshioplusplus.physicsnemo as mpn
from meshioplusplus import DatasetManifest, _gpu
from meshioplusplus._mesh import Mesh
from meshioplusplus._regions import Region


def _mesh(offset=0.0):
    """Two triangles with scalar + vector point data, cell data, regions."""
    points = (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        + offset
    )
    return Mesh(
        points,
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))],
        point_data={
            "T": np.arange(4, dtype=np.float64) + offset,
            "v": np.arange(8, dtype=np.float64).reshape(4, 2),
        },
        cell_data={"mat": [np.array([3.0, 4.0])]},
        regions=[Region("inlet", "point", [0, 1], dim=0, tag=1)],
    )


def _manifest(tmp_path, n=4):
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    manifest = DatasetManifest(base_dir=str(tmp_path))
    for i in range(n):
        meshioplusplus.write(str(cases / f"case_{i}.vtu"), _mesh(float(i)))
        manifest.add(
            f"cases/case_{i}.vtu", id=f"c{i}", split="train" if i < n - 1 else "test"
        )
    return manifest


# --------------------------------------------------------------------------- #
# graph_sample: shapes, dtypes, the contracts                                 #
# --------------------------------------------------------------------------- #
def test_graph_sample_keys_shapes_dtypes():
    s = mpn.graph_sample(_mesh(), fields=["T"], target_fields=["v"])
    assert set(s.arrays) == {"pos", "x", "y", "edge_index", "edge_attr"}
    n = 4
    assert s.arrays["pos"].shape == (n, 3)
    assert s.arrays["x"].shape == (n, 2)  # T + region:inlet one-hot
    assert s.arrays["y"].shape == (n, 2)  # v_0, v_1
    e = s.arrays["edge_index"].shape[1]
    assert s.arrays["edge_index"].shape == (2, e)
    assert s.arrays["edge_attr"].shape == (e, 4)  # dx, dy, dz, norm
    for name in ("pos", "x", "y", "edge_attr"):
        assert s.arrays[name].dtype == np.float32, name
    assert s.arrays["edge_index"].dtype == np.int64


def test_graph_sample_x_columns_are_the_feature_matrix_contract():
    mesh = _mesh()
    s = mpn.graph_sample(mesh, fields=["T"])
    fm = meshioplusplus.feature_matrix(
        mesh, "point", fields=["T"], coords=False, regions=True
    )
    assert s.x_columns == fm.columns == ("T", "region:inlet")
    assert s.schema["x_columns"] == list(fm.columns)
    assert s.schema["x_sources"] == fm.schema["sources"]
    # coordinates never enter x — the MGN convention
    assert "x" not in s.x_columns and "y" not in s.x_columns


def test_graph_sample_edge_attr_matches_the_convention_by_hand():
    s = mpn.graph_sample(_mesh(), float32=False)
    pos, ei, attr = s.arrays["pos"], s.arrays["edge_index"], s.arrays["edge_attr"]
    row, col = ei
    disp = pos[row] - pos[col]  # source minus destination — the sign matters
    expected = np.concatenate(
        (disp, np.linalg.norm(disp, axis=-1, keepdims=True)), axis=-1
    )
    np.testing.assert_array_equal(attr, expected)
    assert attr.dtype == np.float64  # float32=False keeps canonical dtypes


def test_graph_sample_flags():
    mesh = _mesh()
    s = mpn.graph_sample(mesh, edge_features=False, regions=False)
    assert "edge_attr" not in s.arrays and "y" not in s.arrays
    assert s.x_columns == ("T", "v_0", "v_1")  # every point array, no one-hots
    assert s.y_columns == () and s.schema["edge_features"] == []
    directed = mpn.graph_sample(mesh, undirected=False)
    assert (
        directed.arrays["edge_index"].shape[1] == s.arrays["edge_index"].shape[1] // 2
    )
    with pytest.raises(ValueError, match="kind must be"):
        mpn.graph_sample(mesh, kind="bogus")


def test_graph_sample_cell_kind_is_coherent():
    """kind='cell' switches pos/x to centroids and cell-located data, so the
    vertices of edge_index and the rows of pos/x are the same objects."""
    mesh = _mesh()
    s = mpn.graph_sample(mesh, kind="cell", fields=["mat"], float32=False)
    assert s.arrays["pos"].shape == (2, 3)
    np.testing.assert_allclose(s.arrays["pos"][0], mesh.points[[0, 1, 2]].mean(axis=0))
    assert s.x_columns == ("mat",)
    assert s.arrays["x"].shape == (2, 1)
    assert s.arrays["edge_index"].max() < 2


def test_graph_sample_schema_is_json_ready():
    import json

    s = mpn.graph_sample(_mesh(), fields=["T"], target_fields=["v"])
    text = json.dumps(s.schema)
    assert json.loads(text)["y_columns"] == ["v_0", "v_1"]
    assert s.schema["edge_features"] == ["dx", "dy", "dz", "norm"]


# --------------------------------------------------------------------------- #
# manifest walking: iter_samples, field_stats, edge_stats                     #
# --------------------------------------------------------------------------- #
def test_iter_samples_streams_and_filters_by_split(tmp_path):
    manifest = _manifest(tmp_path)
    seen = list(mpn.iter_samples(manifest, split="train", fields=["T"]))
    assert [entry_id for entry_id, _, _ in seen] == ["c0", "c1", "c2"]
    for _, _, sample in seen:
        assert isinstance(sample, mpn.GraphSample)
        # VTU carries no regions, so the re-read mesh has no one-hot column —
        # the format's documented behaviour, not the adapter's.
        assert sample.x_columns == ("T",)
    assert len(list(mpn.iter_samples(manifest))) == 4


def test_field_stats_matches_direct_concatenation(tmp_path):
    manifest = _manifest(tmp_path)
    stats = mpn.field_stats(manifest, split="train", fields=["T", "v"])
    all_t = np.concatenate([np.arange(4, dtype=np.float64) + i for i in range(3)])
    assert stats["T_mean"] == pytest.approx([all_t.mean()])
    assert stats["T_std"] == pytest.approx([all_t.std()])
    assert len(stats["v_mean"]) == len(stats["v_std"]) == 2
    v = np.arange(8, dtype=np.float64).reshape(4, 2)
    assert stats["v_mean"] == pytest.approx(np.tile(v, (3, 1)).mean(axis=0).tolist())
    # the keys are exactly the node_stats.json convention
    assert set(stats) == {"T_mean", "T_std", "v_mean", "v_std"}


def test_edge_stats_matches_direct_computation(tmp_path):
    manifest = _manifest(tmp_path, n=2)
    stats = mpn.edge_stats(manifest)
    attrs = []
    for entry in manifest:
        _, mesh = entry.time_series()[0]
        s = mpn.graph_sample(mesh, float32=False)
        attrs.append(s.arrays["edge_attr"])
    stacked = np.concatenate(attrs)
    assert stats["edge_mean"] == pytest.approx(stacked.mean(axis=0).tolist())
    assert stats["edge_std"] == pytest.approx(stacked.std(axis=0).tolist())


def test_manifest_argument_is_tri_modal(tmp_path):
    manifest = _manifest(tmp_path, n=2)
    path = tmp_path / "m.json"
    manifest.save(path)
    from_path = list(mpn.iter_samples(str(path), fields=["T"]))
    from_obj = list(mpn.iter_samples(manifest, fields=["T"]))
    assert [i for i, _, _ in from_path] == [i for i, _, _ in from_obj]


# --------------------------------------------------------------------------- #
# gating: predicates and the named install errors                             #
# --------------------------------------------------------------------------- #
def test_predicates_return_plain_bools():
    assert mpn.has_physicsnemo() in (True, False)
    assert mpn.has_torch_geometric() in (True, False)


def test_install_errors_name_the_command_and_no_extra(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path, n=1)
    monkeypatch.setattr(_gpu, "_importable", lambda module: False)
    with pytest.raises(ImportError, match="pip install nvidia-physicsnemo") as e:
        mpn.make_reader(manifest)
    # There is deliberately no [physicsnemo] extra — the CuPy/torch precedent.
    assert "meshioplusplus[" not in str(e.value)
    assert "doc/physicsnemo.md" in str(e.value)
    with pytest.raises(ImportError, match="pip install torch_geometric") as e:
        mpn.make_dataset(manifest)
    assert "meshioplusplus[" not in str(e.value)


# --------------------------------------------------------------------------- #
# gated halves — dev GPU box only, never public CI                            #
# --------------------------------------------------------------------------- #
def test_pyg_dataset_yields_data_objects(tmp_path):
    pytest.importorskip("torch_geometric")
    import torch

    manifest = _manifest(tmp_path)
    ds = mpn.make_dataset(manifest, split="train", fields=["T"], target_fields=["v"])
    assert len(ds) == 3
    data = ds[0]
    # VTU carries no regions, so x is the T column alone on the re-read mesh
    assert data.x.shape == (4, 1) and data.x.dtype == torch.float32
    assert data.y.shape == (4, 2)
    assert data.pos.shape == (4, 3)
    assert data.edge_index.dtype == torch.int64
    assert data.edge_attr.shape[1] == 4
    assert ds.schema["x_columns"] == ["T"]
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch.num_graphs == 3 and batch.x.shape == (12, 1)


def test_reader_load_sample_returns_tensor_dict(tmp_path):
    pytest.importorskip("physicsnemo")
    import torch

    manifest = _manifest(tmp_path)
    reader = mpn.make_reader(manifest, split="train", fields=["T"])
    assert len(reader) == 3
    sample = reader._load_sample(0)
    assert isinstance(sample, dict)
    assert set(sample) == {"pos", "x", "edge_index", "edge_attr"}
    assert all(isinstance(t, torch.Tensor) for t in sample.values())
    assert sample["pos"].device.type == "cpu"


# --------------------------------------------------------------------------- #
# t->t+1 pairing (v9.30.0)                                                    #
# --------------------------------------------------------------------------- #
def _series_manifest(tmp_path, steps=3, entries=1):
    """Entries whose cases are k-step series with k-dependent field values."""
    manifest = DatasetManifest(base_dir=str(tmp_path))
    for e in range(entries):
        cases = tmp_path / f"runs{e}"
        cases.mkdir(exist_ok=True)
        for k in range(steps):
            meshioplusplus.write(str(cases / f"out_{k}.vtu"), _mesh(float(k)))
        manifest.add(f"runs{e}/out_*.vtu", id=f"run{e}")
    return manifest


def test_graph_sample_target_mesh_gives_next_step_targets():
    mesh_k, mesh_k1 = _mesh(0.0), _mesh(1.0)
    s = mpn.graph_sample(
        mesh_k,
        fields=["T"],
        target_fields=["T"],
        target_mesh=mesh_k1,
        target_offset=1,
        float32=False,
    )
    # x/pos from the input step, y from the target step
    np.testing.assert_array_equal(s.arrays["x"][:, 0], np.arange(4.0))
    np.testing.assert_array_equal(s.arrays["y"][:, 0], np.arange(4.0) + 1.0)
    np.testing.assert_array_equal(s.arrays["pos"], mesh_k.points)
    assert s.schema["target_offset"] == 1 and s.schema["target_delta"] is False


def test_graph_sample_target_delta_is_the_field_difference():
    mesh_k, mesh_k1 = _mesh(0.0), _mesh(2.5)
    s = mpn.graph_sample(
        mesh_k,
        target_fields=["T", "v"],
        target_mesh=mesh_k1,
        target_delta=True,
        float32=False,
    )
    # T differs by the offset; v is offset-independent so its delta is zero
    np.testing.assert_allclose(s.arrays["y"][:, 0], 2.5)
    np.testing.assert_allclose(s.arrays["y"][:, 1:], 0.0)
    assert s.y_columns == ("T", "v_0", "v_1")
    assert s.schema["target_delta"] is True


def test_graph_sample_pairing_validation_errors():
    mesh = _mesh()
    with pytest.raises(ValueError, match="target_mesh requires target_fields"):
        mpn.graph_sample(mesh, target_mesh=_mesh(1.0))
    with pytest.raises(ValueError, match="target_delta requires target_mesh"):
        mpn.graph_sample(mesh, target_fields=["T"], target_delta=True)
    with pytest.raises(ValueError, match="target_offset must be >= 0"):
        mpn.graph_sample(mesh, target_offset=-1)
    bigger = Mesh(
        np.vstack([mesh.points, [[2.0, 2.0, 0.0]]]),
        mesh.cells,
        point_data={"T": np.arange(5, dtype=np.float64)},
    )
    with pytest.raises(ValueError, match="pairing requires matching meshes"):
        mpn.graph_sample(mesh, target_fields=["T"], target_mesh=bigger)


def test_graph_sample_schema_records_offset_delta_and_edge_options():
    s = mpn.graph_sample(_mesh(), fields=["T"])
    assert s.schema["graph_sample_version"] == mpn.GRAPH_SAMPLE_VERSION == 3
    assert s.schema["target_offset"] == 0
    assert s.schema["target_delta"] is False
    # v3: how the edges were built is part of the recorded contract, so a
    # checkpoint trained on a proximity graph cannot be replayed on a mesh one.
    assert s.schema["proximity"] is None
    assert s.schema["world_edges"] is None
    assert s.schema["world_edge_features"] == []


def test_iter_samples_target_offset_pairs_and_shrinks(tmp_path):
    manifest = _series_manifest(tmp_path, steps=3)
    paired = list(
        mpn.iter_samples(
            manifest,
            fields=["T"],
            target_fields=["T"],
            target_offset=1,
            float32=False,
        )
    )
    assert len(paired) == 2  # 3 steps -> 2 pairs
    for k, (_, _, sample) in enumerate(paired):
        np.testing.assert_array_equal(sample.arrays["x"][:, 0], np.arange(4.0) + k)
        np.testing.assert_array_equal(sample.arrays["y"][:, 0], np.arange(4.0) + k + 1)
        assert sample.schema["target_offset"] == 1
    delta = list(
        mpn.iter_samples(
            manifest,
            target_fields=["T"],
            target_offset=1,
            target_delta=True,
            float32=False,
        )
    )
    for _, _, sample in delta:
        np.testing.assert_allclose(sample.arrays["y"][:, 0], 1.0)


def test_iter_samples_target_offset_skips_short_entries_with_warning(tmp_path):
    manifest = _series_manifest(tmp_path, steps=2)
    with pytest.warns(UserWarning, match="entry 'run0' has 2 step"):
        samples = list(
            mpn.iter_samples(
                manifest, target_fields=["T"], target_offset=5, float32=False
            )
        )
    assert samples == []


def test_field_stats_delta_matches_direct_differences(tmp_path):
    manifest = _series_manifest(tmp_path, steps=4)
    stats = mpn.field_stats(manifest, fields=["T"], delta=True)
    # every consecutive difference is exactly 1.0 per point
    assert set(stats) == {"T_diff_mean", "T_diff_std"}
    assert stats["T_diff_mean"] == pytest.approx([1.0])
    assert stats["T_diff_std"] == pytest.approx([0.0])
    lag2 = mpn.field_stats(manifest, fields=["T"], delta=2)
    assert lag2["T_diff_mean"] == pytest.approx([2.0])
    with pytest.raises(ValueError, match="delta must be True or an int >= 1"):
        mpn.field_stats(manifest, fields=["T"], delta=0)
    # plain stats keep the undecorated keys
    plain = mpn.field_stats(manifest, fields=["T"])
    assert set(plain) == {"T_mean", "T_std"}


def test_pyg_dataset_pairs_targets_with_offset(tmp_path):
    pytest.importorskip("torch_geometric")
    import torch

    manifest = _series_manifest(tmp_path, steps=3)
    ds = mpn.make_dataset(manifest, fields=["T"], target_fields=["T"], target_offset=1)
    assert len(ds) == 2  # 3 steps -> 2 pairs
    for k in range(2):
        data = ds[k]
        torch.testing.assert_close(
            data.y[:, 0], torch.arange(4, dtype=torch.float32) + k + 1
        )
        torch.testing.assert_close(
            data.x[:, 0], torch.arange(4, dtype=torch.float32) + k
        )
    assert ds.schema["target_offset"] == 1
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(ds, batch_size=2)))
    assert batch.num_graphs == 2 and batch.y.shape == (8, 1)


def test_reader_pairs_targets_with_offset(tmp_path):
    pytest.importorskip("physicsnemo")
    import torch

    manifest = _series_manifest(tmp_path, steps=3)
    reader = mpn.make_reader(
        manifest, fields=["T"], target_fields=["T"], target_offset=1
    )
    assert len(reader) == 2
    sample = reader._load_sample(0)
    torch.testing.assert_close(
        sample["y"][:, 0], torch.arange(4, dtype=torch.float32) + 1
    )


# --------------------------------------------------------------------------- #
# the physicsnemo.mesh.Mesh bridge (v9.30.0)                                  #
# --------------------------------------------------------------------------- #
def _mixed_dim_mesh():
    """A hex volume plus a stray line: exercises tessellation AND dropping."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    return Mesh(
        points,
        [
            ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)),
            ("line", np.array([[0, 1]], dtype=np.int64)),
        ],
        point_data={"T": np.arange(8, dtype=np.float64)},
        cell_data={"mat": [np.array([7.0]), np.array([1.0])]},
        field_data={"Re": np.array(100.0), "solver": "kratos"},
    )


def test_physicsnemo_payload_selects_dim_tessellates_and_notes():
    payload = mpn._to_physicsnemo_payload(_mixed_dim_mesh())
    assert payload["manifold_dim"] == 3
    assert payload["cells"].shape == (6, 4)  # hex -> 6 tetra (Freudenthal)
    assert payload["cells"].dtype == np.int64
    assert payload["points"].dtype == np.float32
    # cell_data replicated to the 6 children; the line block's row dropped
    np.testing.assert_array_equal(payload["cell_data"]["mat"], np.full(6, 7.0))
    # numeric field_data lands in global_data; the string is dropped by name
    assert "Re" in payload["global_data"] and "solver" not in payload["global_data"]
    text = " | ".join(payload["notes"])
    assert "tessellated" in text and "line x 1" in text and "'solver'" in text


def test_physicsnemo_payload_dim_selection_and_errors():
    mesh = _mesh()  # triangles with 3-D points
    payload = mpn._to_physicsnemo_payload(mesh, manifold_dim=2, float32=False)
    assert payload["manifold_dim"] == 2
    assert payload["points"].dtype == np.float64
    np.testing.assert_array_equal(payload["cells"], mesh.cells[0].data)
    # regions dropped with a note naming them
    assert any("'inlet'" in n or "inlet" in n for n in payload["notes"])
    with pytest.raises(ValueError, match="no dimension-3 cells"):
        mpn._to_physicsnemo_payload(mesh, manifold_dim=3)
    with pytest.raises(ValueError, match="manifold_dim must be 'auto' or"):
        mpn._to_physicsnemo_payload(mesh, manifold_dim=7)
    flat = Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        [("tetra", np.array([[0, 1, 2, 3]], dtype=np.int64))],
    )
    with pytest.raises(ValueError, match="needs 3-D points"):
        mpn._to_physicsnemo_payload(flat)


def test_from_physicsnemo_payload_maps_types_and_field_data():
    payload = {
        "points": np.eye(4, 3),
        "cells": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        "point_data": {"T": np.arange(4.0)},
        "cell_data": {"mat": np.array([1.0, 2.0])},
        "global_data": {"Re": np.array(100.0)},
    }
    mesh = mpn._from_physicsnemo_payload(payload)
    assert mesh.cells[0].type == "triangle"
    np.testing.assert_array_equal(mesh.cell_data["mat"][0], [1.0, 2.0])
    assert "Re" in mesh.field_data
    with pytest.raises(ValueError, match="map to no simplex"):
        mpn._from_physicsnemo_payload(
            {"points": np.eye(3), "cells": np.zeros((1, 5), dtype=np.int64)}
        )


def test_bridge_install_errors_name_the_command(monkeypatch, tmp_path):
    monkeypatch.setattr(_gpu, "_importable", lambda module: False)
    for fn, arg in ((mpn.to_physicsnemo, _mesh()), (mpn.from_physicsnemo, None)):
        with pytest.raises(ImportError, match="pip install nvidia-physicsnemo") as e:
            fn(arg)
        assert "meshioplusplus[" not in str(e.value)
        assert "doc/physicsnemo.md" in str(e.value)


def test_to_physicsnemo_roundtrip():
    pytest.importorskip("physicsnemo")
    import torch

    mesh = _mesh()
    pm = mpn.to_physicsnemo(mesh)  # emits the dropped-regions warning (stderr)
    assert pm.points.dtype == torch.float32 and pm.points.device.type == "cpu"
    assert pm.cells.dtype == torch.int64 and tuple(pm.cells.shape) == (2, 3)
    torch.testing.assert_close(pm.point_data["T"], torch.arange(4, dtype=torch.float64))
    back = mpn.from_physicsnemo(pm)
    assert back.cells[0].type == "triangle"
    np.testing.assert_array_equal(back.cells[0].data, mesh.cells[0].data)
    np.testing.assert_allclose(back.points, mesh.points)
    np.testing.assert_array_equal(back.point_data["T"], mesh.point_data["T"])
    np.testing.assert_array_equal(back.cell_data["mat"][0], mesh.cell_data["mat"][0])


def test_to_physicsnemo_tessellation_warns(capsys):
    pytest.importorskip("physicsnemo")

    pm = mpn.to_physicsnemo(_mixed_dim_mesh())
    assert tuple(pm.cells.shape) == (6, 4)  # tets
    # notes surface through the repo's rich-console warn(), not the warnings
    # module -- assert on stderr, where every lossy step must be named
    err = capsys.readouterr().err
    assert "tessellated" in err and "line x 1" in err


# --------------------------------------------------------------------------- #
# v10.24.0: the in-package trainer -- dev GPU box only, never public CI        #
# --------------------------------------------------------------------------- #
def test_run_training_and_predict_end_to_end(tmp_path):
    pytest.importorskip("torch_geometric")
    pytest.importorskip("physicsnemo")
    import json
    import os

    manifest = _manifest(tmp_path)
    manifest_path = str(tmp_path / "m.json")
    manifest.save(manifest_path)
    run_dir = tmp_path / "run"
    spec = mpn.default_spec(
        manifest_path,
        ["T"],
        ["v"],
        run_dir=str(run_dir),
        valid_split="test",
        epochs=3,
        batch_size=2,
        hidden_dim=8,
        processor_size=2,
        checkpoint_every=2,
        device="cpu",
    )
    lines = []
    progress = mpn.run_training(spec, log=lines.append)
    assert progress["completed"] and not progress["stopped"]
    assert progress["epoch"] == 3 and progress["epochs"] == 3
    assert any(line.startswith("epoch    0") for line in lines)
    # the run directory layout (physicsnemo/_train.py)
    from meshioplusplus.physicsnemo import _train as t

    rows = t.read_metrics(str(run_dir))
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert all(r["valid_loss"] is not None for r in rows)
    node_stats = json.load(open(run_dir / t.NODE_STATS_FILE))
    assert set(node_stats) == {"T_mean", "T_std", "v_mean", "v_std"}
    assert len(node_stats["v_mean"]) == 2
    ckpts = t.list_checkpoints(str(run_dir), best=progress["best_checkpoint"])
    names = {c["name"] for c in ckpts}
    assert {"best.mdlus", "final.mdlus"} <= names
    assert any(
        c["kind"] == "periodic" for c in ckpts
    ), names  # save_checkpoint at epoch 1
    assert all(c["epoch"] is not None for c in ckpts), ckpts  # every .mdlus has a card
    periodic = [c for c in ckpts if c["kind"] == "periodic"]
    assert os.path.isfile(os.path.join(run_dir, t.CHECKPOINT_DIR, "checkpoint.0.1.pt"))
    card = t.read_json(t.card_path(progress["best_checkpoint"]))
    assert card["x_columns"] == ["T"] and card["y_columns"] == ["v_0", "v_1"]
    assert card["model"]["input_dim_nodes"] == 1 and card["model"]["output_dim"] == 2
    assert card["model"]["input_dim_edges"] == 4
    assert len(card["output_normalization"]["mean"]) == 2

    # predict on the held-out split, written back as data arrays
    out = tmp_path / "pred"
    predictions = mpn.predict(
        progress["best_checkpoint"], manifest_path, split="test", output_dir=str(out)
    )
    assert len(predictions) == 1
    row = predictions[0]
    assert row["entry_id"] == "c3" and row["num_rows"] == 4
    assert row["rmse"] is not None and row["max_error"] >= 0
    mesh = meshioplusplus.read(row["output_path"])
    assert {"v_0_pred", "v_1_pred", "v_0_error", "v_1_error"} <= set(mesh.point_data)
    assert mesh.point_data["v_0_pred"].shape == (4,)
    # v10.32.0: the SAME row from the single-mesh path -- no manifest, no
    # split, no entry. Equality here is what pins the extraction of the
    # per-mesh body out of predict()'s loop as behaviour-preserving.
    single = mpn.predict_file(
        progress["best_checkpoint"],
        str(tmp_path / "cases" / "case_3.vtu"),
        str(out / "single.vtu"),
    )
    assert single["num_rows"] == row["num_rows"]
    assert single["rmse"] == pytest.approx(row["rmse"])
    assert single["max_error"] == pytest.approx(row["max_error"])
    single_mesh = meshioplusplus.read(single["output_path"])
    for name in ("v_0_pred", "v_1_pred"):
        assert np.allclose(single_mesh.point_data[name], mesh.point_data[name])

    # A mesh that carries no truth predicts anyway, and says so rather than
    # measuring an error against its own input.
    truthless = _mesh(3.0)
    del truthless.point_data["v"]
    meshioplusplus.write(str(tmp_path / "truthless.vtu"), truthless)
    bare = mpn.predict_file(
        progress["best_checkpoint"],
        str(tmp_path / "truthless.vtu"),
        str(out / "bare.vtu"),
    )
    assert bare["rmse"] is None and bare["max_error"] is None
    bare_mesh = meshioplusplus.read(bare["output_path"])
    assert {"v_0_pred", "v_1_pred"} <= set(bare_mesh.point_data)
    assert not any(k.endswith("_error") for k in bare_mesh.point_data)

    # and the in-memory form gives the same arrays as the file one
    in_memory, in_row = mpn.predict_mesh(progress["best_checkpoint"], _mesh(3.0))
    assert in_row["num_rows"] == row["num_rows"]
    assert np.allclose(
        in_memory.point_data["v_0_pred"], single_mesh.point_data["v_0_pred"]
    )

    # v10.34.0: a run with a guardrail stores it in the card, and predicting
    # on a wildly different mesh warns while still writing the prediction.
    guarded_dir = tmp_path / "guarded"
    guarded = mpn.run_training(
        t.spec_from_dict(
            {
                **t.spec_to_dict(spec),
                "RunDir": str(guarded_dir),
                "Epochs": 1,
                "Guard": True,
            },
            base_dir=str(tmp_path),
        )
    )
    guarded_card = t.read_json(
        t.card_path(guarded["best_checkpoint"] or guarded["final_checkpoint"])
    )
    assert "guard" in guarded_card and guarded_card["guard"]["threshold"] >= 0

    # A mesh from the training split itself scores in-distribution. (The
    # held-out c3 does NOT, and correctly so: this fixture offsets each case,
    # so with three training cases its centroid really is outside their range
    # -- the guard reporting that is the feature working, not a false alarm.)
    ordinary = mpn.predict_file(
        guarded["best_checkpoint"] or guarded["final_checkpoint"],
        str(tmp_path / "cases" / "case_0.vtu"),
        str(out / "guarded.vtu"),
    )
    assert ordinary["guard"]["verdict"] == "in"

    huge = _mesh(3.0)
    huge.points = np.asarray(huge.points) * 500.0
    meshioplusplus.write(str(tmp_path / "huge.vtu"), huge)
    with pytest.warns(UserWarning, match="out of the distribution"):
        flagged = mpn.predict_file(
            guarded["best_checkpoint"] or guarded["final_checkpoint"],
            str(tmp_path / "huge.vtu"),
            str(out / "huge_pred.vtu"),
        )
    assert flagged["guard"]["verdict"] == "out"
    # ... and the prediction was written anyway: a model cannot refuse to
    # answer, so the honest thing is to answer and say so loudly.
    assert os.path.isfile(flagged["output_path"])

    # a periodic checkpoint predicts too (it carries a card as well)
    again = mpn.predict(
        periodic[0]["path"], manifest_path, split="test", output_dir=str(out / "p")
    )
    assert again[0]["entry_id"] == "c3"
    # feature drift is a named error: a manifest whose meshes lack the field
    with pytest.raises(Exception, match="T"):
        bad = tmp_path / "bad"
        bad.mkdir()
        m = _mesh()
        del m.point_data["T"]
        meshioplusplus.write(str(bad / "x.vtu"), m)
        drift = DatasetManifest(base_dir=str(tmp_path))
        drift.add("bad/x.vtu", id="x", split="test")
        drift.save(str(tmp_path / "drift.json"))
        mpn.predict(
            progress["best_checkpoint"],
            str(tmp_path / "drift.json"),
            split="test",
            output_dir=str(out / "d"),
        )
    # the CLI entry point and the named errors without the frameworks
    from meshioplusplus.physicsnemo import train as trainer

    assert (
        trainer.main(
            [
                "--spec",
                json.dumps(
                    {
                        **t.spec_to_dict(spec),
                        "RunDir": str(tmp_path / "run2"),
                        "Epochs": 1,
                    }
                ),
            ]
        )
        == 0
    )
    assert (tmp_path / "run2" / t.FINAL_CHECKPOINT.replace("final.mdlus", "")).exists()


# --------------------------------------------------------------------------- #
# grid samples: the coarse/fine pair a convolutional model trains on          #
# --------------------------------------------------------------------------- #
def _grid_case(tmp_path, name, phase=0.0, n=8):
    """One high-resolution solve -- the shape a superresolution dataset has."""
    mesh = meshioplusplus.convert_cells(
        meshioplusplus.grid((n, n, n), spacing=(1 / n, 2 / n, 4 / n)),
        mode="simplexify",
    )
    p = mesh.points
    mesh.point_data["T"] = np.sin(3 * p[:, 0] + phase) * np.cos(2 * p[:, 1])
    path = tmp_path / f"{name}.vtu"
    meshioplusplus.write(path, mesh)
    return path


def _grid_manifest(tmp_path, cases=3):
    for c in range(cases):
        _grid_case(tmp_path, f"case_{c}", phase=0.4 * c)
    manifest = meshioplusplus.DatasetManifest(base_dir=str(tmp_path))
    for c in range(cases):
        manifest.add(f"case_{c}.vtu", id=f"c{c}", split="train")
    return manifest


def test_grid_sample_pair_shapes_follow_the_sample_scaling_factor(tmp_path):
    """The pair must agree with what a convolutional upsampler does, which is
    to multiply the *sample* counts -- not the cell counts.

    A coarse 4x4x4-cell grid carries 5x5x5 points, and at s=2 the model emits
    10x10x10. `GridSpec.upscale` would have given 9x9x9 (it multiplies cells,
    which is right for resampling and wrong here), so this pins the one that
    matters.
    """
    mesh = meshioplusplus.read(_grid_case(tmp_path, "one"))
    coarse = meshioplusplus.GridSpec.from_mesh(mesh, resolution=(4, 4, 4))
    sample = mpn.grid_sample_pair(
        mesh, coarse, scaling_factor=2, fields=["T"], target_fields=["T"]
    )
    assert sample.arrays["x"].shape == (1, 5, 5, 5)
    assert sample.arrays["y"].shape == (1, 10, 10, 10)
    assert sample.schema["scaling_factor"] == 2
    assert sample.schema["layout"] == "channels_first_zyx"
    assert coarse.upscale(2).shape == (9, 9, 9)  # the one that would not fit


def test_grid_sample_pair_is_self_supervised_without_a_target(tmp_path):
    """No target mesh means the same mesh supplies both sides -- the ordinary
    shape of a superresolution dataset, not a fallback."""
    mesh = meshioplusplus.read(_grid_case(tmp_path, "one"))
    coarse = meshioplusplus.GridSpec.from_mesh(mesh, resolution=(4, 4, 4))
    sample = mpn.grid_sample_pair(
        mesh, coarse, scaling_factor=2, fields=["T"], target_fields=["T"]
    )
    assert sample.x_channels == ("T",)
    assert sample.y_channels == ("T",)
    assert sample.schema["x_coverage"] == 1.0


def test_grid_sample_pair_refuses_a_mismatched_pair(tmp_path):
    mesh = meshioplusplus.read(_grid_case(tmp_path, "one"))
    coarse = meshioplusplus.GridSpec.from_mesh(mesh, resolution=(4, 4, 4))
    with pytest.raises(ValueError, match="exactly one of fine and scaling_factor"):
        mpn.grid_sample_pair(mesh, coarse, fields=["T"])
    # a fine spec over a different box
    other = meshioplusplus.GridSpec(
        origin=(9.0, 9.0, 9.0), spacing=(1.0, 1.0, 1.0), dims=(9, 9, 9)
    )
    with pytest.raises(ValueError, match="different boxes"):
        mpn.grid_sample_pair(mesh, coarse, fine=other, fields=["T"])
    # same box, but not a whole sample factor
    lo, hi = coarse.bounds
    odd = meshioplusplus.GridSpec(
        origin=lo, spacing=(hi - lo) / np.array([7, 7, 7]), dims=(7, 7, 7)
    )
    with pytest.raises(ValueError, match="upscale_samples"):
        mpn.grid_sample_pair(mesh, coarse, fine=odd, fields=["T"])


def test_grid_sample_pair_squeezes_a_world_axis(tmp_path):
    mesh = meshioplusplus.read(_grid_case(tmp_path, "one"))
    coarse = meshioplusplus.GridSpec.from_mesh(mesh, resolution=(4, 4, 4))
    kwargs = dict(scaling_factor=2, fields=["T"], target_fields=["T"])
    plane = mpn.grid_sample_pair(mesh, coarse, squeeze=2, squeeze_index=0, **kwargs)
    assert plane.arrays["x"].shape == (1, 5, 5)  # world z dropped
    assert plane.arrays["y"].shape == (1, 10, 10)
    averaged = mpn.grid_sample_pair(mesh, coarse, squeeze=2, **kwargs)
    assert averaged.arrays["x"].shape == (1, 5, 5)
    assert not np.array_equal(plane.arrays["x"], averaged.arrays["x"])


def test_iter_grid_samples_walks_a_manifest(tmp_path):
    manifest = _grid_manifest(tmp_path)
    got = list(
        mpn.iter_grid_samples(
            manifest,
            split="train",
            coarse=dict(resolution=(4, 4, 4)),
            scaling_factor=2,
            fields=["T"],
            target_fields=["T"],
        )
    )
    assert [entry_id for entry_id, _, _ in got] == ["c0", "c1", "c2"]
    assert all(s.arrays["y"].shape == (1, 10, 10, 10) for _, _, s in got)


def test_grid_stats_are_not_field_stats(tmp_path):
    """A grid's statistics include the fill wherever the lattice reaches
    outside the mesh, so they are a different number from the nodal ones --
    which is why this is a separate function rather than a mode."""
    manifest = _grid_manifest(tmp_path, cases=2)
    kwargs = dict(
        coarse=dict(resolution=(4, 4, 4)),
        scaling_factor=2,
        fields=["T"],
        target_fields=["T"],
    )
    stats = mpn.grid_stats(manifest, split="train", **kwargs)
    assert set(stats) == {"x_mean", "x_std", "y_mean", "y_std", "coverage"}
    assert len(stats["x_mean"]) == 1 and len(stats["y_mean"]) == 1
    assert stats["coverage"] == 1.0
    # the coarse and fine grids sample the same field at different resolutions,
    # so their statistics are close but not identical
    assert stats["x_mean"][0] != stats["y_mean"][0]


def test_iter_grid_samples_checks_the_pairing_before_reading(tmp_path):
    """A mismatched Target fails once, by name, at index-build time -- not at
    the first epoch that happens to reach the entry."""
    _grid_case(tmp_path, "a")
    _grid_case(tmp_path, "b")
    manifest = meshioplusplus.DatasetManifest(base_dir=str(tmp_path))
    manifest.add(
        {"Paths": ["a.vtu", "b.vtu"]},
        target={"Path": "a.vtu"},
        id="bad",
        validate_source=False,
    )
    with pytest.raises(ValueError, match="pairs 2 source steps with 1 target steps"):
        list(
            mpn.iter_grid_samples(
                manifest,
                coarse=dict(resolution=(2, 2, 2)),
                scaling_factor=2,
                fields=["T"],
            )
        )


def test_grid_pair_shape_matches_a_real_srresnet(tmp_path):
    """The oracle for the whole pairing rule: a real model's output shape.

    `GridSpec.upscale_samples` exists because `upscale` disagrees with what a
    convolutional upsampler does, and the disagreement is an off-by-one that
    surfaces as a torch shape error deep inside a loss. Asserting against the
    model itself is the only way to pin that; a test written against our own
    arithmetic would agree with whichever convention we chose.
    """
    torch = pytest.importorskip("torch")
    srrn = pytest.importorskip("physicsnemo.models.srrn")

    mesh = meshioplusplus.read(_grid_case(tmp_path, "one"))
    for factor in (2, 4):
        coarse = meshioplusplus.GridSpec.from_mesh(mesh, resolution=(4, 4, 4))
        sample = mpn.grid_sample_pair(
            mesh, coarse, scaling_factor=factor, fields=["T"], target_fields=["T"]
        )
        model = srrn.SRResNet(
            in_channels=1,
            out_channels=1,
            conv_layer_size=4,
            n_resid_blocks=1,
            scaling_factor=factor,
        )
        x = torch.from_numpy(sample.arrays["x"]).unsqueeze(0).float()
        out = tuple(model(x).shape[1:])
        assert out == sample.arrays["y"].shape, (
            f"at s={factor} the model emits {out} but the pair's target is "
            f"{sample.arrays['y'].shape}"
        )


# --------------------------------------------------------------------------- #
# proximity and world edges (v10.31.0)                                        #
# --------------------------------------------------------------------------- #
def _cloud(n=40, seed=0):
    """A vertex-only cloud: positions and no connectivity, so a proximity
    neighbourhood is the only graph there is."""
    points = np.random.RandomState(seed).rand(n, 3)
    return Mesh(
        points,
        [("vertex", np.arange(n).reshape(-1, 1))],
        point_data={"T": np.arange(n, dtype=np.float64)},
    )


def test_proximity_replaces_the_mesh_edges():
    cloud = _cloud()
    options = {"method": "radius", "radius": 0.4}
    sample = mpn.graph_sample(cloud, fields=["T"], proximity=options)
    assert np.array_equal(
        sample.arrays["edge_index"], meshioplusplus.proximity_graph(cloud, **options)
    )
    assert np.array_equal(
        sample.arrays["edge_attr"],
        meshioplusplus.edge_vectors(cloud.points, sample.arrays["edge_index"]).astype(
            np.float32
        ),
    )
    assert sample.schema["proximity"] == options


def test_world_edges_ride_beside_the_mesh_edges_rather_than_onto_them():
    mesh = meshioplusplus.grid([3, 3, 3])
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    options = {"method": "knn", "max_neighbors": 4}
    sample = mpn.graph_sample(mesh, fields=["T"], world_edges=options)
    # The mesh edges are untouched -- a world edge set adds, never replaces.
    assert np.array_equal(sample.arrays["edge_index"], meshioplusplus.edge_index(mesh))
    assert np.array_equal(
        sample.arrays["world_edge_index"],
        meshioplusplus.proximity_graph(mesh, **options),
    )
    assert sample.arrays["world_edge_attr"].shape == (
        sample.arrays["world_edge_index"].shape[1],
        4,
    )
    assert sample.schema["world_edges"] == options
    assert sample.schema["world_edge_features"] == ["dx", "dy", "dz", "norm"]


def test_a_periodic_proximity_box_reaches_the_edge_features():
    box = 1.0
    points = np.array([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]])
    cloud = Mesh(
        points,
        [("vertex", np.arange(2).reshape(-1, 1))],
        point_data={"T": np.zeros(2)},
    )
    sample = mpn.graph_sample(
        cloud,
        fields=["T"],
        proximity={"method": "radius", "radius": 0.05, "box_size": box},
    )
    # Hand-computed: the short image is 0.02 across the seam. A box that
    # reached the search but not the features would report 0.98 here.
    assert sample.arrays["edge_attr"][0, 3] == pytest.approx(0.02)


def test_edge_stats_uses_the_same_edges_the_samples_will(tmp_path):
    manifest = _manifest(tmp_path)
    options = {"method": "knn", "max_neighbors": 3}
    stats = mpn.edge_stats(manifest, proximity=options)
    # Recomputed by hand over the same entries, through the public builders.
    accumulated = []
    for entry in manifest.entries():
        for _, mesh in entry.time_series():
            edges = meshioplusplus.proximity_graph(mesh, **options)
            accumulated.append(meshioplusplus.edge_vectors(mesh.points, edges))
    reference = np.concatenate(accumulated)
    assert stats["edge_mean"] == pytest.approx(reference.mean(axis=0).tolist())
    # And they are genuinely different from the mesh-edge statistics.
    assert stats["edge_mean"] != mpn.edge_stats(manifest)["edge_mean"]


def test_edge_stats_reports_world_edges_separately(tmp_path):
    manifest = _manifest(tmp_path)
    stats = mpn.edge_stats(manifest, world_edges={"method": "knn", "max_neighbors": 2})
    assert stats["edge_mean"] == mpn.edge_stats(manifest)["edge_mean"]
    assert len(stats["world_edge_mean"]) == 4
    assert "world_edge_mean" not in mpn.edge_stats(manifest)


def test_proximity_reaches_the_iteration_layer(tmp_path):
    manifest = _manifest(tmp_path)
    samples = list(
        mpn.iter_samples(
            manifest,
            fields=["T"],
            proximity={"method": "radius", "radius": 5.0},
        )
    )
    assert samples and all(s.schema["proximity"] is not None for _, _, s in samples)


def test_a_batch_keeps_the_two_edge_sets_apart(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    mesh = meshioplusplus.grid([3, 3, 3])
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    sample = mpn.graph_sample(
        mesh, fields=["T"], world_edges={"method": "knn", "max_neighbors": 4}
    )
    n = sample.arrays["pos"].shape[0]
    graphs = [
        Data(
            x=torch.from_numpy(sample.arrays["x"]),
            edge_index=torch.from_numpy(sample.arrays["edge_index"]),
            world_edge_index=torch.from_numpy(sample.arrays["world_edge_index"]),
        )
        for _ in range(2)
    ]
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    # PyG increments any attribute whose name contains "index", which is the
    # whole reason the two sets are kept apart rather than concatenated.
    assert int(batch.world_edge_index.max()) >= n
    assert int(batch.edge_index.max()) >= n


# --------------------------------------------------------------------------- #
# single-mesh inference (v10.32.0)                                            #
# --------------------------------------------------------------------------- #
def test_single_mesh_prediction_names_the_missing_framework(monkeypatch):
    # Without the frameworks the entry points fail by name and do not mention a
    # meshioplusplus extra, because there deliberately is none. Monkeypatched
    # rather than skipped, so the assertion holds on a box that HAS them --
    # the `test_install_errors_name_the_command_and_no_extra` convention.
    monkeypatch.setattr(_gpu, "_importable", lambda module: False)
    with pytest.raises(ImportError) as excinfo:
        mpn.predict_file("model.mdlus", "in.vtu", "out.vtu")
    message = str(excinfo.value)
    assert "nvidia-physicsnemo" in message and "meshioplusplus[" not in message
    with pytest.raises(ImportError, match="nvidia-physicsnemo"):
        mpn.predict_mesh("model.mdlus", _mesh())


def test_single_mesh_prediction_is_exported():
    for name in ("predict_mesh", "predict_file"):
        assert name in mpn.__all__ and hasattr(mpn, name)
