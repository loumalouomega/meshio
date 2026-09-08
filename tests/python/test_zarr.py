"""Zarr mesh I/O in PhysicsNeMo's ``io_zarr`` layout.

Three tiers. The install-error tests need nothing at all and run in the
default matrix; the round-trip and layout tests need only ``zarr`` (the
``interop`` CI job and the coverage job install it); the parity tests --
our store against physicsnemo's own ``to_zarr``/``from_zarr`` -- gate on
physicsnemo and are **not run by public CI**.
"""

import json

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import zarr as zarr_format

from . import helpers


def _write(path, mesh, **kwargs):
    kwargs.setdefault("float32", False)
    zarr_format.write(path, mesh, **kwargs)


def test_the_install_error_names_the_extra(monkeypatch):
    # `sys.modules[name] = None` is the documented way to make an import of
    # `name` raise: no builtins are patched, so pytest's own machinery keeps
    # working while the module under test sees a missing zarr.
    import sys

    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(ImportError, match=r"pip install meshioplusplus\[zarr\]"):
        zarr_format._zarr._require_zarr("zarr")


def test_writing_needs_zarr_3(monkeypatch):
    zarr = pytest.importorskip("zarr")
    monkeypatch.setattr(zarr, "__version__", "2.18.0", raising=False)
    with pytest.raises(ImportError, match="zarr>=3"):
        zarr_format._zarr._require_zarr3("zarr")


@pytest.mark.parametrize(
    "mesh",
    [helpers.tri_mesh, helpers.tet_mesh, helpers.line_mesh, helpers.tri_mesh_2d],
)
def test_round_trip(tmp_path, mesh):
    pytest.importorskip("zarr")
    helpers.write_read(
        tmp_path, _write, zarr_format.read, mesh, 1.0e-12, extension=".zarr"
    )


def test_the_store_matches_the_upstream_layout(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "a.zarr"
    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    mesh.field_data["Re"] = np.array(100.0)
    _write(path, mesh)

    root = zarr.open_group(str(path), mode="r")
    assert dict(root.attrs)[zarr_format._zarr.TYPE_ATTR] == "Mesh"
    assert dict(root.attrs)["__tensordict__"] == {"batch_size": [], "version": 1}
    assert set(k for k, _ in root.arrays()) == {"points", "cells"}
    assert dict(root["point_data"].attrs)["__tensordict__"]["batch_size"] == [
        len(mesh.points)
    ]
    assert root["points"].chunks == (len(mesh.points), 3)
    assert np.allclose(np.asarray(root["point_data"]["T"][...]), mesh.point_data["T"])
    # global_data holds a genuine scalar, not a length-1 array.
    assert root["global_data"]["Re"].shape == ()
    # Consolidated metadata, as upstream writes it.
    assert "consolidated_metadata" in json.loads((path / "zarr.json").read_text())


def test_compression_is_zstd_and_can_be_turned_off(tmp_path):
    pytest.importorskip("zarr")
    compressed = tmp_path / "z.zarr"
    plain = tmp_path / "p.zarr"
    _write(compressed, helpers.tet_mesh)
    _write(plain, helpers.tet_mesh, zstd_level=0)
    codecs = json.loads((compressed / "points" / "zarr.json").read_text())["codecs"]
    assert any(c["name"] == "zstd" for c in codecs)
    codecs = json.loads((plain / "points" / "zarr.json").read_text())["codecs"]
    assert not any(c["name"] == "zstd" for c in codecs)


def test_provenance_rides_the_root_attributes(tmp_path):
    pytest.importorskip("zarr")
    from meshioplusplus._provenance import TAG, read_provenance_lines

    path = tmp_path / "a.zarr"
    _write(path, helpers.tri_mesh)
    lines, recognised = read_provenance_lines(path)
    assert recognised and lines[0] == TAG
    # And through the summary, which is how a user actually sees it.
    assert meshioplusplus.read_metadata(path)["provenance_recognised"] is True


def test_a_write_dataset_store_is_refused_by_name(tmp_path):
    pytest.importorskip("zarr")
    from meshioplusplus._ml import write_dataset

    source = tmp_path / "m.vtu"
    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    meshioplusplus.write(source, mesh)
    store = tmp_path / "ds.zarr"
    write_dataset([str(source)], str(store), format="zarr")
    with pytest.raises(meshioplusplus.ReadError, match="write_dataset"):
        zarr_format.read(store)


def test_a_domain_mesh_store_is_refused_by_name(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "d.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs[zarr_format._zarr.TYPE_ATTR] = "DomainMesh"
    with pytest.raises(meshioplusplus.ReadError, match="DomainMesh"):
        zarr_format.read(path)


def test_a_bare_store_with_points_is_accepted(tmp_path):
    # `from_zarr`'s own rule: no type attribute but a points array present.
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "bare.zarr"
    root = zarr.open_group(str(path), mode="w")
    points = np.asarray(helpers.tri_mesh.points, dtype=np.float64)
    array = root.create_array("points", shape=points.shape, dtype=points.dtype)
    array[...] = points
    mesh = zarr_format.read(path)
    assert np.allclose(mesh.points, points)
    assert len(mesh.cells) == 0


def test_an_empty_cells_array_reads_as_a_point_cloud(tmp_path):
    pytest.importorskip("zarr")
    path = tmp_path / "cloud.zarr"
    cloud = meshioplusplus.Mesh(helpers.tri_mesh.points.copy(), [])
    _write(path, cloud)
    back = zarr_format.read(path)
    assert len(back.cells) == 0


def test_an_unrelated_directory_is_not_removed(tmp_path):
    pytest.importorskip("zarr")
    path = tmp_path / "a.zarr"
    path.mkdir()
    (path / "precious.txt").write_text("keep me")
    with pytest.raises(meshioplusplus.WriteError, match="refusing to remove"):
        _write(path, helpers.tri_mesh)
    assert (path / "precious.txt").is_file()


def test_the_reduction_is_warned(tmp_path, capsys):
    pytest.importorskip("zarr")
    _write(tmp_path / "a.zarr", helpers.tri_quad_mesh)
    assert "tessellated" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# Parity with physicsnemo itself. NOT run by public CI.                       #
# --------------------------------------------------------------------------- #
def test_physicsnemo_reads_what_we_write(tmp_path):
    pytest.importorskip("physicsnemo")
    pytest.importorskip("zarr")
    from physicsnemo.mesh.io import from_zarr

    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    path = tmp_path / "a.zarr"
    zarr_format.write(path, mesh)

    loaded = from_zarr(str(path))
    assert np.allclose(loaded.points.numpy(), mesh.points, atol=1e-6)
    assert np.array_equal(
        loaded.cells.numpy(), np.asarray(mesh.cells[0].data, dtype=np.int64)
    )
    assert np.allclose(loaded.point_data["T"].numpy(), mesh.point_data["T"], atol=1e-6)


def test_we_read_what_physicsnemo_writes(tmp_path):
    pytest.importorskip("physicsnemo")
    pytest.importorskip("zarr")
    from physicsnemo.mesh.io import to_zarr

    from meshioplusplus.physicsnemo import to_physicsnemo

    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    path = tmp_path / "theirs.zarr"
    to_zarr(to_physicsnemo(mesh), str(path))

    back = zarr_format.read(path)
    assert np.allclose(back.points, mesh.points, atol=1e-6)
    assert np.allclose(back.point_data["T"], mesh.point_data["T"], atol=1e-6)
