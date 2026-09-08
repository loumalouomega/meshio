"""The PhysicsNeMo ``.pmsh`` memory-mapped directory format.

The pure half runs everywhere: the layout is plain numpy, so the tree, the
dtypes, the empty-array rule and the round trip are all checkable with no
torch, no CUDA and no NVIDIA stack installed. That is the whole point of
hand-rolling the writer rather than delegating to ``Mesh.save``.

The parity half -- our tree against physicsnemo's own ``save``/``load``, in
both directions -- gates on ``pytest.importorskip("physicsnemo")`` and is
**not run by public CI**, which installs neither torch nor physicsnemo. Run it
locally against an interpreter that has them (see doc/formats/pmsh.md).
"""

import json
import os

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import pmsh

from . import helpers


def _write(path, mesh, **kwargs):
    kwargs.setdefault("float32", False)
    pmsh.write(path, mesh, **kwargs)


@pytest.mark.parametrize(
    "mesh",
    [helpers.tri_mesh, helpers.tet_mesh, helpers.line_mesh, helpers.tri_mesh_2d],
)
def test_round_trip(tmp_path, mesh):
    helpers.write_read(tmp_path, _write, pmsh.read, mesh, 1.0e-12, extension=".pmsh")


def test_the_tree_matches_the_tensordict_layout(tmp_path):
    path = tmp_path / "a.pmsh"
    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    _write(path, mesh)

    with open(path / "meta.json") as fh:
        assert json.load(fh)["_type"] == pmsh._pmsh.MESH_TYPE
    with open(path / "_tensordict" / "meta.json") as fh:
        meta = json.load(fh)
    assert meta["points"]["dtype"] == "torch.float64"
    assert meta["cells"]["dtype"] == "torch.int64"
    assert meta["points"]["shape"] == list(mesh.points.shape)
    assert meta["_type"] == pmsh._pmsh.TENSORDICT_TYPE
    for group in pmsh._pmsh.DATA_GROUPS:
        assert meta[group] == {"type": "TensorDict"}
        assert (path / "_tensordict" / group / "meta.json").is_file()

    # Blobs are raw and headerless: size is exactly shape x itemsize.
    blob = path / "_tensordict" / "points.memmap"
    assert blob.stat().st_size == mesh.points.size * 8


def test_blobs_are_little_endian(tmp_path):
    path = tmp_path / "a.pmsh"
    mesh = helpers.tri_mesh.copy()
    _write(path, mesh)
    raw = (path / "_tensordict" / "cells.memmap").read_bytes()
    first = np.asarray(mesh.cells[0].data, dtype=np.int64).ravel()[0]
    assert int.from_bytes(raw[:8], "little") == int(first)


def test_a_zero_element_array_has_no_blob(tmp_path):
    # tensordict does not persist an empty tensor, so `Mesh.save` on a point
    # cloud writes no cells.memmap at all while still declaring [0, 1]. A
    # writer that emits one produces a tree no physicsnemo store has.
    path = tmp_path / "cloud.pmsh"
    cloud = meshioplusplus.Mesh(helpers.tri_mesh.points.copy(), [])
    _write(path, cloud)

    with open(path / "_tensordict" / "meta.json") as fh:
        assert json.load(fh)["cells"]["shape"] == [0, 1]
    assert not (path / "_tensordict" / "cells.memmap").exists()

    back = pmsh.read(path)
    assert len(back.cells) == 0
    assert back.points.shape == cloud.points.shape


@pytest.mark.parametrize("mesh", [helpers.tet_mesh, helpers.tri_quad_mesh])
def test_a_scalar_global_survives(tmp_path, mesh):
    # A 0-d field_data entry stays 0-d, rather than becoming a length-1 array.
    # Both fixtures matter: `tri_quad_mesh` needs tessellating on the way in,
    # which used to destroy the value (NDArray::Size() reported 0 for an empty
    # shape, so the clone copied no bytes) -- fixed in v10.35.0, ABI 11 -> 12.
    path = tmp_path / "a.pmsh"
    mesh = mesh.copy()
    mesh.field_data["Re"] = np.array(12345.678)
    _write(path, mesh)
    back = pmsh.read(path)
    assert back.field_data["Re"].shape == ()
    assert float(back.field_data["Re"]) == 12345.678


def test_a_truncated_store_fails_rather_than_reading_zeros(tmp_path):
    path = tmp_path / "a.pmsh"
    _write(path, helpers.tri_mesh)
    os.unlink(path / "_tensordict" / "points.memmap")
    with pytest.raises(meshioplusplus.ReadError, match="missing"):
        pmsh.read(path)


def test_arrays_are_memory_mapped_and_writeable(tmp_path):
    path = tmp_path / "a.pmsh"
    _write(path, helpers.tri_mesh)
    mesh = pmsh.read(path)
    # copy-on-write: writeable (meshio++'s standing contract) and lazily
    # paged, which is the entire reason this format loads faster than a VTU.
    assert mesh.points.flags.writeable
    assert isinstance(mesh.points.base, np.memmap) or isinstance(mesh.points, np.memmap)
    mesh.points[0, 0] = 42.0  # never reaches the file
    del mesh
    assert pmsh.read(path).points[0, 0] != 42.0


def test_the_single_simplex_reduction_is_warned(tmp_path, capsys):
    path = tmp_path / "a.pmsh"
    _write(path, helpers.tri_quad_mesh)
    assert "tessellated" in capsys.readouterr().err
    back = pmsh.read(path)
    assert [block.type for block in back.cells] == ["triangle"]


def test_a_file_is_refused_by_name(tmp_path):
    path = tmp_path / "a.pmsh"
    path.write_text("not a store")
    with pytest.raises(meshioplusplus.ReadError, match="not a directory"):
        pmsh.read(path)


def test_an_unrelated_directory_is_not_removed(tmp_path):
    path = tmp_path / "a.pmsh"
    path.mkdir()
    (path / "precious.txt").write_text("keep me")
    with pytest.raises(meshioplusplus.WriteError, match="refusing to remove"):
        _write(path, helpers.tri_mesh)
    assert (path / "precious.txt").is_file()


def test_read_metadata_and_provenance_on_a_directory(tmp_path):
    path = tmp_path / "a.pmsh"
    _write(path, helpers.tet_mesh)
    meta = meshioplusplus.read_metadata(path)
    assert meta["num_points"] == len(helpers.tet_mesh.points)
    # pmsh has no free-text slot, and a sidecar would put a file `Mesh.load`
    # never wrote into someone else's layout -- so honestly, nothing found.
    assert meta["provenance"] == []
    assert meta["provenance_recognised"] is False


def test_a_directory_store_is_a_sequence_sample(tmp_path):
    # `os.path.isfile` alone would drop every one of these from the plan.
    from meshioplusplus import read_sequence, write_sequence

    steps = [(float(i), helpers.tri_mesh.copy()) for i in range(3)]
    written = write_sequence(str(tmp_path / "out_{step}.pmsh"), steps)
    assert len(written) == 3
    read_back = list(read_sequence(str(tmp_path / "out_*.pmsh")))
    assert len(read_back) == 3


def test_a_zarr_store_under_a_pmsh_name_is_sniffed(tmp_path):
    # What MeshReader._load_sample does: the extension names the role, the
    # contents name the codec.
    pytest.importorskip("zarr")
    from meshioplusplus import zarr as zarr_format

    path = tmp_path / "a.pmsh"
    zarr_format.write(path, helpers.tri_mesh, float32=False)
    back = pmsh.read(path)
    assert np.allclose(back.points, helpers.tri_mesh.points)


def test_write_to_a_buffer_is_refused():
    import io

    with pytest.raises(meshioplusplus.WriteError, match="multiple files"):
        meshioplusplus.write(io.BytesIO(), helpers.tri_mesh, file_format="pmsh")


# --------------------------------------------------------------------------- #
# Parity with physicsnemo itself. NOT run by public CI.                       #
# --------------------------------------------------------------------------- #
def test_physicsnemo_reads_what_we_write(tmp_path):
    pytest.importorskip("physicsnemo")
    from physicsnemo.mesh import Mesh as PmMesh

    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    mesh.cell_data["mat"] = [np.arange(len(mesh.cells[0].data), dtype=np.int64)]
    path = tmp_path / "a.pmsh"
    pmsh.write(path, mesh)

    loaded = PmMesh.load(str(path))
    assert loaded.points.shape == mesh.points.shape
    assert loaded.cells.shape == np.asarray(mesh.cells[0].data).shape
    assert np.allclose(loaded.points.numpy(), mesh.points)
    assert np.allclose(loaded.point_data["T"].numpy(), mesh.point_data["T"])
    assert np.array_equal(loaded.cell_data["mat"].numpy(), mesh.cell_data["mat"][0])


def test_we_read_what_physicsnemo_writes(tmp_path):
    pytest.importorskip("physicsnemo")
    from meshioplusplus.physicsnemo import to_physicsnemo

    mesh = helpers.tet_mesh.copy()
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    path = tmp_path / "theirs.pmsh"
    to_physicsnemo(mesh).save(prefix=str(path))

    back = pmsh.read(path)
    assert back.points.shape == mesh.points.shape
    assert np.allclose(back.points, mesh.points, atol=1e-6)
    assert np.allclose(back.point_data["T"], mesh.point_data["T"], atol=1e-6)


def test_a_point_cloud_round_trips_through_physicsnemo(tmp_path):
    pytest.importorskip("physicsnemo")
    from physicsnemo.mesh import Mesh as PmMesh

    cloud = meshioplusplus.Mesh(helpers.tri_mesh.points.copy(), [])
    path = tmp_path / "cloud.pmsh"
    pmsh.write(path, cloud)
    loaded = PmMesh.load(str(path))
    # Their own empty-cells sentinel, restored by __post_init__.
    assert tuple(loaded.cells.shape) == (0, 1)
    assert loaded.points.shape[0] == len(cloud.points)
