"""OpenUSD stages: geometry, primvars and time samples.

The gate tests need nothing; everything else needs ``usd-core`` and is run by
the ``interop`` CI job and the coverage job, which install it.
"""

import sys

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import usd

from . import helpers


def test_the_install_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pxr", None)
    with pytest.raises(ImportError, match=r"pip install meshioplusplus\[usd\]"):
        usd._usd._require_usd("usd")


def test_it_is_registered_as_a_series_writer_and_a_timed_reader():
    import inspect

    from meshioplusplus import _sequence

    assert "usd" in _sequence._SERIES_WRITERS
    assert "usd" in _sequence._TIME_CAPABLE_READERS
    # The multi-step guard probes this reader, so it must genuinely take one.
    assert "time_step" in inspect.signature(usd.read).parameters


# `tri_quad_mesh` is deliberately absent: it has THREE blocks (tri, quad, tri)
# and a USD prim holds ONE face list, so block identity is not a thing this
# format can preserve -- the split-by-vertex-count test below pins what it
# does instead.
@pytest.mark.parametrize("mesh", [helpers.tri_mesh, helpers.quad_mesh])
def test_round_trip(tmp_path, mesh):
    pytest.importorskip("pxr")
    helpers.write_read(tmp_path, usd.write, usd.read, mesh, 1.0e-6, extension=".usda")


def test_ngons_are_kept_rather_than_triangulated(tmp_path):
    # USD holds n-gons natively (faceVertexCounts), so unlike the trimesh and
    # physicsnemo bridges this writer never simplexifies.
    pytest.importorskip("pxr")
    path = tmp_path / "a.usda"
    usd.write(path, helpers.quad_mesh)
    assert [block.type for block in usd.read(path).cells] == ["quad"]

    text = path.read_text()
    assert "faceVertexCounts = [4" in text


def test_a_mixed_surface_splits_by_vertex_count(tmp_path):
    pytest.importorskip("pxr")
    path = tmp_path / "a.usda"
    usd.write(path, helpers.tri_quad_mesh)
    back = usd.read(path)
    assert [block.type for block in back.cells] == ["triangle", "quad"]


def test_a_volume_mesh_is_written_as_its_skin(tmp_path, capsys):
    pytest.importorskip("pxr")
    path = tmp_path / "a.usda"
    mesh = meshioplusplus.grid([1, 1, 1])
    mesh.cell_data["mat"] = [np.array([7], dtype=np.int64)]
    mesh.point_data["T"] = np.arange(len(mesh.points), dtype=np.float64)
    usd.write(path, mesh)
    assert "boundary surface" in capsys.readouterr().err

    back = usd.read(path)
    assert [block.type for block in back.cells] == ["quad"]
    assert len(back.cells[0].data) == 6
    # The hexahedron's own tag reaches every one of its faces: extract_surface
    # carries only the parent-cell map, so the values are gathered through it.
    assert np.allclose(back.cell_data["mat"][0], 7)
    assert np.allclose(back.point_data["T"], mesh.point_data["T"])


def test_a_cell_less_mesh_becomes_a_point_cloud(tmp_path):
    pytest.importorskip("pxr")
    path = tmp_path / "cloud.usda"
    cloud = meshioplusplus.Mesh(helpers.tri_mesh.points.copy(), [])
    usd.write(path, cloud)
    assert 'def Points "' in path.read_text()
    back = usd.read(path)
    assert len(back.cells) == 0
    assert np.allclose(back.points, cloud.points)


def test_the_crate_encoding_round_trips(tmp_path):
    # `.usda` is text and `.usd`/`.usdc` the binary crate; the extension picks
    # it, so both must read back.
    pytest.importorskip("pxr")
    path = tmp_path / "a.usdc"
    usd.write(path, helpers.tri_mesh)
    assert not path.read_bytes().startswith(b"#usda")
    assert np.allclose(usd.read(path).points, helpers.tri_mesh.points, atol=1e-6)


def test_primvar_interpolation_and_element_size(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    path = tmp_path / "a.usda"
    mesh = helpers.tri_mesh.copy()
    mesh.point_data["v"] = np.tile([1.0, 2.0, 3.0], (len(mesh.points), 1))
    mesh.point_data["wide"] = np.zeros((len(mesh.points), 6))
    mesh.cell_data["mat"] = [np.arange(len(mesh.cells[0].data), dtype=np.int64)]
    usd.write(path, mesh)

    # The stage must stay referenced: a prim does not keep its own stage
    # alive, and pxr invalidates it the moment the stage is collected.
    stage = Usd.Stage.Open(str(path))
    primvars = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(usd.DEFAULT_PRIM_PATH))
    assert str(primvars.GetPrimvar("v").GetInterpolation()) == "vertex"
    assert str(primvars.GetPrimvar("mat").GetInterpolation()) == "uniform"
    assert primvars.GetPrimvar("wide").GetElementSize() == 6
    assert "float3[] primvars:v" in path.read_text()

    back = usd.read(path)
    assert back.point_data["wide"].shape == (len(mesh.points), 6)
    assert np.allclose(back.point_data["v"], mesh.point_data["v"], atol=1e-6)


def test_stage_metadata_is_honoured(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    path = tmp_path / "a.usda"
    usd.write(
        path,
        helpers.tri_mesh,
        prim_path="/scene/part",
        up_axis="Y",
        meters_per_unit=0.001,
    )
    stage = Usd.Stage.Open(str(path))
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Y"
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.001)
    assert stage.GetPrimAtPath("/scene/part").IsValid()


def test_a_relative_prim_path_is_refused(tmp_path):
    pytest.importorskip("pxr")
    with pytest.raises(meshioplusplus.WriteError, match="absolute"):
        usd.write(tmp_path / "a.usda", helpers.tri_mesh, prim_path="relative/path")


def test_an_unknown_up_axis_is_refused(tmp_path):
    pytest.importorskip("pxr")
    with pytest.raises(meshioplusplus.WriteError, match="up_axis"):
        usd.write(tmp_path / "a.usda", helpers.tri_mesh, up_axis="X")


def test_provenance_rides_the_layer_documentation(tmp_path):
    pytest.importorskip("pxr")
    from meshioplusplus._provenance import TAG, read_provenance_lines

    path = tmp_path / "a.usda"
    usd.write(path, helpers.tri_mesh)
    lines, recognised = read_provenance_lines(path)
    assert recognised and lines[0] == TAG
    assert meshioplusplus.read_metadata(path)["provenance_recognised"] is True


def test_two_prims_are_concatenated_with_a_prim_id(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, Vt

    path = tmp_path / "two.usda"
    usd.write(path, helpers.tri_mesh)
    stage = Usd.Stage.Open(str(path))
    second = UsdGeom.Mesh.Define(stage, "/meshioplusplus/second")
    points = np.asarray(helpers.tri_mesh.points, dtype=np.float32) + 10.0
    second.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    second.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.array([3], np.int32)))
    second.GetFaceVertexIndicesAttr().Set(
        Vt.IntArray.FromNumpy(np.array([0, 1, 2], np.int32))
    )
    stage.GetRootLayer().Save()

    back = usd.read(path)
    assert "usd:prim" in back.cell_data
    assert set(np.concatenate(back.cell_data["usd:prim"]).tolist()) == {0, 1}
    assert len(back.field_data["usd:prim_paths"]) == 2


# --------------------------------------------------------------------------- #
# Time samples                                                                #
# --------------------------------------------------------------------------- #
def _series(tmp_path, remesh=False):
    from meshioplusplus import write_sequence

    def steps():
        for index, time in enumerate([0.0, 0.5, 1.0]):
            mesh = meshioplusplus.grid(
                [1, 1, 1] if not (remesh and index) else [2, 1, 1]
            )
            mesh.point_data["T"] = np.full(len(mesh.points), float(index))
            yield time, mesh

    path = tmp_path / "s.usda"
    write_sequence(str(path), steps())
    return path


def test_a_single_write_authors_no_time_samples(tmp_path):
    pytest.importorskip("pxr")
    path = tmp_path / "a.usda"
    usd.write(path, helpers.tri_mesh)
    # A default-value sample: a viewer with no timeline sees plain geometry.
    assert ".timeSamples" not in path.read_text()
    assert meshioplusplus.read_metadata(path)["time_values"] == []


def test_a_series_reports_its_steps(tmp_path):
    pytest.importorskip("pxr")
    from meshioplusplus import _sequence

    path = _series(tmp_path)
    assert meshioplusplus.read_metadata(path)["time_values"] == [0.0, 0.5, 1.0]
    assert _sequence.num_steps(str(path)) == 3


def test_time_step_selects_a_sample(tmp_path):
    pytest.importorskip("pxr")
    path = _series(tmp_path)
    assert np.allclose(usd.read(path, time_step=0).point_data["T"], 0.0)
    assert np.allclose(usd.read(path, time_step=-1).point_data["T"], 2.0)


def test_an_out_of_range_step_names_the_count(tmp_path):
    pytest.importorskip("pxr")
    path = _series(tmp_path)
    with pytest.raises(meshioplusplus.ReadError, match="carries 3 step"):
        usd.read(path, time_step=9)


def test_topology_is_authored_only_when_it_changes(tmp_path):
    pytest.importorskip("pxr")
    import re

    def samples(path):
        text = path.read_text()
        match = re.search(
            r"int\[\] faceVertexIndices\.timeSamples = \{(.*?)\n\s*\}", text, re.S
        )
        return re.findall(r"^\s*([\d.]+):", match.group(1), re.M) if match else []

    assert samples(_series(tmp_path / "fixed")) == ["0"]
    assert samples(_series(tmp_path / "remeshed", remesh=True)) == ["0", "0.5"]


def test_a_multi_step_stage_refuses_a_single_step_target(tmp_path):
    pytest.importorskip("pxr")
    from meshioplusplus import run_pipeline

    path = _series(tmp_path)
    doc = {
        "Version": 1,
        "Input": {"Path": str(path)},
        "Output": {"Path": str(tmp_path / "out.vtu")},
    }
    with pytest.raises(Exception, match=r"\{step\}|time.step|multi"):
        run_pipeline(doc)
