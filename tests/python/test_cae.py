"""The CAE ``.npz`` sample layout PhysicsNeMo's DoMINO/Transolver pipes read.

Every test here runs in the default matrix: the format is pure numpy, and the
pins below are the ones the Kratos exporter's own suite makes -- 36 flat int32
face indices for a unit cube, a total area of 6, unit normals, and a linear
nodal field averaged onto a triangle equalling its value at the centroid.
"""

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import cae

from . import helpers

FIELD = "T"


def _cube():
    """A unit cube with a linear nodal field and a per-cell tag."""
    mesh = meshioplusplus.grid([1, 1, 1])
    points = np.asarray(mesh.points, dtype=np.float64)
    mesh.point_data[FIELD] = 1.0 + points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2]
    mesh.cell_data["mat"] = [np.array([7], dtype=np.int64)]
    return mesh


def test_the_geometry_keys_match_the_upstream_pins(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube())
    with np.load(path) as data:
        # 6 quads -> 12 triangles -> 36 flat indices, int32, as DoMINO feeds
        # straight into signed_distance_field.
        assert data["stl_faces"].shape == (36,)
        assert data["stl_faces"].dtype == np.int32
        assert data["stl_centers"].shape == (12, 3)
        assert data["stl_areas"].sum() == pytest.approx(6.0, abs=1e-6)
        norms = np.linalg.norm(data["surface_normals"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        # The aliases are the same values under the names Transolver reads.
        assert np.array_equal(data["surface_mesh_centers"], data["stl_centers"])
        assert np.array_equal(data["surface_areas"], data["stl_areas"])
        # Centres are the vertex mean, not the area centroid.
        corners = data["stl_coordinates"][data["stl_faces"].reshape(-1, 3)]
        assert np.allclose(data["stl_centers"], corners.mean(axis=1), atol=1e-6)


def test_a_linear_nodal_field_averages_to_its_centroid_value(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube())
    with np.load(path) as data:
        names = [str(n) for n in data["meshioplusplus:surface_field_names"]]
        column = names.index(FIELD)
        centres = data["stl_centers"]
        expected = 1.0 + centres[:, 0] + 2 * centres[:, 1] + 3 * centres[:, 2]
        assert np.allclose(data["surface_fields"][:, column], expected, atol=1e-5)


def test_cell_data_is_gathered_from_the_parent_cell(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube())
    with np.load(path) as data:
        names = [str(n) for n in data["meshioplusplus:surface_field_names"]]
        column = names.index("mat")
        # The single hexahedron's tag reaches all twelve of its skin triangles.
        assert np.allclose(data["surface_fields"][:, column], 7.0)


def test_the_volume_half_is_the_nodes(tmp_path):
    path = tmp_path / "case_0.npz"
    mesh = _cube()
    cae.write(path, mesh)
    with np.load(path) as data:
        assert data["volume_mesh_centers"].shape == (len(mesh.points), 3)
        assert data["volume_fields"].shape == (len(mesh.points), 1)
        assert np.allclose(
            data["volume_mesh_centers"], mesh.points.astype(np.float32), atol=1e-6
        )


def test_every_array_is_at_least_one_dimensional(tmp_path):
    # The reader does `in_data[key][:]`, which a 0-d array cannot serve.
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube(), global_params={"Re": 100.0}, time=0.5, step=3)
    with np.load(path) as data:
        for key in data.files:
            assert data[key].ndim >= 1, key
        assert data["TIME"].shape == (1,) and data["TIME"].dtype == np.float32
        assert data["STEP"].shape == (1,) and data["STEP"].dtype == np.int64


def test_no_stray_key_contains_volume(tmp_path):
    # The reader takes its volume row count from the FIRST key in file order
    # whose name contains "volume", over every key in the file. A sidecar
    # called `..._volume_field_names` would hand it the length of a names
    # array; ours are `node_field_*` for exactly that reason.
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube())
    with np.load(path) as data:
        offenders = [k for k in data.files if "volume" in k]
        assert offenders == list(cae._cae.VOLUME_KEYS)
        assert data["volume_mesh_centers"].shape[0] == data["volume_fields"].shape[0]


def test_a_global_named_volume_is_refused(tmp_path):
    with pytest.raises(meshioplusplus.WriteError, match="contains 'volume'"):
        cae.write(tmp_path / "c.npz", _cube(), global_params={"volume_flow": 1.0})


def test_the_file_loads_without_pickle(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube(), global_params={"Re": 100.0})
    with np.load(path, allow_pickle=False) as data:
        assert "stl_coordinates" in data.files


def test_provenance_is_the_first_member(tmp_path):
    from meshioplusplus._provenance import TAG, read_provenance_lines

    path = tmp_path / "case_0.npz"
    cae.write(path, _cube())
    with np.load(path) as data:
        assert data.files[0] == cae._cae.PROVENANCE_KEY
    # Found by the ordinary head scanner -- the bytes dtype pads every row
    # with a NUL, so the block does not run into the zip's next header.
    lines, recognised = read_provenance_lines(path)
    assert recognised and lines[0] == TAG


def test_global_parameters(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(
        path,
        _cube(),
        global_params={"stream_velocity": 30.0, "air_density": 1.226},
        global_params_reference={"stream_velocity": 25.0},
        global_params_order=["stream_velocity", "air_density"],
    )
    with np.load(path) as data:
        assert data["stream_velocity"].shape == (1,)
        assert data["global_params_values"].shape == (2, 1)
        assert np.allclose(data["global_params_values"].ravel(), [30.0, 1.226])
        # A reference defaults to the value itself.
        assert np.allclose(data["global_params_reference"].ravel(), [25.0, 1.226])


def test_global_parameter_order_defaults_to_alphabetical(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, _cube(), global_params={"b_second": 2.0, "a_first": 1.0})
    with np.load(path) as data:
        assert np.allclose(data["global_params_values"].ravel(), [1.0, 2.0])


def test_an_orphan_reference_is_refused(tmp_path):
    with pytest.raises(meshioplusplus.WriteError, match="matching global_params"):
        cae.write(
            tmp_path / "c.npz",
            _cube(),
            global_params={"Re": 1.0},
            global_params_reference={"Ma": 1.0},
        )


def test_a_partial_order_is_refused(tmp_path):
    with pytest.raises(meshioplusplus.WriteError, match="exactly the"):
        cae.write(
            tmp_path / "c.npz",
            _cube(),
            global_params={"a": 1.0, "b": 2.0},
            global_params_order=["a"],
        )


def test_a_surface_input_writes_no_volume_keys(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, helpers.tri_mesh)
    with np.load(path) as data:
        assert not any(k in data.files for k in cae._cae.VOLUME_KEYS)
        assert "stl_coordinates" in data.files


def test_quads_are_triangulated(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, helpers.quad_mesh)
    with np.load(path) as data:
        assert data["stl_faces"].size % 3 == 0


def test_a_line_mesh_is_refused_by_name(tmp_path):
    with pytest.raises(meshioplusplus.WriteError, match="surface or volume"):
        cae.write(tmp_path / "c.npz", helpers.line_mesh)


def test_field_selection_and_order(tmp_path):
    path = tmp_path / "case_0.npz"
    mesh = _cube()
    mesh.point_data["S"] = np.zeros(len(mesh.points))
    cae.write(path, mesh, surface_fields=["mat", FIELD], volume_fields=[FIELD])
    with np.load(path) as data:
        names = [str(n) for n in data["meshioplusplus:surface_field_names"]]
        assert names == ["mat", FIELD]
        assert data["volume_fields"].shape[1] == 1
        assert [str(n) for n in data["meshioplusplus:node_field_names"]] == [FIELD]


def test_an_unknown_field_is_refused_by_name(tmp_path):
    with pytest.raises(meshioplusplus.WriteError, match="no surface field"):
        cae.write(tmp_path / "c.npz", _cube(), surface_fields=["nope"])


def test_read_back_the_surface(tmp_path):
    path = tmp_path / "case_0.npz"
    mesh = _cube()
    cae.write(path, mesh, global_params={"Re": 100.0}, time=0.25, step=2)
    back = meshioplusplus.read(path)
    assert [block.type for block in back.cells] == ["triangle"]
    assert len(back.cells[0].data) == 12
    assert set(back.cell_data) >= {FIELD, "mat", "surface_normals", "surface_areas"}
    assert np.allclose(back.cell_data["mat"][0], 7.0)
    assert np.allclose(back.field_data["Re"], [100.0])
    assert np.allclose(back.field_data["TIME"], [0.25])


def test_read_back_the_volume(tmp_path):
    path = tmp_path / "case_0.npz"
    mesh = _cube()
    cae.write(path, mesh)
    back = cae.read(path, part="volume")
    assert [block.type for block in back.cells] == ["vertex"]
    assert back.points.shape == (len(mesh.points), 3)
    assert np.allclose(back.point_data[FIELD], mesh.point_data[FIELD], atol=1e-6)


def test_a_missing_part_is_refused_by_name(tmp_path):
    path = tmp_path / "case_0.npz"
    cae.write(path, helpers.tri_mesh)
    with pytest.raises(meshioplusplus.ReadError, match="no volume half"):
        cae.read(path, part="volume")
    with pytest.raises(meshioplusplus.ReadError, match="'surface' or 'volume'"):
        cae.read(path, part="nope")


def test_a_foreign_file_keeps_its_field_block_whole(tmp_path):
    # No sidecar: the block cannot be split, so it comes back under its own
    # name rather than being guessed at.
    path = tmp_path / "foreign.npz"
    np.savez(
        path,
        stl_coordinates=np.zeros((3, 3), dtype=np.float32),
        stl_faces=np.array([0, 1, 2], dtype=np.int32),
        surface_fields=np.ones((1, 4), dtype=np.float32),
    )
    back = cae.read(path)
    assert "surface_fields" in back.cell_data
    assert back.cell_data["surface_fields"][0].shape == (1, 4)


def test_export_cases_over_a_glob(tmp_path):
    for index in range(3):
        mesh = _cube()
        mesh.point_data[FIELD] = np.full(len(mesh.points), float(index))
        meshioplusplus.write(tmp_path / f"run_{index}.vtu", mesh)
    out = tmp_path / "cases"
    written = cae.export_cases(str(tmp_path / "run_*.vtu"), out)
    assert [p.split("/")[-1] for p in written] == [
        "case_0.npz",
        "case_1.npz",
        "case_2.npz",
    ]
    with np.load(written[1]) as data:
        assert int(data["STEP"][0]) == 1
        assert float(data["TIME"][0]) == pytest.approx(1.0)


def test_the_cli_verb(tmp_path, capsys):
    from meshioplusplus._cli import main

    for index in range(2):
        meshioplusplus.write(tmp_path / f"run_{index}.vtu", _cube())
    out = tmp_path / "cases"
    code = main(
        [
            "data",
            "export-cae",
            str(tmp_path / "run_*.vtu"),
            str(out),
            "--surface-fields",
            FIELD,
            "--global",
            "Re=100",
            "--global-reference",
            "Re=80",
            "--global-order",
            "Re",
        ]
    )
    assert code == 0
    assert "wrote 2 case(s)" in capsys.readouterr().out
    with np.load(out / "case_0.npz") as data:
        assert [str(n) for n in data["meshioplusplus:surface_field_names"]] == [FIELD]
        assert np.allclose(data["global_params_reference"].ravel(), [80.0])
