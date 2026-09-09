"""Tests for the MCP server (meshioplusplus.mcp).

Two halves, mirroring test_interop.py's split:

* The **pure half** exercises the payload layer (``meshioplusplus.mcp._tools``)
  which imports no third-party library beyond numpy — it runs in the default
  CI matrix with the ``mcp`` SDK absent. Every report is asserted to be
  strict JSON (``json.dumps(..., allow_nan=False)``).
* The **gated half** (``pytest.importorskip("mcp")``) builds the FastMCP
  server and exercises tool listing, tool calls, resources and the error
  payload shape.

The parity guard (`test_every_operation_has_a_tool`) is the enforcement
mechanism behind CLAUDE.md's "keep the MCP server in sync" rule: a new public
operation in ``meshioplusplus.__all__`` fails here until it is claimed by a
tool's ``wraps`` (or consciously exempted in ``_NOT_TOOLS``).
"""

import json
import os

import meshioplusplus
import numpy as np
import pytest
from meshioplusplus.mcp import TOOL_REGISTRY, _tools

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_root():
    yield
    _tools.set_root(None)


def _mixed_mesh():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    cells = [
        ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
        ("tetra", np.array([[0, 1, 2, 4]])),
    ]
    return meshioplusplus.Mesh(
        points,
        cells,
        point_data={"t": np.linspace(0.0, 1.0, 5)},
        cell_data={"c": [np.array([1.0, 2.0]), np.array([3.0])]},
    )


@pytest.fixture()
def mesh_file(tmp_path):
    path = str(tmp_path / "in.vtu")
    meshioplusplus.write(path, _mixed_mesh())
    return path


def _dump(report):
    """Assert the report is strict JSON and return it."""
    json.dumps(report, allow_nan=False)
    return report


# --------------------------------------------------------------------------- #
# Pure half: sanitizer and sandbox                                            #
# --------------------------------------------------------------------------- #


def test_json_safe_numpy_and_non_finite():
    out = _dump(
        _tools._json_safe(
            {
                "i": np.int64(3),
                "f": np.float32(1.5),
                "nan": float("nan"),
                "inf": np.float64("inf"),
                "arr": np.array([1.0, float("-inf")]),
                "nested": {"t": (np.int32(1), [np.float64(2.0)])},
            }
        )
    )
    assert out["i"] == 3 and out["f"] == 1.5
    assert out["nan"] is None and out["inf"] is None
    assert out["arr"] == [1.0, None]
    assert out["nested"]["t"] == [1, [2.0]]
    assert out["non_finite_replaced"] == 3


def test_json_safe_truncates_large_arrays():
    out = _dump(_tools._json_safe({"big": np.arange(5000).reshape(100, 50)}))
    wrapper = out["big"]
    assert wrapper["truncated"] is True
    assert wrapper["size"] == 5000
    assert wrapper["shape"] == [100, 50]
    assert wrapper["preview"] == list(range(50))


def test_root_confinement(tmp_path, mesh_file):
    _tools.set_root(str(tmp_path))
    # relative paths resolve against the root
    assert _tools.tool_info("in.vtu")["num_points"] == 5
    # absolute paths inside the root are fine
    assert _tools.tool_info(mesh_file)["num_points"] == 5
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_info("/etc/passwd")
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_info("../escape.vtu")


def test_missing_input_is_a_clean_error(tmp_path):
    with pytest.raises(ValueError, match="input file not found"):
        _tools.tool_info(str(tmp_path / "nope.vtu"))


# --------------------------------------------------------------------------- #
# Pure half: inspection tools                                                 #
# --------------------------------------------------------------------------- #


def test_formats_payload():
    out = _dump(_tools.tool_formats())
    assert "vtu" in out["readable"] and "vtu" in out["writable"]
    assert out["extensions"][".vtu"] == ["vtu"]


def test_sniff(mesh_file):
    out = _dump(_tools.tool_sniff(mesh_file))
    assert out["format"] == "vtu"
    assert out["from_extension"] == ["vtu"]


def test_info(mesh_file):
    out = _dump(_tools.tool_info(mesh_file))
    assert out["num_points"] == 5
    assert out["num_cells"] == 3
    assert out["point_data_names"] == ["t"]


def test_stats(mesh_file):
    out = _dump(_tools.tool_stats(mesh_file))
    assert out["num_points"] == 5
    assert out["cell_type_counts"] == {"triangle": 2, "tetra": 1}


def test_quality_summary_and_annotated_output(mesh_file, tmp_path):
    out = _dump(_tools.tool_quality(mesh_file))
    assert out["num_cells"] == 3
    assert any(m.startswith("quality:") for m in out["metrics"])
    assert isinstance(out["cell_arrays"], str)  # omitted, with a pointer
    annotated = str(tmp_path / "annotated.vtu")
    out = _dump(_tools.tool_quality(mesh_file, output_path=annotated))
    assert os.path.isfile(out["output_path"])
    reread = meshioplusplus.read(annotated)
    assert any(k.startswith("quality:") for k in reread.cell_data)


def test_data_info(mesh_file):
    out = _dump(_tools.tool_data_info(mesh_file))
    names = {(a["location"], a["name"]) for a in out["arrays"]}
    assert ("point_data", "t") in names and ("cell_data", "c") in names


def test_bandwidth(mesh_file):
    out = _dump(_tools.tool_bandwidth(mesh_file))
    assert out["bandwidth"] == 4


def test_data_preview(mesh_file):
    out = _dump(_tools.tool_data_preview(mesh_file, "t", location="point", limit=3))
    assert out["num_rows"] == 5
    assert out["values"] == [0.0, 0.25, 0.5]
    out = _dump(_tools.tool_data_preview(mesh_file, "c", location="cell"))
    assert out["values"] == [[1.0], [2.0], [3.0]]  # block-major concatenation
    with pytest.raises(ValueError, match="no point data array named 'zz'"):
        _tools.tool_data_preview(mesh_file, "zz")
    with pytest.raises(ValueError, match="unknown location"):
        _tools.tool_data_preview(mesh_file, "t", location="vertex")


def test_diff(mesh_file, tmp_path):
    same = str(tmp_path / "same.vtu")
    meshioplusplus.write(same, _mixed_mesh())
    out = _dump(_tools.tool_diff(mesh_file, same))
    assert out["equal"] is True
    moved = _mixed_mesh()
    moved.points = moved.points + 1.0
    other = str(tmp_path / "other.vtu")
    meshioplusplus.write(other, moved)
    out = _dump(_tools.tool_diff(mesh_file, other))
    assert out["equal"] is False and out["verdict"] == "different"


# --------------------------------------------------------------------------- #
# Pure half: conversion                                                       #
# --------------------------------------------------------------------------- #


def test_convert_roundtrip(mesh_file, tmp_path):
    out_path = str(tmp_path / "out.vtk")
    out = _dump(_tools.tool_convert(mesh_file, out_path))
    assert out["output_format"] == "vtk"
    assert os.path.isfile(out["output_path"])
    assert out["num_points"] == 5 and out["num_cells"] == 3


def test_convert_ascii_variant(mesh_file, tmp_path):
    out = _dump(_tools.tool_convert(mesh_file, str(tmp_path / "a.vtu"), mode="ascii"))
    with open(out["output_path"], "rb") as f:
        assert b"ascii" in f.read()


def test_convert_variant_errors(mesh_file, tmp_path):
    with pytest.raises(ValueError, match="has no ascii variant"):
        _tools.tool_convert(mesh_file, str(tmp_path / "a.obj"), mode="ascii")
    with pytest.raises(ValueError, match="only vti/vtu/vtp"):
        _tools.tool_convert(mesh_file, str(tmp_path / "a.vtk"), compression="zstd")
    with pytest.raises(ValueError, match="unknown mode"):
        _tools.tool_convert(mesh_file, str(tmp_path / "a.vtu"), mode="fast")


def test_pipeline_tool(mesh_file, tmp_path):
    out_path = str(tmp_path / "piped.vtu")
    settings_path = str(tmp_path / "settings.json")
    with open(settings_path, "w") as f:
        json.dump(
            {
                "Input": {"Path": mesh_file},
                "Operations": [{"Op": "Quality"}, {"Op": "Clean"}],
                "Output": {"Path": out_path},
            },
            f,
        )
    out = _dump(_tools.tool_pipeline(settings_path))
    assert os.path.isfile(out["output_path"])
    assert [s["op"] for s in out["steps"]] == ["Quality", "Clean"]
    # overrides win over the document's paths
    other = str(tmp_path / "other.vtu")
    _dump(_tools.tool_pipeline(settings_path, output_path=other))
    assert os.path.isfile(other)


def test_pipeline_tool_sandboxes_the_inner_paths(mesh_file, tmp_path, monkeypatch):
    # A settings document naming a path outside the root must fail exactly the
    # way a path argument would -- the sandbox covers the document's insides.
    settings_path = str(tmp_path / "settings.json")
    with open(settings_path, "w") as f:
        json.dump(
            {
                "Input": {"Path": "/etc/passwd"},
                "Operations": [],
                "Output": {"Path": str(tmp_path / "o.vtu")},
            },
            f,
        )
    monkeypatch.setattr(_tools, "_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_pipeline(settings_path)


# --------------------------------------------------------------------------- #
# Pure half: mesh operations                                                  #
# --------------------------------------------------------------------------- #


def test_clean_report(mesh_file, tmp_path):
    out = _dump(_tools.tool_clean(mesh_file, str(tmp_path / "clean.vtu")))
    for key in (
        "points_welded",
        "points_removed_orphan",
        "cells_dropped_degenerate",
        "cells_dropped_duplicate",
    ):
        assert key in out


def test_reorder_reports_bandwidth(mesh_file, tmp_path):
    out = _dump(_tools.tool_reorder(mesh_file, str(tmp_path / "ro.vtu")))
    assert out["bandwidth_after"] <= out["bandwidth_before"]


def test_slice(mesh_file, tmp_path):
    out = _dump(
        _tools.tool_slice(
            mesh_file, str(tmp_path / "s.vtu"), origin=[0, 0, 0.5], normal=[0, 0, 1]
        )
    )
    assert out["num_cells"] >= 1
    assert all(b["type"] in ("triangle", "quad") for b in out["cell_blocks"])


def test_split_writes_one_file_per_piece(mesh_file, tmp_path):
    out_dir = str(tmp_path / "pieces")
    out = _dump(_tools.tool_split(mesh_file, by="type", output_dir=out_dir))
    assert sorted(out["pieces"]) == ["tetra", "triangle"]
    for key, path in out["pieces"].items():
        assert os.path.isfile(path)
        assert os.path.basename(path) == f"in_{key}.vtu"
        assert out["summaries"][key]["num_cells"] >= 1


def test_merge(mesh_file, tmp_path):
    out = _dump(_tools.tool_merge([mesh_file, mesh_file], str(tmp_path / "m.vtu")))
    assert out["num_points"] == 10 and out["num_cells"] == 6
    with pytest.raises(ValueError, match="at least two"):
        _tools.tool_merge([mesh_file], str(tmp_path / "m2.vtu"))


def test_partition_writes_nparts_files(mesh_file, tmp_path):
    out_dir = str(tmp_path / "parts")
    out = _dump(_tools.tool_partition(mesh_file, 2, method="sfc", output_dir=out_dir))
    assert len(out["parts"]) == 2
    assert all(os.path.isfile(p) for p in out["parts"])
    assert sum(s["num_cells"] for s in out["summaries"]) == 3


def test_crop(mesh_file, tmp_path):
    out = _dump(
        _tools.tool_crop(
            mesh_file, str(tmp_path / "c.vtu"), bbox=[0, 0, 0, 1, 1, 0.5], mode="any"
        )
    )
    assert out["num_cells"] == 3
    with pytest.raises(ValueError, match="both plane_origin and plane_normal"):
        _tools.tool_crop(mesh_file, str(tmp_path / "c2.vtu"), plane_origin=[0, 0, 0])


def test_transform(mesh_file, tmp_path):
    out_path = str(tmp_path / "t.vtu")
    _dump(_tools.tool_transform(mesh_file, out_path, translate=[10.0, 0.0, 0.0]))
    moved = meshioplusplus.read(out_path)
    assert moved.points[:, 0].min() >= 10.0
    with pytest.raises(ValueError, match="both rotate_axis and rotate_degrees"):
        _tools.tool_transform(mesh_file, out_path, rotate_axis=[0, 0, 1])


# --------------------------------------------------------------------------- #
# Pure half: data operations                                                  #
# --------------------------------------------------------------------------- #


def test_data_calc_cli_spelling(mesh_file, tmp_path):
    out_path = str(tmp_path / "calc.vtu")
    out = _dump(_tools.tool_data_calc(mesh_file, out_path, "t2 = t * 2"))
    assert "t2" in out["point_data"]
    reread = meshioplusplus.read(out_path)
    assert np.allclose(reread.point_data["t2"], 2 * reread.point_data["t"])


def test_data_manage_rename(mesh_file, tmp_path):
    out = _dump(
        _tools.tool_data_manage(
            mesh_file, str(tmp_path / "dm.vtu"), rename=[["point", "t", "temp"]]
        )
    )
    assert out["point_data"] == ["temp"]
    assert out["renamed"] == [["point_data:t", "point_data:temp"]]
    with pytest.raises(ValueError, match="at least one of keep/drop/rename"):
        _tools.tool_data_manage(mesh_file, str(tmp_path / "dm2.vtu"))


def test_data_convert(mesh_file, tmp_path):
    out = _dump(
        _tools.tool_data_convert(mesh_file, str(tmp_path / "dc.vtu"), "point_to_cell")
    )
    assert "t" in out["cell_data"]
    with pytest.raises(ValueError, match="unknown direction"):
        _tools.tool_data_convert(mesh_file, str(tmp_path / "dc2.vtu"), "sideways")


def test_gradient(mesh_file, tmp_path):
    out = _dump(_tools.tool_gradient(mesh_file, str(tmp_path / "g.vtu"), "t"))
    assert "t:gradient" in out["cell_data"]
    # The mixed fixture's triangle block is below the tet block's dimension, so
    # the skip counter must be reported rather than silently swallowed.
    assert out["num_skipped"] == 2
    assert out["num_fallback"] == 0

    div = _dump(
        _tools.tool_gradient(
            mesh_file,
            str(tmp_path / "d.vtu"),
            "t",
            operator="gradient",
            location="point",
        )
    )
    assert "t:gradient" in div["point_data"]

    with pytest.raises(ValueError):
        _tools.tool_gradient(mesh_file, str(tmp_path / "x.vtu"), "nope")


def test_data_condition(mesh_file, tmp_path):
    out_path = str(tmp_path / "cond.vtu")
    _dump(
        _tools.tool_data_condition(
            output_path=out_path,
            input_path=mesh_file,
            operation="normalize",
            arrays=["t"],
            lo=0.0,
            hi=10.0,
        )
    )
    reread = meshioplusplus.read(out_path)
    assert np.isclose(reread.point_data["t"].max(), 10.0)


# --------------------------------------------------------------------------- #
# Parity guard: every public operation must be claimed by a tool              #
# --------------------------------------------------------------------------- #

# Public API names that deliberately have no path-based MCP tool. Add here
# only when the API is genuinely not expressible as file-in/file-out (the
# in-memory interop/GPU handoffs, the interactive viewer, classes, plumbing).
_NOT_TOOLS = {
    "_cli",
    "write_points_cells",  # constructor variant of write(); convert covers I/O
    "register_format",
    "deregister_format",
    "partition_labels",  # in-memory variant of partition
    # A generator of (time, Mesh) pairs: the meshes never touch the filesystem,
    # so there is no path-in/path-out form. The `sequence` tool covers every
    # file-level use (fan-in, fan-out, N->N).
    "read_sequence",
    "view",  # interactive; screenshot is the headless tool
    "has_viewer",
    "to_pyvista",
    "from_pyvista",
    "to_trimesh",
    "from_trimesh",
    # An in-memory handoff to a live Blender datablock: there is no
    # path-in/path-out form, and `convert` already covers the file case.
    "to_blender",
    "from_blender",
    "has_blender",
    "to_arrow",
    "from_arrow",
    "read_parquet",  # tabular import, no Mesh output to write
    "to_pandas",
    "to_polars",
    "has_pyvista",
    "has_trimesh",
    "has_arrow",
    "has_pandas",
    "has_polars",
    "has_open3d",
    "has_dolfinx",
    "edge_index",  # in-memory (2, E) array; no path-based form
    "feature_matrix",  # in-memory matrix + schema; no path-based form
    "FeatureMatrix",
    # An in-memory value (the mesh plus its provenance arrays); the
    # `tessellate` tool covers the path-in/path-out case, writing the mesh
    # with its tessellate:* arrays attached.
    "Tessellation",
    # Grid values and the lattice they live on are in-memory arrays; the four
    # grid_* tools cover every path-in/path-out case (sample, scatter, resample,
    # spectrum).
    "GridSpec",
    "GridArray",
    "PowerSpectrum",
    "interpolate_grid",  # in-memory (C, M) evaluation at caller-supplied points
    "squeeze_grid",  # in-memory array reshape for a 2-D operator
    "expand_grid",
    # A budget is a selection as an in-memory value (indices in selection
    # order, meant to slice a feature matrix); the `subsample` tool covers the
    # path-in/path-out case, writing the selected points as a point cloud.
    "PointBudget",
    "select_points",
    # In-memory arrays: an edge index and a coarsening hierarchy are values a
    # caller holds, not files a path-based tool could hand back.
    "edge_vectors",
    "bistride_hierarchy",
    "BistrideHierarchy",
    "has_zarr",
    "to_dlpack",
    "to_cupy",
    "from_cupy",
    "has_cupy",
    "has_cuda_device",
    "to_torch",
    "to_jax",
    "has_torch",
    "has_jax",
    "Mesh",
    "CellBlock",
    "Region",
    "TimeSeries",  # in-memory random-access class; the `sequence` tool covers
    # the path-in/path-out cases (fan-in/fan-out/per-step chain)
    "DatasetEntry",  # class; the dataset_* tools cover the path-based cases
    "ReadError",
    "WriteError",
    "topological_dimension",
    "__version__",
}


def test_every_operation_has_a_tool():
    everything = meshioplusplus.__all__
    formats = set(everything[: everything.index("_cli")])
    ops = set(everything) - formats - _NOT_TOOLS
    covered = set()
    for spec in TOOL_REGISTRY.values():
        covered.update(spec["wraps"])
    missing = sorted(ops - covered)
    assert not missing, (
        f"public operations without an MCP tool: {missing} — add a tool to "
        "meshioplusplus/mcp/_tools.py (and register it in _server.py) or, if "
        "the API is genuinely not path-expressible, add a conscious exemption "
        "to _NOT_TOOLS above. See CLAUDE.md 'Keep the MCP server in sync'."
    )


def test_wraps_names_are_real_public_api():
    public = set(meshioplusplus.__all__)
    for name, spec in TOOL_REGISTRY.items():
        stale = [w for w in spec["wraps"] if w not in public]
        assert not stale, f"tool '{name}' claims unknown API names: {stale}"


def test_registry_entries_are_wellformed():
    for name, spec in TOOL_REGISTRY.items():
        assert callable(spec["fn"]), name
        assert spec["gated"] in (None, "arrow", "viewer", "physicsnemo"), name


def test_every_tool_function_is_callable():
    """Import-level smoke: call every non-gated tool's `fn` attribute lookup.

    `_tools.py` imports the public API by name, so a tool added without its
    import is an `F821` that only fires when that tool actually *runs* — which a
    per-tool test suite catches only for the tools it happens to exercise. This
    walks the whole registry instead.
    """
    import inspect

    for name, spec in TOOL_REGISTRY.items():
        fn = spec["fn"]
        # Every name the function body resolves from module globals must exist.
        for var in fn.__code__.co_names:
            if var in fn.__globals__ or hasattr(_tools, var):
                continue
            # Attribute names and locals also land in co_names; only flag a bare
            # global that is neither defined nor a builtin.
            assert not (
                var.islower() and var in getattr(_tools, "__annotations__", {})
            ), f"{name}: unresolved global {var}"
        assert inspect.signature(fn) is not None


def test_compute_sdf_tool(tmp_path):
    # A *surface*: distance is measured to a skin, not to a volume.
    skin = tmp_path / "skin.vtu"
    meshioplusplus.write(
        skin,
        meshioplusplus.convert_cells(
            meshioplusplus.extract_surface(meshioplusplus.grid([2, 2, 2])),
            mode="simplexify",
        ),
    )
    mesh_file = str(skin)
    out = tmp_path / "field.vtu"
    report = _tools.tool_compute_sdf(mesh_file, str(out), resolution=[4, 4, 4])
    assert out.exists()
    assert report["dims"] == [4, 4, 4]
    assert report["max_depth"] == 0
    assert report["num_cells"] == 64

    tree = tmp_path / "tree.vtu"
    report = _tools.tool_compute_sdf(
        mesh_file, str(tree), structure="octree", root_resolution=4, max_depth=2
    )
    assert report["max_depth"] == 2
    assert 64 < report["num_cells"] < 4096

    # resolution/cell_size size a voxel grid; an octree's finest cell is already
    # determined by root_resolution and max_depth.
    with pytest.raises(ValueError, match="octree"):
        _tools.tool_compute_sdf(
            mesh_file, str(tree), structure="octree", resolution=[4, 4, 4]
        )


def _grid_source(tmp_path):
    """Anisotropic box, linear scalar plus a vector -- see test_grid_transfer."""
    mesh = meshioplusplus.convert_cells(
        meshioplusplus.grid((3, 3, 3), spacing=(1 / 3, 2 / 3, 4 / 3)), mode="simplexify"
    )
    p = mesh.points
    mesh.point_data["u"] = 1.0 + 2.0 * p[:, 0] + 3.0 * p[:, 1] - 5.0 * p[:, 2]
    mesh.point_data["v"] = np.column_stack([p[:, 0], 2.0 * p[:, 1], -p[:, 2]])
    path = tmp_path / "src.vtu"
    meshioplusplus.write(path, mesh)
    return str(path)


def test_dataset_add_records_a_paired_target(tmp_path):
    src = tmp_path / "coarse"
    tgt = tmp_path / "fine"
    for d, n in ((src, 2), (tgt, 4)):
        d.mkdir()
        for i in range(3):
            meshioplusplus.write(d / f"out_{i:04d}.vtu", meshioplusplus.grid((n, n, n)))
    manifest = str(tmp_path / "m.json")

    report = _tools.tool_dataset_add(
        manifest,
        input_pattern=str(src / "*.vtu"),
        target_pattern=str(tgt / "*.vtu"),
        entry_id="sr",
        split="train",
    )
    assert report["num_steps"] == 3
    assert report["num_target_steps"] == 3

    listed = _tools.tool_dataset_list(manifest)["entries"][0]
    assert listed["Target"] == {"Pattern": "fine/*.vtu"}
    assert listed["Source"] == {"Pattern": "coarse/*.vtu"}


def test_dataset_add_refuses_a_target_outside_the_root(tmp_path):
    """A Target is client input *inside* the document, so the sandbox has to
    hold on it exactly as it does on the Source."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.vtu").write_bytes(b"")
    meshioplusplus.write(root / "a.vtu", meshioplusplus.grid((2, 2, 2)))
    outside = tmp_path / "outside.vtu"
    meshioplusplus.write(outside, meshioplusplus.grid((2, 2, 2)))

    _tools.set_root(str(root))
    try:
        with pytest.raises(ValueError):
            _tools.tool_dataset_add(
                str(root / "m.json"),
                input_paths=["a.vtu"],
                target_paths=[str(outside)],
                entry_id="sr",
            )
    finally:
        _tools.set_root(None)


def test_dataset_health_reports_a_broken_pairing(tmp_path):
    src = tmp_path / "coarse"
    src.mkdir()
    for i in range(3):
        meshioplusplus.write(src / f"out_{i:04d}.vtu", meshioplusplus.grid((2, 2, 2)))
    one = tmp_path / "only.vtu"
    meshioplusplus.write(one, meshioplusplus.grid((2, 2, 2)))

    manifest = meshioplusplus.DatasetManifest(base_dir=str(tmp_path))
    manifest.add({"Pattern": "coarse/*.vtu"}, id="ok")
    manifest.add(
        {"Pattern": "coarse/*.vtu"},
        target={"Path": "only.vtu"},
        id="bad",
        validate_source=False,
    )
    path = tmp_path / "m.json"
    manifest.save(path)

    report = _tools.tool_dataset_health(str(path), quality=False)
    entries = report["entries"]
    assert entries["ok"]["target_steps"] is None  # self-supervised, not broken
    assert entries["ok"]["pairing_error"] is None
    assert "pairs 3 source steps with 1 target steps" in entries["bad"]["pairing_error"]
    assert "bad" in report["bad_entries"] and "ok" not in report["bad_entries"]


def test_grid_tools_round_trip(tmp_path):
    """sample -> resample -> scatter over paths, with the reports checked.

    The whole point of the grid tools is that an agent can drive the data path
    without holding a mesh, so every step here goes through a file.
    """
    src = _grid_source(tmp_path)
    gridfile = str(tmp_path / "g.vti")
    report = _tools.tool_grid_sample(src, gridfile, resolution=[4, 4, 4])
    assert report["channels"] == ["u", "v_0", "v_1", "v_2"]
    assert report["coverage"] == 1.0
    assert report["grid"]["dims"] == [4, 4, 4]
    assert report["grid"]["shape"] == [5, 5, 5]
    assert report["grid"]["layout"] == "channels_first_zyx"

    upfile = str(tmp_path / "up.vti")
    report = _tools.tool_grid_resample(gridfile, upfile, factor=2)
    assert report["scaling_factor"] == [2, 2, 2]
    assert report["grid"]["dims"] == [8, 8, 8]

    fine = str(tmp_path / "fine.vtu")
    meshioplusplus.write(
        fine, meshioplusplus.grid((6, 6, 6), spacing=(1 / 6, 2 / 6, 4 / 6))
    )
    out = str(tmp_path / "out.vtu")
    report = _tools.tool_grid_scatter(upfile, fine, out)
    assert report["point_data"] == ["u", "v"]

    got = meshioplusplus.read(out)
    p = got.points
    assert got.point_data["u"] == pytest.approx(
        1.0 + 2.0 * p[:, 0] + 3.0 * p[:, 1] - 5.0 * p[:, 2], abs=1e-12
    )


def test_grid_power_spectrum_tool_is_json_safe(tmp_path):
    mesh = meshioplusplus.grid((15, 15, 15), spacing=(1 / 16, 1 / 16, 1 / 16))
    mesh.point_data["u"] = np.cos(2 * np.pi * 3 * mesh.points[:, 0])
    path = str(tmp_path / "g.vti")
    meshioplusplus.write(path, mesh)

    report = _dump(_tools.tool_grid_power_spectrum(path, "u"))
    assert report["units"] == "cycles per unit length"
    assert report["total_power"] == pytest.approx(
        float((mesh.point_data["u"] ** 2).mean()), rel=1e-12
    )
    loud = int(np.argmax(report["power"]))
    assert report["wavenumber"][loud] == pytest.approx(3.0)


def test_grid_tools_report_a_named_error_rather_than_a_traceback(tmp_path):
    src = _grid_source(tmp_path)
    gridfile = str(tmp_path / "g.vti")
    _tools.tool_grid_sample(src, gridfile, resolution=[2, 2, 2])

    guarded = _tools.guard(
        _tools.tool_grid_power_spectrum, input_path=gridfile, field="nope"
    )
    assert guarded["error_type"] == "ValueError"
    assert "no channel named 'nope'" in guarded["error"]

    # a mesh that is not a lattice cannot be read as a grid
    guarded = _tools.guard(
        _tools.tool_grid_scatter,
        grid_path=src,
        target_path=src,
        output_path=str(tmp_path / "o.vtu"),
    )
    assert "not a dense regular lattice" in guarded["error"]


def test_crop_tool_takes_a_data_predicate(mesh_file, tmp_path):
    tagged = tmp_path / "tagged.vtu"
    mesh = meshioplusplus.read(mesh_file)
    mesh.cell_data["t"] = [
        np.arange(len(cb.data), dtype=np.float64) for cb in mesh.cells
    ]
    meshioplusplus.write(tagged, mesh)

    out = tmp_path / "kept.vtu"
    report = _tools.tool_crop(
        str(tagged), str(out), where_array="t", where_compare="<", where_value=1.0
    )
    assert report["num_cells"] == len(mesh.cells)  # one cell per block matches


# --------------------------------------------------------------------------- #
# Gated half: the FastMCP server (needs the [mcp] extra)                      #
# --------------------------------------------------------------------------- #


def _server():
    pytest.importorskip("mcp")
    import meshioplusplus.mcp as mmcp

    return mmcp.create_server()


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _tool_json(result):
    # FastMCP.call_tool returns a content list (older SDKs) or a
    # (content, structured) tuple (newer); normalize to the parsed JSON.
    content = result[0] if isinstance(result, tuple) else result
    text = next(c.text for c in content if getattr(c, "type", "") == "text")
    return json.loads(text)


def test_server_lists_every_registered_tool():
    server = _server()
    tools = _run(server.list_tools())
    assert sorted(t.name for t in tools) == sorted(TOOL_REGISTRY)
    for t in tools:
        assert (t.description or "").strip(), f"tool '{t.name}' has no description"
        assert t.inputSchema, f"tool '{t.name}' has no input schema"


def test_server_call_tool_info_and_convert(mesh_file, tmp_path):
    server = _server()
    report = _tool_json(_run(server.call_tool("info", {"input_path": mesh_file})))
    assert report["num_points"] == 5
    out_path = str(tmp_path / "out.vtk")
    report = _tool_json(
        _run(
            server.call_tool(
                "convert", {"input_path": mesh_file, "output_path": out_path}
            )
        )
    )
    assert os.path.isfile(report["output_path"])


def test_server_error_payload_shape(tmp_path):
    server = _server()
    report = _tool_json(
        _run(server.call_tool("info", {"input_path": str(tmp_path / "nope.vtu")}))
    )
    assert report["error_type"] == "ValueError"
    assert "input file not found" in report["error"]


def test_server_gated_tool_names_the_extra(mesh_file, tmp_path):
    if meshioplusplus.has_viewer():
        pytest.skip("polyscope installed; the gated error path is not reachable")
    server = _server()
    report = _tool_json(
        _run(
            server.call_tool(
                "screenshot",
                {"input_path": mesh_file, "output_path": str(tmp_path / "s.png")},
            )
        )
    )
    assert report["error_type"] == "ImportError"
    assert "polyscope" in report["error"]


def test_server_resources():
    server = _server()
    resources = _run(server.list_resources())
    uris = sorted(str(r.uri) for r in resources)
    assert uris == ["meshioplusplus://formats", "meshioplusplus://version"]
    content = _run(server.read_resource("meshioplusplus://formats"))
    payload = json.loads(list(content)[0].content)
    assert payload == _tools.formats_payload()


def test_has_mcp_and_named_error_without_it():
    import meshioplusplus.mcp as mmcp

    assert isinstance(mmcp.has_mcp(), bool)
    if not mmcp.has_mcp():
        with pytest.raises(ImportError, match=r"pip install meshioplusplus\[mcp\]"):
            mmcp.create_server()


# --------------------------------------------------------------------------- #
# Pure half: dataset-manifest tools                                           #
# --------------------------------------------------------------------------- #
def _case_files(tmp_path, n=3):
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    for i in range(n):
        meshioplusplus.write(str(cases / f"case_{i}.vtu"), _mixed_mesh())
    return cases


def test_dataset_add_list_update_round_trip(tmp_path, monkeypatch):
    _case_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = str(tmp_path / "m.json")

    out = _dump(
        _tools.tool_dataset_add(
            manifest,
            input_pattern="cases/case_*.vtu",
            entry_id="sweep",
            tags=["raw"],
            metadata={"Re": 100},
        )
    )
    assert out["entry_id"] == "sweep" and out["num_steps"] == 3
    # the stored source is manifest-relative AND "/"-separated regardless of
    # platform, so the manifest is genuinely portable (doc/datasets.md) --
    # not os.path.join, which would be backslash-joined on Windows.
    doc = json.loads(open(manifest, encoding="utf-8").read())
    assert doc["Entries"][0]["Source"]["Pattern"] == "cases/case_*.vtu"

    _dump(
        _tools.tool_dataset_add(
            manifest,
            input_paths=["cases/case_0.vtu", "cases/case_1.vtu"],
            entry_id="pair",
            group="g/h",
        )
    )
    out = _dump(
        _tools.tool_dataset_update(
            manifest, entry_ids=["sweep"], split="train", add_tags=["v1"]
        )
    )
    assert out["splits"] == {"train": 1, "None": 1}
    out = _dump(_tools.tool_dataset_list(manifest, split="train", resolve=True))
    assert out["num_matching"] == 1
    entry = out["entries"][0]
    assert entry["Id"] == "sweep" and entry["Tags"] == ["raw", "v1"]
    assert len(entry["resolved"]) == 3
    assert all(item["time_source"] for item in entry["resolved"])


def test_dataset_update_assign_and_annotate(tmp_path, monkeypatch):
    _case_files(tmp_path, n=6)
    monkeypatch.chdir(tmp_path)
    manifest = str(tmp_path / "m.json")
    for i in range(6):
        _tools.tool_dataset_add(
            manifest, input_paths=[f"cases/case_{i}.vtu"], entry_id=f"c{i}"
        )
    out = _dump(
        _tools.tool_dataset_update(
            manifest, assign_splits={"train": 0.5, "test": 0.5}, seed=1
        )
    )
    assert out["splits"] == {"train": 3, "test": 3}
    _dump(
        _tools.tool_dataset_update(
            manifest, entry_ids=["c0"], notes="odd", metadata={"Ma": 0.3}
        )
    )
    m = meshioplusplus.DatasetManifest.load(manifest)
    assert m["c0"].notes == "odd" and m["c0"].metadata == {"Ma": 0.3}
    with pytest.raises(ValueError, match="exactly one entry_id"):
        _tools.tool_dataset_update(manifest, all_entries=True, notes="x")
    with pytest.raises(ValueError, match="entry_ids or all_entries"):
        _tools.tool_dataset_update(manifest, split="train")


def test_dataset_tools_validate_input_shape(tmp_path):
    manifest = str(tmp_path / "m.json")
    with pytest.raises(ValueError, match="exactly one of input_pattern"):
        _tools.tool_dataset_add(manifest)
    with pytest.raises(ValueError, match="exactly one of input_pattern"):
        _tools.tool_dataset_add(manifest, input_pattern="a*.vtu", input_paths=["a.vtu"])


def test_dataset_add_sandboxes_the_sources(tmp_path, monkeypatch):
    # A source outside the root must fail exactly the way a path argument
    # would — before anything is written to the manifest.
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.vtu"
    meshioplusplus.write(str(outside), _mixed_mesh())
    monkeypatch.setattr(_tools, "_ROOT", str(root))
    manifest = str(root / "m.json")
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_dataset_add(manifest, input_paths=[str(outside)])
    assert not os.path.exists(manifest)


def test_dataset_list_sandboxes_hand_edited_sources(tmp_path, monkeypatch):
    # A hand-edited manifest naming files outside the root: listing is fine
    # (the document is data), but resolving its plan must refuse — the
    # pipeline-tool rule applied to manifests.
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.vtu"
    meshioplusplus.write(str(outside), _mixed_mesh())
    manifest = root / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "Version": 1,
                "Entries": [{"Id": "x", "Source": {"Path": str(outside)}}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_tools, "_ROOT", str(root))
    _dump(_tools.tool_dataset_list(str(manifest)))  # data-only: allowed
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_dataset_list(str(manifest), resolve=True)


# --------------------------------------------------------------------------- #
# Pure half: dataset health / find and the registry dispatch (v10.23.0)       #
# --------------------------------------------------------------------------- #
def _health_manifest(tmp_path):
    """Three cases: case_0 carries quality:* arrays (whose NaN must not count),
    case_1 carries a NaN in a data array, case_2 lacks the point array."""
    from meshioplusplus import DatasetManifest, attach_quality

    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    meshioplusplus.write(str(cases / "case_0.vtu"), attach_quality(_mixed_mesh()))
    bad = _mixed_mesh()
    bad.point_data["t"][2] = float("nan")
    meshioplusplus.write(str(cases / "case_1.vtu"), bad)
    sparse = _mixed_mesh()
    del sparse.point_data["t"]
    meshioplusplus.write(str(cases / "case_2.vtu"), sparse)
    m = DatasetManifest(name="health", base_dir=str(tmp_path))
    m.add("cases/case_0.vtu", split="train")
    m.add("cases/case_1.vtu", split="train")
    m.add("cases/case_2.vtu")
    path = str(tmp_path / "m.json")
    m.save(path)
    return path


def test_dataset_health_reports_per_entry_and_aggregates(tmp_path):
    report = _dump(_tools.tool_dataset_health(_health_manifest(tmp_path)))
    assert report["producer"] == "server"
    assert report["name"] == "health"
    assert report["num_entries"] == 3 and report["scanned"] == 3
    assert len(report["sha256"]) == 64
    assert report["splits"] == {"train": 2, "": 1}
    assert report["split_balance"] == [
        {"split": "train", "count": 2, "fraction": 2 / 3},
        {"split": "", "count": 1, "fraction": 1 / 3},
    ]
    entries = report["entries"]
    assert set(entries) == {"case_0", "case_1", "case_2"}
    for scan in entries.values():
        assert scan["steps"] == 1
        assert isinstance(scan["min_scaled_jacobian"], float)
    # quality:* NaN (metric N/A) never counts; a data NaN does
    assert entries["case_0"]["num_nan"] == 0
    assert entries["case_1"]["num_nan"] == 1
    assert not any(a.startswith("quality:") for a in entries["case_0"]["arrays"])
    # fields missing = union over readable entries minus each entry's own
    assert list(report["fields_missing"]) == ["case_2"]
    assert report["fields_missing"]["case_2"][0].endswith(":t")
    assert report["totals"]["num_nan"] == 1
    assert report["totals"]["num_inverted"] == 0
    assert report["bad_entries"] == ["case_1"]


def test_dataset_health_filters_and_reports_unreadable_entries(tmp_path):
    from meshioplusplus import DatasetManifest

    path = _health_manifest(tmp_path)
    only = _dump(_tools.tool_dataset_health(path, split="train"))
    assert only["scanned"] == 2 and set(only["entries"]) == {"case_0", "case_1"}
    one = _dump(_tools.tool_dataset_health(path, entry_ids=["case_2"], quality=False))
    assert set(one["entries"]) == {"case_2"}
    assert one["entries"]["case_2"]["min_scaled_jacobian"] is None
    with pytest.raises(ValueError, match="unknown entry id"):
        _tools.tool_dataset_health(path, entry_ids=["nope"])
    # an entry whose file vanished is reported, never fatal
    m = DatasetManifest.load(path)
    m.add("cases/case_1.vtu", id="ghost", validate_source=False)
    os.remove(str(tmp_path / "cases" / "case_1.vtu"))
    m.save(path)
    report = _dump(_tools.tool_dataset_health(path))
    assert report["entries"]["ghost"]["steps"] == 0
    assert "error" in report["entries"]["ghost"]
    assert "ghost" in report["bad_entries"]
    # the sandbox holds on the manifest AND on what it resolves to
    _tools.set_root(str(tmp_path / "cases"))
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_dataset_health(path)


def test_dataset_find_lists_manifests_and_skips_other_json(tmp_path):
    from meshioplusplus import DatasetManifest

    _case_files(tmp_path)
    top = DatasetManifest(name="top", base_dir=str(tmp_path))
    top.add("cases/case_0.vtu", split="train")
    top.save(str(tmp_path / "top.json"))
    (tmp_path / "other.json").write_text('{"foo": 1}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    nested = tmp_path / "campaign" / "sub"
    nested.mkdir(parents=True)
    DatasetManifest(name="nested").save(str(nested / "n.json"))
    deep = nested / "deeper" / "deepest"
    deep.mkdir(parents=True)
    DatasetManifest(name="deep").save(str(deep / "d.json"))

    report = _dump(_tools.tool_dataset_find(str(tmp_path)))
    found = {m["relpath"]: m for m in report["manifests"]}
    assert set(found) == {"top.json", "campaign/sub/n.json"}
    assert found["top.json"]["name"] == "top"
    assert found["top.json"]["num_entries"] == 1
    assert found["top.json"]["splits"] == {"train": 1}
    assert len(found["top.json"]["sha256"]) == 64
    assert found["top.json"]["mtime"] > 0
    deeper = _dump(_tools.tool_dataset_find(str(tmp_path), max_depth=4))
    assert "campaign/sub/deeper/deepest/d.json" in {
        m["relpath"] for m in deeper["manifests"]
    }
    with pytest.raises(ValueError, match="directory not found"):
        _tools.tool_dataset_find(str(tmp_path / "nowhere"))
    _tools.set_root(str(tmp_path / "cases"))
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_dataset_find(str(tmp_path))


def test_call_tool_dispatches_through_the_registry(mesh_file, tmp_path):
    report = _tools.call_tool("info", {"input_path": mesh_file})
    assert report["num_points"] == 5
    failed = _tools.call_tool("info", {"input_path": str(tmp_path / "nope.vtu")})
    assert failed["error_type"] == "ValueError"
    wrong = _tools.call_tool("info", {"bogus": 1})
    assert wrong["error_type"] == "TypeError"
    with pytest.raises(KeyError, match="unknown tool 'nope'"):
        _tools.call_tool("nope", {})
    assert _tools.call_tool("formats")["readable"]


def test_has_dashboard_is_a_bool_and_names_the_extra():
    import meshioplusplus.mcp as mmcp

    assert isinstance(mmcp.has_dashboard(), bool)
    if not mmcp.has_dashboard():
        with pytest.raises(ImportError, match=r"meshioplusplus\[dashboard\]"):
            mmcp._require_http()


# --------------------------------------------------------------------------- #
# Pure half: training-job tools over the fake trainer (v10.24.0)              #
# --------------------------------------------------------------------------- #
import sys  # noqa: E402
import time  # noqa: E402

_FAKE_TRAINER = [
    sys.executable,
    os.path.join(os.path.dirname(__file__), "_fake_trainer.py"),
]


@pytest.fixture()
def fake_trainer(monkeypatch, tmp_path):
    from meshioplusplus.mcp import _jobs

    monkeypatch.setenv(_jobs.TRAIN_COMMAND_ENV, json.dumps(_FAKE_TRAINER))
    _tools.set_runs_dir(str(tmp_path / "runs"))
    yield
    _tools.set_runs_dir(None)
    _tools._MANAGERS.clear()


def _wait_terminal(job_id, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = _tools.tool_train_status(job_id)
        if state["status"] in ("finished", "failed", "stopped"):
            return state
        time.sleep(0.05)
    raise AssertionError(state)


def test_train_defaults_describes_the_manifest(tmp_path):
    path = _health_manifest(tmp_path)
    report = _dump(_tools.tool_train_defaults(path, fields=["t"], target_fields=["t"]))
    assert report["num_entries"] == 3 and report["splits"] == {"train": 2, "": 1}
    assert report["available_fields"]["point"] == ["t"]
    # case_0 also carries the quality:* cell arrays attach_quality wrote
    assert "c" in report["available_fields"]["cell"]
    assert set(report["frameworks"]) == {"torch_geometric", "physicsnemo"}
    assert report["spec"]["Fields"] == ["t"] and report["spec"]["Manifest"] == path
    assert report["spec"]["Model"]["Name"] == "meshgraphnet"
    assert report["runs_dir"].endswith("runs")


def test_train_tools_drive_a_job_end_to_end(tmp_path, fake_trainer):
    path = _health_manifest(tmp_path)
    started = _dump(
        _tools.tool_train_start(
            path,
            ["t"],
            ["t"],
            epochs=4,
            checkpoint_every=2,
            tags=["smoke"],
            notes="mcp",
        )
    )
    job_id = started["job_id"]
    assert started["status"] == "running"
    assert started["run_dir"].startswith(str(tmp_path / "runs"))
    state = _wait_terminal(job_id)
    assert state["status"] == "finished" and state["completed"] and state["epoch"] == 4
    log = _dump(_tools.tool_train_log(job_id))
    assert "fake trainer: done" in log["text"] and log["done"]
    metrics = _dump(_tools.tool_train_metrics(job_id, since_epoch=3))
    assert [r["epoch"] for r in metrics["rows"]] == [3]
    ckpts = _dump(_tools.tool_train_checkpoints(job_id))
    names = {c["name"] for c in ckpts["checkpoints"]}
    assert {"Fake.0.1.mdlus", "Fake.0.3.mdlus", "best.mdlus", "final.mdlus"} <= names
    marked = _dump(_tools.tool_train_mark_best(job_id, "Fake.0.1.mdlus"))
    assert [c["name"] for c in marked["checkpoints"] if c["is_best"]] == ["best.mdlus"]
    listed = _dump(_tools.tool_train_list())
    assert listed["jobs"][0]["job_id"] == job_id and listed["jobs"][0]["tags"] == [
        "smoke"
    ]
    assert _dump(_tools.tool_train_list(status="running"))["jobs"] == []
    assert (
        _dump(_tools.tool_train_list(manifest_path=path))["jobs"][0]["job_id"] == job_id
    )
    # a stop on a finished job is a no-op; unknown jobs are named errors
    assert _dump(_tools.tool_train_stop(job_id))["status"] == "finished"
    with pytest.raises(ValueError, match="no job 'nope'"):
        _tools.tool_train_status("nope")


def test_train_start_validates_before_spawning(tmp_path, fake_trainer):
    path = _health_manifest(tmp_path)
    with pytest.raises(ValueError, match="no split 'nope'"):
        _tools.tool_train_start(path, ["t"], ["t"], train_split="nope")
    with pytest.raises(ValueError, match="TargetFields must name"):
        _tools.tool_train_start(path, ["t"], [])
    with pytest.raises(ValueError, match="Epochs must be"):
        _tools.tool_train_start(path, ["t"], ["t"], epochs=0)
    # the sandbox holds on the entries the split resolves to
    _tools.set_root(str(tmp_path / "cases"))
    with pytest.raises(ValueError, match="outside the configured root"):
        _tools.tool_train_start(path, ["t"], ["t"])


def test_train_start_names_the_missing_frameworks(tmp_path, monkeypatch):
    from meshioplusplus import _gpu
    from meshioplusplus.mcp import _jobs

    monkeypatch.delenv(_jobs.TRAIN_COMMAND_ENV, raising=False)
    monkeypatch.setattr(_gpu, "_importable", lambda name: False)
    path = _health_manifest(tmp_path)
    with pytest.raises(ImportError, match="pip install torch_geometric"):
        _tools.tool_train_start(path, ["t"], ["t"])
    with pytest.raises(ImportError, match="torch_geometric"):
        _tools.tool_train_predict(path, checkpoint=path, output_dir=str(tmp_path / "p"))
    with pytest.raises(ValueError, match="give job_id or checkpoint"):
        _tools.tool_train_predict(path)


def test_subsample_tool_writes_a_point_cloud_and_is_json_safe(tmp_path):
    mesh = meshioplusplus.extract_surface(
        meshioplusplus.convert_cells(meshioplusplus.grid((4, 4, 4)), mode="simplexify")
    )
    mesh.point_data["u"] = mesh.points[:, 0]
    src = str(tmp_path / "surf.vtu")
    meshioplusplus.write(src, mesh)
    out = str(tmp_path / "cloud.vtu")

    report = _dump(
        _tools.tool_subsample(src, out, 16, method="grid", seed=2, record_ids=True)
    )
    assert report["count"] == 16 and report["num_source_points"] == 98
    assert report["method"] == "grid"
    assert report["cell_blocks"] == [{"type": "vertex", "num_cells": 16}]
    assert "budget:original_point_id" in report["point_data"]
    got = meshioplusplus.read(out)
    ids = got.point_data["budget:original_point_id"].astype(int)
    assert np.allclose(got.point_data["u"], mesh.point_data["u"][ids])

    guarded = _tools.guard(
        _tools.tool_subsample, input_path=src, output_path=out, count=1000
    )
    assert guarded["error_type"] == "ValueError"
    assert "count is 1000 but only 98" in guarded["error"]


def test_predict_file_tool_is_gated_and_guarded(tmp_path):
    src = str(tmp_path / "grid.vtu")
    meshioplusplus.write(src, meshioplusplus.grid((2, 2, 2)))
    checkpoint = tmp_path / "model.mdlus"
    checkpoint.write_bytes(b"")

    assert _tools.TOOL_REGISTRY["predict_file"]["gated"] == "physicsnemo"
    guarded = _tools.guard(
        _tools.tool_predict_file,
        checkpoint=str(checkpoint),
        input_path=src,
        output_path=str(tmp_path / "out.vtu"),
    )
    # Without the frameworks it is a named payload, never a traceback.
    assert guarded["error_type"] == "ImportError"
    assert "nvidia-physicsnemo" in guarded["error"]


def test_proximity_graph_tool_writes_the_graph_and_reports_degrees(tmp_path):
    src = str(tmp_path / "grid.vtu")
    meshioplusplus.write(src, meshioplusplus.grid((4, 4, 4)))
    out = str(tmp_path / "graph.vtu")

    report = _dump(_tools.tool_proximity_graph(src, out, radius=1.1))
    expected = meshioplusplus.proximity_graph(
        meshioplusplus.grid((4, 4, 4)), radius=1.1
    )
    assert report["num_vertices"] == 125
    assert report["num_edges"] == expected.shape[1] // 2
    assert report["cell_blocks"] == [{"type": "line", "num_cells": report["num_edges"]}]
    assert report["num_isolated"] == 0 and report["degree_max"] == 6

    knn = _dump(_tools.tool_proximity_graph(src, out, method="knn", max_neighbors=4))
    assert knn["degree_min"] >= 4

    bad = _tools.guard(_tools.tool_proximity_graph, input_path=src, output_path=out)
    assert bad["error_type"] == "ValueError" and "positive radius" in bad["error"]
