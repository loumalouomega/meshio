import json

import numpy as np
import pytest

import meshioplusplus

from . import helpers


def is_same_mesh(mesh0, mesh1, atol):
    if not np.allclose(mesh0.points, mesh1.points, atol=atol, rtol=0.0):
        return False
    for cells0, cells1 in zip(mesh0.cells, mesh1.cells):
        if cells0.type != cells1.type or not np.allclose(cells0.data, cells1.data):
            return False
    return True


def test_info(tmp_path):
    infile = tmp_path / "out.msh"
    meshioplusplus.write(infile, helpers.tri_mesh, file_format="gmsh")
    meshioplusplus._cli.main(["info", str(infile), "--input-format", "gmsh"])


def test_convert(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.msh"
    meshioplusplus.write(infile, helpers.tri_mesh, file_format="gmsh")

    outfile = tmp_path / "out.msh"
    meshioplusplus._cli.main(
        [
            "convert",
            str(infile),
            str(outfile),
            "--input-format",
            "gmsh",
            "--output-format",
            "vtk",
            "--sets-to-int-data",
        ]
    )

    mesh = meshioplusplus.read(outfile, file_format="vtk")

    atol = 1.0e-15
    assert np.allclose(input_mesh.points, mesh.points, atol=atol, rtol=0.0)

    for cells0, cells1 in zip(input_mesh.cells, mesh.cells):
        assert cells0.type == cells1.type
        assert np.allclose(cells0.data, cells1.data)


def test_compress(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, input_mesh)

    meshioplusplus._cli.main(["decompress", str(infile)])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)

    meshioplusplus._cli.main(["compress", str(infile)])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-15)


def test_ascii_binary(tmp_path):
    input_mesh = helpers.tri_mesh

    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, input_mesh)

    meshioplusplus._cli.main(["ascii", str(infile)])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-12)

    meshioplusplus._cli.main(["binary", str(infile)])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(input_mesh, mesh, atol=1.0e-12)


def test_version():
    # `--version` is an argparse action that prints and exits 0.
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main(["--version"])
    assert exc.value.code == 0


def test_no_command():
    # A required subcommand is missing -> argparse exits with code 2.
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main([])
    assert exc.value.code == 2


def test_convert_format_inference(tmp_path):
    # No --input-format/--output-format: the formats are inferred from the
    # extensions, and the `c` alias is used.
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.vtk"
    meshioplusplus.write(infile, helpers.tri_mesh)

    meshioplusplus._cli.main(["c", str(infile), str(outfile)])

    mesh = meshioplusplus.read(outfile)
    assert is_same_mesh(helpers.tri_mesh, mesh, atol=1.0e-12)


def test_info_inference(tmp_path):
    # `info` with extension inference, via the `i` alias.
    infile = tmp_path / "mesh.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh)
    meshioplusplus._cli.main(["i", str(infile)])


def test_decompress_standalone(tmp_path):
    # `decompress` on a compressed VTU produces a still-readable, larger file.
    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh, compression="zlib")
    meshioplusplus._cli.main(["decompress", str(infile)])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(helpers.tri_mesh, mesh, atol=1.0e-12)


def test_compress_max_flag(tmp_path):
    # The `--max` flag switches VTU to lzma; the result must round-trip.
    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh, binary=True, compression=None)
    meshioplusplus._cli.main(["compress", str(infile), "--max"])
    mesh = meshioplusplus.read(infile)
    assert is_same_mesh(helpers.tri_mesh, mesh, atol=1.0e-12)


def test_convert_ascii_and_float_format(tmp_path):
    # Exercise the `-a`/`--ascii` and `-f`/`--float-format` flag plumbing.
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.vtk"
    meshioplusplus.write(infile, helpers.tri_mesh)
    meshioplusplus._cli.main(["convert", str(infile), str(outfile), "-a", "-f", ".8e"])
    mesh = meshioplusplus.read(outfile)
    assert is_same_mesh(helpers.tri_mesh, mesh, atol=1.0e-7)


def test_convert_int_data_to_sets(tmp_path):
    # The `-d`/`--int-data-to-sets` path (converse of `--sets-to-int-data`).
    # Regression test: `convert -d` used to raise "dictionary changed size
    # during iteration" because *_data_to_sets(key) mutates the dict being
    # iterated (fixed by snapshotting the keys in _cli/_convert.py).
    mesh = helpers.tri_mesh.copy()
    # Both loops (point_data and cell_data) must survive the in-place mutation.
    mesh.point_data = {"fixed-loose": np.array([0, 0, 1, 1])}
    mesh.cell_data = {"grain0-grain1": [np.array([0, 1])]}

    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.vtu"
    meshioplusplus.write(infile, mesh)
    meshioplusplus._cli.main(["convert", str(infile), str(outfile), "-d"])
    # The command completes and the geometry survives.
    out = meshioplusplus.read(outfile)
    assert np.allclose(out.points, mesh.points)


def test_ascii_unsupported_format_returns_1(tmp_path):
    # `.off` is not in the ascii command's whitelist -> error + `return 1`.
    infile = tmp_path / "in.off"
    meshioplusplus.write(infile, helpers.tri_mesh)
    rc = meshioplusplus._cli.main(["ascii", str(infile)])
    assert rc == 1


@pytest.mark.parametrize("command", ["binary", "compress", "decompress"])
def test_unsupported_format_exits_1(command, tmp_path):
    # binary/compress/decompress use `exit(1)` (SystemExit), unlike ascii.
    infile = tmp_path / "in.off"
    meshioplusplus.write(infile, helpers.tri_mesh)
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main([command, str(infile)])
    assert exc.value.code == 1


def test_info_inconsistent_points_warns(monkeypatch, capsys):
    # A mesh whose cells reference a nonexistent point triggers the
    # "Inconsistent mesh" warning branch of `info` (_info.py:28).
    bad = meshioplusplus.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        [("triangle", np.array([[0, 1, 9]]))],
    )
    monkeypatch.setattr(meshioplusplus._cli._info, "read", lambda *a, **k: bad)
    meshioplusplus._cli.main(["info", "ignored"])
    assert "Inconsistent" in capsys.readouterr().err


def test_info_unused_points_warns(monkeypatch, capsys):
    # A consistent mesh with an unused point triggers the second warning
    # branch of `info` (_info.py:37).
    mesh = meshioplusplus.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [5.0, 5.0, 5.0]],
        [("triangle", np.array([[0, 1, 2]]))],
    )
    monkeypatch.setattr(meshioplusplus._cli._info, "read", lambda *a, **k: mesh)
    meshioplusplus._cli.main(["info", "ignored"])
    assert "not part of any cell" in capsys.readouterr().err


# --- the nested `data` command group ---------------------------------------


def _data_cli_mesh():
    """A two-block mesh carrying point and cell data, for the data verbs."""
    m = meshioplusplus.Mesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        ),
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]])),
            ("quad", np.array([[1, 4, 5, 2]])),
        ],
    )
    m.point_data["T"] = np.array([0.0, 1.0, 11.0, 10.0, 2.0, 12.0])
    m.point_data["velocity"] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    m.cell_data["mat"] = [np.array([1.0, 2.0]), np.array([3.0])]
    return m


@pytest.fixture()
def data_infile(tmp_path):
    path = tmp_path / "in.vtu"
    meshioplusplus.write(path, _data_cli_mesh())
    return path


def test_data_info(data_infile, capsys):
    rc = meshioplusplus._cli.main(["data", "info", str(data_infile)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "T" in out
    assert "velocity" in out
    assert "mat" in out


def test_data_info_json(data_infile, capsys):
    rc = meshioplusplus._cli.main(["data", "info", str(data_infile), "--json"])
    assert rc == 0
    arrays = json.loads(capsys.readouterr().out)
    names = {(a["location"], a["name"]) for a in arrays}
    assert ("point_data", "T") in names
    assert ("cell_data", "mat") in names


def test_data_calc(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        [
            "data",
            "calc",
            str(data_infile),
            str(out),
            "--point",
            "speed = norm(velocity)",
        ]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    expected = np.linalg.norm(_data_cli_mesh().point_data["velocity"], axis=1)
    assert np.allclose(mesh.point_data["speed"], expected)


def test_data_to_cell(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        [
            "data",
            "to-cell",
            str(data_infile),
            str(out),
            "--keys",
            "T",
            "--target-suffix",
            "_c",
        ]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert "T_c" in mesh.cell_data


def test_data_to_point_weighted(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        ["data", "to-point", str(data_infile), str(out), "--keys", "mat", "--weighted"]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    # Point 2 is shared by both triangles (area 1/2) and the quad (area 1).
    assert mesh.point_data["mat"][2] == pytest.approx(2.25)


def test_data_clamp(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        [
            "data",
            "clamp",
            str(data_infile),
            str(out),
            "--point",
            "T",
            "--min",
            "0",
            "--max",
            "10",
        ]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert mesh.point_data["T"].max() == pytest.approx(10.0)


def test_data_normalize(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        [
            "data",
            "normalize",
            str(data_infile),
            str(out),
            "--cell",
            "mat",
            "--to",
            "0,1",
        ]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    values = np.concatenate([np.asarray(b) for b in mesh.cell_data["mat"]])
    assert values.min() == pytest.approx(0.0)
    assert values.max() == pytest.approx(1.0)


def test_data_normalize_zero_mean(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        ["data", "normalize", str(data_infile), str(out), "--point", "T", "--zero-mean"]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert mesh.point_data["T"].mean() == pytest.approx(0.0, abs=1e-12)
    assert mesh.point_data["T"].std() == pytest.approx(1.0)


def test_data_rename(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        ["data", "rename", str(data_infile), str(out), "--point", "T:temperature"]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert "temperature" in mesh.point_data
    assert "T" not in mesh.point_data


def test_data_rename_splits_on_the_last_colon(tmp_path):
    # Data names routinely contain colons, so `--point gmsh:physical:tag` must
    # rename `gmsh:physical` to `tag`.
    mesh = _data_cli_mesh()
    mesh.point_data["gmsh:physical"] = np.arange(6.0)
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.vtu"
    meshioplusplus.write(infile, mesh)
    rc = meshioplusplus._cli.main(
        ["data", "rename", str(infile), str(outfile), "--point", "gmsh:physical:tag"]
    )
    assert rc == 0
    back = meshioplusplus.read(outfile)
    assert "tag" in back.point_data
    assert "gmsh:physical" not in back.point_data


def test_data_drop(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        ["data", "drop", str(data_infile), str(out), "--point", "T,velocity"]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert "T" not in mesh.point_data
    assert "velocity" not in mesh.point_data
    assert "mat" in mesh.cell_data


def test_data_keep(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        ["data", "keep", str(data_infile), str(out), "--point", "T"]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert set(mesh.point_data) == {"T"}
    # A location the whitelist does not mention is untouched.
    assert "mat" in mesh.cell_data


def test_data_estimate_error(data_infile, tmp_path, capsys):
    out = tmp_path / "out.vtu"
    rc = meshioplusplus._cli.main(
        [
            "data",
            "estimate-error",
            str(data_infile),
            str(out),
            "--array",
            "T",
            "--marking",
            "absolute",
            "--marking-value",
            "1e-9",
        ]
    )
    assert rc == 0
    mesh = meshioplusplus.read(out)
    assert "error:zz" in mesh.cell_data
    assert "error:marked" in mesh.cell_data
    out_text = capsys.readouterr().out
    assert "global error" in out_text
    assert "cells marked" in out_text


def test_data_estimate_error_rejects_cell_data(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    with pytest.raises(ValueError, match="cell_data_to_point_data"):
        meshioplusplus._cli.main(
            ["data", "estimate-error", str(data_infile), str(out), "--array", "mat"]
        )


def test_data_without_a_verb_exits_2():
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main(["data"])
    assert exc.value.code == 2


def test_data_unknown_verb_exits_2():
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main(["data", "nonsense"])
    assert exc.value.code == 2


def test_data_geometry_is_never_modified(data_infile, tmp_path):
    out = tmp_path / "out.vtu"
    meshioplusplus._cli.main(
        [
            "data",
            "calc",
            str(data_infile),
            str(out),
            "--point",
            "speed = norm(velocity)",
        ]
    )
    before = meshioplusplus.read(data_infile)
    after = meshioplusplus.read(out)
    assert np.array_equal(before.points, after.points)
    for a, b in zip(before.cells, after.cells):
        assert a.type == b.type
        assert np.array_equal(np.asarray(a.data), np.asarray(b.data))


def test_view_browser_writes_a_page(tmp_path, monkeypatch):
    """The browser backend needs neither polyscope nor a display."""
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)

    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh)
    assert meshioplusplus._cli.main(["view", str(infile), "--backend", "browser"]) == 0
    assert len(opened) == 1


def test_view_rejects_an_unknown_backend_at_parse_time(tmp_path):
    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh)
    with pytest.raises(SystemExit) as exc:
        meshioplusplus._cli.main(["view", str(infile), "--backend", "opengl"])
    assert exc.value.code == 2


def test_screenshot_without_polyscope_says_how_to_install_it(tmp_path, monkeypatch):
    from meshioplusplus._cli import _view

    monkeypatch.setattr(_view, "has_viewer", lambda: False)
    infile = tmp_path / "in.vtu"
    meshioplusplus.write(infile, helpers.tri_mesh)
    with pytest.raises(SystemExit, match=r"pip install meshioplusplus\[viewer\]"):
        meshioplusplus._cli.main(["screenshot", str(infile), str(tmp_path / "o.png")])


@pytest.mark.viewer
def test_screenshot_verb_writes_a_png(tmp_path):
    pytest.importorskip("polyscope")
    infile = tmp_path / "in.vtu"
    out = tmp_path / "out.png"
    meshioplusplus.write(infile, helpers.tet_mesh)
    try:
        rc = meshioplusplus._cli.main(
            ["screenshot", str(infile), str(out), "--size", "160", "120"]
        )
    except RuntimeError as e:  # pragma: no cover - depends on the GL runtime
        pytest.skip(str(e))
    assert rc == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# --- convert --color-by ------------------------------------------------------


def _colored_mesh():
    return meshioplusplus.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [("tetra", [[0, 1, 2, 3]])],
        cell_data={"tag": [np.array([1.0])]},
        point_data={"T": np.array([0.0, 1.0, 2.0, 3.0])},
    )


@pytest.mark.parametrize("ext, cmap", [("svg", "viridis"), ("tikz", "turbo")])
def test_convert_color_by(tmp_path, ext, cmap):
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / f"out.{ext}"
    meshioplusplus.write(infile, _colored_mesh())
    meshioplusplus._cli.main(
        [
            "convert",
            str(infile),
            str(outfile),
            "--color-by",
            "T",
            "--cmap",
            cmap,
            "--colorbar",
        ]
    )
    text = outfile.read_text()
    assert text
    # The default flat fill is gone from the drawn faces: every one carries a
    # mapped colour instead.
    if ext == "svg":
        assert 'fill="#' in text
        assert "<rect " in text  # the colorbar
    else:
        assert "fill={rgb,255:red," in text
        assert "\\fill[" in text


def test_convert_color_by_requires_svg_or_tikz(tmp_path):
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.vtk"
    meshioplusplus.write(infile, _colored_mesh())
    with pytest.raises(ValueError, match="only supported for svg/tikz"):
        meshioplusplus._cli.main(
            ["convert", str(infile), str(outfile), "--color-by", "tag"]
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--cmap", "turbo"],
        ["--vmin", "0"],
        ["--vmax", "1"],
        ["--component", "0"],
        ["--nan-color", "red"],
        ["--colorbar"],
    ],
)
def test_convert_color_modifiers_require_color_by(tmp_path, flags):
    # A modifier without --color-by is a mistake, not a silent no-op.
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.svg"
    meshioplusplus.write(infile, _colored_mesh())
    with pytest.raises(ValueError, match="require"):
        meshioplusplus._cli.main(["convert", str(infile), str(outfile), *flags])


def test_convert_color_by_unknown_array(tmp_path):
    infile = tmp_path / "in.vtu"
    outfile = tmp_path / "out.svg"
    meshioplusplus.write(infile, _colored_mesh())
    with pytest.raises(ValueError, match="no point_data or cell_data array"):
        meshioplusplus._cli.main(
            ["convert", str(infile), str(outfile), "--color-by", "nope"]
        )


# --------------------------------------------------------------------------- #
# the `dataset` group (dataset manifests, doc/datasets.md)                    #
# --------------------------------------------------------------------------- #
def _dataset_cases(tmp_path, n=4):
    cases = tmp_path / "cases"
    cases.mkdir(exist_ok=True)
    for i in range(n):
        meshioplusplus.write(cases / f"case_{i}.vtu", helpers.tri_mesh)
    return cases


def test_dataset_group_end_to_end(tmp_path, monkeypatch, capsys):
    _dataset_cases(tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = "m.json"
    # add: a glob and an explicit two-path case
    assert (
        meshioplusplus._cli.main(
            [
                "dataset",
                "add",
                manifest,
                "cases/case_*.vtu",
                "--id",
                "sweep",
                "--tag",
                "raw",
                "--meta",
                "Re=100",
            ]
        )
        == 0
    )
    assert (
        meshioplusplus._cli.main(
            [
                "dataset",
                "add",
                manifest,
                "cases/case_0.vtu",
                "cases/case_1.vtu",
                "--id",
                "pair",
                "--group",
                "g/h",
            ]
        )
        == 0
    )
    doc = json.loads((tmp_path / manifest).read_text(encoding="utf-8"))
    assert [e["Id"] for e in doc["Entries"]] == ["sweep", "pair"]
    # sources are stored relative to the manifest's directory
    assert doc["Entries"][0]["Source"]["Pattern"] == "cases/case_*.vtu"
    assert doc["Entries"][0]["Metadata"] == {"Re": 100}
    # split --set, then tag / annotate
    assert (
        meshioplusplus._cli.main(
            ["dataset", "split", manifest, "--id", "sweep", "--set", "train"]
        )
        == 0
    )
    assert (
        meshioplusplus._cli.main(
            ["dataset", "tag", manifest, "--all", "--add", "v1", "--remove", "raw"]
        )
        == 0
    )
    assert (
        meshioplusplus._cli.main(
            [
                "dataset",
                "annotate",
                manifest,
                "--id",
                "pair",
                "--notes",
                "odd",
                "--meta",
                "Ma=0.3",
            ]
        )
        == 0
    )
    m = meshioplusplus.DatasetManifest.load(str(tmp_path / manifest))
    assert m["sweep"].split == "train" and m["sweep"].tags == ("v1",)
    assert m["pair"].notes == "odd" and m["pair"].metadata == {"Ma": 0.3}
    # list --json round-trips the document's entries, with resolved plans
    capsys.readouterr()
    assert (
        meshioplusplus._cli.main(["dataset", "list", manifest, "--resolve", "--json"])
        == 0
    )
    listing = json.loads(capsys.readouterr().out)
    assert [e["Id"] for e in listing] == ["sweep", "pair"]
    assert len(listing[0]["Resolved"]) == 4 and len(listing[1]["Resolved"]) == 2
    # filters reach the CLI
    capsys.readouterr()
    assert (
        meshioplusplus._cli.main(
            ["dataset", "list", manifest, "--split", "train", "--json"]
        )
        == 0
    )
    assert [e["Id"] for e in json.loads(capsys.readouterr().out)] == ["sweep"]


def test_dataset_split_assign_is_deterministic(tmp_path, monkeypatch):
    _dataset_cases(tmp_path, n=10)
    monkeypatch.chdir(tmp_path)
    for i in range(10):
        meshioplusplus._cli.main(
            ["dataset", "add", "m.json", f"cases/case_{i}.vtu", "--id", f"c{i}"]
        )
    assert (
        meshioplusplus._cli.main(
            [
                "dataset",
                "split",
                "m.json",
                "--assign",
                "train=0.8,test=0.2",
                "--seed",
                "3",
            ]
        )
        == 0
    )
    m1 = meshioplusplus.DatasetManifest.load("m.json")
    counts = m1.splits()
    assert counts == {"train": 8, "test": 2}
    meshioplusplus._cli.main(
        ["dataset", "split", "m.json", "--assign", "train=0.8,test=0.2", "--seed", "3"]
    )
    m2 = meshioplusplus.DatasetManifest.load("m.json")
    assert [e.split for e in m1] == [e.split for e in m2]


def test_dataset_add_empty_glob_fails_by_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(meshioplusplus.ReadError):
        meshioplusplus._cli.main(["dataset", "add", "m.json", "missing_*.vtu"])
    # nothing half-written
    assert not (tmp_path / "m.json").exists()


def test_dataset_hand_edit_survives_cli_edit(tmp_path, monkeypatch):
    _dataset_cases(tmp_path, n=1)
    monkeypatch.chdir(tmp_path)
    meshioplusplus._cli.main(["dataset", "add", "m.json", "cases/case_0.vtu"])
    doc = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    doc["Entries"][0]["Notes"] = "hand-written"
    (tmp_path / "m.json").write_text(json.dumps(doc, indent=2) + "\n")
    meshioplusplus._cli.main(
        ["dataset", "split", "m.json", "--id", "case_0", "--set", "train"]
    )
    m = meshioplusplus.DatasetManifest.load("m.json")
    assert m["case_0"].notes == "hand-written" and m["case_0"].split == "train"


# --------------------------------------------------------------------------- #
# grid transfer                                                               #
# --------------------------------------------------------------------------- #
@pytest.fixture
def grid_source(tmp_path):
    """A tetrahedral mesh over an anisotropic box, carrying a linear field."""
    mesh = meshioplusplus.convert_cells(
        meshioplusplus.grid((3, 3, 3), spacing=(1 / 3, 2 / 3, 4 / 3)), mode="simplexify"
    )
    p = mesh.points
    mesh.point_data["u"] = 1.0 + 2.0 * p[:, 0] + 3.0 * p[:, 1] - 5.0 * p[:, 2]
    mesh.point_data["v"] = np.column_stack([p[:, 0], 2.0 * p[:, 1], -p[:, 2]])
    path = tmp_path / "src.vtu"
    meshioplusplus.write(path, mesh)
    return path


def test_cli_grid_round_trip(grid_source, tmp_path, capsys):
    """sample -> resample -> scatter, the whole chain through the CLI.

    A linear field must survive all three steps, and the multi-component array
    must come back with its shape.
    """
    gridfile = tmp_path / "g.vti"
    fine = tmp_path / "fine.vtu"
    upfile = tmp_path / "up.vti"
    outfile = tmp_path / "out.vtu"
    meshioplusplus.write(
        fine, meshioplusplus.grid((6, 6, 6), spacing=(1 / 6, 2 / 6, 4 / 6))
    )

    meshioplusplus._cli.main(
        ["grid-sample", str(grid_source), str(gridfile), "--resolution", "4,4,4"]
    )
    assert "coverage:" in capsys.readouterr().out

    meshioplusplus._cli.main(
        ["grid-resample", str(gridfile), str(upfile), "--factor", "2"]
    )
    assert "(2, 2, 2)" in capsys.readouterr().out

    meshioplusplus._cli.main(["grid-scatter", str(upfile), str(fine), str(outfile)])

    out = meshioplusplus.read(outfile)
    p = out.points
    assert out.point_data["u"] == pytest.approx(
        1.0 + 2.0 * p[:, 0] + 3.0 * p[:, 1] - 5.0 * p[:, 2], abs=1e-12
    )
    assert out.point_data["v"].shape == (len(p), 3)


def test_cli_grid_spectrum_json(tmp_path, capsys):
    mesh = meshioplusplus.grid((15, 15, 15), spacing=(1 / 16, 1 / 16, 1 / 16))
    mesh.point_data["u"] = np.cos(2 * np.pi * 3 * mesh.points[:, 0])
    path = tmp_path / "g.vti"
    meshioplusplus.write(path, mesh)

    meshioplusplus._cli.main(["grid-spectrum", str(path), "--field", "u", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["units"] == "cycles per unit length"
    # Parseval, and the mode lands in the bin its wavenumber names
    assert report["total_power"] == pytest.approx(
        float((mesh.point_data["u"] ** 2).mean()), rel=1e-12
    )
    loud = int(np.argmax(report["power"]))
    assert report["wavenumber"][loud] == pytest.approx(3.0)


def test_cli_grid_resample_requires_exactly_one_target(grid_source, tmp_path):
    gridfile = tmp_path / "g.vti"
    meshioplusplus._cli.main(
        ["grid-sample", str(grid_source), str(gridfile), "--resolution", "2,2,2"]
    )
    with pytest.raises(ValueError, match="exactly one of --factor and --resolution"):
        meshioplusplus._cli.main(
            ["grid-resample", str(gridfile), str(tmp_path / "x.vti")]
        )


def test_cli_subsample_writes_a_point_cloud(tmp_path, capsys):
    mesh = meshioplusplus.extract_surface(
        meshioplusplus.convert_cells(meshioplusplus.grid((4, 4, 4)), mode="simplexify")
    )
    mesh.point_data["u"] = mesh.points[:, 0]
    src = tmp_path / "surf.vtu"
    meshioplusplus.write(src, mesh)
    out = tmp_path / "cloud.vtp"

    meshioplusplus._cli.main(
        ["subsample", str(src), str(out), "--count", "20", "--record-ids"]
    )
    assert "subsampled 98 points to 20" in capsys.readouterr().out
    got = meshioplusplus.read(out)
    assert len(got.points) == 20
    assert [cb.type for cb in got.cells] == ["vertex"]
    ids = got.point_data["budget:original_point_id"].astype(int)
    assert np.allclose(got.points, mesh.points[ids])
    assert np.allclose(got.point_data["u"], mesh.point_data["u"][ids])

    # --method grid, a random start and a box, all accepted
    meshioplusplus._cli.main(
        [
            "subsample",
            str(src),
            str(out),
            "--count",
            "5",
            "--method",
            "grid",
            "--start",
            "random",
            "--seed",
            "3",
            "--bounds=0,0,0,4,4,2",
            "-q",
        ]
    )
    got = meshioplusplus.read(out)
    assert len(got.points) == 5 and got.points[:, 2].max() <= 2.0
    with pytest.raises(ValueError, match="count is 1000 but only"):
        meshioplusplus._cli.main(["subsample", str(src), str(out), "--count", "1000"])
