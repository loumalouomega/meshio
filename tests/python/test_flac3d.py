import pathlib

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import _core
from meshioplusplus.flac3d import _flac3d

from . import helpers

THIS_DIR = pathlib.Path(__file__).resolve().parent
MESH_DIR = THIS_DIR / "meshes" / "flac3d"
REFERENCE_FILES = ["flac3d_mesh_ex.f3grid", "flac3d_mesh_ex_bin.f3grid"]

# The reference mesh's five groups, as *global* (block-major) cell indices.
# Recomputed from the raw file rather than from a meshio++ read: the ZGROUP /
# FGROUP member ids resolved through the file's own zone and face id tables.
# These are the numbers issue #76 reports, minus one for 0-basing.
REFERENCE_GROUPS = {
    "face:bottom:Default": (0, 18),
    "zone:Brick1:Default": (18, 45),
    "zone:Pyramid2:Default": (45, 72),
    "zone:Wedge3:Default": (72, 99),
    "zone:Tetrahedron4:Default": (99, 126),
}


def cell_regions(mesh):
    """``{name: entries}`` over the cell regions.

    A dict, not a list: the C++ core keeps regions sorted by
    ``(kind, name, dim, tag)`` while the Python ``Mesh`` keeps insertion
    order, so any cross-engine comparison has to be order-insensitive.
    """
    return {r.name: r.entries.tolist() for r in mesh.regions if r.kind == "cell"}


def expected_groups():
    return {k: list(range(*v)) for k, v in REFERENCE_GROUPS.items()}


# Both engines, explicitly: `meshioplusplus.read` prefers the C++ core, so a
# test that only goes through it says nothing about the numpy reference reader
# (verified -- sabotaging the Python rebase leaves such a test green).
ENGINES = {
    "core": lambda path: _core.flac3d_read(str(path)),
    "python": lambda path: _flac3d.read(str(path)),
}


@pytest.mark.parametrize(
    "mesh",
    [
        helpers.empty_mesh,
        helpers.tet_mesh,
        helpers.hex_mesh,
        helpers.tet_mesh,
        helpers.add_cell_sets(helpers.tet_mesh),
    ],
)
@pytest.mark.parametrize("binary", [False, True])
def test(mesh, binary, tmp_path):
    # mesh.write("out.f3grid")
    helpers.write_read(
        tmp_path,
        lambda f, m: meshioplusplus.flac3d.write(f, m, binary=binary),
        meshioplusplus.flac3d.read,
        mesh,
        1.0e-15,
    )


@pytest.mark.parametrize("filename", REFERENCE_FILES)
def test_reference_file(filename):
    filename = MESH_DIR / filename

    mesh = meshioplusplus.read(filename)

    # points
    assert np.isclose(mesh.points.sum(), 307.0)

    # cells
    ref_num_cells = [
        ("quad", 15),
        ("triangle", 3),
        ("hexahedron", 45),
        ("pyramid", 9),
        ("hexahedron", 18),
        ("wedge", 9),
        ("hexahedron", 6),
        ("wedge", 3),
        ("hexahedron", 6),
        ("wedge", 3),
        ("pyramid", 6),
        ("tetra", 3),
    ]
    assert [
        (cell_block.type, len(cell_block)) for cell_block in mesh.cells
    ] == ref_num_cells
    # Groups. Asserting the *membership*, not just the block count -- the old
    # `len(arr) == 12` check compared the number of blocks, which
    # `global_to_blocks` guarantees for any region, so it passed while three of
    # these five groups were empty and the other two were shifted (issue #76).
    assert cell_regions(mesh) == expected_groups()


@pytest.mark.parametrize("filename", REFERENCE_FILES)
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_reference_file_groups(filename, engine):
    """The membership the issue reports, asserted on each engine separately."""
    mesh = ENGINES[engine](MESH_DIR / filename)
    assert cell_regions(mesh) == expected_groups()


@pytest.mark.parametrize("filename", REFERENCE_FILES)
def test_reference_file_cell_sets_are_block_local(filename):
    """`cell_sets` holds per-block LOCAL indices; `regions` holds global ones.

    This is the convention the whole region layer is built on, and getting it
    wrong is what issue #76 was: the reader emitted global indices into the
    per-block slots, so `blocks_to_global` added the block base a second time
    and dropped everything that overshot its own block.
    """
    mesh = meshioplusplus.read(MESH_DIR / filename)

    blocks = mesh.cell_sets["face:bottom:Default"]
    assert [len(b) for b in blocks] == [15, 3] + [0] * 10
    assert blocks[0].tolist() == list(range(15))  # the 15 quads
    assert blocks[1].tolist() == list(range(3))  # the 3 triangles

    # Brick1 is the first 27 cells of the 45-hexahedron block, not the last.
    blocks = mesh.cell_sets["zone:Brick1:Default"]
    assert blocks[2].tolist() == list(range(27))


def test_ascii_and_binary_agree():
    """The same mesh in the two encodings must give the same group names.

    The ascii slot arrives quoted (`SLOT "Default"`) and the binary one bare,
    so without stripping, `face:bottom:"Default"` and `face:bottom:Default`
    were two different regions for the same group.
    """
    for read in ENGINES.values():
        assert cell_regions(read(MESH_DIR / REFERENCE_FILES[0])) == cell_regions(
            read(MESH_DIR / REFERENCE_FILES[1])
        )


@pytest.mark.parametrize("filename", REFERENCE_FILES)
@pytest.mark.parametrize("binary", [False, True])
def test_reference_file_round_trip(filename, binary, tmp_path):
    mesh = meshioplusplus.read(MESH_DIR / filename)

    path = tmp_path / "out.f3grid"
    meshioplusplus.flac3d.write(path, mesh, binary=binary)
    back = meshioplusplus.read(path)

    assert np.allclose(back.points, mesh.points)
    assert [(c.type, len(c)) for c in back.cells] == [
        (c.type, len(c)) for c in mesh.cells
    ]
    assert cell_regions(back) == expected_groups()


@pytest.mark.parametrize("binary", [False, True])
def test_group_names_are_a_fixed_point(binary, tmp_path):
    """`<zone|face>:<name>:<slot>` must survive repeated round trips.

    The writer used to hardcode the flag and slot, so each pass re-prefixed
    them: `zone:Brick1:Default` became `zone:zone:Brick1:Default:1`.
    """
    mesh = meshioplusplus.read(MESH_DIR / REFERENCE_FILES[0])
    names = sorted(cell_regions(mesh))

    for i in range(2):
        path = tmp_path / f"pass{i}.f3grid"
        meshioplusplus.flac3d.write(path, mesh, binary=binary)
        mesh = meshioplusplus.read(path)
        assert sorted(cell_regions(mesh)) == names


@pytest.mark.parametrize("filename", REFERENCE_FILES)
def test_cpp_matches_python_read(filename):
    path = str(MESH_DIR / filename)
    cpp = _core.flac3d_read(path)
    py = _flac3d.read(path)

    assert cell_regions(cpp) == cell_regions(py)
    assert np.allclose(cpp.points, py.points)
    assert [(c.type, len(c)) for c in cpp.cells] == [(c.type, len(c)) for c in py.cells]


@pytest.mark.parametrize("binary", [False, True])
def test_cpp_matches_python_write(binary, tmp_path):
    """Both engines must produce the same bytes, groups included."""
    mesh = _flac3d.read(str(MESH_DIR / REFERENCE_FILES[0]))

    cpp_path = tmp_path / "cpp.f3grid"
    py_path = tmp_path / "py.f3grid"
    _core.flac3d_write(str(cpp_path), mesh, ".16e", binary)
    _flac3d.write(py_path, mesh, binary=binary)

    assert cpp_path.read_bytes() == py_path.read_bytes()
    assert cell_regions(_core.flac3d_read(str(cpp_path))) == expected_groups()


@pytest.mark.parametrize("binary", [False, True])
def test_mixed_zone_and_face_mesh(binary, tmp_path):
    """Writing a mesh with both zones and faces used to raise outright.

    `split_f_z` zipped one category's block sizes against the whole cell list,
    and both sections shared one running cell-id counter even though a FLAC3D
    file numbers zones and faces independently.
    """
    mesh = meshioplusplus.Mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [("tetra", [[0, 1, 2, 3]]), ("triangle", [[0, 1, 2]])],
        regions=[meshioplusplus.Region("mixed", "cell", [0, 1])],
    )

    path = tmp_path / "mixed.f3grid"
    meshioplusplus.flac3d.write(path, mesh, binary=binary)
    back = meshioplusplus.read(path)

    # The blocks come back faces-first, so the triangle is global 0.
    assert [c.type for c in back.cells] == ["triangle", "tetra"]
    # A region spanning both categories splits into an FGROUP and a ZGROUP.
    assert cell_regions(back) == {
        "face:mixed:Default": [0],
        "zone:mixed:Default": [1],
    }


def test_unknown_group_ids_are_dropped(tmp_path, capsys):
    """A group member the file never defines is dropped, never guessed at."""
    path = tmp_path / "unknown.f3grid"
    path.write_text(
        "* GRIDPOINTS\n"
        "G 1 0 0 0\n"
        "G 2 1 0 0\n"
        "G 3 0 1 0\n"
        "G 4 0 0 1\n"
        "* ZONES\n"
        "Z T4 1 1 2 3 4\n"
        "* ZONE GROUPS\n"
        'ZGROUP "g" SLOT "Default"\n'
        " 1 999\n"
    )
    mesh = _flac3d.read(path)
    assert cell_regions(mesh) == {"zone:g:Default": [0]}
    assert "does not define" in capsys.readouterr().err


@pytest.mark.parametrize("binary", [False, True])
def test_named_cell_set_round_trips(binary, tmp_path):
    """A bare set name is namespaced into the file's own vocabulary."""
    mesh = helpers.add_cell_sets(helpers.tet_mesh)
    n = len(mesh.cells[0])

    path = tmp_path / "sets.f3grid"
    meshioplusplus.flac3d.write(path, mesh, binary=binary)
    back = meshioplusplus.read(path)

    assert cell_regions(back) == {
        "zone:grain0:Default": list(range(0, n // 2)),
        "zone:grain1:Default": list(range(n // 2, n)),
    }
