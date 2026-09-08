"""The cross-format region round-trip matrix.

One fixture mesh carrying **all three** region kinds is written to and read back
from every format meshio++ can currently map regions onto, and the table below
declares what each format keeps and what it loses. The table is the point: it
makes the lossiness executable documentation rather than prose that drifts, and
a format that silently starts or stops carrying a kind fails here.

Phase 1 maps Gmsh, Abaqus and MED. Exodus reads regions (element blocks, node
sets and side sets) but does not yet write them, so it is a **read-only** entry
recorded below rather than a row here -- this matrix is a round-trip table, and
a format that cannot write cannot round-trip. FLAC3D round-trips a cell
region's *membership* but rewrites its *name* into the file's own
``<zone|face>:<name>:<slot>`` vocabulary, so it gets its own bucket too rather
than weakening this table's exact-name assertion. UNV, Ansys, OpenFOAM and XDMF
are deferred entirely. See ``doc/regions.md``.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import meshioplusplus

try:
    import h5py  # noqa: F401

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


def fixture_mesh():
    """Two tetrahedra carrying a point, a cell and a side region.

    Tetrahedra so that side entries have somewhere to point: each cell has four
    faces, numbered as ``detail/cell_faces.hpp`` numbers them.
    """
    return meshioplusplus.Mesh(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        [("tetra", [[0, 1, 2, 4], [0, 2, 3, 4]])],
        regions=[
            meshioplusplus.Region("clamped", "point", [0, 3]),
            meshioplusplus.Region("solid", "cell", [0, 1], dim=3, tag=42),
            meshioplusplus.Region("wall", "side", [[0, 1], [1, 3]], dim=2),
        ],
    )


# --------------------------------------------------------------------------- #
# THE MATRIX                                                                   #
#                                                                              #
# Per format: which region kinds survive a write + read, and whether the        #
# format-native integer `tag` comes back with them. Every "no" is a real,       #
# understood limitation of the file format or of the Phase-1 mapping, spelled   #
# out in `why` so the table doubles as the lossiness documentation.             #
# --------------------------------------------------------------------------- #
MATRIX = [
    pytest.param(
        "abaqus",
        ".inp",
        {"point": True, "cell": True, "side": True},
        {"tag": False},
        "*NSET / *ELSET / *SURFACE map onto the three kinds exactly. Abaqus "
        "names its groups but has no integer id for them, so `tag` is lost.",
        id="abaqus",
    ),
    pytest.param(
        "gmsh22",
        ".msh",
        {"point": False, "cell": True, "side": False},
        {"tag": True},
        "A gmsh physical group is a named, tagged, per-dimension group of "
        "*elements*: the per-element tag column carries cell membership and "
        "$PhysicalNames the name/tag/dim. There is no node-set and no "
        "side-set concept, so point and side regions are dropped. (A gmsh "
        "dimension-0 group tags `vertex` cells, which is a cell region.)",
        id="gmsh22",
    ),
    pytest.param(
        "med",
        ".med",
        {"point": True, "cell": True, "side": False},
        {"tag": False},
        "A MED family (FAS/NOEUD or FAS/ELEME) is a named group of nodes or "
        "elements, so point and cell regions map directly -- one family per "
        "unique combination of region names an entity belongs to. A family "
        "id is per-combination rather than per-name, and a name may span "
        "several ids, so the format-native `tag` is not carried. MED has no "
        "facet-group concept, so side regions are dropped.",
        marks=pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed"),
        id="med",
    ),
]


@pytest.mark.parametrize("fmt, suffix, survives, carries, why", MATRIX)
def test_region_round_trip(fmt, suffix, survives, carries, why, tmp_path):
    mesh = fixture_mesh()
    path = tmp_path / ("regions" + suffix)
    meshioplusplus.write(path, mesh, file_format=fmt)
    back = meshioplusplus.read(path)

    before = {(r.name, r.kind): r for r in mesh.regions}
    after = {(r.name, r.kind): r for r in back.regions}

    for name, kind in before:
        expected = survives[kind]
        got = (name, kind) in after
        assert got == expected, (
            f"{fmt}: region '{name}' ({kind}) "
            f"{'vanished' if expected else 'unexpectedly survived'} — {why}"
        )
        if not got:
            continue
        # Membership must survive exactly. Entries are canonical on both sides
        # (sorted, de-duplicated), so this is an equality, not a set compare.
        assert_array_equal(
            after[(name, kind)].entries,
            before[(name, kind)].entries,
            err_msg=f"{fmt}: region '{name}' ({kind}) changed membership",
        )
        if carries["tag"]:
            assert (
                after[(name, kind)].tag == before[(name, kind)].tag
            ), f"{fmt}: region '{name}' lost its format-native tag"


@pytest.mark.parametrize("fmt, suffix, survives, carries, why", MATRIX)
def test_geometry_is_unaffected_by_regions(
    fmt, suffix, survives, carries, why, tmp_path
):
    """Carrying regions must not perturb points or connectivity."""
    mesh = fixture_mesh()
    path = tmp_path / ("regions" + suffix)
    meshioplusplus.write(path, mesh, file_format=fmt)
    back = meshioplusplus.read(path)

    assert np.allclose(back.points, mesh.points)
    assert len(back.cells) == len(mesh.cells)
    assert_array_equal(np.asarray(back.cells[0].data), np.asarray(mesh.cells[0].data))


@pytest.mark.parametrize("fmt, suffix, survives, carries, why", MATRIX)
def test_no_regions_writes_the_same_bytes(
    fmt, suffix, survives, carries, why, tmp_path
):
    """A mesh with no regions must be byte-identical to one whose regions were
    stripped — the guarantee that this feature costs existing files nothing."""
    mesh = fixture_mesh()
    plain = meshioplusplus.Mesh(
        mesh.points, [("tetra", np.asarray(mesh.cells[0].data))]
    )

    a = tmp_path / ("a" + suffix)
    b = tmp_path / ("b" + suffix)
    meshioplusplus.write(a, plain, file_format=fmt)
    stripped = fixture_mesh()
    stripped.regions.clear()
    meshioplusplus.write(b, stripped, file_format=fmt)

    assert a.read_bytes() == b.read_bytes()


def test_side_regions_are_the_new_capability():
    """No format could express a side set before; Abaqus now can.

    Spelled out separately because it is the one kind with no `point_sets` /
    `cell_sets` equivalent at all — it is only reachable through `.regions`.
    """
    side_capable = [p.values[0] for p in MATRIX if p.values[2]["side"]]
    assert side_capable == ["abaqus"]


# --------------------------------------------------------------------------- #
# Deferred to Phase 2 — recorded so the gap is explicit, not forgotten.        #
# --------------------------------------------------------------------------- #
PHASE_2 = {
    "unv": "groups (absorbing UnvInfo)",
    "ansysInp": "components (absorbing AnsysInfo)",
    "openfoam": "boundary patches, which are face groups (side regions)",
    "xdmf": "XDMF Sets",
    "vtu": "no native set concept — a convention has to be chosen, not invented silently",
}


@pytest.mark.parametrize("fmt", sorted(PHASE_2))
def test_phase_2_formats_do_not_carry_regions_yet(fmt, tmp_path):
    """These formats keep working; they simply do not map regions yet.

    The assertion is deliberately weak — it only pins that nothing *claims* to
    round-trip a region it cannot. When a format graduates to Phase 2, move it
    into MATRIX and delete its entry here.
    """
    assert fmt not in {
        p.values[0] for p in MATRIX
    }, f"{fmt} now carries regions: move it from PHASE_2 into MATRIX"


# --------------------------------------------------------------------------- #
# Read-only region support: the reader maps regions but the writer does not     #
# emit them yet, so these formats can be a *source* of regions but not a        #
# round-trip target. Recorded here so the half-finished state is explicit.      #
# --------------------------------------------------------------------------- #
READ_ONLY_REGIONS = {
    "exodus": (
        "reads element blocks -> Cell, node sets -> Point and side sets -> Side "
        "(see tests/python/test_exodus.py); the writer still emits neither "
        "eb_names nor side sets, so a region written here would not come back"
    ),
}


# --------------------------------------------------------------------------- #
# Round-trips membership but RENAMES: the format's own namespace is part of a  #
# group's identity, so a region's name is rewritten into it on write and is a  #
# fixed point only thereafter. Not a MATRIX row -- that table asserts the name #
# survives *exactly*, which is the right assertion for every format that can   #
# make it, and weakening it for one exception would cost the other three.      #
# --------------------------------------------------------------------------- #
NAMESPACED_REGIONS = {
    "flac3d": (
        ".f3grid",
        "zone:solid:Default",
        "A FLAC3D group is identified by all three of (ZGROUP vs FGROUP, name, "
        "slot) -- zones and faces are separate namespaces and a slot partitions "
        "the groups within one -- so meshio++ names it "
        "`<zone|face>:<name>:<slot>`. A region named `solid` on a zone mesh "
        "comes back as `zone:solid:Default`; a name already in that form "
        "round-trips unchanged. FLAC3D has no node-set and no facet-group "
        "concept, so point and side regions are dropped.",
    ),
}


@pytest.mark.parametrize("fmt", sorted(NAMESPACED_REGIONS))
def test_namespaced_region_formats_keep_membership(fmt, tmp_path):
    """Membership survives exactly; the name is rewritten, then stable."""
    suffix, renamed, why = NAMESPACED_REGIONS[fmt]
    mesh = fixture_mesh()
    path = tmp_path / ("regions" + suffix)
    meshioplusplus.write(path, mesh, file_format=fmt)
    back = meshioplusplus.read(path)

    cells = {r.name: r for r in back.regions if r.kind == "cell"}
    assert list(cells) == [renamed], why
    assert_array_equal(cells[renamed].entries, [0, 1], err_msg=why)
    assert [r for r in back.regions if r.kind in ("point", "side")] == [], why

    # ... and the rewritten name is a fixed point from there on.
    again = tmp_path / ("again" + suffix)
    meshioplusplus.write(again, back, file_format=fmt)
    assert [r.name for r in meshioplusplus.read(again).regions] == [renamed], why


@pytest.mark.parametrize("fmt", sorted(NAMESPACED_REGIONS))
def test_namespaced_region_formats_are_not_round_trip_rows(fmt):
    """Pins which bucket this format belongs to, as the others do."""
    assert fmt not in {
        p.values[0] for p in MATRIX
    }, f"{fmt} now preserves region names: move it into MATRIX"
    assert fmt not in PHASE_2, f"{fmt} carries regions: it is no longer a Phase-2 gap"
    assert fmt not in READ_ONLY_REGIONS, f"{fmt} writes regions too"


@pytest.mark.parametrize("fmt", sorted(READ_ONLY_REGIONS))
def test_read_only_region_formats_are_not_round_trip_rows(fmt):
    """Pins the asymmetry rather than letting it be mistaken for full support.

    When such a format's writer learns to emit regions, move it into MATRIX and
    delete its entry here.
    """
    assert fmt not in {
        p.values[0] for p in MATRIX
    }, f"{fmt} now round-trips regions: move it from READ_ONLY_REGIONS into MATRIX"
    assert fmt not in PHASE_2, f"{fmt} is read-capable: it is no longer a Phase-2 gap"
