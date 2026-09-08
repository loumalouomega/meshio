"""Tests for geometry guardrails (pure numpy, default CI matrix).

The load-bearing oracle is **non-invariance**. The usual instinct with shape
descriptors is to normalize away position and scale; here that is exactly
backwards, because a scaled part is a different part and a model trained at
one scale will interpolate confidently into a regime it never saw. Every test
that scales or translates a mesh and expects the descriptors to *change* fails
against a bounding-box-normalized implementation.
"""

import random

import numpy as np
import pytest

import meshioplusplus
import meshioplusplus as mio
from meshioplusplus import DatasetManifest, GeometryGuard, geometry_descriptors


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _manifest(tmp_path, count=6, seed=0, low=0.9, high=1.1):
    """A family of boxes whose spacing varies a little -- the training set."""
    rng = random.Random(seed)
    manifest = DatasetManifest(base_dir=str(tmp_path))
    for i in range(count):
        spacing = rng.uniform(low, high)
        mesh = mio.grid([4, 4, 4], spacing=(spacing, spacing, spacing))
        meshioplusplus.write(str(tmp_path / f"case_{i}.vtu"), mesh)
        manifest.add(f"case_{i}.vtu", id=f"c{i}", split="train")
    return manifest


def _inverted_box():
    mesh = mio.grid([2, 2, 2])
    cells = np.array(mesh.cells[0].data)
    cells[0, [0, 1]] = cells[0, [1, 0]]  # flip one hexahedron
    return meshioplusplus.Mesh(mesh.points, [("hexahedron", cells)])


# --------------------------------------------------------------------------- #
# the descriptors are NOT invariant, and that is the design                   #
# --------------------------------------------------------------------------- #
def test_scaling_a_part_changes_its_descriptors():
    mesh = mio.grid([3, 3, 3])
    plain = geometry_descriptors(mesh)
    scaled = geometry_descriptors(mio.transform(mesh, scale=2.0))
    # A bounding-box-normalized descriptor set would report these equal, which
    # is precisely the failure this design exists to avoid.
    assert scaled["extent_x"] == pytest.approx(2 * plain["extent_x"])
    assert scaled["total_area"] == pytest.approx(4 * plain["total_area"])
    assert scaled["unsigned_volume"] == pytest.approx(8 * plain["unsigned_volume"])
    # The counts do not change: it is the same mesh, moved.
    assert scaled["log10_num_cells"] == plain["log10_num_cells"]


def test_translating_a_part_moves_its_centroid_and_nothing_else():
    mesh = mio.grid([3, 3, 3])
    plain = geometry_descriptors(mesh)
    moved = geometry_descriptors(mio.transform(mesh, translate=(50.0, 0.0, 0.0)))
    assert moved["centroid_x"] == pytest.approx(plain["centroid_x"] + 50.0)
    assert moved["extent_x"] == pytest.approx(plain["extent_x"])
    assert moved["unsigned_volume"] == pytest.approx(plain["unsigned_volume"])


def test_a_descriptor_that_does_not_apply_is_nan_never_zero():
    # A point cloud has no surface and no volume cells; those descriptors must
    # be missing rather than fabricated, or a guard would score a made-up 0.
    cloud = mio.subsample_points(mio.grid([3, 3, 3]), 8)
    values = geometry_descriptors(cloud)
    assert np.isnan(values["surface_area"])
    assert np.isnan(values["scaled_jacobian_min"])


def test_every_named_descriptor_is_produced():
    values = geometry_descriptors(mio.grid([2, 2, 2]))
    assert set(values) == set(meshioplusplus._guard.GUARD_DESCRIPTORS)


# --------------------------------------------------------------------------- #
# fitting and scoring                                                         #
# --------------------------------------------------------------------------- #
def test_every_training_mesh_is_in_distribution(tmp_path):
    manifest = _manifest(tmp_path)
    guard = GeometryGuard.fit(manifest)
    assert guard.schema["num_samples"] == 6
    for entry in manifest.entries():
        _, mesh = entry.time_series()[0]
        assert guard.check(mesh)["verdict"] == "in"


def test_a_much_larger_part_is_out_of_distribution(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    report = guard.check(mio.grid([4, 4, 4], spacing=(3.0, 3.0, 3.0)))
    assert report["verdict"] == "out"
    assert report["score"] > report["threshold"]
    assert report["worst"][0]["name"] in {
        "extent_x",
        "extent_y",
        "extent_z",
        "total_area",
        "unsigned_volume",
        "surface_area",
    }


def test_a_displaced_part_is_out_on_its_centroid(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    moved = mio.transform(mio.grid([4, 4, 4]), translate=(50.0, 0.0, 0.0))
    report = guard.check(moved)
    assert report["verdict"] == "out"
    # Dropping the centroid descriptors would make this mesh look ordinary.
    assert any(item["name"].startswith("centroid") for item in report["worst"])


def test_an_inverted_cell_is_out_of_distribution(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    report = guard.check(_inverted_box())
    assert report["verdict"] == "out"
    assert any(
        item["name"] in ("num_inverted", "scaled_jacobian_min")
        for item in report["worst"]
    )


def test_the_worst_descriptors_name_what_is_unusual(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    report = guard.check(mio.grid([4, 4, 4], spacing=(3.0, 3.0, 3.0)), top=2)
    assert len(report["worst"]) == 2
    for item in report["worst"]:
        # Actionable: the value, what was expected, and by how much it differs.
        assert {"name", "value", "mean", "std", "z"} <= set(item)
    assert abs(report["worst"][0]["z"]) >= abs(report["worst"][1]["z"])


def test_a_descriptor_neither_side_can_compare_is_reported_missing(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    cloud = mio.subsample_points(mio.grid([4, 4, 4]), 8)
    report = guard.check(cloud)
    assert "surface_area" in report["missing"]
    assert np.isfinite(report["score"]), "the comparable descriptors still score"


def test_an_exactly_constant_descriptor_scores_finitely(tmp_path):
    # `num_inverted` is exactly 0.0 for every well-formed training box -- not
    # nearly zero, exactly -- so its variance is exactly zero and dividing by
    # it would report an infinite score for any mesh that differs. The floor
    # keeps the score large and finite, which is what makes it comparable and
    # what lets `worst` rank the descriptors at all.
    guard = GeometryGuard.fit(_manifest(tmp_path, low=1.0, high=1.0))
    index = guard.names.index("num_inverted")
    assert guard.std[index] > 0.0, "an exactly constant descriptor kept a zero std"

    report = guard.check(_inverted_box())
    assert report["verdict"] == "out"
    assert np.isfinite(report["score"])
    assert all(np.isfinite(item["z"]) for item in report["worst"])


def test_a_tiny_training_set_warns(tmp_path, capsys):
    # `_common.warn` writes to the rich console, not the warnings module, so
    # this asserts on stderr -- the repo-wide convention for these.
    GeometryGuard.fit(_manifest(tmp_path, count=2))
    assert "tight by construction" in capsys.readouterr().err


def test_an_unreadable_entry_is_reported_and_skipped(tmp_path, capsys):
    manifest = _manifest(tmp_path)
    (tmp_path / "broken.vtu").write_text("not a mesh")
    manifest.add("broken.vtu", id="bad", split="train")
    guard = GeometryGuard.fit(manifest)
    assert "could not describe" in capsys.readouterr().err
    # The fit went on with what it could read, rather than failing outright.
    assert guard.schema["num_samples"] == 6


def test_fitting_on_nothing_is_refused_by_name(tmp_path):
    manifest = _manifest(tmp_path)
    with pytest.raises(ValueError, match="no entries could be described"):
        GeometryGuard.fit(manifest, split="valid")


def test_a_bad_margin_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="margin must be positive"):
        GeometryGuard.fit(_manifest(tmp_path), margin=0.0)


# --------------------------------------------------------------------------- #
# the document                                                                #
# --------------------------------------------------------------------------- #
def test_a_guard_round_trips_through_json(tmp_path):
    guard = GeometryGuard.fit(_manifest(tmp_path))
    probe = mio.grid([4, 4, 4], spacing=(2.0, 2.0, 2.0))
    path = tmp_path / "guard.json"
    guard.save(str(path))
    again = GeometryGuard.load(str(path))
    assert again.score(probe) == guard.score(probe)
    assert again.threshold == guard.threshold
    assert again.names == guard.names


def test_a_guard_can_be_read_out_of_a_model_card(tmp_path):
    import json

    guard = GeometryGuard.fit(_manifest(tmp_path))
    card = tmp_path / "best.mdlus.card.json"
    card.write_text(json.dumps({"version": 1, "guard": guard.to_dict()}))
    assert GeometryGuard.load(str(card)).threshold == guard.threshold


def test_a_malformed_document_is_refused_by_name():
    with pytest.raises(ValueError, match="missing key"):
        GeometryGuard.from_dict({"mean": [1.0]})


# --------------------------------------------------------------------------- #
# the surfaces                                                                #
# --------------------------------------------------------------------------- #
def test_the_health_report_flags_an_unusual_entry_without_calling_it_bad(tmp_path):
    from meshioplusplus.mcp import _health

    manifest = _manifest(tmp_path)
    guard = GeometryGuard.fit(manifest)
    odd = mio.grid([4, 4, 4], spacing=(3.0, 3.0, 3.0))
    meshioplusplus.write(str(tmp_path / "odd.vtu"), odd)
    manifest.add("odd.vtu", id="odd", split="train")

    report = _health.manifest_health(manifest, guard=guard)
    assert report["out_of_distribution"] == ["odd"]
    # An unusual shape is a fact about the dataset, not a defect in the mesh.
    assert "odd" not in report["bad_entries"]
    assert report["entries"]["odd"]["guard"]["verdict"] == "out"
    assert report["entries"]["c0"]["guard"]["verdict"] == "in"


def test_the_health_report_is_unchanged_without_a_guard(tmp_path):
    from meshioplusplus.mcp import _health

    report = _health.manifest_health(_manifest(tmp_path))
    assert report["out_of_distribution"] == []
    assert all("guard" not in scan for scan in report["entries"].values())


def test_the_tools_fit_and_check(tmp_path):
    from meshioplusplus.mcp import _tools

    manifest = _manifest(tmp_path)
    manifest_path = str(tmp_path / "m.json")
    manifest.save(manifest_path)
    guard_path = str(tmp_path / "guard.json")

    fitted = _tools.tool_guard_fit(manifest_path, guard_path)
    assert fitted["num_samples"] == 6 and fitted["threshold"] > 0

    meshioplusplus.write(
        str(tmp_path / "big.vtu"), mio.grid([4, 4, 4], spacing=(3.0, 3.0, 3.0))
    )
    report = _tools.tool_guard_check(str(tmp_path / "big.vtu"), guard_path)
    assert report["verdict"] == "out"
    # Without a guard it still describes the mesh.
    bare = _tools.tool_guard_check(str(tmp_path / "big.vtu"))
    assert "descriptors" in bare and "verdict" not in bare


def test_the_public_api_is_exported():
    for name in ("GeometryGuard", "geometry_descriptors"):
        assert name in mio.__all__ and hasattr(mio, name)
