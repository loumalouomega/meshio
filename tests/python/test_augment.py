"""Tests for dataset-level augmentation (pure numpy, default CI matrix).

The discriminating oracle is the **cell** vector: before v10.33.0
``transform(rotate_vector_data=True)`` rotated point data only, so a rotated
mesh carried an unrotated per-cell velocity — a physically impossible sample.
Every assertion here that mentions ``cell_data`` fails against that behaviour.
"""

import numpy as np
import pytest

import meshioplusplus
import meshioplusplus.physicsnemo as mpn
from meshioplusplus.physicsnemo._augment import Augmentation


def _mesh():
    mesh = meshioplusplus.grid([2, 2, 2])
    cells = len(mesh.cells[0].data)
    mesh.point_data["v"] = np.tile([1.0, 0.0, 0.0], (len(mesh.points), 1))
    mesh.cell_data["cv"] = [np.tile([1.0, 0.0, 0.0], (cells, 1))]
    mesh.cell_data["stress"] = [
        np.tile(np.diag([1.0, 2.0, 3.0]).reshape(9), (cells, 1))
    ]
    mesh.cell_data["mat"] = [np.arange(cells, dtype=np.int64)]
    return mesh


_QUARTER_TURN = {"rotate": ["z", 90.0]}


# --------------------------------------------------------------------------- #
# coherence: the point of the whole exercise                                  #
# --------------------------------------------------------------------------- #
def test_a_rotation_reaches_point_and_cell_vectors_alike():
    out, _ = Augmentation(rotation=True).apply(_mesh(), params=_QUARTER_TURN)
    # (1,0,0) under a quarter turn about z is (0,1,0), everywhere it appears.
    assert np.allclose(out.point_data["v"][0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(out.cell_data["cv"][0][0], [0.0, 1.0, 0.0], atol=1e-12)


def test_a_rotation_reaches_a_cell_tensor():
    out, _ = Augmentation(rotation=True).apply(_mesh(), params=_QUARTER_TURN)
    rotated = out.cell_data["stress"][0][0].reshape(3, 3)
    assert np.allclose(rotated, np.diag([2.0, 1.0, 3.0]), atol=1e-12)


def test_an_integer_cell_array_is_never_rotated():
    source = _mesh()
    out, _ = Augmentation(rotation=True).apply(source, params=_QUARTER_TURN)
    assert np.array_equal(out.cell_data["mat"][0], source.cell_data["mat"][0])


# --------------------------------------------------------------------------- #
# determinism                                                                 #
# --------------------------------------------------------------------------- #
def test_a_draw_is_a_pure_function_of_seed_epoch_and_index():
    aug = Augmentation(rotation=True, scale=True, translation=True, seed=7)
    assert aug.params(epoch=2, index=5) == aug.params(epoch=2, index=5)
    assert aug.params(epoch=2, index=5) != aug.params(epoch=3, index=5)
    assert aug.params(epoch=2, index=5) != aug.params(epoch=2, index=6)
    assert aug.params(epoch=2, index=5) != Augmentation(
        rotation=True, scale=True, translation=True, seed=8
    ).params(epoch=2, index=5)


def test_replaying_params_reproduces_the_transform_exactly():
    aug = Augmentation(rotation=True, scale=True, translation=True, seed=1)
    first, params = aug.apply(_mesh(), epoch=0, index=0)
    again, _ = aug.apply(_mesh(), params=params)
    assert np.array_equal(first.points, again.points)
    assert np.array_equal(first.cell_data["cv"][0], again.cell_data["cv"][0])


def test_an_inactive_augmentation_is_the_identity():
    aug = Augmentation()
    assert not aug.active
    source = _mesh()
    out, _ = aug.apply(source)
    assert out is source


# --------------------------------------------------------------------------- #
# the description                                                             #
# --------------------------------------------------------------------------- #
def test_true_means_the_documented_defaults():
    aug = Augmentation(rotation=True, scale=True, translation=True)
    assert aug.rotation == {"axis": "any", "max_degrees": 180.0}
    assert aug.scale == {"low": 0.9, "high": 1.1}
    assert aug.translation == {"max": 0.1}


def test_an_axis_rotation_stays_on_its_axis():
    aug = Augmentation(rotation={"axis": "z", "max_degrees": 30.0}, seed=3)
    out, _ = aug.apply(_mesh(), epoch=0, index=0)
    source = _mesh()
    # A rotation about z moves x and y and leaves z alone.
    assert np.allclose(out.points[:, 2], source.points[:, 2])
    assert not np.allclose(out.points[:, 0], source.points[:, 0])


def test_an_arbitrary_axis_draw_is_a_unit_vector():
    params = Augmentation(rotation={"axis": "any"}, seed=0).params()
    axis = np.asarray(params["rotate"][:3])
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_the_document_round_trips():
    aug = Augmentation(rotation={"axis": "y"}, scale=True, seed=5)
    assert Augmentation.from_dict(aug.to_dict()) == aug


def test_the_spec_block_maps_pascal_case():
    aug = Augmentation.from_spec(
        {
            "Rotation": {"Axis": "z", "MaxDegrees": 45.0},
            "Scale": {"Low": 0.5, "High": 2.0},
            "Seed": 9,
        }
    )
    assert aug.rotation == {"axis": "z", "max_degrees": 45.0}
    assert aug.scale == {"low": 0.5, "high": 2.0}
    assert aug.seed == 9
    assert Augmentation.from_spec(None) is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(rotation={"axis": "w"}), "rotation axis must be"),
        (dict(rotation={"angle": 1}), "unknown key"),
        (dict(scale={"low": 2.0, "high": 1.0}), "0 < low <= high"),
        (dict(scale={"low": -1.0, "high": 1.0}), "0 < low <= high"),
        (dict(rotation=3), "must be true, false or an object"),
    ],
)
def test_a_malformed_description_is_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Augmentation(**kwargs)


@pytest.mark.parametrize(
    "block, message",
    [
        ({"Rotate": True}, r"unknown key\(s\)"),
        ({"Rotation": {"Axe": "z"}}, "unknown key 'Axe'"),
    ],
)
def test_a_malformed_spec_block_is_refused_by_name(block, message):
    with pytest.raises(ValueError, match=message):
        Augmentation.from_spec(block)


# --------------------------------------------------------------------------- #
# pair coherence through the iteration layer                                  #
# --------------------------------------------------------------------------- #
def test_a_paired_sample_gets_ONE_draw_for_both_meshes(tmp_path):
    from meshioplusplus import DatasetManifest
    from meshioplusplus._mesh import Mesh

    def step(value):
        return Mesh(
            np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]]),
            [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
            point_data={
                "T": np.full(4, float(value)),
                "v": np.tile([1.0, 0.0, 0.0], (4, 1)),
            },
        )

    case = tmp_path / "case"
    case.mkdir()
    for s in range(3):
        meshioplusplus.write(str(case / f"out_{s}.vtu"), step(s))
    manifest = DatasetManifest(base_dir=str(tmp_path))
    manifest.add("case/out_*.vtu", id="c0", split="train")

    aug = Augmentation(rotation={"axis": "z", "max_degrees": 90.0}, seed=2)
    samples = list(
        mpn.iter_samples(
            manifest,
            fields=["T"],
            target_fields=["v"],
            target_offset=1,
            augmentation=aug,
            epoch=0,
        )
    )
    assert len(samples) == 2
    for index, (_, _, sample) in enumerate(samples):
        # y is the TARGET's rotated velocity, and it must sit at the angle the
        # INPUT's own draw used: augmenting the pair independently would put
        # the two meshes in different frames, which is a physically impossible
        # sample the model would happily learn from.
        rotated = sample.arrays["y"][0]
        assert np.linalg.norm(rotated) == pytest.approx(1.0, rel=1e-5)
        angle = np.deg2rad(aug.params(epoch=0, index=index)["rotate"][1])
        assert rotated[0] == pytest.approx(np.cos(angle), rel=1e-4)
        assert rotated[1] == pytest.approx(np.sin(angle), rel=1e-4)
        # ... and each sample draws its own pose, not the first one's.
        assert index == 0 or not np.allclose(samples[0][2].arrays["y"][0], rotated)


def test_an_epoch_changes_the_poses_reproducibly(tmp_path):
    from meshioplusplus import DatasetManifest
    from meshioplusplus._mesh import Mesh

    mesh = Mesh(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        point_data={"T": np.arange(4, dtype=float)},
    )
    meshioplusplus.write(str(tmp_path / "a.vtu"), mesh)
    manifest = DatasetManifest(base_dir=str(tmp_path))
    manifest.add("a.vtu", id="a", split="train")

    aug = Augmentation(rotation=True, seed=0)

    def positions(epoch):
        return [
            s.arrays["pos"]
            for _, _, s in mpn.iter_samples(
                manifest, fields=["T"], augmentation=aug, epoch=epoch
            )
        ]

    assert np.array_equal(positions(0)[0], positions(0)[0])
    assert not np.allclose(positions(0)[0], positions(1)[0])


def test_augmentation_is_exported():
    assert "Augmentation" in mpn.__all__ and hasattr(mpn, "Augmentation")
