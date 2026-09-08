"""Tests for temporal windows and rollout (pure numpy, default CI matrix).

The oracles are chosen so a plausible bug fails them:

* the window layout is checked against a *hand-built* concatenation, because
  reversing it produces an array of the same shape;
* the rollout drift is checked with two injected models whose behaviour is
  known exactly -- a perfect one (zero error at every step) and persistence
  (error growing 1, 2, 3, ...), the second of which distinguishes a genuine
  autoregressive rollout from teacher forcing, which would report a constant
  error instead;
* the streaming invariant is checked by counting reads, not by inspecting
  memory.
"""

import numpy as np
import pytest

import meshioplusplus
import meshioplusplus.physicsnemo as mpn
from meshioplusplus import DatasetManifest
from meshioplusplus._mesh import Mesh


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _step_mesh(value):
    """Two triangles whose field is exactly the step index."""
    return Mesh(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2], [0, 2, 3]]))],
        point_data={"T": np.full(4, float(value))},
    )


class _Series:
    """A TimeSeries-shaped stand-in that counts its reads."""

    def __init__(self, steps):
        self._steps = steps
        self.reads = 0

    def __len__(self):
        return self._steps

    def __getitem__(self, index):
        self.reads += 1
        return float(index), _step_mesh(index)


def _series_manifest(tmp_path, entries=2, steps=5):
    manifest = DatasetManifest(base_dir=str(tmp_path))
    for e in range(entries):
        case = tmp_path / f"case_{e}"
        case.mkdir(exist_ok=True)
        for s in range(steps):
            meshioplusplus.write(str(case / f"out_{s}.vtu"), _step_mesh(s))
        manifest.add(f"case_{e}/out_*.vtu", id=f"c{e}", split="train")
    return manifest


# --------------------------------------------------------------------------- #
# the window layout                                                           #
# --------------------------------------------------------------------------- #
def test_a_window_lays_its_states_oldest_first():
    states = [np.full((3, 2), k, dtype=float) for k in range(3)]
    window = mpn.make_window(states)
    assert window.shape == (3, 6)
    # Hand-built: the LAST two columns are the most recent state. Reversing the
    # order gives an array of the same shape and different meaning.
    assert np.array_equal(window, np.concatenate(states, axis=1))
    assert np.all(window[:, -2:] == 2.0)
    assert np.all(window[:, :2] == 0.0)


def test_make_window_refuses_mismatched_states():
    with pytest.raises(ValueError, match="same"):
        mpn.make_window([np.zeros((3, 2)), np.zeros((4, 2))])
    with pytest.raises(ValueError, match="at least one"):
        mpn.make_window([])


# --------------------------------------------------------------------------- #
# the three schemes                                                           #
# --------------------------------------------------------------------------- #
def test_single_step_pairs_each_window_with_the_state_that_followed():
    series = _Series(5)
    samples = list(mpn.window_samples(series, history_size=2, fields=["T"]))
    assert len(samples) == 5 - 2
    for offset, (time, sample) in enumerate(samples):
        step = offset + 2
        assert time == float(step)
        # The field IS the step index, so the window's contents are checkable.
        assert np.allclose(sample.arrays["x"][:, -1], step - 1)
        assert np.allclose(sample.arrays["x"][:, 0], step - 2)
        assert np.allclose(sample.arrays["y"], step)
    assert samples[0][1].x_columns == ("T@-1", "T@-0")
    assert samples[0][1].y_columns == ("T",)


def test_one_shot_pairs_the_first_window_with_the_last_state():
    series = _Series(6)
    samples = list(
        mpn.window_samples(series, scheme="one_shot", history_size=2, fields=["T"])
    )
    assert len(samples) == 1
    _, sample = samples[0]
    assert np.allclose(sample.arrays["x"][:, 0], 0.0)
    assert np.allclose(sample.arrays["x"][:, -1], 1.0)
    assert np.allclose(sample.arrays["y"], 5.0)


def test_time_conditional_appends_a_normalized_time_channel():
    series = _Series(5)
    samples = list(
        mpn.window_samples(
            series, scheme="time_conditional", history_size=2, fields=["T"]
        )
    )
    assert len(samples) == 3
    for offset, (_, sample) in enumerate(samples):
        step = offset + 2
        assert sample.x_columns[-1] == "time"
        assert np.allclose(sample.arrays["x"][:, -1], step / 4.0)
        # The window itself is the INITIAL one at every step -- the model reads
        # the trajectory as a function of time, not of its own last output.
        assert np.allclose(sample.arrays["x"][:, 0], 0.0)
        assert np.allclose(sample.arrays["y"], step)


def test_a_series_too_short_for_a_window_yields_nothing():
    assert list(mpn.window_samples(_Series(2), history_size=2, fields=["T"])) == []


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(scheme="rollout"), "unknown scheme"),
        (dict(history_size=1), "at least 2"),
    ],
)
def test_bad_window_requests_are_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        list(mpn.window_samples(_Series(5), fields=["T"], **kwargs))


def test_each_step_is_read_exactly_once():
    series = _Series(6)
    list(mpn.window_samples(series, history_size=3, fields=["T"]))
    assert series.reads == 6, "the streaming invariant reads each step once"


def test_windows_walk_a_manifest_and_skip_short_entries(tmp_path):
    manifest = _series_manifest(tmp_path, entries=2, steps=5)
    samples = list(mpn.iter_windows(manifest, fields=["T"], history_size=2))
    assert len(samples) == 2 * (5 - 2)
    assert {entry_id for entry_id, _, _ in samples} == {"c0", "c1"}

    short = tmp_path / "short"
    short.mkdir()
    meshioplusplus.write(str(short / "out_0.vtu"), _step_mesh(0))
    manifest.add("short/out_*.vtu", id="tiny", split="train")
    with pytest.warns(UserWarning, match="too few for a window"):
        again = list(mpn.iter_windows(manifest, fields=["T"], history_size=2))
    assert len(again) == len(samples), "the short entry contributed nothing"


def test_window_stats_reports_per_column_moments(tmp_path):
    manifest = _series_manifest(tmp_path, entries=1, steps=5)
    stats = mpn.window_stats(manifest, fields=["T"], history_size=2)
    # Windows are (T@-1, T@-0) over steps (0,1), (1,2), (2,3): means 1 and 2.
    assert stats["x_mean"] == pytest.approx([1.0, 2.0])
    assert stats["y_mean"] == pytest.approx([3.0])


# --------------------------------------------------------------------------- #
# rollout                                                                     #
# --------------------------------------------------------------------------- #
def _trajectory(steps=6, rows=4, width=1):
    return np.stack([np.full((rows, width), float(k)) for k in range(steps)])


def test_a_perfect_model_rolls_out_without_drift():
    # The state advances by exactly 1, and so does this model.
    states = _trajectory()
    result = mpn.rollout(lambda w: w[:, -1:] + 1.0, states, history_size=2)
    assert result.predictions.shape == (4, 4, 1)
    assert np.allclose(result.errors, 0.0)


def test_persistence_drifts_and_the_drift_compounds():
    # Persistence predicts "the same as last time". Fed its OWN output, its
    # error grows 1, 2, 3, 4; under teacher forcing it would stay at 1, which
    # is what distinguishes a real rollout from a one-step evaluation.
    states = _trajectory()
    result = mpn.rollout(lambda w: w[:, -1:], states, history_size=2)
    assert result.errors == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_rollout_walks_a_real_series_one_step_at_a_time():
    series = _Series(6)
    result = mpn.rollout(
        lambda w: w[:, -1:] + 1.0, series, history_size=2, fields=["T"]
    )
    assert np.allclose(result.errors, 0.0)
    assert result.times == (2.0, 3.0, 4.0, 5.0)
    assert series.reads == 6


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(history_size=1), "at least 2"),
        (dict(history_size=9), "too few"),
    ],
)
def test_bad_rollout_requests_are_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mpn.rollout(lambda w: w[:, -1:], _trajectory(), **kwargs)


def test_a_wrongly_shaped_prediction_is_refused_by_name():
    with pytest.raises(ValueError, match="returned"):
        mpn.rollout(lambda w: w, _trajectory(), history_size=2)


def test_the_public_api_is_exported():
    for name in (
        "make_window",
        "window_samples",
        "iter_windows",
        "window_stats",
        "rollout",
        "Rollout",
        "WindowSample",
    ):
        assert name in mpn.__all__ and hasattr(mpn, name)
