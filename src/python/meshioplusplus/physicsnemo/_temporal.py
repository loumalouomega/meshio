"""Temporal windows and rollout over a transient series (pure, no torch).

A steady-state surrogate maps one mesh to one answer. A *transient* one maps a
short history to the next state, and the shape it wants is a **window**: the
last K states of every node, laid side by side along the channel axis. This
module builds those windows over a :class:`~meshioplusplus.TimeSeries` or a
:class:`~meshioplusplus.DatasetManifest`, and evaluates a trained model's
**rollout** -- feeding its own prediction back in and measuring how the error
grows, which is the number that says whether a transient surrogate is usable
for more than one step.

Oldest first
------------
A window of K states is ``(N, K*W)``: node row, then the K states concatenated
**oldest first**, so the last ``W`` columns are always the most recent state.
That is the upstream convention and it is shared by every function here --
:func:`make_window`, the three schemes, and :func:`rollout`'s own history --
because a model trained on one ordering and fed the other produces plausible,
wrong numbers with nothing to flag them. The recorded ``x_columns`` name the
age explicitly (``T@-2``, ``T@-1``, ``T@-0``), so a stored contract catches the
disagreement rather than leaving it to a convention nobody re-reads.

Three schemes
-------------
``"single_step"`` -- every window of K states paired with the state that
followed it. The autoregressive shape: ``T - K`` samples per series.

``"one_shot"`` -- the *first* window paired with the *last* state. One sample
per series, for a model that jumps to the end rather than stepping there.

``"time_conditional"`` -- the first window plus a normalized time channel,
paired with the state at that time. ``T - K`` samples, and the model learns
the trajectory as a function of time rather than of its own previous output,
so its error does not compound.

Streaming
---------
The sequence invariant holds: :func:`window_samples` keeps a ``deque`` of at
most K **feature matrices** and one mesh alive, never K meshes, and reads each
step exactly once. :func:`rollout` reads the truth one step at a time as it
goes.

Public API:

* :func:`make_window` -- K states -> one ``(N, K*W)`` window.
* :class:`WindowSample` / :func:`window_samples` / :func:`iter_windows`.
* :func:`window_stats` -- per-column normalization moments.
* :class:`Rollout` / :func:`rollout` -- autoregressive drift.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..__about__ import __version__
from .._ml import FEATURE_SCHEMA_VERSION, feature_matrix

__all__ = [
    "TEMPORAL_WINDOW_VERSION",
    "WindowSample",
    "make_window",
    "window_samples",
    "iter_windows",
    "window_stats",
    "Rollout",
    "rollout",
]

#: Bumped when the recorded window schema changes meaning.
TEMPORAL_WINDOW_VERSION = 1

_SCHEMES = ("single_step", "time_conditional", "one_shot")


@dataclass(frozen=True)
class WindowSample:
    """One window's worth of arrays, plus the recorded column contract.

    ``arrays`` holds ``x`` (the window, ``(N, K*W)`` -- plus one normalized
    time column under ``"time_conditional"``), ``y`` (the target state,
    ``(N, W)``) and ``pos``. ``x_columns`` name each column's field **and its
    age**, oldest first.
    """

    arrays: dict
    x_columns: tuple
    y_columns: tuple
    schema: dict = field(default_factory=dict)


def make_window(states):
    """K state matrices -> one ``(N, K*W)`` window, **oldest first**.

    The single owner of the layout: every scheme here and :func:`rollout`'s own
    history build their windows through it, so they cannot disagree about which
    end is recent.
    """
    states = [np.asarray(s, dtype=np.float64) for s in states]
    if not states:
        raise ValueError("meshio++: make_window: needs at least one state")
    rows = states[0].shape[0]
    for s in states:
        if s.ndim != 2 or s.shape[0] != rows:
            raise ValueError(
                "meshio++: make_window: every state must be (N, W) with the same "
                f"N; got {[tuple(s.shape) for s in states]}"
            )
    return np.ascontiguousarray(np.concatenate(states, axis=1))


def _states_of(mesh, fields, location, regions):
    fm = feature_matrix(mesh, location, fields=fields, coords=False, regions=regions)
    return fm.matrix.astype(np.float64, copy=False), fm.columns


def _positions(mesh, location):
    from .._proximity import _graph_positions

    return _graph_positions(mesh, "node" if location == "point" else "cell")


def _aged_columns(columns, history):
    """``T@-2``, ``T@-1``, ``T@-0`` -- the age is part of the contract."""
    return tuple(
        f"{name}@-{history - 1 - k}" for k in range(history) for name in columns
    )


def window_samples(
    series,
    *,
    scheme="single_step",
    history_size=2,
    fields=None,
    location="point",
    regions=False,
    float32=True,
):
    """Yield ``(time, WindowSample)`` over ONE series.

    ``series`` is a :class:`~meshioplusplus.TimeSeries` (or anything indexable
    yielding ``(time, mesh)``). At most ``history_size`` feature matrices and
    one mesh are alive at a time, and each step is read exactly once.
    """
    prefix = "meshio++: window_samples: "
    if scheme not in _SCHEMES:
        raise ValueError(
            f"{prefix}unknown scheme {scheme!r} (expected one of "
            f"{', '.join(_SCHEMES)})"
        )
    history = int(history_size)
    if history < 2:
        raise ValueError(
            f"{prefix}history_size must be at least 2 -- a window of one state "
            "carries no history at all"
        )
    total = len(series)
    if total <= history:
        return
    float_dtype = np.float32 if float32 else np.float64

    history_states = deque(maxlen=history)
    initial = None
    columns = None
    pos = None
    for step in range(total):
        time, mesh = series[step]
        state, cols = _states_of(mesh, fields, location, regions)
        if columns is None:
            columns, pos = cols, _positions(mesh, location)
        elif cols != columns:
            raise ValueError(
                f"{prefix}the columns change across the series ({columns} then "
                f"{cols}) -- every step must carry the same fields"
            )
        elif state.shape[0] != pos.shape[0]:
            raise ValueError(
                f"{prefix}step {step} has {state.shape[0]} rows but the first "
                f"has {pos.shape[0]} -- a remeshed series cannot be windowed"
            )
        del mesh

        if len(history_states) < history:
            history_states.append(state)
            if len(history_states) == history:
                initial = make_window(history_states)
            continue

        # `step` is now the first state AFTER a full window.
        if scheme == "single_step":
            x = make_window(history_states)
        elif scheme == "time_conditional":
            fraction = step / float(total - 1)
            x = np.concatenate(
                (initial, np.full((initial.shape[0], 1), fraction)), axis=1
            )
        else:  # one_shot: the first window, paired with the last state only
            if step != total - 1:
                history_states.append(state)
                continue
            x = initial

        x_columns = _aged_columns(columns, history)
        if scheme == "time_conditional":
            x_columns = x_columns + ("time",)
        schema = {
            "temporal_window_version": TEMPORAL_WINDOW_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "meshioplusplus_version": str(__version__),
            "scheme": scheme,
            "history_size": history,
            "location": location,
            "step": int(step),
            "num_steps": int(total),
            "float32": bool(float32),
            "x_columns": list(x_columns),
            "y_columns": list(columns),
        }
        yield time, WindowSample(
            arrays={
                "x": x.astype(float_dtype, copy=False),
                "y": state.astype(float_dtype, copy=False),
                "pos": pos.astype(float_dtype, copy=False),
            },
            x_columns=x_columns,
            y_columns=tuple(columns),
            schema=schema,
        )
        history_states.append(state)


def iter_windows(manifest, *, split=None, **kwargs):
    """Yield ``(entry_id, time, WindowSample)`` over a manifest's entries.

    An entry with too few steps to fill a window contributes nothing, with one
    warning naming it -- emitted once, when the entry is reached, never per
    epoch. ``kwargs`` split into :func:`window_samples` parameters and
    ``read()`` kwargs.
    """
    from . import _as_manifest

    window_keys = (
        "scheme",
        "history_size",
        "fields",
        "location",
        "regions",
        "float32",
    )
    window_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in window_keys}
    history = int(window_kwargs.get("history_size", 2))
    for entry in _as_manifest(manifest).entries(split=split):
        series = entry.time_series(**kwargs)
        if len(series) <= history:
            warnings.warn(
                f"meshio++: iter_windows: entry '{entry.id}' has {len(series)} "
                f"step(s), too few for a window of {history} plus a target -- "
                "it contributes no samples",
                stacklevel=2,
            )
            continue
        for time, sample in window_samples(series, **window_kwargs):
            yield entry.id, time, sample


def window_stats(manifest, *, split=None, **kwargs):
    """Per-column mean/std over a manifest's windows, streaming.

    ``{"x_mean": [...], "x_std": [...], "y_mean": [...], "y_std": [...]}``.
    A window's field columns are lagged copies of one another, so
    :func:`~meshioplusplus.physicsnemo.field_stats` already covers the usual
    case; this exists for the ``"time_conditional"`` channel, whose statistics
    have no field to borrow from.
    """
    from . import _Moments

    x_acc = y_acc = None
    for _, _, sample in iter_windows(manifest, split=split, **kwargs):
        x = np.asarray(sample.arrays["x"], dtype=np.float64)
        y = np.asarray(sample.arrays["y"], dtype=np.float64)
        if x_acc is None:
            x_acc, y_acc = _Moments(x.shape[1]), _Moments(y.shape[1])
        x_acc.add(x)
        y_acc.add(y)
    if x_acc is None:
        return {"x_mean": [], "x_std": [], "y_mean": [], "y_std": []}
    return {
        "x_mean": x_acc.mean().tolist(),
        "x_std": x_acc.std().tolist(),
        "y_mean": y_acc.mean().tolist(),
        "y_std": y_acc.std().tolist(),
    }


@dataclass(frozen=True)
class Rollout:
    """An autoregressive rollout and the drift it accumulated.

    ``predictions`` is ``(T - K, N, W)`` and ``errors`` the per-step RMSE
    against the truth, so ``errors`` rising is the model's own error
    compounding -- the thing a one-step validation loss cannot show.
    """

    predictions: np.ndarray
    errors: np.ndarray
    times: tuple
    schema: dict = field(default_factory=dict)


def rollout(
    predict_fn,
    series_or_states,
    *,
    history_size=2,
    fields=None,
    location="point",
    regions=False,
    **read_kwargs,
):
    """Feed a model its own predictions and measure the drift.

    ``predict_fn`` takes one ``(N, K*W)`` float64 window and returns the next
    ``(N, W)`` state. Injecting it keeps this module torch-free: the caller
    owns the framework, normalization and device, and this owns the loop.

    Seeded with the first ``history_size`` **true** states, then every later
    step is predicted from the model's own output. ``series_or_states`` is a
    :class:`~meshioplusplus.TimeSeries` or a plain ``(T, N, W)`` array.
    """
    prefix = "meshio++: rollout: "
    history = int(history_size)
    if history < 2:
        raise ValueError(f"{prefix}history_size must be at least 2")

    array = None
    if not hasattr(series_or_states, "__len__") or isinstance(
        series_or_states, np.ndarray
    ):
        array = np.asarray(series_or_states, dtype=np.float64)
        if array.ndim != 3:
            raise ValueError(
                f"{prefix}a plain trajectory must be (T, N, W), got shape "
                f"{array.shape}"
            )
        total = array.shape[0]
    else:
        total = len(series_or_states)
    if total <= history:
        raise ValueError(
            f"{prefix}the series has {total} step(s), too few for a history of "
            f"{history} plus a target"
        )

    def truth_at(step):
        if array is not None:
            return None, array[step]
        time, mesh = series_or_states[step]
        state, _ = _states_of(mesh, fields, location, regions)
        return time, state

    window = deque(maxlen=history)
    times = []
    for step in range(history):
        _, state = truth_at(step)
        window.append(state)

    predictions = []
    errors = []
    rows = window[0].shape[0]
    for step in range(history, total):
        time, truth = truth_at(step)
        if truth.shape[0] != rows:
            raise ValueError(
                f"{prefix}step {step} has {truth.shape[0]} rows but the first "
                f"has {rows} -- a remeshed series cannot be rolled out"
            )
        pred = np.asarray(predict_fn(make_window(window)), dtype=np.float64)
        if pred.shape != truth.shape:
            raise ValueError(
                f"{prefix}predict_fn returned {pred.shape} at step {step} but "
                f"the state is {truth.shape}"
            )
        predictions.append(pred)
        errors.append(float(np.sqrt(np.mean(np.square(pred - truth)))))
        times.append(time)
        # The prediction, not the truth: that is what makes this a rollout.
        window.append(pred)

    schema = {
        "temporal_window_version": TEMPORAL_WINDOW_VERSION,
        "meshioplusplus_version": str(__version__),
        "history_size": history,
        "num_steps": int(total),
        "location": location,
    }
    return Rollout(
        predictions=np.asarray(predictions),
        errors=np.asarray(errors, dtype=np.float64),
        times=tuple(times),
        schema=schema,
    )
