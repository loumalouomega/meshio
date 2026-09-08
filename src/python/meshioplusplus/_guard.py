"""Geometry guardrails (``meshioplusplus.GeometryGuard``).

A trained surrogate will answer any mesh you give it. It has no way to say
"this part is nothing like what I was trained on", and the answer it returns in
that case is finite, plausible and wrong -- which is worse than an error,
because nothing downstream flags it. A guardrail is the missing check: fit a
description of the *shapes* a model was trained on, and score a new one against
it before trusting the prediction.

The library is unusually well placed for this. :func:`compute_stats`,
:func:`compute_quality` and :func:`extract_surface` already produce every
descriptor, ``dataset_health`` already walks a manifest one mesh at a time, and
fitting a density over those summaries is a small amount of code on top. That
is all this module is.

The descriptors are NOT invariant, deliberately
-----------------------------------------------
The usual instinct with shape descriptors is to normalize away position, scale
and orientation. Here that is exactly backwards: **a scaled part is a different
part**. A model trained on brackets 10 cm across has learnt physics at that
scale, and handed one 3 m across it will interpolate confidently into a regime
it never saw. So extents, centroid, area and volume enter raw, and a guard
fitted on centimetre parts flags a metre one -- which is the whole point.

The score
---------
A diagonal Gaussian: each descriptor is standardized against the training
mean and standard deviation, and the score is the root mean square of those
z-values. It is deliberately the simplest thing that works -- a full
covariance over ~16 descriptors needs far more cases than a training set
usually has, and a mixture model needs a choice of components nobody can
justify from the data. The threshold is ``margin`` times the **worst training
score**, so "out" means "further from the training set than anything in it
was, by a stated factor" rather than a probability nobody calibrated.

What comes back names the worst descriptors, so a flag is actionable: "this
part is 40x wider than anything I trained on" is a different problem from
"this mesh has inverted cells".

Public API:

* :func:`geometry_descriptors` -- one mesh -> its raw descriptors.
* :class:`GeometryGuard` -- ``fit`` / ``score`` / ``check``, JSON round-trip.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._common import warn

__all__ = [
    "GEOMETRY_GUARD_VERSION",
    "GUARD_DESCRIPTORS",
    "geometry_descriptors",
    "GeometryGuard",
]

#: Bumped when the descriptor set or the scoring rule changes meaning.
GEOMETRY_GUARD_VERSION = 1

#: The descriptors, in a fixed order. Every one is **non-invariant** on
#: purpose (see the module docstring): nothing here is normalized by the
#: bounding box, and that is what lets a guard notice a rescaled part.
GUARD_DESCRIPTORS = (
    "extent_x",
    "extent_y",
    "extent_z",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "log10_num_points",
    "log10_num_cells",
    "total_area",
    "unsigned_volume",
    "num_inverted",
    "surface_area",
    "log10_num_surface_cells",
    "scaled_jacobian_min",
    "scaled_jacobian_mean",
    "aspect_ratio_mean",
)

#: A standard deviation is floored relative to its own mean, so a descriptor
#: that never varied across the training set flags ANY change in it rather
#: than dividing by ~0 and reporting infinity.
_STD_FLOOR_RELATIVE = 1e-6
_STD_FLOOR_ABSOLUTE = 1e-12

#: Below this many training meshes the threshold rests on too few samples to
#: mean much; the fit still happens, with a warning.
_MIN_SAMPLES = 3


def _log10_count(value):
    """Counts span orders of magnitude, so they enter as logarithms -- a mesh
    with twice the cells is a small change, one with a thousand times is not."""
    return math.log10(float(value)) if value > 0 else 0.0


def _quality_summary(report, metric, key):
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    entry = metrics.get(f"quality:{metric}")
    if not entry:
        return float("nan")
    value = entry.get(key)
    return float("nan") if value is None else float(value)


def geometry_descriptors(mesh, *, quality: bool = True) -> dict:
    """One mesh -> its raw shape descriptors, as a plain ``{name: float}``.

    A descriptor that does not apply to this mesh is ``NaN`` (never zero, and
    never silently omitted), so a guard can exclude it from the score and name
    it as missing rather than scoring a fabricated value.
    """
    from ._quality import compute_quality
    from ._stats import compute_stats
    from ._surface import extract_surface

    stats = compute_stats(mesh)
    extent = list(stats["extent"])
    centroid = list(stats["centroid"])
    out = {
        "extent_x": float(extent[0]),
        "extent_y": float(extent[1]),
        "extent_z": float(extent[2]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
        "log10_num_points": _log10_count(stats["num_points"]),
        "log10_num_cells": _log10_count(stats["num_cells"]),
        "total_area": float(stats["total_area"]),
        "unsigned_volume": float(stats["unsigned_volume"]),
        "num_inverted": float(stats["num_inverted"]),
    }

    # The outer surface: its area and facet count describe how convoluted the
    # shape is in a way the bounding box cannot.
    surface_area = float("nan")
    surface_cells = float("nan")
    try:
        surface = extract_surface(mesh)
        surface_stats = compute_stats(surface)
        surface_area = float(surface_stats["total_area"])
        surface_cells = _log10_count(surface_stats["num_cells"])
    except Exception:
        # A mesh with no skinnable cells (a point cloud, a curve) has no
        # surface to describe; that is a missing descriptor, not a failure.
        pass
    out["surface_area"] = surface_area
    out["log10_num_surface_cells"] = surface_cells

    if quality:
        try:
            report = compute_quality(mesh)
        except Exception:
            report = {}
        out["scaled_jacobian_min"] = _quality_summary(report, "scaled_jacobian", "min")
        out["scaled_jacobian_mean"] = _quality_summary(
            report, "scaled_jacobian", "mean"
        )
        out["aspect_ratio_mean"] = _quality_summary(report, "aspect_ratio", "mean")
    else:
        for name in (
            "scaled_jacobian_min",
            "scaled_jacobian_mean",
            "aspect_ratio_mean",
        ):
            out[name] = float("nan")
    return out


@dataclass(frozen=True)
class GeometryGuard:
    """A description of the shapes a model was trained on.

    ``mean``/``std`` are per-descriptor over the training set, ``threshold``
    the score above which a mesh is out of distribution, and ``schema``
    records how the fit was made (including every training score, so a caller
    can re-threshold without re-fitting).
    """

    names: tuple
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    schema: dict = field(default_factory=dict)

    def __post_init__(self):
        names = tuple(str(n) for n in self.names)
        mean = np.ascontiguousarray(np.asarray(self.mean, dtype=np.float64))
        std = np.ascontiguousarray(np.asarray(self.std, dtype=np.float64))
        if mean.shape != (len(names),) or std.shape != (len(names),):
            raise ValueError(
                f"meshio++: GeometryGuard: mean and std must each have "
                f"{len(names)} entries, got {mean.shape} and {std.shape}"
            )
        mean.flags.writeable = False
        std.flags.writeable = False
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "schema", dict(self.schema))

    # ----------------------------------------------------------------- fit --
    @classmethod
    def fit(
        cls,
        manifest,
        *,
        split: Optional[str] = "train",
        margin: float = 1.5,
        quality: bool = True,
        step: int = 0,
        **read_kwargs,
    ) -> "GeometryGuard":
        """Fit over a manifest's meshes, one alive at a time.

        An entry that cannot be described is skipped with a warning naming it,
        never silently: a guard fitted on half a dataset would be tighter than
        the caller believes.
        """
        from ._dataset import DatasetManifest

        prefix = "meshio++: GeometryGuard.fit: "
        if not isinstance(manifest, DatasetManifest):
            manifest = DatasetManifest.load(manifest)
        rows = []
        skipped = []
        for entry in manifest.entries(split=split):
            try:
                series = entry.time_series(**read_kwargs)
                _, mesh = series[step]
                rows.append(geometry_descriptors(mesh, quality=quality))
                del mesh
            except Exception as exc:  # noqa: BLE001 -- reported, never fatal
                skipped.append(f"{entry.id} ({type(exc).__name__})")
        if skipped:
            warn(
                "GeometryGuard.fit: could not describe "
                f"{len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'}: "
                + ", ".join(skipped)
            )
        if not rows:
            raise ValueError(
                f"{prefix}no entries could be described"
                + (f" in split '{split}'" if split else "")
            )
        if len(rows) < _MIN_SAMPLES:
            warn(
                f"GeometryGuard.fit: fitted on {len(rows)} mesh(es); a threshold "
                f"from fewer than {_MIN_SAMPLES} is tight by construction -- "
                "raise margin, or fit on more cases"
            )
        return cls.from_descriptors(rows, margin=margin)

    @classmethod
    def from_descriptors(cls, rows, *, margin: float = 1.5) -> "GeometryGuard":
        """Fit from already-computed descriptor dicts."""
        prefix = "meshio++: GeometryGuard: "
        rows = [dict(r) for r in rows]
        if not rows:
            raise ValueError(f"{prefix}needs at least one set of descriptors")
        if not margin > 0:
            raise ValueError(f"{prefix}margin must be positive, got {margin}")
        names = tuple(n for n in GUARD_DESCRIPTORS if n in rows[0])
        extra = tuple(sorted(set(rows[0]) - set(GUARD_DESCRIPTORS)))
        names = names + extra
        table = np.array(
            [[float(r.get(n, float("nan"))) for n in names] for r in rows],
            dtype=np.float64,
        )
        finite = np.isfinite(table)
        with np.errstate(invalid="ignore"):
            mean = np.where(
                finite.any(axis=0),
                np.nansum(np.where(finite, table, 0.0), axis=0)
                / np.maximum(finite.sum(axis=0), 1),
                np.nan,
            )
            centred = np.where(finite, table - mean, 0.0)
            variance = (centred**2).sum(axis=0) / np.maximum(finite.sum(axis=0), 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.maximum(
            std, np.maximum(np.abs(mean) * _STD_FLOOR_RELATIVE, _STD_FLOOR_ABSOLUTE)
        )
        guard = cls(names, mean, std, 0.0, {})
        scores = [guard.score(row) for row in rows]
        worst = max(scores) if scores else 0.0
        schema = {
            "version": GEOMETRY_GUARD_VERSION,
            "num_samples": len(rows),
            "margin": float(margin),
            "training_scores": [float(s) for s in scores],
            "descriptors": list(names),
        }
        return cls(names, mean, std, float(margin) * float(worst), schema)

    # --------------------------------------------------------------- score --
    def _row(self, mesh_or_descriptors):
        if isinstance(mesh_or_descriptors, dict):
            return dict(mesh_or_descriptors)
        return geometry_descriptors(mesh_or_descriptors)

    def _z(self, row):
        values = np.array(
            [float(row.get(n, float("nan"))) for n in self.names], dtype=np.float64
        )
        usable = np.isfinite(values) & np.isfinite(self.mean)
        z = np.zeros_like(values)
        z[usable] = (values[usable] - self.mean[usable]) / self.std[usable]
        return values, z, usable

    def score(self, mesh_or_descriptors) -> float:
        """The root-mean-square z-value over the descriptors both sides have."""
        _, z, usable = self._z(self._row(mesh_or_descriptors))
        if not usable.any():
            return float("nan")
        return float(np.sqrt(np.mean(np.square(z[usable]))))

    def check(self, mesh_or_descriptors, *, top: int = 3) -> dict:
        """Score a mesh and name the descriptors that put it there.

        Advisory by design: this reports, it never refuses. ``verdict`` is
        ``"in"`` or ``"out"``; ``worst`` names the ``top`` descriptors by
        absolute z-value, and ``missing`` those neither side could compare.
        """
        row = self._row(mesh_or_descriptors)
        values, z, usable = self._z(row)
        score = self.score(row)
        order = np.argsort(-np.abs(np.where(usable, z, 0.0)))
        worst = [
            {
                "name": self.names[i],
                "value": float(values[i]),
                "mean": float(self.mean[i]),
                "std": float(self.std[i]),
                "z": float(z[i]),
            }
            for i in order[: max(int(top), 0)]
            if usable[i]
        ]
        return {
            "score": score,
            "threshold": self.threshold,
            "verdict": "out" if score > self.threshold else "in",
            "worst": worst,
            "missing": [n for n, ok in zip(self.names, usable) if not ok],
        }

    # ---------------------------------------------------------------- I/O --
    def to_dict(self) -> dict:
        doc = {
            "version": GEOMETRY_GUARD_VERSION,
            "descriptors": list(self.names),
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "threshold": self.threshold,
        }
        for key, value in self.schema.items():
            doc.setdefault(key, value)
        return doc

    @classmethod
    def from_dict(cls, doc) -> "GeometryGuard":
        if isinstance(doc, GeometryGuard):
            return doc
        try:
            names = tuple(doc["descriptors"])
            mean = doc["mean"]
            std = doc["std"]
            threshold = doc["threshold"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "meshio++: GeometryGuard.from_dict: missing key "
                f"{exc.args[0]!r} (descriptors, mean, std and threshold are "
                "required)"
            ) from None
        schema = {
            k: v
            for k, v in doc.items()
            if k not in ("descriptors", "mean", "std", "threshold")
        }
        return cls(names, mean, std, threshold, schema)

    def save(self, path) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
        return os.fspath(path)

    @classmethod
    def load(cls, path) -> "GeometryGuard":
        """Read a guard from a JSON file, a dict, or a model card holding one."""
        if isinstance(path, (dict, GeometryGuard)):
            return cls.from_dict(path)
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        # A model card carries its guard under "guard"; accept either.
        if "descriptors" not in doc and isinstance(doc.get("guard"), dict):
            doc = doc["guard"]
        return cls.from_dict(doc)
