"""Dataset-level augmentation: seeded, per-epoch, coherent (pure, no torch).

A surrogate trained on a hundred solves of the same part in the same pose
learns the pose. Rotating, scaling and translating each case per epoch is the
cheap answer, and the part meshio++ already had is the hard half:
:func:`~meshioplusplus.transform` with ``rotate_vector_data=True`` rotates
vector and rank-2 tensor fields **coherently with the geometry**, which is what
upstream augmentation pipelines routinely get wrong -- a rotated mesh carrying
an unrotated velocity field is a physically impossible sample the model will
happily learn from.

What was missing is only the wrapper, and it is this module: a description of
the randomness, a rule for drawing it, and the guarantee that the *same* draw
reaches a paired input and target.

Determinism
-----------
Every draw is a pure function of ``(seed, epoch, index)`` -- no global
generator, nothing carried between calls, so a run is reproducible, two
processes of a distributed job agree, and re-running epoch 3 gives epoch 3's
meshes rather than the next ones in a stream. ``random.Random`` is seeded on
the *string* ``"seed/epoch/index"``: a tuple seed is a ``TypeError`` from
Python 3.11, and a string hashes stably through ``sha512`` across versions.

Pair coherence
--------------
``target_offset``/`Target` pairing means a sample is two meshes, and augmenting
them independently would teach the model that a part rotates between one step
and the next. :func:`Augmentation.apply` therefore returns the ``params`` it
drew, and the iteration layer replays them on the target.

Public API:

* :class:`Augmentation` -- the description, ``params`` and ``apply``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .._transform import transform

__all__ = ["AUGMENTATION_VERSION", "Augmentation"]

#: Bumped when the recorded augmentation document changes meaning.
AUGMENTATION_VERSION = 1

_AXES = ("x", "y", "z", "any")

_DEFAULT_ROTATION = {"axis": "any", "max_degrees": 180.0}
_DEFAULT_SCALE = {"low": 0.9, "high": 1.1}
_DEFAULT_TRANSLATION = {"max": 0.1}


def _block(value, default, name):
    """``None`` | ``True`` | a dict -> a dict or ``None``."""
    if value is None or value is False:
        return None
    if value is True:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError(
            f"meshio++: Augmentation: {name} must be true, false or an object, "
            f"not {type(value).__name__}"
        )
    unknown = set(value) - set(default)
    if unknown:
        raise ValueError(
            f"meshio++: Augmentation: unknown key(s) {sorted(unknown)} in {name} "
            f"(known: {', '.join(sorted(default))})"
        )
    out = dict(default)
    out.update(value)
    return out


@dataclass(frozen=True)
class Augmentation:
    """Per-epoch rigid-and-scale augmentation of a whole manifest.

    ``rotation`` is ``{"axis": "x"|"y"|"z"|"any", "max_degrees": 180.0}``,
    ``scale`` is ``{"low": 0.9, "high": 1.1}`` (isotropic -- an anisotropic
    scale changes what a vector field *means*, so it is deliberately not
    offered), and ``translation`` is ``{"max": 0.1}`` per axis. Any of them may
    be ``True`` for the defaults shown, or ``None`` to leave that degree of
    freedom alone.
    """

    rotation: Optional[Dict[str, Any]] = None
    scale: Optional[Dict[str, Any]] = None
    translation: Optional[Dict[str, Any]] = None
    seed: int = 0

    def __post_init__(self):
        rotation = _block(self.rotation, _DEFAULT_ROTATION, "rotation")
        if rotation is not None and rotation["axis"] not in _AXES:
            raise ValueError(
                f"meshio++: Augmentation: rotation axis must be one of "
                f"{', '.join(_AXES)}, not {rotation['axis']!r}"
            )
        scale = _block(self.scale, _DEFAULT_SCALE, "scale")
        if scale is not None and not 0 < scale["low"] <= scale["high"]:
            raise ValueError(
                "meshio++: Augmentation: scale needs 0 < low <= high, got "
                f"{scale['low']} and {scale['high']}"
            )
        translation = _block(self.translation, _DEFAULT_TRANSLATION, "translation")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def active(self) -> bool:
        """Whether anything is actually drawn."""
        return any(
            block is not None for block in (self.rotation, self.scale, self.translation)
        )

    def params(self, *, epoch: int = 0, index: int = 0) -> dict:
        """The draw for one (epoch, sample), as a plain dict.

        A pure function of ``(seed, epoch, index)``: the same three give the
        same transform in any process, in any order, on any run.
        """
        rng = random.Random(f"{self.seed}/{int(epoch)}/{int(index)}")
        out = {"epoch": int(epoch), "index": int(index), "seed": self.seed}
        if self.rotation is not None:
            limit = float(self.rotation["max_degrees"])
            angle = rng.uniform(-limit, limit)
            axis = self.rotation["axis"]
            if axis == "any":
                # A uniform direction on the sphere: z uniform in [-1, 1] with
                # the azimuth uniform, which is Archimedes' rule -- drawing the
                # polar angle uniformly instead would cluster at the poles.
                z = rng.uniform(-1.0, 1.0)
                phi = rng.uniform(0.0, 2.0 * math.pi)
                r = math.sqrt(max(0.0, 1.0 - z * z))
                out["rotate"] = [r * math.cos(phi), r * math.sin(phi), z, angle]
            else:
                out["rotate"] = [axis, angle]
        if self.scale is not None:
            out["scale"] = rng.uniform(
                float(self.scale["low"]), float(self.scale["high"])
            )
        if self.translation is not None:
            limit = float(self.translation["max"])
            out["translate"] = [rng.uniform(-limit, limit) for _ in range(3)]
        return out

    def apply(self, mesh, *, epoch: int = 0, index: int = 0, params=None):
        """Augment one mesh -> ``(mesh, params)``.

        Pass a previous call's ``params`` back to replay the *same* transform,
        which is how a paired input and target stay coherent.
        """
        if params is None:
            params = self.params(epoch=epoch, index=index)
        if not self.active:
            return mesh, params
        rotate = params.get("rotate")
        if rotate is not None and len(rotate) == 4:
            rotate = (rotate[:3], rotate[3])
        elif rotate is not None:
            rotate = (rotate[0], rotate[1])
        out = transform(
            mesh,
            translate=params.get("translate"),
            scale=params.get("scale"),
            rotate=rotate,
            rotate_vector_data=True,
        )
        return out, params

    def to_dict(self) -> dict:
        """snake_case, for a model card or a run's own record."""
        doc = {"version": AUGMENTATION_VERSION, "seed": self.seed}
        for name in ("rotation", "scale", "translation"):
            block = getattr(self, name)
            if block is not None:
                doc[name] = dict(block)
        return doc

    @classmethod
    def from_dict(cls, doc) -> "Augmentation":
        if isinstance(doc, Augmentation):
            return doc
        doc = dict(doc or {})
        doc.pop("version", None)
        return cls(**doc)

    @classmethod
    def from_spec(cls, block) -> Optional["Augmentation"]:
        """The PascalCase ``Augmentation`` spec block -> an instance."""
        if block is None or block is False:
            return None
        if block is True:
            return cls(rotation=True, scale=True, translation=True)
        if not isinstance(block, dict):
            raise ValueError(
                "meshio++: train: Augmentation must be true, false or an object"
            )
        known = {"Rotation", "Scale", "Translation", "Seed"}
        unknown = set(block) - known
        if unknown:
            raise ValueError(
                f"meshio++: train: unknown key(s) {sorted(unknown)} in "
                f"Augmentation (known: {', '.join(sorted(known))})"
            )

        def _lower(value, mapping):
            if not isinstance(value, dict):
                return value
            out = {}
            for key, item in value.items():
                if key not in mapping:
                    raise ValueError(
                        f"meshio++: train: unknown key '{key}' in Augmentation "
                        f"(known: {', '.join(sorted(mapping))})"
                    )
                out[mapping[key]] = item
            return out

        return cls(
            rotation=_lower(
                block.get("Rotation"), {"Axis": "axis", "MaxDegrees": "max_degrees"}
            ),
            scale=_lower(block.get("Scale"), {"Low": "low", "High": "high"}),
            translation=_lower(block.get("Translation"), {"Max": "max"}),
            seed=int(block.get("Seed", 0)),
        )
