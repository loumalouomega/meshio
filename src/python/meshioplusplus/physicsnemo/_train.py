"""The training spec and run-directory layout: the pure half of the trainer.

Everything here is stdlib + numpy and Python 3.8 -- the spec parser, the run
layout, the metrics/progress files, the model card -- so it is tested in the
default CI matrix with no framework installed. ``train.py`` (the module run
as ``python -m meshioplusplus.physicsnemo.train``) is the only place torch,
PhysicsNeMo and PyTorch Geometric are imported.

**The spec is a hand-editable input, so it is a settings-family document**
(PascalCase keys, ``"Version": 1``, strict unknown-key refusal -- the
``DatasetManifest``/pipeline convention); everything a run *writes*
(``job.json``, ``progress.json``, ``metrics.jsonl``, ``*.card.json``) is a
machine artefact and is snake_case (the ``write_dataset`` precedent).

Run directory layout (``RunDir``)::

    spec.json           the spec the run was started from (PascalCase)
    job.json            written by the job manager (mcp/_jobs.py), not here
    progress.json       {epoch, epochs, best_epoch, best_valid_loss, ...}
    metrics.jsonl       one JSON object per epoch
    log.txt             stdout+stderr of the trainer (the manager redirects)
    node_stats.json     the PhysicsNeMo-convention normalization stats
    edge_stats.json
    checkpoints/        <Model>.0.<epoch>.mdlus + checkpoint.0.<epoch>.pt
                        (physicsnemo's save_checkpoint), best.mdlus,
                        final.mdlus -- every .mdlus with a .card.json sidecar
    predictions/        <entry_id>.vtu written by predict()

The **model card** (``<checkpoint>.card.json``, the Kratos
PhysicsNeMoApplication convention) records what a checkpoint's channels
mean -- field names, column order, the normalization the model was trained
under -- so a prediction can never silently write normalized numbers onto a
physical field.
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..__about__ import __version__

SPEC_VERSION = 1
SPEC_FILE = "spec.json"
JOB_FILE = "job.json"
PROGRESS_FILE = "progress.json"
METRICS_FILE = "metrics.jsonl"
LOG_FILE = "log.txt"
CHECKPOINT_DIR = "checkpoints"
NODE_STATS_FILE = "node_stats.json"
EDGE_STATS_FILE = "edge_stats.json"
PREDICTIONS_DIR = "predictions"
BEST_CHECKPOINT = "best.mdlus"
FINAL_CHECKPOINT = "final.mdlus"
CARD_SUFFIX = ".card.json"
STATS_STD_FLOOR = 1e-8

_ERR = "meshio++: train: "
_TOP_KEYS = (
    "Version",
    "Manifest",
    "RunDir",
    "Fields",
    "TargetFields",
    "TrainSplit",
    "ValidSplit",
    "Epochs",
    "BatchSize",
    "LearningRate",
    "Seed",
    "CheckpointEvery",
    "Device",
    "Model",
    "Graph",
    "Grid",
    "Read",
    "Notes",
    "Tags",
    "Augmentation",
    "Guard",
)
#: Every ``Model`` key, across families. Which of them are *legal* depends on
#: ``Model.Name`` -- see ``_MODEL_FAMILY_KEYS`` -- so that a graph
#: hyperparameter on a CNN is refused by name rather than silently ignored.
_MODEL_KEYS = (
    "Name",
    # meshgraphnet
    "ProcessorSize",
    "HiddenDim",
    "Aggregation",
    # srresnet
    "ScalingFactor",
    "ConvLayerSize",
    "ResidBlocks",
    "LargeKernelSize",
    "SmallKernelSize",
    "ActivationFn",
)
_MODEL_FAMILY_KEYS = {
    "meshgraphnet": ("Name", "ProcessorSize", "HiddenDim", "Aggregation"),
    "srresnet": (
        "Name",
        "ScalingFactor",
        "ConvLayerSize",
        "ResidBlocks",
        "LargeKernelSize",
        "SmallKernelSize",
        "ActivationFn",
    ),
}
_GRAPH_KEYS = (
    "Regions",
    "Kind",
    "Undirected",
    "EdgeFeatures",
    "Float32",
    "TargetOffset",
    "TargetDelta",
    "Proximity",
)
_GRID_KEYS = (
    "Resolution",
    "CellSize",
    "Bounds",
    "Padding",
    "PaddingRelative",
    "Extrapolate",
    "FillValue",
    "Squeeze",
    "SqueezeIndex",
    "MaxCells",
    "Float32",
)
_MODELS = ("meshgraphnet", "srresnet")
#: Which spec block each family reads. The other one is refused rather than
#: ignored: a Grid block on a graph model means the author expected something
#: that will not happen.
_FAMILY_BLOCK = {"meshgraphnet": "Graph", "srresnet": "Grid"}
#: `SRResNet` accepts only these, and says so itself -- but failing here names
#: the spec key instead of surfacing from inside torch.
_SCALING_FACTORS = (2, 4, 8)
_KINDS = ("node", "cell")
#: `Graph.Proximity` builds the edges from geometry instead of connectivity,
#: in `meshioplusplus.proximity_graph`'s vocabulary. There is deliberately no
#: `Graph.WorldEdges` or bistride block: no shipped model family reads a
#: second edge set, and a key a run would silently ignore is exactly what the
#: strict unknown-key refusal exists to prevent.
_PROXIMITY_KEYS = ("Method", "Radius", "MaxNeighbors", "BoxSize")
_PROXIMITY_METHODS = ("radius", "knn")
_AGGREGATIONS = ("sum", "mean")
_DEVICES_PREFIX = ("auto", "cpu", "cuda")


def _check_keys(obj, where, allowed):
    for key in obj:
        if key not in allowed:
            raise ValueError(
                f"{_ERR}unknown key '{key}' in {where} (known: {', '.join(allowed)})"
            )


def _names(value, where):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(v, str) and v for v in value
    ):
        raise ValueError(f"{_ERR}{where} must be a list of non-empty strings")
    return tuple(value)


def _int(value, where, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{_ERR}{where} must be an integer >= {minimum}")
    return int(value)


def _triple(value, where):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{_ERR}{where} must be three cell counts")
    return tuple(_int(v, where, 1) for v in value)


def _bool(value, where):
    if not isinstance(value, bool):
        raise ValueError(f"{_ERR}{where} must be true or false")
    return value


def _proximity(value, where):
    """`Graph.Proximity` -> the snake_case dict `proximity_graph` takes.

    Validated here rather than left to the first sample: a malformed
    neighbourhood should fail when the document is read, not an epoch later.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{_ERR}{where} must be an object")
    _check_keys(value, where, _PROXIMITY_KEYS)
    method = value.get("Method", "radius")
    if method not in _PROXIMITY_METHODS:
        raise ValueError(
            f"{_ERR}{where}.Method must be one of "
            f"{', '.join(_PROXIMITY_METHODS)}, not {method!r}"
        )
    out = {"method": method}
    if method == "radius":
        if "MaxNeighbors" in value:
            raise ValueError(f"{_ERR}{where}.MaxNeighbors belongs to Method 'knn'")
        radius = value.get("Radius")
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise ValueError(f"{_ERR}{where}.Radius must be a positive number")
        if radius <= 0:
            raise ValueError(f"{_ERR}{where}.Radius must be a positive number")
        out["radius"] = float(radius)
    else:
        if "Radius" in value:
            raise ValueError(f"{_ERR}{where}.Radius belongs to Method 'radius'")
        out["max_neighbors"] = _int(
            value.get("MaxNeighbors", 0), f"{where}.MaxNeighbors", 1
        )
    box = value.get("BoxSize")
    if box not in (None, []):
        if isinstance(box, (int, float)) and not isinstance(box, bool):
            box = [float(box)]
        elif isinstance(box, (list, tuple)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
            for v in box
        ):
            box = [float(v) for v in box]
        else:
            raise ValueError(
                f"{_ERR}{where}.BoxSize must be a positive number or a list of them"
            )
        out["box_size"] = box
    return out


@dataclass(frozen=True)
class TrainSpec:
    """A training run's inputs -- the PascalCase document, typed.

    ``base_dir`` (never serialized, excluded from equality) is the directory
    a spec loaded from a file resolves its relative ``Manifest``/``RunDir``
    against; a dict/JSON-text load resolves against the CWD.
    """

    manifest: str
    fields: Tuple[str, ...]
    target_fields: Tuple[str, ...]
    run_dir: str = "runs/run"
    train_split: str = "train"
    valid_split: str = "valid"
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    seed: int = 0
    checkpoint_every: int = 10
    device: str = "auto"
    model_name: str = "meshgraphnet"
    processor_size: int = 8
    hidden_dim: int = 64
    aggregation: str = "sum"
    # srresnet
    scaling_factor: int = 2
    conv_layer_size: int = 32
    resid_blocks: int = 8
    large_kernel_size: int = 7
    small_kernel_size: int = 3
    activation_fn: str = "prelu"
    regions: bool = False
    kind: str = "node"
    undirected: bool = True
    edge_features: bool = True
    float32: bool = True
    target_offset: int = 0
    target_delta: bool = False
    proximity: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None
    guard: Optional[Dict[str, Any]] = None
    # Grid (srresnet); resolution/cell_size are the `GridSpec.from_mesh` pair
    resolution: Optional[Tuple[int, int, int]] = None
    cell_size: Optional[float] = None
    bounds: Optional[Tuple[float, ...]] = None
    padding: float = 0.0
    padding_relative: float = 0.0
    extrapolate: bool = False
    fill_value: float = 0.0
    squeeze: Optional[int] = None
    squeeze_index: Optional[int] = None
    max_cells: int = 20000000
    read: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None
    tags: Tuple[str, ...] = ()
    base_dir: Optional[str] = field(default=None, compare=False, repr=False)

    def resolved_manifest(self) -> str:
        return _resolve_against(self.manifest, self.base_dir)

    def resolved_run_dir(self) -> str:
        return _resolve_against(self.run_dir, self.base_dir)

    def grid_kwargs(self) -> Dict[str, Any]:
        """The :func:`grid_sample_pair` parameters this spec describes.

        :meth:`graph_kwargs`' sibling, and like it the single thing recorded on
        the model card and replayed verbatim at inference, so training and
        prediction cannot disagree about how the grid was built.
        """
        coarse = {
            "resolution": list(self.resolution) if self.resolution else None,
            "cell_size": self.cell_size,
            "bounds": list(self.bounds) if self.bounds else None,
            "padding": self.padding,
            "padding_relative": self.padding_relative,
            "max_cells": self.max_cells,
        }
        return {
            "fields": list(self.fields),
            "target_fields": list(self.target_fields),
            "coarse": coarse,
            "scaling_factor": self.scaling_factor,
            "extrapolate": self.extrapolate,
            "fill_value": self.fill_value,
            "squeeze": self.squeeze,
            "squeeze_index": self.squeeze_index,
            "float32": self.float32,
        }

    def graph_kwargs(self) -> Dict[str, Any]:
        """The :func:`graph_sample`/``make_dataset`` keyword arguments."""
        return {
            "fields": list(self.fields),
            "target_fields": list(self.target_fields),
            "regions": self.regions,
            "kind": self.kind,
            "undirected": self.undirected,
            "edge_features": self.edge_features,
            "float32": self.float32,
            "target_offset": self.target_offset,
            "target_delta": self.target_delta,
            "proximity": None if self.proximity is None else dict(self.proximity),
        }


def _proximity_to_document(proximity):
    """The inverse of :func:`_proximity`: snake_case back to the document."""
    doc = {"Method": proximity["method"]}
    if "radius" in proximity:
        doc["Radius"] = proximity["radius"]
    if "max_neighbors" in proximity:
        doc["MaxNeighbors"] = proximity["max_neighbors"]
    if "box_size" in proximity:
        doc["BoxSize"] = list(proximity["box_size"])
    return doc


def _resolve_against(path, base_dir):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir or os.getcwd(), path))


def spec_from_dict(doc, *, base_dir=None) -> TrainSpec:
    """Parse a PascalCase spec document, strictly."""
    if not isinstance(doc, dict):
        raise ValueError(f"{_ERR}the spec must be a JSON object")
    _check_keys(doc, "the spec", _TOP_KEYS)
    version = doc.get("Version", SPEC_VERSION)
    if version != SPEC_VERSION:
        raise ValueError(
            f"{_ERR}unsupported Version {version!r} (this build knows {SPEC_VERSION})"
        )
    if not isinstance(doc.get("Manifest"), str) or not doc["Manifest"]:
        raise ValueError(f"{_ERR}Manifest is required (a dataset manifest path)")
    fields = _names(doc.get("Fields", ()), "Fields")
    targets = _names(doc.get("TargetFields", ()), "TargetFields")
    if not fields:
        raise ValueError(f"{_ERR}Fields must name at least one input array")
    if not targets:
        raise ValueError(f"{_ERR}TargetFields must name at least one target array")
    model = doc.get("Model", {})
    if not isinstance(model, dict):
        raise ValueError(f"{_ERR}Model must be an object")
    graph = doc.get("Graph", {})
    if not isinstance(graph, dict):
        raise ValueError(f"{_ERR}Graph must be an object")
    _check_keys(graph, "Graph", _GRAPH_KEYS)
    grid = doc.get("Grid", {})
    if not isinstance(grid, dict):
        raise ValueError(f"{_ERR}Grid must be an object")
    _check_keys(grid, "Grid", _GRID_KEYS)
    read = doc.get("Read", {})
    if not isinstance(read, dict):
        raise ValueError(f"{_ERR}Read must be an object of read() keyword arguments")
    name = str(model.get("Name", "meshgraphnet")).lower()
    if name not in _MODELS:
        raise ValueError(f"{_ERR}Model.Name must be one of {', '.join(_MODELS)}")

    # Cross-family strictness. A hyperparameter meant for the other family is
    # refused by name rather than ignored: the author expected something that
    # is not going to happen, and a silently dropped key is how a run ends up
    # training a model nobody asked for.
    _check_keys(model, f"Model (with Name '{name}')", _MODEL_FAMILY_KEYS[name])
    wanted = _FAMILY_BLOCK[name]
    unwanted = "Grid" if wanted == "Graph" else "Graph"
    if doc.get(unwanted):
        raise ValueError(
            f"{_ERR}a '{name}' model reads the {wanted} block, not {unwanted}; "
            f"remove {unwanted} or change Model.Name"
        )

    # Augmentation rotates and rescales the geometry, which moves a grid
    # sample's own lattice relative to the mesh and so changes its coverage.
    # Refused by name for the grid family rather than silently applied.
    augmentation = doc.get("Augmentation")
    if augmentation is not None and augmentation is not False:
        if name == "srresnet":
            raise ValueError(
                f"{_ERR}Augmentation applies to the graph families; a "
                "'srresnet' samples a fixed lattice, so rotating the mesh "
                "under it would change the pair's own coverage"
            )
        from ._augment import Augmentation

        augmentation = Augmentation.from_spec(augmentation).to_dict()
    else:
        augmentation = None

    # A geometry guardrail is fitted on the training split at the start of a
    # run and stored in the card. Both families can carry one -- it describes
    # the shapes, not the sample.
    guard = doc.get("Guard")
    if guard is None or guard is False:
        guard = None
    elif guard is True:
        guard = {}
    elif isinstance(guard, dict):
        known = {"Margin", "Quality"}
        unknown = set(guard) - known
        if unknown:
            raise ValueError(
                f"{_ERR}unknown key(s) {sorted(unknown)} in Guard "
                f"(known: {', '.join(sorted(known))})"
            )
        margin = guard.get("Margin", 1.5)
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            raise ValueError(f"{_ERR}Guard.Margin must be a positive number")
        if margin <= 0:
            raise ValueError(f"{_ERR}Guard.Margin must be a positive number")
        block = {"margin": float(margin)}
        if "Quality" in guard:
            block["quality"] = _bool(guard["Quality"], "Guard.Quality")
        guard = block
    else:
        raise ValueError(f"{_ERR}Guard must be true, false or an object")

    aggregation = str(model.get("Aggregation", "sum"))
    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            f"{_ERR}Model.Aggregation must be one of {', '.join(_AGGREGATIONS)}"
        )
    kind = str(graph.get("Kind", "node"))
    if kind not in _KINDS:
        raise ValueError(f"{_ERR}Graph.Kind must be one of {', '.join(_KINDS)}")

    scaling_factor = _int(model.get("ScalingFactor", 2), "Model.ScalingFactor", 1)
    if name == "srresnet" and scaling_factor not in _SCALING_FACTORS:
        raise ValueError(
            f"{_ERR}Model.ScalingFactor must be one of "
            f"{', '.join(str(s) for s in _SCALING_FACTORS)} (SRResNet accepts no "
            f"others), got {scaling_factor}"
        )
    resolution = _triple(grid.get("Resolution"), "Grid.Resolution")
    cell_size = grid.get("CellSize")
    if cell_size is not None:
        if isinstance(cell_size, bool) or not isinstance(cell_size, (int, float)):
            raise ValueError(f"{_ERR}Grid.CellSize must be a number")
        cell_size = float(cell_size)
    if name == "srresnet" and (resolution is None) == (cell_size is None):
        raise ValueError(f"{_ERR}give exactly one of Grid.Resolution and Grid.CellSize")
    bounds = grid.get("Bounds")
    if bounds is not None:
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 6:
            raise ValueError(
                f"{_ERR}Grid.Bounds must be [xlo, ylo, zlo, xhi, yhi, zhi]"
            )
        bounds = tuple(float(v) for v in bounds)
    squeeze = grid.get("Squeeze")
    if squeeze is not None:
        squeeze = _int(squeeze, "Grid.Squeeze")
        if squeeze not in (0, 1, 2):
            raise ValueError(f"{_ERR}Grid.Squeeze must be a world axis 0, 1 or 2")
        if name == "srresnet":
            raise ValueError(
                f"{_ERR}Grid.Squeeze does not apply to 'srresnet', which is 3-D "
                "throughout (Conv3d); it is for a 2-D operator"
            )
    squeeze_index = grid.get("SqueezeIndex")
    if squeeze_index is not None:
        squeeze_index = _int(squeeze_index, "Grid.SqueezeIndex")
    device = str(doc.get("Device", "auto"))
    if not device.startswith(_DEVICES_PREFIX):
        raise ValueError(f"{_ERR}Device must be auto, cpu, cuda or cuda:N")
    lr = doc.get("LearningRate", 1e-3)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"{_ERR}LearningRate must be a positive number")
    notes = doc.get("Notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"{_ERR}Notes must be a string")
    return TrainSpec(
        manifest=doc["Manifest"],
        fields=fields,
        target_fields=targets,
        run_dir=str(doc.get("RunDir", "runs/run")),
        train_split=str(doc.get("TrainSplit", "train")),
        valid_split=str(doc.get("ValidSplit", "valid")),
        epochs=_int(doc.get("Epochs", 100), "Epochs", 1),
        batch_size=_int(doc.get("BatchSize", 8), "BatchSize", 1),
        learning_rate=float(lr),
        seed=_int(doc.get("Seed", 0), "Seed"),
        checkpoint_every=_int(doc.get("CheckpointEvery", 10), "CheckpointEvery"),
        device=device,
        model_name=name,
        processor_size=_int(model.get("ProcessorSize", 8), "Model.ProcessorSize", 1),
        hidden_dim=_int(model.get("HiddenDim", 64), "Model.HiddenDim", 1),
        aggregation=aggregation,
        scaling_factor=scaling_factor,
        conv_layer_size=_int(model.get("ConvLayerSize", 32), "Model.ConvLayerSize", 1),
        resid_blocks=_int(model.get("ResidBlocks", 8), "Model.ResidBlocks", 1),
        large_kernel_size=_int(
            model.get("LargeKernelSize", 7), "Model.LargeKernelSize", 1
        ),
        small_kernel_size=_int(
            model.get("SmallKernelSize", 3), "Model.SmallKernelSize", 1
        ),
        activation_fn=str(model.get("ActivationFn", "prelu")),
        regions=_bool(graph.get("Regions", False), "Graph.Regions"),
        kind=kind,
        undirected=_bool(graph.get("Undirected", True), "Graph.Undirected"),
        edge_features=_bool(graph.get("EdgeFeatures", True), "Graph.EdgeFeatures"),
        float32=_bool(graph.get("Float32", True), "Graph.Float32"),
        target_offset=_int(graph.get("TargetOffset", 0), "Graph.TargetOffset"),
        target_delta=_bool(graph.get("TargetDelta", False), "Graph.TargetDelta"),
        proximity=_proximity(graph.get("Proximity"), "Graph.Proximity"),
        resolution=resolution,
        cell_size=cell_size,
        bounds=bounds,
        padding=float(grid.get("Padding", 0.0)),
        padding_relative=float(grid.get("PaddingRelative", 0.0)),
        extrapolate=_bool(grid.get("Extrapolate", False), "Grid.Extrapolate"),
        fill_value=float(grid.get("FillValue", 0.0)),
        squeeze=squeeze,
        squeeze_index=squeeze_index,
        max_cells=_int(grid.get("MaxCells", 20000000), "Grid.MaxCells", 1),
        read=dict(read),
        notes=notes,
        tags=_names(doc.get("Tags", ()), "Tags"),
        augmentation=augmentation,
        guard=guard,
        base_dir=base_dir,
    )


def spec_to_dict(spec: TrainSpec) -> dict:
    """The PascalCase document (round-trip exact through :func:`spec_from_dict`)."""
    doc = {
        "Version": SPEC_VERSION,
        "Manifest": spec.manifest,
        "RunDir": spec.run_dir,
        "Fields": list(spec.fields),
        "TargetFields": list(spec.target_fields),
        "TrainSplit": spec.train_split,
        "ValidSplit": spec.valid_split,
        "Epochs": spec.epochs,
        "BatchSize": spec.batch_size,
        "LearningRate": spec.learning_rate,
        "Seed": spec.seed,
        "CheckpointEvery": spec.checkpoint_every,
        "Device": spec.device,
    }
    # Only the family's own blocks are emitted. Round-tripping is still exact
    # (the other family's fields keep their dataclass defaults), and it keeps a
    # spec readable: a superresolution document that listed graph aggregation
    # would invite someone to change it and wonder why nothing happened.
    if spec.model_name == "srresnet":
        doc["Model"] = {
            "Name": spec.model_name,
            "ScalingFactor": spec.scaling_factor,
            "ConvLayerSize": spec.conv_layer_size,
            "ResidBlocks": spec.resid_blocks,
            "LargeKernelSize": spec.large_kernel_size,
            "SmallKernelSize": spec.small_kernel_size,
            "ActivationFn": spec.activation_fn,
        }
        grid = {
            "Padding": spec.padding,
            "PaddingRelative": spec.padding_relative,
            "Extrapolate": spec.extrapolate,
            "FillValue": spec.fill_value,
            "MaxCells": spec.max_cells,
            "Float32": spec.float32,
        }
        if spec.resolution is not None:
            grid["Resolution"] = list(spec.resolution)
        if spec.cell_size is not None:
            grid["CellSize"] = spec.cell_size
        if spec.bounds is not None:
            grid["Bounds"] = list(spec.bounds)
        if spec.squeeze is not None:
            grid["Squeeze"] = spec.squeeze
        if spec.squeeze_index is not None:
            grid["SqueezeIndex"] = spec.squeeze_index
        doc["Grid"] = grid
    else:
        doc["Model"] = {
            "Name": spec.model_name,
            "ProcessorSize": spec.processor_size,
            "HiddenDim": spec.hidden_dim,
            "Aggregation": spec.aggregation,
        }
        doc["Graph"] = {
            "Regions": spec.regions,
            "Kind": spec.kind,
            "Undirected": spec.undirected,
            "EdgeFeatures": spec.edge_features,
            "Float32": spec.float32,
            "TargetOffset": spec.target_offset,
            "TargetDelta": spec.target_delta,
        }
        if spec.proximity is not None:
            doc["Graph"]["Proximity"] = _proximity_to_document(spec.proximity)
    if spec.read:
        doc["Read"] = dict(spec.read)
    if spec.augmentation is not None:
        block = {"Seed": spec.augmentation.get("seed", 0)}
        for key, name in (
            ("rotation", "Rotation"),
            ("scale", "Scale"),
            ("translation", "Translation"),
        ):
            inner = spec.augmentation.get(key)
            if inner is not None:
                block[name] = {
                    "".join(part.title() for part in k.split("_")): v
                    for k, v in inner.items()
                }
        doc["Augmentation"] = block
    if spec.guard is not None:
        block = {}
        if "margin" in spec.guard:
            block["Margin"] = spec.guard["margin"]
        if "quality" in spec.guard:
            block["Quality"] = spec.guard["quality"]
        doc["Guard"] = block if block else True
    if spec.notes is not None:
        doc["Notes"] = spec.notes
    if spec.tags:
        doc["Tags"] = list(spec.tags)
    return doc


def load_spec(settings) -> TrainSpec:
    """Load a spec from a dict, JSON text, or a file path (tri-modal, the
    ``DatasetManifest.load`` rule); a file's directory becomes ``base_dir``."""
    if isinstance(settings, TrainSpec):
        return settings
    if isinstance(settings, dict):
        return spec_from_dict(settings)
    text = str(settings)
    if text.lstrip()[:1] in ("{", "["):
        try:
            return spec_from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"{_ERR}the spec is not valid JSON: {e}") from None
    if not os.path.isfile(text):
        raise ValueError(f"{_ERR}spec file not found: '{text}'")
    with open(text, encoding="utf-8") as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(f"{_ERR}'{text}' is not valid JSON: {e}") from None
    return spec_from_dict(doc, base_dir=os.path.dirname(os.path.abspath(text)))


def save_spec(spec: TrainSpec, path) -> str:
    write_json_atomic(path, spec_to_dict(spec))
    return str(path)


def default_spec(manifest, fields, target_fields, *, run_dir="runs/run", **overrides):
    """A spec with every default filled in; ``overrides`` are field names."""
    spec = TrainSpec(
        manifest=str(manifest),
        fields=_names(fields, "Fields"),
        target_fields=_names(target_fields, "TargetFields"),
        run_dir=str(run_dir),
    )
    if overrides:
        unknown = [k for k in overrides if k not in spec.__dataclass_fields__]
        if unknown:
            raise ValueError(f"{_ERR}unknown spec field(s) {unknown}")
        spec = replace(spec, **overrides)
    # Re-validate through the document, so overrides obey the same rules.
    return spec_from_dict(spec_to_dict(spec), base_dir=spec.base_dir)


# --------------------------------------------------------------------------- #
# Run-directory files                                                         #
# --------------------------------------------------------------------------- #
def write_json_atomic(path, obj) -> None:
    """Write JSON via a sibling temp file + rename, so a reader (the job
    manager polling ``progress.json``) never sees a torn file."""
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def append_metrics(run_dir, row: dict) -> None:
    """Append one epoch's row to ``metrics.jsonl``."""
    with open(os.path.join(run_dir, METRICS_FILE), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_metrics(run_dir, since_epoch: int = 0) -> List[dict]:
    """Every metrics row with ``epoch >= since_epoch`` (a torn last line, from
    a trainer mid-write, is skipped rather than fatal)."""
    path = os.path.join(run_dir, METRICS_FILE)
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if int(row.get("epoch", -1)) >= since_epoch:
                    rows.append(row)
    except OSError:
        return []
    return rows


def metrics_row(epoch, train_loss, valid_loss, lr, elapsed, epoch_seconds) -> dict:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "valid_loss": None if valid_loss is None else float(valid_loss),
        "lr": float(lr),
        "elapsed": float(elapsed),
        "epoch_seconds": float(epoch_seconds),
        "timestamp": time.time(),
    }


# --------------------------------------------------------------------------- #
# Normalization stats and the model card                                      #
# --------------------------------------------------------------------------- #
def stat_vectors(stats: dict, names: Sequence[str], suffix: str) -> np.ndarray:
    """Concatenate ``{name}_{suffix}`` vectors (the ``node_stats.json``
    convention) for ``names``, in order; a ``std`` is floored at
    ``STATS_STD_FLOOR`` so a constant field never divides by zero."""
    values: List[float] = []
    for name in names:
        key = f"{name}_{suffix}"
        if key not in stats:
            raise KeyError(f"{_ERR}no '{key}' in the normalization stats")
        values.extend(float(v) for v in stats[key])
    out = np.asarray(values, dtype=np.float64)
    if suffix.endswith("std"):
        out = np.maximum(out, STATS_STD_FLOOR)
    return out


def _input_normalization(schema: dict, node_stats: dict, fields) -> Dict[str, list]:
    """The input normalizer, aligned to the recorded ``x_columns``.

    Built column by column rather than field by field, because they are not the
    same width: with ``Graph.Regions`` on, ``x`` carries a ``region:<name>``
    one-hot per region and the field statistics know nothing about those. A
    field-wide vector is then **narrower than the batch** and the broadcast in
    the training loop fails -- which is what happened before this function
    existed.

    A region one-hot contributes mean 0 / std 1: it is already 0/1 and must not
    be shifted, since a shifted one-hot no longer says "member".
    """
    sources = schema.get("x_sources")
    if not sources:
        # A schema from before sources were recorded: fall back to the old
        # field-wide vectors, which are correct whenever no region widened x.
        return {
            "mean": stat_vectors(node_stats, fields, "mean").tolist(),
            "std": stat_vectors(node_stats, fields, "std").tolist(),
        }
    mean, std = [], []
    for source in sources:
        if source.get("kind") == "region":
            mean.append(0.0)
            std.append(1.0)
            continue
        name = source["source"]
        component = source.get("component") or 0
        for key, out, floor in (("mean", mean, False), ("std", std, True)):
            values = node_stats.get(f"{name}_{key}")
            if values is None:
                raise KeyError(f"{_ERR}no '{name}_{key}' in the normalization stats")
            value = float(values[component])
            out.append(max(value, STATS_STD_FLOOR) if floor else value)
    return {"mean": mean, "std": std}


def card_from_run(
    spec: TrainSpec,
    schema: dict,
    node_stats: dict,
    edge_stats: dict,
    *,
    epoch: int,
    valid_loss: Optional[float],
    checkpoint: str,
    input_dims: Optional[Tuple[int, int, int]] = None,
) -> dict:
    """The model card for a checkpoint: what its channels mean and how they
    were normalized. ``input_dims`` = (nodes, edges, output)."""
    x_cols = list(schema["x_columns"])
    y_cols = list(schema["y_columns"])
    y_suffix = "diff_" if spec.target_delta else ""
    return {
        "version": 1,
        "meshioplusplus_version": __version__,
        "model": {
            "name": spec.model_name,
            "processor_size": spec.processor_size,
            "hidden_dim": spec.hidden_dim,
            "aggregation": spec.aggregation,
            "input_dim_nodes": input_dims[0] if input_dims else len(x_cols),
            "input_dim_edges": (
                input_dims[1] if input_dims else len(edge_stats.get("edge_mean", []))
            ),
            "output_dim": input_dims[2] if input_dims else len(y_cols),
        },
        "schema": schema,
        "x_columns": x_cols,
        "y_columns": y_cols,
        "fields": list(spec.fields),
        "target_fields": list(spec.target_fields),
        "graph": spec.graph_kwargs(),
        "read": dict(spec.read),
        # Informational: a checkpoint says what poses it was shown. Inference
        # never augments, so nothing reads this back.
        "augmentation": spec.augmentation,
        "input_normalization": _input_normalization(schema, node_stats, spec.fields),
        "output_normalization": {
            "mean": stat_vectors(
                node_stats, spec.target_fields, y_suffix + "mean"
            ).tolist(),
            "std": stat_vectors(
                node_stats, spec.target_fields, y_suffix + "std"
            ).tolist(),
        },
        "edge_normalization": {
            "mean": [float(v) for v in edge_stats.get("edge_mean", [])],
            "std": [
                max(float(v), STATS_STD_FLOOR) for v in edge_stats.get("edge_std", [])
            ],
        },
        "epoch": int(epoch),
        "valid_loss": None if valid_loss is None else float(valid_loss),
        "checkpoint": os.path.basename(checkpoint),
    }


def normalizers_from_card(card: dict) -> Dict[str, np.ndarray]:
    """``x_mean/x_std/y_mean/y_std/e_mean/e_std`` as float64 arrays."""
    out = {}
    for prefix, key in (
        ("x", "input_normalization"),
        ("y", "output_normalization"),
        # A grid model has no edges at all, so its card carries no edge block;
        # an absent one is empty rather than an error.
        ("e", "edge_normalization"),
    ):
        block = card.get(key) or {"mean": [], "std": []}
        out[f"{prefix}_mean"] = np.asarray(block["mean"], dtype=np.float64)
        out[f"{prefix}_std"] = np.maximum(
            np.asarray(block["std"], dtype=np.float64), STATS_STD_FLOOR
        )
    return out


def card_path(checkpoint) -> str:
    return str(checkpoint) + CARD_SUFFIX


def list_checkpoints(run_dir, best: Optional[str] = None) -> List[dict]:
    """Every ``.mdlus`` under ``checkpoints/`` with its card's epoch and
    validation loss (never parsed from the file name), oldest epoch first;
    ``best`` names the one the job marks best."""
    directory = os.path.join(run_dir, CHECKPOINT_DIR)
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.mdlus"))):
        name = os.path.basename(path)
        card = read_json(card_path(path), {}) or {}
        kind = (
            "best"
            if name == BEST_CHECKPOINT
            else "final" if name == FINAL_CHECKPOINT else "periodic"
        )
        out.append(
            {
                "path": path,
                "name": name,
                "kind": kind,
                "epoch": card.get("epoch"),
                "valid_loss": card.get("valid_loss"),
                "size": os.path.getsize(path),
                "is_best": best is not None and os.path.basename(best) == name,
            }
        )
    out.sort(key=lambda c: (c["epoch"] is None, c["epoch"] or 0, c["name"]))
    return out
