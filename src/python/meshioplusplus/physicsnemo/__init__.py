"""PhysicsNeMo adapter: meshes -> the tensors its datapipes expect.

The ``mcp/`` split, applied again: this ``__init__`` is **pure** -- numpy,
stdlib and meshioplusplus only -- and holds all the behaviour (`graph_sample`,
the stats accumulators, the manifest iteration), so it is testable in the
default CI matrix with nothing optional installed. ``_reader.py`` is the only
module importing ``physicsnemo`` (the Gen-2 ``Reader`` subclass), ``_pyg.py``
the only one importing ``torch_geometric`` (the PyG ``Dataset``), and both
are reached through the lazy ``make_reader``/``make_dataset`` factories.

Deliberately **not** imported by ``import meshioplusplus`` (the ``mcp``
subpackage precedent): the gated halves have their own Python floor
(PhysicsNeMo requires >= 3.11 against the package's 3.8) and heavyweight
dependencies, and a class subclassing an optional-dep base cannot live in an
eagerly imported module. Use ``import meshioplusplus.physicsnemo as mpn``.

There is deliberately no ``meshioplusplus[physicsnemo]`` pip extra:
``nvidia-physicsnemo`` hard-depends on ``torch``, whose default Linux wheel
bundles CUDA at multiple GB -- the exact dependency the repo already refuses
to pin for ``to_torch``/``to_jax`` (the CuPy packaging precedent). See
``doc/physicsnemo.md``.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from ..__about__ import __version__
from .._dataset import DatasetManifest, check_pairing
from .._gpu import _require_framework
from .._grid_transfer import GridSpec
from .._interop import _emit, _importable
from .._ml import FEATURE_SCHEMA_VERSION, edge_index, feature_matrix
from .._proximity import _graph_positions, edge_vectors, proximity_graph
from ._train import TrainSpec, default_spec, load_spec

__all__ = [
    "GraphSample",
    "graph_sample",
    "iter_samples",
    "GridSample",
    "grid_sample_pair",
    "iter_grid_samples",
    "grid_stats",
    "field_stats",
    "edge_stats",
    "make_reader",
    "make_dataset",
    "to_physicsnemo",
    "from_physicsnemo",
    "has_physicsnemo",
    "has_torch_geometric",
    "TrainSpec",
    "default_spec",
    "load_spec",
    "run_training",
    "predict",
    "predict_mesh",
    "predict_file",
]

# Version 2 (v9.30.0): the schema gained `target_offset`/`target_delta` --
# the documented bump rule (a stored v1 schema now compares unequal, which is
# the drift guard doing its job).
# Version 3 (v10.31.0): and `proximity`/`world_edges`/`world_edge_features`.
GRAPH_SAMPLE_VERSION = 3

_DOC = "doc/physicsnemo.md"

#: keyword arguments consumed by :func:`graph_sample`; everything else a
#: manifest-walking entry point receives is forwarded to ``read()``.
#: ``target_offset`` is also read (without popping) by the iteration layer,
#: which turns it into the per-sample ``target_mesh`` -- ``target_mesh``
#: itself is never an iteration-level kwarg, being derived per step.
_GRAPH_KWARGS = (
    "fields",
    "target_fields",
    "target_delta",
    "target_offset",
    "regions",
    "kind",
    "undirected",
    "edge_features",
    "float32",
    "proximity",
    "world_edges",
)


def has_physicsnemo():
    """Whether PhysicsNeMo is importable. There is deliberately no pip extra
    (install ``nvidia-physicsnemo`` directly; it imports as ``physicsnemo``)."""
    return _importable("physicsnemo")


def has_torch_geometric():
    """Whether PyTorch Geometric is importable. There is deliberately no pip
    extra (install ``torch_geometric`` directly)."""
    return _importable("torch_geometric")


@dataclass(frozen=True)
class GraphSample:
    """One graph's worth of arrays, plus the recorded feature contract.

    ``arrays`` holds plain numpy: ``pos``, ``x``, ``edge_index``, and
    optionally ``y`` and ``edge_attr``. ``x_columns``/``y_columns`` are the
    :func:`~meshioplusplus.feature_matrix` column contract carried through;
    ``schema`` is the JSON-serializable record to store at training time and
    compare at inference time (the feature-drift guard).
    """

    arrays: dict
    x_columns: tuple
    y_columns: tuple
    schema: dict


def _positions(mesh, kind):
    """Graph-vertex positions: mesh points for the node graph, block-major
    cell centroids (the `partition` numbering) for the cell dual.

    One line over ``_proximity._graph_positions``, which is the single owner:
    a proximity graph and a mesh graph over the same mesh must agree on where
    its vertices are, or their edge features describe different geometry.
    """
    return _graph_positions(mesh, kind)


def _edge_attributes(pos, ei, box_size=None):
    """The PhysicsNeMo MeshGraphNet edge-feature convention, exactly:
    relative displacement (source minus destination) plus its norm.

    :func:`~meshioplusplus.edge_vectors` is the owner; ``box_size`` carries a
    periodic box through, so an edge across a boundary is short rather than
    box-length.
    """
    return edge_vectors(pos, ei, box_size=box_size)


def _sample_edges(mesh, kind, undirected, proximity, world_edges):
    """The edge sets of one sample: ``(edge_index, world_edge_index)``.

    The single owner of "which edges does this configuration mean", shared by
    :func:`graph_sample` and :func:`edge_stats` -- statistics gathered over a
    different graph than the model trains on normalize the wrong thing, and
    nothing downstream would report the discrepancy.

    ``proximity`` **replaces** the mesh edges: a particle state has positions
    and no connectivity, so there is nothing to replace them with but a
    neighbourhood rule. ``world_edges`` **adds** a second set alongside them --
    contact between surfaces that are near but not connected.
    """
    if proximity is None:
        ei = edge_index(mesh, kind=kind, undirected=undirected)
    else:
        ei = proximity_graph(mesh, kind=kind, undirected=undirected, **dict(proximity))
    world = (
        None
        if world_edges is None
        else proximity_graph(
            mesh, kind=kind, undirected=undirected, **dict(world_edges)
        )
    )
    return ei, world


def graph_sample(
    mesh,
    *,
    fields=None,
    target_fields=None,
    target_mesh=None,
    target_delta=False,
    target_offset=0,
    regions=True,
    kind="node",
    undirected=True,
    edge_features=True,
    float32=True,
    proximity=None,
    world_edges=None,
):
    """One mesh as one GNN training sample.

    - ``pos`` -- vertex positions ``(N, dim)``; never folded into ``x`` (the
      MeshGraphNet convention: coordinates enter through edge features only).
    - ``x`` -- :func:`~meshioplusplus.feature_matrix` at the graph's location
      with ``coords=False``: ``fields`` in stated order, then the
      ``region:<name>`` one-hots (the node-type slot); ``regions`` as there.
    - ``y`` -- a second ``feature_matrix`` over ``target_fields``; absent
      when ``None``. Three target modes: same-step (the default, the
      steady-state surrogate shape), **next-step** (``target_mesh`` -- the
      autoregressive t->t+1 pairing; the target's row count must match the
      input's, so a remeshed series fails by name rather than mis-pairing),
      and **delta** (``target_delta=True`` with a ``target_mesh``: y is the
      target's values minus this mesh's own -- the MeshGraphNet increment
      convention; normalize with ``field_stats(delta=...)``).
    - ``edge_index`` -- :func:`~meshioplusplus.edge_index` verbatim
      (``kind``/``undirected``); ``kind="cell"`` switches ``pos``/``x``/``y``
      to cell centroids and cell-located data, so the sample stays coherent.
    - ``edge_attr`` -- ``cat((pos[row] - pos[col], ||.||))``, the stable
      PhysicsNeMo convention; ``edge_features=False`` skips it.
    - ``world_edge_index``/``world_edge_attr`` -- present only with
      ``world_edges``, and kept as **separate arrays** rather than
      concatenated onto the mesh edges. PyTorch Geometric increments any
      attribute whose name contains ``index`` when it batches, so two edge
      sets batch correctly for free, while a concatenated one could not be
      split apart again afterwards. A model wanting the single edge set
      HybridMeshGraphNet expects concatenates them itself, mesh edges first.

    ``proximity`` and ``world_edges`` are dicts in
    :func:`~meshioplusplus.proximity_graph`'s vocabulary
    (``{"method", "radius", "max_neighbors", "box_size"}``). ``proximity``
    **replaces** the mesh edges -- the particle case, where a cloud has
    positions and no connectivity -- while ``world_edges`` **adds** a second
    set beside them.

    ``target_offset`` is pure metadata recorded in the schema -- this
    function sees two meshes, not a series, so the iteration layer
    (:func:`iter_samples`/`make_dataset`/`make_reader`) supplies both the
    ``target_mesh`` and the offset it used. ``float32=True`` (default -- the
    training convention) casts the float arrays; ``edge_index`` is int64
    always. Returns a :class:`GraphSample`.
    """
    if kind not in ("node", "cell"):
        raise ValueError(
            f"meshio++: graph_sample: kind must be 'node' or 'cell', not {kind!r}"
        )
    if target_mesh is not None and target_fields is None:
        raise ValueError(
            "meshio++: graph_sample: target_mesh requires target_fields "
            "(there is nothing to pair without target arrays)"
        )
    if target_delta and target_mesh is None:
        raise ValueError(
            "meshio++: graph_sample: target_delta requires target_mesh -- "
            "a delta against the same step is identically zero"
        )
    if target_offset < 0:
        raise ValueError(
            f"meshio++: graph_sample: target_offset must be >= 0, not {target_offset}"
        )
    location = "point" if kind == "node" else "cell"
    float_dtype = np.float32 if float32 else np.float64

    pos = _positions(mesh, kind)
    fmx = feature_matrix(mesh, location, fields=fields, coords=False, regions=regions)
    ei, world_ei = _sample_edges(mesh, kind, undirected, proximity, world_edges)
    box = None if proximity is None else proximity.get("box_size")

    arrays = {
        "pos": pos.astype(float_dtype, copy=False),
        "x": fmx.matrix.astype(float_dtype, copy=False),
        "edge_index": ei,
    }
    if world_ei is not None:
        arrays["world_edge_index"] = world_ei
    y_columns = ()
    y_sources = []
    if target_fields is not None:
        y_mesh = mesh if target_mesh is None else target_mesh
        fmy = feature_matrix(
            y_mesh, location, fields=target_fields, coords=False, regions=False
        )
        if fmy.matrix.shape[0] != pos.shape[0]:
            raise ValueError(
                f"meshio++: graph_sample: target_mesh yields "
                f"{fmy.matrix.shape[0]} {location} rows but mesh yields "
                f"{pos.shape[0]} -- t->t+1 pairing requires matching meshes"
            )
        y = fmy.matrix
        if target_delta:
            fmb = feature_matrix(
                mesh, location, fields=target_fields, coords=False, regions=False
            )
            if fmy.columns != fmb.columns:
                raise ValueError(
                    "meshio++: graph_sample: target_delta needs the same "
                    f"columns at both steps, got {fmy.columns} vs {fmb.columns}"
                )
            y = fmy.matrix - fmb.matrix
        arrays["y"] = y.astype(float_dtype, copy=False)
        y_columns = fmy.columns
        y_sources = fmy.schema["sources"]
    if edge_features:
        arrays["edge_attr"] = _edge_attributes(pos, ei, box).astype(
            float_dtype, copy=False
        )
        if world_ei is not None:
            arrays["world_edge_attr"] = _edge_attributes(
                pos, world_ei, world_edges.get("box_size")
            ).astype(float_dtype, copy=False)

    schema = {
        "graph_sample_version": GRAPH_SAMPLE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "meshioplusplus_version": str(__version__),
        "kind": kind,
        "undirected": bool(undirected),
        "float32": bool(float32),
        "target_offset": int(target_offset),
        "target_delta": bool(target_delta),
        "x_columns": list(fmx.columns),
        "y_columns": list(y_columns),
        "edge_features": (
            ["d" + "xyz"[i] for i in range(pos.shape[1])] + ["norm"]
            if edge_features
            else []
        ),
        "proximity": None if proximity is None else dict(proximity),
        "world_edges": None if world_edges is None else dict(world_edges),
        "world_edge_features": (
            ["d" + "xyz"[i] for i in range(pos.shape[1])] + ["norm"]
            if edge_features and world_ei is not None
            else []
        ),
        "x_sources": fmx.schema["sources"],
        "y_sources": y_sources,
    }
    return GraphSample(
        arrays=arrays,
        x_columns=fmx.columns,
        y_columns=tuple(y_columns),
        schema=schema,
    )


# --------------------------------------------------------------------------- #
# Walking a dataset manifest                                                  #
# --------------------------------------------------------------------------- #
def _as_manifest(manifest):
    if isinstance(manifest, DatasetManifest):
        return manifest
    return DatasetManifest.load(manifest)


def _split_kwargs(kwargs):
    graph_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _GRAPH_KWARGS}
    return graph_kwargs, kwargs


def _flat_items(manifest, split, read_kwargs, offset=0):
    """The flat (entry, step) index: one ``(entry_id, TimeSeries, step)`` per
    sample, resolved once from the manifest's plans -- no mesh read.

    With ``offset >= 1`` (t->t+offset pairing) each entry contributes
    ``len(series) - offset`` samples; an entry too short to pair contributes
    nothing, with one warning naming it -- emitted here at index-build time,
    never once per epoch.
    """
    items = []
    for entry in _as_manifest(manifest).entries(split=split):
        series = entry.time_series(**read_kwargs)
        if offset and len(series) <= offset:
            warnings.warn(
                f"meshio++: physicsnemo: entry '{entry.id}' has "
                f"{len(series)} step(s), fewer than target_offset+1="
                f"{offset + 1} -- it contributes no samples",
                stacklevel=2,
            )
            continue
        for step in range(max(len(series) - offset, 0)):
            items.append((entry.id, series, step))
    return items


def _read_sample(series, step, graph_kwargs):
    """One (or two, when pairing) mesh reads -> ``(time, GraphSample)``.

    The single owner of the read-and-sample step, shared by
    :func:`iter_samples`, the Gen-2 ``Reader`` and the PyG ``Dataset`` so
    the three cannot drift. ``TimeSeries`` caches nothing, so a paired
    sample (``target_offset >= 1``) costs exactly two reads -- **at most two
    meshes alive**, the pairing amendment to the streaming invariant.
    """
    offset = int(graph_kwargs.get("target_offset", 0))
    time, mesh = series[step]
    target = series[step + offset][1] if offset else None
    return time, graph_sample(mesh, target_mesh=target, **graph_kwargs)


def iter_samples(manifest, *, split=None, **kwargs):
    """Yield ``(entry_id, time, GraphSample)`` over a manifest's entries.

    A generator honouring the sequence streaming invariant: one mesh is
    alive per yielded sample -- at most two when ``target_offset >= 1``
    pairs step k with step k+offset (the yielded ``time`` is the *input*
    step's). ``kwargs`` split into :func:`graph_sample` parameters and
    ``read()`` kwargs (``arrays=...`` narrowing, ...); ``manifest`` is a
    :class:`~meshioplusplus.DatasetManifest` or anything its ``load``
    accepts.
    """
    graph_kwargs, read_kwargs = _split_kwargs(dict(kwargs))
    offset = int(graph_kwargs.get("target_offset", 0))
    for entry_id, series, step in _flat_items(manifest, split, read_kwargs, offset):
        time, sample = _read_sample(series, step, graph_kwargs)
        yield entry_id, time, sample


# --------------------------------------------------------------------------- #
# Grid samples: the coarse/fine pair a convolutional model trains on          #
# --------------------------------------------------------------------------- #
#: Bumped when the recorded grid-sample contract changes meaning.
GRID_SAMPLE_VERSION = 1

_GRID_KWARGS = (
    "fields",
    "target_fields",
    "coarse",
    "fine",
    "scaling_factor",
    "extrapolate",
    "fill_value",
    "squeeze",
    "squeeze_index",
    "float32",
)


@dataclass(frozen=True)
class GridSample:
    """One coarse/fine pair, plus the contract that names its channels.

    :class:`GraphSample`'s shape for a grid-shaped model. ``arrays`` holds
    ``x`` (the coarse grid, ``(C, D, H, W)``) and, when targets were asked
    for, ``y`` (the fine grid, ``(C', sD, sH, sW)``).
    """

    arrays: dict
    x_channels: tuple
    y_channels: tuple
    schema: dict


def _grid_specs(mesh, coarse, fine, scaling_factor):
    """Resolve the coarse and fine specs of a pair, and check they agree.

    Exactly one of ``fine`` and ``scaling_factor`` decides the fine grid.
    ``scaling_factor`` goes through :meth:`GridSpec.upscale_samples`, **not**
    ``upscale``: a convolutional upsampler multiplies sample counts, not cells
    (see that method's own note).
    """
    if not isinstance(coarse, GridSpec):
        coarse = GridSpec.from_mesh(mesh, **dict(coarse))
    if (fine is None) == (scaling_factor is None):
        raise ValueError(
            "meshio++: grid_sample_pair: give exactly one of fine and " "scaling_factor"
        )
    if scaling_factor is not None:
        fine = coarse.upscale_samples(int(scaling_factor))
    elif not isinstance(fine, GridSpec):
        fine = GridSpec.from_mesh(mesh, **dict(fine))
    if not coarse.same_bounds(fine):
        raise ValueError(
            "meshio++: grid_sample_pair: the coarse and fine grids cover "
            f"different boxes ({coarse.bounds} vs {fine.bounds}); take the box "
            "from the fine mesh so every fine node is inside the coarse grid"
        )
    factor = coarse.sample_scaling_factor(fine)
    if factor is None:
        raise ValueError(
            f"meshio++: grid_sample_pair: the coarse grid's shape {coarse.shape} "
            f"does not scale to the fine grid's {fine.shape} by one whole factor "
            "on every axis; build the fine spec with "
            "GridSpec.upscale_samples(factor)"
        )
    return coarse, fine, factor


def grid_sample_pair(
    mesh,
    coarse,
    *,
    fine=None,
    scaling_factor=None,
    target_mesh=None,
    fields=None,
    target_fields=None,
    extrapolate=False,
    fill_value=0.0,
    squeeze=None,
    squeeze_index=None,
    float32=True,
):
    """Sample a mesh (and optionally its paired target) onto a coarse/fine pair.

    The grid counterpart of :func:`graph_sample`. ``x`` is ``mesh`` on the
    coarse grid; ``y`` is the fine grid, taken from ``target_mesh`` when there
    is one and from ``mesh`` itself when there is not -- **self-supervision is
    the ordinary case**, since a superresolution dataset is usually one
    high-resolution solve per case with the coarse input made by sampling it
    less finely.

    Give exactly one of ``fine`` (a spec, or the kwargs to build one) and
    ``scaling_factor`` (an integer, resolved through
    :meth:`GridSpec.upscale_samples`). ``squeeze`` collapses a world axis for a
    2-D operator: an integer index keeps that plane, ``"mean"`` averages.
    """
    from .._grid_transfer import sample_grid, squeeze_grid

    coarse, fine, factor = _grid_specs(mesh, coarse, fine, scaling_factor)
    target_fields = list(target_fields) if target_fields else None
    y_mesh = mesh if target_mesh is None else target_mesh

    x = sample_grid(
        mesh,
        coarse,
        fields=fields,
        extrapolate=extrapolate,
        fill_value=fill_value,
        float32=float32,
    )
    arrays = {"x": x.values}
    y_channels = ()
    y = None
    if target_fields is not None:
        y = sample_grid(
            y_mesh,
            fine,
            fields=target_fields,
            extrapolate=extrapolate,
            fill_value=fill_value,
            float32=float32,
        )
        arrays["y"] = y.values
        y_channels = y.channels

    if squeeze is not None:
        axis, reduce = _squeeze_args(squeeze, squeeze_index)
        for key in list(arrays):
            arrays[key] = squeeze_grid(
                arrays[key], axis, index=squeeze_index, reduce=reduce
            )

    schema = {
        "grid_sample_version": GRID_SAMPLE_VERSION,
        "grid_schema_version": x.schema["grid_schema_version"],
        "meshioplusplus_version": str(__version__),
        "layout": x.schema["layout"],
        "coarse": coarse.to_dict(),
        "fine": fine.to_dict(),
        "scaling_factor": factor,
        "squeeze": squeeze,
        "squeeze_index": squeeze_index,
        "float32": bool(float32),
        "x_channels": list(x.channels),
        "y_channels": list(y_channels),
        "x_coverage": x.coverage,
        "y_coverage": None if y is None else y.coverage,
    }
    return GridSample(
        arrays=arrays,
        x_channels=tuple(x.channels),
        y_channels=tuple(y_channels),
        schema=schema,
    )


def _squeeze_args(squeeze, squeeze_index):
    """``squeeze`` is a world axis; ``squeeze_index`` picks a plane, else mean."""
    axis = int(squeeze)
    if axis not in (0, 1, 2):
        raise ValueError(
            f"meshio++: grid_sample_pair: squeeze must be a world axis 0, 1 or 2, "
            f"got {squeeze!r}"
        )
    return axis, (None if squeeze_index is not None else "mean")


def _grid_flat_items(manifest, split, read_kwargs):
    """The flat (entry, step) index for grid pairs -- no mesh read.

    Every entry contributes one sample per step. A ``Target`` is checked
    against its ``Source`` **here**, at index-build time, so a mismatched pair
    fails once by name rather than at the first epoch that reaches it.
    """
    items = []
    for entry in _as_manifest(manifest).entries(split=split):
        check_pairing(entry)
        series = entry.time_series(**read_kwargs)
        target = entry.target_time_series(**read_kwargs) if entry.target else None
        for step in range(len(series)):
            items.append((entry.id, series, target, step))
    return items


def _read_grid_sample(series, target, step, grid_kwargs):
    """One or two mesh reads -> ``(time, GridSample)``.

    :func:`_read_sample`'s sibling and the single owner of the grid
    read-and-sample step. A self-supervised entry (no ``Target``) reads **one**
    mesh; a paired one reads two. Nothing is cached between steps, so the
    streaming invariant holds either way.
    """
    time, mesh = series[step]
    target_mesh = target[step][1] if target is not None else None
    return time, grid_sample_pair(mesh, target_mesh=target_mesh, **grid_kwargs)


def iter_grid_samples(manifest, *, split=None, **kwargs):
    """Yield ``(entry_id, time, GridSample)`` over a manifest's entries.

    :func:`iter_samples`' grid counterpart, honouring the same streaming
    invariant: one mesh alive per yielded sample, two for a paired entry.
    """
    grid_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _GRID_KWARGS}
    read_kwargs = kwargs
    for entry_id, series, target, step in _grid_flat_items(
        manifest, split, read_kwargs
    ):
        time, sample = _read_grid_sample(series, target, step, grid_kwargs)
        yield entry_id, time, sample


def grid_stats(manifest, *, split=None, **kwargs):
    """Per-channel mean/std over a manifest's **grids**, in the same key
    convention as :func:`field_stats`.

    Deliberately a separate function rather than a mode of :func:`field_stats`,
    and the distinction is not cosmetic: a grid's statistics include the
    **fill** wherever the lattice reaches outside the mesh, so they are a
    different number from the nodal ones. Normalizing a grid with node stats
    would be silently wrong, which is exactly the kind of error a model absorbs
    without complaining.

    Returns ``{"x_mean", "x_std", "y_mean", "y_std", "coverage"}`` -- one value
    per channel, plus the mean coverage over the dataset, since a dataset whose
    grids are mostly fill is a dataset a model will learn the fill from.
    """
    x_acc = y_acc = None
    x_channels = y_channels = None
    coverage, seen = 0.0, 0
    for _, _, sample in iter_grid_samples(manifest, split=split, **kwargs):
        for key, channels in (("x", "x_channels"), ("y", "y_channels")):
            if key not in sample.arrays:
                continue
            names = list(getattr(sample, channels))
            rows = np.asarray(sample.arrays[key], dtype=np.float64)
            rows = rows.reshape(rows.shape[0], -1).T  # (samples, channels)
            if key == "x":
                if x_channels is None:
                    x_channels, x_acc = names, _Moments(len(names))
                elif names != x_channels:
                    raise ValueError(
                        "meshio++: grid_stats: the x channels changed across the "
                        f"dataset ({x_channels} -> {names})"
                    )
                x_acc.add(rows)
            else:
                if y_channels is None:
                    y_channels, y_acc = names, _Moments(len(names))
                elif names != y_channels:
                    raise ValueError(
                        "meshio++: grid_stats: the y channels changed across the "
                        f"dataset ({y_channels} -> {names})"
                    )
                y_acc.add(rows)
        cov = sample.schema.get("x_coverage")
        if cov is not None:
            coverage += float(cov)
            seen += 1
    stats = {
        "x_mean": [] if x_acc is None else x_acc.mean().tolist(),
        "x_std": [] if x_acc is None else x_acc.std().tolist(),
        "y_mean": [] if y_acc is None else y_acc.mean().tolist(),
        "y_std": [] if y_acc is None else y_acc.std().tolist(),
        "coverage": (coverage / seen) if seen else None,
    }
    return stats


class _Moments:
    """Streaming per-component mean/std, the PhysicsNeMo convention:
    ``std = sqrt(mean(x^2) - mean(x)^2)`` accumulated in float64."""

    def __init__(self, width):
        self.count = 0
        self.total = np.zeros(width, dtype=np.float64)
        self.total_sq = np.zeros(width, dtype=np.float64)

    def add(self, rows):
        rows = np.asarray(rows, dtype=np.float64)
        self.count += rows.shape[0]
        self.total += rows.sum(axis=0)
        self.total_sq += np.square(rows).sum(axis=0)

    def mean(self):
        return self.total / max(self.count, 1)

    def std(self):
        variance = self.total_sq / max(self.count, 1) - np.square(self.mean())
        return np.sqrt(np.maximum(variance, 0.0))


def field_stats(
    manifest, *, split=None, location="point", fields=None, delta=False, **read_kwargs
):
    """Per-field normalization stats over a manifest, streaming.

    Returns ``{"<field>_mean": [...], "<field>_std": [...]}`` with one value
    per component -- the Gen-1 ``node_stats.json`` key convention, directly
    usable by PhysicsNeMo's datapipes and by its Gen-2 ``Normalize``
    transform. Plain JSON away from disk: ``json.dump(stats, fh)``. One mesh
    alive at a time; the multi-component grouping comes from the
    ``feature_matrix`` schema, so it is the same rule as the columns.

    ``delta`` accumulates the statistics of **consecutive-step differences**
    instead -- what normalizing ``graph_sample(target_delta=True)`` targets
    needs -- under ``{field}_diff_mean``/``{field}_diff_std`` keys (still
    plain JSON, no new stats file format). ``True`` means lag 1; an int
    ``n >= 1`` is the lag matching ``target_offset=n``. Only the previous
    feature *matrices* (numpy) are held between steps, so one mesh stays
    alive; entries with too few steps contribute nothing.
    """
    # Identity checks, not `in`: 0 == False in Python, and a `delta=0` almost
    # certainly meant something -- name it rather than silently doing nothing.
    if delta is False or delta is None:
        lag = 0
    else:
        lag = 1 if delta is True else int(delta)
        if lag < 1:
            raise ValueError(
                f"meshio++: field_stats: delta must be True or an int >= 1, "
                f"not {delta!r}"
            )
    suffix = "_diff" if lag else ""
    accumulators = {}
    order = []
    for entry in _as_manifest(manifest).entries(split=split):
        previous = []  # the last `lag` feature matrices (numpy, never meshes)
        for _, mesh in entry.time_series(**read_kwargs):
            fm = feature_matrix(
                mesh, location, fields=fields, coords=False, regions=False
            )
            if lag:
                previous.append(fm)
                if len(previous) <= lag:
                    continue
                oldest = previous.pop(0)
                if oldest.columns != fm.columns:
                    raise ValueError(
                        f"meshio++: field_stats: entry '{entry.id}' changes "
                        f"columns across steps ({oldest.columns} vs "
                        f"{fm.columns}) -- delta stats need a stable field set"
                    )
                if oldest.matrix.shape[0] != fm.matrix.shape[0]:
                    raise ValueError(
                        f"meshio++: field_stats: entry '{entry.id}' changes "
                        f"row count across steps ({oldest.matrix.shape[0]} vs "
                        f"{fm.matrix.shape[0]}) -- delta stats need a stable mesh"
                    )
                matrix = fm.matrix - oldest.matrix
            else:
                matrix = fm.matrix
            groups = {}
            for column, source in zip(fm.columns, fm.schema["sources"]):
                groups.setdefault(source["source"], []).append(column)
            for name, columns in groups.items():
                idx = [fm.columns.index(c) for c in columns]
                acc = accumulators.get(name)
                if acc is None:
                    acc = accumulators[name] = _Moments(len(idx))
                    order.append(name)
                elif acc.total.shape[0] != len(idx):
                    raise ValueError(
                        f"meshio++: field_stats: '{name}' changes component "
                        f"count across the dataset ({acc.total.shape[0]} vs "
                        f"{len(idx)})"
                    )
                acc.add(matrix[:, idx])
    stats = {}
    for name in order:
        acc = accumulators[name]
        stats[f"{name}{suffix}_mean"] = acc.mean().tolist()
        stats[f"{name}{suffix}_std"] = acc.std().tolist()
    return stats


def edge_stats(
    manifest,
    *,
    split=None,
    kind="node",
    undirected=True,
    proximity=None,
    world_edges=None,
    **read_kwargs,
):
    """Edge-feature normalization stats over a manifest, streaming.

    Returns ``{"edge_mean": [...], "edge_std": [...]}`` per component of the
    ``(disp, norm)`` edge attribute -- the Gen-1 ``edge_stats.json`` key
    convention -- plus ``world_edge_mean``/``world_edge_std`` when
    ``world_edges`` is given.

    ``proximity``/``world_edges`` must be the SAME options training will use:
    the edges are rebuilt here, through the same :func:`_sample_edges`, and
    statistics gathered over a different graph normalize the wrong thing.
    """
    acc = None
    world_acc = None
    box = None if proximity is None else proximity.get("box_size")
    world_box = None if world_edges is None else world_edges.get("box_size")

    def _fold(accumulator, attr, name):
        if accumulator is None:
            accumulator = _Moments(attr.shape[1])
        elif accumulator.total.shape[0] != attr.shape[1]:
            raise ValueError(
                f"meshio++: edge_stats: the {name} width changes across the "
                f"dataset ({accumulator.total.shape[0]} vs {attr.shape[1]})"
            )
        accumulator.add(attr)
        return accumulator

    for entry in _as_manifest(manifest).entries(split=split):
        for _, mesh in entry.time_series(**read_kwargs):
            pos = _positions(mesh, kind)
            ei, world_ei = _sample_edges(mesh, kind, undirected, proximity, world_edges)
            acc = _fold(acc, _edge_attributes(pos, ei, box), "edge-attribute")
            if world_ei is not None:
                world_acc = _fold(
                    world_acc,
                    _edge_attributes(pos, world_ei, world_box),
                    "world-edge-attribute",
                )
    out = (
        {"edge_mean": [], "edge_std": []}
        if acc is None
        else {"edge_mean": acc.mean().tolist(), "edge_std": acc.std().tolist()}
    )
    if world_edges is not None:
        out["world_edge_mean"] = [] if world_acc is None else world_acc.mean().tolist()
        out["world_edge_std"] = [] if world_acc is None else world_acc.std().tolist()
    return out


# --------------------------------------------------------------------------- #
# Framework objects (lazy factories over the gated modules)                   #
# --------------------------------------------------------------------------- #
def make_reader(manifest, *, split=None, **kwargs):
    """A PhysicsNeMo Gen-2 ``Reader`` over a manifest: ``__len__`` +
    ``_load_sample(i) -> dict[str, torch.Tensor]`` (the sanctioned datapipe
    extension point). Needs ``nvidia-physicsnemo``; raises a named install
    error otherwise. ``kwargs`` as in :func:`iter_samples`."""
    _require_framework(
        "make_reader", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    from ._reader import MeshManifestReader

    return MeshManifestReader(manifest, split=split, **kwargs)


# --------------------------------------------------------------------------- #
# The physicsnemo.mesh.Mesh bridge (v9.30.0)                                  #
# --------------------------------------------------------------------------- #
#: physicsnemo.mesh.Mesh holds exactly ONE simplex kind, named by the cells
#: tensor's trailing dim (n_manifold_dims = k - 1).
_SIMPLEX_BY_DIM = {0: "vertex", 1: "line", 2: "triangle", 3: "tetra"}
_TYPE_BY_WIDTH = {1: "vertex", 2: "line", 3: "triangle", 4: "tetra"}


def _to_physicsnemo_payload(mesh, *, manifold_dim="auto", float32=True):
    """The pure half of :func:`to_physicsnemo`: plain numpy, no torch.

    ``physicsnemo.mesh.Mesh`` is single-manifold-dim and simplicial-only (its
    own docs: subdivide non-simplicial elements before use), so this selects
    ONE topological dimension, tessellates it with the existing
    ``convert_cells`` operations (which replicate ``cell_data`` rows to
    children natively), and drops everything the target cannot hold -- each
    drop recorded in ``notes``, never silent. Every returned buffer is
    freshly owned: upstream in-place edits invalidate that Mesh's caches, so
    sharing memory with the source mesh would be a spooky-action bug.
    """
    from .._convert_cells import convert_cells

    float_dtype = np.float32 if float32 else np.float64
    notes = []

    dims = sorted({blk.dim for blk in mesh.cells})
    if manifold_dim == "auto":
        target = dims[-1] if dims else 0
    else:
        target = int(manifold_dim)
        if target not in _SIMPLEX_BY_DIM:
            raise ValueError(
                f"meshio++: to_physicsnemo: manifold_dim must be 'auto' or "
                f"0..3, not {manifold_dim!r}"
            )
        if dims and target not in dims:
            raise ValueError(
                f"meshio++: to_physicsnemo: the mesh has no dimension-{target} "
                f"cells (dimensions present: {dims or 'none'})"
            )
    spatial = mesh.points.shape[1] if mesh.points.ndim == 2 else 0
    if target > spatial:
        raise ValueError(
            f"meshio++: to_physicsnemo: manifold dimension {target} exceeds "
            f"the spatial dimension {spatial} -- a tetra Mesh needs 3-D "
            f"points (physicsnemo.mesh.Mesh's own constraint)"
        )

    simplex = _SIMPLEX_BY_DIM[target]
    converted = mesh
    if dims and any(
        blk.dim == target and blk.type != simplex for blk in converted.cells
    ):
        # Higher-order first, then decompose -- both idempotent on cells they
        # do not apply to, and both replicate cell_data rows to children.
        converted = convert_cells(converted, mode="linearize")
        converted = convert_cells(converted, mode="simplexify")
        notes.append(
            f"non-{simplex} dimension-{target} cells were tessellated "
            f"(linearize + simplexify); cell_data rows replicated to children"
        )

    kept = [
        i
        for i, blk in enumerate(converted.cells)
        if blk.dim == target and blk.type == simplex
    ]
    dropped = [
        (blk.type, len(blk.data))
        for i, blk in enumerate(converted.cells)
        if i not in kept
    ]
    if dropped:
        listing = ", ".join(f"{t} x {n}" for t, n in dropped)
        notes.append(
            f"physicsnemo.mesh.Mesh is single-manifold-dim; dropped {listing} "
            f"-- export each dimension as its own Mesh"
        )

    if kept:
        cells = np.ascontiguousarray(
            np.concatenate([np.asarray(converted.cells[i].data) for i in kept]),
            dtype=np.int64,
        )
    else:
        cells = None
        if dims:
            notes.append("no cells at the selected dimension survive; points only")

    cell_data = {}
    for name, arrays in converted.cell_data.items():
        rows = [np.asarray(arrays[i]) for i in kept]
        if not rows:
            continue
        if not all(r.dtype.kind in "fiub" for r in rows):
            notes.append(f"cell_data '{name}' is non-numeric and was dropped")
            continue
        cell_data[name] = np.ascontiguousarray(np.concatenate(rows))

    point_data = {}
    for name, array in converted.point_data.items():
        array = np.asarray(array)
        if array.dtype.kind not in "fiub":
            notes.append(f"point_data '{name}' is non-numeric and was dropped")
            continue
        point_data[name] = np.array(array, copy=True)

    global_data = {}
    for name, value in converted.field_data.items():
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray(None)
        if array.dtype.kind not in "fiub":
            notes.append(f"field_data '{name}' is non-numeric and was dropped")
            continue
        global_data[name] = np.array(array, copy=True)

    if getattr(converted, "regions", None):
        names = sorted({r.name for r in converted.regions})
        notes.append(
            f"regions {names} have no physicsnemo.mesh slot and were dropped "
            f"-- encode membership as features via graph_sample's one-hots"
        )

    return {
        "points": np.array(converted.points, dtype=float_dtype, copy=True),
        "cells": cells,
        "point_data": point_data,
        "cell_data": cell_data,
        "global_data": global_data,
        "manifold_dim": target,
        "notes": notes,
    }


def _from_physicsnemo_payload(payload):
    """The pure half of :func:`from_physicsnemo`: numpy payload -> Mesh."""
    from .._mesh import Mesh

    cells = payload.get("cells")
    blocks = []
    if cells is not None and len(cells):
        cells = np.ascontiguousarray(np.asarray(cells), dtype=np.int64)
        width = cells.shape[1] if cells.ndim == 2 else 0
        if width not in _TYPE_BY_WIDTH:
            raise ValueError(
                f"meshio++: from_physicsnemo: cells with {width} nodes per "
                f"row map to no simplex (vertex/line/triangle/tetra)"
            )
        blocks = [(_TYPE_BY_WIDTH[width], cells)]
    cell_data = {
        name: [np.asarray(array)]
        for name, array in payload.get("cell_data", {}).items()
    }
    return Mesh(
        np.asarray(payload["points"]),
        blocks,
        point_data={
            name: np.asarray(a) for name, a in payload.get("point_data", {}).items()
        },
        cell_data=cell_data if blocks else {},
        field_data={
            name: np.asarray(a) for name, a in payload.get("global_data", {}).items()
        },
    )


def to_physicsnemo(mesh, *, manifold_dim="auto", float32=True):
    """A meshio++ mesh as a ``physicsnemo.mesh.Mesh`` (CPU tensors).

    The target holds exactly one simplex kind: ``manifold_dim`` selects it
    (``"auto"`` = the highest dimension present), non-simplex cells at that
    dimension are tessellated through the existing ``convert_cells``
    operations, and everything the target cannot hold (other-dim blocks,
    regions, non-numeric data) is dropped **with a warning**, never
    silently. Points follow the upstream float32 convention
    (``float32=False`` keeps float64); cells are int64; move to a device
    with the upstream ``Mesh.to(device)``.

    Needs ``nvidia-physicsnemo`` (no pip extra, deliberately); pin
    ``>=2.1,<2.2`` in training projects -- ``physicsnemo.mesh`` is the
    framework's newest surface. See ``doc/physicsnemo.md``.
    """
    _require_framework(
        "to_physicsnemo", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    payload = _to_physicsnemo_payload(mesh, manifold_dim=manifold_dim, float32=float32)
    _emit("to_physicsnemo", payload["notes"])
    from ._mesh import to_mesh

    return to_mesh(payload)


def from_physicsnemo(pm):
    """A ``physicsnemo.mesh.Mesh`` back as a meshio++ mesh.

    The cells tensor's trailing dim names the one simplex block
    (vertex/line/triangle/tetra); ``point_data``/``cell_data`` leaves come
    back as numpy arrays and ``global_data`` lands in ``field_data``.
    Needs ``nvidia-physicsnemo``; see :func:`to_physicsnemo`.
    """
    _require_framework(
        "from_physicsnemo", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    from ._mesh import from_mesh

    return _from_physicsnemo_payload(from_mesh(pm))


def make_dataset(manifest, *, split=None, device=None, **kwargs):
    """A ``torch.utils.data.Dataset`` of ``torch_geometric.data.Data``
    (``pos``/``x``/``y``/``edge_index``/``edge_attr``) over a manifest --
    what MeshGraphNet training consumes, batched natively by PyG's
    ``DataLoader``. Needs ``torch_geometric``; raises a named install error
    otherwise. ``device=`` moves each sample's tensors on access."""
    _require_framework(
        "make_dataset", "torch_geometric", "pip install torch_geometric", doc=_DOC
    )
    from ._pyg import MeshManifestDataset

    return MeshManifestDataset(manifest, split=split, device=device, **kwargs)


# --------------------------------------------------------------------------- #
# Training and prediction (v10.24.0; the gated module is ``train.py``)        #
# --------------------------------------------------------------------------- #
def run_training(spec, *, log=print):
    """Train MeshGraphNet per a :class:`TrainSpec` (or its dict / JSON text /
    file path) into the spec's run directory; returns the final progress
    record. The same thing ``python -m meshioplusplus.physicsnemo.train
    --spec`` and the dashboard's ``train_start`` run. Needs ``torch_geometric``
    and ``nvidia-physicsnemo``; raises a named install error otherwise.
    Named ``run_training`` rather than ``train`` so importing the ``train``
    submodule can never shadow it. See ``doc/physicsnemo.md``."""
    _require_framework(
        "run_training", "torch_geometric", "pip install torch_geometric", doc=_DOC
    )
    from .train import run

    return run(spec, log=log)


def predict(
    checkpoint,
    manifest,
    *,
    entry_ids=None,
    split="test",
    step=0,
    output_dir,
    device="auto",
):
    """Predict with a trained ``.mdlus`` checkpoint (and its model card) over
    a manifest's entries, writing ``<column>_pred``/``<column>_error`` back
    as data arrays into ``output_dir/<entry_id>.vtu``; returns one row per
    entry (``entry_id``, ``output_path``, ``rmse``, ``max_error``). Needs
    ``nvidia-physicsnemo`` and ``torch_geometric``."""
    _require_framework(
        "predict", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    from .train import predict as _predict

    return _predict(
        checkpoint,
        manifest,
        entry_ids=entry_ids,
        split=split,
        step=step,
        output_dir=output_dir,
        device=device,
    )


def predict_mesh(checkpoint, mesh, *, target_mesh=None, device="auto", label="mesh"):
    """Predict with a trained ``.mdlus`` checkpoint on ONE in-memory mesh,
    returning ``(mesh, row)`` -- the mesh carrying ``<column>_pred`` (and
    ``<column>_error`` where the truth is present) plus a report row. The
    card says which family wrote the checkpoint, so a caller does not have
    to, and nothing here consults a manifest. Needs ``nvidia-physicsnemo``
    (and ``torch_geometric`` for a graph checkpoint)."""
    _require_framework(
        "predict_mesh", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    from .train import predict_mesh as _predict_mesh

    return _predict_mesh(
        checkpoint, mesh, target_mesh=target_mesh, device=device, label=label
    )


def predict_file(
    checkpoint,
    input_path,
    output_path,
    *,
    time_step=None,
    target_path=None,
    input_format=None,
    output_format=None,
    device="auto",
):
    """Predict with a trained ``.mdlus`` checkpoint on ONE mesh file, writing
    the result. The single-mesh counterpart of :func:`predict`: no manifest,
    no split, no entry -- point a checkpoint at a file that was never
    catalogued and get the prediction written back. A file carrying no truth
    predicts anyway, with ``rmse``/``max_error`` reported ``None``. Needs
    ``nvidia-physicsnemo`` (and ``torch_geometric`` for a graph checkpoint)."""
    _require_framework(
        "predict_file", "physicsnemo", "pip install nvidia-physicsnemo", doc=_DOC
    )
    from .train import predict_file as _predict_file

    return _predict_file(
        checkpoint,
        input_path,
        output_path,
        time_step=time_step,
        target_path=target_path,
        input_format=input_format,
        output_format=output_format,
        device=device,
    )
