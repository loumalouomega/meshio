"""Machine-learning data handling: graphs, feature matrices, datasets.

The three features here close roadmap §1 together with the v9.26.0 frames and
the tensor handoff in :mod:`meshioplusplus._gpu`:

* :func:`edge_index` — the mesh as a graph, in the ``(2, E)`` int64 layout
  PyTorch Geometric and DGL expect. Node graph (unique mesh edges) or
  facet-sharing cell dual.
* :func:`feature_matrix` — a canonical per-node / per-cell feature matrix
  (coordinates, selected fields, region one-hots) with a **stable, recorded
  column order**. The :class:`FeatureMatrix` payload's ``columns`` tuple is the
  contract: record it at training time, assert equality at inference time, and
  the two can never silently disagree.
* :func:`write_dataset` — a *set* of meshes (a glob, a directory listing, a
  transient series) exported as one on-disk dataset keyed by ``mesh_id``:
  hive-partitioned Parquet (the format a training loop's dataloader wants),
  or chunked Zarr / HDF5 groups for datasets too large to hold in memory.

Everything is pure Python over existing machinery — the table payload and
suffix-expansion rules from :mod:`meshioplusplus._interop`, the edge table from
:mod:`meshioplusplus._smooth` (the C++ ``detail/node_adjacency.hpp`` twin), the
facet-dual from :mod:`meshioplusplus._partition`, and the sequence plan from
:mod:`meshioplusplus._sequence` (whose streaming invariant — one mesh alive at
a time — :func:`write_dataset` preserves). The C++/WASM/C/Fortran core is
untouched. Optional libraries (pyarrow, zarr, h5py) are imported lazily inside
the entry points, the interop convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from ._common import warn
from ._interop import _emit, _frame_columns, _importable, _require, _to_table_payload
from ._regions import block_bases
from ._smooth import _REFINE_EDGES

#: Version of the feature-matrix column-order contract. Bumped only when the
#: rules that determine column names/order change — a consumer comparing
#: recorded ``columns`` catches any drift regardless, but the explicit version
#: makes "trained under a different contract" a checkable fact.
FEATURE_SCHEMA_VERSION = 1

#: Version of the on-disk dataset layout written by :func:`write_dataset`.
DATASET_LAYOUT_VERSION = 1

#: Name of the JSON manifest :func:`write_dataset` writes beside a Parquet
#: dataset (and stores in the root attrs of a Zarr/HDF5 one).
DATASET_MANIFEST_NAME = "meshioplusplus_dataset.json"


# --------------------------------------------------------------------------- #
# graph export                                                                #
# --------------------------------------------------------------------------- #
def edge_index(mesh, kind: str = "node", *, undirected: bool = True):
    """The mesh as a graph, in the layout GNN frameworks expect.

    Returns a ``(2, E)`` C-contiguous int64 array — rows are source and target
    indices, directly usable as PyTorch Geometric's / DGL's ``edge_index``.

    ``kind="node"`` (default): vertices are mesh points, edges are the unique
    cell edges — the same edge topology :func:`~meshioplusplus.smooth` relaxes
    over (``_REFINE_EDGES``, the twin of the C++ ``detail/node_adjacency.hpp``
    Edge kind). Ragged polygon blocks contribute their closed rings. Cell types
    with no known edge topology (the higher-order family, VTK-Lagrange,
    ``custom``) contribute nothing, with one warning naming them — linearize
    first (:func:`~meshioplusplus.convert_cells`) to keep their corners.

    ``kind="cell"``: vertices are cells in the **global block-major** numbering
    (the one regions and ``partition_labels`` use), edges join cells sharing a
    facet — faces in 3D, edges in 2D. Ragged/polyhedron blocks raise
    ``NotImplementedError`` (the dual-graph helper's documented limit).

    ``undirected=True`` (default) emits **both** directions of every edge —
    the convention PyG uses for undirected graphs — sorted lexicographically
    by (source, target). ``undirected=False`` emits each edge once with
    ``source < target``.
    """
    if kind == "node":
        pairs = _node_edge_pairs(mesh)
    elif kind == "cell":
        pairs = _cell_dual_pairs(mesh)
    else:
        raise ValueError(
            f"meshio++: edge_index: unknown kind '{kind}' (expected node or cell)"
        )

    return _canonical_edges(pairs[0], pairs[1], undirected)


def _canonical_edges(a, b, undirected):
    """Raw endpoint arrays -> the canonical ``(2, E)`` int64 edge index.

    De-duplicated on the unordered pair, then emitted either once per edge
    with ``source < target`` or in both directions, lexsorted by (source,
    target). The single owner of that layout: :func:`edge_index` and
    :func:`~meshioplusplus.proximity_graph` both go through it, so a mesh graph
    and a proximity graph over the same pair set are byte-identical.
    """
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    order = np.lexsort((hi, lo))
    lo, hi = lo[order], hi[order]
    if lo.size:
        uniq = np.ones(lo.size, dtype=bool)
        uniq[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
        lo, hi = lo[uniq], hi[uniq]

    if not undirected:
        return np.ascontiguousarray(np.vstack([lo, hi]))
    src = np.concatenate([lo, hi])
    dst = np.concatenate([hi, lo])
    order = np.lexsort((dst, src))
    return np.ascontiguousarray(np.vstack([src[order], dst[order]]))


def _node_edge_pairs(mesh):
    """Raw (a, b) endpoint arrays of every cell edge, unfiltered duplicates.

    The collection rules are ``_smooth._edge_graph``'s, deliberately: same
    table, same closed-ring treatment of ragged polygons, same silence on
    unknown types — except the silence is replaced by one warning, since a
    missing graph edge is a wrong training input, not just a pinned node.
    """
    n = len(mesh.points)
    src, dst = [], []
    skipped = []
    for block in mesh.cells:
        data = block.data
        rectangular = isinstance(data, np.ndarray) and data.ndim == 2
        if block.type in _REFINE_EDGES and rectangular:
            for a, b in _REFINE_EDGES[block.type]:
                src.append(np.asarray(data[:, a], dtype=np.int64).ravel())
                dst.append(np.asarray(data[:, b], dtype=np.int64).ravel())
        elif block.type.startswith("polyhedron"):
            # A cell -> faces -> node-ids nesting; its rows are not rings.
            skipped.append(block.type)
        elif block.type.startswith("polygon") or not rectangular:
            # Closed rings. Tested by TYPE, not raggedness: a uniform n-gon
            # block stores rectangularly (the cgns.cpp:476 lesson) and would
            # otherwise be silently skipped.
            for row in data:
                row = np.asarray(row, dtype=np.int64).ravel()
                if row.size >= 2:
                    src.append(row)
                    dst.append(np.roll(row, -1))
        else:
            skipped.append(block.type)
    if skipped:
        warn(
            "edge_index: no edge topology for cell type(s) "
            f"{sorted(set(skipped))}; they contribute no edges — linearize "
            "with convert_cells to keep their corners"
        )
    if not src:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    a = np.concatenate(src)
    b = np.concatenate(dst)
    keep = (a >= 0) & (a < n) & (b >= 0) & (b < n) & (a != b)
    return a[keep], b[keep]


def _cell_dual_pairs(mesh):
    """Facet-sharing cell-dual edges as raw (a, b) arrays."""
    from ._partition import _dual_graph

    total = int(block_bases(mesh.cells)[-1])
    xadj, adjncy = _dual_graph(mesh, total)
    counts = np.diff(np.asarray(xadj, dtype=np.int64))
    a = np.repeat(np.arange(total, dtype=np.int64), counts)
    b = np.asarray(adjncy, dtype=np.int64)
    return a, b


# --------------------------------------------------------------------------- #
# feature matrix                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureMatrix:
    """A feature matrix plus the contract that makes it reproducible.

    ``columns`` is the recorded column order — the whole value of this payload.
    Record it (or ``schema``) at training time; at inference time build the
    matrix the same way and ``assert fm.columns == recorded`` before the model
    sees a single row. ``schema`` additionally carries the contract version and
    a per-column source description.
    """

    #: ``(N, F)`` float64, C-contiguous.
    matrix: np.ndarray
    #: final flat column names, in matrix order — the contract.
    columns: tuple
    #: ``{"feature_schema_version", "meshioplusplus_version", "location",
    #:   "columns", "sources"}`` — JSON-serializable.
    schema: dict


def feature_matrix(
    mesh,
    location: str = "point",
    *,
    fields=None,
    coords: bool = True,
    regions=True,
):
    """Assemble a canonical per-node / per-cell feature matrix.

    Column order rule (this IS the contract, versioned as
    ``FEATURE_SCHEMA_VERSION``):

    1. coordinates — ``x``/``y``/``z`` (as many as the mesh has), point
       location only, skipped with ``coords=False``;
    2. data arrays — ``fields=None`` means every array at that location in
       sorted-name order; an explicit list is honoured **in the given order**
       (it is recorded either way). Multi-component arrays expand into the
       suffixed columns the pandas export uses (``v`` → ``v_0``/``v_1``/…);
    3. region one-hots — 0/1 membership columns ``region:<name>`` for each
       region of the matching kind (point/cell; ``side`` never), in the
       deterministic ``Region.key`` order. ``regions=False`` skips them, a
       list of names selects.

    Everything is cast to float64. The structural cell-table columns
    (``block``/``cell_type``/``cell``) are identifiers, not features, and are
    never included. Object-dtype arrays are dropped by the table payload with
    a warning, as everywhere else.

    Returns a :class:`FeatureMatrix`.
    """
    payload = _to_table_payload(mesh, location, op="feature_matrix")

    data_cols = [c for c in payload.columns if not c.derived]
    if fields is not None:
        by_name = {c.name: c for c in data_cols}
        missing = [name for name in fields if name not in by_name]
        if missing:
            raise ValueError(
                f"meshio++: feature_matrix: no {location} data array named "
                f"{missing} (available: {sorted(by_name)})"
            )
        data_cols = [by_name[name] for name in fields]

    selected = []
    if location == "point" and coords:
        selected.extend(c for c in payload.columns if c.derived)
    selected.extend(data_cols)

    # Reuse the pandas suffix expansion so `v` -> v_0/v_1/v_2 is ONE rule
    # repo-wide, collision handling included.
    from ._interop import TablePayload

    flat, groups = _frame_columns(
        TablePayload(location=location, columns=selected, notes=payload.notes)
    )

    columns = []
    sources = []
    parts = []
    expanded = {name for emitted in groups.values() for name in emitted}
    for col in flat:
        kind = "coords" if col.derived else "data"
        source = col.name
        component = None
        if col.name in expanded:
            source, _, comp = col.name.rpartition("_")
            component = int(comp)
        columns.append(col.name)
        sources.append(
            {"name": col.name, "source": source, "kind": kind, "component": component}
        )
        parts.append(np.asarray(col.values, dtype=np.float64))

    num_rows = payload.num_rows
    region_kind = location
    wanted = regions if isinstance(regions, (list, tuple, set)) else None
    if regions:
        for region in sorted(
            (r for r in getattr(mesh, "regions", []) if r.kind == region_kind),
            key=lambda r: r.key,
        ):
            if wanted is not None and region.name not in wanted:
                continue
            mask = np.zeros(num_rows, dtype=np.float64)
            entries = np.asarray(region.entries, dtype=np.int64)
            entries = entries[(entries >= 0) & (entries < num_rows)]
            mask[entries] = 1.0
            name = f"region:{region.name}"
            columns.append(name)
            sources.append(
                {
                    "name": name,
                    "source": region.name,
                    "kind": "region",
                    "component": None,
                }
            )
            parts.append(mask)

    if parts:
        matrix = np.ascontiguousarray(np.column_stack(parts))
    else:
        matrix = np.zeros((num_rows, 0), dtype=np.float64)
    _emit("feature_matrix", payload.notes)

    from .__about__ import __version__

    schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "meshioplusplus_version": str(__version__),
        "location": location,
        "columns": list(columns),
        "sources": sources,
    }
    return FeatureMatrix(matrix=matrix, columns=tuple(columns), schema=schema)


# --------------------------------------------------------------------------- #
# dataset export                                                              #
# --------------------------------------------------------------------------- #
def has_zarr() -> bool:
    """Whether zarr is installed (``pip install meshioplusplus[zarr]``)."""
    return _importable("zarr")


def write_dataset(
    source,
    path,
    *,
    location: str = "point",
    format: str = "parquet",
    mesh_id: str = "stem",
    file_format=None,
    times=None,
    time_from: str = "auto",
    sort: bool = False,
):
    """Export a *set* of meshes as one on-disk dataset keyed by ``mesh_id``.

    ``source`` is anything :func:`~meshioplusplus.sequence_entries` accepts —
    a glob pattern (natural-numeric ordering), a list of paths, or a
    multi-step file whose steps become entries. Meshes are read **streaming**
    (one alive at a time, the sequence invariant), each exported as one table
    of the chosen data ``location`` exactly as :func:`~meshioplusplus.to_arrow`
    would build it.

    The **schema is strict**: the first mesh defines the column names and
    dtypes, and a later mesh that disagrees is a named error rather than a
    silently widened union — the whole point of a dataset is that every row
    group means the same thing.

    ``mesh_id="stem"`` uses each file's stem (multi-step files append
    ``_<step>``; a duplicate id is an error naming both paths);
    ``"index"`` uses the zero-padded entry index.

    Layouts (``DATASET_LAYOUT_VERSION`` 1):

    * ``format="parquet"`` (needs ``[arrow]``): a directory in hive layout —
      ``path/mesh_id=<id>/data.parquet``, one file per mesh, each carrying a
      ``mesh_id`` string column plus the usual ``meshioplusplus:*`` schema
      metadata and per-file ``meshioplusplus:mesh_id`` / ``:time`` keys. A
      ``meshioplusplus_dataset.json`` manifest at the root records the layout
      version, column order and every entry (id, source path, time, rows).
      Read it back with ``pyarrow.dataset.dataset(path, partitioning="hive")``,
      ``pandas.read_parquet(path)`` or any Parquet-aware engine.
    * ``format="zarr"`` (needs ``[zarr]``) / ``format="hdf5"`` (needs h5py,
      the ``[all]`` extra): one root group whose attrs carry the same manifest
      JSON, one subgroup per ``mesh_id`` with one chunked dataset per column
      and per-group ``time`` / ``num_rows`` attrs. **Numeric columns only** —
      the object-dtype ``cell_type`` column is dropped with a warning (its
      information is in the per-group ``cell_types`` attr), which keeps the
      layout clear of the h5py/zarr variable-string dtype swamp.

    Returns the manifest dict.
    """
    from ._helpers import read
    from ._sequence import _resolve_time, sequence_entries

    if format not in ("parquet", "zarr", "hdf5"):
        raise ValueError(
            f"meshio++: write_dataset: unknown format '{format}' "
            "(expected parquet, zarr or hdf5)"
        )
    if mesh_id not in ("stem", "index"):
        raise ValueError(
            f"meshio++: write_dataset: unknown mesh_id rule '{mesh_id}' "
            "(expected stem or index)"
        )

    entries = sequence_entries(
        source, file_format=file_format, times=times, time_from=time_from, sort=sort
    )
    ids = _mesh_ids(entries, mesh_id)

    writer = {
        "parquet": _ParquetDatasetWriter,
        "zarr": _ZarrDatasetWriter,
        "hdf5": _Hdf5DatasetWriter,
    }[format](path, location)

    manifest_entries = []
    try:
        for entry, entry_id in zip(entries, ids):
            mesh = read(
                entry["path"],
                file_format=file_format or None,
                time_step=entry["step"],
            )
            time = _resolve_time(entry, mesh, time_from)
            num_rows = writer.write_mesh(entry_id, mesh, time)
            manifest_entries.append(
                {
                    "mesh_id": entry_id,
                    "source_path": str(entry["path"]),
                    "time": time,
                    "time_source": entry["time_source"],
                    "num_rows": num_rows,
                }
            )
            del mesh  # the streaming invariant: one mesh alive at a time
    except BaseException:
        writer.abort()  # close handles; a partial dataset gets NO manifest
        raise
    return writer.finalize(manifest_entries)


def _mesh_ids(entries, rule):
    """One id per entry, unique by construction or by named error."""
    if rule == "index":
        width = max(4, len(str(max(len(entries) - 1, 0))))
        return [f"{i:0{width}d}" for i in range(len(entries))]
    multi = {}
    for entry in entries:
        multi[entry["path"]] = multi.get(entry["path"], 0) + 1
    ids, seen = [], {}
    for entry in entries:
        stem = os.path.splitext(os.path.basename(entry["path"]))[0]
        if multi[entry["path"]] > 1:
            stem = f"{stem}_{entry['step']}"
        if stem in seen:
            raise ValueError(
                "meshio++: write_dataset: mesh_id 'stem' is not unique: "
                f"'{stem}' names both {seen[stem]} and {entry['path']} — "
                "use mesh_id='index'"
            )
        seen[stem] = entry["path"]
        ids.append(stem)
    return ids


class _DatasetWriterBase:
    """Shared schema strictness + manifest assembly for the three layouts."""

    format_name = ""

    def __init__(self, path, location):
        self.path = str(path)
        self.location = location
        self.columns = None  # (name, dtype-str) pairs; first mesh defines them

    def abort(self):
        """Best-effort cleanup after a failed write; no manifest is emitted."""

    def check_schema(self, entry_id, columns):
        described = [(name, str(dtype)) for name, dtype in columns]
        if self.columns is None:
            self.columns = described
            return
        if described != self.columns:
            raise ValueError(
                f"meshio++: write_dataset: mesh '{entry_id}' does not match "
                f"the dataset schema: expected {self.columns}, got {described} "
                "— the first mesh defines the schema, and every mesh must "
                "carry the same columns"
            )

    def manifest(self, manifest_entries):
        from .__about__ import __version__

        return {
            "layout_version": DATASET_LAYOUT_VERSION,
            "meshioplusplus_version": str(__version__),
            "location": self.location,
            "columns": [name for name, _ in (self.columns or [])],
            "entries": manifest_entries,
        }


class _ParquetDatasetWriter(_DatasetWriterBase):
    format_name = "parquet"

    def __init__(self, path, location):
        super().__init__(path, location)
        _require("pyarrow", "arrow", "write_dataset")
        os.makedirs(self.path, exist_ok=True)

    def write_mesh(self, entry_id, mesh, time):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from ._interop import to_arrow

        table = to_arrow(mesh, location=self.location)
        self.check_schema(
            entry_id, [(n, table.schema.field(n).type) for n in table.schema.names]
        )
        ids = pa.array([entry_id] * table.num_rows, type=pa.string())
        table = table.append_column("mesh_id", ids)
        metadata = dict(table.schema.metadata or {})
        metadata[b"meshioplusplus:mesh_id"] = entry_id.encode()
        metadata[b"meshioplusplus:time"] = repr(float(time)).encode()
        table = table.replace_schema_metadata(metadata)
        part_dir = os.path.join(self.path, f"mesh_id={entry_id}")
        os.makedirs(part_dir, exist_ok=True)
        pq.write_table(table, os.path.join(part_dir, "data.parquet"))
        return table.num_rows

    def finalize(self, manifest_entries):
        manifest = self.manifest(manifest_entries)
        with open(os.path.join(self.path, DATASET_MANIFEST_NAME), "w") as handle:
            json.dump(manifest, handle, indent=1)
            handle.write("\n")
        return manifest


class _GroupDatasetWriter(_DatasetWriterBase):
    """Common column preparation for the Zarr/HDF5 group layouts."""

    def numeric_columns(self, mesh):
        payload = _to_table_payload(mesh, self.location, op="write_dataset")
        columns = []
        dropped = []
        for col in payload.columns:
            if col.values.dtype == object:
                dropped.append(col.name)
                continue
            columns.append(col)
        if dropped:
            warn(
                f"write_dataset: non-numeric column(s) {dropped} dropped from "
                f"the {self.format_name} layout (the information is in the "
                "group attrs)"
            )
        return payload, columns

    def group_attrs(self, payload, time):
        return {
            "time": float(time),
            "num_rows": int(payload.num_rows),
            "cell_types": payload.metadata["meshioplusplus:cell_types"],
        }


class _ZarrDatasetWriter(_GroupDatasetWriter):
    format_name = "zarr"

    def __init__(self, path, location):
        super().__init__(path, location)
        zarr = _require("zarr", "zarr", "write_dataset")
        self.root = zarr.open_group(str(path), mode="w")

    def write_mesh(self, entry_id, mesh, time):
        payload, columns = self.numeric_columns(mesh)
        self.check_schema(entry_id, [(c.name, c.values.dtype) for c in columns])
        group = self.root.create_group(entry_id)
        for col in columns:
            _zarr_write_array(group, col.name, col.values)
        for key, value in self.group_attrs(payload, time).items():
            group.attrs[key] = value
        return payload.num_rows

    def finalize(self, manifest_entries):
        manifest = self.manifest(manifest_entries)
        self.root.attrs["meshioplusplus_dataset"] = json.dumps(manifest)
        return manifest


def _zarr_write_array(group, name, values):
    """One column into a zarr group, across the zarr 2.x / 3.x API split."""
    if hasattr(group, "create_array"):  # zarr-python >= 3
        array = group.create_array(name, shape=values.shape, dtype=values.dtype)
        array[...] = values
    else:  # zarr-python 2.x
        group.array(name, values)


class _Hdf5DatasetWriter(_GroupDatasetWriter):
    format_name = "hdf5"

    def __init__(self, path, location):
        super().__init__(path, location)
        h5py = _require("h5py", "all", "write_dataset")
        self.file = h5py.File(str(path), "w")

    def abort(self):
        try:
            self.file.close()
        except Exception:
            pass

    def write_mesh(self, entry_id, mesh, time):
        payload, columns = self.numeric_columns(mesh)
        self.check_schema(entry_id, [(c.name, c.values.dtype) for c in columns])
        group = self.file.create_group(entry_id)
        for col in columns:
            group.create_dataset(
                col.name,
                data=col.values,
                chunks=True if col.values.size else None,
            )
        for key, value in self.group_attrs(payload, time).items():
            group.attrs[key] = value
        return payload.num_rows

    def finalize(self, manifest_entries):
        manifest = self.manifest(manifest_entries)
        self.file.attrs["meshioplusplus_dataset"] = json.dumps(manifest)
        self.file.close()
        return manifest


# `_column` is re-exported for the tests' convenience only; the public surface
# of this module is edge_index / feature_matrix / FeatureMatrix / write_dataset
# / has_zarr plus the three version/name constants above.
__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "DATASET_LAYOUT_VERSION",
    "DATASET_MANIFEST_NAME",
    "FeatureMatrix",
    "edge_index",
    "feature_matrix",
    "has_zarr",
    "write_dataset",
]
