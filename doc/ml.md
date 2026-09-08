# Machine-learning data handling

Turn meshes into the shapes an ML pipeline consumes — graphs, feature matrices, datasets, and framework tensors — without a bespoke glue script per project:

```python
import meshioplusplus as mio

mesh = mio.read("bracket.msh")

ei = mio.edge_index(mesh)                     # (2, E) int64 — PyG / DGL layout
fm = mio.feature_matrix(mesh, "point")        # (N, F) float64 + recorded columns
mio.write_dataset("out_*.vtu", "dataset/")    # partitioned Parquet, mesh_id keyed
t  = mio.to_torch(mesh)                       # torch tensors, adopted zero-copy
```

Everything here is pure Python over existing machinery — the [interop table payload](./interop), the smoothing layer's edge topology, the [sequence machinery](./sequences) — and the C++/WASM/C/Fortran core is untouched. Together with the v9.26.0 [pandas/polars frames](./interop#pandas-and-polars) and the [GPU handoff](./gpu), this closes the roadmap's machine-learning section.

## Graphs for GNNs: `edge_index`

```python
ei = mio.edge_index(mesh)                     # node graph, both directions
ei = mio.edge_index(mesh, undirected=False)   # unique pairs, source < target
ei = mio.edge_index(mesh, kind="cell")        # facet-sharing cell dual
```

Returns a `(2, E)` C-contiguous **int64** array — rows are source and target indices, exactly what PyTorch Geometric's and DGL's `edge_index` expect.

- **`kind="node"`** (default): vertices are mesh points, edges are the unique cell edges — the same edge topology `smooth` relaxes over (the Python twin of the C++ `detail/node_adjacency.hpp` Edge kind). Ragged *and* rectangular `polygon` blocks contribute their closed rings (routed by type name — a uniform n-gon block stores rectangularly). Cell types with no known edge topology (the higher-order family, VTK-Lagrange, `custom`, `polyhedron`) contribute nothing, with one warning naming them — linearize with `convert_cells` first to keep their corners.
- **`kind="cell"`**: vertices are cells in the **global block-major** numbering (the one regions and `partition_labels` use), edges join cells sharing a facet — faces in 3D, edges in 2D. Ragged/polyhedron blocks raise `NotImplementedError`.

`undirected=True` (default) emits **both** directions of every edge — the PyG convention for undirected graphs — sorted lexicographically by (source, target), so the output is deterministic and byte-identical across runs.

A graph built from *connectivity* is the wrong graph for a particle state, which has positions and no cells: there the interaction radius decides what a neighbour is. [`proximity_graph`](proximity_graphs.md) builds that one — radius or k-nearest-neighbour, with a periodic minimum-image mode — through the same canonicalization, so its output is byte-identical to `edge_index`'s for the same pair set.

```python
import torch
from torch_geometric.data import Data

fm = mio.feature_matrix(mesh, "point")
data = Data(
    x=torch.from_numpy(fm.matrix),
    edge_index=torch.from_numpy(mio.edge_index(mesh)),
)
```

## The feature matrix and its contract: `feature_matrix`

```python
fm = mio.feature_matrix(mesh, "point")
fm.matrix      # (N, F) float64, C-contiguous
fm.columns     # ('x', 'y', 'z', 'T', 'v_0', 'v_1', 'v_2', 'region:inlet')
fm.schema      # JSON-serializable: version, columns, per-column sources
```

**The column order is the whole value.** Training and inference silently disagreeing about which column is which is the classic mesh-ML failure mode, so the order is a stated, versioned rule (`FEATURE_SCHEMA_VERSION`, currently
1) and the payload records it:

1. **coordinates** — `x`/`y`/`z` (as many as the mesh has); point location only; `coords=False` skips them;
2. **data arrays** — `fields=None` means every array at that location in sorted-name order; an explicit `fields=[...]` list is honoured in the given order (and recorded either way). Multi-component arrays expand into the same suffixed columns the [pandas export](./interop#pandas-and-polars) uses (`v` → `v_0`/`v_1`/`v_2`) — one rule repo-wide;
3. **region one-hots** — 0/1 columns `region:<name>` for each region of the matching kind, in the deterministic `Region.key` order; `regions=False` skips them, a list of names selects.

Everything is float64. The cell table's structural columns (`block`/`cell_type`/`cell`) are identifiers, not features, and are never included.

At training time, store `fm.schema` (it is plain JSON). At inference time:

```python
fm = mio.feature_matrix(new_mesh, "point")
assert list(fm.columns) == trained_schema["columns"], "feature drift"
```

Derived quantities compose rather than being re-implemented: attach them first, then select them —

```python
mesh = mio.attach_quality(mesh)                       # quality:* cell_data
mesh = mio.gradient(mesh, "T", location="cell")       # T:gradient cell_data
fm = mio.feature_matrix(mesh, "cell", fields=["mat", "quality:skewness"])
```

A **token budget** composes the same way: a point-cloud model that wants 4096 tokens gets `mio.select_points(mesh, 4096).take(fm.matrix)`, the budget's rows of the full matrix with the column contract intact — see [point-cloud budgets](point_budgets.md).

## Dataset export: `write_dataset`

```python
manifest = mio.write_dataset("out_*.vtu", "dataset/", location="point")
```

Exports a *set* of meshes — a glob (natural-numeric ordering), a list of paths, or a multi-step transient file — as **one on-disk dataset keyed by `mesh_id`**. Sources go through the [sequence machinery](./sequences), so ordering, time metadata and the streaming invariant (one mesh in memory at a time) are inherited, not re-implemented.

Two rules apply to every layout:

- **The schema is strict.** The first mesh defines the column names and dtypes; a later mesh that disagrees is a named error, never a silently widened union. A dataset where every row group means the same thing is the entire point.
- **`mesh_id` is stable.** `"stem"` (default) uses each file's stem (multi-step files append `_<step>`; duplicates are a named error), `"index"` uses the zero-padded entry index.

### Parquet layout (`format="parquet"`, needs `[arrow]`)

```
dataset/
├── meshioplusplus_dataset.json      # layout version, columns, entries
├── mesh_id=case_0/data.parquet
└── mesh_id=case_1/data.parquet
```

Hive-partitioned — the layout every Parquet engine discovers natively. Each file is exactly what [`to_arrow`](./interop#arrow-and-parquet) builds (same columns, same `meshioplusplus:*` schema metadata) plus a `mesh_id` string column and per-file `meshioplusplus:mesh_id` / `meshioplusplus:time` metadata. Read it back with anything:

```python
import pyarrow.dataset as ds
table = ds.dataset("dataset/", partitioning="hive").to_table()

import pandas as pd
df = pd.read_parquet("dataset/")     # mesh_id arrives as a column
```

The JSON manifest records the layout version, the column order, and one entry per mesh (`mesh_id`, source path, time, time source, row count). A run that fails partway leaves **no manifest** — a partial dataset must not look complete.

### Chunked Zarr / HDF5 layouts (`format="zarr"` / `format="hdf5"`)

For datasets too large to hold in memory: one root group, one subgroup per `mesh_id`, one chunked dataset per column, written streaming.

```
dataset.zarr/                        # or dataset.h5
├── attrs: meshioplusplus_dataset    # the same manifest, as JSON
├── case_0/
│   ├── x, y, z, T                   # one chunked array per column
│   └── attrs: time, num_rows, cell_types
└── case_1/ …
```

Zarr needs the `[zarr]` extra (`pip install meshioplusplus[zarr]`; both zarr-python 2.x and 3.x work); HDF5 rides h5py from the `[all]` extra. **Numeric columns only**: the object-dtype `cell_type` column of a cell table is dropped with a warning — its information is in each group's `cell_types` attr — which keeps the layout clear of variable-length-string dtype portability problems.

### CLI and MCP

```bash
meshioplusplus data export-dataset 'out_*.vtu' dataset/ --location cell
meshioplusplus data export-dataset a.vtu b.vtu ds.zarr --format zarr
```

A sub-verb of the [`data` group](./cli#meshioplusplus-data) (Python CLI only, like `data export`), and an `export_dataset` [MCP tool](./mcp) with the same sandboxed glob handling as the `sequence` tool.

## PyTorch / JAX tensors: `to_torch`, `to_jax`

```python
t = mio.to_torch(mesh)               # torch tensors, host, zero-copy
t = mio.to_torch(mesh, device="cuda")  # + one bus transfer per array
j = mio.to_jax(mesh)                 # jax arrays on JAX's default device
```

Both return the same `DevicePayload` the [GPU handoff](./gpu) uses — points, per-block connectivity, data dicts, `block_bases`, regions as index arrays — with every array adopted through DLPack. On the host, `to_torch`'s adoption is **genuinely zero-copy** (the tensors share memory with the mesh; measured in the tests, not assumed). `device=` then moves each tensor with `.to(device)`, which per the GPU honesty rule is one bus transfer per array, recorded in the payload's `derived` list.

`to_jax` has no `device=` parameter: placement belongs to JAX (`jax.default_device`, `jax.device_put`), and JAX puts adopted arrays on its default device. One JAX-specific note: under JAX's default configuration (`jax_enable_x64=False`) 64-bit arrays cannot exist, so meshio++'s canonical float64/int64 arrays fall back to `jnp.asarray` — a copy plus JAX's own 32-bit demotion, counted and warned once. Pass `float32=True`/`int32=True` to make the narrowing explicit on the meshio++ side, or enable x64 in JAX.

### Packaging: deliberately no `[torch]` / `[jax]` extra

The CuPy precedent, applied again: torch's default Linux wheel bundles CUDA at roughly 900 MB, and JAX's accelerator story lives in its own extras (`jax[cuda12]`, …). A `meshioplusplus[torch]` extra pinning either would surprise far more people than it would help (the same reasoning that keeps Open3D's ~400 MB wheel out of the interop extras). Install the framework directly — `pip install torch` / `pip install jax` — and `has_torch()` / `has_jax()` answer availability without raising; a missing install raises a named error pointing here.

**CI note:** like the CuPy device tests, the torch/jax adoption tests `importorskip` and are not exercised by public CI — the frameworks are too heavy for the runners. The payload construction they share with `to_dlpack` *is* fully covered.

## From here to PhysicsNeMo

Everything above is per-mesh. For a *collection* of solution outputs — splits, tags, per-case metadata — the [dataset manifests](./datasets) page is next, and the [PhysicsNeMo adapter](./physicsnemo) builds directly on this page's contracts: `graph_sample` composes `feature_matrix` + `edge_index` into the MeshGraphNet tensor set, and the manifest replaces the ad-hoc glob in every training script.
