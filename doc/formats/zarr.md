# Zarr

A reader and writer for [Zarr](https://zarr.dev/) stores in NVIDIA [PhysicsNeMo](https://developer.nvidia.com/physicsnemo)'s own mesh layout — the one `physicsnemo.mesh.io.from_zarr` reads and its `MeshReader` accepts wherever it accepts a [`.pmsh`](./pmsh.md). Like `.pmsh` a store is a **directory**, but a chunked and compressed one, so a large mesh can be read in pieces.

| | |
|---|---|
| **Format name** | `zarr` |
| **Extensions** | `.zarr` |
| **Read / Write** | ✓ / ✓ |
| **Extra dependencies** | `zarr` (writing needs 3.x) |

## Reading & writing

```python
import meshioplusplus

meshioplusplus.write("case.zarr", mesh)                  # a directory named case.zarr
mesh = meshioplusplus.read("case.zarr")
```

```python
meshioplusplus.zarr.write(
    "case.zarr", mesh, manifold_dim="auto", float32=True, chunk_rows=200_000, zstd_level=3
)
```

`manifold_dim`/`float32` are [`pmsh`](./pmsh.md)'s. `chunk_rows` and `zstd_level` are upstream's own chunk and compression policy — matching them keeps a store written here indistinguishable from one written by `to_zarr` — and `zstd_level=0` writes uncompressed.

**Writing requires zarr 3.x**, and the error says so by name: zarr-python 2.x cannot produce a v3 store at all, and a v2 store is not what upstream reads. Reading uses only `open_group` and is not version-gated in the same way.

## File structure

```
case.zarr/
├── zarr.json          group metadata; root attributes carry the type tag
├── points/            (n_points, dim) float32/float64
├── cells/             (n_cells, k) int64
├── point_data/<name>
├── cell_data/<name>
└── global_data/<name> scalars keep shape ()
```

Root attributes: `physicsnemo_mesh_type: "Mesh"` (what upstream dispatches on), `__tensordict__` (`{"batch_size": [...], "version": 1}`, also on each data group, so the store opens with `tensordict.from_zarr` too) and `meshioplusplus:provenance`. Metadata is consolidated, as upstream does.

## Cell types

Identical to [`pmsh`](./pmsh.md): one simplex kind per store, with the same tessellate-and-warn reduction.

## Data mapping

`point_data`/`cell_data` map onto the groups of the same name and `field_data` onto `global_data`, with scalars kept genuinely 0-d rather than promoted to length-1 arrays. Regions and non-numeric arrays are dropped with a warning.

## Quirks & limitations

* **Not `write_dataset(format="zarr")`.** That writes a *training dataset* — one subgroup per mesh, tabular columns, a manifest in the root attributes — which is a different thing entirely (see [ML data handling](../ml.md)). Such a store is refused here by name rather than half-read.
* A `DomainMesh` store is refused by name.
* A store with no type attribute but a `points` array is accepted, matching `from_zarr`'s own rule.
* Nested groups inside a data group are dropped with a note: meshio++ data dictionaries are flat.
* Provenance rides `meshioplusplus:provenance` in the root attributes, which is plain JSON — so `read_metadata` recovers it without importing zarr at all.
