# CAE sample `.npz`

A writer and reader for the per-case `.npz` layout NVIDIA [PhysicsNeMo](https://developer.nvidia.com/physicsnemo)'s CAE datapipes read — one file per case, carrying the *superset* of the keys `DoMINODataPipe` and `TransolverDataPipe` consume, since each selects its own subset through `keys_to_read` and ignores the rest. The layout is transcribed from the Kratos `PhysicsNeMoApplication`'s own exporter, so a meshio++ case and a Kratos case are interchangeable in one dataset directory.

| | |
|---|---|
| **Format name** | `cae` |
| **Extensions** | `.npz` |
| **Read / Write** | ✓ / ✓ |
| **Extra dependencies** | — |

This is a **tabular sample export, not a mesh format in the usual sense**. It reads back, but a round trip is lossy by construction: the surface half is a triangulation of the input's skin, and the volume half is node coordinates plus nodal fields with no volume connectivity at all.

## Reading & writing

```python
import meshioplusplus

meshioplusplus.write("case_0.npz", mesh)                 # one case
mesh = meshioplusplus.read("case_0.npz")                 # the surface half
volume = meshioplusplus.cae.read("case_0.npz", part="volume")
```

```python
meshioplusplus.cae.write(
    "case_0.npz",
    mesh,
    surface_fields=["p", "wallShearStress"],
    volume_fields=["U"],
    global_params={"stream_velocity": 30.0, "air_density": 1.226},
    global_params_reference={"stream_velocity": 25.0},
    global_params_order=["stream_velocity", "air_density"],
    time=0.5,
    step=3,
)
```

A whole dataset at once, from a glob, a path list or a multi-step file:

```python
meshioplusplus.cae.export_cases("run_*.vtu", "dataset/")
```

which is also the `data export-cae` CLI verb and the `export_cae` MCP tool. One mesh is alive at a time however large the run.

## File structure

An uncompressed `numpy.savez` archive. A mesh with 3-D cells is the *volume*: its nodes become `volume_mesh_centers`, its numeric `point_data` becomes `volume_fields`, and its skin — extracted and triangulated here — becomes the surface half, with the volume's own `cell_data` gathered onto each triangle through its parent cell. A surface mesh is written as the surface alone, which is what DoMINO's surface-only model type reads.

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `stl_coordinates` | (P, 3) | float32 | surface vertices |
| `stl_faces` | (3T,) | int32 | **flattened** — DoMINO feeds it straight to `signed_distance_field` |
| `stl_centers` | (T, 3) | float32 | vertex mean, not the area centroid |
| `stl_areas` | (T,) | float32 | `0.5 * ‖(v1-v0) × (v2-v0)‖` |
| `surface_normals` | (T, 3) | float32 | unit length; winding is the input's |
| `surface_mesh_centers` / `surface_areas` | | | the aliases Transolver reads |
| `surface_fields` | (T, ΣW) | float32 | omitted when empty |
| `volume_mesh_centers` | (N, 3) | float32 | the volume's **nodes** |
| `volume_fields` | (N, ΣW) | float32 | nodal only; omitted when empty |
| `<param>` | (1,) | float32 | one per global parameter |
| `global_params_values` / `_reference` | (k, 1) | float32 | stacked, in `global_params_order` |
| `TIME` / `STEP` | (1,) | float32 / int64 | only when given |

Point fields are averaged from a triangle's three vertices; cell fields are replicated from the parent cell. Widths flatten, so a 3-vector contributes three columns, and the blocks are concatenated in the requested order.

meshio++ adds four sidecar keys (`meshioplusplus:surface_field_names`, `..._widths`, `meshioplusplus:node_field_names`, `..._widths`) so a read can split the concatenated blocks back into named arrays. The datapipes ignore unknown keys, so they cost the consumer nothing; a file written elsewhere, without them, comes back with its block whole rather than guessed at.

## Cell types

Triangles only, on the surface side. Quads and n-gons are triangulated and higher-order cells linearized first; a mesh whose highest cell dimension is 1 or 0 is refused by name. The volume half has no cells at all — it reads back as a `vertex` point cloud.

## Data mapping

`surface_fields` becomes per-triangle `cell_data`, `volume_fields` per-node `point_data`, and the global parameters plus `TIME`/`STEP` become `field_data`. `surface_normals`, `surface_areas` and `stl_centers` come back as `cell_data` too.

## Quirks & limitations

Three rules belong to the reader on the other side, and each produces silently wrong results rather than an error when broken:

* **Every array must have `ndim >= 1`**, because `NpzFileReader` does `in_data[key][:]`. Hence `TIME`/`STEP` and every scalar parameter as shape-`(1,)` arrays.
* **No key but `volume_mesh_centers` and `volume_fields` may contain the substring `volume`.** That reader takes the volume row count from `next(key for key in in_data.keys() if "volume" in key)` — the first such key in file order, over *every* key in the file, not just the requested ones. meshio++'s own sidecars are therefore called `node_field_*`, and a global parameter whose name contains `volume` is refused by name.
* **The two volume keys must share their row count**, since the slice computed from one is applied to the other.

Provenance rides a bytes array written as the archive's first member, so the ordinary head scanner finds it with no `.npz`-specific code — each row is NUL-padded so the block does not run into the zip's next local-file header.
