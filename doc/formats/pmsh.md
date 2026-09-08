# PhysicsNeMo `.pmsh`

A reader and writer for NVIDIA [PhysicsNeMo](https://developer.nvidia.com/physicsnemo)'s memory-mapped mesh layout — the format its own `MeshReader`/`MeshDataset` glob for (`**/*.pmsh`) and the one a training loop reads a dataset from. A `.pmsh` is a **directory**, not a file: two small JSON manifests describing shapes and dtypes, and one raw headerless blob per array, memory-mapped on the way back in.

| | |
|---|---|
| **Format name** | `pmsh` |
| **Extensions** | `.pmsh` |
| **Read / Write** | ✓ / ✓ |
| **Extra dependencies** | — |

Written in pure numpy rather than through `physicsnemo.mesh.Mesh.save`, and that is the point: a solver box with no torch, no CUDA and no NVIDIA stack installed can still write the training set the framework reads. Parity with upstream is pinned in both directions by tests that gate on an installed physicsnemo.

## Reading & writing

```python
import meshioplusplus

meshioplusplus.write("case.pmsh", mesh)                  # a directory named case.pmsh
mesh = meshioplusplus.read("case.pmsh")
```

```python
meshioplusplus.pmsh.write("case.pmsh", mesh, manifold_dim="auto", float32=True)
```

`manifold_dim` selects the one simplex kind the target holds (`"auto"` = the highest dimension present, or `0`/`1`/`2`/`3` for vertex/line/triangle/tetra); `float32` follows the upstream convention for point coordinates, and `float32=False` keeps float64. Cells are always int64.

## File structure

```
case.pmsh/
├── meta.json                      {"_type": "<class 'physicsnemo.mesh.mesh.Mesh'>"}
└── _tensordict/
    ├── meta.json                  points/cells leaf entries + the three data groups
    ├── points.memmap              raw little-endian C-order float32/float64
    ├── cells.memmap               raw little-endian C-order int64
    ├── point_data/{meta.json, <name>.memmap}
    ├── cell_data/{meta.json, <name>.memmap}
    └── global_data/{meta.json, <name>.memmap}
```

Every leaf entry carries `device`, `shape`, `dtype` (the `str(torch.dtype)` spelling, e.g. `"torch.float32"`) and `is_nested`; the blobs carry no header at all, so the manifests are the only description of them. The optional `_cache` group upstream writes is not produced and not required to load.

**A zero-element array has no blob.** tensordict does not persist an empty tensor, so a point cloud's `cells.memmap` is absent while `meta.json` still declares `"shape": [0, 1]` — the empty-cells sentinel `Mesh.__post_init__` restores. A writer that emits the file anyway produces a tree no upstream store has; a reader that treats the absence as corruption rejects every point cloud. A blob missing for a *non-empty* array is an error, never silently zeros.

**A `.pmsh` directory containing `zarr.json` is read as a [Zarr store](./zarr.md)**, which is what upstream's `MeshReader._load_sample` does: the extension names the role, the contents name the codec.

## Cell types

One simplex kind per file — `vertex`, `line`, `triangle` or `tetra` — named by the cells array's trailing dimension. Non-simplex cells at the selected dimension are tessellated on the way out (linearize, then simplexify), and blocks at other dimensions are dropped. Every reduction is warned, never silent; export each dimension as its own file when both matter.

## Data mapping

`point_data` and `cell_data` map onto the groups of the same name; `field_data` becomes `global_data`. Named regions, `point_sets`/`cell_sets` and non-numeric arrays have no counterpart and are dropped with a warning.

Arrays come back memory-mapped copy-on-write, so they are writeable (meshio++'s standing contract) and paged in only as they are touched — which is what makes this format load faster than an equivalent VTU. Writing to one never reaches the file.

## Quirks & limitations

* The reduction to one simplex kind is lossy by construction, and a round trip therefore does not reproduce a mixed-topology mesh.
* There is no provenance slot: a `.pmsh` carries no free-text field, and adding a sidecar file would put something upstream's own loader never wrote into its layout. `read_metadata` honestly reports none.
* A `physicsnemo` `DomainMesh` (`.pdmsh`, an interior mesh plus named boundaries) is refused by name rather than half-read.
