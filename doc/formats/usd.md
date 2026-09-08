# OpenUSD

A reader and writer for [OpenUSD](https://openusd.org/) stages — the interchange format Omniverse, usdview and every USD-aware DCC tool read. A solve, or a surrogate's prediction, becomes a scrubbable 3-D asset: points are time-sampled per step and topology is re-authored only on the steps where it actually changes.

| | |
|---|---|
| **Format name** | `usd` |
| **Extensions** | `.usd`, `.usda`, `.usdc` |
| **Read / Write** | ✓ / ✓ |
| **Extra dependencies** | `usd-core` (the `[usd]` extra) |

`usd-core` is Pixar's self-contained PyPI build of USD — no Omniverse install — and is imported lazily, so a build without it is unaffected until the format is actually used, when it raises a named install error.

## Reading & writing

```python
import meshioplusplus

meshioplusplus.write("part.usda", mesh)          # text
meshioplusplus.write("part.usdc", mesh)          # binary crate; the extension picks it
mesh = meshioplusplus.read("part.usda")
```

```python
meshioplusplus.usd.write(
    "part.usda", mesh, prim_path="/scene/part", up_axis="Z", meters_per_unit=0.001
)
```

A time-sampled stage from a sequence:

```python
meshioplusplus.write_sequence("solve.usda", ((t, mesh) for t, mesh in steps))
mesh = meshioplusplus.read("solve.usda", time_step=-1)   # the last step
```

`read` takes `time_step` (0 is the first, negatives count from the end) and attaches the full list of authored times to the mesh, so `read_metadata(...)["time_values"]` and the sequence engine's step count both work.

## File structure

One `UsdGeomMesh` per prim (or a `UsdGeomPoints` for a cell-less mesh) under `prim_path`, with `points`, `extent`, `faceVertexCounts`/`faceVertexIndices` and one `primvars:<name>` per data array. Stage metadata carries the up axis, metres per unit and the time-code range; the root layer's `documentation` field carries the provenance block.

Topology is written only when its hash changes, recorded in the prim's custom data under `physicsNemo:lastTopologyHash` (upstream's own spelling, so a stage written by either tool reads the same). A fixed-topology series therefore authors `faceVertexIndices` once; a remeshed one authors it per change, and stays valid at every step.

## Cell types

**n-gons are kept**, not triangulated: USD stores a face-vertex count per face, so a quad-dominant hex skin arrives as quads. A mesh with 3-D cells is written as its boundary surface; higher-order surface cells are linearized; a mesh with no 2-D or 3-D cells becomes a point cloud.

On the way back, faces are grouped by vertex count into `triangle`, `quad` and one ragged `polygon` block, in that fixed order.

## Data mapping

`point_data` becomes `vertex`-interpolated primvars and `cell_data` `uniform` ones — getting the interpolation token wrong renders plausible, wrong colours, so it is a contract rather than a detail. A width-3 array is a `Float3Array`, an integer array an `IntArray`, and anything else a flat array with `SetElementSize`. `faceVarying` and `constant` primvars have no meshio++ counterpart and are skipped with a warning.

A volume mesh's `cell_data` is gathered onto its skin through the parent-cell map, so a solid arrives coloured by its per-cell material.

## Quirks & limitations

* **Block identity is not preserved.** A USD prim holds one face list, so a mesh with three blocks (triangle, quad, triangle) comes back as two, grouped by vertex count.
* Several prims on one stage are concatenated, with a `usd:prim` cell array naming which is which and `usd:prim_paths` in `field_data`.
* A single write authors a *default* value rather than a time sample, so a viewer with no timeline sees plain geometry.
* Provenance lives in the root layer's `documentation`. A text `.usda` therefore also surrenders it to the ordinary byte scanner; a binary crate layer needs `usd-core` installed for `read_metadata` to recover it.
* The step count comes from a full read (there is no cheap metadata path), which is the same arrangement Exodus has.
