# Named regions

A **region** is a named group of mesh entities. Every format that models one spells it differently — gmsh calls it a physical group, Exodus an element block or node set or side set, Abaqus an `*NSET` / `*ELSET` / `*SURFACE`, MED a family or group, UNV a group, Ansys a component, OpenFOAM a boundary patch, Kratos a SubModelPart — and converting between any two of them used to lose or mangle the grouping. `Region` is the one model they all map onto.

Before meshio++ 8.1 these lived only on the Python `Mesh`, as `point_sets` / `cell_sets`. They never crossed the pybind11 boundary, so the C API, Fortran, WebAssembly and the native CLI dropped them entirely, and each set-capable format smuggled them through its own side-channel struct. Regions close that gap.

## The model

```python
import meshioplusplus as mio

region = mio.Region("wall", "side", [[0, 1], [1, 3]], dim=2, tag=7)
```

| Field     | Meaning |
|-----------|---------|
| `name`    | the group's name, as the originating format spelled it |
| `kind`    | `"point"`, `"cell"` or `"side"` — what `entries` indexes |
| `dim`     | the topological dimension the group was declared for, or `-1` if the format does not say |
| `tag`     | the format-native integer id (gmsh physical tag, MED family id, Exodus set id), or `-1` |
| `entries` | int64 indices — see below |

`dim` and `tag` exist because gmsh needs them: a physical group is *per-dimension*, two groups of different dimensions may share a name, and the integer tag is what a round-trip back to gmsh must restore. They are carried through conversion even by formats that have no use for them.

### The three kinds

**`"point"`** — entries are point indices, shape `(n,)`.

**`"cell"`** — entries are **global cell indices**, shape `(n,)`, numbered *block-major*: block 0's cells first, then block 1's, and so on. This is the same numbering [`partition_labels`](partition.md) uses for its flat buffer. `detail/cell_index.hpp` is the single owner of the conversion to and from `(block, row)`, so nothing re-derives it — the browser viewer's `_dimension_rows` counter bug (a running index that failed to advance over *skipped* blocks, so every per-cell quantity silently shifted) is the cautionary tale for what a second copy of that arithmetic costs.

**`"side"`** — entries are `(global cell index, local facet index)` pairs, shape `(n, 2)`. Facets are numbered exactly as `detail/cell_faces.hpp` numbers the faces of a 3-D cell and `detail/cell_edges.hpp` the edges of a 2-D one. **This kind has no equivalent in the legacy `point_sets`/`cell_sets` model at all** — it is only reachable through `.regions`.

### Canonical ordering

Entries are stored **ascending and de-duplicated** (lexicographically on the pair, for `"side"`). Two regions describing the same group are therefore bitwise equal, which is what makes the cross-format round-trip matrix an equality assertion rather than a set-comparison heuristic, and what makes region output byte-identical across the MESHIO / NATIVE / KRATOS backends and across thread counts.

The visible consequence: a group that survives an operation comes back **sorted**, not in whatever order the operation's permutation produced. Set semantics are what a named group means.

## Compatibility with `point_sets` / `cell_sets`

`mesh.point_sets` and `mesh.cell_sets` keep working exactly as they always have. They are now **views** over `mesh.regions`: reading one materializes the historical shape (a flat index array for `point_sets`, a per-block list of local index arrays for `cell_sets`), and writing one creates or replaces the matching regions.

```python
mesh.point_sets = {"fixed": [0, 3]}     # -> a "point" region
mesh.cell_sets["wall"] = [[0], [1, 2]]  # -> a "cell" region, global indices
mesh.regions                            # both, plus any side regions
```

The views are `dict` **subclasses**, not bare mappings, because plenty of code — including `numpy.testing.assert_equal` — branches on `isinstance(x, dict)`.

Two accommodations make the compat lossless rather than nearly so:

- **The passthrough escape hatch.** Formats stash things in `cell_sets` that are not cell indices at all: gmsh's `gmsh:bounding_entities` holds entity tags, orientation sign included. Such an entry cannot be a region, so it is kept verbatim in a private passthrough dict and returned unchanged. The test is *non-negative integers*, not "integers in range" — a negative value is the reliable tell (in the bundled `example.msh`, 475 of the 476 populated blocks contain one), while requiring every index to be in range would reject a genuine but sloppily-built cell set, which meshio has always accepted.
- **`None` blocks become empty arrays.** A `cell_sets` value may carry `None` for a block with no members (gmsh's reader builds `[None] * n`). Global indices cannot express "absent" separately from "empty". Every consumer in the tree already treated the two identically — and the Abaqus writer's `len(v[ic])` would in fact have raised on a genuine `None`.

One sharp edge, shared with the old model: indices are captured against the mesh's blocks *at the time you set them*. Appending a cell block is safe; inserting, removing or reordering one invalidates them.

## The format matrix

| Format | `point` | `cell` | `side` | `tag` | Notes |
|--------|:---:|:---:|:---:|:---:|-------|
| **abaqus** | ✅ | ✅ | ✅ | ❌ | `*NSET` / `*ELSET` / `*SURFACE`. Abaqus names its groups but has no integer id for them. |
| **gmsh 2.2** | ❌ | ✅ | ❌ | ✅ | A physical group is a named, tagged, per-dimension group of *elements*. No node-set and no side-set concept. |
| **gmsh 4.1** | ❌ | ✅ | ❌ | ✅ | Membership lives in `$Entities`, which describes the *geometry* — so it round-trips (since v9.7.0) for a mesh that says which entity each node belongs to, i.e. one carrying `point_data["gmsh:dim_tags"]`, as any mesh read from a 4.1 file does. A mesh that came from another format has no entity structure to write, so no `$Entities` is emitted and only the group *names* survive; write 2.2 for that case. |
| **exodus** | 📖 | 📖 | 📖 | ✅ | **Read only** (v8.6.0): element blocks → `cell`, node sets → `point`, side sets → `side`, each tagged with its `eb_prop1`/`ns_prop1`/`ss_prop1` id. The writer emits neither `eb_names` nor side sets, so nothing written comes back. See [`doc/formats/exodus.md`](./formats/exodus.md#named-regions). |
| **flac3d** | ❌ | 🏷️ | ❌ | ❌ | A `ZGROUP`/`FGROUP` is a named group of zone/face *cells*. FLAC3D identifies a group by all three of (`ZGROUP` vs `FGROUP`, name, slot), so the region is named `<zone|face>:<name>:<slot>` — membership round-trips exactly, the name is rewritten into that vocabulary and is a fixed point thereafter. No node-set and no facet-group concept, so point and side regions are dropped. See [`doc/formats/flac3d.md`](./formats/flac3d.md#group-names-on-write). |
| **med** | ✅ | ✅ | ❌ | ❌ | A MED family (`FAS/NOEUD`/`FAS/ELEME`) is a named group of nodes/elements — one region per group *name*, a multi-group family contributing to all of its names. A family id is per unique *combination* of names rather than per name, and a name may span several ids, so the format-native `tag` is not carried. MED has no facet-group concept, so side regions are dropped. See [`doc/formats/med.md`](./formats/med.md#named-regions). |

📖 = read only. Exodus is a region *source* rather than a round-trip target, so it is recorded in `tests/python/test_region_roundtrip.py`'s `READ_ONLY_REGIONS` rather than as a row in the round-trip matrix — that matrix writes and reads back, and a format that cannot write cannot round-trip.

🏷️ = membership round-trips but the *name* is rewritten into the format's own namespace. FLAC3D is the one such format, recorded in that file's `NAMESPACED_REGIONS` bucket for the same reason: the matrix asserts that a region's name survives exactly, which is the right assertion for every format that can make it, and weakening it for one exception would cost the other three.

Deferred to Phase 2, and listed in `tests/python/test_region_roundtrip.py` so the gap stays on the record: **UNV** and **Ansys** (absorbing `UnvInfo`/`AnsysInfo`), **OpenFOAM** (boundary patches, which are face groups and therefore side regions), **XDMF** (Sets), and **VTU/VTP** — which have no native set concept at all, so a convention has to be chosen and documented rather than invented silently. Exodus's **writer** belongs on this list too.

### Gmsh precedence

The gmsh mapping is *derived and additive*, never authoritative:

- **Read**: regions are derived from the existing `gmsh:physical` cell_data plus the `$PhysicalNames` entries in `field_data`. Both stay exactly as they were, so every existing consumer is unaffected.
- **Write**: `field_data` and `gmsh:physical` are consulted **first**; regions only fill gaps. A mesh carrying gmsh's own metadata therefore writes byte-identical bytes, and a mesh whose groups came from another format still gets real physical groups.

A gmsh dimension-0 physical group tags `vertex` *cells*, not points, so it becomes a `"cell"` region with `dim == 0` rather than a `"point"` one.

### Abaqus face identifiers

Abaqus numbers element faces `S1`..`S6`, and that numbering is **not** meshio++'s. `C3D8`'s `S1` is the 1-2-3-4 face — local nodes `{0,1,2,3}` — which is meshio++'s face 4. The mapping is spelled out per type in both `abaqus.cpp` (`abq_face_index`) and `_abaqus.py` (`_ABAQUS_FACE_ORDER`), which are twins and must stay in step: getting it wrong yields a plausible-looking side set pointing at the wrong faces. Shell elements' `SPOS`/`SNEG` name a side rather than a facet and map to facet 0 and 1.

## How operations treat regions

| Operations | Treatment |
|---|---|
| `crop`, `split`, `merge`, `reorder`, `clean`, `partition`, `convert_cells`, `refine`, `decimate` | **Remapped** through `detail/region_remap.hpp` |
| `transform`, `smooth`, `interpolate` (the target's), the `data` operations, `attach_quality` | **Passed through** — nothing is renumbered |
| `slice`, `isosurface`, `extract_surface`, `extract_skin` | **Dropped**, with a `log::warn` naming the operation |

`merge` namespaces colliding names by source index (`0:wall`, `1:wall`), the same rule it already applies to `field_data` keys.

`detail/region_remap.hpp` is the single owner of the carry. It needs to be told which *shape* an operation's cell map is in, because they genuinely differ:

- **`Direct`** — `map[c]` is *the* output cell, or `-1`. (crop, split, clean, partition, reorder.)
- **`FirstChild`** — `map[c]` is the first of a contiguous run of children, ending at the next non-negative entry. (convert_cells' simplexify, refine, decimate.) This shape also handles 1:1 maps correctly, but `Direct` must not be replaced by it: a permutation's map is not monotone, so the "next entry" rule would invent nonsense ranges.
- **`Global`** — one flat array indexed by input *global* cell index. (merge.)

A **side** entry survives only if its cell survives *and* the facet still exists: the output cell must have the same type as the input cell, and the facet index must still be in range. Under `FirstChild` a parent's children are new cells of a subdivided or different topology, so there is no facet correspondence and side regions are dropped by name.

A region that loses every entry is still carried, as an empty group — the name is information in its own right, and that is what the per-operation Python shims did before the shared helper replaced them.

## The other language bindings

**C API** ([`doc/c_api.md`](c_api.md)) — an opaque snapshot handle:

```c
mio_regions* r = mio_regions_create(mesh);
mio_region_info info;
mio_regions_info(r, 0, &info);
int64_t count;
const int64_t* entries = mio_regions_entries(r, 0, &count);
mio_regions_free(r);

mio_mesh_add_region(mesh, "wall", MIO_REGION_SIDE, 2, -1, pairs, 4);
```

**Fortran** ([`doc/fortran.md`](fortran.md)) — `m%regions(keys=, entries=)` and `m%add_region(...)`. Point and cell indices are shifted to Fortran's 1-based convention by the copying getter; the *facet* column of a side region is not, matching the `partition_labels` rule that a value which is not an index stays as it is.

**WebAssembly** ([`doc/wasm.md`](wasm.md)) — regions ride on the mesh object, so `readMesh` / `writeMesh` / `convert` carry them with no new call:

```js
const mesh = mio.readMesh('/in.inp');
mesh.regions; // [{ name, kind, dim, tag, entries }]
```

**Native CLI** — `meshioplusplus info` prints `Point sets:` / `Cell sets:` / `Side sets:`, and `meshioplusplus diff` reports regions added, removed and changed, folding them into its nonzero exit code. (`meshes_equal` deliberately does **not** consider them: it is documented to compare geometry and data.)

## Enumerating and splitting by region

Two things build directly on the model above, both added in v8.7.0:

- **`read_metadata(...)["regions"]`** (Python; `readMetadata(...).regions` in WASM; `mio_read_metadata_num_regions`/`_region_name`/`_region_info` in the C API; `mio_metadata%regions` in Fortran; the equivalent shape in Julia/R) — each region's `name`/`kind`/`dim`/`tag`/`num_entries`, **without the entries themselves**. Populated from whatever's already on an in-memory mesh, so it costs nothing extra whenever the summary came from a fallback read (every format lacking a native metadata path, plus Exodus, which always falls back); empty on the VTU/VTP/XDMF native metadata paths, since none of those map regions at all. **Gmsh 4.1's native path does report them** (since v9.7.0): `$Entities` and `$PhysicalNames` are both small and both precede `$Elements`, so the group names, dimensions, tags and entry counts come out of the block headers alone, and the summary agrees with a real read. This is what lets a caller build a SubModelPart tree — or just decide whether it's worth reading the mesh at all — without paying for a full read first.
- **`meshioplusplus regions FILE`** (both CLIs) lists the same summary directly (`--json` for machine consumption):

  ```bash
  meshioplusplus regions bracket.inp
  # <meshio++ mesh regions> (2)
  #   fixed (point, 12 entries, tag=1)
  #   solid (cell, 340 entries, dim=3, tag=2)
  ```

- **`split(mesh, by="regions")`** (see [`doc/split.md`](split.md)) turns each named `Cell` region into its own submesh — the natural next step after listing them. Unlike every other `split` criterion this is not a partition: regions may overlap, so a cell can land in zero, one, or several output pieces; `Point`/`Side` regions produce no piece, since there is no sound default for "these facets alone".

## Nested groups: the `/` convention

Regions are flat — a `Region` has a name, not a parent. Formats with genuinely nested groups (Kratos SubModelParts, and MDPA's nested `Begin SubModelPart`) therefore **flatten the path into the name with `/`**:

```
Structure/Loads/PointLoad3D
```

This is a convention, not a parser: the region is one name that happens to contain slashes, and it round-trips as such. The KRATOS mesh backend is the one place that interprets it — `BuildSubModelPartsFromRegions` splits on `/` and walks or creates the chain, adding members to the leaf, and `RestoreRegions` walks the tree back into the same flattened names.

`.` is **reserved** and cannot appear in a segment: `ModelPart::FullName()` joins ancestors with it. A region name containing one is left on the mesh but not materialized as a SubModelPart, with a warning — the same treatment `Side` regions already get. This is the exact interaction MED's own Kratos `MedApplication` runs into: it uses `.` as *its* group-name nesting separator when writing MED family names from Kratos, so a MED file carrying such a name round-trips its region unaffected through every mesh backend except KRATOS's SubModelPart materialization, which — per this rule — leaves it flat with a warning rather than guessing at the intended nesting.

One thing a SubModelPart cannot carry back is a region's `mDim`/`mTag`: it stores a name and member ids and nothing else. A region reconstructed from one reports `-1` for both unless a staged region of the same name and kind supplied them. That is the truthful answer rather than a gap — synthesizing a value would be inventing data.
