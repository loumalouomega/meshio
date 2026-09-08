# FLAC3D (`.f3grid`)

The [Itasca FLAC3D](https://www.itascacg.com/software/flac3d) grid format (`.f3grid`): ASCII or binary, with separate ZONE (3D) and FACE (2D) sections and named cell groups.

| | |
|---|---|
| **Format name** | `flac3d` |
| **Extensions** | `.f3grid` |
| **Read / Write** | ✓ / ✓ |
| **Extra dependencies** | — |

## Reading & writing

```python
import meshioplusplus

mesh = meshioplusplus.read("grid.f3grid")
meshioplusplus.flac3d.write("out.f3grid", mesh, float_fmt=".16e", binary=False)
```

- **`float_fmt`** — coordinate format (ASCII only).
- **`binary`** — binary (`True`) or ASCII (`False`, default).

## File structure

**Format auto-detection**: the first 8 bytes are checked for a null byte; if found, the file is binary, else ASCII.

**Binary** (little-endian): 8 bytes of a header pair whose meaning is undocumented on read (`_read`'s comment: "not sure what the first bytes represent") but whose exact values (`1375135718, 3`) **are** reproduced by the writer as a magic number; then `uint32` node count, per-node `(point_id: uint32, x,y,z: float64×3)`; then for `zone` and `face` in that order: `uint32` cell count, per-cell `(cell_id: uint32, num_verts: uint32, node_ids: uint32×num_verts)` — if `num_verts == 7` (a degenerate "B7" hexahedron-as-7-node encoding), the last node is duplicated to make 8; then `uint32` group count and, per group, length-prefixed name/slot strings plus a `uint32`-counted id list.

**ASCII**: `G <id> <x> <y> <z>` per point; `Z <TypeAbbrev> <id> <n0> <n1> ...` per zone (`F` for faces), with the same `"B7"` degenerate-hex handling; `ZGROUP "<name>" SLOT <slot>` / `FGROUP "<name>" SLOT <slot>` followed by whitespace-separated member id lines until a line starting `*`, `ZGROUP`, or `FGROUP`.

**Zone and face cell ids are separate 1-based namespaces.** A file numbers its zones `1..nz` and its faces `1..nf` independently, and a `ZGROUP`'s members are zone ids while an `FGROUP`'s are face ids — the two never share a numbering. Both readers and both writers honour that; sharing one counter produces a file whose group lists point at the wrong cells (fixed in v10.36.0, [issue #76](https://github.com/loumalouomega/meshioplusplus/issues/76)).

**Cell-block grouping**: cells are grouped into blocks of **consecutive same-typed cells in file order** — not all cells of one type merged globally — so a file alternating types produces alternating blocks (see the reference-file cell list below for a concrete example).

**Right-handed zone reorder** (the format's central quirk): FLAC3D requires each zone's first four corner nodes to form a right-handed system. For every zone cell, meshio++ computes the scalar triple product of the first three edge vectors (from the "primary" meshio++→FLAC3D order's first four nodes); if positive, the primary order is used, otherwise a "flipped" alternate order. This check only happens **on write** — the read-side reorder is a fixed, unconditional permutation, since a well-formed file is assumed to already store correctly-handed zones.

## Cell types & node ordering

| meshio++ type | FLAC3D abbrev | primary order | flipped order (write-only) |
|---|---|---|---|
| `triangle` | `T3` | `[0,1,2]` | — |
| `quad` | `Q4` | `[0,1,2,3]` | — |
| `tetra` | `T4` | `[0,1,2,3]` | `[0,2,1,3]` |
| `pyramid` | `P5` | `[0,1,3,4,2]` | `[0,3,1,4,2]` |
| `wedge` | `W6` | `[0,1,3,2,4,5]` | `[0,2,3,1,5,4]` |
| `hexahedron` | `B8` | `[0,1,3,4,2,7,5,6]` | `[0,3,1,4,2,5,7,6]` |

Read-side order (FLAC3D → meshio++): `triangle/quad/tetra` unchanged; `pyramid: [0,1,4,2,3]`; `wedge: [0,1,3,2,4,5]`; `hexahedron: [0,1,4,2,3,6,7,5]`.

## Data mapping

- `cell_data["cell_ids"]` — the original FLAC3D global cell id, split per block (faces are numbered first, then zones, in the reader's internal concatenation order).
- `mesh.regions` — one [`RegionKind::Cell` region](../regions.md) per `ZGROUP`/`FGROUP`, named `<zone|face>:<name>:<slot>`, with `dim`/`tag` left unset (`-1`). A FLAC3D group is identified by all three of those parts: `ZGROUP` and `FGROUP` are separate namespaces, and a slot partitions the groups within one, so the name alone would not identify a group. `mesh.cell_sets[name]` is the usual compat view over those regions — a list of one array per cell block holding **block-local** indices, while the region's own `entries` are global block-major ones.
- A group member the file never defines is dropped with a warning rather than mapped to something else, and an empty group is still carried, since the name is information.

### Group names on write

The writer decomposes a region name by the same rule, which is what makes a file read from disk a **fixed point**: `zone:Brick1:Default` is written back as `ZGROUP "Brick1" SLOT "Default"` and reads back under the same name. The split is on the *last* colon, since a FLAC3D group name may contain one while a slot may not.

A region whose name is in any other shape — `solid`, say, carried in from Abaqus or Gmsh — keeps its whole self and takes the `Default` slot, so it is written as `ZGROUP "solid" SLOT "Default"` and comes back as `zone:solid:Default`. If such a region's members span both zone and face cells it is written into **both** sections and reads back as two regions, one per namespace. `Point` and `Side` regions have no FLAC3D equivalent and are ignored.

## Quirks & limitations

- **Read/write section-order asymmetry**: the writer emits `* ZONES` before `* FACES` in the ascii file, but the reader concatenates faces-then-zones internally when building `Mesh.cells` — so the returned cell-block order after a read does not match the on-disk section order. This is harmless (the reader doesn't depend on write order) but a genuine structural asymmetry worth knowing about when comparing a file's raw section layout to `mesh.cells`.
- **A group name is rewritten into the format's own vocabulary.** A region called `solid` comes back as `zone:solid:Default`; only a name already in `<zone|face>:<name>:<slot>` form survives verbatim. This is why FLAC3D is recorded in `tests/python/test_region_roundtrip.py`'s `NAMESPACED_REGIONS` bucket rather than as a row in the round-trip matrix, which asserts that names survive exactly.
- **`mesh.cell_sets` values changed meaning in v10.36.0** ([issue #76](https://github.com/loumalouomega/meshioplusplus/issues/76)). They are now per-block **local** indices, the convention every other format and the whole region layer already used; before, this reader alone put *global* indices in the per-block slots, which `_regions.blocks_to_global` then rebased a second time — silently emptying three of the five groups in the bundled reference file. Code that read those values as global indices should read `mesh.regions` instead.

## Notes

- `tests/python/meshes/flac3d/flac3d_mesh_ex.f3grid` (ascii, "FLAC3D 7.00 Release 118") and `flac3d_mesh_ex_bin.f3grid` (its binary counterpart) — checked for point-sum ≈307.0 and an exact ordered list of 12 alternating cell blocks: `[(quad,15), (triangle,3), (hexahedron,45), (pyramid,9), (hexahedron,18), (wedge,9), (hexahedron,6), (wedge,3), (hexahedron,6), (wedge,3), (pyramid,6), (tetra,3)]` — a direct illustration of the "consecutive same-type runs" block-splitting rule above — plus 5 named cell regions whose exact global memberships are asserted: `face:bottom:Default` (cells 0-17), `zone:Brick1:Default` (18-44), `zone:Pyramid2:Default` (45-71), `zone:Wedge3:Default` (72-98) and `zone:Tetrahedron4:Default` (99-125).
- The C++ core handles points, zone/face cells **and** cell groups in both ASCII and binary (with the determinant-based reorder), so groups are reachable from WASM, the C API, Fortran, Julia, R and the native CLI. Its output is byte-identical to the Python reference writer's in both encodings, pinned by `tests/python/test_flac3d.py::test_cpp_matches_python_write`.
