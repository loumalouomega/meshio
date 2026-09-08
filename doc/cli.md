# CLI Reference

The `meshioplusplus` command-line tool is installed alongside the Python package.

```
meshioplusplus --version
meshioplusplus --help
meshioplusplus <subcommand> --help
```

The same verbs are also available as a **standalone native C++ binary** that needs no Python interpreter and no pybind11 extension — see [Native CLI (C API)](./c_api.md#native-command-line-binary) for how to build it (`build/configure.sh --cli --build`), or download a ready-to-run, statically-linked build for Linux/macOS/Windows from the [GitHub Releases](https://github.com/loumalouomega/meshioplusplus/releases) page. It shares the [C API](./c_api.md)'s remaining flat-surface limitation — `convert -s/-d` is unavailable there (it lives only in the Python `Mesh`) — and has no Python fallback for formats whose C++ reader raises. Named **sets** are carried since v8.1.0: they are [regions](./regions.md) in the core, so `info` prints them and `diff` compares them. Everything below otherwise applies identically to both.

---

## meshioplusplus convert

Convert a mesh file from one format to another.

```
meshioplusplus convert [options] INFILE OUTFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format (skip extension detection) |
| `--output-format FORMAT` | `-o` | Force output format |
| `--ascii` | `-a` | Write ASCII variant (default: binary where available) |
| `--float-format FMT` | `-f` | Float format string for ASCII output (default: `.16e`) |
| `--sets-to-int-data` | `-s` | Convert point/cell sets to integer data arrays |
| `--int-data-to-sets` | `-d` | Convert integer data arrays to point/cell sets |

**Transient sequences** — treat a set of files, or the steps inside one file, as one dataset (see [sequences](sequences.md)):

| Option | Description |
|--------|-------------|
| `--input FILE` | An extra input file, appended after `INFILE`; repeatable |
| `--times T1,T2,...` | Explicit per-step times; the count must match |
| `--time-from WHICH` | `auto` (default), `file`, `filename` or `index` |
| `--sequence` | Force sequence handling |
| `--no-sequence` | Force single-file handling (for a filename containing `*` or `{step}`) |

```bash
meshioplusplus convert 'out_*.vtu' series.xdmf          # fan-in (quote the glob!)
meshioplusplus convert series.xdmf 'step_{step}.vtu'    # fan-out
meshioplusplus convert a.vtu --input b.vtu out.xdmf     # pre-expanded argv
```

**Quote the pattern.** A shell expands `out_*.vtu` before the CLI sees it, so the unquoted form arrives as a dozen positionals and fails on the argument count; use `--input` when something else has already expanded the glob.

Ordering is natural-numeric, so `out_10.vtu` follows `out_9.vtu`. A multi-step input aimed at a single-step output is an **error** naming `{step}` and `--time-step`, never a silent write of step 0 — pass `--time-step=N` when you genuinely want one step.

**Data-driven colouring** (SVG/TikZ output only):

| Option | Description |
|--------|-------------|
| `--color-by NAME` | `point_data` or `cell_data` array to colour the faces by |
| `--component I` | Component of a multi-component array (default: its magnitude) |
| `--cmap NAME` | `viridis` (default), `coolwarm` or `turbo` |
| `--vmin V` / `--vmax V` | Colour range (default: the drawn faces' finite range) |
| `--nan-color C` | Colour for NaN/infinite values (default: `#808080` / `gray`) |
| `--colorbar` | Append a gradient bar with min/max labels |

Point data colours a face by the mean of its corner values, cell data by its owning cell's value — for a volume mesh, found through the skin's `surface:parent_cell` provenance. `--color-by` with any other output format is an error, as is any of the modifier flags without `--color-by`. See the [SVG](./formats/svg.md#data-driven-colouring) and [TikZ](./formats/tikz.md) format pages for the full semantics.

**Examples:**

```sh
meshioplusplus convert mesh.msh mesh.vtu
meshioplusplus convert -i gmsh -o vtk mesh.msh mesh.vtk
meshioplusplus convert --ascii mesh.msh mesh.vtu
meshioplusplus convert --sets-to-int-data mesh.inp mesh.xdmf
meshioplusplus convert mesh.msh skin.stl   # volume mesh -> boundary-skin STL

# colour a vector figure by a field
meshioplusplus convert mesh.vtu figure.svg --color-by temperature --colorbar
meshioplusplus convert mesh.vtu figure.tikz --color-by damage --cmap coolwarm \
    --vmin 0 --vmax 1
```

Converting a 3D volume mesh to STL or PLY writes its extracted boundary skin (the writers' default — see [Skin extraction](./extract_skin.md)); converting to SVG or TikZ renders it with the default isometric camera.

---

## meshioplusplus info

Print a summary of a mesh file.

```
meshioplusplus info [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |

Output includes: number of points, cell blocks and their types/counts, point/cell/**side** sets (see [Named regions](./regions.md)), point/cell data names, field data names. It also warns if cells reference nonexistent points or if there are unused points.

**Example:**

```sh
meshioplusplus info mesh.msh
```

---

## meshioplusplus quality

Print a per-cell [mesh quality](./mesh_quality.md) report (min/mean/max and counts of inverted/degenerate cells).

```
meshioplusplus quality [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |
| `--output FILE` | `-o` | Also write the metrics into `FILE` as `cell_data` |

**Examples:**

```sh
meshioplusplus quality part.vtu
meshioplusplus quality part.vtu -o part_quality.vtu
```

---

## meshioplusplus extract-surface

Extract the [boundary surface/edges](./extract_surface.md) of a mesh (volume → faces, 2D surface → edges) and write it out.

```
meshioplusplus extract-surface [options] INFILE OUTFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |
| `--output-format FORMAT` | `-o` | Force output format |
| `--parent-ids` | `-p` | Record each facet's parent cell id as `cell_data` |

**Examples:**

```sh
meshioplusplus extract-surface part.vtu surface.stl
meshioplusplus extract-surface --parent-ids part.vtu surface.vtu
```

---

## meshioplusplus reorder

[Renumber](./reorder.md) a mesh's nodes/elements to reduce matrix bandwidth (RCM) or improve cache locality (Morton / Hilbert), as a pure permutation.

```
meshioplusplus reorder [options] INFILE OUTFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--method METHOD` | `-m` | `rcm` (default), `morton`, or `hilbert` |
| `--report` | `-r` | Print the connectivity bandwidth before and after |
| `--input-format FORMAT` | `-i` | Force input format |
| `--output-format FORMAT` | `-o` | Force output format |

**Examples:**

```sh
meshioplusplus reorder part.vtu reordered.vtu
meshioplusplus reorder part.vtu reordered.vtu --method hilbert
meshioplusplus reorder part.vtu reordered.vtu --method rcm --report
```

---

## meshioplusplus diff

[Compare](./diff.md) two meshes and report whether they are equivalent within a tolerance. The **exit code is nonzero when the meshes differ** and zero when they are equal, so it drops straight into CI / shell scripts / Makefiles.

```
meshioplusplus diff [options] INFILE_A INFILE_B
```

| Option | Short | Description |
|--------|-------|-------------|
| `--atol ATOL` | | Absolute tolerance in `abs_err <= atol + rtol*|expected|` (default `1e-12`) |
| `--rtol RTOL` | | Relative tolerance (default `1e-9`) |
| `--unordered` | | Match points by spatial proximity (tolerant to a shuffled node order) |
| `--exact` | | Only a bitwise-identical result passes (tolerated drift exits nonzero) |
| `--quiet` | `-q` | Print nothing; communicate equality only via the exit code |
| `--input-format-a FORMAT` | | Force the format of the first file |
| `--input-format-b FORMAT` | | Force the format of the second file |

**Examples:**

```sh
meshioplusplus diff a.vtu b.vtu
meshioplusplus diff a.vtu b.vtu --atol 1e-8 --rtol 1e-6
meshioplusplus diff a.msh b.vtu --unordered
meshioplusplus diff expected.vtu actual.vtu --quiet || echo "regression!"
```

---

## meshioplusplus merge

Merge two or more mesh files into one.

```
meshioplusplus merge [options] FILE... OUTFILE
```

Takes two or more input meshes followed by the output file.

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format (applied to every input) |
| `--output-format FORMAT` | `-o` | Force output format |
| `--weld` | | Merge coincident nodes within `--atol` |
| `--atol ATOL` | | Coincidence tolerance for `--weld` (default `1e-8`) |
| `--data-policy POLICY` | | `intersection` (default, keep only data keys present in every input) or `fill` (keep every key, filling missing rows with NaN) |
| `--drop-duplicate-cells` | | With `--weld`, drop cells that become identical after welding |
| `--no-source-tag` | | Do not add the per-cell `source_mesh_id` tag |
| `--quiet` | `-q` | Do not print the merge summary |

Prints a summary of points/cells in and out (and points welded, with `--weld`) unless `--quiet` is given.

**Examples:**

```sh
meshioplusplus merge a.vtu b.vtu merged.vtu
meshioplusplus merge a.vtu b.vtu c.vtu merged.vtu --weld --atol 1e-6
meshioplusplus merge a.vtu b.vtu merged.vtu --data-policy fill
```

---

## meshioplusplus transform

Apply an affine transform to a mesh's point coordinates (see [transform](/transform)).

```
meshioplusplus transform [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--translate x,y,z` | Translation |
| `--scale sx,sy,sz` | Per-axis scale (or a single scalar for uniform) |
| `--rotate axis,deg` | Rotation; `axis` is `x`/`y`/`z` or `nx,ny,nz` (angle in degrees) |
| `--matrix m00,...,m33` | A row-major 4×4 affine matrix (16 values) |
| `--scale-units FACTOR` | Uniform unit-scale factor (e.g. `0.001`) |
| `--rotate-data` | Also rotate vector/tensor `point_data` by the transform |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Give exactly one transform source. Values starting with `-` need the `=` form (`--translate=-1,0,0`).

**Examples:**

```sh
meshioplusplus transform in.vtu out.vtu --translate 1,2,3
meshioplusplus transform in.vtu out.vtu --rotate z,90
meshioplusplus transform in.vtu out.vtu --scale-units 0.001
```

---

## meshioplusplus clean

Weld / prune / de-dup a mesh in one pass (see [clean](/clean)).

```
meshioplusplus clean [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--weld` | Fuse coincident points within `--atol` |
| `--atol ATOL` | Weld tolerance (default `1e-8`) |
| `--remove-orphans` | Drop unused points |
| `--drop-degenerate` | Drop degenerate cells |
| `--drop-duplicates` | Drop exact-duplicate cells |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

With no step flags, the default set runs (remove-orphans + drop-degenerate + drop-duplicates, **no** weld). A removal summary is printed.

**Examples:**

```sh
meshioplusplus clean in.vtu out.vtu
meshioplusplus clean in.vtu out.vtu --weld --atol 1e-6
```

---

## meshioplusplus crop

Extract part of a mesh — inside a bounding box, inside a half-space, or the cells a `cell_data` comparison selects (see [crop](/crop)).

```
meshioplusplus crop [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--bbox xmin,ymin,zmin,xmax,ymax,zmax` | Axis-aligned bounding box |
| `--plane px,py,pz,nx,ny,nz` | Half-space (point + normal), keep `(p−point)·normal ≥ 0` |
| `--where 'NAME OP VALUE'` | A scalar `cell_data` predicate, `OP` one of `<`, `<=`, `>`, `>=`, `==`, `!=`. A non-finite cell value never matches |
| `--mode all\|any` | Keep a cell if ALL (default) or ANY node is inside. `--bbox`/`--plane` only |
| `--record-ids` | Attach original point/cell ids as data arrays |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Give exactly one of `--bbox`/`--plane`/`--where`. Negative values need the `=` form (`--bbox=-1,-1,-1,1,1,1`). `--mode` alongside `--where` is an **error** rather than being ignored: a per-cell value has nothing for an all/any rule to reduce.

**Examples:**

```sh
meshioplusplus crop in.vtu out.vtu --bbox 0,0,0,1,1,1
meshioplusplus crop in.vtu out.vtu --plane 0.5,0,0,1,0,0 --mode any
meshioplusplus crop in.vtu out.vtu --where 'quality:scaled_jacobian < 0.3'

# inside a surface, composed:
meshioplusplus sdf shell.stl field.vtu --resolution 64,64,64 --location center
meshioplusplus crop field.vtu inside.vtu --where 'sdf:distance < 0'
```

---

## meshioplusplus slice

Compute the planar cross-section of a mesh — the intersection with a plane, one dimension below the cut cells (a volume mesh → a `triangle`/`quad` surface, a 2D surface → a `line` mesh). See [slice](/slice).

```
meshioplusplus slice [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--origin x,y,z` | A point on the cutting plane (default `0,0,0`) |
| `--normal x,y,z` | The plane normal (default `0,0,1`; non-zero) |
| `--record-parent-ids` | Attach `slice:parent_cell` (the input cell each section cell was cut from) |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Negative components need the `=` form (`--normal=0,0,-1`). Unlike `crop`, which keeps whole cells on one side, `slice` computes the intersection itself.

**Examples:**

```sh
meshioplusplus slice in.vtu section.vtu --origin 0,0,0.5 --normal 0,0,1
meshioplusplus slice part.msh section.vtu --normal=0,0,-1 --record-parent-ids
```

---

## meshioplusplus isosurface

Compute the level set(s) of a scalar `point_data` field — the data-driven sibling of `slice`, and like it one dimension below the cut cells (a volume mesh → a `triangle`/`quad` surface, a 2D surface → a `line` contour). See [isosurface](/isosurface).

```
meshioplusplus isosurface [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--array NAME` | The `point_data` array to contour (**required**) |
| `--values v1,v2,…` | The isovalues (**required**); sorted ascending, duplicates dropped |
| `--component I` | Component of a multi-component array; the row magnitude by default |
| `--record-parent-ids` | Attach `iso:parent_cell` (the input cell each contour cell was cut from) |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Negative isovalues need the `=` form (`--values=-1.5`). A `cell_data` name is rejected: cell data is piecewise constant and has no level set — convert it with `meshioplusplus data to-point` first. Every contour cell is tagged with `iso:value` (Float64) and `iso:index` (Int64, the ordinal — the integer tag `split --by region --tag …` needs).

**Examples:**

```sh
meshioplusplus isosurface part.vtu shell.vtu --array T --values 350
meshioplusplus isosurface part.vtu shells.vtu --array T --values 300,350,400
meshioplusplus isosurface part.vtu shell.vtu --array v --values=-1.5 --component 2

# one file per contour, via the integer ordinal tag
meshioplusplus split shells.vtu 'contour_{key}.vtu' --by region --tag iso:index
```

---

## meshioplusplus split

Partition a mesh into several files by type, region, named Cell regions, or connected component (see [split](/split)).

```
meshioplusplus split [options] INFILE OUTPATTERN
```

`OUTPATTERN` must contain `{key}`, replaced by each piece's key.

| Option | Description |
|--------|-------------|
| `--by type\|region\|regions\|component` | Split criterion (default `type`) |
| `--tag NAME` | For `--by region`: the integer `cell_data` name to split on (unused by `regions`) |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Prints how many pieces were produced and their sizes. `--by regions` (plural) is one piece per named **Cell** [region](/regions) and is **not a partition** — a cell in several regions lands in several pieces, and `Point`/`Side` regions produce no piece at all. Run [`meshioplusplus regions`](#meshioplusplus-regions) first to see what a mesh's regions are.

**Examples:**

```sh
meshioplusplus split in.vtu 'out_{key}.vtu' --by type
meshioplusplus split in.vtu 'out_{key}.vtu' --by component
meshioplusplus split in.inp 'part_{key}.vtu' --by regions
```

---

## meshioplusplus regions

List a mesh's named [regions](/regions) — name, kind, dimension, tag, and entry count (not the entries themselves).

```
meshioplusplus regions [options] INFILE
```

| Option | Description |
|--------|-------------|
| `--input-format` (`-i`) | Force input format |
| `--json` | Emit the regions as JSON |

Goes through the same cheap path `info --fast`/`read_metadata` use rather than a full read: whenever the summary already comes from an in-memory mesh (every format lacking a native metadata path, plus Exodus, which always falls back), regions cost nothing extra to report; a native metadata path (VTU/VTP/XDMF/Gmsh 4.1) reports none, since none of those currently map regions at all.

**Example:**

```sh
meshioplusplus regions bracket.inp
# <meshio++ mesh regions> (2)
#   fixed (point, 12 entries, tag=1)
#   solid (cell, 340 entries, dim=3, tag=2)
```

---

## meshioplusplus stats

Print geometric statistics of a mesh (see [stats](/stats)).

```
meshioplusplus stats [options] INFILE
```

| Option | Description |
|--------|-------------|
| `--json` | Emit the statistics as JSON |
| `--input-format` (`-i`) | Force input format |

Prints the bounding box, extent, centroid, per-cell-type counts, total area, signed/unsigned volume, and inverted-cell count. This complements `info` (which is topological) with geometric measures.

**Examples:**

```sh
meshioplusplus stats mesh.vtu
meshioplusplus stats mesh.vtu --json
```

---

## meshioplusplus convert-cells

Convert a mesh's element representation (see [convert_cells](/convert_cells)). Distinct from `convert`, which changes the *file format*.

```
meshioplusplus convert-cells [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--mode linearize\|simplexify\|elevate` | Conversion to perform (default `linearize`) |
| `--record-parent-ids` | Attach `convert:parent_cell` cell_data of the source cell indices |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

`linearize` drops higher-order nodes (`tetra10` → `tetra`) and prunes the points that become unreferenced; `simplexify` decomposes cells into simplices of the same dimension (`hexahedron` → 6 `tetra`); `elevate` promotes linear cells to serendipity quadratic (`triangle` → `triangle6`), adding a node per unique edge. A polyhedron block under `simplexify`, and `quad9`/`hexahedron27` under `elevate`, are errors.

**Examples:**

```sh
meshioplusplus convert-cells in.msh out.vtu --mode linearize
meshioplusplus convert-cells in.msh out.vtu --mode simplexify --record-parent-ids
meshioplusplus convert-cells in.msh out.vtu --mode elevate
```

---

## meshioplusplus subdivide

Polyhedrally refine a mesh: split every eligible 3D cell into one polyhedral child per face, connected to a new interior point (see [subdivide](/subdivide)). Distinct from `refine`, which is built on fixed same-type templates and raises by name on a polyhedron.

```
meshioplusplus subdivide [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--record-parent-ids` | Attach `subdivide:parent_cell` cell_data of the source cell indices |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

No per-type template table is needed: tabulated types (reduced to corners for a quadratic variant) and existing polyhedron blocks are handled uniformly. Automatically conforming — no closure flag, unlike `refine`. Non-3D blocks and the full-Lagrange family (no face table) pass through unchanged.

**Example:**

```sh
meshioplusplus subdivide bracket.msh bracket_subdivided.vtu --record-parent-ids
```

---

## meshioplusplus agglomerate

Polyhedrally coarsen a mesh: merge groups of cells into single larger polyhedral cells via greedy seed-and-grow over the shared-face dual (see [agglomerate](/agglomerate)). Distinct from `decimate`, whose fixed-template QEM edge collapse has no analogue for merging arbitrary polyhedral cells.

```
meshioplusplus agglomerate [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--target-group-size N` | Approximate member cells per output group (default `8`) |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Non-volume blocks pass through unchanged; points are never pruned or renumbered (`clean --remove-orphans` is the follow-up for a minimal point set). Conserves volume exactly. `--target-group-size 1` groups every cell by itself.

**Example:**

```sh
meshioplusplus agglomerate fine.vtu coarse.vtu --target-group-size 8
```

---

## meshioplusplus refine

Refine a mesh, subdividing cells into same-type children — every cell, or a selected subset with a conforming closure (see [refine](/refine)).

```
meshioplusplus refine [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--levels N` | How many times to subdivide (default `1`) |
| `--record-parent-ids` | Attach `refine:parent_cell` cell_data of the original cell indices |
| `--cells i,j,k` | Refine only these global (block-major) cells |
| `--region NAME` | Refine a named region (a cell region selects its cells, a point region every cell touching it; a side region is an error) |
| `--where "EXPR"` | Refine the cells satisfying a threshold on a scalar cell_data array, e.g. `"quality:scaled_jacobian < 0.3"` |
| `--closure redgreen\|propagate\|balanced` | How to resolve hanging nodes (default `redgreen`) |
| `--record-levels` | Attach `refine:level` cell_data of each cell's refinement depth |
| `--record-hierarchy` | Attach `refine:cell_id`/`refine:parent_id` cell_data — the persistent parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps; also forces `refine:entity` to be attached even when the closure leaves no hanging node |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

At most one of `--cells`, `--region` and `--where` may be given; with none, every cell is refined. `--closure redgreen` keeps the extra refinement local — a single refined quadrilateral costs one row of a structured grid, a hexahedron one dual sheet. `--closure propagate` is defined for every cell type but reaches the whole edge-connected component, so on a connected mesh it *is* the uniform refinement. `--closure balanced` does not close at all: it keeps the hanging nodes and only enforces 2:1 balance, so the output is **not conforming** (the constrained nodes are listed in `refine:hanging`) but the cost is bounded by the selection — one cell of a 4×4×4 block costs 7 extra cells, against 61 and 448.

One level splits a `triangle`/`quad` into 4 and a `tetra`/`wedge`/`hexahedron` into 8, inserting nodes at edge, quad-face and body midpoints. Those nodes are shared between neighbouring cells, so the result has no hanging nodes, and `point_data` is interpolated onto them. Higher-order cells, `pyramid`, and ragged polygon/polyhedron blocks have no same-type subdivision and are errors — `convert-cells --mode linearize` (or `--mode simplexify`) first.

Note the cell count grows as `4^levels` (2D) or `8^levels` (3D), so `--levels 3` is already a 512× increase on a volume mesh.

**Examples:**

```sh
meshioplusplus refine in.msh out.vtu
meshioplusplus refine in.msh out.vtu --levels 2
meshioplusplus refine coarse.vtu fine.vtu --levels 2 --record-parent-ids
meshioplusplus refine coarse.vtu graded.vtu --cells 12,13,44 --record-levels
meshioplusplus refine coarse.vtu graded.vtu --where "quality:scaled_jacobian < 0.3"
meshioplusplus refine coarse.vtu fine.vtu --cells 12,13 --record-hierarchy
```

---

## meshioplusplus undo-green

Restore a transitional (green) cell back to its coarse parent, read verbatim from the COARSE mesh — the standard "restore and re-split from scratch" rule for a selective refinement pass over a region a prior pass already closed up (see [green-element undo](/undo_green)). A two-mesh verb, like `interpolate`.

```
meshioplusplus undo-green [options] COARSE FINE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--quiet` (`-q`) | Suppress the undo summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input (both files) / output format |

`FINE` must be the output of a prior `refine COARSE FINE --record-hierarchy --record-levels` call (both flags — `--record-hierarchy` alone does not imply `--record-levels`); it fails by name otherwise, or when a `refine:parent_id` cannot be resolved against `COARSE`'s id space. The six reserved `refine:*` arrays are dropped from the output. Only a single-pass (`--levels 1`) hierarchy is supported.

**Examples:**

```sh
meshioplusplus refine coarse.vtu fine.vtu --cells 12,13 --record-hierarchy --record-levels
meshioplusplus undo-green coarse.vtu fine.vtu restored.vtu
meshioplusplus refine restored.vtu regraded.vtu --cells 44,58 --record-hierarchy --record-levels
```

---

## meshioplusplus decimate

Reduce a surface mesh's face count by quadric-error-metric edge collapse — the resolution-reducing inverse of `refine` (see [decimation](/decimate)).

```
meshioplusplus decimate [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--ratio R` | Fraction of the (triangulated) faces to KEEP, in (0, 1] |
| `--target-faces N` | Absolute face count to stop at (within one collapse) |
| `--max-error E` | Collapse only while the cheapest quadric error is at most `E` |
| `--placement P` | `optimal` (default), `midpoint`, or `endpoint` |
| `--no-preserve-boundary` | Allow boundary vertices to collapse (the outline may change) |
| `--no-preserve-features` | Allow sharp corners/creases to collapse |
| `--feature-angle A` | Degrees between face normals above which a vertex is a feature (default `30`) |
| `--quiet` (`-q`) | Do not print the collapse summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Exactly one of `--ratio`, `--target-faces` and `--max-error` must be given. Surface meshes only: `quad`/`polygon` blocks are triangulated first (the output is all-triangle), and a volume mesh is an error — run `extract-surface` first. Boundary and feature vertices are pinned by default, and the link condition plus a normal-flip guard reject any collapse that would change topology or fold the surface.

**Examples:**

```sh
meshioplusplus decimate scan.stl coarse.stl --ratio 0.25
meshioplusplus decimate skin.vtu coarse.vtu --target-faces 5000
meshioplusplus decimate skin.vtu coarse.vtu --max-error 1e-6 --placement midpoint
meshioplusplus decimate open_patch.vtu out.vtu --ratio 0.1 --no-preserve-features -q
```

---

## meshioplusplus decimate-volume

Reduce a tetrahedral mesh's cell count by quadric-error-metric **tet**-edge collapse — the volume-mesh sibling of `decimate`, a separate verb rather than a mode on it (see [volume decimation](/decimate_volume)).

```
meshioplusplus decimate-volume [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--ratio R` | Fraction of the tets to KEEP, in (0, 1] |
| `--target-cells N` | Absolute tet count to stop at (within one collapse) |
| `--max-error E` | Collapse only while the cheapest boundary-touching quadric error is at most `E` |
| `--placement P` | `optimal` (default), `midpoint`, or `endpoint` |
| `--preserve-boundary` | Pin every boundary vertex outright, reproducing `decimate`'s own default instead of letting boundary vertices participate |
| `--no-preserve-features` | Allow sharp boundary corners/creases to collapse |
| `--feature-angle A` | Degrees between boundary-triangle normals above which a vertex is a feature (default `30`) |
| `--quiet` (`-q`) | Do not print the collapse summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Exactly one of `--ratio`, `--target-cells` and `--max-error` must be given. Tet meshes only: a non-tetra 3D block is an error — run `convert-cells --mode simplexify` first. Note `--preserve-boundary` is opt-**in** here, the mirror image of `decimate`'s opt-out `--no-preserve-boundary` — boundary vertices participate in decimation by real quadric error by default. Validity guards reject any collapse that would change topology, invert a tet, or (for boundary-touching collapses) fold the outer surface.

**Examples:**

```sh
meshioplusplus decimate-volume solid.vtu coarse.vtu --ratio 0.25
meshioplusplus decimate-volume solid.vtu coarse.vtu --target-cells 5000
meshioplusplus decimate-volume solid.vtu coarse.vtu --max-error 1e-6 --placement midpoint
meshioplusplus decimate-volume solid.vtu coarse.vtu --ratio 0.1 --preserve-boundary -q
```

---

## meshioplusplus remesh

Replace a **surface** mesh's own triangulation with a new, near-uniformly-sized, well-shaped one at a caller-chosen vertex count, by approximated centroidal Voronoi diagram (ACVD) clustering (see [remeshing](/remesh)).

```
meshioplusplus remesh [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--num-clusters N` | Target output vertex count (required) |
| `--subdivide N` | Uniform `refine` passes applied before clustering; `-1` (default) auto-picks from `--subsample-ratio` |
| `--subsample-ratio R` | Target items per cluster driving the auto-`subdivide` pick (default `10`) |
| `--max-subdivide N` | Cap on the auto-picked subdivide count (default `4`) |
| `--iterations N` | Energy-minimisation sweeps (default `10`) |
| `--repair-passes N` | Max repair passes for disconnected/non-manifold clusters (default `3`) |
| `--metric isotropic\|quadric\|anisotropic` | Clustering metric (default `isotropic`) |
| `--gradation G` | Curvature-weighting exponent; `0` (default) disables curvature weighting entirely |
| `--no-preserve-boundary` | Let boundary-adjacent triangles drop from the dual instead of emitting a boundary `line` block |
| `--max-anisotropy A` | `metric anisotropic` only: cap on a vertex's principal-curvature-length ratio (default `4`) |
| `--quiet` (`-q`) | Do not print the clustering summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

The output is a brand-new mesh with new points and new connectivity — there is no point/cell map, and `point_data`/`cell_data`/named regions are dropped (`field_data` passes through); transfer a field with `interpolate`/`conservative-interpolate` afterwards. `--metric anisotropic` shapes clusters with a per-vertex curvature tensor rather than isotropic distance, so elongated features (a fillet, a pipe, a rib) get elongated elements; `--max-anisotropy` is an error under any other `--metric`.

**Examples:**

```sh
meshioplusplus remesh bracket.stl out.vtu --num-clusters 5000
meshioplusplus remesh bracket.stl out.vtu --num-clusters 5000 --metric quadric
meshioplusplus remesh bracket.stl out.vtu --num-clusters 5000 --metric anisotropic --max-anisotropy 4
meshioplusplus remesh bracket.stl out.vtu --num-clusters 2000 --gradation 1.5 --no-preserve-boundary -q
```

---

## meshioplusplus remesh-volume

Retetrahedralize a **volume** mesh (or a closed surface) at a caller-chosen resolution by isosurface stuffing over a body-centered cubic (BCC) lattice — the volumetric sibling of `remesh` (see [volumetric remeshing](/remesh_volume)). Unlike `remesh`, this accepts a volume mesh directly; its boundary is extracted internally.

```
meshioplusplus remesh-volume [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--resolution nx,ny,nz` | Cell counts of the root lattice (exactly one of `--resolution`/`--cell-size`) |
| `--cell-size S` | Cubic cell size of the root lattice |
| `--bounds xlo,ylo,zlo,xhi,yhi,zhi` | Explicit bounds; the mesh's own bounding box by default (negatives need the `--bounds=` form) |
| `--padding P` | Grow the box by this on every side, in world units |
| `--padding-relative R` | Grow the box by this fraction of its diagonal (default `0.1`) |
| `--max-cells N` | Refuse to generate a root lattice above this many cells |
| `--max-tets N` | Refuse an output with more tets than this (checked after cutting; unlike `--max-cells`, has no "lifts the limit" value) |
| `--warp-fraction F` | Fraction of a lattice vertex's shortest incident edge within which it may be warped onto the surface; `0` disables warping (default `0.35`; negatives need the `--warp-fraction=` form) |
| `--sign pseudonormal\|winding-number` | How lattice vertices are classified inside/outside |
| `--watertight-check off\|warn\|error` | What to do about a non-watertight input surface (default `warn`) |
| `--quiet` (`-q`) | Do not print the run summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

The output is a brand-new single-`tetra`-block mesh with new points and new connectivity — there is no point/cell map, and `point_data`/`cell_data`/named regions are dropped (`field_data` passes through); transfer a field with `interpolate`/`conservative-interpolate` afterwards. Warping (`--warp-fraction`) trades a small, measured chance of non-manifold boundary edges (reported as `non-manifold edges` in the summary) for substantially better boundary tet quality; `--warp-fraction 0` gives an exactly watertight but lower-quality boundary. See [the measured tradeoff](/remesh_volume#the-warp-quality-tradeoff-measured).

**Examples:**

```sh
meshioplusplus remesh-volume part.vtu out.vtu --cell-size 0.5
meshioplusplus remesh-volume part.vtu out.vtu --resolution 64,64,64 --warp-fraction 0.2
meshioplusplus remesh-volume surface.stl out.vtu --cell-size 0.3 --watertight-check error
```

---

## meshioplusplus optimize-volume

ODT-remesh a **tetrahedral** mesh: raise its worst element quality by relocating vertices AND flipping connectivity (2-3/3-2, predicate-free) — the genuine "ODT remeshing" sibling of `remesh-volume` (which generates a fresh lattice mesh) and of `smooth --method odt` (which only moves points on fixed connectivity). See [ODT remeshing](/optimize_volume).

```sh
meshioplusplus optimize-volume [options] INFILE OUTFILE
```

| Option | Description |
| --- | --- |
| `--max-iterations N` | optimisation sweeps (relocation + flips); stops early at a fixed point (default: 10) |
| `--no-relocate` | skip the ODT vertex-relocation half (flips only) |
| `--no-flip` | skip the topological-flip half (reduces to ODT smoothing) |
| `--no-preserve-boundary` | allow boundary vertices to move during relocation (may drift off the surface) |
| `--min-improvement E` | strict scaled-Jacobian gain a flip must deliver to be accepted (default: 1e-6) |
| `--quiet`, `-q` | suppress the summary |

The point set is invariant, so `point_data`/`field_data` and named Point regions carry through; `cell_data` and Cell/Side regions are dropped (a flip has no cell correspondence). With `--no-preserve-boundary` off (the default) the boundary surface is exactly preserved. Tet-only: a non-`tetra` block errors pointing at `convert-cells --mode simplexify`.

**Examples:**

```sh
meshioplusplus optimize-volume volume.vtu optimized.vtu
meshioplusplus optimize-volume volume.vtu optimized.vtu --max-iterations 20
```

---

## meshioplusplus smooth

Relax point coordinates toward their edge-neighbour centroids to improve element shape (see [smoothing](/smooth)).

```
meshioplusplus smooth [options] INFILE OUTFILE
```

| Option | Description |
|--------|-------------|
| `--method taubin\|laplacian\|odt` | Smoothing operator (default `taubin`, which does not shrink; `laplacian` is stronger per pass but shrinks; `odt` is optimal-Delaunay-triangulation smoothing, tet-only and C++-core only, moving each free interior vertex toward the volume-weighted average of its incident tets' circumcenters) |
| `--iterations N` | How many iterations to run; for `taubin` one iteration is two passes (default `10`) |
| `--lambda L` | Relaxation factor of the smoothing pass, in (0, 1); the default depends on `--method` (`0.5` laplacian, `0.33` taubin, `0.9` odt) |
| `--mu=M` | `taubin` only: un-shrinking factor, must satisfy `mu < -lambda < 0` (default `-0.34`; the negative value needs the `--mu=` form); silently ignored by `laplacian` and `odt` |
| `--no-fix-boundary` | Let boundary nodes move too (they are pinned by default) |
| `--no-preserve-features` | Do not pin feature nodes (sharp corners/creases are kept by default) |
| `--feature-angle A` | Angle in degrees between boundary facet normals above which their shared nodes are pinned as features (default `30`) |
| `--no-guard-inversion` | Do not reject moves that would invert an incident cell |
| `--quiet` (`-q`) | Suppress the summary output |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Only the point coordinates move: connectivity, `cell_data`, `field_data` and `point_data` values come through unchanged, the point and cell counts are unchanged, and the points array keeps its input dtype. Neighbours are the nodes joined by an actual cell *edge*, so a structured hex block is a fixed point. Nodes of blocks whose edge topology is unknown — the higher-order family, the VTK-Lagrange types, `custom` — are pinned rather than guessed at. Unless suppressed, the summary reports the number of nodes moved, the largest net displacement, and how many moves the inversion guard rejected.

**Examples:**

```sh
meshioplusplus smooth noisy.vtu smooth.vtu
meshioplusplus smooth noisy.vtu smooth.vtu --iterations 40
meshioplusplus smooth in.msh out.vtu --method laplacian --lambda 0.4
meshioplusplus smooth in.msh out.vtu --mu=-0.4 --feature-angle 45
meshioplusplus smooth in.msh out.vtu --no-fix-boundary --no-guard-inversion -q
meshioplusplus smooth tangled.vtu out.vtu --method odt --iterations 10
```

---

## meshioplusplus interpolate

Sample data arrays from a SOURCE mesh onto a TARGET mesh (see [interpolation](/interpolate)). The output is a copy of the target — geometry, connectivity and its own data preserved exactly — with the requested source arrays sampled onto it.

```
meshioplusplus interpolate [options] SOURCE TARGET OUTFILE
```

| Option | Description |
|--------|-------------|
| `--method nearest\|barycentric` | Nearest source point (default; dtype-preserving) or linear interpolation in a simplexified source (exact on a linear field; Float64) |
| `--arrays a,b` | Comma-separated source array names to transfer (default: every source `point_data` array; `cell_data` transfers only when named) |
| `--extrapolate` | `barycentric` only: give a target point outside the source domain the nearest source point's value instead of the default value |
| `--default-value=V` | `barycentric` only: fill value for target points outside the source domain (default `0`; negative values need the `--default-value=` form) |
| `--on-conflict error\|overwrite\|suffix` | What to do when a transferred name already exists on the target (default `error`; `suffix` writes to `NAME_interp`) |
| `--quiet` (`-q`) | Suppress the transfer summary |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input (both files) / output format |

Source `point_data` is sampled at the target's points, source `cell_data` at the target's cell centroids — always by nearest source-cell centroid, whatever the method. Under `barycentric` the source is simplexified first, so on a quad/hex source the result is the simplex-linear interpolant, and triangle sources are evaluated in the xy-plane (use `nearest` for a curved surface embedded in 3D).

**Examples:**

```sh
meshioplusplus interpolate coarse.vtu fine.vtu out.vtu
meshioplusplus interpolate coarse.vtu fine.vtu out.vtu --method barycentric
meshioplusplus interpolate a.msh b.msh out.vtu --arrays T,v --on-conflict suffix
meshioplusplus interpolate a.msh b.msh out.vtu --method barycentric --extrapolate
```

---

## meshioplusplus partition

Decompose a mesh into N balanced parts for domain decomposition (see [partitioning](/partition)) — the count-driven complement to `split`.

```
meshioplusplus partition [options] INFILE OUTPATTERN
```

`OUTPATTERN` must contain `{part}` (e.g. `out_{part}.vtu`), expanded once per piece — except with `--labels-only`, where it is a single plain path.

| Option | Description |
|--------|-------------|
| `--nparts N` (`-n`) | Number of parts (required, `>= 1`) |
| `--method M` | `sfc` (Hilbert curve cut, always available), `kahip` (KaHIP `kaffpa`; needs a KaHIP-enabled build, fails by name otherwise), or `auto` (default: kahip when available, else sfc) |
| `--imbalance F` | KaHIP only: allowed imbalance fraction in (0, 1) (default `0.03`) |
| `--mode M` | KaHIP only: `fast` / `eco` / `strong` (default `eco`) |
| `--seed N` | KaHIP only: random seed (deterministic per seed; default `0`) |
| `--weights NAME` | Scalar `cell_data` array of per-cell weights to balance instead of the cell count |
| `--record-ids` | Attach `partition:original_point_id` / `partition:original_cell_id` arrays to every piece |
| `--labels-only` | Write the input once with the Int64 `partition:part` cell_data attached instead of writing pieces |
| `--ghost-layers N` | Reserved; only `0` is supported |
| `--input-format` / `--output-format` (`-i`/`-o`) | Force input/output format |

Every piece keeps the input's cell-block structure 1:1 (empty blocks included, unlike `split`), so the pieces recombine into the input — each cell lands in exactly one piece.

**Examples:**

```sh
meshioplusplus partition domain.msh 'domain_{part}.vtu' --nparts 4
meshioplusplus partition domain.msh 'domain_{part}.vtu' -n 16 --method kahip --mode strong
meshioplusplus partition domain.msh labelled.vtu --nparts 4 --labels-only
meshioplusplus partition domain.msh 'p_{part}.vtu' -n 8 --weights cost --record-ids
```

---

## meshioplusplus data

A nested group of ten verbs operating on a mesh's `point_data` / `cell_data` / `field_data` arrays (see [data operations](/data_operations)). **The geometry is never modified** by any of them — points, connectivity, block order and block types come through bit-identical.

```
meshioplusplus data <subcommand> [options]
```

| Subcommand | Description |
|------------|-------------|
| `info` | Summarize every data array (see [data summary](/data_info)) |
| `rename` | Rename data arrays (see [array management](/data_manage)) |
| `drop` | Drop data arrays by name |
| `keep` | Keep only the named data arrays |
| `to-cell` | Average `point_data` onto the cells (see [averaging](/data_average)) |
| `to-point` | Average `cell_data` onto the points |
| `calc` | Derive an array from an expression (see [expressions](/data_calc)) |
| `clamp` | Clamp values into a range (see [conditioning](/data_condition)) |
| `normalize` | Rescale values to a target range |
| `gradient` | Differentiate a `point_data` field (see [field derivatives](/gradient)) |
| `hessian` | Second derivative of a scalar `point_data` field (see [second derivatives](/hessian)) |
| `estimate-error` | ZZ recovery-based error indicator, plus marking (see [error estimation](/error)) |
| `integrate` | Cell-measure-weighted total/mean over cells, per region (see [field integration](/field_integration)) |
| `export` | Export the arrays to Parquet (see [interoperability](/interop)) |
| `export-dataset` | Export a *set* of meshes as one `mesh_id`-keyed dataset (see [ML data handling](/ml)) |

Every verb takes `--input-format` (`-i`), and every verb but `info` and `integrate` takes an `OUTFILE` and `--output-format` (`-o`) — the mesh is never modified by either of those two, so there is nothing to write. `export` and `export-dataset` are the exceptions on the output side: they write Parquet / zarr / hdf5, so they take no `--output-format` (`export-dataset` has `--format parquet|zarr|hdf5` instead, plus `--mesh-id stem|index`, and its input is several paths, one quoted glob, or one multi-step file — the sequence source language). Both are **Python CLI only**; both need the matching optional extra.

::: tip `data gradient`, `data hessian`, `data estimate-error` and `data integrate` are mesh operations
Every other verb in this group belongs to the `data_*` family, which by definition never touches geometry. `gradient` consumes and produces data arrays but **reads** geometry and topology (face areas, cell volumes, cell adjacency), so it lives in the mesh-operations layer; `hessian` is `gradient`'s companion one order further, a composition of two `gradient` calls; `estimate-error` composes `gradient` itself; `integrate` reads the same cell measures to weight its totals. All four are grouped here because that is where a user looks for them. See [field derivatives](/gradient), [second derivatives](/hessian), [error estimation](/error) and [field integration](/field_integration).
:::

::: warning `data export` is not a mesh conversion
It writes `point_data` / `cell_data` to Parquet **for analytics** (pandas, polars, DuckDB) and does not round-trip geometry — Parquet is deliberately not in the format registry, so `meshioplusplus convert mesh.vtu out.parquet` does not work. It needs `pip install meshioplusplus[arrow]`, and it exists in the **Python CLI only**: the native binary has no counterpart.
:::

::: warning Colons in names
Data names routinely contain colons (`gmsh:physical`). `data rename` therefore splits its `OLD:NEW` value on the **last** colon — `--point gmsh:physical:tag` renames `gmsh:physical` to `tag`. `data calc` splits `NAME = EXPR` on the **first** `=`. `drop`/`keep` take a comma-separated name list with no prefix, so colons there are unambiguous. The Python CLI and the native binary implement identical rules.
:::

### data info

| Option | Description |
|--------|-------------|
| `--json` | Emit the summary as JSON |

Prints location, name, dtype, component count, entry count, min/max/mean and NaN/inf counts for every array. Read-only.

### data rename / drop / keep

| Option | Description |
|--------|-------------|
| `--point`, `--cell`, `--field` | `OLD:NEW` for `rename` (repeatable); a comma-separated name list for `drop`/`keep` |
| `--ignore-missing` | Skip names that do not exist instead of failing (`drop`/`keep`) |

For `keep`, a location that is not named at all is left untouched; naming it with an empty list drops everything there.

### data to-cell / to-point

| Option | Description |
|--------|-------------|
| `--keys` | Comma-separated names to convert (default: all at the source location) |
| `--target-suffix` | Append this to each output name (default: keep the same name) |
| `--weighted` | `to-point` only: weight by cell measure (area/volume) instead of counting cells equally |

The output is always `float64` — the mean of an integer field is not an integer.

### data calc

| Option | Description |
|--------|-------------|
| `--point`, `--cell`, `--field` | `NAME = EXPRESSION` (repeatable) |
| `--overwrite` | Allow replacing an array that already exists |

The expression grammar accepts `+ - * /`, unary minus, parentheses, numeric literals, array names, and `abs`/`sqrt`/`min`/`max`/`norm` — nothing else is evaluated.

### data export

| Option | Description |
|--------|-------------|
| `--location` | `point` (default) or `cell` |

### data clamp / normalize

| Option | Description |
|--------|-------------|
| `--point`, `--cell`, `--field` | Comma-separated names (default: all at that location) |
| `--min`, `--max` | `clamp` only: the bounds (both required) |
| `--to LO,HI` | `normalize` only: target range (default `0,1`) |
| `--zero-mean` | `normalize` only: standardize to zero mean / unit std instead |
| `--magnitude` | Condition by row magnitude instead of per component |
| `--nan` | `ignore` (default), `replace` or `fail` |
| `--nan-value` | Replacement used with `--nan replace` |
| `--suffix` | Store as `NAME+SUFFIX` instead of replacing in place |

### data gradient

| Option | Description |
|--------|-------------|
| `--array NAME` | The `point_data` array to differentiate (**required**) |
| `--op` | `gradient` (default), `divergence` or `curl` |
| `--method` | `green-gauss` (default) or `least-squares` |
| `--location` | `cell` (default) or `point` |
| `--output NAME` | Output array name (default `<array>:<op>`) |
| `--component I` | Differentiate only this component (gradient only; default all) |
| `--overwrite` | Replace an existing array of the output name instead of failing |
| `--quiet`, `-q` | Suppress the summary |

Naming a `cell_data` array is an error pointing at `data to-point`: a piecewise constant field has no derivative. Divergence and curl need a 2- or 3-component field. Cells that cannot be differentiated are reported as `cells skipped (NaN)`; least-squares cells with a degenerate neighbourhood fall back to Green-Gauss and are reported separately. See [field derivatives](/gradient) for the exactness guarantees and caveats.

### data hessian

| Option | Description |
|--------|-------------|
| `--array NAME` | The scalar `point_data` array to differentiate twice (**required**) |
| `--method` | `green-gauss` (default) or `least-squares`, forwarded to both internal `gradient` passes |
| `--location` | `cell` (default) or `point` |
| `--output NAME` | Output array name (default `<array>:hessian`) |
| `--overwrite` | Replace an existing array of the output name instead of failing |
| `--quiet`, `-q` | Suppress the summary |

The Hessian (second derivative) of a scalar `point_data` field — `gradient`'s companion one order further, a composition of two `gradient` calls, not a new kernel. Naming a `cell_data` array is an error pointing at `data to-point`, for the same reason `data gradient` rejects one; a multi-component array is an error naming the per-component workaround (Hessian is scalar-only). Cells that cannot be evaluated are reported as `cells skipped (NaN)`. See [second derivatives](/hessian) for the exactness guarantees and the curvature-driven refinement composition.

### data estimate-error

| Option | Description |
|--------|-------------|
| `--array NAME` | The `point_data` array to estimate the error of (**required**) |
| `--method` | `zz` (default; the only estimator family today) |
| `--marking` | `none` (default), `absolute`, `fraction` or `dorfler` |
| `--marking-value V` | Meaning depends on `--marking`: an absolute threshold, a fraction in `(0, 1]` of cells, or the Dörfler bulk fraction theta in `(0, 1]` (ignored for `none`) |
| `--output NAME` | Indicator array name (default `error:zz`) |
| `--marked NAME` | Marking array name (default `error:marked`; ignored when `--marking none`) |
| `--overwrite` | Replace an existing array of an output name instead of failing |
| `--quiet`, `-q` | Suppress the summary |

The Zienkiewicz-Zhu recovery-based error indicator of a `point_data` field: a composition of `gradient` and the point↔cell averaging round trip, not a new kernel. Naming a `cell_data` array is an error pointing at `data to-point`, for the same reason `data gradient` rejects one. Cells that cannot be evaluated are reported as `cells skipped (NaN)` and read `NaN` in `error:zz`, `0` in `error:marked`. With `--marking` not `none`, a second `cell_data` array is attached so [`refine`](/refine)'s own `--where` selector needs no change at all — the intended use is `meshioplusplus refine estimated.vtu adapted.vtu --where "error:marked > 0.5"`. See [error estimation](/error) for the composition, the marking policies and the byte-identity tolerance.

### data integrate

| Option | Description |
|--------|-------------|
| `--array NAME` | `cell_data` array to integrate (repeatable; default all `cell_data` arrays) |
| `--json` | Emit the report as JSON |

Cell-measure-weighted total and mean of one or more `cell_data` arrays — `gradient`'s integration counterpart. Every sum is weighted by the cell's own length/area/volume; a cell whose measure is not computable, or a component whose value is non-finite, is excluded from that component's numerator **and** denominator, never given a fallback weight of 1. Reported for the whole mesh and independently for every named `Cell` region — a cell in two regions contributes fully to both, one in none contributes to neither. Naming a `point_data` array is an error pointing at `data to-cell`. Read-only, like `data info`: there is no `OUTFILE`. See [field integration](/field_integration).

**Examples:**

```sh
meshioplusplus data info mesh.vtu
meshioplusplus data info mesh.vtu --json

meshioplusplus data gradient in.vtu out.vtu --array T
meshioplusplus data gradient in.vtu out.vtu --array u --op curl --location point
meshioplusplus data gradient in.vtu out.vtu --array T --method least-squares --output dT

meshioplusplus data hessian in.vtu out.vtu --array T
meshioplusplus data hessian in.vtu out.vtu --array T --location point

meshioplusplus data estimate-error in.vtu out.vtu --array T
meshioplusplus data estimate-error in.vtu out.vtu --array T --marking dorfler --marking-value 0.6

meshioplusplus data integrate mesh.vtu --array density
meshioplusplus data integrate mesh.vtu --array density --array pressure --json

meshioplusplus data rename in.vtu out.vtu --point T:temperature
meshioplusplus data drop   in.vtu out.vtu --point a,b --cell c
meshioplusplus data keep   in.vtu out.vtu --point T,p --cell mat

meshioplusplus data to-cell  in.vtu out.vtu --keys T,p --target-suffix _c
meshioplusplus data to-point in.vtu out.vtu --keys stress --weighted

meshioplusplus data calc in.vtu out.vtu --point "speed = norm(velocity)"
meshioplusplus data calc in.vtu out.vtu --cell  "dp = p_new - p_old"

meshioplusplus data clamp     in.vtu out.vtu --point T --min 0 --max 100
meshioplusplus data normalize in.vtu out.vtu --cell damage --to 0,1
meshioplusplus data normalize in.vtu out.vtu --point T --zero-mean

meshioplusplus data export in.vtu points.parquet
meshioplusplus data export in.vtu cells.parquet --location cell
```

---

## meshioplusplus dataset

The second nested group: curate a hand-editable [dataset manifest](datasets.md) — the JSON cataloguing many cases (each possibly a time series) with splits, tags, groups and notes. Python CLI only, like `data export`. Every mutating verb is load → mutate → save against the same file a text editor uses, so hand edits made between two CLI calls survive; sources given on the command line are stored **relative to the manifest's directory** (absolute paths stay absolute), keeping the manifest portable.

```
meshioplusplus dataset <subcommand> [options]
```

| verb | does |
|---|---|
| `add MANIFEST SOURCE...` | add a case — one quoted glob, one file, or several paths; `--id` (default: the stem), `--format`, `--times T,T`, `--time-from`, `--sort`, plus curation `--split`/`--tag` (repeatable)/`--group`/`--notes`/`--meta K=V` (repeatable; `V` parses as JSON when it can). The source is expanded once so an empty glob fails now, by name (`--no-validate` skips). Creates the manifest file if absent. `--target SOURCE` (repeatable) records a paired coarse/fine series, with its own `--target-format`/`--target-times`/`--target-time-from`/`--target-sort`; the two must have the same steps at the same instants, checked here. Omit it for the ordinary self-supervised case |
| `list MANIFEST` | entries filtered by `--split`/`--tag`/`--group`; `--resolve` expands each plan (checks files exist, reads no mesh); `--json` emits the entries (plus `Resolved` plans) as JSON |
| `split MANIFEST` | `--set S` on `--id` (repeatable) or `--all`; or `--assign train=0.8,valid=0.1,test=0.1` over every entry — deterministic (`--seed`), `--by-group` keeps entries sharing a `Group` together |
| `tag MANIFEST` | `--add T,T` / `--remove T,T` on `--id` (repeatable) or `--all` |
| `annotate MANIFEST --id ID` | set `--notes`, `--group`, merge `--meta K=V`, drop `--del-meta K` |

```sh
meshioplusplus dataset add m.json 'runs/c42/out_*.vtu' --split train --meta Re=100
meshioplusplus dataset add m.json a.vtu b.vtu --id pair --tag coarse
meshioplusplus dataset add m.json 'coarse/*.vtu' --target 'fine/*.vtu' --id sr
meshioplusplus dataset split m.json --assign train=0.8,valid=0.1,test=0.1 --seed 0
meshioplusplus dataset list m.json --split train --resolve
meshioplusplus dataset annotate m.json --id pair --notes "restarted at t=0.3"
```

---

## meshioplusplus pipeline

Run a whole [settings pipeline](pipeline.md): read `Input.Path`, apply the `Operations` chain, write `Output.Path` — one `settings.json` instead of N verb invocations with intermediate files.

```bash
meshioplusplus pipeline settings.json
meshioplusplus pipeline settings.json --input other.msh --output out.vtu
meshioplusplus pipeline settings.json --json          # machine-readable report
meshioplusplus pipeline settings.json --quiet
```

A document whose `Input` is a `Pattern`/`Paths`, or whose `Output.Path` carries `{step}`/`{index}`, runs the chain **per step** over a whole transient dataset — see [sequences](sequences.md). The verb routes it automatically; a plain single-file document takes the unchanged path.

- `--input` / `--output` override the two paths in the settings file (the document itself is untouched).
- The report lists each step with its counters (`step 3: Clean (PointsWelded=12, ...)`) plus any warnings; `--json` prints the same as JSON.
- Parsing is strict: an unknown op, an unknown key, or an `Output` option the format cannot honour is an error naming the offender.
- The exit code is nonzero on any error, so the verb composes with `make`/CI.

The verb exists in **both** CLIs. The Python CLI runs the pure-Python engine (and so inherits the per-format Python fallbacks); the native CLI needs a build with the JSON parser (`-DMESHIOPLUSPLUS_WITH_JSON=ON`, the default when the `src/cpp/third_party/json` submodule is checked out — release binaries carry it) and otherwise reports the flag by name. See [the settings pipeline](pipeline.md) for the schema and the full op table.

---

## meshioplusplus compress

Compress the data in a mesh file (formats that support compression, e.g. VTU).

```
meshioplusplus compress [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |

---

## meshioplusplus decompress

Decompress the data in a mesh file.

```
meshioplusplus decompress [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |

---

## meshioplusplus ascii

Convert a mesh file to its ASCII representation (in-place).

```
meshioplusplus ascii [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |

---

## meshioplusplus binary

Convert a mesh file to its binary representation (in-place).

```
meshioplusplus binary [options] INFILE
```

| Option | Short | Description |
|--------|-------|-------------|
| `--input-format FORMAT` | `-i` | Force input format |

---

## Format names

The `--input-format` and `--output-format` options accept any of the registered format names. The full list is shown by `meshioplusplus convert --help`. Common values:

`abaqus`, `ansys`, `avsucd`, `cgns`, `dolfin-xml`, `exodus`, `flac3d`, `gmsh`, `gmsh22`, `h5m`, `hmf`, `mdpa`, `med`, `medit`, `nastran`, `netgen`, `obj`, `off`, `permas`, `ply`, `stl`, `su2`, `svg`, `tecplot`, `tetgen`, `ugrid`, `vtk`, `vtk42`, `vtk51`, `vtu`, `wkt`, `xdmf`

## Selective reads and fast summaries

`info --fast` summarizes a file from its header instead of loading it, and `convert` can narrow what it reads:

```bash
meshioplusplus info --fast big.vtu
meshioplusplus convert --points-only in.vtu out.vtu     # geometry, no data arrays
meshioplusplus convert --arrays u,p in.vtu out.vtu      # only these data arrays
meshioplusplus convert --time-step=-1 run.exo last.vtu  # the last step of a time series
```

`--points-only` keeps connectivity — it narrows data, not topology. `arrays` with an empty list keeps no arrays; omitting the flag keeps every array.

`--time-step=N` picks one step of a multi-step file: `0` (the default) is the first, negative counts from the end. A negative value needs the `--time-step=-1` form, as with the other negative-valued options. Out of range is an error naming the available count, never a silent clamp; `info --fast` prints `Time steps: N [...]` when a file records more than one. Honoured by formats carrying a time series (currently `exodus`); a format whose reader has no time concept refuses rather than quietly returning the first step.

`--lenient` (**native CLI only**) downgrades "this reader cannot represent construct X" errors to a warning plus a skip — currently MDPA's `Table`, `Geometries`, `Mesh` and `Constraints` blocks, which nearly every production `.mdpa` carries. It is *not* "ignore all errors": a malformed row, a bad node reference or a duplicate node id still fail, because continuing past those returns a mesh that is quietly wrong rather than merely incomplete. The Python CLI has no such flag, deliberately: its MDPA reader is the pure-Python reference, which already accepts every construct the flag covers.

Formats without a header-only path are read in full and `info --fast` says so explicitly (`no header-only path for this format; the file was read in full`) rather than implying a saving that did not happen. See [Selective reads](selective_read.md).

`--points-only`/`--arrays` are rejected alongside `-s`/`-d`, which convert exactly the data arrays that were skipped.

## Compression codecs

```bash
meshioplusplus compress --codec lz4 mesh.vtu
```

`--codec zlib|lz4|zstd` selects the VTK XML block codec for `.vtu`/`.vtp`. zlib is the default; `lz4` stays ParaView-readable, `zstd` is a meshio++ extension that ParaView cannot read. The flag is **rejected** for formats with no block codec rather than silently ignored. See [Compression codecs](codecs.md).

Both CLIs — the Python one and the native `meshioplusplus` binary — accept these identically.

## `view` / `screenshot` — the native viewer

```sh
meshioplusplus view part.msh
meshioplusplus view part.msh --kind surface --color-by material
meshioplusplus screenshot part.msh out.png --size 1600x1200
```

Options: `--input-format/-i`, `--kind {auto,surface,volume,curve,points}`, `--color-by NAME`, `--name NAME`; `screenshot` adds `--size WIDTHxHEIGHT` and `--transparent`.

These mirror the Python CLI's verbs, but in the **native binary** they are only functional in a build configured with [Polyscope](https://polyscope.run):

```sh
git submodule update --init --recursive     # Polyscope vendors its own submodules
build/configure.sh --cli --with-polyscope --build
```

They are listed in `--help` in every build; without the flag they report it rather than silently not existing. The **prebuilt release binaries do not include the viewer** — they are deliberately dependency-free single files, and Polyscope needs OpenGL, GLFW and X11. Use the Python CLI (`pip install meshioplusplus[viewer]`) or the [browser viewer](/viewer) if you would rather not build from source.

## `voxelize`

```bash
meshioplusplus voxelize bunny.stl shell.vtu --resolution 64,64,64 --fill surface
meshioplusplus voxelize bunny.stl solid.vtu --cell-size 0.5 --fill inside
```

| flag | meaning |
|---|---|
| `--resolution nx,ny,nz` | cell counts; give exactly one of this and `--cell-size` |
| `--cell-size S` | cubic cell size |
| `--bounds=xlo,...,zhi` | explicit bounds; the mesh's own by default (negatives need the `=` form) |
| `--padding` / `--padding-relative` | grow the box on every side |
| `--fill all\|surface\|inside` | which cells to keep |
| `--sign pseudonormal\|winding-number` | how `--fill=inside` decides what is inside |
| `--attach-occupancy` | attach the `voxel:occupancy` array |
| `--max-cells N` | refuse above this many cells (default ~256³) |

See [`doc/voxelize.md`](voxelize.md) and [`doc/sdf.md`](sdf.md).

## `grid-sample`, `grid-scatter`, `grid-resample`, `grid-spectrum`

The mesh-to-grid data path a convolutional or superresolution model needs.

```bash
meshioplusplus grid-sample   case.vtu grid.vti --resolution 64,64,64
meshioplusplus grid-resample grid.vti fine.vti --factor 2
meshioplusplus grid-scatter  fine.vti case.vtu predicted.vtu
meshioplusplus grid-spectrum grid.vti --field T --json
```

Write grids as **`.vti`**: it stores the lattice as origin/spacing/extent and regenerates the points arithmetically, so the grid is recovered exactly on the way back in. A grid written to `.vtu` is a lossy round trip for this purpose and reading one back as a grid fails by name.

| flag | verb | meaning |
|---|---|---|
| `--resolution nx,ny,nz` / `--cell-size S` | sample | the lattice; exactly one, the same six-field vocabulary as `voxelize` and `sdf` |
| `--bounds=xlo,...,zhi` | sample | explicit bounds. For a coarse/fine pair take the box from the **fine** mesh, so every fine node is inside the coarse grid |
| `--padding` / `--padding-relative` | sample | grow the box on every side |
| `--fields a,b` | all | which point data to carry; every array by default |
| `--extrapolate` | sample | give a grid point outside the mesh its nearest value instead of the fill |
| `--fill-value=V` | sample | what an outside point gets (`nan` makes the fill visible downstream) |
| `--factor N` / `--resolution nx,ny,nz` | resample | integer upscale, or explicit counts, over the same box; exactly one |
| `--on-conflict error\|overwrite\|suffix` | scatter | when the target already has an array of that name |
| `--field NAME` | spectrum | required; a multi-component array's components are summed |
| `--max-bins N` / `--json` | spectrum | truncate the printed table / emit JSON |

`grid-sample` reports the **coverage**, the fraction of grid points inside the mesh — a grid around a concave domain is mostly fill, and a model trained on it learns the fill. `cell_data` is refused in both directions: convert it with `data to-point` first.

## `subsample`

Reduce a mesh to a point cloud of exactly `--count` points, so a large surface fits a transformer's token budget.

```bash
meshioplusplus subsample wing.stl wing_4096.vtp --count 4096                 # farthest-point sampling
meshioplusplus subsample wing.stl wing_4096.vtp --count 4096 --method grid   # O(N + count^2), for big clouds
meshioplusplus subsample wing.stl tip.vtp --count 512 --bounds=0.8,-1,-1,1,1,1 --record-ids
```

| flag | meaning |
|---|---|
| `--count N` | required; how many points to keep |
| `--method farthest\|grid\|random` | exact farthest-point sampling (`O(N*count)`, the most uniform coverage), lattice representatives then farthest-point sampling (`O(N + count^2)`, the scalable choice above a few hundred thousand points), or a uniform draw |
| `--start N` / `--start random` | the first selected point for `farthest`/`grid`; `random` draws it from `--seed` |
| `--seed S` | drives `random`, and `--start random` |
| `--bounds=xlo,...,zhi` | only points inside this box are candidates |
| `--record-ids` | attach `budget:original_point_id`, each point's index in the source |

The output keeps the selected points, their `point_data` and Point regions, and gets a `vertex` block so every format can hold it; cells, `cell_data` and Cell/Side regions are dropped with a warning naming what went. See [point-cloud budgets](point_budgets.md).

See [`doc/grids.md`](grids.md).

## `proximity-graph`

Build a graph from geometry rather than connectivity: the neighbourhoods a particle method needs, where the interaction radius and not a shared element is what links two points.

```bash
meshioplusplus proximity-graph particles.vtu graph.vtu --radius 0.015
meshioplusplus proximity-graph particles.vtu graph.vtu --knn 16
meshioplusplus proximity-graph particles.vtu graph.vtu --radius 0.015 --box 2.0   # periodic
```

| flag | meaning |
|---|---|
| `--radius R` | link every pair closer than `R`; one of `--radius`/`--knn` is required |
| `--knn K` | link each point to its `K` nearest, then symmetrize (near-constant degree) |
| `--box L` / `--box Lx,Ly,Lz` | a periodic box: pairs are linked by their minimum image, and a radius past half the smallest side is refused |
| `--kind node\|cell` | vertices are mesh points (default) or block-major cell centroids |

The graph is written as `line` cells over those positions with a `degree` point array, and the summary prints the vertex and edge counts, the degree spread and how many vertices came out isolated — the numbers that say whether the radius was well chosen. See [proximity graphs](proximity_graphs.md).

## `sdf`

```bash
meshioplusplus sdf bunny.stl field.vti --resolution 128,128,128
meshioplusplus sdf bunny.stl tree.vtu  --structure octree --max-depth 5
```

| flag | meaning |
|---|---|
| `--structure voxel\|octree` | a dense lattice, or one refined near the surface |
| `--resolution nx,ny,nz` / `--cell-size S` | size a **voxel** grid; exactly one, and an **error** with `--structure octree` |
| `--root-resolution N` / `--max-depth N` / `--band-cells R` | octree: the root lattice, how many halving passes, and the band width in cell diagonals |
| `--bounds=xlo,...,zhi` | explicit bounds; the surface's own by default |
| `--padding` / `--padding-relative` | grow the box on every side (relative defaults to 0.1) |
| `--sign` / `--location` / `--band` / `--watertight-check` | as `distance_to_surface` |
| `--max-cells N` | refuse above this many cells, re-checked after every octree pass |

Write the result as `.vti` to keep the grid header — no other format carries it. The octree's output is 1-irregular (it has hanging nodes). See [`doc/sdf.md`](sdf.md).
