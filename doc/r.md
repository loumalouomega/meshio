# R

meshio++ ships an R package, `meshioplusplus`, layered on the [C API](/c_api) — the same flat C library the [Fortran](/fortran) and [Julia](/julia) bindings sit on:

```r
library(meshioplusplus)

m <- mio_read("bracket.msh")
m
#> <mio_mesh: 9231 points, 42145 cells in 3 block(s)>

surf <- mio_extract_surface(m)
mio_write(surf, "bracket_surface.vtu")
mio_release(m)
```

Every exported function is prefixed `mio_`, which keeps the package clear of base R names such as `points()`, `stats()`, `split()` and `merge()`.

The package is MIT-licensed like the rest of meshio++. (The sibling [Julia](/julia) binding is deliberately not — see its page.)

## Building and installing

The package binds the **installed** C library; it compiles only its own thin C shim.

```sh
./build/configure.sh --c-api --build
cmake --install build/cpp-release --prefix /opt/meshioplusplus
```

Then make it discoverable, either through pkg-config (the relocatable `.pc` the C API already installs):

```sh
export PKG_CONFIG_PATH=/opt/meshioplusplus/lib/pkgconfig
```

or by naming the prefix directly:

```sh
export MESHIOPLUSPLUS_HOME=/opt/meshioplusplus
```

and install:

```sh
R CMD INSTALL bindings/r/meshioplusplus
```

At run time the shared library must also be on the loader path (`LD_LIBRARY_PATH` on Linux, `DYLD_LIBRARY_PATH` on macOS). On Windows `configure` is not run: set `MESHIOPLUSPLUS_HOME` and put `libmeshioplusplus.dll` on `PATH`.

The package's `configure` script probes those two routes and generates `src/Makevars` from `src/Makevars.in`. That indirection is deliberate: a committed `Makevars` containing a `$(shell ...)` earns a *non-portable flags* NOTE from `R CMD check --as-cran`, while generating it from a `.in` is the pattern CRAN packages such as **sf** and **xml2** use.

## Array layout — column-major, no transpose

R matrices are column-major and the C core row-major, so

```r
mio_points(m)                     # dim x num_points
mio_connectivity(m, i)            # nodes_per_cell x num_cells
mio_point_data(m, "displacement") # components x num_points
```

are the **same layout** as the C API's row-major `(num_points, dim)`, `(num_cells, nodes_per_cell)` and `(num_points, components)`. One `memcpy` and the shape is already right — nothing is transposed on either side, exactly as in the Fortran and Julia bindings.

The one deliberate exception is `mio_transform()`, whose 4×4 affine matrix *is* transposed on the way out: that is a mathematical object, not a mesh array.

## R is copy-only

::: warning No zero-copy borrow in R
R vectors are R-managed, so the C API's zero-copy borrow **does not survive into R** without ALTREP machinery that is out of scope here. **Every accessor in this package copies.** That is a real difference from the [Julia](/julia) and [Fortran](/fortran) bindings, which do hand out live views into the C++ core's buffers.
:::

Because of that there is **no `_ptr` accessor at all**. The 0-based reader is named `mio_connectivity_raw()`, deliberately *not* `mio_connectivity_ptr()`, so nobody reads it as a borrow:

| accessor | copies? | node indices |
|---|---|---|
| `mio_points(m)` | yes | n/a |
| `mio_connectivity(m, i)` | yes | **1-based** |
| `mio_connectivity_raw(m, i)` | yes | **0-based** — the ABI's own |

The ±1 shift happens in that copy, which is where the other two bindings put it too.

## 64-bit integers

R has no native 64-bit integer type. The `int64` arrays the C API returns — connectivity, region entries, index maps, permutations — therefore arrive as `double`. That is exact to 2^53, far beyond any mesh this library will meet, and it avoids a hard dependency on **bit64** for a theoretical case. It is a limitation of the R side, not of the ABI.

The dtype a data array was actually stored as is reported separately rather than silently lost:

```r
attr(mio_point_data(m, "temperature"), "dtype")
#> [1] "float64"
```

## Indexing

Cell-block indices, data-name indices, connectivity and region entries are all **1-based**. Index maps and permutations are 1-based too, with the C API's `-1` "pruned / absent" sentinel becoming **`0`** — never a valid 1-based index, so it stays unambiguous. That is verbatim the Fortran and Julia rule.

`mio_partition_labels()` is the exception: those are part **ids**, not indices, so they stay in `0:(nparts - 1)`.

Region entries are 1-based with one further exception (see [`doc/regions.md`](/regions)): for a `"side"` region the second row is a **facet ordinal within the cell type**, not a mesh index, so it is passed through unshifted.

## Memory management

A mesh handle is an **external pointer** (`R_MakeExternalPtr`) with a registered finalizer calling `mio_mesh_free`, so it is released when garbage-collected. `mio_release()` frees it immediately and is idempotent; using a released handle raises a clean R error rather than crashing, and so does passing something that is not a mesh handle at all.

`mio_refine()` takes an optional cell selection: at most one of `cells` (global block-major, **1-based** here), `region` (a cell region selects its cells, a point region every cell with any node in it; a side region is an error) and `where_array` + `where_op` + `where_value`, plus `closure` (`"redgreen"`, local, or `"propagate"`, which reaches the whole edge-connected component) and `record_levels`. With no selector every cell is refined. `record_hierarchy` attaches `refine:cell_id`/`refine:parent_id` — the persistent parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps — and forces `refine:entity` (the multigrid prolongation stencil) to be attached even when the closure leaves no hanging node. See [refine](/refine#refinecell_id-and-refineparent_id). Note that R's data setters always write `Float64`, so a predicate over an array built fresh in R still works — the comparison is numeric — but `mio_split(by = "region")`'s integer-tag restriction does not apply here.

Operations producing an opaque C result (`mio_split()`, `mio_partition()`, `mio_reorder()`, `mio_refine()`, `mio_decimate()`, `mio_convert_cells()`, `mio_subdivide()`, `mio_agglomerate()`) always **transfer ownership** of the mesh out of that result rather than returning a borrow into it, so a piece stays valid after the result is gone.

## Error handling

Every failure becomes an R condition carrying the C API's own thread-local message via `Rf_error()`; a `mio_status` code never reaches R.

```r
mio_read("/nope.vtu")
#> Error: meshio++: cannot open file '/nope.vtu'
```

## Named regions

```r
mio_add_region(m, "inlet", "point", c(1, 3, 5))
mio_add_region(m, "solid", "cell", c(1, 2), dim = 3L, tag = 17)
mio_add_region(m, "wall", "side", matrix(c(1, 0, 2, 2), nrow = 2))

for (r in mio_regions(m)) {
  cat(r$name, r$kind, ncol(r$entries), "entries\n")
}
```

## Why plain `.Call`, not Rcpp

`.Call` with R's own C API keeps the dependency footprint at exactly zero — the package has no `Imports` and needs no C++ toolchain during `R CMD check` — and it matches the flat C ABI the rest of meshio++'s bindings sit on. The whole surface is scalars, vectors and matrices, where `Rf_allocVector` / `Rf_allocMatrix` plus a `memcpy` is all that is required; Rcpp would buy convenience this surface does not need.

## Selective reads and time steps

`mio_read()` narrows what it materializes, and since v8.6.0 also picks which time step of a multi-step file to decode:

```r
m <- mio_read("big.vtu", points_only = TRUE)

# 0 (the default) is the first step; negative counts from the end.
last <- mio_read("run.exo", time_step = -1)

meta <- mio_read_metadata("run.exo")
meta$time_values   # c(0, 0.5, 1) -- how many steps `time_step` may name
```

Out of range is an error naming the available count, never a silent clamp. `meta$time_values` has length 0 for a format with no time concept. Honoured by `exodus`; see [Selective reads](/selective_read).

## Transient (time-series) XDMF writing

`mio_xdmf_series()` is the write half of the above, and the one writer `mio_write()` cannot express: a series is a **stateful** multi-call object, so it gets its own handle rather than an argument. The grid goes out once and each solve appends a cheap step. See [XDMF time series](/xdmf_time_series).

```r
s <- mio_xdmf_series("simulation.xdmf")        # "HDF" by default
mio_xdmf_series_write_points_cells(s, mesh)    # the static grid, once
for (k in 0:9) {
  solve(mesh)
  mio_xdmf_series_write_data(s, k * dt, mesh)  # point_data/cell_data only
}
mio_xdmf_series_num_steps(s)
#> [1] 10
mio_xdmf_series_finalize(s)                    # release() would do this too
mio_xdmf_series_release(s)
```

`data_format` is `"HDF"` (the default; needs an HDF5-enabled library), `"XML"` (everything inline in the `.xdmf`) or `"Binary"`; `gzip_level` applies to `"HDF"` datasets only and is negative (no compression) by default. An unknown format, or `"HDF"` against a library built without HDF5, is an error carrying the C API's own message.

Two things worth knowing before reading the result back:

- the `.xdmf` light data is **buffered until the series is finalized**, so the file is only readable after `mio_xdmf_series_finalize()` (or `mio_xdmf_series_release()`, which finalizes first);
- a write failure during that implicit finalize cannot be reported from a GC finalizer at all, which is exactly why `mio_xdmf_series_finalize()` exists as a separate call. Prefer it, and release the handle before the directory it writes into goes away.

The handle is an external pointer with its **own** tag, so a `mio_mesh` and a `mio_xdmf_series` are never accepted for one another; `mio_xdmf_series_release()` is the idempotent deterministic free, and `mio_xdmf_series_is_open()` the predicate. Reading a finished series back is the ordinary `mio_read(path, time_step = k)`.

## Documented gaps

These are gaps in the **C ABI**, shared with the [Fortran](/fortran) and [Julia](/julia) bindings; the R package invents no workaround for any of them:

- **Four Python-only formats.** `pmsh`, `zarr`, `cae` and `usd` (v10.35.0, the physics-ML data path) are registered in the Python layer only, not in the shared C++ dispatch registry, so this surface cannot read or write them. They are export targets for a training pipeline rather than interchange formats; see [formats](/formats).

- **point / cell sets beyond regions** never reach the C++ core at all;
- the **`frozen` pin mask** of `mio_smooth()` and `mio_decimate()`;
- **per-cell-type counts** in `mio_stats()` — use `mio_cell_block_types()` with `mio_cell_block_info()`;
- ~~ragged block connectivity~~ — **closed in v9.15.0**: `mio_polygon_block()` / `mio_polyhedron_block()` read them as nested 1-based lists and `mio_add_polygon_block()` / `mio_add_polyhedron_block()` build them. `mio_connectivity()` still raises, since a ragged block has no matrix. See [Polyhedra and ragged cells](/polyhedra);
- the combined **`data_manage`** — `mio_data_drop()` / `mio_data_keep()` / `mio_data_rename()` compose to the same effect;
- **Exodus provenance strings** (`qa_records` / `info_records`): they ride the `ExodusInfo` side channel, which like `MedInfo` does not cross the flat ABI, so `mesh$info` has no counterpart here. Geometry, data, regions and time steps are unaffected.

As on the Julia page: **gmsh does not currently round-trip named regions** (the `$PhysicalNames` entry is written, but no physical tag is attached to an entity in `$Elements`, so a reader cannot rebuild the group). This is pre-existing meshio++ behaviour, reproducible from Python; `abaqus` round-trips regions correctly.

One further limitation is specific to this binding rather than the C ABI: **the data *setters* always write `Float64`.** `mio_add_point_data()`, `mio_append_cell_data()` and `mio_add_field_data()` copy through `REALSXP` regardless of the R vector's own storage mode, because R has no integer type reaching the C ABI here — the same reason [64-bit integers](#bit-integers) come back as `double` on the *reading* side. The practical consequence: `mio_split(by = "region")` needs a genuinely **integer** cell-data tag, so a tag array built fresh in R cannot drive it — only an integer tag already present in a *read* file (a gmsh physical group, an Exodus block id, an `mio_isosurface()`-produced `iso:index`, …) can. `example/r/03_mesh_operations.ipynb`'s split demo uses `by = "type"` for exactly this reason.

## Tests

```sh
R CMD build bindings/r/meshioplusplus
R CMD check --as-cran meshioplusplus_*.tar.gz
```

with `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH` pointed at the install prefix. The `testthat` suite mirrors the Julia one on the same deliberately non-square fixture, so a transposed mapping or a missed shift cannot cancel out.

## v10.9.0 additions

- `mio_hessian(mesh, array, method = "green-gauss", location = "cell", output = "", overwrite = FALSE)` — the Hessian (second derivative) of a **scalar point-data** field, `mio_gradient()`'s companion one order further. A composition of TWO `mio_gradient()` calls, not a new numerical kernel: the field is differentiated once (point location), then that `(n, 3)` gradient is differentiated again with the default `"gradient"` operator, producing `(n, 9)` — the flattened row-major 3x3 Hessian, `H[i,j]` at index `i*3+j`. `method` is forwarded to BOTH internal passes. Returns a list of `mesh`, `num_skipped` and `num_fallback`. See [`doc/hessian.md`](hessian.md).

A field that is at most LINEAR has an exactly zero Hessian everywhere — the one mesh-shape-independent guarantee. For a genuinely quadratic field the composition is exact on a structured/symmetric mesh away from its own boundary and a good, standard, but genuinely approximate curvature estimate on an irregular mesh. Input must have exactly one component — a vector field's Hessian is a separate quantity per component.

A curvature-driven refinement indicator needs no new function: `norm(...)` in `mio_data_calc()` on the 9-component output is exactly its Frobenius norm, ready for `mio_refine()`'s `where` selector.

## v10.8.0 additions

- `mio_data_integrate(mesh, names = NULL)` — cell-measure-weighted total/mean of one or more `cell_data` arrays, `mio_gradient()`'s integration counterpart (`mio_gradient` differentiates a field, this integrates one), returning a list of per-array summaries — each with `name`, `num_components`, `domain` (`num_cells`, `num_skipped`, and a `num_components x 4` `total`/`mean`/`domain_measure`/`num_nan` `components` matrix) and `regions` (a list of the same shape, one per named Cell region present). See [`doc/field_integration.md`](field_integration.md).

Every sum is weighted by `|measure(cell)|`; a cell whose measure is not computable, or a component whose value is non-finite, is excluded from that component's numerator **and** denominator, never given a fallback weight of 1. Regions are not a partition: a cell in two regions contributes fully to both. A `point_data`-only name fails, naming `mio_data_point_to_cell()` as the fix.

## v10.5.0 additions

- `mio_undo_green(coarse, fine)` — green-element undo: restores `fine`'s transitional (closure-only) cells back to their original parent, read verbatim from `coarse` — a **lookup, not a reconstruction**, since `mio_refine()` never renumbers or prunes points, so a green parent's exact connectivity and cell_data are already sitting, byte-for-byte, in `coarse` at the row `fine`'s `refine:parent_id` names. Returns a list of `mesh`, `num_groups_undone` and `num_cells_removed`. See [`doc/undo_green.md`](undo_green.md).

A **two-mesh** operation, like `mio_interpolate()`: `coarse` is the mesh a prior `mio_refine(coarse, ..., record_hierarchy = TRUE, record_levels = TRUE)` call was run on, `fine` is that call's output — both flags are required, `record_hierarchy` alone does not imply `record_levels`. The six reserved `refine:*` arrays are always dropped from the output; only a single-pass (`levels = 1`) hierarchy is supported, a deeper multi-level hierarchy being refused by name. Unlike `mio_subdivide()`/`mio_agglomerate()`, this operation has no winding repair or discrete sign branch anywhere in it — pure array bookkeeping, which on the Python side means a full numpy reference implementation rather than C++-core-only (this binding always calls the installed C library either way).

- `mio_conservative_interpolate(source, target, arrays = NULL, default_value = 0, on_conflict = "error")` — mass-preserving field transfer, `mio_interpolate()`'s sibling: conserves `sum(target value * target measure) == sum(source value * source measure)` over the region the two meshes share, a property `mio_interpolate()`'s pointwise sampling does not have. Both meshes are simplexified first, accepting ragged/polyhedron blocks for free. Unlike `mio_interpolate()`, a `NULL` `arrays` means every source point_data **and** cell_data array. Output arrays are always Float64. Like `mio_undo_green()`, a **two-mesh** operation, and like `mio_subdivide()`/`mio_agglomerate()` this is C++-core only — the 3D clip kernel's discrete branches could disagree with a second implementation near a degenerate overlap, so there is no pure-R fallback either; the installed C library is always called. See [`doc/conservative_interpolate.md`](conservative_interpolate.md).

## v10.4.0 additions

- `mio_agglomerate(mesh, target_group_size = 8)` — polyhedral coarsening, the many-to-one counterpart to `mio_subdivide()`: greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until each group reaches `target_group_size` members, then emitting one polyhedron per group whose faces are exactly its external boundary — conserving volume exactly, since internal faces are simply dropped rather than re-triangulated. Returns a list of `mesh` and `cell_map`. See [`doc/agglomerate.md`](agglomerate.md).

Unlike every other opaque-result operation here, `cell_map` is a **single flat**, not per-block, 1-based vector — an agglomerated cell's output index is a function of which group it joined, not which input block it came from — so there is no `subdivide_cell_maps()`-style per-block helper here, only the same single-array shift `mio_split()`'s node map already uses. Like `mio_subdivide()`, there is **no `point_map`**: points are never pruned or renumbered, so `mio_clean(mesh, remove_orphans = TRUE)` is the documented follow-up for a minimal point set. A non-manifold input (a face shared by three or more cells) raises an R error naming the face rather than guessing.

## v10.3.0 additions

- `mio_subdivide(mesh, record_parent_ids = FALSE)` — polyhedral refinement: one polyhedral child per face of every eligible 3D cell, connected to a new interior point, returning a list of `mesh` and `cell_maps`. Needs no per-type template table — tabulated types (reduced to corners for a quadratic variant) and existing polyhedron blocks are handled uniformly — and is automatically conforming, unlike `mio_refine()`. See [`doc/subdivide.md`](subdivide.md).

Unlike `mio_convert_cells()`, there is **no `point_map`**: subdivide never prunes or renumbers an original point. `cell_maps[[b]]` is 1-based input cell → the index of its **first** child (one per face) in the corresponding output block, the same shape `mio_convert_cells()` already uses for its own one-to-many splits.

## v10.2.0 additions

- `mio_estimate_error(mesh, array, method = "zz", marking = "none", marking_value = 0.0, output = "", marked = "", overwrite = FALSE)` — the Zienkiewicz-Zhu recovery-based error indicator of a **point-data** field, plus optional marking, returning a list of `mesh`, `global_error`, `num_skipped` and `num_marked`. A composition of `mio_gradient` with the point↔cell averaging round trip, not a new kernel. See [`doc/error.md`](error.md).

`error:zz` is always attached; `error:marked` too when `marking` is not `"none"`, so `mio_refine`'s own predicate needs no change to consume it. Cells that cannot be evaluated read `NaN` in `error:zz` and `0` (never `NaN`) in `error:marked`, counted in `num_skipped` and excluded from `global_error`/`num_marked`. The three counters come back as `double`, like every other 64-bit integer in this binding (see [64-bit integers](#bit-integers)).

## v9.11.0 additions

- The [settings pipeline](pipeline.md): `mio_pipeline_run_file(settings_path)` and `mio_pipeline_run_json(json_text)` run a whole `settings.json` (read → operation chain → write; PascalCase ops/keys) through the C++ engine, and `mio_pipeline_has_json()` reports whether the loaded library carries the JSON parser — a build without it signals an R error naming `-DMESHIOPLUSPLUS_WITH_JSON=ON` rather than missing a symbol. The flat ABI carries JSON text only; the structured run report is a recorded follow-up.

## v9.10.0 additions

- `mio_gradient(mesh, array, op = "gradient", method = "green-gauss", location = "cell", output = "", component = -1L, overwrite = FALSE)` — the gradient, divergence or curl of a **point-data** field, returning a list of `mesh`, `num_skipped` and `num_fallback`. See [`doc/gradient.md`](gradient.md).

The two counters come back as `double`, like every other 64-bit integer in this binding (see [64-bit integers](#bit-integers)); they are exact well past any plausible cell count. `component` is negative for **every** component here — deliberately the opposite of `mio_isosurface()`, where negative means the row magnitude.

## v9.1.0 additions

- `mio_read(..., lenient = TRUE)` — see [`doc/selective_read.md`](selective_read.md).
- XDMF series: `mio_xdmf_series_flush()`, `mio_xdmf_series_finalized()`, and `mio_xdmf_series(..., mode = "append", auto_flush = FALSE)`.

As elsewhere in this binding, remember to release a series *before* its tempdir is removed: a write failure during the implicit finalize in a GC finalizer cannot be reported. `MdpaInfo` is not exposed (as for every flat binding).

## Sequences (transient / multi-file datasets)

`mio_sequence()` wraps the C API's ordered plan over a set of files (or the steps inside one multi-step file). Ordering is natural-numeric, so `out_9.vtu` precedes `out_10.vtu`; nothing is read until `mio_sequence_read()`, which hands back a mesh independent of the sequence — the sequence caches nothing, so a 500-step dataset is traversable without materialising it.

```r
seq <- mio_sequence("out_*.vtu")            # or mio_sequence_list(c("a.vtu", ...))
for (i in seq_len(mio_sequence_count(seq))) {
  mesh <- mio_sequence_read(seq, i)         # one mesh alive at a time
  cat(mio_sequence_time(seq, i), mio_sequence_time_source(seq, i), "\n")
}
mio_sequence_to_timeseries(seq, "series.xdmf")            # fan-in
mio_sequence_free(seq)

# ascii=TRUE selects XDMF's "XML" data format (no HDF5 needed) -- the option
# this package's own HDF5-off notebook environment needs; see below.

mio_timeseries_to_sequence("series.xdmf", "step_{step}.vtu")   # fan-out
mio_sequence_pipeline_run_file("transient.json")               # per-step chain
```

Indices are 1-based, like every other R accessor. The external pointer has its own tag, so a `mio_mesh`, a `mio_xdmf_series` and a `mio_sequence` can never be passed for one another; a released handle is an R error, never a dereference.

See [sequences](sequences.md) for the ordering rule, the time-value precedence and the streaming guarantee.

- `mio_grid()`, `mio_voxelize()`, `mio_sample_distance()`, `mio_distance_to_surface()` and `mio_surface_watertight_check()` (v9.24.0) — regular grids and signed distance. The `mio_` prefix keeps `mio_grid()` clear of base R's own `grid` package. Counters come back as `double`, as everywhere else here. See [`doc/voxelize.md`](voxelize.md) and [`doc/sdf.md`](sdf.md).
- `mio_compute_sdf()` (v9.25.0) — the grid and the field in one call, returning a named list `(mesh, dims, origin, spacing, max_depth, num_banded, quality)`. `structure = "octree"` refines only near the surface and sizes itself from `root_resolution`/`max_depth`, so passing `resolution` or `cell_size` with it is an error; its output is 1-irregular.
- `mio_crop_predicate(mesh, array, compare, value)` (v9.25.0) — keep the cells whose scalar `cell_data` value satisfies a comparison. There is deliberately no `mode`: a per-cell value has nothing for an all/any rule to reduce. See [`doc/crop.md`](crop.md).
