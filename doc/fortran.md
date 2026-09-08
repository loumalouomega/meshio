# Fortran

meshio++ ships a modern object-oriented Fortran 2008 module, `meshioplusplus`, layered on the [C API](/c_api) via `ISO_C_BINDING` — in the HDF5/PETSc style, aimed at Fortran HPC codes:

```fortran
use meshioplusplus
type(mio_mesh) :: m

call m%read("bracket.msh")
print *, m%num_points(), "points,", m%num_cell_blocks(), "cell blocks"
call m%write("bracket.vtu")
call m%free()
```

## Building

```sh
build/configure.sh --fortran --build     # implies --c-api
cmake --install build/cpp-release --prefix /opt/meshioplusplus
```

installs `libmeshioplusplus_fortran.so` next to `libmeshioplusplus.so`, plus `include/meshioplusplus/fortran/meshioplusplus.mod` **and the module source** `meshioplusplus.f90`. Compile and link:

```sh
gfortran my_solver.f90 -I /opt/meshioplusplus/include/meshioplusplus/fortran \
  -L /opt/meshioplusplus/lib -lmeshioplusplus_fortran -lmeshioplusplus -o my_solver
```

::: warning .mod files are compiler-specific
A `.mod` compiled by gfortran N is unreadable by gfortran N±2, ifort, or flang. If your compiler rejects the installed `.mod`, recompile the module from the installed `meshioplusplus.f90` with your own compiler and link the same libraries — that is exactly why the source is installed (the HDF5 approach). CMake consumers can instead `find_package(meshioplusplus)` and link `meshioplusplus::meshioplusplus_fortran` from the same build.
:::

## Array layout and 1-based indexing

The C core stores points as row-major `(num_points, dim)` and connectivity as `(num_cells, nodes_per_cell)`. Because Fortran is column-major, the **same memory** is naturally the Fortran arrays

```fortran
real(real64)   :: points(dim, num_points)     ! points(:, i) = coordinates of point i
integer(int64) :: conn(nodes_per_cell, num_cells)
```

so nothing is ever transposed. Node indices are **1-based** in this module; the ±1 shift happens inside the copying setters/getters (where a copy is made anyway):

```fortran
call m%set_points(points)                     ! copies
call m%add_cell_block("tetra", conn)          ! copies, shifts to the core's 0-based
call m%get_cell_block(1, rconn)               ! allocates, copies, shifts back to 1-based
```

Vector data uses the same rule — a per-point field of `c` components is `data(c, num_points)` in Fortran and round-trips as the C API's `(num_points, c)`.

Zero-copy borrows exist where no index shift is needed:

```fortran
real(real64), pointer :: p(:, :)
call m%points_ptr(p)     ! p(dim, num_points) aliases mesh memory --
                         ! valid until the next mutating call or m%free()
```

## Error handling

Every fallible procedure takes optional `stat` and `errmsg` arguments:

```fortran
integer :: ierr
character(:), allocatable :: msg

call m%read("missing.vtu", stat=ierr, errmsg=msg)
if (ierr /= 0) print *, "read failed: ", msg
```

If `stat` is **absent** and the call fails, the message is printed and the program `error stop`s — convenient for straight-line tools, pass `stat` when you need to recover. `mio_error_message()` returns the most recent failure message on the calling thread.

## API summary

| | |
| --- | --- |
| Lifecycle | `m%create()` (implicit in `read`/setters), `m%read(path [, format])`, `m%write(path [, format])`, `m%free()`, `m%is_valid()` |
| Building | `m%set_points(points)`, `m%add_cell_block(type, conn)` (int32 or int64), `m%add_point_data(name, data)` (rank-1 or rank-2), `m%add_cell_data(name, data)` (once per block, in order), `m%add_field_data(name, data)` |
| Counts | `m%num_points()`, `m%point_dim()`, `m%num_cell_blocks()`, `m%num_point_data()`, `m%num_cell_data()`, `m%num_field_data()`, `m%cell_data_num_blocks(name)` |
| Cell blocks (1-based) | `m%cell_block_type(i)`, `m%cell_block_num_cells(i)`, `m%cell_block_nodes_per_cell(i)`, `m%cell_block_is_ragged(i)`, `m%cell_block_is_polyhedron(i)`, `m%get_cell_block(i, conn)` |
| Ragged cell blocks | Cells of varying size, carried as a **1-based** flat CSR pair or triple (see [Polyhedra and ragged cells](/polyhedra)). Building: `m%add_polygon_block(type, row_offsets, nodes)`, `m%add_polyhedron_block(type, cell_offsets, face_offsets, nodes)`. Reading: `m%get_polygon_block(i, row_offsets, nodes)`, `m%get_polyhedron_block(i, cell_offsets, face_offsets, nodes)`. The offsets are shifted to 1-based along with the node ids, because they index a Fortran array — so cell `c` is `nodes(row_offsets(c) : row_offsets(c + 1) - 1)`, and face `f` of cell `c` is `nodes(face_offsets(j) : face_offsets(j + 1) - 1)` with `j = cell_offsets(c) + f - 1` |
| Data (copies, real64) | `m%get_points(points)`, `m%get_point_data(name, data)` (rank-1 or rank-2), `m%get_cell_data(name, block, data)`, `m%get_field_data(name, data)`; names via `m%point_data_name(i)` etc. (sorted order) |
| Zero-copy | `m%points_ptr(p)` |
| Operations | `m%extract_surface([record_parent_ids])`, `m%extract_skin([linearize])`, `m%attach_quality()`, `m%quality_counts(...)`, `m%reorder(method [, node_perm])` (method `"rcm"`/`"morton"`/`"hilbert"`; `node_perm` is an optional allocatable 1-based old→new node permutation), `m%compute_bandwidth()`, and comparison: `m%equals(other [, atol, rtol, unordered])` (logical) / `m%diff(other [, atol, rtol, unordered])` (verdict `0`=identical / `1`=equal within tolerance / `2`=different), and merging: the module-level `mio_merge(meshes [, weld, atol, source_tag, data_policy, drop_duplicate_cells])` (an array of `type(mio_mesh)` → a new `type(mio_mesh)`; `data_policy` is `"intersection"` or `"fill"`), and the editing/stats bundle: `m%transform(matrix(16) [, rotate_vector_data])`, `m%clean([weld, atol, remove_orphans, drop_degenerate, drop_duplicate_cells, points_welded, ...])`, `m%crop_bbox(lo, hi [, mode, record_ids])` / `m%crop_plane(point, normal [, mode, record_ids])` (`mode` `"all"`/`"any"`) / `m%crop_predicate(array [, compare, value, record_ids])` (v9.25.0; keeps the cells whose scalar `cell_data` value satisfies a comparison — `"<"`, `"<="`, `">"`, `">="`, `"=="`, `"!="`, the same vocabulary `m%refine`'s `where_op` uses. A **non-finite** cell value never matches. There is deliberately no `mode`: a per-cell value has nothing for an all/any rule to reduce), `m%split(by [, tag_name, keys])` (→ an allocatable array of `type(mio_mesh)`), `m%stats()` (→ a `type(mio_stats_report)` of scalar geometric measures), and `m%convert_cells(mode [, record_parent_ids, point_map])` (mode `"linearize"`/`"simplexify"`/`"elevate"`; `point_map` is an optional allocatable 1-based input→output point map, `0` where the point was pruned), `m%subdivide([record_parent_ids])` (polyhedral refinement: one polyhedral child per face of every eligible 3D cell, connected to a new interior point — no per-type template table, tabulated types and existing polyhedron blocks handled uniformly, and automatically conforming, unlike `m%refine`; → a new `type(mio_mesh)` with no point map, unlike `m%convert_cells`, since subdivide never prunes or renumbers a point; fails through `stat`/`errmsg` naming the cell when its faces are not a closed orientable surface), `m%agglomerate([target_group_size, stat, errmsg])` (polyhedral coarsening, the many-to-one counterpart to `m%subdivide`: greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until each group reaches `target_group_size` (default 8) members, then emitting one polyhedron per group whose faces are exactly its external boundary — conserving volume exactly, since internal faces are dropped rather than re-triangulated; → a new `type(mio_mesh)` with, like `m%subdivide`, no point map at all — the flat input→output cell map the underlying `mio_agglomerate_result` carries is a documented flat-binding gap here, not exposed on this wrapper, since an agglomerated cell's output index is a function of which group it joined rather than a per-block quantity Fortran's existing `point_map`/allocatable-array idioms fit; fails through `stat`/`errmsg` naming the face when the input is non-manifold), and `m%refine(levels [, record_parent_ids, point_map, cells, region, where_array, where_op, where_value, closure, record_levels, record_hierarchy])` (subdivision into same-type children; `levels` defaults to 1. At most one of `cells` (global block-major, **1-based** here), `region` and `where_array` + `where_op` + `where_value` selects a **subset** to refine, and the hanging nodes that leaves are resolved by `closure` — `"redgreen"` keeps the extra refinement local, `"propagate"` works for every cell type but reaches the whole edge-connected component. `record_levels` attaches the `refine:level` array; `record_hierarchy` attaches `refine:cell_id`/`refine:parent_id` — the persistent parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps, and it also forces `refine:entity` to be attached even when the closure leaves no hanging node), and partitioning: `m%partition(nparts [, method, imbalance, mode, seed, record_ids, ghost_layers, weights_key])` (→ an allocatable array of exactly `nparts` `type(mio_mesh)` pieces, part id = index − 1; `method` `"sfc"`/`"kahip"`/`"auto"`) and `m%partition_labels(nparts [, method, imbalance, mode, seed, weights_key])` (→ a flat block-major `integer(int64)` array of part ids in `[0, nparts)` — the values are ids, not indices, so no 1-based shift applies), and smoothing: `m%smooth(method, iterations [, lambda, mu, fix_boundary, preserve_features, feature_angle, guard_inversion, nodes_moved, max_displacement, skipped_inversion])` (method `"taubin"`/`"laplacian"`; a negative `lambda` means the method's own default; → a new `type(mio_mesh)` with the same cells and data and only its point coordinates moved, plus optional run-summary out-arguments), and green-element undo: the module-level `mio_undo_green(coarse, fine [, num_groups_undone, num_cells_removed, stat, errmsg])` (a TWO-mesh function like `mio_interpolate`/`mio_merge`, not type-bound; `fine` must carry `refine:cell_id`/`refine:parent_id`/`refine:level`, i.e. must come from `m%refine(..., record_hierarchy=.true., record_levels=.true.)`, and `coarse` must be the mesh that call was run on) → a new `type(mio_mesh)` restoring `fine`'s transitional (green) sibling groups to the row `coarse` already has for them verbatim — a lookup, not a reconstruction, since `refine` never renumbers or prunes points — with the six reserved `refine:*` arrays always dropped and only a single-pass (`levels=1`) hierarchy supported; the per-block cell maps are a documented flat-ABI gap here, like `m%agglomerate`'s, and mass-preserving field transfer: the module-level `mio_conservative_interpolate(source, target [, arrays, default_value, on_conflict, stat, errmsg])` (a TWO-mesh function like `mio_interpolate`/`mio_merge`, not type-bound; unlike `mio_interpolate`, an omitted/zero-sized `arrays` means every source point_data AND cell_data array — one algorithm regardless of location) → a new `type(mio_mesh)` conserving `sum(value * measure)` over the region the two meshes share (both meshes simplexified first, accepting ragged/polyhedron blocks for free), unlike `mio_interpolate`'s pointwise sampling; deliberately no `extrapolate` argument, since a silent fallback would break that guarantee, and the two cutters: `m%slice(origin, normal [, record_parent_ids])` (the planar cross-section, one topological dimension below the cut cells) and its data-driven sibling `m%isosurface(array, isovalues(:) [, component, record_parent_ids])` (the level set of a scalar `point_data` array; `component` defaults to −1 meaning the row magnitude, several isovalues land in one mesh tagged per cell with `iso:value`/`iso:index`, and a `cell_data` name fails through `stat` — convert it with `m%data_cell_to_point` first), and field differentiation: `m%gradient(array [, op, method, location, output, component, overwrite, num_skipped, num_fallback])` (the gradient, divergence or curl of a `point_data` array; `op` `"gradient"`/`"divergence"`/`"curl"`, `method` `"green-gauss"`/`"least-squares"`, `location` `"cell"`/`"point"`. `component` defaults to −1 meaning **every** component — the opposite of `m%isosurface`'s sentinel. Cells that cannot be differentiated yield NaN and are reported in `num_skipped`; least-squares cells with a degenerate neighbourhood fall back to Green-Gauss and are reported in `num_fallback`), and second derivatives: `m%hessian(array [, method, location, output, overwrite, num_skipped, num_fallback])` (the Hessian of a **scalar** `point_data` field -- `m%gradient`'s companion one order further, a composition of TWO `m%gradient` calls, not a new kernel: the field is differentiated once at `location="point"` regardless of the caller's own `location`, then that `(n,3)` gradient is differentiated again with the default operator, producing `(n,9)` -- the flattened row-major 3x3 Hessian. `method` is forwarded to BOTH internal passes. A field that is at most linear has an exactly zero Hessian everywhere -- the one mesh-shape-independent guarantee; a genuinely quadratic field's composition is exact on a structured/symmetric mesh away from its own boundary and approximate on an irregular mesh (see `doc/hessian.md`). Input must have exactly one component -- a vector field's Hessian is a separate quantity per component, rejected naming the per-component workaround. A curvature-driven refinement indicator needs no new procedure: `norm(...)` in `m%data_calc` on the 9-component output is exactly its Frobenius norm, ready for `m%refine`'s `where_array`/`where_op`/`where_value` selector), and error estimation: `m%estimate_error(array [, method, marking, marking_value, output, marked, overwrite, global_error, num_skipped, num_marked])` (the Zienkiewicz-Zhu recovery-based error indicator of a `point_data` field — a composition of `m%gradient` with the point↔cell averaging round trip, not a new kernel — plus optional marking; `method` `"zz"`, `marking` `"none"`/`"absolute"`/`"fraction"`/`"dorfler"`. `error:zz` is always attached; `error:marked` too when `marking` is not `"none"`, so `m%refine`'s own `where_array`/`where_op`/`where_value` needs no change to consume it. Cells that cannot be evaluated read NaN in `error:zz` and `0` in `error:marked`, counted in `num_skipped` and excluded from `global_error`/`num_marked`), and field integration: `m%data_integrate([arrays, keys, totals, means, domain_measures, num_nans])` (→ an allocatable array of `type(mio_field_integral_info){num_components, num_cells, num_skipped}`, one per requested `cell_data` array (omitted `arrays` = every array); the four optional out-arguments are flat, array-major, per-component buffers, `mesh_regions`'s own flat-buffer idiom — `gradient`'s integration counterpart, a cell-measure-weighted total/mean, never a fallback weight of 1 for an unmeasurable cell or non-finite value) plus `m%data_integrate_region(array_name [, keys, totals, means, domain_measures, num_nans])` for one named array's **per-region** breakdown — split into a second call rather than nesting array × region × component into one, mirroring `m%data_integrate`'s own shape but one array at a time; every named **Cell** region present gets its own independent entry (not a partition: a cell in two regions contributes fully to both), and a `point_data`-only name fails through `stat`/`errmsg` naming `m%data_point_to_cell` as the fix | `mio_grid(...)`, `m%voxelize(...)`, `m%watertight_check()`, `m%sample_distance(points)`, `m%distance_to_surface(surface)` (v9.24.0), and `m%compute_sdf([structure, resolution, cell_size, bounds, padding, padding_relative, root_resolution, max_depth, band_cells, record_levels, max_cells, sign, weight, location, band, watertight_check, dims, origin, spacing, depth, num_banded, quality])` (v9.25.0 — the grid and the field in one call; `structure` `"voxel"`/`"octree"`, the latter sizing itself from `root_resolution`/`max_depth`, so `resolution`/`cell_size` with it is an error. `spacing` reports the **finest** cell, `dims` the **root** counts).
| Data operations | Act on the mesh's data arrays; the geometry is never modified (see [data operations](/data_operations)). `m%data_drop(location, names [, ignore_missing])`, `m%data_keep(...)`, `m%data_rename(location, from, to)`, `m%data_point_to_cell([names, suffix])`, `m%data_cell_to_point([names, weight, suffix])`, `m%data_calc(expression, output_name [, location, overwrite])`, `m%data_condition(location [, names, mode, lo, hi, scope, nan_policy, nan_replacement, suffix])` — each returns a new `type(mio_mesh)` — and `m%data_info([keys])` → an allocatable array of `type(mio_data_array_info)`. `location` is `MIO_DATA_POINT` / `_CELL` / `_FIELD`; the other enumerations are `MIO_WEIGHT_*`, `MIO_COND_*`, `MIO_SCOPE_*` and `MIO_NAN_*`. The module also exports `STRBUF_LEN`, the fixed width of the `keys` out-argument arrays used by `m%split`, `m%data_info` and `m%regions` |
| Named regions | Named groups of points, cells or cell facets (see [Named regions](/regions)). `m%regions([keys, entries])` → an allocatable array of `type(mio_region_info)`, with the group names in `keys` and the flat `integer(int64)` entries in `entries`, laid out back to back in report order; `m%add_region(name, kind, entries [, dim, tag])`. `kind` is `MIO_REGION_POINT` / `_CELL` / `_SIDE`. Point and cell indices are shifted to Fortran's 1-based convention by the copying getter and setter; the **facet** column of a side region is not, matching the `partition_labels` rule that a value which is not an index stays as it is |
| Module-level | `mio_convert(in, out [, in_format, out_format])`, `mio_version()`, `mio_mesh_backend()`, `mio_format_readable(f)`, `mio_format_writable(f)`, `mio_error_message()`, and the [settings pipeline](/pipeline): `mio_pipeline_run_file(settings_path [, stat, errmsg])` / `mio_pipeline_run_json(json_text [, stat, errmsg])` run a whole settings.json (read → operation chain → write), and `mio_pipeline_has_json()` reports whether the linked library carries the JSON parser (a build without it fails through `stat` naming `-DMESHIOPLUSPLUS_WITH_JSON=ON`) |

The complete CI-tested example lives at [`doc/examples/fortran_example.f90`](https://github.com/loumalouomega/meshioplusplus/blob/main/doc/examples/fortran_example.f90); format support and the remaining limitations (side-channel metadata) are identical to the [C API](/c_api#format-support), which this module wraps; ragged blocks are no longer among them (see the table above). Copy-getters deliver `real(real64)` regardless of the stored dtype (float32/int32/int64 are converted); Fortran on Windows/MSVC is untested in v1.

## Selective reads and file summaries

`read` takes optional `points_only`, `arrays` and `time_step`, and the module-level `mio_read_metadata` returns a `type(mio_metadata)`:

```fortran
type(mio_mesh) :: m
type(mio_metadata) :: meta

call m%read('big.msh', points_only=.true.)
call m%read('big.msh', arrays=['u   ', 'p   '])   ! fixed-length, as Fortran requires
call m%read('run.exo', time_step=-1)              ! the last step; 0 (default) is the first

meta = mio_read_metadata('run.exo')
print *, meta%num_points, meta%num_cells
print *, meta%fell_back_to_full_read      ! .true. => the file was read in full
print *, size(meta%time_values)           ! how many steps `time_step` may name
```

`time_step` picks a step of a multi-step file (0 = the first, negative counts from the end). Out of range is an error naming the available count, never a silent clamp; `meta%time_values` is allocated to size 0 for a format with no time concept.

A zero-sized `arrays` means *no* arrays, which is distinct from omitting the argument (every array). Names are trimmed and NUL-terminated internally; the copies outlive the call.

`meta%cell_blocks` is an array of `type(mio_cell_block_info)`, and the name lists use the exported `STRBUF_LEN` convention, as `split` and `data_info` do.

## Transient (time-series) XDMF writing

`type(mio_xdmf_series)` is the write half of what `time_step` and `meta%time_values` expose above, and the one writer `m%write()` cannot express: a series is a **stateful** multi-call object, so it gets its own handle rather than a `write` argument. The grid goes out once and each solve appends a cheap step. See [XDMF time series](/xdmf_time_series).

```fortran
type(mio_mesh) :: m
type(mio_xdmf_series) :: series
integer :: k

call series%create('simulation.xdmf')          ! "HDF" by default
call series%write_points_cells(m)              ! the static grid, once
do k = 0, nsteps - 1
    call solve(m)
    call series%write_data(k*dt, m)            ! point_data/cell_data only
end do
print *, series%num_steps()
call series%finalize()                         ! free() would do this too
call series%free()
```

| | |
| --- | --- |
| Lifecycle | `series%create(path [, data_format, gzip_level])`, `series%free()`, `series%is_valid()` |
| Writing | `series%write_points_cells(mesh)`, `series%write_data(time, mesh)`, `series%finalize()`, `series%num_steps()` |

`data_format` is `'HDF'` (the default; needs an HDF5-enabled build), `'XML'` (everything inline in the `.xdmf`) or `'Binary'`; `gzip_level` applies to `'HDF'` datasets only and is negative (no compression) by default. An unknown format, or `'HDF'` against a library built without HDF5, fails through `stat`/`errmsg` like every other fallible procedure.

Two things worth knowing before reading the result back:

- the `.xdmf` light data is **buffered until the series is finalized**, so the file is only readable after `series%finalize()` (or `series%free()`, which finalizes first);
- `free()` swallows a write failure during that implicit finalize — call `finalize()` explicitly, with `stat`, to see one.

Handles are freed explicitly, exactly like `type(mio_mesh)`; there is no finalizer. Reading a finished series back is the ordinary `m%read(path, time_step=k)`, with `mio_read_metadata(path)%time_values` reporting the steps.

## v9.1.0 additions

- `m%read(..., lenient=.true.)` — see [`doc/selective_read.md`](selective_read.md).
- XDMF series: `s%flush()`, `s%finalized()`, and `s%create(..., mode='append', auto_flush=...)`.

**Gap, deliberate:** four Python-only formats — `pmsh`, `zarr`, `cae` and `usd` (v10.35.0, the physics-ML data path) — are registered in the Python layer only, not in the shared C++ dispatch registry, so this surface cannot read or write them; see [formats](/formats). And there is no Fortran counterpart to the solver-array `write_data` overload. An array of derived types holding interop pointers is a poor fit for Fortran, and a Fortran solver already holds an `mio_mesh` handle it can `add_point_data` into before `write_data`. `MdpaInfo` is likewise absent, as for every flat binding.

## Sequences (transient / multi-file datasets)

`type(mio_sequence)` wraps the C API's ordered plan over a set of files (or the steps inside one multi-step file). Ordering is natural-numeric, so `out_9.vtu` precedes `out_10.vtu`; nothing is read until `%read_step`, and the sequence caches nothing, so a 500-step dataset is traversable without materialising it.

```fortran
type(mio_sequence) :: seq
type(mio_mesh) :: step
integer :: i

call seq%open('out_*.vtu')
do i = 1, int(seq%count())
    step = seq%read_step(i)          ! OWNED; free it yourself
    print *, seq%time(i), seq%time_source(i)
    call step%free()
end do
call seq%to_timeseries('series.xdmf')            ! fan-in
call seq%free()

call mio_timeseries_to_sequence('series.xdmf', 'step_{step}.vtu')  ! fan-out
call mio_sequence_pipeline_run_file('transient.json')              ! per-step chain
```

Indices are 1-based, like every other Fortran accessor, and handles are freed explicitly (`%free()`) exactly as `mio_mesh` is — there is no finalizer. Every fallible procedure takes the usual optional `stat`/`errmsg`.

See [sequences](sequences.md) for the ordering rule, the time-value precedence and the streaming guarantee.
