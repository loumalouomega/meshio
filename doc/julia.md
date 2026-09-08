# Julia

meshio++ ships a Julia package, `MeshioPlusPlus`, layered on the [C API](/c_api) via `ccall` — the same way the [Fortran](/fortran) module is, and aimed at the same HPC audience:

```julia
import MeshioPlusPlus as mio
using MeshioPlusPlus

m = mio.read("bracket.msh")
println(m)                     # Mesh(9231 points, 42145 cells in 3 blocks)
surf = extract_surface(m)
mio.write(surf, "bracket_surface.vtu")
```

::: warning This binding is not MIT
`bindings/julia/` is released under the **GNU General Public License, version 3 (GPL-3.0)**, not the MIT licence covering the rest of meshio++. GPL-3.0 is a **copyleft** license, not a permission-required one: anyone — including a company — may use, modify or sell it commercially with **no permission needed**, but *distributing* it or a modified version of it must be under GPL-3.0 too, with source available. Purely private/internal use that is never distributed carries no obligation.

The C API and C++ core this binding calls are unaffected and stay MIT. Calling that stable, non-GPL C ABI via `ccall`/`dlopen` at runtime is the standard "linking exception" case — it does not require the C library to also be GPL. See [`bindings/julia/LICENSE`](https://github.com/loumalouomega/meshioplusplus/blob/master/bindings/julia/LICENSE).
:::

## Building and installing

The package binds the **installed** C library; no C++ is compiled by it.

```sh
./build/configure.sh --c-api --build
cmake --install build/cpp-release --prefix /opt/meshioplusplus
```

Then either put `/opt/meshioplusplus/lib` on the loader path, or point the package straight at the library:

```julia
ENV["MESHIOPLUSPLUS_LIB"] = "/opt/meshioplusplus/lib/libmeshioplusplus.so"

using Pkg
Pkg.develop(path="bindings/julia/MeshioPlusPlus")
```

`MESHIOPLUSPLUS_LIB` is checked first, then the standard loader path (`Libdl.find_library`). Failing both raises an error naming these two build commands.

::: warning A build with HDF5 can fail to load from Julia on Debian/Ubuntu
If the C API was built with `-DMESHIOPLUSPLUS_WITH_HDF5=ON`, loading it from a Julia process can fail with `libcurl.so.4: version 'CURL_OPENSSL_4' not found`. This is a real Debian/Ubuntu + Julia interaction, not a meshio++ bug: `libhdf5-dev` there pulls in `libhdf5_openmpi`, which links the *system* `libcurl`, while Julia bundles its own `libcurl` — and depending on the process's library-resolution order (an IJulia kernel subprocess is more likely to hit it than an interactive `julia` invocation), the two can conflict. Build with `-DMESHIOPLUSPLUS_WITH_HDF5=OFF -DMESHIOPLUSPLUS_WITH_NETCDF=OFF` if this binding is all you need from the library, as the `julia` CI job does.
:::

::: tip No registration yet, and no JLL
GPL-3.0 **is** OSI-approved, so the package is eligible for Julia's General registry — but registering it is a separate follow-up step, not done yet, so `Pkg.add("MeshioPlusPlus")` will not work until then; install by path or by URL in the meantime.

There is also deliberately no [JLL](https://docs.binarybuilder.org/stable/jll/): shipping a binary artifact through BinaryBuilder/Yggdrasil is a real distribution step and belongs in a follow-up rather than being faked here.
:::

## Array layout and 1-based indexing

The C core stores points row-major as `(num_points, dim)` and connectivity as `(num_cells, nodes_per_cell)`. Because Julia is column-major, the **same memory** is naturally

```julia
points(m)                      # (dim, num_points)
connectivity(m, 1)             # (nodes_per_cell, num_cells)
point_data(m, "displacement")  # (components, num_points)
```

so nothing is ever transposed — exactly the reasoning the Fortran module documents. Node indices are **1-based** here, and the ±1 shift happens inside the **copying** accessors, where a copy is made anyway:

| accessor | copies? | node indices |
|---|---|---|
| `points(m)` | yes | n/a — coordinates carry no indices |
| `points_ptr(m)` | **no** (zero-copy borrow) | n/a |
| `connectivity(m, i)` | yes | **1-based** |
| `connectivity_ptr(m, i)` | **no** | **0-based** — the ABI's own |

A borrow cannot be shifted without copying it, which is the whole point of a borrow — hence the two names. `point_data_ptr`, `cell_data_ptr` and `field_data_ptr` follow the same pattern.

Index maps and permutations (`refine`, `convert_cells`, `subdivide`, `agglomerate`, `decimate`, `partition`, `reorder`) are copies, so they are 1-based too, and the C API's `-1` "pruned / absent" sentinel becomes **`0`** — never a valid 1-based index. That is verbatim the Fortran rule; the bindings agree deliberately.

`partition_labels` is the exception: those are part **ids**, not indices, so they stay in `0:nparts-1`.

## The borrow window

`points_ptr` and friends return a `MeshBorrow`, a real zero-copy view of the buffer inside the C++ core (`unsafe_wrap`). The C API's rule is that such a pointer stays valid until the next **mutating** `mio_mesh_*` call on that mesh, or until it is freed; read-only accessors never invalidate it.

`MeshBorrow` enforces exactly that rather than trusting the caller:

```julia
b = points_ptr(m)
b[1, 2]                                # fine
add_point_data!(m, "T", temperatures)  # a MUTATING call ends the window
b[1, 2]                                # BorrowError -- not a stale read
```

It also holds a reference to the owning `Mesh`, so the mesh cannot be garbage-collected while a view of it is alive. `parent(b)` returns the raw `Array` for hot loops; that escape hatch is **unchecked** and valid only inside the window.

## Memory management

A `Mesh` releases its handle through a **finalizer** — the one real difference from the Fortran module, where meshes are freed explicitly with `call m%free()`. `close(m)` releases one deterministically and is idempotent.

`refine` takes an optional cell selection: at most one of `cells` (global block-major, **1-based** here), `region` (a cell region selects its cells, a point region every cell with any node in it; a side region is an error) and `where_array` + `where_op` + `where_value`, plus `closure` (`"redgreen"`, local, or `"propagate"`, which reaches the whole edge-connected component) and `record_levels`. With no selector every cell is refined. `record_hierarchy` attaches `refine:cell_id`/`refine:parent_id` — the persistent parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps, riding the raw C-side numbering unshifted like `partition_labels`' part ids — and forces `refine:entity` (the multigrid prolongation stencil) to be attached even when the closure leaves no hanging node. See [refine](/refine#refinecell_id-and-refineparent_id).

Operations producing an opaque C result (`split`, `partition`, `reorder`, `refine`, `decimate`, `convert_cells`, `subdivide`, `agglomerate`) always **transfer ownership** of the mesh out of that result rather than handing back a borrow into it, so a piece stays valid after the result is gone:

```julia
for (key, piece) in mio.split(m; by="type")
    mio.write(piece, "part_$key.vtu")   # still valid; the result is long gone
end
```

## Names that would shadow `Base`

`read`, `write`, `convert`, `merge`, `split` and `diff` are **not exported**, because they collide with `Base`. Call them qualified:

```julia
import MeshioPlusPlus as mio
m = mio.read("bracket.msh")
```

Everything else — the accessors, the setters, `regions`, and the remaining operations — is exported. `close` and `isopen` are proper `Base` method extensions.

## Error handling

Every failure raises a `MeshioError` carrying the C API's own thread-local message; a status code never reaches the caller.

```julia
julia> mio.read("/nope.vtu")
ERROR: MeshioError (read error): meshio++: cannot open file '/nope.vtu'
```

Using a borrow outside its window raises `BorrowError` instead.

## Named regions

```julia
add_region!(m, "inlet", :point, [1, 3, 5])
add_region!(m, "solid", :cell, [1, 2]; dim=3, tag=17)
add_region!(m, "wall", :side, Int64[1 2; 0 2])   # (cell, facet) pairs

for r in regions(m)
    println(r.name, " ", r.kind, " ", size(r.entries, 2), " entries")
end
```

Entries are 1-based, with one exception documented in [`doc/regions.md`](/regions): for a `:side` region the second row is a **facet ordinal within the cell type**, not a mesh index, so it is passed through unshifted — the same rule Fortran's `partition_labels` follows.

## Selective reads and time steps

`ReadOptions` narrows what a read materializes, and since v8.6.0 also picks which time step of a multi-step file to decode:

```julia
using MeshioPlusPlus

m = MeshioPlusPlus.read("big.vtu"; options = ReadOptions(points_only = true))

# 0 (the default) is the first step; negative counts from the end.
last = MeshioPlusPlus.read("run.exo"; options = ReadOptions(time_step = -1))

meta = read_metadata("run.exo")
meta.time_values          # [0.0, 0.5, 1.0] -- how many steps `time_step` may name
```

Out of range is an error naming the available count, never a silent clamp. `meta.time_values` is empty for a format with no time concept, so `length(...)` is always safe. Honoured by `exodus`; see [Selective reads](/selective_read).

## Transient (time-series) XDMF writing

`XdmfSeries` is the write half of the above, and the one writer `write` cannot express: a series is a **stateful** multi-call object, so it gets its own handle rather than a keyword. The grid goes out once and each solve appends a cheap step. See [XDMF time series](/xdmf_time_series).

```julia
using MeshioPlusPlus

s = XdmfSeries("simulation.xdmf")          # "HDF" by default
write_points_cells!(s, mesh)               # the static grid, once
for k in 0:nsteps-1
    solve!(mesh)
    write_data!(s, k * dt, mesh)           # point_data/cell_data only
end
num_steps(s)                               # 10
finalize!(s)                               # close(s) would do this too
close(s)
```

The function form closes the series afterwards even if the body throws:

```julia
XdmfSeries("simulation.xdmf"; data_format = "XML") do s
    write_points_cells!(s, mesh)
    write_data!(s, 0.0, mesh)
end
```

`data_format` is `"HDF"` (the default; needs an HDF5-enabled library), `"XML"` (everything inline in the `.xdmf`) or `"Binary"`; `gzip_level` applies to `"HDF"` datasets only and is negative (no compression) by default. An unknown format, or `"HDF"` against a library built without HDF5, throws a `MeshioError` from the constructor.

Two things worth knowing before reading the result back:

- the `.xdmf` light data is **buffered until the series is finalized**, so the file is only readable after `finalize!(s)` (or `close(s)`, which finalizes first);
- `close` swallows a write failure during that implicit finalize — call `finalize!` explicitly to see one.

Like `Mesh`, the handle is released by a GC finalizer and `close` is the deterministic, idempotent form. The name is `finalize!` rather than `finalize` because `Base.finalize` runs an object's GC finalizer and means something quite different. Reading a finished series back is the ordinary `MeshioPlusPlus.read(path; options = ReadOptions(time_step = k))`.

## Documented gaps

These are gaps in the **C ABI**, shared with the [Fortran](/fortran) and [R](/r) bindings; the Julia package invents no workaround for any of them:

- **Four Python-only formats.** `pmsh`, `zarr`, `cae` and `usd` (v10.35.0, the physics-ML data path) are registered in the Python layer only, not in the shared C++ dispatch registry, so this surface cannot read or write them. They are export targets for a training pipeline rather than interchange formats; see [formats](/formats).

- **point / cell sets beyond regions** never reach the C++ core at all;
- the **`frozen` pin mask** of `smooth` and `decimate`;
- **per-cell-type counts** in `stats` — use `cell_block_types` with `cell_block_info`;
- ~~ragged block connectivity~~ — **closed in v9.15.0**: `polygon_block` / `polyhedron_block` read them as nested 1-based vectors and `add_polygon_block!` / `add_polyhedron_block!` build them. `connectivity` still throws, since a ragged block has no matrix. See [Polyhedra and ragged cells](/polyhedra);
- the combined **`data_manage`** — `data_drop` / `data_keep` / `data_rename` compose to the same effect;
- **Exodus provenance strings** (`qa_records` / `info_records`): they ride the `ExodusInfo` side channel, which like `MedInfo` does not cross the flat ABI, so `mesh.info` has no counterpart here. Geometry, data, regions and time steps are unaffected.

One further limitation is a *format* one rather than an ABI one, and easy to trip over: **gmsh does not currently round-trip named regions.** The writer emits the `$PhysicalNames` entry but does not attach the physical tag to an entity in `$Elements`, so a reader finds nothing to rebuild the group from. This is pre-existing meshio++ behaviour, reproducible from Python; `abaqus` round-trips regions correctly.

## Tests

```sh
MESHIOPLUSPLUS_LIB=/opt/meshioplusplus/lib/libmeshioplusplus.so \
  julia --project=bindings/julia/MeshioPlusPlus -e 'using Pkg; Pkg.test()'
```

The suite uses the same deliberately non-square fixture as [`tests/fortran/test_fortran_api.f90`](https://github.com/loumalouomega/meshioplusplus/blob/master/tests/fortran/test_fortran_api.f90) — 5 points × 3 dims, 2 tetrahedra × 4 nodes, 3-component vector data — so a transposed mapping or a missed shift cannot cancel out and pass anyway. It pins the column-major identity, the 1-based/0-based accessor pair, the borrow window, regions, and every operation.

## v10.9.0 additions

- `hessian(mesh, array; method=:green_gauss, location=:cell, output="", overwrite=false) -> (; mesh, num_skipped, num_fallback)` — the Hessian (second derivative) of a **scalar point-data** field, [`gradient`](@ref)'s companion one order further. A composition of TWO `gradient` calls, not a new numerical kernel: the field is differentiated once (point location), then that `(n, 3)` gradient is differentiated again with the default `:gradient` operator, producing `(n, 9)` — the flattened row-major 3x3 Hessian, `H[i,j]` at index `i*3+j`. `method` is forwarded to BOTH internal passes. See [`doc/hessian.md`](hessian.md).

A field that is at most LINEAR has an exactly zero Hessian everywhere — the one mesh-shape-independent guarantee. For a genuinely quadratic field the composition is exact on a structured/symmetric mesh away from its own boundary and a good, standard, but genuinely approximate curvature estimate on an irregular mesh. Input must have exactly one component — a vector field's Hessian is a separate quantity per component.

A curvature-driven refinement indicator needs no new function: `norm(...)` in [`data_calc`](@ref) on the 9-component output is exactly its Frobenius norm, ready for [`refine`](@ref)'s `where` selector.

`hessian` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v10.8.0 additions

- `data_integrate(mesh, names=String[])` — cell-measure-weighted total/mean of one or more `cell_data` arrays, [`gradient`](@ref)'s integration counterpart (`gradient` differentiates a field, this integrates one), returning a `Vector{NamedTuple}` with `name`, `num_components`, `domain` (`num_cells`, `num_skipped`, `components` — a vector of `(total, mean, domain_measure, num_nan)` NamedTuples) and `regions` (the same shape, one per named Cell region present). See [`doc/field_integration.md`](field_integration.md).

Every sum is weighted by `|measure(cell)|`; a cell whose measure is not computable, or a component whose value is non-finite, is excluded from that component's numerator **and** denominator, never given a fallback weight of 1. Regions are not a partition: a cell in two regions contributes fully to both. A `point_data`-only name throws a `MeshioError` naming `data_point_to_cell` as the fix.

`data_integrate` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v10.5.0 additions

- `undo_green(coarse, fine)` — green-element undo: restores `fine`'s transitional (closure-only) cells back to their original parent, read verbatim from `coarse` — a **lookup, not a reconstruction**, since [`refine`](@ref) never renumbers or prunes points, so a green parent's exact connectivity and cell_data are already sitting, byte-for-byte, in `coarse` at the row `fine`'s `refine:parent_id` names. Returns `(; mesh, num_groups_undone, num_cells_removed)`. See [`doc/undo_green.md`](undo_green.md).

A **two-mesh** operation, like [`interpolate`](@ref): `coarse` is the mesh a prior `refine(coarse, ...; record_hierarchy=true, record_levels=true)` call was run on, `fine` is that call's output — both flags are required, `record_hierarchy` alone does not imply `record_levels`. The six reserved `refine:*` arrays are always dropped from the output; only a single-pass (`levels=1`) hierarchy is supported, a deeper multi-level hierarchy being refused by name. Unlike `subdivide`/`agglomerate`, this operation has **no winding repair or discrete sign branch anywhere in it** — it is pure array bookkeeping, which on the Python side means it has a full numpy reference implementation rather than being C++-core-only (this binding always calls the installed C library either way).

`undo_green` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

- `conservative_interpolate(source, target; arrays=String[], default_value=0.0, on_conflict="error")` — mass-preserving field transfer, [`interpolate`](@ref)'s sibling: conserves `sum(target value * target measure) == sum(source value * source measure)` over the region the two meshes share, a property `interpolate`'s pointwise sampling does not have. Both meshes are simplexified first, accepting ragged/polyhedron blocks for free. Unlike `interpolate`, an empty `arrays` means every source point_data **and** cell_data array. Output arrays are always Float64. Like `undo_green`, a **two-mesh** operation, and like `subdivide`/`agglomerate` this is C++-core only — the 3D clip kernel's discrete branches could disagree with a second implementation near a degenerate overlap, so there is no pure-Julia fallback either; the installed C library is always called. See [`doc/conservative_interpolate.md`](conservative_interpolate.md).

## v10.4.0 additions

- `agglomerate(mesh; target_group_size=8)` — polyhedral coarsening, the many-to-one counterpart to [`subdivide`](@ref): greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until each group reaches `target_group_size` members, then emitting one polyhedron per group whose faces are exactly its external boundary — conserving volume exactly, since internal faces are simply dropped rather than re-triangulated. Returns `(; mesh, cell_map)`. See [`doc/agglomerate.md`](agglomerate.md).

Unlike every other opaque-result operation here, `cell_map` is a **single flat**, not block-indexed, 1-based array: an agglomerated cell's output index is a function of which group it joined, not which input block it came from, so `_result_map` (the same single-array reader `split`'s node map uses) reads it rather than `_result_cell_maps`. Like `subdivide`, there is no `point_map` — points are never pruned or renumbered, so `clean(mesh; remove_orphans=true)` is the documented follow-up for a minimal point set. A non-manifold input (a face shared by three or more cells) raises a `MeshioError` naming the face rather than guessing.

`agglomerate` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v10.3.0 additions

- `subdivide(mesh; record_parent_ids=false)` — polyhedral refinement: one polyhedral child per face of every eligible 3D cell, connected to a new interior point, returning `(; mesh, cell_maps)`. Needs no per-type template table — tabulated types (reduced to corners for a quadratic variant) and existing polyhedron blocks are handled uniformly through the same `detail::cell_rings`/`orient_rings` machinery `gradient` uses — and is automatically conforming, unlike [`refine`](@ref). See [`doc/subdivide.md`](subdivide.md).

Unlike [`convert_cells`](@ref), there is no `point_map`: `subdivide` never prunes or renumbers an original point. `cell_maps[b]` is 1-based input cell → the index of its **first** child (one per face) in the corresponding output block, the same `FirstChild` shape `convert_cells` already uses for its own one-to-many splits.

`subdivide` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v10.2.0 additions

- `estimate_error(mesh, array; method=:zz, marking=:none, marking_value=0.0, output="", marked="", overwrite=false)` — the Zienkiewicz-Zhu recovery-based error indicator of a **point-data** field, plus optional marking, returning `(; mesh, global_error, num_skipped, num_marked)`. A composition of `gradient` with the point↔cell averaging round trip, not a new kernel. See [`doc/error.md`](error.md).

`error:zz` is always attached; `error:marked` too when `marking` is not `:none`, so `refine`'s own `where` selector needs no change to consume it — `refine(mesh, where="error:marked > 0.5")`. Cells that cannot be evaluated read `NaN` in `error:zz` and `0` (never `NaN`) in `error:marked`, counted in `num_skipped` and excluded from `global_error`/`num_marked`.

`estimate_error` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v9.11.0 additions

- The [settings pipeline](pipeline.md): `run_pipeline_file(settings_path)` and `run_pipeline_json(json_text)` run a whole `settings.json` (read → operation chain → write; PascalCase ops/keys) through the C++ engine, and `pipeline_has_json()` reports whether the loaded library carries the JSON parser — a build without it raises a `MeshioError` naming `-DMESHIOPLUSPLUS_WITH_JSON=ON` rather than missing a symbol. The flat ABI carries JSON text only; the structured run report is a recorded follow-up. All three are exported (none shadows `Base`).

## v9.10.0 additions

- `gradient(mesh, array; operator=:gradient, method=:green_gauss, location=:cell, output="", component=-1, overwrite=false)` — the gradient, divergence or curl of a **point-data** field, returning `(; mesh, num_skipped, num_fallback)`. See [`doc/gradient.md`](gradient.md).

Two spellings to note. `method` is given as an underscored symbol (`:green_gauss`, `:least_squares`) and translated to the C API's hyphenated name by the binding, because a Julia symbol cannot carry a hyphen. And `component` is negative for **every** component here — deliberately the opposite of `isosurface`, where negative means the row magnitude.

`gradient` shadows nothing in `Base`, so unlike `read`/`write`/`split` it is exported.

## v9.1.0 additions

- `ReadOptions(; lenient=true)` — see [`doc/selective_read.md`](selective_read.md).
- XDMF series: `flush!(s)`, `finalized(s)`, `XdmfSeries(path; mode=:append, auto_flush=false)`, and `write_data!(s, t, Dict("u" => values))` for writing a step from raw arrays with no `Mesh` in between. An `n x k` matrix is transposed on the way out, since Julia is column-major and the C ABI expects `k` components per entity row-major.

`flush!` is named with a bang for the same reason `finalize!` is: `Base.flush` means "flush this IO stream". `MdpaInfo` is not exposed (as for every flat binding).

## Sequences (transient / multi-file datasets)

`Sequence` wraps the C API's ordered plan over a set of files (or the steps inside one multi-step file). Ordering is natural-numeric, so `out_9.vtu` precedes `out_10.vtu`; nothing is read until `read_step`, and the sequence caches nothing, so a 500-step dataset is traversable without materialising it.

```julia
seq = Sequence("out_*.vtu")               # or Sequence(["a.vtu", "b.vtu"])
for i in 1:length(seq)
    mesh = read_step(seq, i)              # OWNED; one mesh alive at a time
    @show MeshioPlusPlus.time(seq, i), MeshioPlusPlus.time_source(seq, i)
end
to_timeseries(seq, "series.xdmf")         # fan-in
to_timeseries(seq, "series2.xdmf"; ascii=true)  # "XML" data format, no HDF5 needed
close(seq)

timeseries_to_sequence("series.xdmf", "step_{step}.vtu")   # fan-out
run_sequence_file("transient.json")                        # per-step chain
```

`Sequence` is also iterable (`for mesh in seq`), and has a `do`-block form that closes the handle even if the body throws.

**Not exported**, because they shadow `Base`: `path`, `step`, `time`, `time_source`. Reach them as `MeshioPlusPlus.time(seq, i)` — the same rule the binding already applies to `read`/`write`/`split`.

See [sequences](sequences.md) for the ordering rule, the time-value precedence and the streaming guarantee.

- `grid(dims; origin, spacing)`, `voxelize(m; resolution, fill, ...)`, `sample_distance(surface, points)`, `distance_to_surface(query, surface)` and `surface_watertight_check(m)` (v9.24.0) — regular grids and signed distance. `grid` shadows nothing in `Base`, so it is exported plainly. See [`doc/voxelize.md`](voxelize.md) and [`doc/sdf.md`](sdf.md).
- `compute_sdf(surface; structure, resolution, root_resolution, max_depth, ...)` (v9.25.0) — the grid and the field in one call, returning `(; mesh, dims, origin, spacing, max_depth, num_banded, quality)`. `structure=:octree` refines only near the surface and sizes itself from `root_resolution`/`max_depth`, so passing `resolution` or `cell_size` with it is an error; its output is 1-irregular.
- `crop_predicate(m, array; compare, value, record_ids)` (v9.25.0) — keep the cells whose scalar `cell_data` value satisfies a comparison, the same vocabulary `refine`'s `where_op` uses. There is deliberately no `mode`: a per-cell value has nothing for an all/any rule to reduce. See [`doc/crop.md`](crop.md).
