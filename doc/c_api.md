# C API

The C++ core also ships as an installable shared library, `libmeshioplusplus`, with a stable pure-C99 header — the natural entry point for HPC codes written in C (and the foundation of the [Fortran interface](/fortran)). Like the [WebAssembly binding](/wasm), it is a flat, whole-mesh API over the same C++ core, built exclusively on the uniform mesh API, so it works identically under every [mesh backend](/cpp_backends).

## Building and installing

The C API is off by default (a plain `pip install` never builds it). From the repo:

```sh
build/configure.sh --c-api --build          # add --fortran for the Fortran module
cmake --install build/cpp-release --prefix /opt/meshioplusplus
```

(or pass `-DMESHIOPLUSPLUS_BUILD_C_API=ON` to a direct CMake configure). The install lays out:

```
include/meshioplusplus/meshioplusplus.h     # the only installed header
lib/libmeshioplusplus.so[.0.6.1.0]
lib/cmake/meshioplusplus/                   # find_package(meshioplusplus)
lib/pkgconfig/meshioplusplus.pc             # pkg-config
```

Consume it with pkg-config:

```sh
gcc my_solver.c $(pkg-config --cflags --libs meshioplusplus) -o my_solver
```

or CMake:

```cmake
find_package(meshioplusplus REQUIRED)  # pin a minimum version if you need one, e.g. `meshioplusplus 10.0`
                                       # (compatibility is SameMajorVersion: a 9.x request rejects a 10.x install)
target_link_libraries(my_solver PRIVATE meshioplusplus::meshioplusplus)
```

HDF5/netCDF/zlib are detected at configure time exactly as for the Python build; they are private dependencies of the shared library (consumers never link them directly).

## Package managers (Conan, vcpkg & Spack)

The C API also ships as a **Conan** recipe (root [`conanfile.py`](https://github.com/loumalouomega/meshioplusplus/blob/main/conanfile.py)) and a **vcpkg** overlay port ([`packages/vcpkg/meshioplusplus/`](https://github.com/loumalouomega/meshioplusplus/tree/main/packages/vcpkg/meshioplusplus)). Both are self-hosted in the repo and drive the same `-DMESHIOPLUSPLUS_BUILD_C_API=ON -DMESHIOPLUSPLUS_BUILD_PYTHON=OFF` install path, so consumers get the identical `meshioplusplus::meshioplusplus` target:

```sh
# Conan (options: with_hdf5 / with_netcdf / with_zlib / with_eigen / fortran)
conan create . -o meshioplusplus/*:with_hdf5=True -o meshioplusplus/*:with_netcdf=True

# vcpkg (features: hdf5 / netcdf / zlib -- all on by default)
vcpkg install meshioplusplus --overlay-ports=packages/vcpkg
```

`packages/vcpkg/` also carries a second overlay, `packages/vcpkg/libaec/`, alongside `packages/vcpkg/meshioplusplus/` — it replaces vcpkg registry's own `libaec` port (a transitive dependency of `hdf5`'s `szip` feature) with an otherwise-identical one that sources the same release from `libaec`'s official GitHub mirror instead of its `gitlab.dkrz.de` host, which had been intermittently rate-limiting this project's release CI. `--overlay-ports=packages/vcpkg` picks up both automatically.

Both are validated in CI on every PR and on `v*` release tags (`.github/workflows/packages.yml`). Two caveats: the shared library is **shared-only** (no static build yet), and the vendored **Eigen** submodule is off in both recipes (the MED transpose falls back to a hand-written loop), since it is absent from a source tarball. Neither is submitted to Conan Center / the upstream vcpkg registry today, so nothing resolves `meshioplusplus` as a plain requirement out of the box; as a stopgap, every `v*` release attaches ready-to-use Linux artifacts, for both x86_64 and arm64, to its [GitHub Release](https://github.com/loumalouomega/meshioplusplus/releases) that supply the missing recipe:

```sh
# Conan: restore the release archive (adds the meshioplusplus recipe + a matching
# prebuilt binary to your local cache), then install normally -- Conan Center
# still supplies the transitive hdf5/netcdf/zlib/openssl binaries over the
# network as usual; --build=missing rebuilds meshioplusplus itself from the
# archive's recipe if your profile doesn't match the prebuilt one.
conan cache restore meshioplusplus-conan-full-linux-x86_64.tgz   # or ...-linux-arm64.tgz
conan install --requires=meshioplusplus/<version> --build=missing \
  -o meshioplusplus/*:with_hdf5=True -o meshioplusplus/*:with_netcdf=True -o meshioplusplus/*:with_zlib=True

# vcpkg, from the release archive (unzip first):
vcpkg install meshioplusplus --overlay-ports=meshioplusplus-vcpkg-overlay-port-<version>
```

Unlike Conan/vcpkg, the **Spack** recipe is not self-hosted in this repo — it lives in the [`spack/spack-packages`](https://github.com/spack/spack-packages) upstream repo ([`meshioplusplus`](https://github.com/spack/spack-packages/blob/develop/repos/spack_repo/builtin/packages/meshioplusplus/package.py), [PR #5624](https://github.com/spack/spack-packages/pull/5624)), so it needs no local checkout and resolves `meshioplusplus` as a plain requirement:

```sh
spack install meshioplusplus +fortran +hdf5 +netcdf +zlib
```

Variants mirror the Conan options / vcpkg features above (`+hdf5`/`+netcdf`/`+zlib`/`+zstd`/`+lz4`/`+kahip`, all named the same way) plus `+fortran`, `+cxx_api` (the [installable C++ API](https://loumalouomega.github.io/meshioplusplus/cpp_api), with `cxx_api_backends=meshio,native,kratos` selecting which mesh-backend libraries to install), `parallel=auto|seq|stl|openmp|tbb|kokkos`, and `mesh_backend=meshio|native|kratos` for the standalone build. See [Installation → Spack](installation.md#spack) for the matching `py-meshioplusplus` Python package.

## Example

The complete, CI-tested example lives at [`doc/examples/c_api_example.c`](https://github.com/loumalouomega/meshioplusplus/blob/main/doc/examples/c_api_example.c):

```c
#include <meshioplusplus/meshioplusplus.h>

/* Build a mesh from raw arrays and write it (setters copy your buffers). */
mio_mesh* m = mio_mesh_create();
mio_mesh_set_points(m, MIO_FLOAT64, num_points, 3, xyz);            /* (n, 3) row-major */
mio_mesh_add_cell_block(m, "tetra", num_cells, 4, MIO_INT64, conn); /* 0-based indices */
mio_mesh_add_point_data(m, "temperature", MIO_FLOAT64, 1, (int64_t[]){num_points}, temp);
if (mio_write("out.vtu", m, NULL) != MIO_OK)          /* NULL: infer format from extension */
    fprintf(stderr, "%s\n", mio_last_error());
mio_mesh_free(m);

/* Read one back (getters are zero-copy borrows into mesh-owned memory). */
mio_mesh* r = mio_read("in.msh", NULL);               /* .msh defaults to gmsh */
const void* pts; mio_dtype dt;
mio_mesh_get_points(r, &pts, &dt);                    /* valid until mutation/free */
mio_mesh_free(r);

/* Or convert file-to-file without touching the data. */
mio_convert("in.msh", NULL, "out.vtk", NULL);
```

## The contract in five rules

1. **Errors**: every fallible function returns a `mio_status` (`MIO_OK == 0`) or `NULL`/`-1`; the message is retrievable via `mio_last_error()` (thread-local, valid until the next `mio_*` call on the same thread). No C++ exception ever crosses the ABI.
2. **Setters copy** — your buffers can be freed as soon as the call returns.
3. **Getters are zero-copy** — returned data pointers alias mesh-owned memory and stay valid until the next *mutating* call on that mesh or `mio_mesh_free()`. Read-only accessors never invalidate them.
4. **Arrays are row-major** (C order): points `(num_points, dim)`, connectivity `(num_cells, nodes_per_cell)` with **0-based** node indices.
5. **String getters** use the snprintf convention: copy at most `buflen - 1` bytes plus a NUL, return the full length (excluding the NUL), `-1` on error.

## API reference

| Group | Functions |
| --- | --- |
| Introspection | `mio_version`, `mio_mesh_backend`, `mio_format_readable`, `mio_format_writable`, `mio_last_error` |
| Cell-type metadata | `mio_cell_type_name`, `mio_cell_type_from_name`, `mio_cell_type_num_nodes`, `mio_cell_type_dimension` (the `mio_cell_type` enum mirrors the C++ table; strings like `"tetra10"` are the primary representation) |
| Lifecycle & I/O | `mio_mesh_create`, `mio_mesh_free`, `mio_read`, `mio_write`, `mio_convert` (format `NULL`/`""` = infer from extension; `.msh` → gmsh, `.inp` → abaqus, `.node`/`.ele` → tetgen — pass `"triangle"` for 2D Triangle pairs) |
| Building | `mio_mesh_set_points`, `mio_mesh_add_cell_block` (int32 connectivity is widened to the core's int64), `mio_mesh_add_point_data`, `mio_mesh_append_cell_data` (one call per cell block, in block order), `mio_mesh_add_field_data` |
| Points/cells | `mio_mesh_num_points`, `mio_mesh_point_dim`, `mio_mesh_get_points`, `mio_mesh_num_cell_blocks`, `mio_mesh_cell_block_info`, `mio_mesh_cell_block_type`, `mio_mesh_cell_block_conn` |
| Named data | `mio_mesh_num_{point,cell,field}_data`, `mio_mesh_{point,cell,field}_data_name` (names in sorted order — identical on every backend), `mio_mesh_get_{point,cell,field}_data`, `mio_mesh_cell_data_num_blocks` |
| Version | `MIO_VERSION_MAJOR` / `_MINOR` / `_PATCH` and the ordered integer `MIO_VERSION` are **compile-time** macros in the header, with `MIO_VERSION_AT_LEAST(major, minor, patch)` / `MIO_VERSION_BEFORE(...)` for feature detection (`#if MIO_VERSION_AT_LEAST(9, 5, 0)` guards a call to `mio_refine_ex`, new in 9.5.0). `mio_version()` returns the version of the **linked library** instead, which with a shared build can genuinely differ from the header's — compile against the macros, report the call. A `static_assert` in `c_api.cpp` pins these to the C++ `MESHIOPLUSPLUS_VERSION_*` macros and CMake hard-fails if either disagrees with the project version, so the copies cannot drift |
| Operations | `mio_extract_surface`, `mio_extract_skin`, `mio_attach_quality`, `mio_quality_counts`, `mio_sniff_format`, `mio_compute_bandwidth`, and reordering: `mio_reorder` (method `"rcm"`/`"morton"`/`"hilbert"`) → `mio_reorder_result` (`mio_reorder_result_mesh` borrow, `mio_reorder_result_node_perm` / `mio_reorder_result_num_cell_perms` / `mio_reorder_result_cell_perm` zero-copy borrows, `mio_reorder_result_take_mesh` ownership transfer, `mio_reorder_result_free`), and comparison: `mio_meshes_equal(a, b, atol, rtol, unordered, &equal)` plus `mio_diff(a, b, atol, rtol, unordered, &result)` → `mio_diff_result` (`mio_diff_result_verdict`, `mio_diff_result_point_summary`, `mio_diff_result_num_block_diffs` / `mio_diff_result_block`, `mio_diff_result_free`), and merging: `mio_merge(meshes, count, weld, atol, source_tag, data_policy, drop_duplicate_cells)` (array of mesh handles → a new `mio_mesh*`; `data_policy` `0`=intersection / `1`=fill), and the editing/stats bundle: `mio_transform(mesh, matrix[16], rotate_vector_data)` → `mio_mesh*`; `mio_clean(mesh, weld, atol, remove_orphans, drop_degenerate, drop_duplicate_cells, &n_welded, &n_orphan, &n_degen, &n_dup)` → `mio_mesh*`; `mio_crop_bbox(mesh, lo, hi, mode, record_ids)` / `mio_crop_plane(mesh, point, normal, mode, record_ids)` (`mode` `0`=all / `1`=any) / `mio_crop_predicate(mesh, array, compare, value, record_ids)` (v9.25.0; keeps the cells whose scalar `cell_data` value satisfies a `mio_refine_compare` — the same vocabulary `mio_refine_ex` uses, evaluated by the same C++ function, and a **non-finite** cell value never matches whatever the comparison. There is deliberately no `mode`: bbox and half-space test *points* and then need an all/any rule, whereas a `cell_data` predicate is already one value per cell. Inside/outside a surface composes as `mio_distance_to_surface` at `MIO_SDF_CENTER` then this on `"sdf:distance"` < 0) → `mio_mesh*`; `mio_split(mesh, by, tag_name)` (`by` `"type"`/`"component"`/`"region"`/`"tag"`/`"regions"` — the last, added v8.7.0, is one piece per named **Cell** region and, uniquely, not a partition: a cell in several regions lands in several pieces, `tag_name` unused) → `mio_split_result` (`mio_split_result_count` / `_key` / `_mesh` borrow / `_take_mesh` / `_free`); `mio_stats(mesh, &report)` → fills a `mio_stats_report` struct (scalar geometric measures; per-cell-type counts are not carried); `mio_convert_cells(mesh, mode, record_parent_ids)` (`mode` `"linearize"`/`"simplexify"`/`"elevate"`) → `mio_convert_cells_result` (`mio_convert_cells_result_mesh` borrow, `_take_mesh` ownership transfer, `_point_map` / `_num_cell_maps` / `_cell_map` zero-copy borrows, `_free`); `mio_subdivide(mesh, record_parent_ids)` (polyhedral refinement: one polyhedral child per face of every eligible 3D cell, connected to a new interior point -- no per-type template table, tabulated types and existing polyhedron blocks handled uniformly through the same `detail::cell_rings`/`orient_rings` machinery `mio_gradient` uses, and automatically conforming, unlike `mio_refine`) → `mio_subdivide_result` (`mio_subdivide_result_mesh` borrow, `_take_mesh` ownership transfer, `_num_cell_maps` / `_cell_map` zero-copy borrows, `_free` -- deliberately **no point map**, unlike `mio_convert_cells_result`, since subdivide never prunes or renumbers a point; fails with `mio_last_error()` naming the cell when its faces are not a closed orientable surface); `mio_agglomerate(mesh, target_group_size)` (polyhedral coarsening, the many-to-one counterpart to `mio_subdivide`: greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until each group reaches `target_group_size` members, then emitting one polyhedron per group whose faces are exactly its external boundary -- conserving volume exactly, since internal faces are simply dropped rather than re-triangulated) → `mio_agglomerate_result` (`mio_agglomerate_result_mesh` borrow, `_take_mesh` ownership transfer, `_cell_map` a **single flat** zero-copy borrow with **no `block` argument and no `_num_cell_maps` accessor** -- unlike every other opaque-result type here, an agglomerated cell's output index is a function of which group it joined, not which input block it came from -- `_free`; fails with `mio_last_error()` naming the face when the input is non-manifold; points are never pruned or renumbered, so there is no point map either, `clean(..., remove_orphans=1)` being the documented follow-up for a minimal point set); `mio_refine(mesh, levels, record_parent_ids)` (subdivision into same-type children) → `mio_refine_result` (the same accessor set: `mio_refine_result_mesh` borrow, `_take_mesh`, `_point_map` / `_num_cell_maps` / `_cell_map`, `_free`), plus `mio_refine_ex(mesh, &opts)` taking a `mio_refine_opts` (`mio_read_opts`' append-only reserved-tail discipline; zero-initialize through `mio_refine_opts_init` and it reproduces `mio_refine` exactly) with a **cell selection** — at most one of a global block-major index list, a `region` name, or a scalar `cell_data` `predicate_array` + `predicate_op` + `predicate_value` — a `closure` (`MIO_REFINE_CLOSURE_REDGREEN`; `_PROPAGATE` for the always-works baseline that reaches the whole edge-connected component; `_BALANCED` to keep the hanging nodes and only enforce 2:1 balance, which makes the output **non-conforming** and reports the constrained nodes in the `refine:hanging` `point_data` array) and `record_levels` for the `refine:level` array, plus `record_hierarchy` (v10.1.0; an `int64_t` taking one of the struct's former `reserved` slots, `sizeof(mio_refine_opts)` unchanged at 112 bytes) for the persistent `refine:cell_id`/`refine:parent_id` parent/child hierarchy a multigrid caller resolves across the sequence of meshes it keeps (a link between two meshes, not a tree inside one; an unsplit cell keeps its id and is its own parent, a split cell's children each get a fresh id and carry the parent's id) — it also forces `refine:entity` to be attached even when the closure leaves no hanging node, since that array already records the coarse corners each new fine node is the mean of; `mio_partition(mesh, nparts, method, imbalance, mode, seed, record_ids, ghost_layers, weights_key)` (`method` `"sfc"`/`"kahip"`/`"auto"`; NULL/`""` strings mean the defaults) → `mio_partition_result` (`mio_partition_result_num_pieces` / `_part_id` / `_mesh` borrow / `_take_mesh` / per-piece zero-copy `_point_map` / `_num_cell_maps` / `_cell_map` / `_free`), plus `mio_partition_labels(mesh, nparts, method, imbalance, mode, seed, weights_key, labels, labels_size)`, which fills a caller-allocated flat int64 buffer in block-major global cell order (`labels_size` must equal the total cell count; recover block alignment from the mesh's own per-block cell counts). `method="kahip"` in a build without `MESHIOPLUSPLUS_WITH_KAHIP` fails with `mio_last_error()` naming the option; and smoothing: `mio_smooth(mesh, method, iterations, lambda, mu, fix_boundary, preserve_features, feature_angle, guard_inversion, &nodes_moved, &max_displacement, &skipped_inversion)` (`method` `"taubin"`/`"laplacian"`; a negative `lambda` means the method's own default — `0.5` laplacian, `0.33` taubin) → a new `mio_mesh*` with the same cells and data and only its point coordinates moved, the three counter out-params being nullable exactly like `mio_clean`'s. The C++ `mFrozen` pin mask is **not** exposed on the flat ABI (a documented gap, like the `point_sets`/`cell_sets` gaps in diff/merge/split); and green-element undo: `mio_undo_green(coarse, fine, &num_groups_undone, &num_cells_removed)` (a TWO-mesh function, like `mio_interpolate` — `fine` must carry one Int64 scalar `cell_data` array per block for each of `refine:cell_id`/`refine:parent_id`/`refine:level`, i.e. must come from an `mio_refine_ex` call with `record_hierarchy` AND `record_levels` both set; `coarse` must be the mesh that call was run on) → a new `mio_mesh*` restoring `fine`'s transitional (green) sibling groups to the row `coarse` already has for them verbatim — a lookup, not a reconstruction, since `mio_refine`/`mio_refine_ex` never renumber or prune points — with the two counters nullable exactly like `mio_smooth`'s; the six reserved `refine:*` arrays are always dropped from the output, and only a single-pass (`levels=1`) hierarchy is supported. Its per-block cell maps are **not** exposed on the flat ABI, a documented gap like `mio_smooth`'s `mFrozen`; and mass-preserving field transfer: `mio_conservative_interpolate(source, target, arrays, arrays_count, default_value, on_conflict)` → `mio_mesh*` (a TWO-mesh function like `mio_interpolate`; unlike `mio_interpolate`, NULL/`<= 0` `arrays_count` means every source point_data AND cell_data array — one algorithm regardless of location. Both meshes are simplexified first, which is what lets ragged/polyhedron blocks through for free. Output is always Float64, conserving `sum(value * measure)` over the region the two meshes share, unlike `mio_interpolate`'s pointwise sampling; there is deliberately no `extrapolate` flag, since a silent fallback would break that guarantee); and the two cutters: `mio_slice(mesh, origin[3], normal[3], record_parent_ids)` → `mio_mesh*` (the planar cross-section, one dimension below the cut cells) and its data-driven sibling `mio_isosurface(mesh, array_name, isovalues, n_isovalues, component, record_parent_ids)` → `mio_mesh*` (the level set of a scalar `point_data` array; `component` negative = row magnitude, several isovalues land in one mesh tagged per cell with `iso:value`/`iso:index`, and a `cell_data` name fails with `mio_last_error()` naming `mio_data_cell_to_point` as the fix); and field differentiation: `mio_gradient(mesh, array_name, op, method, location, output_name, component, overwrite, &num_skipped, &num_fallback)` → `mio_mesh*` (the gradient, divergence or curl of a `point_data` array; `op` `"gradient"`/`"divergence"`/`"curl"`, `method` `"green-gauss"`/`"least-squares"`, `location` `"cell"`/`"point"`, NULL/`""` strings meaning the defaults, and the two counters nullable exactly like `mio_clean`'s. `component` is negative for **every** component — deliberately the opposite of `mio_isosurface`'s sentinel. An `nc`-component input yields `3*nc` gradient components laid out `[component][derivative]`; divergence gives 1 and curl 3, both needing 2 or 3 components. Cells that cannot be differentiated yield NaN and are counted, and a `cell_data` name fails naming `mio_data_cell_to_point`); and second derivatives: `mio_hessian(mesh, array_name, method, location, output_name, overwrite, &num_skipped, &num_fallback)` → `mio_mesh*` (the Hessian of a **scalar** `point_data` field — `mio_gradient`'s companion one order further, a composition of TWO `mio_gradient` calls, not a new kernel: the field is differentiated once at `location="point"` regardless of the caller's own `location`, then that `(n,3)` gradient is differentiated again with the default operator, producing `(n,9)` — the flattened row-major 3x3 Hessian. `method` is forwarded to BOTH internal passes; NULL/`""` strings mean the defaults, and the two counters are nullable exactly like `mio_clean`'s. A field that is at most linear has an exactly zero Hessian everywhere — the one mesh-shape-independent guarantee; a genuinely quadratic field's composition is exact on a structured/symmetric mesh away from its own boundary and approximate on an irregular mesh (see `doc/hessian.md`). Input must have exactly one component — a vector field's Hessian is a separate quantity per component, rejected naming the per-component workaround. A curvature-driven refinement indicator needs no new API: `` mio_data_calc(mesh, "curv = norm(`<array>:hessian`)", ...) `` on the 9-component output is exactly its Frobenius norm, ready for `mio_refine_ex`'s predicate selector); and error estimation: `mio_estimate_error(mesh, array_name, method, marking, marking_value, output_name, marked_name, overwrite, &global_error, &num_skipped, &num_marked)` → `mio_mesh*` (the Zienkiewicz-Zhu recovery-based error indicator of a `point_data` field — a composition of `mio_gradient` with the measure-weighted point↔cell averaging round trip, not a new kernel — plus optional marking; `method` `"zz"` (the only one today), `marking` `"none"`/`"absolute"`/`"fraction"`/`"dorfler"`, NULL/`""` strings meaning the defaults, and all three counters nullable exactly like `mio_clean`'s. `error:zz` (Float64) is always attached; `error:marked` (Int64 0/1) too when `marking` is not `"none"`, so `mio_refine_ex`'s own `predicate_array`/`predicate_op`/`predicate_value` needs no change at all to consume it. Cells that cannot be evaluated read NaN in `error:zz` and `0` (never NaN) in `error:marked`, counted in `num_skipped` and excluded from `global_error` and every marking policy's count); and field integration: `mio_data_integrate_create(mesh, names, count)` (NULL `names` = every `cell_data` array) → an opaque `mio_data_integrate*` — `gradient`'s integration counterpart, a cell-measure-weighted total/mean, read with `mio_data_integrate_count` / `_name` (caller-buffer protocol) / `_entry` (fills a `mio_field_integral_info{num_components, num_cells, num_skipped}`, the whole-mesh entry) / `_component(index, comp, &total, &mean, &domain_measure, &num_nan)`, plus the region axis `_region_count` / `_region_name` / `_region_entry` / `_region_component` (one independent entry per named **Cell** region present — not a partition), released with `mio_data_integrate_free`. A cell whose measure cannot be computed, or a component whose value is non-finite, is excluded from that component's numerator and denominator, never given a fallback weight of 1; a `point_data`-only name fails naming `mio_data_point_to_cell` as the fix. Read-only — no ABI bump, wholly new entry points | `mio_grid`, `mio_voxelize`, `mio_surface_watertight_check`, `mio_sample_distance`, `mio_distance_to_surface` (v9.24.0) and `mio_compute_sdf(surface, &opts, dims_out, origin_out, spacing_out, &max_depth, &num_banded, &quality)` (v9.25.0 — the umbrella that generates its own grid: `MIO_SDF_VOXEL` is a dense lattice sized by `resolution`/`cell_size`, `MIO_SDF_OCTREE` refines near the surface and sizes itself from `root_resolution`/`max_depth`, so passing either sizing with it is an error rather than a preference; the octree's output is 1-irregular). `mio_voxel_opts`/`mio_sdf_opts`/`mio_compute_sdf_opts` all follow `mio_refine_opts`' append-only `reserved` discipline — note `mio_compute_sdf_opts` **does** embed a `mio_sdf_opts` by value, matching the C++ struct rather than papering over the nesting — and `mio_sample_distance` writes into a caller-supplied buffer like `mio_partition_labels`.
| Data operations | Act on `point_data`/`cell_data`/`field_data`; the geometry is never modified (see [data operations](/data_operations)). Name lists cross as `const char* const* names` plus an explicit `int64_t count` (`0` = every array). Array management: `mio_data_drop` / `mio_data_keep` / `mio_data_rename` → a new `mio_mesh*`. Averaging: `mio_data_point_to_cell(mesh, names, count, suffix)` / `mio_data_cell_to_point(mesh, names, count, weight, suffix)` (`weight` `MIO_WEIGHT_UNIFORM`/`_MEASURE`). Expressions: `mio_data_calc(mesh, expression, location, output_name, overwrite)`. Conditioning: `mio_data_condition(mesh, location, names, count, mode, lo, hi, scope, nan_policy, nan_replacement, suffix)` with the `MIO_COND_*` / `MIO_SCOPE_*` / `MIO_NAN_*` enumerations. Summary: `mio_data_info_create(mesh)` → an opaque `mio_data_info*`, read with `mio_data_info_count` / `_name` (caller-buffer protocol) / `_entry` (fills a `mio_data_array_info`) / `_component`, released with `mio_data_info_free` |

| Settings pipeline | `mio_pipeline_run_file(settings_path)` / `mio_pipeline_run_json(json_text)` run a whole [settings pipeline](/pipeline) (read → operation chain → write; PascalCase ops/keys) and return `mio_status` — the flat ABI carries JSON text only, no step struct, and schema errors reach `mio_last_error()` naming the offending op/key. Needs a build with the JSON parser (`-DMESHIOPLUSPLUS_WITH_JSON=ON`, the default when the `src/cpp/third_party/json` submodule is checked out; **off** in the conan/vcpkg packages, whose source export has no submodule — the Eigen rule); otherwise the calls fail naming the flag, and `mio_pipeline_has_json()` reports which build this is. A structured report accessor is a recorded follow-up |
| Sequences (transient) | `mio_sequence_open(pattern)` / `_open_list` / `_count` / `_path` / `_step` / `_time` / `_time_source` / `_read` / `_free`, plus `mio_sequence_to_timeseries`/`_ex` (fan-in), `mio_timeseries_to_sequence` (fan-out) and `mio_sequence_pipeline_run_file`/`_json`. The handle is an ordered **plan** (paths, per-file step indices, times) and owns no mesh, so `mio_sequence_read` hands back an **owned** `mio_mesh*` you free — deliberately unlike `mio_split_result_mesh`'s borrow, which would force the handle to cache everything it produced. That is the C ABI's expression of the streaming guarantee. `mio_sequence_to_timeseries_ex` takes a `mio_write_opts*` for the one option the transient writer has anywhere to put: `MIO_ENCODING_ASCII` selects XDMF's `"XML"` data format (no HDF5 needed) over the default `"HDF"` — `Codec`/`FloatFormat` fail by name, since the writer bypasses the registry and has nowhere to send them. See [sequences](sequences.md). |

> **Four Python-only formats are absent here.** `pmsh`, `zarr`, `cae` and `usd` (v10.35.0, the physics-ML data path — see [formats](formats.md)) are registered in the Python layer only, not in the shared `registry.cpp` dispatch tables, so this ABI does not see them.

Every function is documented in the installed header, [`bindings/c/include/meshioplusplus/meshioplusplus.h`](https://github.com/loumalouomega/meshioplusplus/blob/main/bindings/c/include/meshioplusplus/meshioplusplus.h).

## Format support

All formats with a C++ implementation are available — the same set as the [WASM binding](/wasm#format-support) **plus**, when the build found the dependencies, the HDF5-backed formats (`cgns`, `h5m`, `hmf`, `med`, XDMF's HDF heavy-data path) and the netCDF-backed `exodus`. This includes the write-only visualization formats `svg` and `tikz` (writable, not readable; emitted with the fixed default styling — 3D meshes render their projected boundary skin with the default isometric camera). Probe at runtime with `mio_format_readable()`/`mio_format_writable()`; requesting a compiled-out format fails with a message naming the missing dependency. Formats that only exist in Python (`neuroglancer`, …) are not reachable from C. `mdpa` **is** reachable (since v9.0.0), but only its mesh-level blocks — the C++ reader declines an MDPA file using `Tables`/`Geometries`/`Mesh` blocks or non-empty `Properties`/`ModelPartData` metadata, naming the construct, since the C++ `Mesh` cannot carry them; see [MDPA](/formats/mdpa#c-core).

Parameterized writers use each format's Python-reference default (VTU: binary+zlib, gmsh: 4.1 binary, STL: ASCII, XDMF: HDF when built with HDF5 else XML, …). In particular `stl`/`ply` extract and write the boundary **skin** of a 3D volume mesh, matching the Python default (see [Skin extraction](/extract_skin)); a standalone skin-extraction entry point is not part of the C API yet (documented follow-up). Per-call writer options are a possible future addition.

## Native command-line binary

The core also ships a **Python-free CLI** — the same verbs as the Python command-line tool, built directly on the C++ core, so it needs neither a Python interpreter nor the pybind11 extension.

**Prebuilt binaries:** every [GitHub Release](https://github.com/loumalouomega/meshioplusplus/releases) tag (`v*`) ships a ready-to-run, statically-linked `meshioplusplus` executable for Linux (x86_64 and arm64), macOS (universal x86_64+arm64), and Windows (x86_64) — no install step, no runtime dependency to satisfy (no `libstdc++`/vcredist). These release binaries build with `MESHIOPLUSPLUS_WITH_HDF5/NETCDF/ZLIB=OFF` and the sequential parallel backend, trading those for a single dependency-free file (see `.github/workflows/cli.yml`); build from source (below) for the full HDF5/netCDF/multi-threaded feature set.

**Build from source:** off by default; build it alongside a standalone C++ tree:

```sh
build/configure.sh --cli --build          # -DMESHIOPLUSPLUS_BUILD_CLI=ON
build/cpp-release/meshioplusplus --help
```

Pass `-DMESHIOPLUSPLUS_STATIC_RUNTIME=ON` (alongside `--cli`, via a direct CMake configure) for the same statically-linked-runtime behavior as the release binaries: static `libgcc`/`libstdc++` on GNU, static MSVC CRT (`/MT`) on Windows.

The installed executable is named `meshioplusplus` and mirrors the Python verbs:

```sh
meshioplusplus convert in.msh out.vtu
meshioplusplus info out.vtu
meshioplusplus ascii out.vtu             # in-place; also: binary / compress / decompress
meshioplusplus quality mesh.vtu
meshioplusplus extract-surface vol.vtu surf.vtu
meshioplusplus reorder in.vtu out.vtu --method rcm --report
meshioplusplus diff a.vtu b.vtu          # nonzero exit if different
meshioplusplus merge a.vtu b.vtu out.vtu
```

Format dispatch reuses the shared registry (with a content-sniff fallback on read); ASCII/binary/compress variants call the per-format writers directly. Because it links only the C++ core it inherits this C API's remaining flat-surface limitation: the `convert -s/-d` sets↔data conversions are unavailable (they live only in the Python `Mesh`). Named **sets** are no longer on that list — since v8.1.0 they are [regions](./regions.md) in the core, so `info` prints point/cell/side sets and `diff` compares them. There is also no Python fallback, so a file whose C++ reader raises is not readable by the native CLI — use the Python CLI for those. (Gmsh 4.1's `$Entities` used to be the headline example; since v9.7.0 the C++ reader handles it, and `$Periodic` is what remains.) It works under any [mesh backend](/cpp_backends) and any optional-dependency configuration; verbs that touch a compiled-out HDF5/netCDF format report the missing dependency by name.

## Limitations (v1)

- **Ragged cell blocks are carried** (since v9.15.0): build one with `mio_mesh_add_polygon_block` / `mio_mesh_add_polyhedron_block` and read it back through the `mio_poly_conn_create` / `_get_shape` / `_nodes` / `_face_offsets` / `_cell_offsets` / `_free` snapshot. `mio_mesh_cell_block_info_ex` reports `is_polyhedron` alongside the counts, and `mio_mesh_cell_block_conn` still fails with `MIO_ERR_UNSUPPORTED` because a ragged block has no rectangular buffer to borrow. See [Polyhedra and ragged cells](./polyhedra.md). Note the snapshot is an owning copy rather than a rule-3 borrow -- so, unusually here, it stays valid across mutating calls -- because the `MESHIO` mesh backend stores ragged blocks as nested vectors with no offsets array to point at.
- **`is_ragged`'s type differs between the two paths**, a pre-existing wart kept for compatibility: `mio_mesh_cell_block_info` takes `int32_t*` and `mio_read_metadata_cell_block` takes `int*`. The metadata path also reports no `is_polyhedron`, since `mio_read_metadata`'s per-block record is layout-frozen; create the mesh if you need it.
- **Named regions are carried** (since v8.1.0): `mio_regions_create` / `_count` / `_name` / `_info` / `_entries` / `_free` snapshot a mesh's groups, and `mio_mesh_add_region` adds one. Kinds are `MIO_REGION_POINT` / `_CELL` / `_SIDE`; cell entries are global block-major indices and side entries `(cell, facet)` pairs. See [Named regions](./regions.md). Gmsh, Abaqus and MED (since v9.6.0 — one region per `FAS`/`GRO` group name) map onto them fully (read + write); Exodus (since v8.6.0) reads them (element blocks/node sets/side sets) but does not yet write them. Since v8.7.0, `mio_read_metadata_num_regions` / `_region_name` / `_region_info` (the latter reusing `mio_region_info`'s shape, minus entries) let a caller enumerate a mesh's regions from `mio_read_metadata_create` without a full read wherever the summary already comes from an in-memory mesh.
- **Remaining side-channel metadata** is still dropped: MED's family *ids*/link names and mesh-level metadata (`mesh_name`/`description`/`unit_time`/`unit_coords` — the `MedInfo` fields regions don't carry; group *names* and *membership* are unaffected, see above), OpenFOAM cell tags, and the `ansysinp`/`unv` set channels pending their Phase-2 region mapping. Use the Python API when you need those.
- A `mio_mesh` handle is not thread-safe; distinct handles may be used from distinct threads freely.

## Selective reads and file summaries

`mio_read` is unchanged. `mio_read_ex(path, format, opts)` adds the selective-read options, and `mio_read_metadata_create` returns an opaque summary handle in the style of `mio_split_result` / `mio_convert_cells_result` / `mio_subdivide_result` / `mio_agglomerate_result` / `mio_refine_result` / `mio_partition_result` / `mio_data_info`.

```c
mio_read_opts opts;
mio_read_opts_init(&opts);      /* always -- never zero-fill by hand */
opts.points_only = 1;
opts.time_step = -1;            /* the last step; 0 (the default) is the first */
mio_mesh* mesh = mio_read_ex("big.vtu", "vtu", &opts);

mio_read_metadata* meta = mio_read_metadata_create("run.exo", NULL);
int64_t np = mio_read_metadata_num_points(meta);
int fell_back = mio_read_metadata_fell_back(meta);   /* 1 => the file was read in full */

/* How many time steps may `opts.time_step` name? */
int64_t nsteps = mio_read_metadata_num_time_values(meta);
double* times = malloc((size_t)nsteps * sizeof(double));
mio_read_metadata_time_values(meta, times, nsteps);
mio_read_metadata_free(meta);
```

`opts.time_step` selects a step of a multi-step file (0 = the first, negative counts from the end); out of range fails the call rather than clamping. It was added in v8.6.0 by consuming one of the struct's six `reserved` `int64_t` slots, so **`sizeof(mio_read_opts)` and every preceding field's offset are unchanged** — a caller compiled against the older header still passes a correctly-sized object. This is the only sanctioned way to grow the struct; a gtest pins the tail's width.

A `NULL` `opts.arrays` means *every* array; a non-`NULL` pointer with `num_arrays == 0` means *none*. That distinction is load-bearing and is preserved throughout.

`mio_read_metadata_bbox` returns `MIO_ERR_NOT_FOUND` when no bounding box was computed — the normal case for a native summary, which never decodes the point coordinates. It does not report a zero box.

**ABI note:** `mio_read_opts` is part of the installed library's permanent ABI. It carries reserved capacity and may only ever grow additively — never reorder, resize or repurpose an existing field. Always initialize through `mio_read_opts_init()` so fields added later default sensibly in code compiled against an older header.

### Optional codecs in packages

Conan gains `with_zstd` / `with_lz4` and vcpkg gains `zstd` / `lz4` features, both **off by default** (unlike `with_hdf5`/`with_netcdf`/`with_zlib`). zlib remains the default codec, so existing package IDs and consumers are unaffected. See [Compression codecs](codecs.md).

## v9.1.0 additions

- `mio_read_opts.lenient` — downgrade "this reader cannot represent construct X" to a warning plus a skip (currently mdpa only). It took a second former `reserved` slot, so `sizeof(mio_read_opts)` and every preceding offset are unchanged. See [`doc/selective_read.md`](selective_read.md).
- Transient XDMF: `mio_xdmf_series_create_ex` + `mio_xdmf_series_opts` (append mode, auto-flush), `mio_xdmf_series_flush`, `mio_xdmf_series_finalized`, and `mio_xdmf_series_write_data_arrays` + `mio_named_array` for writing a step from raw solver arrays. `mio_xdmf_series_create` is unchanged. See [`doc/xdmf_time_series.md`](xdmf_time_series.md).

**Not on this ABI, by design:** `MdpaInfo` (MDPA properties bodies and entity names) is dropped by the registry, exactly as `MedInfo`/`ExodusInfo` are — the C ABI cannot hand out a variable-length tree of typed values. `ModelPart`, entity names and the Kratos bridge are likewise unreachable here; that is the reason the installable C++ API (`MESHIOPLUSPLUS_INSTALL_CPP`) exists at all.
