# The operations layer: computations *on* a mesh, not file formats. Full parity
# with the Fortran module's `type(mio_mesh)` procedures.
#
# Ownership rule for the opaque C results (`mio_split_result`,
# `mio_partition_result`, `mio_reorder_result`, `mio_refine_result`,
# `mio_convert_cells_result`, `mio_decimate_result`): read the maps and
# counters first, then TAKE the mesh(es) out of the result, then free it. Every
# `Mesh` handed to the caller therefore owns its handle, and no Julia value can
# dangle when the result goes away.

"""
Copy an int64 index map, shifting to 1-based. The C API's `-1` sentinel
("pruned" / "absent" / "not in this piece") becomes `0`, which is never a valid
1-based index and so stays unambiguous. This is exactly the Fortran module's
rule; the two bindings must agree.
"""
function _shift_map(ptr::Ptr{Cvoid}, n::Integer)
    n <= 0 && return Int[]
    src = unsafe_wrap(Array, Ptr{Int64}(ptr), (Int(n),); own=false)
    out = Vector{Int}(undef, Int(n))
    @inbounds for i in eachindex(out)
        v = src[i]
        out[i] = v < 0 ? 0 : Int(v) + 1
    end
    out
end

"""Read a `(result) -> (data, dtype, n)` zero-copy map getter into a 1-based copy."""
function _result_map(result::Ptr{Cvoid}, sym::Symbol)
    data = Ref{Ptr{Cvoid}}(C_NULL); dt = Ref{Cint}(0); n = Ref{Int64}(0)
    _check(ccall(_sym(sym), Cint,
                 (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Ptr{Cint}, Ptr{Int64}), result, data, dt, n))
    _shift_map(data[], n[])
end

"""Read a `(result, block) -> (data, dtype, n)` map getter into a 1-based copy."""
function _result_block_map(result::Ptr{Cvoid}, sym::Symbol, block::Integer)
    data = Ref{Ptr{Cvoid}}(C_NULL); dt = Ref{Cint}(0); n = Ref{Int64}(0)
    _check(ccall(_sym(sym), Cint,
                 (Ptr{Cvoid}, Int64, Ptr{Ptr{Cvoid}}, Ptr{Cint}, Ptr{Int64}),
                 result, Int64(block), data, dt, n))
    _shift_map(data[], n[])
end

"""Collect all per-block cell maps of an opaque result."""
function _result_cell_maps(result::Ptr{Cvoid}, count_sym::Symbol, map_sym::Symbol)
    n = _check_count(ccall(_sym(count_sym), Int64, (Ptr{Cvoid},), result), string(count_sym))
    [_result_block_map(result, map_sym, b) for b in 0:(n-1)]
end

# --- surface / skin / quality ------------------------------------------------

"""
    extract_surface(mesh; record_parent_ids=false) -> Mesh

Boundary of the mesh's highest-dimension cells: a volume mesh yields its
boundary faces, a 2-D surface mesh its boundary edges.

`record_parent_ids` attaches an Int64 `"surface:parent_cell"` cell-data array
naming each output facet's owning input cell.

Named regions are **not** carried through (the operation warns); see
`doc/extract_surface.md`.
"""
extract_surface(m::Mesh; record_parent_ids::Bool=false) =
    Mesh(_check_ptr(ccall(_sym(:mio_extract_surface), Ptr{Cvoid}, (Ptr{Cvoid}, Cint),
                          _handle(m), record_parent_ids ? Cint(1) : Cint(0))))

"""
    extract_skin(mesh; linearize=false) -> Mesh

Boundary skin of a volume mesh as a new surface mesh. `linearize` emits corner
nodes only (triangle/quad output).
"""
extract_skin(m::Mesh; linearize::Bool=false) =
    Mesh(_check_ptr(ccall(_sym(:mio_extract_skin), Ptr{Cvoid}, (Ptr{Cvoid}, Cint),
                          _handle(m), linearize ? Cint(1) : Cint(0))))

"""
    attach_quality(mesh) -> Mesh

A copy of the mesh with the per-cell quality metrics attached as cell data
(names `"quality:<metric>"`). See `doc/mesh_quality.md`.
"""
attach_quality(m::Mesh) =
    Mesh(_check_ptr(ccall(_sym(:mio_attach_quality), Ptr{Cvoid}, (Ptr{Cvoid},), _handle(m))))

"""
    quality_counts(mesh) -> (; num_cells, num_inverted, num_degenerate)
"""
function quality_counts(m::Mesh)
    nc = Ref{Int64}(0); ni = Ref{Int64}(0); nd = Ref{Int64}(0)
    _check(ccall(_sym(:mio_quality_counts), Cint,
                 (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}), _handle(m), nc, ni, nd))
    (num_cells=Int(nc[]), num_inverted=Int(ni[]), num_degenerate=Int(nd[]))
end

# --- geometry-preserving transforms -----------------------------------------

"""
    transform(mesh, matrix; rotate_vector_data=false) -> Mesh

Affine transform of the point coordinates. `matrix` is 4×4; a point `p` maps to
`M * [p.x, p.y, p.z, 1]`. Connectivity and data are carried through; only the
points change, unless `rotate_vector_data` also rotates vector (trailing
dimension 3) and tensor (trailing dimension 9) point data by the transform's
3×3 linear block.

The matrix is passed to the C API row-major, so a `4×4` Julia matrix is
transposed on the way out — this is the one place a transpose is correct,
because a 4×4 affine matrix is a mathematical object, not a mesh array.
"""
function transform(m::Mesh, matrix::AbstractMatrix{<:Real};
                   rotate_vector_data::Bool=false)
    size(matrix) == (4, 4) || throw(ArgumentError(
        "the affine matrix must be 4x4, got $(size(matrix))"))
    rowmajor = Float64[matrix[i, j] for i in 1:4 for j in 1:4]
    ptr = GC.@preserve rowmajor ccall(_sym(:mio_transform), Ptr{Cvoid},
                                      (Ptr{Cvoid}, Ptr{Cdouble}, Cint),
                                      _handle(m), pointer(rowmajor),
                                      rotate_vector_data ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    clean(mesh; weld=false, atol=1e-12, remove_orphans=true, drop_degenerate=true,
          drop_duplicate_cells=true)
        -> (; mesh, points_welded, points_removed_orphan,
              cells_dropped_degenerate, cells_dropped_duplicate)

One-pass weld / prune / de-duplicate. The default set matches the CLI's
no-flag behaviour: no welding, the other three steps on.
"""
function clean(m::Mesh; weld::Bool=false, atol::Real=1e-12, remove_orphans::Bool=true,
               drop_degenerate::Bool=true, drop_duplicate_cells::Bool=true)
    w = Ref{Int64}(0); o = Ref{Int64}(0); dg = Ref{Int64}(0); du = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_clean), Ptr{Cvoid},
                (Ptr{Cvoid}, Cint, Cdouble, Cint, Cint, Cint,
                 Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
                _handle(m), weld ? Cint(1) : Cint(0), Float64(atol),
                remove_orphans ? Cint(1) : Cint(0), drop_degenerate ? Cint(1) : Cint(0),
                drop_duplicate_cells ? Cint(1) : Cint(0), w, o, dg, du)
    (mesh=Mesh(_check_ptr(ptr)), points_welded=Int(w[]),
     points_removed_orphan=Int(o[]), cells_dropped_degenerate=Int(dg[]),
     cells_dropped_duplicate=Int(du[]))
end

"""
    smooth(mesh; method="taubin", iterations=10, lambda=-1.0, mu=-0.34,
           fix_boundary=true, preserve_features=true, feature_angle=30.0,
           guard_inversion=true)
        -> (; mesh, num_nodes_moved, max_displacement, num_skipped_inversion)

Relax the point coordinates toward their edge-neighbour centroids. Topology and
every data value pass through unchanged — only the points move.

`method` is `"taubin"` (default, shrink-free), `"laplacian"` (stronger per
pass, shrinks the mesh), or `"odt"` (optimal-Delaunay-triangulation
smoothing — tet-only, raises on any other block; moves each free interior
vertex toward the volume-weighted average of its incident tets'
circumcenters).

A **negative `lambda` means "this method's own default"**: 0.5 for
`"laplacian"`, 0.33 for `"taubin"`, 0.9 for `"odt"`. Taubin additionally
needs `mu < -lambda < 0`; `mu` is silently ignored by `"laplacian"` and
`"odt"`. See `doc/smooth.md`.

The `frozen` pin mask of the C++ API is **not reachable across the C ABI** (a
documented flat-ABI gap shared with Fortran).
"""
function smooth(m::Mesh; method::AbstractString="taubin", iterations::Integer=10,
                lambda::Real=-1.0, mu::Real=-0.34, fix_boundary::Bool=true,
                preserve_features::Bool=true, feature_angle::Real=30.0,
                guard_inversion::Bool=true)
    moved = Ref{Int64}(0); disp = Ref{Cdouble}(0.0); skipped = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_smooth), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Cint, Cdouble, Cdouble, Cint, Cint, Cdouble, Cint,
                 Ptr{Int64}, Ptr{Cdouble}, Ptr{Int64}),
                _handle(m), method, Cint(iterations), Float64(lambda), Float64(mu),
                fix_boundary ? Cint(1) : Cint(0), preserve_features ? Cint(1) : Cint(0),
                Float64(feature_angle), guard_inversion ? Cint(1) : Cint(0),
                moved, disp, skipped)
    (mesh=Mesh(_check_ptr(ptr)), num_nodes_moved=Int(moved[]),
     max_displacement=Float64(disp[]), num_skipped_inversion=Int(skipped[]))
end

"""
    optimize_volume(mesh; max_iterations=10, relocate=true, flip=true,
                    preserve_boundary=true, min_improvement=1e-6)
        -> (; mesh, num_flips, num_23_flips, num_32_flips, num_vertices_moved,
              num_tets, min_quality_before, min_quality_after)

ODT-remesh a tetrahedral mesh: raise its worst element quality by relocating
vertices AND flipping connectivity (2-3/3-2, predicate-free). The genuine "ODT
remeshing" sibling of [`remesh_volume`](@ref) (which generates a fresh lattice
mesh) and of `smooth(method="odt")` (which only moves points on fixed
connectivity). Tet-only, raises on any other block. The point set is invariant,
so point data and named Point regions carry; cell data and Cell/Side regions are
dropped. With `preserve_boundary` the boundary surface is exactly preserved.
See `doc/optimize_volume.md`.
"""
function optimize_volume(m::Mesh; max_iterations::Integer=10, relocate::Bool=true,
                         flip::Bool=true, preserve_boundary::Bool=true,
                         min_improvement::Real=1e-6)
    nflips = Ref{Int64}(0); n23 = Ref{Int64}(0); n32 = Ref{Int64}(0)
    nmoved = Ref{Int64}(0); ntets = Ref{Int64}(0)
    qb = Ref{Cdouble}(0.0); qa = Ref{Cdouble}(0.0)
    ptr = ccall(_sym(:mio_optimize_volume), Ptr{Cvoid},
                (Ptr{Cvoid}, Cint, Cint, Cint, Cint, Cdouble,
                 Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                 Ptr{Cdouble}, Ptr{Cdouble}),
                _handle(m), Cint(max_iterations), relocate ? Cint(1) : Cint(0),
                flip ? Cint(1) : Cint(0), preserve_boundary ? Cint(1) : Cint(0),
                Float64(min_improvement), nflips, n23, n32, nmoved, ntets, qb, qa)
    (mesh=Mesh(_check_ptr(ptr)), num_flips=Int(nflips[]), num_23_flips=Int(n23[]),
     num_32_flips=Int(n32[]), num_vertices_moved=Int(nmoved[]), num_tets=Int(ntets[]),
     min_quality_before=Float64(qb[]), min_quality_after=Float64(qa[]))
end

# --- subsetting --------------------------------------------------------------

_crop_mode(mode::Symbol) = mode === :all ? Cint(0) : mode === :any ? Cint(1) :
                           throw(ArgumentError("mode must be :all or :any, got :$mode"))

"""
    crop_bbox(mesh, lo, hi; mode=:all, record_ids=false) -> Mesh

Keep the cells inside the axis-aligned box `lo .<= p .<= hi` (3-element
vectors). `mode` is `:all` (every node inside) or `:any` (at least one).
`record_ids` attaches `crop:original_point_id` / `crop:original_cell_id`.
"""
function crop_bbox(m::Mesh, lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real};
                   mode::Symbol=:all, record_ids::Bool=false)
    l = Float64[lo[1], lo[2], lo[3]]; h = Float64[hi[1], hi[2], hi[3]]
    ptr = GC.@preserve l h ccall(_sym(:mio_crop_bbox), Ptr{Cvoid},
                                 (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
                                 _handle(m), pointer(l), pointer(h), _crop_mode(mode),
                                 record_ids ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    crop_plane(mesh, point, normal; mode=:all, record_ids=false) -> Mesh

Keep the cells in the half-space `(p - point) . normal >= 0`. Unlike
[`slice`](@ref), whole cells are kept and the dimension is unchanged.
"""
function crop_plane(m::Mesh, point::AbstractVector{<:Real},
                    normal::AbstractVector{<:Real}; mode::Symbol=:all,
                    record_ids::Bool=false)
    p = Float64[point[1], point[2], point[3]]; n = Float64[normal[1], normal[2], normal[3]]
    ptr = GC.@preserve p n ccall(_sym(:mio_crop_plane), Ptr{Cvoid},
                                 (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Cint),
                                 _handle(m), pointer(p), pointer(n), _crop_mode(mode),
                                 record_ids ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    crop_predicate(mesh, array; compare="<", value=0.0, record_ids=false) -> Mesh

Crop to the cells whose value in the scalar `cell_data` array `array` satisfies
a comparison -- one of `"<"`, `"<="`, `">"`, `">="`, `"=="`, `"!="`, the same
vocabulary [`refine`](@ref)'s `where_op` uses, evaluated by the same C++
function. **A non-finite cell value never matches**, whatever the comparison.

There is deliberately no `mode`: `crop_bbox`/`crop_plane` test points and then
need an all/any rule, whereas a `cell_data` predicate is already one value per
cell and has nothing to reduce.

Inside/outside a surface composes rather than being its own function:

```julia
f = distance_to_surface(m, skin; location=:center).mesh
inside = crop_predicate(f, "sdf:distance"; compare="<", value=0.0)
```
"""
function crop_predicate(m::Mesh, array::AbstractString; compare="<", value::Real=0.0,
                        record_ids::Bool=false)
    code = get(_REFINE_COMPARES, String(compare)) do
        throw(ArgumentError("meshio++: unknown comparison '$(compare)'"))
    end
    ptr = ccall(_sym(:mio_crop_predicate), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Cint, Cdouble, Cint),
                _handle(m), String(array), code, Cdouble(value),
                record_ids ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    slice(mesh, origin, normal; record_parent_ids=false) -> Mesh

Planar cross-section: the actual **intersection** with the plane, one
topological dimension below the cut cells (a volume mesh yields a
triangle/quad surface, a 2-D surface mesh a line mesh). Contrast
[`crop_plane`](@ref), which keeps whole cells on one side.

Empty when the plane misses the geometry — not an error. See `doc/slice.md`.
"""
function slice(m::Mesh, origin::AbstractVector{<:Real}, normal::AbstractVector{<:Real};
               record_parent_ids::Bool=false)
    o = Float64[origin[1], origin[2], origin[3]]
    n = Float64[normal[1], normal[2], normal[3]]
    ptr = GC.@preserve o n ccall(_sym(:mio_slice), Ptr{Cvoid},
                                 (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Cint),
                                 _handle(m), pointer(o), pointer(n),
                                 record_parent_ids ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    isosurface(mesh, array, isovalues; component=-1, record_parent_ids=false) -> Mesh

Level set of a scalar **point-data** field — the data-driven sibling of
[`slice`](@ref), sharing its marching-tetrahedra cutter.

`isovalues` may be a single number or a vector; all contours land in the one
returned mesh, tagged per cell with a Float64 `"iso:value"` and an Int64
`"iso:index"`. `component` selects a component of a multi-component array;
negative means the row magnitude.

Naming a *cell-data* array fails by name: cell data is piecewise constant and
has no level set — convert it with [`data_cell_to_point`](@ref) first. See
`doc/isosurface.md`.
"""
function isosurface(m::Mesh, array::AbstractString, isovalues;
                    component::Integer=-1, record_parent_ids::Bool=false)
    vals = Float64[Float64(v) for v in (isovalues isa Number ? (isovalues,) : isovalues)]
    isempty(vals) && throw(ArgumentError("at least one isovalue is required"))
    ptr = GC.@preserve vals ccall(_sym(:mio_isosurface), Ptr{Cvoid},
                                  (Ptr{Cvoid}, Cstring, Ptr{Cdouble}, Cint, Cint, Cint),
                                  _handle(m), array, pointer(vals), Cint(length(vals)),
                                  Cint(component), record_parent_ids ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    gradient(mesh, array; operator=:gradient, method=:green_gauss, location=:cell,
             output="", component=-1, overwrite=false)
        -> (; mesh, num_skipped, num_fallback)

Differentiate a **point-data** field: its gradient, divergence or curl.

`operator` is `:gradient`, `:divergence` or `:curl`; `method` is `:green_gauss`
(exact for a linear field on any cell) or `:least_squares` (a fit over the
node-sharing neighbours); `location` is `:cell` or `:point`. The result is named
`"<array>:<operator>"` unless `output` overrides it.

An `nc`-component input yields `3 * nc` gradient components, flat and row-major
as `[component i][derivative j]`; divergence yields 1 and curl always 3, both
needing a 2- or 3-component field. `component` selects a single component of the
gradient; **negative means every component** — deliberately the opposite of
[`isosurface`](@ref), where negative means the row magnitude.

Cells that cannot be differentiated yield `NaN` and are counted in
`num_skipped`; least-squares cells with a degenerate neighbourhood fall back to
Green-Gauss and are counted in `num_fallback`.

Naming a *cell-data* array fails by name: cell data is piecewise constant and
has no derivative — convert it with [`data_cell_to_point`](@ref) first. See
`doc/gradient.md`.
"""
function gradient(m::Mesh, array::AbstractString; operator=:gradient,
                  method=:green_gauss, location=:cell, output::AbstractString="",
                  component::Integer=-1, overwrite::Bool=false)
    skipped = Ref{Int64}(0)
    fallback = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_gradient), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Cstring, Cstring, Cstring, Cstring, Cint, Cint,
                 Ptr{Int64}, Ptr{Int64}),
                _handle(m), array, _op_name(operator), _method_name(method),
                String(location), output, Cint(component),
                overwrite ? Cint(1) : Cint(0), skipped, fallback)
    (mesh=Mesh(_check_ptr(ptr)), num_skipped=Int(skipped[]),
     num_fallback=Int(fallback[]))
end

# The C API spells these with hyphens; Julia symbols cannot carry one, so the
# idiomatic underscore form is translated here rather than forcing callers to
# pass strings.
_op_name(op) = String(op)
_method_name(m) = replace(String(m), '_' => '-')

"""
    hessian(mesh, array; method=:green_gauss, location=:cell, output="",
            overwrite=false) -> (; mesh, num_skipped, num_fallback)

The Hessian (second derivative) of a **scalar point-data** field —
[`gradient`](@ref)'s companion one order further, for curvature-based
adaptive refinement.

A composition of TWO `gradient` calls, not a new numerical kernel: the field
is differentiated once (point location), then that `(n, 3)` gradient is
differentiated again with the default `:gradient` operator, producing
`(n, 9)` — the flattened row-major 3x3 Hessian, `H[i,j]` at index `i*3+j`.
`method` is forwarded to BOTH internal passes. The result is named
`"<array>:hessian"` unless `output` overrides it.

A field that is at most LINEAR has an exactly zero Hessian everywhere — the
one mesh-shape-independent guarantee. For a genuinely quadratic field the
composition is exact on a structured/symmetric mesh away from its own
boundary and a good, standard, but genuinely approximate curvature estimate
on an irregular mesh (see `doc/hessian.md`). Input must have exactly one
component — a vector field's Hessian is a separate quantity per component.

Cells that cannot be evaluated yield `NaN` and are counted in `num_skipped`;
least-squares cells with a degenerate neighbourhood in either internal pass
fall back to Green-Gauss and are counted in `num_fallback` (summed over both
passes).

A curvature-driven refinement indicator needs no new function: `norm(...)`
in [`data_calc`](@ref) on the 9-component output is exactly its Frobenius
norm, ready for [`refine`](@ref)'s `where` selector.
"""
function hessian(m::Mesh, array::AbstractString; method=:green_gauss, location=:cell,
                 output::AbstractString="", overwrite::Bool=false)
    skipped = Ref{Int64}(0)
    fallback = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_hessian), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Cstring, Cstring, Cstring, Cint, Ptr{Int64}, Ptr{Int64}),
                _handle(m), array, _method_name(method), String(location), output,
                overwrite ? Cint(1) : Cint(0), skipped, fallback)
    (mesh=Mesh(_check_ptr(ptr)), num_skipped=Int(skipped[]), num_fallback=Int(fallback[]))
end

"""
    estimate_error(mesh, array; method=:zz, marking=:none, marking_value=0.0,
                   output="", marked="", overwrite=false)
        -> (; mesh, global_error, num_skipped, num_marked)

Estimate the per-cell recovered-gradient (Zienkiewicz-Zhu) error of a
**point-data** field, and optionally mark cells for refinement.

A composition of [`gradient`](@ref) (Green-Gauss, cell location) with the
measure-weighted point<->cell averaging round trip: the indicator is
`sqrt(|measure| * sum((recovered - raw)^2))` per cell, attached as `output`
(default `"error:zz"`). `marking` is `:none` (default), `:absolute`,
`:fraction`, or `:dorfler`; when not `:none` a second Int64 0/1 array `marked`
(default `"error:marked"`) is attached too, so [`refine`](@ref)'s own `where`
selector needs no change at all — the intended use is
`refine(mesh, where="error:marked > 0.5")`. `marking_value`'s meaning depends
on `marking`: an absolute indicator threshold, a fraction in `(0, 1]` of
cells, or the Dörfler bulk fraction theta in `(0, 1]`.

Cells that cannot be evaluated read `NaN` in the indicator array and `0`
(never `NaN`) in the marking array, and are counted in `num_skipped`
(excluded from `global_error` and from `num_marked`).

Naming a *cell-data* array fails by name, for the same reason `gradient` does.
See `doc/gradient.md`.
"""
function estimate_error(m::Mesh, array::AbstractString; method=:zz, marking=:none,
                        marking_value::Real=0.0, output::AbstractString="",
                        marked::AbstractString="", overwrite::Bool=false)
    global_error = Ref{Cdouble}(0.0)
    skipped = Ref{Int64}(0)
    nmarked = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_estimate_error), Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Cstring, Cstring, Cdouble, Cstring, Cstring, Cint,
                 Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}),
                _handle(m), array, String(method), String(marking), Cdouble(marking_value),
                output, marked, overwrite ? Cint(1) : Cint(0), global_error, skipped, nmarked)
    (mesh=Mesh(_check_ptr(ptr)), global_error=Float64(global_error[]),
     num_skipped=Int(skipped[]), num_marked=Int(nmarked[]))
end

"""
    remesh(mesh, num_clusters; subdivide=-1, subsample_ratio=10.0,
           max_subdivide=4, max_iterations=100, max_repair_passes=10,
           metric=:isotropic, gradation=0.0, preserve_boundary=true,
           max_anisotropy=4.0)
        -> (; mesh, num_clusters, num_iterations, subdivide_applied,
             num_isolated_clusters, num_non_manifold_vertices)

Remesh a surface uniformly by approximated centroidal Voronoi diagram (ACVD)
clustering: replace its triangulation with a new, near-uniformly-sized,
well-shaped one at `num_clusters` vertices. Unlike every other
resolution-changing operation, the output has NEW points and NEW
connectivity with no correspondence to `mesh` — point/cell data and named
regions are dropped, field data is carried.

`metric` is `:isotropic` (default; area-weighted centroidal distance, fast,
rounds sharp features), `:quadric` (Garland-Heckbert quadric error,
preserves sharp edges/corners at extra cost per candidate move), or
`:anisotropic` (clusters shaped by a local curvature tensor — elongated
along low-curvature directions — see `max_anisotropy`). `subdivide` defaults
to automatic (`-1`): the smallest count of uniform `refine` passes reaching
`subsample_ratio` items per cluster, capped at `max_subdivide`; `0` disables
subdivision. `gradation` is the curvature-gradation exponent `gamma` in the
item weight `area * kappa^gamma`; `0.0` (default) disables gradation
entirely and reproduces plain area weighting. `preserve_boundary` (default
`true`) detects the input's open boundary (if any), seeds it before the
interior, and emits a `line` dual cell along boundary edges whose endpoints
land in different clusters — a no-op on a closed mesh. `max_anisotropy`
(default `4.0`, a measured value — see `kRemeshDefaultMaxAnisotropy`'s doc
comment in `remesh.hpp`) is, under `metric=:anisotropic`, the maximum ratio
between the two in-plane target edge lengths a curvature tensor may
request; `1.0` recovers the isotropic shape exactly. Must be at least `1.0`,
and an error to set away from the default under any other metric.
`num_isolated_clusters` and `num_non_manifold_vertices` are two distinct
causes of non-manifold output (disconnected clusters vs. "bowtie" vertices)
that repair could not fully fix; check both rather than assuming.

Goes through `mio_remesh_ex`/`mio_remesh_opts` rather than the flat
`mio_remesh` — the `refine`/`mio_refine_ex` precedent, needed because
`mio_remesh` is a flat C function with no room to grow (it already changed
once).

Attribution: the isotropic clustering engine is derived from
[pyacvd](https://github.com/pyvista/pyacvd) (MIT, (c) 2017-2024 The PyVista
Developers), itself an independent implementation of the published research
of S. Valette and J.-M. Chassery, not of the CeCILL-B licensed ACVD project.
See `doc/remesh.md`.
"""
function remesh(m::Mesh, num_clusters::Integer; subdivide::Integer=-1,
                subsample_ratio::Real=10.0, max_subdivide::Integer=4,
                max_iterations::Integer=100, max_repair_passes::Integer=10,
                metric::Symbol=:isotropic, gradation::Real=0.0,
                preserve_boundary::Bool=true, max_anisotropy::Real=4.0)
    metric_c = Vector{UInt8}(codeunits(String(metric)))
    push!(metric_c, 0x00)
    report = Ref{_CRemeshReport}()
    ptr = GC.@preserve metric_c begin
        opts = _CRemeshOpts(Int64(num_clusters), Int32(subdivide), Int32(max_subdivide),
                            Cdouble(subsample_ratio), Int32(max_iterations),
                            Int32(max_repair_passes), Cstring(pointer(metric_c)),
                            Cdouble(gradation), preserve_boundary ? Int32(1) : Int32(0),
                            Int32(0), Cdouble(max_anisotropy),
                            (Cdouble(0), Cdouble(0), Cdouble(0)),
                            (Int64(0), Int64(0), Int64(0), Int64(0)))
        ccall(_sym(:mio_remesh_ex), Ptr{Cvoid},
              (Ptr{Cvoid}, Ref{_CRemeshOpts}, Ptr{_CRemeshReport}),
              _handle(m), Ref(opts), report)
    end
    r = _check_ptr(ptr)
    rep = report[]
    (mesh=Mesh(r), num_clusters=Int(rep.num_clusters),
     num_iterations=Int(rep.num_iterations), subdivide_applied=Int(rep.subdivide_applied),
     num_isolated_clusters=Int(rep.num_isolated_clusters),
     num_non_manifold_vertices=Int(rep.num_non_manifold_vertices))
end

"""
    remesh_volume(mesh; resolution=nothing, cell_size=nothing, bounds=nothing,
                  padding=0.0, padding_relative=0.1, max_cells=20_000_000,
                  max_tets=20_000_000, warp_fraction=0.35, sign=:pseudonormal,
                  watertight_check=:warn)
        -> (; mesh, num_tets, num_vertices_warped, num_tets_rejected,
             num_non_manifold_edges, input_quality)

Retetrahedralize `mesh` (a volume mesh, or a closed surface) at a
caller-chosen resolution by isosurface stuffing — the volumetric sibling of
[`remesh`](@ref). Unlike `remesh`, `mesh` may be a VOLUME mesh directly (its
boundary is extracted internally) as well as a closed surface. Same
no-correspondence-with-the-input output contract as `remesh`: new points, new
connectivity, `point_data`/`cell_data`/named regions dropped, `field_data`
carried.

Exactly one of `resolution`/`cell_size` must be given, sizing a
body-centered cubic (BCC) lattice whose uncut tets have dihedral angles from
a fixed, mesh-size-independent set. `warp_fraction` (default `0.35`) moves
lattice vertices near the surface onto it, trading a small, measured chance
of non-manifold boundary edges (reported as `num_non_manifold_edges`) for
substantially better boundary tet quality; `0` disables warping and gives an
exactly watertight but lower-quality boundary. See `doc/remesh_volume.md`
for the measured tradeoff.

Attribution: implemented from the PUBLISHED DESCRIPTION of Labelle &
Shewchuk, "Isosurface Stuffing" (SIGGRAPH 2007) only — neither the paper's
own reference implementation (Stellar) nor TetGen (AGPL-3.0) is read or
vendored here. See `doc/remesh_volume.md`.
"""
function remesh_volume(m::Mesh; resolution=nothing, cell_size=nothing, bounds=nothing,
                       padding::Real=0.0, padding_relative::Real=0.1,
                       max_cells::Integer=20_000_000, max_tets::Integer=20_000_000,
                       warp_fraction::Real=0.35, sign=:pseudonormal, weight=:angle,
                       watertight_check=:warn, surface_region::AbstractString="",
                       grid_cell_size::Real=0.0, max_winding_work::Real=2.0e9)
    res = resolution === nothing ? Int64[] : Int64[Int64(v) for v in resolution]
    (resolution === nothing || length(res) == 3) ||
        throw(ArgumentError("resolution must have three entries"))
    bnd = bounds === nothing ? Cdouble[] : Cdouble[Cdouble(v) for v in bounds]
    (bounds === nothing || length(bnd) == 6) ||
        throw(ArgumentError("bounds must have six entries"))

    region_buf = Vector{UInt8}(codeunits(String(surface_region)))
    push!(region_buf, 0x00)
    report = Ref{_CRemeshVolumeReport}()
    ptr = GC.@preserve res bnd region_buf begin
        inner = _sdf_opts(; sign, weight, location=:corner, band=0.0,
                          record_closest_cell=false, record_inside=false, watertight_check,
                          surface_region, grid_cell_size, max_winding_work, region_buf)
        opts = _CRemeshVolumeOpts(isempty(res) ? Ptr{Int64}(C_NULL) : pointer(res),
                                  isempty(bnd) ? Ptr{Cdouble}(C_NULL) : pointer(bnd),
                                  Cdouble(cell_size === nothing ? 0.0 : cell_size),
                                  Cdouble(padding), Cdouble(padding_relative),
                                  Int64(max_cells), Int64(max_tets), Cdouble(warp_fraction),
                                  (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)),
                                  inner)
        ccall(_sym(:mio_remesh_volume_ex), Ptr{Cvoid},
              (Ptr{Cvoid}, Ref{_CRemeshVolumeOpts}, Ptr{_CRemeshVolumeReport}),
              _handle(m), Ref(opts), report)
    end
    r = _check_ptr(ptr)
    rep = report[]
    q = rep.input_quality
    (mesh=Mesh(r), num_tets=Int(rep.num_tets),
     num_vertices_warped=Int(rep.num_vertices_warped),
     num_tets_rejected=Int(rep.num_tets_rejected),
     num_non_manifold_edges=Int(rep.num_non_manifold_edges),
     input_quality=(boundary_edges=Int(q.boundary_edges),
                    non_manifold_edges=Int(q.non_manifold_edges),
                    inconsistent_pairs=Int(q.inconsistent_pairs),
                    degenerate_triangles=Int(q.degenerate_triangles),
                    watertight=q.watertight != 0))
end

# --- combining / comparing ---------------------------------------------------

"""
    merge(meshes; weld=false, atol=1e-12, source_tag=true,
          data_policy=:intersection, drop_duplicate_cells=false) -> Mesh

Combine two or more meshes into one. `data_policy` is `:intersection` (keep
only keys present in every input) or `:fill` (union, NaN for missing rows).
`source_tag` adds an Int64 `"source_mesh_id"` cell-data array.

Not exported (it would shadow `Base.merge`): call it qualified.
"""
function merge(meshes::AbstractVector{Mesh}; weld::Bool=false, atol::Real=1e-12,
               source_tag::Bool=true, data_policy::Symbol=:intersection,
               drop_duplicate_cells::Bool=false)
    isempty(meshes) && throw(ArgumentError("merge needs at least one mesh"))
    policy = data_policy === :intersection ? Cint(0) : data_policy === :fill ? Cint(1) :
             throw(ArgumentError("data_policy must be :intersection or :fill"))
    handles = Ptr{Cvoid}[_handle(x) for x in meshes]
    ptr = GC.@preserve meshes handles ccall(
        _sym(:mio_merge), Ptr{Cvoid},
        (Ptr{Ptr{Cvoid}}, Int64, Cint, Cdouble, Cint, Cint, Cint),
        pointer(handles), Int64(length(handles)), weld ? Cint(1) : Cint(0),
        Float64(atol), source_tag ? Cint(1) : Cint(0), policy,
        drop_duplicate_cells ? Cint(1) : Cint(0))
    Mesh(_check_ptr(ptr))
end

"""
    undo_green(coarse, fine) -> (; mesh, num_groups_undone, num_cells_removed)

Green-element undo: restore `fine`'s transitional (closure-only) cells back to
their original parent, read verbatim from `coarse` — a lookup, not a
reconstruction, since `refine` never renumbers or prunes points. Undoes
`refine`'s known quality-degradation issue with repeated selective passes over
the same region ("restore the parent and re-split from scratch").

`fine` must carry `refine:cell_id`/`refine:parent_id`/`refine:level` (i.e. it
must come from `refine(coarse, ...; record_hierarchy=true,
record_levels=true)`); `coarse` must be the exact mesh that call was run on.
The six reserved `refine:*` arrays are dropped from the output. Only a
single-pass (`levels=1`) hierarchy is supported. See `doc/undo_green.md`.
"""
function undo_green(coarse::Mesh, fine::Mesh)
    ngroups = Ref{Int64}(0); nremoved = Ref{Int64}(0)
    ptr = ccall(_sym(:mio_undo_green), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}),
                _handle(coarse), _handle(fine), ngroups, nremoved)
    (mesh=Mesh(_check_ptr(ptr)), num_groups_undone=Int(ngroups[]),
     num_cells_removed=Int(nremoved[]))
end

"""
    interpolate(source, target; method="nearest", arrays=String[],
                extrapolate=false, default_value=0.0, on_conflict="error") -> Mesh

Sample data arrays from `source` onto `target` (cross-mesh field transfer).
The result is a copy of `target` — geometry, connectivity and its own data
preserved exactly — with the requested source arrays sampled onto it.

An empty `arrays` means every source point-data array; cell data transfers only
when named explicitly. `method` is `"nearest"` or `"barycentric"`;
`on_conflict` is `"error"`, `"overwrite"` or `"suffix"`. See
`doc/interpolate.md`.
"""
function interpolate(source::Mesh, target::Mesh; method::AbstractString="nearest",
                     arrays=String[], extrapolate::Bool=false,
                     default_value::Real=0.0, on_conflict::AbstractString="error")
    sh = _handle(source); th = _handle(target)
    ptr = _with_names(arrays) do names_ptr, count
        ccall(_sym(:mio_interpolate), Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cvoid}, Cstring, Ptr{Cstring}, Int64, Cint, Cdouble, Cstring),
              sh, th, method, names_ptr, count, extrapolate ? Cint(1) : Cint(0),
              Float64(default_value), on_conflict)
    end
    Mesh(_check_ptr(ptr))
end

"""
    conservative_interpolate(source, target; arrays=String[],
                             default_value=0.0, on_conflict="error") -> Mesh

Mass-preserving cross-mesh field transfer: an exact overlap-measure weighted
remap, so that over the region the two meshes share, `sum(target value *
target measure)` equals `sum(source value * source measure)` — the property
[`interpolate`](@ref)'s `"barycentric"` mode does not have. Both meshes are
simplexified first (accepting ragged/polyhedron blocks for free).

Unlike `interpolate`, an empty `arrays` means every source point_data AND
cell_data array — there is one algorithm regardless of location. Output
arrays are always Float64. `on_conflict` is `"error"`, `"overwrite"` or
`"suffix"`. See `doc/conservative_interpolate.md`.
"""
function conservative_interpolate(source::Mesh, target::Mesh; arrays=String[],
                                   default_value::Real=0.0,
                                   on_conflict::AbstractString="error")
    sh = _handle(source); th = _handle(target)
    ptr = _with_names(arrays) do names_ptr, count
        ccall(_sym(:mio_conservative_interpolate), Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cstring}, Int64, Cdouble, Cstring),
              sh, th, names_ptr, count, Float64(default_value), on_conflict)
    end
    Mesh(_check_ptr(ptr))
end

"""
    meshes_equal(a, b; atol=0.0, rtol=0.0, unordered=false) -> Bool

Are two meshes equal within tolerance (i.e. the diff verdict is not
`:different`)? Named regions are **not** considered — they never reach the C++
comparison.
"""
function meshes_equal(a::Mesh, b::Mesh; atol::Real=0.0, rtol::Real=0.0,
                      unordered::Bool=false)
    out = Ref{Cint}(0)
    _check(ccall(_sym(:mio_meshes_equal), Cint,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Cint}),
                 _handle(a), _handle(b), Float64(atol), Float64(rtol),
                 unordered ? Cint(1) : Cint(0), out))
    out[] != 0
end

"""
    DiffReport

Structured result of [`diff`](@ref). `verdict` is `:identical`,
`:equal_within_tolerance` or `:different`.
"""
struct DiffReport
    verdict::Symbol
    points::NamedTuple{(:max_abs_error, :max_rel_error, :worst_index, :num_exceeding),
                       Tuple{Float64,Float64,Int,Int}}
    blocks::Vector{NamedTuple{(:type_mismatch, :count_mismatch, :conn_mismatch_count),
                              Tuple{Bool,Bool,Int}}}
end

const _VERDICTS = (:identical, :equal_within_tolerance, :different)

"""
    diff(a, b; atol=0.0, rtol=0.0, unordered=false) -> DiffReport

Compare two meshes: points, cell blocks and point/cell/field data, within
`abs_err <= atol + rtol*|expected|` with `expected` taken from `a`.
`unordered` matches points by spatial proximity instead of by index.

Named regions are **not** compared (they never reach the C++ core). Not
exported (it would shadow `Base.diff`): call it qualified.
"""
function diff(a::Mesh, b::Mesh; atol::Real=0.0, rtol::Real=0.0, unordered::Bool=false)
    out = Ref{Ptr{Cvoid}}(C_NULL)
    _check(ccall(_sym(:mio_diff), Cint,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Cdouble, Cdouble, Cint, Ptr{Ptr{Cvoid}}),
                 _handle(a), _handle(b), Float64(atol), Float64(rtol),
                 unordered ? Cint(1) : Cint(0), out))
    result = out[]
    try
        v = ccall(_sym(:mio_diff_result_verdict), Cint, (Ptr{Cvoid},), result)
        mabs = Ref{Cdouble}(0.0); mrel = Ref{Cdouble}(0.0)
        worst = Ref{Int64}(0); nex = Ref{Int64}(0)
        _check(ccall(_sym(:mio_diff_result_point_summary), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}),
                     result, mabs, mrel, worst, nex))
        nb = ccall(_sym(:mio_diff_result_num_block_diffs), Int64, (Ptr{Cvoid},), result)
        nb < 0 && (nb = 0)
        blocks = NamedTuple{(:type_mismatch, :count_mismatch, :conn_mismatch_count),
                            Tuple{Bool,Bool,Int}}[]
        for i in 0:(nb-1)
            tm = Ref{Cint}(0); cm = Ref{Cint}(0); cc = Ref{Int64}(0)
            _check(ccall(_sym(:mio_diff_result_block), Cint,
                         (Ptr{Cvoid}, Int64, Ptr{Cint}, Ptr{Cint}, Ptr{Int64}),
                         result, i, tm, cm, cc))
            push!(blocks, (type_mismatch=tm[] != 0, count_mismatch=cm[] != 0,
                           conn_mismatch_count=Int(cc[])))
        end
        # worst_index is a flat index into the coordinate buffer: 1-based here,
        # with the C API's -1 "no offender" becoming 0.
        w = worst[] < 0 ? 0 : Int(worst[]) + 1
        DiffReport(_VERDICTS[v+1],
                   (max_abs_error=Float64(mabs[]), max_rel_error=Float64(mrel[]),
                    worst_index=w, num_exceeding=Int(nex[])), blocks)
    finally
        ccall(_sym(:mio_diff_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

# --- statistics --------------------------------------------------------------

"""
    stats(mesh) -> NamedTuple

Read-only geometric statistics: counts, bounding box, extent, centroid, total
area, signed / unsigned volume and the inverted-cell count.

Per-cell-type counts are **not carried across the C ABI** (a documented gap);
use [`cell_block_types`](@ref) with [`cell_block_info`](@ref) instead.
"""
function stats(m::Mesh)
    out = Ref{_CStatsReport}()
    _check(ccall(_sym(:mio_stats), Cint, (Ptr{Cvoid}, Ptr{_CStatsReport}), _handle(m), out))
    r = out[]
    (num_points=Int(r.num_points), num_cells=Int(r.num_cells),
     bbox_min=r.bbox_min, bbox_max=r.bbox_max, extent=r.extent, centroid=r.centroid,
     total_area=r.total_area, signed_volume=r.signed_volume,
     unsigned_volume=r.unsigned_volume, num_inverted=Int(r.num_inverted))
end

"""
    compute_bandwidth(mesh) -> Int

Connectivity bandwidth: the maximum over all cells of (max node index − min
node index) within a cell. A before/after measure for [`reorder`](@ref).
"""
compute_bandwidth(m::Mesh) =
    Int(_check_count(ccall(_sym(:mio_compute_bandwidth), Int64, (Ptr{Cvoid},), _handle(m)),
                     "mio_compute_bandwidth"))

# --- opaque-result operations ------------------------------------------------

"""
    reorder(mesh, method) -> (; mesh, node_perm, cell_perms)

Renumber a mesh as a **pure permutation** of node and element indices, to
reduce sparse-matrix bandwidth or improve cache locality. `method` is `"rcm"`,
`"morton"` or `"hilbert"`.

`node_perm` and each entry of `cell_perms` are 1-based old→new index vectors.
See `doc/reorder.md`.
"""
function reorder(m::Mesh, method::AbstractString)
    result = _check_ptr(ccall(_sym(:mio_reorder), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring),
                              _handle(m), method))
    try
        node_perm = _result_map(result, :mio_reorder_result_node_perm)
        cell_perms = _result_cell_maps(result, :mio_reorder_result_num_cell_perms,
                                       :mio_reorder_result_cell_perm)
        out = Mesh(_check_ptr(ccall(_sym(:mio_reorder_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, node_perm=node_perm, cell_perms=cell_perms)
    finally
        ccall(_sym(:mio_reorder_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    split(mesh; by="type", tag="") -> Vector{Pair{String,Mesh}}

Partition a mesh into submeshes by a criterion: `by` is `"type"` (one piece per
cell type), `"component"` (connected components) or `"region"`/`"tag"` (the
distinct values of an integer cell-data array, named by `tag`, or the first
integer array when `tag` is empty).

Each pair is `key => piece`. Every piece owns its handle. Not exported (it
would shadow `Base.split`): call it qualified. See `doc/split.md`.
"""
function split(m::Mesh; by::AbstractString="type", tag::AbstractString="")
    result = _check_ptr(ccall(_sym(:mio_split), Ptr{Cvoid}, (Ptr{Cvoid}, Cstring, Cstring),
                              _handle(m), by, tag))
    try
        n = _check_count(ccall(_sym(:mio_split_result_count), Int64, (Ptr{Cvoid},), result),
                         "mio_split_result_count")
        out = Pair{String,Mesh}[]
        for i in 0:(n-1)
            key = _getstring() do buf, len
                ccall(_sym(:mio_split_result_key), Int64,
                      (Ptr{Cvoid}, Int64, Ptr{UInt8}, Int64), result, i, buf, len)
            end
            piece = Mesh(_check_ptr(ccall(_sym(:mio_split_result_take_mesh), Ptr{Cvoid},
                                          (Ptr{Cvoid}, Int64), result, i)))
            push!(out, key => piece)
        end
        out
    finally
        ccall(_sym(:mio_split_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    convert_cells(mesh, mode; record_parent_ids=false) -> (; mesh, point_map, cell_maps)

Convert the **element representation** (not the file format). `mode` is
`"linearize"` (higher-order cells → their linear base, pruning orphaned
nodes), `"simplexify"` (decompose into same-dimension simplices) or
`"elevate"` (linear → serendipity quadratic).

`point_map` is 1-based input-point → output-point, `0` where the point was
pruned; each `cell_maps[b]` is input cell → its **first child** in the
corresponding output block. See `doc/convert_cells.md`.
"""
function convert_cells(m::Mesh, mode::AbstractString; record_parent_ids::Bool=false)
    result = _check_ptr(ccall(_sym(:mio_convert_cells), Ptr{Cvoid},
                              (Ptr{Cvoid}, Cstring, Cint), _handle(m), mode,
                              record_parent_ids ? Cint(1) : Cint(0)))
    try
        pm = _result_map(result, :mio_convert_cells_result_point_map)
        cm = _result_cell_maps(result, :mio_convert_cells_result_num_cell_maps,
                               :mio_convert_cells_result_cell_map)
        out = Mesh(_check_ptr(ccall(_sym(:mio_convert_cells_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, point_map=pm, cell_maps=cm)
    finally
        ccall(_sym(:mio_convert_cells_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    subdivide(mesh; record_parent_ids=false) -> (; mesh, cell_maps)

Polyhedrally refine: split every eligible 3-D cell into one polyhedral child
per face, connected to a new interior point. Needs no per-type template
table -- tabulated types (reduced to corners for a quadratic variant) and
existing polyhedron blocks are handled uniformly. Automatically conforming,
unlike [`refine`](@ref): a shared face between two input cells is never
touched. Non-3-D blocks and the full-Lagrange family (no face table) pass
through unchanged.

Unlike [`convert_cells`](@ref), there is no `point_map` -- subdivide never
prunes or renumbers an original point. Each `cell_maps[b]` is input cell →
the index of its **first** child (one per face) in the corresponding output
block. See `doc/subdivide.md`.
"""
function subdivide(m::Mesh; record_parent_ids::Bool=false)
    result = _check_ptr(ccall(_sym(:mio_subdivide), Ptr{Cvoid},
                              (Ptr{Cvoid}, Cint), _handle(m),
                              record_parent_ids ? Cint(1) : Cint(0)))
    try
        cm = _result_cell_maps(result, :mio_subdivide_result_num_cell_maps,
                               :mio_subdivide_result_cell_map)
        out = Mesh(_check_ptr(ccall(_sym(:mio_subdivide_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, cell_maps=cm)
    finally
        ccall(_sym(:mio_subdivide_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    agglomerate(mesh; target_group_size=8) -> (; mesh, cell_map)

Polyhedrally coarsen: merge groups of cells into single larger polyhedral
cells via greedy seed-and-grow over the mesh's shared-face dual, absorbing
face-adjacent neighbours into a group until it reaches `target_group_size`
(a short group at a mesh boundary or pocket is expected, not an error).
Non-volume blocks pass through unchanged; points are never pruned or
renumbered ([`clean`](@ref) with `remove_orphans=true` is the follow-up for
a minimal point set). `target_group_size=1` groups every cell by itself.

Unlike [`subdivide`](@ref)'s per-block `cell_maps`, `cell_map` is a single
**flat** 1-based array (input global cell → output global cell) -- an
output cell's index is a function of which group it joined, not which
input block it came from. See `doc/agglomerate.md`.
"""
function agglomerate(m::Mesh; target_group_size::Integer=8)
    result = _check_ptr(ccall(_sym(:mio_agglomerate), Ptr{Cvoid},
                              (Ptr{Cvoid}, Int64), _handle(m), Int64(target_group_size)))
    try
        cm = _result_map(result, :mio_agglomerate_result_cell_map)
        out = Mesh(_check_ptr(ccall(_sym(:mio_agglomerate_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, cell_map=cm)
    finally
        ccall(_sym(:mio_agglomerate_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

const _REFINE_CLOSURES = Dict("" => Int32(0), "redgreen" => Int32(0), "red-green" => Int32(0),
                              "green" => Int32(0), "propagate" => Int32(1), "red" => Int32(1),
                              "balanced" => Int32(2), "2:1" => Int32(2))

const _REFINE_COMPARES = Dict("<" => Int32(0), "lt" => Int32(0), "<=" => Int32(1),
                              "le" => Int32(1), ">" => Int32(2), "gt" => Int32(2),
                              ">=" => Int32(3), "ge" => Int32(3), "==" => Int32(4),
                              "=" => Int32(4), "eq" => Int32(4), "!=" => Int32(5),
                              "ne" => Int32(5))

"""
    refine(mesh; levels=1, record_parent_ids=false, cells=nothing, region="",
           where_array="", where_op="<", where_value=0.0, closure="redgreen",
           record_levels=false, record_hierarchy=false)
        -> (; mesh, point_map, cell_maps)

Subdivide cells into same-type children (line → 2, triangle → 4, quad → 4,
tetra → 8, wedge → 8, hexahedron → 8), sharing the new edge/face/body nodes so
the result has no hanging nodes.

With no selector every cell is refined. Give at most **one** of `cells` (global,
block-major, **1-based** indices), `region` (a cell region selects its own cells;
a point region every cell with any node in it; a side region is an error) or
`where_array` + `where_op` + `where_value` (a threshold on a scalar `cell_data`
array), and only those cells are refined — the hanging nodes that leaves are
resolved by `closure`, so the output is still conforming. `"redgreen"` keeps the
extra refinement local; `"propagate"` is always available for every cell type but
converges to uniform refinement of the whole edge-connected component;
`"balanced"` keeps the hanging nodes and only enforces 2:1 balance, so the output
is **not conforming** but the cost is bounded by the selection (the constrained
nodes come back in `refine:hanging`).

Higher-order cells, pyramids and ragged blocks have no same-type subdivision
and fail by name. See `doc/refine.md`.

`record_hierarchy` attaches the `refine:cell_id`/`refine:parent_id` `cell_data`
arrays -- the persistent parent/child hierarchy a multigrid caller resolves
across the sequence of meshes it keeps ("a link between two meshes, not a tree
inside one"): an unsplit cell keeps its id and is its own parent; a split
cell's children each get a fresh id and carry the parent's id. An input
already carrying `refine:cell_id` is updated whatever this says. Also forces
`refine:entity` to be attached even when the closure leaves no hanging node,
since it already records the coarse corners each new fine node is the mean of
-- the multigrid prolongation weights, which `"redgreen"`/`"propagate"` would
otherwise never expose.
"""
function refine(m::Mesh; levels::Integer=1, record_parent_ids::Bool=false,
                cells=nothing, region::AbstractString="",
                where_array::AbstractString="", where_op::AbstractString="<",
                where_value::Real=0.0, closure::AbstractString="redgreen",
                record_levels::Bool=false, record_hierarchy::Bool=false)
    closure_code = get(_REFINE_CLOSURES, String(closure)) do
        throw(ArgumentError("refine: unknown closure '$closure' " *
                            "(expected 'redgreen'/'green', 'propagate' or 'balanced')"))
    end
    compare_code = get(_REFINE_COMPARES, String(where_op)) do
        throw(ArgumentError("refine: unknown comparison '$where_op' " *
                            "(expected '<', '<=', '>', '>=', '==' or '!=')"))
    end
    # 1-based on this side, 0-based across the ABI -- the binding's standing rule
    # for every index array.
    ids = cells === nothing ? Int64[] : Vector{Int64}(vec(collect(cells))) .- Int64(1)
    region_c = Vector{UInt8}(codeunits(String(region)))
    push!(region_c, 0x00)
    array_c = Vector{UInt8}(codeunits(String(where_array)))
    push!(array_c, 0x00)
    result = GC.@preserve ids region_c array_c begin
        opts = _CRefineOpts(isempty(ids) ? Ptr{Int64}(C_NULL) : pointer(ids),
                            Int64(length(ids)),
                            Cstring(pointer(region_c)), Cstring(pointer(array_c)),
                            Cdouble(where_value), Int32(levels),
                            record_parent_ids ? Int32(1) : Int32(0),
                            record_levels ? Int32(1) : Int32(0),
                            closure_code, compare_code, Int32(0),
                            record_hierarchy ? Int64(1) : Int64(0),
                            (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)))
        _check_ptr(ccall(_sym(:mio_refine_ex), Ptr{Cvoid},
                         (Ptr{Cvoid}, Ref{_CRefineOpts}), _handle(m), Ref(opts)))
    end
    try
        pm = _result_map(result, :mio_refine_result_point_map)
        cm = _result_cell_maps(result, :mio_refine_result_num_cell_maps,
                               :mio_refine_result_cell_map)
        out = Mesh(_check_ptr(ccall(_sym(:mio_refine_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, point_map=pm, cell_maps=cm)
    finally
        ccall(_sym(:mio_refine_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    decimate(mesh; ratio=-1.0, target_faces=-1, max_error=-1.0, placement="optimal",
             preserve_boundary=true, preserve_features=true, feature_angle=30.0)
        -> (; mesh, point_map, cell_maps, faces_removed, points_removed,
              collapses_rejected, max_error_applied)

Reduce a **surface** mesh's face count by greedy quadric-error-metric edge
collapse — the inverse of [`refine`](@ref). Exactly one of `ratio` (fraction of
faces to KEEP), `target_faces` and `max_error` must be set; a negative value
means "unset".

A 3-D volume mesh fails by name (extract the surface first), as do
higher-order, ragged and line/vertex blocks. The caller `frozen` mask is not
exposed across the C ABI. See `doc/decimate.md`.
"""
function decimate(m::Mesh; ratio::Real=-1.0, target_faces::Integer=-1,
                  max_error::Real=-1.0, placement::AbstractString="optimal",
                  preserve_boundary::Bool=true, preserve_features::Bool=true,
                  feature_angle::Real=30.0)
    result = _check_ptr(ccall(_sym(:mio_decimate), Ptr{Cvoid},
                              (Ptr{Cvoid}, Cdouble, Int64, Cdouble, Cstring, Cint, Cint,
                               Cdouble),
                              _handle(m), Float64(ratio), Int64(target_faces),
                              Float64(max_error), placement,
                              preserve_boundary ? Cint(1) : Cint(0),
                              preserve_features ? Cint(1) : Cint(0), Float64(feature_angle)))
    try
        pm = _result_map(result, :mio_decimate_result_point_map)
        cm = _result_cell_maps(result, :mio_decimate_result_num_cell_maps,
                               :mio_decimate_result_cell_map)
        fr = ccall(_sym(:mio_decimate_result_faces_removed), Int64, (Ptr{Cvoid},), result)
        pr = ccall(_sym(:mio_decimate_result_points_removed), Int64, (Ptr{Cvoid},), result)
        cr = ccall(_sym(:mio_decimate_result_collapses_rejected), Int64, (Ptr{Cvoid},), result)
        me = ccall(_sym(:mio_decimate_result_max_error_applied), Cdouble, (Ptr{Cvoid},), result)
        out = Mesh(_check_ptr(ccall(_sym(:mio_decimate_result_take_mesh), Ptr{Cvoid},
                                    (Ptr{Cvoid},), result)))
        (mesh=out, point_map=pm, cell_maps=cm, faces_removed=Int(fr),
         points_removed=Int(pr), collapses_rejected=Int(cr),
         max_error_applied=Float64(me))
    finally
        ccall(_sym(:mio_decimate_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    partition(mesh, nparts; method="auto", imbalance=0.03, mode="eco", seed=0,
              record_ids=false, ghost_layers=0, weights="")
        -> Vector{@NamedTuple{part_id::Int, mesh::Mesh, point_map::Vector{Int},
                              cell_maps::Vector{Vector{Int}}}}

Decompose a mesh into exactly `nparts` balanced pieces (the count-driven
complement to [`split`](@ref)). `method` is `"sfc"` (Hilbert curve, always
available), `"kahip"` (only in a build configured with KaHIP; requesting it
elsewhere fails by name) or `"auto"`.

Every piece keeps the input's cell-block structure 1:1, so the cell maps index
by input block and the pieces recombine into the input. `ghost_layers > 0`
instead grows each piece by that many shared-node BFS layers of other parts'
cells (a halo), tagged `partition:ghost` (0 = owned, L = reached at layer L),
in which case the pieces overlap. See `doc/partition.md`.
"""
function partition(m::Mesh, nparts::Integer; method::AbstractString="auto",
                   imbalance::Real=0.03, mode::AbstractString="eco", seed::Integer=0,
                   record_ids::Bool=false, ghost_layers::Integer=0,
                   weights::AbstractString="")
    result = _check_ptr(ccall(_sym(:mio_partition), Ptr{Cvoid},
                              (Ptr{Cvoid}, Cint, Cstring, Cdouble, Cstring, Cint, Cint,
                               Cint, Cstring),
                              _handle(m), Cint(nparts), method, Float64(imbalance), mode,
                              Cint(seed), record_ids ? Cint(1) : Cint(0),
                              Cint(ghost_layers), weights))
    try
        n = _check_count(ccall(_sym(:mio_partition_result_num_pieces), Int64,
                               (Ptr{Cvoid},), result), "mio_partition_result_num_pieces")
        nmaps = _check_count(ccall(_sym(:mio_partition_result_num_cell_maps), Int64,
                                   (Ptr{Cvoid},), result),
                             "mio_partition_result_num_cell_maps")
        out = @NamedTuple{part_id::Int, mesh::Mesh, point_map::Vector{Int},
                          cell_maps::Vector{Vector{Int}}}[]
        for i in 0:(n-1)
            pid = ccall(_sym(:mio_partition_result_part_id), Cint, (Ptr{Cvoid}, Int64),
                        result, i)
            data = Ref{Ptr{Cvoid}}(C_NULL); dt = Ref{Cint}(0); len = Ref{Int64}(0)
            _check(ccall(_sym(:mio_partition_result_point_map), Cint,
                         (Ptr{Cvoid}, Int64, Ptr{Ptr{Cvoid}}, Ptr{Cint}, Ptr{Int64}),
                         result, i, data, dt, len))
            pm = _shift_map(data[], len[])
            cms = Vector{Int}[]
            for b in 0:(nmaps-1)
                d2 = Ref{Ptr{Cvoid}}(C_NULL); t2 = Ref{Cint}(0); l2 = Ref{Int64}(0)
                _check(ccall(_sym(:mio_partition_result_cell_map), Cint,
                             (Ptr{Cvoid}, Int64, Int64, Ptr{Ptr{Cvoid}}, Ptr{Cint},
                              Ptr{Int64}), result, i, b, d2, t2, l2))
                push!(cms, _shift_map(d2[], l2[]))
            end
            piece = Mesh(_check_ptr(ccall(_sym(:mio_partition_result_take_mesh), Ptr{Cvoid},
                                          (Ptr{Cvoid}, Int64), result, i)))
            push!(out, (part_id=Int(pid), mesh=piece, point_map=pm, cell_maps=cms))
        end
        out
    finally
        ccall(_sym(:mio_partition_result_free), Cvoid, (Ptr{Cvoid},), result)
    end
end

"""
    partition_labels(mesh, nparts; method="auto", imbalance=0.03, mode="eco",
                     seed=0, weights="") -> Vector{Int}

The per-cell part assignment only, without building the pieces: one entry per
cell in **block-major global cell order** (block 1's cells first, then block
2's, …), with values in `0:nparts-1`.

These are part **ids**, not indices, so they are deliberately **not** shifted
to 1-based — the same rule the Fortran module follows.
"""
function partition_labels(m::Mesh, nparts::Integer; method::AbstractString="auto",
                          imbalance::Real=0.03, mode::AbstractString="eco",
                          seed::Integer=0, weights::AbstractString="")
    n = num_cells(m)
    labels = Vector{Int64}(undef, n)
    _check(GC.@preserve labels ccall(_sym(:mio_partition_labels), Cint,
                                     (Ptr{Cvoid}, Cint, Cstring, Cdouble, Cstring, Cint,
                                      Cstring, Ptr{Int64}, Int64),
                                     _handle(m), Cint(nparts), method, Float64(imbalance),
                                     mode, Cint(seed), weights, pointer(labels), Int64(n)))
    Int[Int(x) for x in labels]
end

# --- data operations ---------------------------------------------------------
#
# These act on point_data / cell_data / field_data and never touch geometry:
# points, connectivity and block order come through bit-identical.

const _LOCATIONS = Dict(:point => MIO_DATA_POINT, :cell => MIO_DATA_CELL,
                        :field => MIO_DATA_FIELD)

function _location(loc::Symbol)
    haskey(_LOCATIONS, loc) || throw(ArgumentError(
        "location must be :point, :cell or :field, got :$loc"))
    _LOCATIONS[loc]
end

"""
    data_drop(mesh, location, names; ignore_missing=false) -> Mesh

Drop the named data arrays at one location (`:point`, `:cell` or `:field`).

The C ABI exposes the three primitives (`drop` / `keep` / `rename`) but not the
combined `data_manage`; they compose to the same effect. See
`doc/data_manage.md`.
"""
function data_drop(m::Mesh, location::Symbol, names; ignore_missing::Bool=false)
    h = _handle(m); loc = _location(location)
    ptr = _with_names(names) do p, c
        ccall(_sym(:mio_data_drop), Ptr{Cvoid},
              (Ptr{Cvoid}, Cint, Ptr{Cstring}, Int64, Cint), h, loc, p, c,
              ignore_missing ? Cint(1) : Cint(0))
    end
    Mesh(_check_ptr(ptr))
end

"""
    data_keep(mesh, location, names; ignore_missing=false) -> Mesh

Keep only the named arrays at one location, dropping the rest **there**; the
other two locations are untouched. An empty `names` drops everything at
`location`.
"""
function data_keep(m::Mesh, location::Symbol, names; ignore_missing::Bool=false)
    h = _handle(m); loc = _location(location)
    ptr = _with_names(names) do p, c
        ccall(_sym(:mio_data_keep), Ptr{Cvoid},
              (Ptr{Cvoid}, Cint, Ptr{Cstring}, Int64, Cint), h, loc, p, c,
              ignore_missing ? Cint(1) : Cint(0))
    end
    Mesh(_check_ptr(ptr))
end

"""
    data_rename(mesh, location, from, to) -> Mesh

Rename one data array, preserving its values, dtype and shape.
"""
data_rename(m::Mesh, location::Symbol, from::AbstractString, to::AbstractString) =
    Mesh(_check_ptr(ccall(_sym(:mio_data_rename), Ptr{Cvoid},
                          (Ptr{Cvoid}, Cint, Cstring, Cstring),
                          _handle(m), _location(location), from, to)))

"""
    data_point_to_cell(mesh, names=String[]; suffix="") -> Mesh

Average point data onto the cells: each cell's value is the mean over its own
nodes. The output is always `Float64`. An empty `names` converts every
point-data array. See `doc/data_average.md`.
"""
function data_point_to_cell(m::Mesh, names=String[]; suffix::AbstractString="")
    h = _handle(m)
    ptr = _with_names(names) do p, c
        ccall(_sym(:mio_data_point_to_cell), Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cstring}, Int64, Cstring), h, p, c, suffix)
    end
    Mesh(_check_ptr(ptr))
end

"""
    data_cell_to_point(mesh, names=String[]; weight=:uniform, suffix="") -> Mesh

Average cell data onto the points: each point's value is the (optionally
measure-weighted) mean over the incident cells. `weight` is `:uniform` or
`:measure`. A point touched by no cell gets `NaN`; the output is always
`Float64`.
"""
function data_cell_to_point(m::Mesh, names=String[]; weight::Symbol=:uniform,
                            suffix::AbstractString="")
    w = weight === :uniform ? Cint(0) : weight === :measure ? Cint(1) :
        throw(ArgumentError("weight must be :uniform or :measure, got :$weight"))
    h = _handle(m)
    ptr = _with_names(names) do p, c
        ccall(_sym(:mio_data_cell_to_point), Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cstring}, Int64, Cint, Cstring), h, p, c, w, suffix)
    end
    Mesh(_check_ptr(ptr))
end

"""
    data_calc(mesh, expression, output_name; location=:point, overwrite=false) -> Mesh

Evaluate an elementwise expression over the arrays at `location` and store the
result there as `output_name`.

The grammar accepts `+ - * /`, unary minus, parentheses, numeric literals,
array names, and the functions `abs`, `sqrt`, `min`, `max` and `norm`. Nothing
else is evaluated — there is **no arbitrary-code path**. See `doc/data_calc.md`.
"""
data_calc(m::Mesh, expression::AbstractString, output_name::AbstractString;
          location::Symbol=:point, overwrite::Bool=false) =
    Mesh(_check_ptr(ccall(_sym(:mio_data_calc), Ptr{Cvoid},
                          (Ptr{Cvoid}, Cstring, Cint, Cstring, Cint),
                          _handle(m), expression, _location(location), output_name,
                          overwrite ? Cint(1) : Cint(0))))

"""
    data_condition(mesh, location, names=String[]; mode=:clamp, lo=0.0, hi=1.0,
                   scope=:component, nan_policy=:ignore, nan_replacement=0.0,
                   suffix="") -> Mesh

Condition the values of the selected arrays. `mode` is `:clamp`, `:normalize`
or `:standardize`; `scope` is `:component` (each trailing component
independently) or `:magnitude` (rescale whole rows, preserving direction);
`nan_policy` is `:ignore`, `:replace` or `:fail`.

For cell data the statistics are computed **jointly across all blocks**. An
empty `names` conditions every array at `location`. See
`doc/data_condition.md`.
"""
function data_condition(m::Mesh, location::Symbol, names=String[]; mode::Symbol=:clamp,
                        lo::Real=0.0, hi::Real=1.0, scope::Symbol=:component,
                        nan_policy::Symbol=:ignore, nan_replacement::Real=0.0,
                        suffix::AbstractString="")
    md = mode === :clamp ? Cint(0) : mode === :normalize ? Cint(1) :
         mode === :standardize ? Cint(2) :
         throw(ArgumentError("mode must be :clamp, :normalize or :standardize"))
    sc = scope === :component ? Cint(0) : scope === :magnitude ? Cint(1) :
         throw(ArgumentError("scope must be :component or :magnitude"))
    np = nan_policy === :ignore ? Cint(0) : nan_policy === :replace ? Cint(1) :
         nan_policy === :fail ? Cint(2) :
         throw(ArgumentError("nan_policy must be :ignore, :replace or :fail"))
    h = _handle(m); loc = _location(location)
    ptr = _with_names(names) do p, c
        ccall(_sym(:mio_data_condition), Ptr{Cvoid},
              (Ptr{Cvoid}, Cint, Ptr{Cstring}, Int64, Cint, Cdouble, Cdouble, Cint, Cint,
               Cdouble, Cstring),
              h, loc, p, c, md, Float64(lo), Float64(hi), sc, np,
              Float64(nan_replacement), suffix)
    end
    Mesh(_check_ptr(ptr))
end

"""
    data_info(mesh) -> Vector{NamedTuple}

Summarize every data array the mesh carries: location, name, dtype, shape,
counts, whole-array and per-component min/max/mean, and the NaN/inf/finite
counts. Read-only, and it **never throws on non-finite values** — it counts
them. See `doc/data_info.md`.
"""
function data_info(m::Mesh)
    handle = _check_ptr(ccall(_sym(:mio_data_info_create), Ptr{Cvoid}, (Ptr{Cvoid},),
                              _handle(m)))
    try
        n = _check_count(ccall(_sym(:mio_data_info_count), Int64, (Ptr{Cvoid},), handle),
                         "mio_data_info_count")
        out = []
        for i in 0:(n-1)
            name = _getstring() do buf, len
                ccall(_sym(:mio_data_info_name), Int64,
                      (Ptr{Cvoid}, Int64, Ptr{UInt8}, Int64), handle, i, buf, len)
            end
            entry = Ref{_CDataArrayInfo}()
            _check(ccall(_sym(:mio_data_info_entry), Cint,
                         (Ptr{Cvoid}, Int64, Ptr{_CDataArrayInfo}), handle, i, entry))
            e = entry[]
            comps = NamedTuple{(:min, :max, :mean),Tuple{Float64,Float64,Float64}}[]
            for c in 0:(e.num_components-1)
                lo = Ref{Cdouble}(0.0); hi = Ref{Cdouble}(0.0); mu = Ref{Cdouble}(0.0)
                _check(ccall(_sym(:mio_data_info_component), Cint,
                             (Ptr{Cvoid}, Int64, Int64, Ptr{Cdouble}, Ptr{Cdouble},
                              Ptr{Cdouble}), handle, i, c, lo, hi, mu))
                push!(comps, (min=Float64(lo[]), max=Float64(hi[]), mean=Float64(mu[])))
            end
            push!(out, (name=name, location=_LOCATION_SYMS[e.location+1],
                        dtype=_jltype(e.dtype), num_blocks=Int(e.num_blocks),
                        num_entries=Int(e.num_entries),
                        num_components=Int(e.num_components),
                        num_values=Int(e.num_values), min=e.min, max=e.max, mean=e.mean,
                        num_nan=Int(e.num_nan), num_inf=Int(e.num_inf),
                        num_finite=Int(e.num_finite),
                        inconsistent_blocks=e.inconsistent_blocks != 0,
                        components=comps))
        end
        out
    finally
        ccall(_sym(:mio_data_info_free), Cvoid, (Ptr{Cvoid},), handle)
    end
end

function _field_integral_components(handle, i::Int64, region, nc::Int64)
    comps = NamedTuple{(:total, :mean, :domain_measure, :num_nan),
                        Tuple{Float64,Float64,Float64,Int64}}[]
    for c in 0:(nc - 1)
        total = Ref{Cdouble}(0.0); mean = Ref{Cdouble}(0.0)
        dm = Ref{Cdouble}(0.0); nnan = Ref{Int64}(0)
        if region === nothing
            _check(ccall(_sym(:mio_data_integrate_component), Cint,
                         (Ptr{Cvoid}, Int64, Int64, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
                          Ptr{Int64}), handle, i, c, total, mean, dm, nnan))
        else
            _check(ccall(_sym(:mio_data_integrate_region_component), Cint,
                         (Ptr{Cvoid}, Int64, Int64, Int64, Ptr{Cdouble}, Ptr{Cdouble},
                          Ptr{Cdouble}, Ptr{Int64}), handle, i, region, c, total, mean, dm, nnan))
        end
        push!(comps, (total=total[], mean=mean[], domain_measure=dm[], num_nan=Int(nnan[])))
    end
    comps
end

"""
    data_integrate(mesh, names=String[]) -> Vector{NamedTuple}

Cell-measure-weighted total/mean of one or more `cell_data` arrays --
`gradient`'s integration counterpart (`gradient` differentiates a field, this
integrates one). Every sum is weighted by the cell's own length/area/volume,
excluding both a cell with unmeasurable geometry and a component with a
non-finite value from that component's numerator *and* denominator. `names`
empty means every `cell_data` array. Reported for the whole mesh (`domain`)
and independently for every named Cell region (`regions`) -- a cell in two
regions contributes fully to both, one in none contributes to neither. A
`point_data`-only name throws naming `data_point_to_cell` as the fix. See
`doc/field_integration.md`.
"""
function data_integrate(m::Mesh, names=String[])
    h = _handle(m)
    handle = _with_names(names) do p, c
        _check_ptr(ccall(_sym(:mio_data_integrate_create), Ptr{Cvoid},
                          (Ptr{Cvoid}, Ptr{Cstring}, Int64), h, p, c))
    end
    try
        n = _check_count(ccall(_sym(:mio_data_integrate_count), Int64, (Ptr{Cvoid},), handle),
                         "mio_data_integrate_count")
        out = []
        for i in 0:(n - 1)
            name = _getstring() do buf, len
                ccall(_sym(:mio_data_integrate_name), Int64,
                      (Ptr{Cvoid}, Int64, Ptr{UInt8}, Int64), handle, i, buf, len)
            end
            entry = Ref{_CFieldIntegralInfo}()
            _check(ccall(_sym(:mio_data_integrate_entry), Cint,
                         (Ptr{Cvoid}, Int64, Ptr{_CFieldIntegralInfo}), handle, i, entry))
            e = entry[]
            domain = (num_cells=Int(e.num_cells), num_skipped=Int(e.num_skipped),
                      components=_field_integral_components(handle, i, nothing, e.num_components))

            nregions = _check_count(ccall(_sym(:mio_data_integrate_region_count), Int64,
                                          (Ptr{Cvoid}, Int64), handle, i),
                                    "mio_data_integrate_region_count")
            regions = []
            for r in 0:(nregions - 1)
                rname = _getstring() do buf, len
                    ccall(_sym(:mio_data_integrate_region_name), Int64,
                          (Ptr{Cvoid}, Int64, Int64, Ptr{UInt8}, Int64), handle, i, r, buf, len)
                end
                rentry = Ref{_CFieldIntegralInfo}()
                _check(ccall(_sym(:mio_data_integrate_region_entry), Cint,
                             (Ptr{Cvoid}, Int64, Int64, Ptr{_CFieldIntegralInfo}), handle, i, r,
                             rentry))
                re = rentry[]
                push!(regions,
                      (name=rname, num_cells=Int(re.num_cells), num_skipped=Int(re.num_skipped),
                       components=_field_integral_components(handle, i, r, re.num_components)))
            end
            push!(out, (name=name, num_components=Int(e.num_components), domain=domain,
                        regions=regions))
        end
        out
    finally
        ccall(_sym(:mio_data_integrate_free), Cvoid, (Ptr{Cvoid},), handle)
    end
end

const _LOCATION_SYMS = (:point, :cell, :field)

# ---------------------------------------------------------------------------
# Settings pipeline (v9.11.0): run a whole settings.json (read -> operation
# chain -> write; PascalCase vocabulary, see doc/pipeline.md). The flat ABI
# carries JSON text only.
# ---------------------------------------------------------------------------

"""
    run_pipeline_file(settings_path)

Run the settings pipeline stored in the file at `settings_path` (a
`settings.json`; PascalCase ops/keys, see `doc/pipeline.md`). Needs a build
with the JSON parser (`-DMESHIOPLUSPLUS_WITH_JSON=ON`); otherwise the raised
error names the flag — [`pipeline_has_json`](@ref) reports which build this is.
"""
function run_pipeline_file(settings_path::AbstractString)
    _check(ccall(_sym(:mio_pipeline_run_file), Cint, (Cstring,), settings_path))
    nothing
end

"""
    run_pipeline_json(json_text)

Run a settings pipeline given as JSON text (see [`run_pipeline_file`](@ref)).
"""
function run_pipeline_json(json_text::AbstractString)
    _check(ccall(_sym(:mio_pipeline_run_json), Cint, (Cstring,), json_text))
    nothing
end

"""
    pipeline_has_json() -> Bool

Whether the loaded library carries the JSON pipeline parser.
"""
pipeline_has_json() = ccall(_sym(:mio_pipeline_has_json), Cint, ()) != 0

# --- regular grids and signed distance ---------------------------------------

const _SDF_SIGNS = Dict("unsigned" => Int32(0), "pseudonormal" => Int32(1),
                        "winding-number" => Int32(2), "winding_number" => Int32(2))
const _SDF_WEIGHTS = Dict("angle" => Int32(0), "area" => Int32(1))
const _SDF_LOCATIONS = Dict("corner" => Int32(0), "point" => Int32(0),
                            "center" => Int32(1), "centre" => Int32(1), "cell" => Int32(1))
const _SDF_CHECKS = Dict("off" => Int32(0), "warn" => Int32(1), "error" => Int32(2))
const _VOXEL_FILLS = Dict("all" => Int32(0), "surface" => Int32(1), "inside" => Int32(2))

function _code(table, value, what)
    key = replace(String(value), '_' => '-')
    haskey(table, key) && return table[key]
    haskey(table, String(value)) && return table[String(value)]
    throw(ArgumentError("meshio++: unknown $what '$(value)'"))
end

"""
    grid(dims; origin=(0,0,0), spacing=(1,1,1), max_cells=20_000_000) -> Mesh

Build a regular hexahedron lattice from nothing -- the only constructor here
that takes no input mesh. Points run x fastest, then y, then z.

A zero cell count on any axis gives an empty mesh, which is a legal request
rather than an error.
"""
function grid(dims; origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),
              max_cells::Integer=20_000_000)
    d = Int64[Int64(v) for v in dims]
    length(d) == 3 || throw(ArgumentError("dims must have three entries"))
    o = Cdouble[Cdouble(v) for v in origin]
    sp = Cdouble[Cdouble(v) for v in spacing]
    ptr = GC.@preserve d o sp ccall(_sym(:mio_grid), Ptr{Cvoid},
                                    (Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble}, Int64),
                                    pointer(d), pointer(o), pointer(sp), Int64(max_cells))
    Mesh(_check_ptr(ptr))
end

"""
    voxelize(m; resolution=nothing, cell_size=nothing, bounds=nothing, padding=0.0,
             padding_relative=0.0, fill=:all, sign=:pseudonormal,
             watertight_check=:warn, attach_occupancy=false,
             max_cells=20_000_000) -> NamedTuple

Build a regular grid around `m`. Exactly one of `resolution` and `cell_size`
must be given. `fill` is `:all`, `:surface` or `:inside`.

Returns `(; mesh, dims, origin, spacing, num_occupied)`.
"""
function voxelize(m::Mesh; resolution=nothing, cell_size=nothing, bounds=nothing,
                  padding::Real=0.0, padding_relative::Real=0.0, fill=:all,
                  sign=:pseudonormal, watertight_check=:warn,
                  attach_occupancy::Bool=false, max_cells::Integer=20_000_000)
    res = resolution === nothing ? Int64[] : Int64[Int64(v) for v in resolution]
    (resolution === nothing || length(res) == 3) ||
        throw(ArgumentError("resolution must have three entries"))
    bnd = bounds === nothing ? Cdouble[] : Cdouble[Cdouble(v) for v in bounds]
    (bounds === nothing || length(bnd) == 6) ||
        throw(ArgumentError("bounds must have six entries"))

    dims = Vector{Int64}(undef, 3)
    origin = Vector{Cdouble}(undef, 3)
    spacing = Vector{Cdouble}(undef, 3)
    occupied = Ref{Int64}(0)
    ptr = GC.@preserve res bnd dims origin spacing begin
        opts = _CVoxelOpts(isempty(res) ? Ptr{Int64}(C_NULL) : pointer(res),
                           isempty(bnd) ? Ptr{Cdouble}(C_NULL) : pointer(bnd),
                           Cdouble(cell_size === nothing ? 0.0 : cell_size),
                           Cdouble(padding), Cdouble(padding_relative),
                           Int64(max_cells), _code(_VOXEL_FILLS, fill, "fill"),
                           attach_occupancy ? Int32(1) : Int32(0),
                           _code(_SDF_SIGNS, sign, "sign"),
                           _code(_SDF_CHECKS, watertight_check, "watertight check"),
                           (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)))
        ccall(_sym(:mio_voxelize), Ptr{Cvoid},
              (Ptr{Cvoid}, Ref{_CVoxelOpts}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble},
               Ptr{Int64}),
              _handle(m), Ref(opts), pointer(dims), pointer(origin), pointer(spacing),
              occupied)
    end
    (mesh=Mesh(_check_ptr(ptr)), dims=Tuple(Int.(dims)), origin=Tuple(Float64.(origin)),
     spacing=Tuple(Float64.(spacing)), num_occupied=Int(occupied[]))
end

const _CURVATURE_DUAL_AREAS = Dict(:mixed_voronoi => Int32(0), :barycentric => Int32(1))

"""
    compute_curvature(m; mean=true, gaussian=true, dual_area=:mixed_voronoi,
                      include_boundary=false, record_area=false,
                      record_principal=false, region="")
        -> (; mesh, quality, num_boundary, num_isolated, num_degenerate,
             total_angle_defect)

Per-vertex mean (`H`) and Gaussian (`K`) curvature of a surface mesh, by the
angle defect for `K` and the cotangent Laplace-Beltrami operator for `H` — the
signed distance's natural companion as a node feature.

Writes `curvature:mean` and `curvature:gaussian` as point data, optionally
`curvature:area` (the dual area each was divided by) and `curvature:principal`
(`(n, 2)`, `k1 >= k2`). Geometry, connectivity and existing data are carried
through unchanged.

`total_angle_defect` is the oracle: on a CLOSED surface it is `2*pi*chi`
exactly — `4*pi` for anything sphere-like — whatever the tessellation and
whichever `dual_area`, so a value that is not that means the input is not
closed or the result is not sane.

`H` is orientation-dependent and `K` is not, so check
`quality.inconsistent_pairs` before trusting a sign: a nonzero count means
facets disagree about which side is out. This never repairs its input.
"""
function compute_curvature(m::Mesh; mean::Bool=true, gaussian::Bool=true,
                           dual_area::Symbol=:mixed_voronoi,
                           include_boundary::Bool=false, record_area::Bool=false,
                           record_principal::Bool=false, region::AbstractString="")
    haskey(_CURVATURE_DUAL_AREAS, dual_area) ||
        throw(ArgumentError("meshio++: curvature: unknown dual area '$(dual_area)' " *
                            "(expected :mixed_voronoi or :barycentric)"))
    report = Ref{_CCurvatureReport}()
    region_c = Vector{UInt8}(codeunits(String(region) * "\0"))
    ptr = GC.@preserve region_c begin
        opts = _CCurvatureOpts(Cstring(pointer(region_c)),
                               mean ? Int32(1) : Int32(0),
                               gaussian ? Int32(1) : Int32(0),
                               _CURVATURE_DUAL_AREAS[dual_area],
                               include_boundary ? Int32(1) : Int32(0),
                               record_area ? Int32(1) : Int32(0),
                               record_principal ? Int32(1) : Int32(0),
                               (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)))
        ccall(_sym(:mio_compute_curvature), Ptr{Cvoid},
              (Ptr{Cvoid}, Ref{_CCurvatureOpts}, Ptr{_CCurvatureReport}),
              _handle(m), Ref(opts), report)
    end
    r = _check_ptr(ptr)
    rep = report[]
    q = rep.quality
    (mesh=Mesh(r),
     quality=(boundary_edges=Int(q.boundary_edges),
              non_manifold_edges=Int(q.non_manifold_edges),
              inconsistent_pairs=Int(q.inconsistent_pairs),
              degenerate_triangles=Int(q.degenerate_triangles),
              watertight=q.watertight != 0),
     num_boundary=Int(rep.num_boundary), num_isolated=Int(rep.num_isolated),
     num_degenerate=Int(rep.num_degenerate),
     total_angle_defect=Float64(rep.total_angle_defect))
end

"""
    surface_watertight_check(m) -> NamedTuple

What is wrong with a surface, in numbers: boundary edges, non-manifold edges,
inconsistently wound pairs and degenerate triangles. A signed distance is only
meaningful where these are zero.
"""
function surface_watertight_check(m::Mesh)
    out = Ref{_CSurfaceQuality}()
    _check(ccall(_sym(:mio_surface_watertight_check), Cint,
                 (Ptr{Cvoid}, Ptr{_CSurfaceQuality}), _handle(m), out))
    q = out[]
    (boundary_edges=Int(q.boundary_edges), non_manifold_edges=Int(q.non_manifold_edges),
     inconsistent_pairs=Int(q.inconsistent_pairs),
     degenerate_triangles=Int(q.degenerate_triangles), watertight=q.watertight != 0)
end

function _sdf_opts(; sign, weight, location, band, record_closest_cell, record_inside,
                   watertight_check, surface_region, grid_cell_size, max_winding_work,
                   region_buf)
    _CSdfOpts(Cstring(pointer(region_buf)), Cdouble(band), Cdouble(grid_cell_size),
              Cdouble(max_winding_work), _code(_SDF_SIGNS, sign, "sign"),
              _code(_SDF_WEIGHTS, weight, "weight"),
              _code(_SDF_LOCATIONS, location, "location"),
              _code(_SDF_CHECKS, watertight_check, "watertight check"),
              record_closest_cell ? Int32(1) : Int32(0),
              record_inside ? Int32(1) : Int32(0),
              (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)))
end

"""
    sample_distance(surface, points; sign=:pseudonormal, band=0.0, ...) -> Vector{Float64}

Signed distances from `points` -- a `3 x n` matrix, the column-major memory the
C ABI reads as `n` rows of three -- to `surface`. Negative is inside.
"""
function sample_distance(surface::Mesh, points::AbstractMatrix{<:Real};
                         sign=:pseudonormal, weight=:angle, band::Real=0.0,
                         watertight_check=:warn, surface_region::AbstractString="",
                         grid_cell_size::Real=0.0, max_winding_work::Real=2.0e9)
    size(points, 1) == 3 || throw(ArgumentError("points must be a 3 x n matrix"))
    n = size(points, 2)
    pts = Matrix{Cdouble}(points)
    out = Vector{Cdouble}(undef, max(n, 1))
    region_buf = Vector{UInt8}(codeunits(String(surface_region)))
    push!(region_buf, 0x00)
    GC.@preserve pts out region_buf begin
        opts = _sdf_opts(; sign, weight, location=:corner, band,
                         record_closest_cell=false, record_inside=false, watertight_check,
                         surface_region, grid_cell_size, max_winding_work, region_buf)
        _check(ccall(_sym(:mio_sample_distance), Cint,
                     (Ptr{Cvoid}, Ptr{Cdouble}, Int64, Ref{_CSdfOpts}, Ptr{Cdouble}),
                     _handle(surface), pointer(pts), Int64(n), Ref(opts), pointer(out)))
    end
    Float64[out[i] for i in 1:n]
end

"""
    distance_to_surface(query, surface; sign=:pseudonormal, location=:corner, ...) -> NamedTuple

Attach the signed distance from `query`'s points (or cell centres) to `surface`
as the `sdf:distance` array. Returns `(; mesh, num_banded, quality)`.
"""
function distance_to_surface(query::Mesh, surface::Mesh; sign=:pseudonormal, weight=:angle,
                             location=:corner, band::Real=0.0,
                             record_closest_cell::Bool=false, record_inside::Bool=false,
                             watertight_check=:warn, surface_region::AbstractString="",
                             grid_cell_size::Real=0.0, max_winding_work::Real=2.0e9)
    banded = Ref{Int64}(0)
    quality = Ref{_CSurfaceQuality}()
    region_buf = Vector{UInt8}(codeunits(String(surface_region)))
    push!(region_buf, 0x00)
    ptr = GC.@preserve region_buf begin
        opts = _sdf_opts(; sign, weight, location, band, record_closest_cell, record_inside,
                         watertight_check, surface_region, grid_cell_size, max_winding_work,
                         region_buf)
        ccall(_sym(:mio_distance_to_surface), Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cvoid}, Ref{_CSdfOpts}, Ptr{Int64}, Ptr{_CSurfaceQuality}),
              _handle(query), _handle(surface), Ref(opts), banded, quality)
    end
    q = quality[]
    (mesh=Mesh(_check_ptr(ptr)), num_banded=Int(banded[]),
     quality=(boundary_edges=Int(q.boundary_edges),
              non_manifold_edges=Int(q.non_manifold_edges),
              inconsistent_pairs=Int(q.inconsistent_pairs),
              degenerate_triangles=Int(q.degenerate_triangles),
              watertight=q.watertight != 0))
end

const _SDF_STRUCTURES = Dict("voxel" => Int32(0), "octree" => Int32(1))

"""
    compute_sdf(surface; structure=:voxel, resolution=nothing, cell_size=nothing,
                bounds=nothing, padding=0.0, padding_relative=0.1,
                root_resolution=8, max_depth=4, band_cells=1.0, record_levels=true,
                max_cells=20_000_000, sign=:pseudonormal, ...) -> NamedTuple

Generate a grid over `surface` and fill it with signed distances -- the one call
that turns a surface into a field.

`structure` is `:voxel` (a dense lattice) or `:octree` (refined near the
surface, and therefore **1-irregular**: it has hanging nodes; see
[`refine`](@ref)'s `closure=:balanced`). `resolution`/`cell_size` size a voxel
grid and are an **error** with `:octree`, whose finest cell is
`root_resolution / 2^max_depth` and is therefore already determined.

Returns `(; mesh, dims, origin, spacing, max_depth, num_banded, quality)`, where
`dims` are the ROOT cell counts and `spacing` the FINEST cell size. The mesh
also carries the numeric `sdf:*` `field_data` header; no file format persists
arbitrary `field_data`, so write it as `.vti`, whose Origin/Spacing/WholeExtent
attributes are the same information.
"""
function compute_sdf(surface::Mesh; structure=:voxel, resolution=nothing,
                     cell_size=nothing, bounds=nothing, padding::Real=0.0,
                     padding_relative::Real=0.1, root_resolution::Integer=8,
                     max_depth::Integer=4, band_cells::Real=1.0,
                     record_levels::Bool=true, max_cells::Integer=20_000_000,
                     sign=:pseudonormal, weight=:angle, location=:corner, band::Real=0.0,
                     record_closest_cell::Bool=false, record_inside::Bool=false,
                     watertight_check=:warn, surface_region::AbstractString="",
                     grid_cell_size::Real=0.0, max_winding_work::Real=2.0e9)
    res = resolution === nothing ? Int64[] : Int64[Int64(v) for v in resolution]
    (resolution === nothing || length(res) == 3) ||
        throw(ArgumentError("resolution must have three entries"))
    bnd = bounds === nothing ? Cdouble[] : Cdouble[Cdouble(v) for v in bounds]
    (bounds === nothing || length(bnd) == 6) ||
        throw(ArgumentError("bounds must have six entries"))

    dims = Vector{Int64}(undef, 3)
    origin = Vector{Cdouble}(undef, 3)
    spacing = Vector{Cdouble}(undef, 3)
    depth = Ref{Int64}(0)
    banded = Ref{Int64}(0)
    quality = Ref{_CSurfaceQuality}()
    region_buf = Vector{UInt8}(codeunits(String(surface_region)))
    push!(region_buf, 0x00)
    ptr = GC.@preserve res bnd dims origin spacing region_buf begin
        inner = _sdf_opts(; sign, weight, location, band, record_closest_cell,
                          record_inside, watertight_check, surface_region, grid_cell_size,
                          max_winding_work, region_buf)
        opts = _CComputeSdfOpts(isempty(res) ? Ptr{Int64}(C_NULL) : pointer(res),
                                isempty(bnd) ? Ptr{Cdouble}(C_NULL) : pointer(bnd),
                                Cdouble(cell_size === nothing ? 0.0 : cell_size),
                                Cdouble(padding), Cdouble(padding_relative),
                                Cdouble(band_cells), Int64(max_cells),
                                Int64(root_resolution), Int64(max_depth),
                                _code(_SDF_STRUCTURES, structure, "structure"),
                                record_levels ? Int32(1) : Int32(0),
                                (Int64(0), Int64(0), Int64(0), Int64(0), Int64(0), Int64(0)),
                                inner)
        ccall(_sym(:mio_compute_sdf), Ptr{Cvoid},
              (Ptr{Cvoid}, Ref{_CComputeSdfOpts}, Ptr{Int64}, Ptr{Cdouble}, Ptr{Cdouble},
               Ptr{Int64}, Ptr{Int64}, Ptr{_CSurfaceQuality}),
              _handle(surface), Ref(opts), pointer(dims), pointer(origin),
              pointer(spacing), depth, banded, quality)
    end
    q = quality[]
    (mesh=Mesh(_check_ptr(ptr)), dims=Tuple(Int.(dims)), origin=Tuple(Float64.(origin)),
     spacing=Tuple(Float64.(spacing)), max_depth=Int(depth[]), num_banded=Int(banded[]),
     quality=(boundary_edges=Int(q.boundary_edges),
              non_manifold_edges=Int(q.non_manifold_edges),
              inconsistent_pairs=Int(q.inconsistent_pairs),
              degenerate_triangles=Int(q.degenerate_triangles),
              watertight=q.watertight != 0))
end
