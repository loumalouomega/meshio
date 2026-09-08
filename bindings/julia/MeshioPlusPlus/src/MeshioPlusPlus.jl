"""
    MeshioPlusPlus

Julia bindings for [meshio++](https://github.com/loumalouomega/meshioplusplus):
read and write ~40 mesh formats used in scientific computing and FEM, and run
meshio++'s operations layer on the meshes.

This package binds the **installed C API** (`libmeshioplusplus`), exactly as
the Fortran module does. Build and install it first:

```sh
./build/configure.sh --c-api --build
cmake --install build/cpp-release --prefix /your/prefix
```

then either put `/your/prefix/lib` on the loader path or set
`ENV["MESHIOPLUSPLUS_LIB"]` to the shared library before `using`.

# Two conventions

**Column-major.** Julia is column-major and the C core row-major, so points
shaped `(dim, num_points)` and connectivity `(nodes_per_cell, num_cells)` are
the *same memory* as the C API's `(num_points, dim)` and
`(num_cells, nodes_per_cell)`. Nothing is transposed anywhere.

**1-based.** Connectivity is 0-based at the ABI. The ±1 shift happens inside
the **copying** accessors only — [`connectivity`](@ref) is a 1-based copy,
[`connectivity_ptr`](@ref) the un-shifted zero-copy borrow. A borrow cannot be
shifted without copying it, which is the whole point of a borrow.

# Names that would shadow `Base`

`read`, `write`, `convert`, `merge`, `split` and `diff` are **not exported**.
Call them qualified:

```julia
import MeshioPlusPlus as mio
m = mio.read("bracket.msh")
mio.write(m, "bracket.vtu")
```

# Licence

This binding is **not** MIT like the rest of meshio++: it is released under
the GNU General Public License, version 3 (GPL-3.0), a copyleft license.
Anyone may use, modify or sell it commercially with no permission needed;
distributing it or a modified version of it must be under GPL-3.0 too, with
source available. Purely private use carries no obligation. See
`bindings/julia/LICENSE`.
"""
module MeshioPlusPlus

using Libdl

include("libmio.jl")
include("errors.jl")
include("mesh.jl")
include("regions.jl")
include("operations.jl")
include("sequence.jl")
include("xdmf_series.jl")
include("provenance.jl")

function __init__()
    _load_library()
    _check_abi_layout()
    return
end

# Handles, errors, borrows
export Mesh, MeshBorrow, MeshioError, BorrowError, Region
export ReadOptions, MeshMetadata, DiffReport

# Introspection
export version, mesh_backend, last_error, library_path
export provenance, provenance_note, provenance_source, provenance_target
export PROVENANCE_OFF, PROVENANCE_BEST_EFFORT, PROVENANCE_REQUIRED
export format_readable, format_writable, sniff_format
export cell_type_num_nodes, cell_type_dimension, reader_supports_options, read_metadata

# Inspecting
export num_points, point_dim, num_cells, num_cell_blocks
export cell_block_info, cell_block_type, cell_block_types
export points, points_ptr, connectivity, connectivity_ptr
export polygon_block, polyhedron_block
export num_point_data, num_cell_data, num_field_data
export point_data_names, cell_data_names, field_data_names
export point_data, point_data_ptr, cell_data, cell_data_ptr, field_data, field_data_ptr
export cell_data_num_blocks

# Building
export set_points!, add_cell_block!, add_point_data!, append_cell_data!, add_field_data!
export add_polygon_block!, add_polyhedron_block!
export regions, add_region!

# Operations
export extract_surface, extract_skin, attach_quality, quality_counts
export transform, clean, smooth, optimize_volume, crop_bbox, crop_plane, crop_predicate, slice, isosurface
export gradient
export hessian
export estimate_error
export remesh
export remesh_volume
export grid, voxelize
export sample_distance, distance_to_surface, surface_watertight_check, compute_sdf
export compute_curvature
export interpolate, conservative_interpolate, meshes_equal, stats, compute_bandwidth
export reorder, convert_cells, subdivide, agglomerate, refine, undo_green, decimate, partition, partition_labels
export data_drop, data_keep, data_rename, data_point_to_cell, data_cell_to_point
export data_calc, data_condition, data_info, data_integrate
export run_pipeline_file, run_pipeline_json, pipeline_has_json
# Sequences (multi-file / transient datasets). `read`/`step`/`time`/`path` and
# friends would shadow Base, so only the non-colliding names are exported;
# reach the rest as MeshioPlusPlus.step(seq, i) etc.
export Sequence, read_step, to_timeseries, timeseries_to_sequence
export run_sequence_file, run_sequence_json

# Transient (time-series) XDMF writing
export XdmfSeries, write_points_cells!, write_data!, flush!, finalize!, finalized,
    num_steps

end # module
