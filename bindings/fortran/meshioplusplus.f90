!  ██████   ██████ ██████████  █████████  █████   █████ █████    ███████
! ░░██████ ██████ ░░███░░░░░█ ███░░░░░███░░███   ░░███ ░░███   ███░░░░░███      ███         ███
!  ░███░█████░███  ░███  █ ░ ░███    ░░░  ░███    ░███  ░███  ███     ░░███    ░███        ░███
!  ░███░░███ ░███  ░██████   ░░█████████  ░███████████  ░███ ░███      ░███ ███████████ ███████████
!  ░███ ░░░  ░███  ░███░░█    ░░░░░░░░███ ░███░░░░░███  ░███ ░███      ░███░░░░░███░░░ ░░░░░███░░░
!  ░███      ░███  ░███ ░   █ ███    ░███ ░███    ░███  ░███ ░░███     ███     ░███        ░███
!  █████     █████ ██████████░░█████████  █████   █████ █████ ░░░███████░      ░░░         ░░░
! ░░░░░     ░░░░░ ░░░░░░░░░░  ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░    ░░░░░░░
!
!
!  License:         MIT License
!                   meshio++ default license: LICENSE
!
!  Main authors:    Vicente Mataix Ferrandiz
!
!
! The meshio++ Fortran interface: a modern OO Fortran 2008 module layered on
! the C API (bindings/c/include/meshioplusplus/meshioplusplus.h) via
! ISO_C_BINDING, in the HDF5/PETSc style:
!
!     use meshioplusplus
!     type(mio_mesh) :: m
!     call m%read("bracket.msh")
!     print *, m%num_points()
!     call m%write("bracket.vtu")
!     call m%free()
!
! Conventions (differences from the C API):
!  - Arrays are Fortran-shaped: points are `points(dim, num_points)` and
!    connectivity `conn(nodes_per_cell, num_cells)`. Because Fortran is
!    column-major and the C core row-major, this is the SAME memory -- no
!    transpose happens anywhere in this module.
!  - Connectivity is 1-based here; the +-1 shift happens inside the copying
!    setters/getters (where a copy is made anyway). Zero-copy borrows
!    (points_ptr) carry no indices, so nothing zero-copy ever needs shifting.
!  - All indices (cell blocks, data names) are 1-based.
!  - Every fallible procedure takes `optional` `stat` (integer, 0 = success)
!    and `errmsg` (deferred-length character) arguments. If `stat` is absent
!    and the call fails, the message is printed and the program error stops
!    -- pass `stat` to handle errors yourself (the stdlib pattern).
!  - Handles are freed explicitly with `call m%free()` (no finalizer).
!
! Compiled .mod files are compiler-(major-version-)specific; this source file
! is installed next to the .mod so consumers on a different compiler can
! simply recompile the module (the HDF5 approach).
module meshioplusplus
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64, error_unit
    implicit none
    private

    public :: mio_mesh
    public :: mio_xdmf_series
    public :: mio_stats_report
    public :: mio_surface_quality
    public :: mio_grid
    public :: mio_data_array_info
    public :: mio_field_integral_info
    public :: mio_convert, mio_version, mio_mesh_backend, mio_error_message
    public :: mio_pipeline_run_file, mio_pipeline_run_json, mio_pipeline_has_json
    ! `mio_sequence`'s fan-in is the type-bound `%to_timeseries`; only the
    ! fan-out (which starts from a path, not a handle) is module-level.
    public :: mio_sequence, mio_timeseries_to_sequence
    public :: mio_sequence_pipeline_run_file, mio_sequence_pipeline_run_json
    public :: mio_format_readable, mio_format_writable
    public :: mio_sniff_format
    public :: mio_read_metadata, mio_metadata, mio_cell_block_info
    public :: mio_merge
    public :: mio_provenance_begin, mio_provenance_end
    public :: mio_provenance_note, mio_provenance_source, mio_provenance_target
    public :: MIO_PROVENANCE_OFF, MIO_PROVENANCE_BEST_EFFORT, MIO_PROVENANCE_REQUIRED
    public :: mio_interpolate
    public :: mio_conservative_interpolate
    public :: mio_undo_green
    public :: mio_shrinkwrap
    ! Length of the fixed-width string buffers the `keys` out-arguments of
    ! `split` and `data_info` use; consumers need it to declare those arrays.
    public :: STRBUF_LEN
    ! Data-operation enumerations (must match the C enums in meshioplusplus.h).
    public :: MIO_DATA_POINT, MIO_DATA_CELL, MIO_DATA_FIELD
    public :: MIO_WEIGHT_UNIFORM, MIO_WEIGHT_MEASURE
    public :: MIO_COND_CLAMP, MIO_COND_NORMALIZE, MIO_COND_STANDARDIZE
    public :: MIO_SCOPE_COMPONENT, MIO_SCOPE_MAGNITUDE
    public :: MIO_NAN_IGNORE, MIO_NAN_REPLACE, MIO_NAN_FAIL
    ! Named regions (see doc/regions.md).
    public :: MIO_REGION_POINT, MIO_REGION_CELL, MIO_REGION_SIDE
    public :: mio_region_info

    ! What is wrong with a surface (bind(c); layout must match
    ! mio_surface_quality in meshioplusplus.h). The four counts are separate
    ! because they need different fixes: "12 boundary edges" is actionable,
    ! "not watertight" is not.
    type, bind(c) :: mio_surface_quality
        integer(c_int64_t) :: boundary_edges = 0
        integer(c_int64_t) :: non_manifold_edges = 0
        integer(c_int64_t) :: inconsistent_pairs = 0
        integer(c_int64_t) :: degenerate_triangles = 0
        integer(c_int32_t) :: watertight = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_curvature_opts`. Field order and types are ABI
    !> and must match bindings/c/include/meshioplusplus/meshioplusplus.h
    !> exactly; `reserved` is padding for additive growth and must stay zero.
    !> The defaults here mirror `mio_curvature_opts_init`, which is called
    !> anyway -- an all-zero struct is NOT the default, since `mean` and
    !> `gaussian` are on.
    type, bind(c) :: mio_curvature_opts_t
        type(c_ptr) :: region = c_null_ptr
        integer(c_int32_t) :: mean = 1
        integer(c_int32_t) :: gaussian = 1
        integer(c_int32_t) :: dual_area = 0
        integer(c_int32_t) :: include_boundary = 0
        integer(c_int32_t) :: record_area = 0
        integer(c_int32_t) :: record_principal = 0
        integer(c_int64_t) :: reserved(6) = 0
    end type

    !> Interop mirror of C `mio_curvature_report`. Field order/types are ABI.
    type, bind(c) :: mio_curvature_report_t
        type(mio_surface_quality) :: quality
        integer(c_int64_t) :: num_boundary = 0
        integer(c_int64_t) :: num_isolated = 0
        integer(c_int64_t) :: num_degenerate = 0
        real(c_double) :: total_angle_defect = 0.0_c_double
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_repair_opts`. Field order/types are ABI; the
    !> defaults mirror `mio_repair_opts_init` (every pass on), which is called
    !> anyway -- an all-zero struct is NOT the default.
    type, bind(c) :: mio_repair_opts_t
        integer(c_int32_t) :: fix_orientation = 1
        integer(c_int32_t) :: orient_outward = 1
        integer(c_int32_t) :: fill_holes = 1
        integer(c_int32_t) :: split_non_manifold = 1
        integer(c_int32_t) :: record_provenance = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: max_hole_edges = 10
        real(c_double) :: weld_tolerance = 0.0_c_double
        integer(c_int64_t) :: reserved(5) = 0
    end type

    !> Interop mirror of C `mio_repair_report`. Field order/types are ABI.
    type, bind(c) :: mio_repair_report_t
        type(mio_surface_quality) :: quality_before
        type(mio_surface_quality) :: quality_after
        integer(c_int64_t) :: num_flipped = 0
        integer(c_int64_t) :: num_components = 0
        integer(c_int64_t) :: largest_component = 0
        integer(c_int64_t) :: num_oriented_outward = 0
        integer(c_int64_t) :: num_unorientable = 0
        integer(c_int64_t) :: num_vertices_split = 0
        integer(c_int64_t) :: num_holes_detected = 0
        integer(c_int64_t) :: num_holes_filled = 0
        integer(c_int64_t) :: num_holes_skipped = 0
        integer(c_int64_t) :: num_faces_added = 0
        integer(c_int64_t) :: num_points_added = 0
        integer(c_int64_t) :: points_welded = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_shrinkwrap_opts`. Field order/types are ABI.
    type, bind(c) :: mio_shrinkwrap_opts_t
        type(c_ptr) :: weights = c_null_ptr
        type(c_ptr) :: target_region = c_null_ptr
        real(c_double) :: offset = 0.0_c_double
        real(c_double) :: max_distance = 0.0_c_double
        real(c_double) :: grid_cell_size = 0.0_c_double
        integer(c_int32_t) :: normal_weight = 0
        integer(c_int32_t) :: record_distance = 0
        integer(c_int32_t) :: record_closest_cell = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: reserved(5) = 0
    end type

    !> Interop mirror of C `mio_shrinkwrap_report`. Field order/types are ABI.
    type, bind(c) :: mio_shrinkwrap_report_t
        type(mio_surface_quality) :: quality
        integer(c_int64_t) :: num_projected = 0
        integer(c_int64_t) :: num_missed = 0
        integer(c_int64_t) :: num_skipped = 0
        real(c_double) :: max_displacement = 0.0_c_double
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_sobolev_opts`. Field order/types are ABI.
    type, bind(c) :: mio_sobolev_opts_t
        type(c_ptr) :: array = c_null_ptr
        type(c_ptr) :: fixed_points_array = c_null_ptr
        real(c_double) :: length_scale = 0.0_c_double
        real(c_double) :: tolerance = 1.0e-10_c_double
        integer(c_int32_t) :: max_iterations = 128
        integer(c_int32_t) :: fix_boundary = 0
        integer(c_int32_t) :: record_filtered = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: reserved(5) = 0
    end type

    !> Interop mirror of C `mio_sobolev_report`. Field order/types are ABI.
    type, bind(c) :: mio_sobolev_report_t
        integer(c_int64_t) :: num_iterations = 0
        integer(c_int64_t) :: num_fixed = 0
        integer(c_int64_t) :: num_isolated = 0
        real(c_double) :: residual = 0.0_c_double
        real(c_double) :: max_displacement = 0.0_c_double
        integer(c_int32_t) :: converged = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_sdf_opts`. Field order and types are ABI and
    !> must match bindings/c/include/meshioplusplus/meshioplusplus.h exactly;
    !> `reserved` is padding for additive growth and must stay zero.
    type, bind(c) :: mio_sdf_opts_t
        type(c_ptr) :: surface_region = c_null_ptr
        real(c_double) :: band = 0.0_c_double
        real(c_double) :: grid_cell_size = 0.0_c_double
        real(c_double) :: max_winding_work = 2.0e9_c_double
        integer(c_int32_t) :: sign = 1
        integer(c_int32_t) :: weight = 0
        integer(c_int32_t) :: location = 0
        integer(c_int32_t) :: watertight_check = 1
        integer(c_int32_t) :: record_closest_cell = 0
        integer(c_int32_t) :: record_inside = 0
        integer(c_int64_t) :: reserved(6) = 0
    end type

    !> Interop mirror of C `mio_voxel_opts`. Same ABI rules as above. Exactly
    !> one of `resolution` and `cell_size` may be given.
    type, bind(c) :: mio_voxel_opts_t
        type(c_ptr) :: resolution = c_null_ptr
        type(c_ptr) :: bounds = c_null_ptr
        real(c_double) :: cell_size = 0.0_c_double
        real(c_double) :: padding = 0.0_c_double
        real(c_double) :: padding_relative = 0.0_c_double
        integer(c_int64_t) :: max_cells = 20000000
        integer(c_int32_t) :: fill = 0
        integer(c_int32_t) :: attach_occupancy = 0
        integer(c_int32_t) :: sign = 1
        integer(c_int32_t) :: watertight_check = 1
        integer(c_int64_t) :: reserved(6) = 0
    end type

    !> Interop mirror of C `mio_compute_sdf_opts`. Same ABI rules as above. For
    !> `structure = 0` (voxel) exactly one of `resolution` and `cell_size` may be
    !> given; for `structure = 1` (octree) NEITHER may be -- its finest cell is
    !> root/2**depth and is therefore already determined.
    type, bind(c) :: mio_compute_sdf_opts_t
        type(c_ptr) :: resolution = c_null_ptr
        type(c_ptr) :: bounds = c_null_ptr
        real(c_double) :: cell_size = 0.0_c_double
        real(c_double) :: padding = 0.0_c_double
        real(c_double) :: padding_relative = 0.1_c_double
        real(c_double) :: band_cells = 1.0_c_double
        integer(c_int64_t) :: max_cells = 20000000
        integer(c_int64_t) :: root_resolution = 8
        integer(c_int64_t) :: max_depth = 4
        integer(c_int32_t) :: structure = 0
        integer(c_int32_t) :: record_levels = 1
        integer(c_int64_t) :: reserved(6) = 0
        type(mio_sdf_opts_t) :: distance
    end type

    ! Geometric statistics (bind(c); layout must match mio_stats_report in
    ! meshioplusplus.h). Per-cell-type counts are not carried across the C ABI.
    type, bind(c) :: mio_stats_report
        integer(c_int64_t) :: num_points
        integer(c_int64_t) :: num_cells
        real(c_double) :: bbox_min(3)
        real(c_double) :: bbox_max(3)
        real(c_double) :: extent(3)
        real(c_double) :: centroid(3)
        real(c_double) :: total_area
        real(c_double) :: signed_volume
        real(c_double) :: unsigned_volume
        integer(c_int64_t) :: num_inverted
    end type

    ! Summary of one data array (bind(c); layout must match
    ! mio_data_array_info in meshioplusplus.h). Per-component statistics are
    ! retrieved separately via the `data_info` procedure's optional arguments.
    type, bind(c) :: mio_data_array_info
        integer(c_int) :: location            !< a MIO_DATA_* value
        integer(c_int) :: dtype               !< a MIO_* dtype, as stored
        integer(c_int64_t) :: num_blocks      !< cell_data: cell blocks; else 1
        integer(c_int64_t) :: num_entries     !< rows (points / cells / length)
        integer(c_int64_t) :: num_components  !< product of trailing dimensions
        integer(c_int64_t) :: num_values      !< num_entries * num_components
        real(c_double) :: min                 !< over finite values, else NaN
        real(c_double) :: max                 !< over finite values, else NaN
        real(c_double) :: mean                !< over finite values, else NaN
        integer(c_int64_t) :: num_nan
        integer(c_int64_t) :: num_inf
        integer(c_int64_t) :: num_finite
        integer(c_int) :: inconsistent_blocks !< nonzero if blocks disagree
    end type

    ! Whole-mesh (or one named Cell region's) reduction of one array (bind(c);
    ! layout must match mio_field_integral_info in meshioplusplus.h). Per-
    ! component total/mean/domain_measure/num_nan are retrieved separately via
    ! the `data_integrate`/`data_integrate_region` procedures' optional
    ! arguments.
    type, bind(c) :: mio_field_integral_info
        integer(c_int64_t) :: num_components !< product of trailing dimensions
        integer(c_int64_t) :: num_cells      !< cells with a computable measure
        integer(c_int64_t) :: num_skipped    !< cells excluded: unmeasurable geometry
    end type

    ! mio_dtype values (must match the C enum in meshioplusplus.h).
    integer(c_int), parameter :: MIO_FLOAT32 = 0, MIO_FLOAT64 = 1
    integer(c_int), parameter :: MIO_INT32 = 4, MIO_INT64 = 5

    ! Data-operation enumerations (must match the C enums in meshioplusplus.h).
    integer(c_int), parameter :: MIO_DATA_POINT = 0, MIO_DATA_CELL = 1, MIO_DATA_FIELD = 2
    integer(c_int), parameter :: MIO_WEIGHT_UNIFORM = 0, MIO_WEIGHT_MEASURE = 1
    integer(c_int), parameter :: MIO_COND_CLAMP = 0, MIO_COND_NORMALIZE = 1
    !> Provenance modes (see doc/provenance.md). OFF is the default
    !> everywhere: a writer emits only the one-line credit.
    integer(c_int), parameter :: MIO_PROVENANCE_OFF = 0
    integer(c_int), parameter :: MIO_PROVENANCE_BEST_EFFORT = 1
    integer(c_int), parameter :: MIO_PROVENANCE_REQUIRED = 2
    integer(c_int), parameter :: MIO_COND_STANDARDIZE = 2
    integer(c_int), parameter :: MIO_SCOPE_COMPONENT = 0, MIO_SCOPE_MAGNITUDE = 1
    integer(c_int), parameter :: MIO_NAN_IGNORE = 0, MIO_NAN_REPLACE = 1, MIO_NAN_FAIL = 2

    ! Region kinds (must match the C enum mio_region_kind).
    integer(c_int), parameter :: MIO_REGION_POINT = 0, MIO_REGION_CELL = 1
    integer(c_int), parameter :: MIO_REGION_SIDE = 2

    ! Description of one named region (bind(c); layout must match
    ! mio_region_info in meshioplusplus.h). The entries themselves come back
    ! from the `regions` procedure as a separate allocatable array.
    type, bind(c) :: mio_region_info
        integer(c_int) :: kind               !< a MIO_REGION_* value
        integer(c_int) :: dim                !< topological dimension, or -1
        integer(c_int64_t) :: tag            !< format-native id, or -1
        integer(c_int64_t) :: num_entries    !< grouped entities (rows)
        integer(c_int64_t) :: stride         !< 2 for SIDE, else 1
    end type

    ! One cell block's full shape (bind(c); layout must match
    ! mio_cell_block_info in meshioplusplus.h). Deliberately NOT named
    ! mio_cell_block_info -- that name is already this module's own metadata
    ! summary type, which is a different thing (it carries the type NAME and
    ! comes from a file summary, not from a live mesh). Private, like every
    ! other c_mio_* interop detail; the public surface is the scalar
    ! cell_block_* queries below.
    type, bind(c) :: c_mio_cell_block_info
        integer(c_int64_t) :: num_cells
        integer(c_int64_t) :: nodes_per_cell   !< 0 when ragged
        integer(c_int32_t) :: is_ragged
        integer(c_int32_t) :: is_polyhedron    !< implies is_ragged
        integer(c_int64_t) :: num_faces
        integer(c_int64_t) :: num_nodes
        integer(c_int64_t) :: reserved(6)
    end type

    ! One ragged block's CSR shape (bind(c); matches mio_poly_conn_shape).
    type, bind(c) :: c_mio_poly_conn_shape
        integer(c_int32_t) :: is_polyhedron
        integer(c_int32_t) :: reserved0
        integer(c_int64_t) :: num_cells
        integer(c_int64_t) :: num_faces
        integer(c_int64_t) :: num_nodes
        integer(c_int64_t) :: reserved(4)
    end type

    integer, parameter :: MIO_MAX_NDIM = 8
    integer, parameter :: STRBUF_LEN = 4096

    !> Interop mirror of C `mio_read_opts`. Field order and types are ABI and
    !> must match bindings/c/include/meshioplusplus/meshioplusplus.h exactly;
    !> `reserved` is padding for additive growth and must stay zero.
    type, bind(c) :: mio_read_opts_t
        integer(c_int) :: points_only = 0
        integer(c_int) :: metadata_only = 0
        type(c_ptr) :: arrays = c_null_ptr
        integer(c_int64_t) :: num_arrays = 0
        integer(c_int) :: mmap_mode = 0
        !> Which step of a multi-step file to materialize: 0 (default) is the
        !> first, negative counts from the end. Takes one of the former
        !> `reserved` slots, so the struct size and every preceding field's
        !> offset are unchanged.
        integer(c_int64_t) :: time_step = 0
        !> Nonzero downgrades "this reader cannot represent construct X" errors
        !> to a warning plus a skip (currently mdpa's Table/Geometries/Mesh/
        !> Constraints blocks). Not "ignore all errors": a malformed file still
        !> fails. Takes a second former `reserved` slot; size unchanged.
        integer(c_int64_t) :: lenient = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_refine_opts`. Field order and types are ABI and
    !> must match bindings/c/include/meshioplusplus/meshioplusplus.h exactly;
    !> `reserved` is padding for additive growth and must stay zero.
    !>
    !> All zero (plus `levels = 1`) means "refine every cell once". At most ONE
    !> of `cells`, `region` and `predicate_array` may be given.
    type, bind(c) :: mio_refine_opts_t
        type(c_ptr) :: cells = c_null_ptr
        integer(c_int64_t) :: num_cells = 0
        type(c_ptr) :: region = c_null_ptr
        type(c_ptr) :: predicate_array = c_null_ptr
        real(c_double) :: predicate_value = 0.0_c_double
        integer(c_int32_t) :: levels = 1
        integer(c_int32_t) :: record_parent_ids = 0
        integer(c_int32_t) :: record_levels = 0
        integer(c_int32_t) :: closure = 0
        integer(c_int32_t) :: predicate_op = 0
        integer(c_int32_t) :: reserved_pad = 0
        !> Nonzero to attach refine:cell_id/refine:parent_id -- the persistent
        !> parent/child hierarchy a multigrid caller resolves across the
        !> sequence of meshes it keeps. Also forces refine:entity to be
        !> attached even when the closure leaves no hanging node. Takes one of
        !> the former `reserved` slots; size unchanged.
        integer(c_int64_t) :: record_hierarchy = 0
        integer(c_int64_t) :: reserved(5) = 0
    end type

    !> Interop mirror of C `mio_remesh_opts`. Field order and types are ABI
    !> and must match bindings/c/include/meshioplusplus/meshioplusplus.h
    !> exactly; `reserved`/`reserved_d` are padding for additive growth and
    !> must stay zero. `metric` is a C string, so it needs the same
    !> TARGET-buffer + c_loc idiom `region`/`predicate_array` use in
    !> mio_refine_opts_t -- see mesh_remesh below.
    type, bind(c) :: mio_remesh_opts_t
        integer(c_int64_t) :: num_clusters = 0
        integer(c_int32_t) :: subdivide = -1
        integer(c_int32_t) :: max_subdivide = 4
        real(c_double) :: subsample_ratio = 10.0_c_double
        integer(c_int32_t) :: max_iterations = 100
        integer(c_int32_t) :: max_repair_passes = 10
        type(c_ptr) :: metric = c_null_ptr
        real(c_double) :: gradation = 0.0_c_double
        integer(c_int32_t) :: preserve_boundary = 1
        integer(c_int32_t) :: reserved_pad = 0
        real(c_double) :: max_anisotropy = 4.0_c_double
        real(c_double) :: reserved_d(3) = 0.0_c_double
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_remesh_report`. Field order/types are ABI.
    type, bind(c) :: mio_remesh_report_t
        integer(c_int64_t) :: num_clusters = 0
        integer(c_int64_t) :: num_iterations = 0
        integer(c_int32_t) :: subdivide_applied = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: num_isolated_clusters = 0
        integer(c_int64_t) :: num_non_manifold_vertices = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_remesh_volume_opts`. Field order and types are
    !> ABI and must match bindings/c/include/meshioplusplus/meshioplusplus.h
    !> exactly; `reserved` is padding for additive growth and must stay zero.
    !> `distance` is embedded by value, the `mio_compute_sdf_opts_t` precedent
    !> above -- its own `reserved` tail absorbs future growth without shifting
    !> anything here.
    type, bind(c) :: mio_remesh_volume_opts_t
        type(c_ptr) :: resolution = c_null_ptr
        type(c_ptr) :: bounds = c_null_ptr
        real(c_double) :: cell_size = 0.0_c_double
        real(c_double) :: padding = 0.0_c_double
        real(c_double) :: padding_relative = 0.1_c_double
        integer(c_int64_t) :: max_cells = 20000000
        integer(c_int64_t) :: max_tets = 20000000
        real(c_double) :: warp_fraction = 0.35_c_double
        integer(c_int64_t) :: reserved(6) = 0
        type(mio_sdf_opts_t) :: distance
    end type

    !> Interop mirror of C `mio_remesh_volume_report`. Field order/types ABI.
    type, bind(c) :: mio_remesh_volume_report_t
        type(mio_surface_quality) :: input_quality
        integer(c_int64_t) :: num_tets = 0
        integer(c_int64_t) :: num_vertices_warped = 0
        integer(c_int64_t) :: num_tets_rejected = 0
        integer(c_int64_t) :: num_non_manifold_edges = 0
        integer(c_int64_t) :: reserved(4) = 0
    end type

    !> Interop mirror of C `mio_xdmf_series_opts`. Field order and types are ABI
    !> and must match bindings/c/include/meshioplusplus/meshioplusplus.h exactly;
    !> `reserved` is padding for additive growth and must stay zero.
    type, bind(c) :: mio_xdmf_series_opts_t
        type(c_ptr) :: data_format = c_null_ptr
        integer(c_int32_t) :: gzip_level = -1
        integer(c_int32_t) :: mode = 0
        integer(c_int32_t) :: auto_flush = 0
        integer(c_int32_t) :: reserved_pad = 0
        integer(c_int64_t) :: reserved(6) = 0
    end type

    !> One named region's shape, without its entries (see `mio_metadata%regions`).
    type :: mio_region_summary
        character(len=STRBUF_LEN) :: name = ''
        integer(c_int) :: kind = MIO_REGION_POINT
        integer(c_int) :: dim = -1
        integer(c_int64_t) :: tag = -1
        integer(c_int64_t) :: num_entries = 0
    end type

    !> One cell block's shape, without its connectivity.
    type :: mio_cell_block_info
        character(len=STRBUF_LEN) :: cell_type = ''
        integer(c_int64_t) :: num_cells = 0
        integer(c_int64_t) :: nodes_per_cell = 0
        logical :: ragged = .false.
    end type

    !> A file summary produced without loading the heavy arrays.
    !> `has_bbox` is false for a native summary, which never decodes the point
    !> coordinates; `fell_back_to_full_read` says whether the answer was cheap.
    type :: mio_metadata
        integer(c_int64_t) :: num_points = 0
        integer(c_int64_t) :: point_dim = 0
        integer(c_int64_t) :: num_cells = 0
        type(mio_cell_block_info), allocatable :: cell_blocks(:)
        character(len=STRBUF_LEN), allocatable :: point_data_names(:)
        character(len=STRBUF_LEN), allocatable :: cell_data_names(:)
        character(len=STRBUF_LEN), allocatable :: field_data_names(:)
        logical :: has_bbox = .false.
        logical :: fell_back_to_full_read = .false.
        !> The file's recorded time-series values; size 0 for a format with no
        !> time concept. This is the count `time_step` may name.
        real(c_double), allocatable :: time_values(:)
        !> The file's named regions, without their entries. Populated whenever
        !> the summary came from an already-read mesh; size 0 on a native
        !> metadata path, since none of those formats currently map regions.
        type(mio_region_summary), allocatable :: regions(:)
    end type

    type :: mio_mesh
        private
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => mesh_create
        procedure :: free => mesh_free
        procedure :: is_valid => mesh_is_valid
        procedure :: read => mesh_read
        procedure :: write => mesh_write
        ! -- operations --
        procedure :: extract_surface => mesh_extract_surface
        procedure :: extract_skin => mesh_extract_skin
        procedure :: attach_quality => mesh_attach_quality
        procedure :: quality_counts => mesh_quality_counts
        procedure :: reorder => mesh_reorder
        procedure :: transform => mesh_transform
        procedure :: clean => mesh_clean
        procedure :: smooth => mesh_smooth
        procedure :: optimize_volume => mesh_optimize_volume
        procedure :: crop_bbox => mesh_crop_bbox
        procedure :: crop_plane => mesh_crop_plane
        procedure :: crop_predicate => mesh_crop_predicate
        procedure :: slice => mesh_slice
        procedure :: isosurface => mesh_isosurface
        procedure :: gradient => mesh_gradient
        procedure :: hessian => mesh_hessian
        procedure :: estimate_error => mesh_estimate_error
        procedure :: remesh => mesh_remesh
        procedure :: curvature => mesh_curvature
        procedure :: repair => mesh_repair
        procedure :: sobolev_deform => mesh_sobolev_deform
        procedure :: remesh_volume => mesh_remesh_volume
        procedure :: split => mesh_split
        procedure :: convert_cells => mesh_convert_cells
        procedure :: subdivide => mesh_subdivide
        procedure :: agglomerate => mesh_agglomerate
        procedure :: refine => mesh_refine
        procedure :: decimate => mesh_decimate
        procedure :: partition => mesh_partition
        procedure :: partition_labels => mesh_partition_labels
        procedure :: stats => mesh_stats
        procedure :: voxelize => mesh_voxelize
        procedure :: compute_sdf => mesh_compute_sdf
        procedure :: watertight_check => mesh_watertight_check
        procedure :: sample_distance => mesh_sample_distance
        procedure :: distance_to_surface => mesh_distance_to_surface
        !> Named regions (doc/regions.md): the groups a set-capable format
        !> carries. `regions` returns one mio_region_info per group, with the
        !> names in `keys` and the flat int64 entries in `entries`.
        procedure :: regions => mesh_regions
        procedure :: add_region => mesh_add_region
        procedure :: compute_bandwidth => mesh_compute_bandwidth
        procedure :: equals => mesh_equals
        procedure :: diff => mesh_diff
        ! -- data operations (act on data arrays; geometry is never modified) --
        procedure :: data_drop => mesh_data_drop
        procedure :: data_keep => mesh_data_keep
        procedure :: data_rename => mesh_data_rename
        procedure :: data_point_to_cell => mesh_data_point_to_cell
        procedure :: data_cell_to_point => mesh_data_cell_to_point
        procedure :: data_calc => mesh_data_calc
        procedure :: data_condition => mesh_data_condition
        procedure :: data_info => mesh_data_info
        procedure :: data_integrate => mesh_data_integrate
        procedure :: data_integrate_region => mesh_data_integrate_region
        ! -- building --
        procedure :: set_points => mesh_set_points
        procedure, private :: mesh_add_cell_block_i32
        procedure, private :: mesh_add_cell_block_i64
        generic :: add_cell_block => mesh_add_cell_block_i32, mesh_add_cell_block_i64
        procedure :: add_polygon_block => mesh_add_polygon_block
        procedure :: add_polyhedron_block => mesh_add_polyhedron_block
        procedure, private :: mesh_add_point_data_r1
        procedure, private :: mesh_add_point_data_r2
        generic :: add_point_data => mesh_add_point_data_r1, mesh_add_point_data_r2
        procedure :: add_cell_data => mesh_add_cell_data_r1
        procedure :: add_field_data => mesh_add_field_data_r1
        ! -- inspecting --
        procedure :: num_points => mesh_num_points
        procedure :: point_dim => mesh_point_dim
        procedure :: num_cell_blocks => mesh_num_cell_blocks
        procedure :: get_points => mesh_get_points
        procedure :: points_ptr => mesh_points_ptr
        procedure :: cell_block_type => mesh_cell_block_type
        procedure :: cell_block_num_cells => mesh_cell_block_num_cells
        procedure :: cell_block_nodes_per_cell => mesh_cell_block_nodes_per_cell
        procedure :: cell_block_is_ragged => mesh_cell_block_is_ragged
        procedure :: cell_block_is_polyhedron => mesh_cell_block_is_polyhedron
        procedure :: get_cell_block => mesh_get_cell_block
        procedure :: get_polygon_block => mesh_get_polygon_block
        procedure :: get_polyhedron_block => mesh_get_polyhedron_block
        procedure :: num_point_data => mesh_num_point_data
        procedure :: point_data_name => mesh_point_data_name
        procedure, private :: mesh_get_point_data_r1
        procedure, private :: mesh_get_point_data_r2
        generic :: get_point_data => mesh_get_point_data_r1, mesh_get_point_data_r2
        procedure :: num_cell_data => mesh_num_cell_data
        procedure :: cell_data_name => mesh_cell_data_name
        procedure :: cell_data_num_blocks => mesh_cell_data_num_blocks
        procedure :: get_cell_data => mesh_get_cell_data_r1
        procedure :: num_field_data => mesh_num_field_data
        procedure :: field_data_name => mesh_field_data_name
        procedure :: get_field_data => mesh_get_field_data_r1
    end type mio_mesh

    !> A multi-file / transient dataset: an ordered PLAN over a set of files
    !> (or the steps inside one multi-step file), read ONE STEP AT A TIME.
    !>
    !> That is the whole point -- a 500-step dataset must be traversable
    !> without materializing it -- so the handle stores paths, per-file step
    !> indices and time values, and never a mesh. `read_step` therefore hands
    !> back an OWNED `mio_mesh` the caller frees, rather than a borrow the
    !> sequence would have to cache.
    !>
    !> Ordering is natural-numeric, so `out_9.vtu` precedes `out_10.vtu`.
    !> Handles are freed explicitly, exactly like `mio_mesh` -- no finalizer.
    !>
    !> ```fortran
    !> type(mio_sequence) :: seq
    !> type(mio_mesh) :: step
    !> integer :: i
    !> call seq%open('out_*.vtu')
    !> do i = 1, seq%count()
    !>     step = seq%read_step(i)      ! 1-based, Fortran style
    !>     ! ... use step, whose time is seq%time(i) ...
    !>     call step%free()
    !> end do
    !> call seq%free()
    !> ```
    !>
    !> See doc/sequences.md.
    type :: mio_sequence
        private
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: open => sequence_open
        procedure :: open_list => sequence_open_list
        procedure :: free => sequence_free
        procedure :: is_valid => sequence_is_valid
        procedure :: count => sequence_count
        procedure :: path => sequence_path
        procedure :: step => sequence_step
        procedure :: time => sequence_time
        procedure :: time_source => sequence_time_source
        procedure :: read_step => sequence_read_step
        procedure :: to_timeseries => sequence_to_timeseries
    end type mio_sequence

    !> A transient (time-series) XDMF writer: the write half of what the
    !> `time_step` read option and `mio_metadata%time_values` expose on the read
    !> side. A solver writes the mesh ONCE and then one cheap step per solve, so
    !> this is a stateful handle rather than a `write` call -- the one writer
    !> here that `m%write()` cannot express.
    !>
    !>     type(mio_xdmf_series) :: series
    !>     call series%create('simulation.xdmf')        ! "HDF" by default
    !>     call series%write_points_cells(m)            ! the static grid, once
    !>     do k = 1, nsteps
    !>         call solve(m)
    !>         call series%write_data(k*dt, m)          ! point_data/cell_data only
    !>     end do
    !>     call series%finalize()                       ! free() would do it too
    !>     call series%free()
    !>
    !> The `.xdmf` light data is buffered and written at finalize, so by default
    !> a series is only readable after `finalize()` (or `free()`, which finalizes
    !> first). Call `flush()` to make it readable *now* -- what keeps a run that
    !> is killed or still going from leaving nothing openable -- or pass
    !> `auto_flush=.true.` to `create()` to do that after every `write_data`.
    !> `mode='append'` continues a series already at the path instead of
    !> overwriting it; a path with no file yet is simply a fresh series, so a
    !> restartable solver can pass it unconditionally.
    !> Heavy data for `"HDF"` goes to a `<path minus extension>.h5`
    !> SIBLING of the `.xdmf`. Handles are freed explicitly, exactly like
    !> `mio_mesh` -- there is no finalizer.
    type :: mio_xdmf_series
        private
        type(c_ptr) :: handle = c_null_ptr
    contains
        procedure :: create => xdmf_series_create
        procedure :: free => xdmf_series_free
        procedure :: is_valid => xdmf_series_is_valid
        procedure :: write_points_cells => xdmf_series_write_points_cells
        procedure :: write_data => xdmf_series_write_data
        procedure :: flush => xdmf_series_flush
        procedure :: finalize => xdmf_series_finalize
        procedure :: finalized => xdmf_series_finalized
        procedure :: num_steps => xdmf_series_num_steps
    end type mio_xdmf_series

    ! ------------------------------------------------------------------
    ! Raw bind(c) interfaces to libmeshioplusplus (private; the OO layer
    ! above is the public surface). Data pointers cross as type(c_ptr) via
    ! c_loc() so one C symbol serves every Fortran type/kind.
    ! ------------------------------------------------------------------
    interface
        function c_mio_version() bind(c, name="mio_version") result(p)
            import :: c_ptr
            type(c_ptr) :: p
        end function

        function c_mio_mesh_backend() bind(c, name="mio_mesh_backend") result(p)
            import :: c_ptr
            type(c_ptr) :: p
        end function

        function c_mio_last_error() bind(c, name="mio_last_error") result(p)
            import :: c_ptr
            type(c_ptr) :: p
        end function

        function c_mio_format_readable(format) bind(c, name="mio_format_readable") result(r)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: format
            integer(c_int) :: r
        end function

        function c_mio_format_writable(format) bind(c, name="mio_format_writable") result(r)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: format
            integer(c_int) :: r
        end function

        function c_mio_mesh_create() bind(c, name="mio_mesh_create") result(h)
            import :: c_ptr
            type(c_ptr) :: h
        end function

        subroutine c_mio_mesh_free(h) bind(c, name="mio_mesh_free")
            import :: c_ptr
            type(c_ptr), value :: h
        end subroutine

        function c_mio_read(path, format) bind(c, name="mio_read") result(h)
            import :: c_ptr, c_char
            character(kind=c_char), dimension(*), intent(in) :: path, format
            type(c_ptr) :: h
        end function

        function c_mio_read_ex(path, format, opts) bind(c, name="mio_read_ex") result(h)
            import :: c_ptr, c_char, mio_read_opts_t
            character(kind=c_char), dimension(*), intent(in) :: path, format
            type(mio_read_opts_t), intent(in) :: opts
            type(c_ptr) :: h
        end function

        subroutine c_mio_read_opts_init(opts) bind(c, name="mio_read_opts_init")
            import :: mio_read_opts_t
            type(mio_read_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_read_metadata_create(path, format) &
                bind(c, name="mio_read_metadata_create") result(h)
            import :: c_ptr, c_char
            character(kind=c_char), dimension(*), intent(in) :: path, format
            type(c_ptr) :: h
        end function

        function c_mio_read_metadata_num_points(h) &
                bind(c, name="mio_read_metadata_num_points") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_point_dim(h) &
                bind(c, name="mio_read_metadata_point_dim") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_num_cells(h) &
                bind(c, name="mio_read_metadata_num_cells") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_num_cell_blocks(h) &
                bind(c, name="mio_read_metadata_num_cell_blocks") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_cell_block(h, index, num_cells, npc, ragged) &
                bind(c, name="mio_read_metadata_cell_block") result(s)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index
            integer(c_int64_t), intent(out) :: num_cells, npc
            integer(c_int), intent(out) :: ragged
            integer(c_int) :: s
        end function

        function c_mio_read_metadata_cell_block_type(h, index, buf, buflen) &
                bind(c, name="mio_read_metadata_cell_block_type") result(n)
            import :: c_ptr, c_int64_t, c_char
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_num_names(h, location) &
                bind(c, name="mio_read_metadata_num_names") result(n)
            import :: c_ptr, c_int64_t, c_int
            type(c_ptr), value :: h
            integer(c_int), value :: location
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_num_time_values(h) &
                bind(c, name="mio_read_metadata_num_time_values") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_time_values(h, out, count) &
                bind(c, name="mio_read_metadata_time_values") result(n)
            import :: c_ptr, c_int64_t, c_double
            type(c_ptr), value :: h
            real(c_double), intent(out) :: out(*)
            integer(c_int64_t), value :: count
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_num_regions(h) &
                bind(c, name="mio_read_metadata_num_regions") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_region_name(h, index, buf, buflen) &
                bind(c, name="mio_read_metadata_region_name") result(n)
            import :: c_ptr, c_int64_t, c_char
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index
            character(kind=c_char), intent(out) :: buf(*)
            integer(c_int64_t), value :: buflen
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_region_info(h, index, out) &
                bind(c, name="mio_read_metadata_region_info") result(s)
            import :: c_ptr, c_int64_t, c_int, mio_region_info
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index
            type(mio_region_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_read_metadata_name(h, location, index, buf, buflen) &
                bind(c, name="mio_read_metadata_name") result(n)
            import :: c_ptr, c_int64_t, c_int, c_char
            type(c_ptr), value :: h
            integer(c_int), value :: location
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_read_metadata_fell_back(h) &
                bind(c, name="mio_read_metadata_fell_back") result(n)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            integer(c_int) :: n
        end function

        subroutine c_mio_read_metadata_free(h) bind(c, name="mio_read_metadata_free")
            import :: c_ptr
            type(c_ptr), value :: h
        end subroutine

        function c_mio_write(path, h, format) bind(c, name="mio_write") result(s)
            import :: c_ptr, c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: path, format
            type(c_ptr), value :: h
            integer(c_int) :: s
        end function

        function c_mio_convert(in_path, in_format, out_path, out_format) &
                bind(c, name="mio_convert") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: in_path, in_format
            character(kind=c_char), dimension(*), intent(in) :: out_path, out_format
            integer(c_int) :: s
        end function

        function c_mio_pipeline_run_file(settings_path) &
                bind(c, name="mio_pipeline_run_file") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: settings_path
            integer(c_int) :: s
        end function

        function c_mio_pipeline_run_json(json_text) &
                bind(c, name="mio_pipeline_run_json") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: json_text
            integer(c_int) :: s
        end function

        function c_mio_sequence_open(pattern) &
                bind(c, name="mio_sequence_open") result(p)
            import :: c_char, c_ptr
            character(kind=c_char), dimension(*), intent(in) :: pattern
            type(c_ptr) :: p
        end function

        function c_mio_sequence_open_list(paths, num_paths) &
                bind(c, name="mio_sequence_open_list") result(p)
            import :: c_ptr, c_int64_t
            type(c_ptr), dimension(*), intent(in) :: paths
            integer(c_int64_t), value :: num_paths
            type(c_ptr) :: p
        end function

        function c_mio_sequence_count(seq) &
                bind(c, name="mio_sequence_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: seq
            integer(c_int64_t) :: n
        end function

        function c_mio_sequence_path(seq, index, buf, buflen) &
                bind(c, name="mio_sequence_path") result(n)
            import :: c_ptr, c_int64_t, c_char
            type(c_ptr), value :: seq
            integer(c_int64_t), value :: index
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t), value :: buflen
            integer(c_int64_t) :: n
        end function

        function c_mio_sequence_step(seq, index) &
                bind(c, name="mio_sequence_step") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: seq
            integer(c_int64_t), value :: index
            integer(c_int64_t) :: n
        end function

        function c_mio_sequence_time(seq, index, out_time) &
                bind(c, name="mio_sequence_time") result(s)
            import :: c_ptr, c_int64_t, c_double, c_int
            type(c_ptr), value :: seq
            integer(c_int64_t), value :: index
            real(c_double), intent(out) :: out_time
            integer(c_int) :: s
        end function

        function c_mio_sequence_time_source(seq, index) &
                bind(c, name="mio_sequence_time_source") result(r)
            import :: c_ptr, c_int64_t, c_int32_t
            type(c_ptr), value :: seq
            integer(c_int64_t), value :: index
            integer(c_int32_t) :: r
        end function

        function c_mio_sequence_read(seq, index) &
                bind(c, name="mio_sequence_read") result(p)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: seq
            integer(c_int64_t), value :: index
            type(c_ptr) :: p
        end function

        subroutine c_mio_sequence_free(seq) bind(c, name="mio_sequence_free")
            import :: c_ptr
            type(c_ptr), value :: seq
        end subroutine

        function c_mio_sequence_to_timeseries(seq, out_path, out_format) &
                bind(c, name="mio_sequence_to_timeseries") result(s)
            import :: c_ptr, c_char, c_int
            type(c_ptr), value :: seq
            character(kind=c_char), dimension(*), intent(in) :: out_path
            character(kind=c_char), dimension(*), intent(in) :: out_format
            integer(c_int) :: s
        end function

        function c_mio_timeseries_to_sequence(in_path, in_format, out_pattern, out_format) &
                bind(c, name="mio_timeseries_to_sequence") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: in_path
            character(kind=c_char), dimension(*), intent(in) :: in_format
            character(kind=c_char), dimension(*), intent(in) :: out_pattern
            character(kind=c_char), dimension(*), intent(in) :: out_format
            integer(c_int) :: s
        end function

        function c_mio_sequence_pipeline_run_file(settings_path) &
                bind(c, name="mio_sequence_pipeline_run_file") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: settings_path
            integer(c_int) :: s
        end function

        function c_mio_sequence_pipeline_run_json(json_text) &
                bind(c, name="mio_sequence_pipeline_run_json") result(s)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: json_text
            integer(c_int) :: s
        end function

        function c_mio_pipeline_has_json() &
                bind(c, name="mio_pipeline_has_json") result(r)
            import :: c_int32_t
            integer(c_int32_t) :: r
        end function

        function c_mio_extract_surface(h, record_parent_ids) &
                bind(c, name="mio_extract_surface") result(r)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            integer(c_int), value :: record_parent_ids
            type(c_ptr) :: r
        end function

        function c_mio_extract_skin(h, linearize) bind(c, name="mio_extract_skin") result(r)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            integer(c_int), value :: linearize
            type(c_ptr) :: r
        end function

        function c_mio_attach_quality(h) bind(c, name="mio_attach_quality") result(r)
            import :: c_ptr
            type(c_ptr), value :: h
            type(c_ptr) :: r
        end function

        function c_mio_quality_counts(h, nc, ninv, ndeg) &
                bind(c, name="mio_quality_counts") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), intent(out) :: nc, ninv, ndeg
            integer(c_int) :: s
        end function

        function c_mio_merge(meshes, count, weld, atol, source_tag, data_policy, drop_dup) &
                bind(c, name="mio_merge") result(r)
            import :: c_ptr, c_int, c_int64_t, c_double
            type(c_ptr), intent(in) :: meshes(*)
            integer(c_int64_t), value :: count
            integer(c_int), value :: weld, source_tag, data_policy, drop_dup
            real(c_double), value :: atol
            type(c_ptr) :: r
        end function

        function c_mio_interpolate(source, target, method, arrays, count, extrapolate, &
                                   default_value, on_conflict) &
                bind(c, name="mio_interpolate") result(r)
            import :: c_ptr, c_char, c_int, c_int64_t, c_double
            type(c_ptr), value :: source, target
            character(kind=c_char), dimension(*), intent(in) :: method
            type(c_ptr), value :: arrays
            integer(c_int64_t), value :: count
            integer(c_int), value :: extrapolate
            real(c_double), value :: default_value
            character(kind=c_char), dimension(*), intent(in) :: on_conflict
            type(c_ptr) :: r
        end function

        function c_mio_conservative_interpolate(source, target, arrays, count, &
                                                default_value, on_conflict) &
                bind(c, name="mio_conservative_interpolate") result(r)
            import :: c_ptr, c_char, c_int64_t, c_double
            type(c_ptr), value :: source, target
            type(c_ptr), value :: arrays
            integer(c_int64_t), value :: count
            real(c_double), value :: default_value
            character(kind=c_char), dimension(*), intent(in) :: on_conflict
            type(c_ptr) :: r
        end function

        function c_mio_undo_green(coarse, fine, ngroups, nremoved) &
                bind(c, name="mio_undo_green") result(r)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: coarse, fine
            integer(c_int64_t), intent(out) :: ngroups, nremoved
            type(c_ptr) :: r
        end function

        function c_mio_reorder(h, method) bind(c, name="mio_reorder") result(r)
            import :: c_ptr, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: method
            type(c_ptr) :: r
        end function

        function c_mio_transform(h, matrix, rotate_data) bind(c, name="mio_transform") result(r)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h
            real(c_double), intent(in) :: matrix(*)
            integer(c_int), value :: rotate_data
            type(c_ptr) :: r
        end function

        function c_mio_clean(h, weld, atol, rmorph, ddeg, ddup, nweld, norph, ndeg, ndup) &
                bind(c, name="mio_clean") result(r)
            import :: c_ptr, c_int, c_int64_t, c_double
            type(c_ptr), value :: h
            integer(c_int), value :: weld, rmorph, ddeg, ddup
            real(c_double), value :: atol
            integer(c_int64_t), intent(out) :: nweld, norph, ndeg, ndup
            type(c_ptr) :: r
        end function

        function c_mio_smooth(h, method, iters, lambda, mu, fixb, feat, fangle, guard, &
                              nmoved, maxdisp, nskip) bind(c, name="mio_smooth") result(r)
            import :: c_ptr, c_char, c_int, c_int64_t, c_double
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: method
            integer(c_int), value :: iters, fixb, feat, guard
            real(c_double), value :: lambda, mu, fangle
            integer(c_int64_t), intent(out) :: nmoved, nskip
            real(c_double), intent(out) :: maxdisp
            type(c_ptr) :: r
        end function

        function c_mio_optimize_volume(h, maxiter, relocate, flip, presb, minimp, nflips, n23, &
                                       n32, nmoved, ntets, qbefore, qafter) &
                bind(c, name="mio_optimize_volume") result(r)
            import :: c_ptr, c_int, c_int64_t, c_double
            type(c_ptr), value :: h
            integer(c_int), value :: maxiter, relocate, flip, presb
            real(c_double), value :: minimp
            integer(c_int64_t), intent(out) :: nflips, n23, n32, nmoved, ntets
            real(c_double), intent(out) :: qbefore, qafter
            type(c_ptr) :: r
        end function

        function c_mio_crop_bbox(h, lo, hi, mode, record_ids) &
                bind(c, name="mio_crop_bbox") result(r)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h
            real(c_double), intent(in) :: lo(*), hi(*)
            integer(c_int), value :: mode, record_ids
            type(c_ptr) :: r
        end function

        function c_mio_crop_plane(h, point, normal, mode, record_ids) &
                bind(c, name="mio_crop_plane") result(r)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h
            real(c_double), intent(in) :: point(*), normal(*)
            integer(c_int), value :: mode, record_ids
            type(c_ptr) :: r
        end function

        function c_mio_slice(h, origin, normal, record_parent_ids) &
                bind(c, name="mio_slice") result(r)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: h
            real(c_double), intent(in) :: origin(*), normal(*)
            integer(c_int), value :: record_parent_ids
            type(c_ptr) :: r
        end function

        function c_mio_isosurface(h, array_name, isovalues, n_isovalues, component, &
                                  record_parent_ids) &
                bind(c, name="mio_isosurface") result(r)
            import :: c_ptr, c_int, c_double, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: array_name
            real(c_double), intent(in) :: isovalues(*)
            integer(c_int), value :: n_isovalues, component, record_parent_ids
            type(c_ptr) :: r
        end function

        function c_mio_gradient(h, array_name, op, method, location, output_name, &
                               component, overwrite, n_skipped, n_fallback) &
                bind(c, name="mio_gradient") result(r)
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: array_name, op, method
            character(kind=c_char), dimension(*), intent(in) :: location, output_name
            integer(c_int), value :: component, overwrite
            integer(c_int64_t), intent(out) :: n_skipped, n_fallback
            type(c_ptr) :: r
        end function

        function c_mio_hessian(h, array_name, method, location, output_name, &
                              overwrite, n_skipped, n_fallback) &
                bind(c, name="mio_hessian") result(r)
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: array_name, method, location
            character(kind=c_char), dimension(*), intent(in) :: output_name
            integer(c_int), value :: overwrite
            integer(c_int64_t), intent(out) :: n_skipped, n_fallback
            type(c_ptr) :: r
        end function

        function c_mio_estimate_error(h, array_name, method, marking, marking_value, &
                                      output_name, marked_name, overwrite, global_error, &
                                      n_skipped, n_marked) &
                bind(c, name="mio_estimate_error") result(r)
            import :: c_ptr, c_int, c_int64_t, c_char, c_double
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: array_name, method, marking
            character(kind=c_char), dimension(*), intent(in) :: output_name, marked_name
            real(c_double), value :: marking_value
            integer(c_int), value :: overwrite
            real(c_double), intent(out) :: global_error
            integer(c_int64_t), intent(out) :: n_skipped, n_marked
            type(c_ptr) :: r
        end function
        function c_mio_remesh(h, num_clusters, subdivide, subsample_ratio, max_subdivide, &
                              max_iterations, max_repair_passes, metric, gradation, &
                              preserve_boundary, num_clusters_out, num_iterations, &
                              subdivide_applied, num_isolated_clusters, &
                              num_non_manifold_vertices) &
                bind(c, name="mio_remesh") result(r)
            import :: c_ptr, c_int, c_int64_t, c_char, c_double
            type(c_ptr), value :: h
            integer(c_int64_t), value :: num_clusters
            integer(c_int), value :: subdivide, max_subdivide, max_iterations, max_repair_passes
            real(c_double), value :: subsample_ratio
            character(kind=c_char), dimension(*), intent(in) :: metric
            real(c_double), value :: gradation
            integer(c_int), value :: preserve_boundary
            integer(c_int64_t), intent(out) :: num_clusters_out, num_iterations
            integer(c_int), intent(out) :: subdivide_applied
            integer(c_int64_t), intent(out) :: num_isolated_clusters
            integer(c_int64_t), intent(out) :: num_non_manifold_vertices
            type(c_ptr) :: r
        end function

        subroutine c_mio_remesh_opts_init(opts) bind(c, name="mio_remesh_opts_init")
            import :: mio_remesh_opts_t
            type(mio_remesh_opts_t), intent(out) :: opts
        end subroutine

        subroutine c_mio_curvature_opts_init(opts) bind(c, name="mio_curvature_opts_init")
            import :: mio_curvature_opts_t
            type(mio_curvature_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_compute_curvature(h, opts, report) &
                bind(c, name="mio_compute_curvature") result(r)
            import :: c_ptr, mio_curvature_opts_t, mio_curvature_report_t
            type(c_ptr), value :: h
            type(mio_curvature_opts_t), intent(in) :: opts
            type(mio_curvature_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        subroutine c_mio_repair_opts_init(opts) bind(c, name="mio_repair_opts_init")
            import :: mio_repair_opts_t
            type(mio_repair_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_repair(h, opts, report) bind(c, name="mio_repair") result(r)
            import :: c_ptr, mio_repair_opts_t, mio_repair_report_t
            type(c_ptr), value :: h
            type(mio_repair_opts_t), intent(in) :: opts
            type(mio_repair_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        subroutine c_mio_shrinkwrap_opts_init(opts) bind(c, name="mio_shrinkwrap_opts_init")
            import :: mio_shrinkwrap_opts_t
            type(mio_shrinkwrap_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_shrinkwrap(h, target, opts, report) bind(c, name="mio_shrinkwrap") result(r)
            import :: c_ptr, mio_shrinkwrap_opts_t, mio_shrinkwrap_report_t
            type(c_ptr), value :: h, target
            type(mio_shrinkwrap_opts_t), intent(in) :: opts
            type(mio_shrinkwrap_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        subroutine c_mio_sobolev_opts_init(opts) bind(c, name="mio_sobolev_opts_init")
            import :: mio_sobolev_opts_t
            type(mio_sobolev_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_sobolev_deform(h, opts, report) bind(c, name="mio_sobolev_deform") result(r)
            import :: c_ptr, mio_sobolev_opts_t, mio_sobolev_report_t
            type(c_ptr), value :: h
            type(mio_sobolev_opts_t), intent(in) :: opts
            type(mio_sobolev_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        function c_mio_remesh_ex(h, opts, report) bind(c, name="mio_remesh_ex") result(r)
            import :: c_ptr, mio_remesh_opts_t, mio_remesh_report_t
            type(c_ptr), value :: h
            type(mio_remesh_opts_t), intent(in) :: opts
            type(mio_remesh_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        subroutine c_mio_remesh_volume_opts_init(opts) &
                bind(c, name="mio_remesh_volume_opts_init")
            import :: mio_remesh_volume_opts_t
            type(mio_remesh_volume_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_remesh_volume_ex(h, opts, report) &
                bind(c, name="mio_remesh_volume_ex") result(r)
            import :: c_ptr, mio_remesh_volume_opts_t, mio_remesh_volume_report_t
            type(c_ptr), value :: h
            type(mio_remesh_volume_opts_t), intent(in) :: opts
            type(mio_remesh_volume_report_t), intent(out) :: report
            type(c_ptr) :: r
        end function

        function c_mio_split(h, by, tag_name) bind(c, name="mio_split") result(r)
            import :: c_ptr, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: by, tag_name
            type(c_ptr) :: r
        end function

        function c_mio_split_result_count(r) bind(c, name="mio_split_result_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: n
        end function

        function c_mio_split_result_key(r, index, buf, buflen) &
                bind(c, name="mio_split_result_key") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_split_result_take_mesh(r, index) &
                bind(c, name="mio_split_result_take_mesh") result(m)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            type(c_ptr) :: m
        end function

        subroutine c_mio_split_result_free(r) bind(c, name="mio_split_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_convert_cells(h, mode, record_parent_ids) &
                bind(c, name="mio_convert_cells") result(r)
            import :: c_ptr, c_char, c_int
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: mode
            integer(c_int), value :: record_parent_ids
            type(c_ptr) :: r
        end function

        function c_mio_convert_cells_result_take_mesh(r) &
                bind(c, name="mio_convert_cells_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        function c_mio_convert_cells_result_point_map(r, data, dtype, n) &
                bind(c, name="mio_convert_cells_result_point_map") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: r
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int64_t), intent(out) :: n
            integer(c_int) :: s
        end function

        subroutine c_mio_convert_cells_result_free(r) &
                bind(c, name="mio_convert_cells_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_subdivide(h, record_parent_ids) &
                bind(c, name="mio_subdivide") result(r)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            integer(c_int), value :: record_parent_ids
            type(c_ptr) :: r
        end function

        function c_mio_subdivide_result_take_mesh(r) &
                bind(c, name="mio_subdivide_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        subroutine c_mio_subdivide_result_free(r) &
                bind(c, name="mio_subdivide_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_agglomerate(h, target_group_size) &
                bind(c, name="mio_agglomerate") result(r)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: target_group_size
            type(c_ptr) :: r
        end function

        function c_mio_agglomerate_result_take_mesh(r) &
                bind(c, name="mio_agglomerate_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        subroutine c_mio_agglomerate_result_free(r) &
                bind(c, name="mio_agglomerate_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_refine(h, levels, record_parent_ids) &
                bind(c, name="mio_refine") result(r)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            integer(c_int), value :: levels
            integer(c_int), value :: record_parent_ids
            type(c_ptr) :: r
        end function

        subroutine c_mio_sdf_opts_init(opts) bind(c, name="mio_sdf_opts_init")
            import :: mio_sdf_opts_t
            type(mio_sdf_opts_t), intent(out) :: opts
        end subroutine

        subroutine c_mio_voxel_opts_init(opts) bind(c, name="mio_voxel_opts_init")
            import :: mio_voxel_opts_t
            type(mio_voxel_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_grid(dims, origin, spacing, max_cells) bind(c, name="mio_grid") result(r)
            import :: c_ptr, c_int64_t, c_double
            integer(c_int64_t), intent(in) :: dims(*)
            real(c_double), intent(in) :: origin(*), spacing(*)
            integer(c_int64_t), value :: max_cells
            type(c_ptr) :: r
        end function

        function c_mio_voxelize(h, opts, dims, origin, spacing, occupied) &
                bind(c, name="mio_voxelize") result(r)
            import :: c_ptr, c_int64_t, c_double, mio_voxel_opts_t
            type(c_ptr), value :: h
            type(mio_voxel_opts_t), intent(in) :: opts
            integer(c_int64_t), intent(out) :: dims(*)
            real(c_double), intent(out) :: origin(*), spacing(*)
            integer(c_int64_t), intent(out) :: occupied
            type(c_ptr) :: r
        end function

        subroutine c_mio_compute_sdf_opts_init(opts) &
                bind(c, name="mio_compute_sdf_opts_init")
            import :: mio_compute_sdf_opts_t
            type(mio_compute_sdf_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_compute_sdf(h, opts, dims, origin, spacing, max_depth, banded, q) &
                bind(c, name="mio_compute_sdf") result(r)
            import :: c_ptr, c_int64_t, c_double, mio_compute_sdf_opts_t, mio_surface_quality
            type(c_ptr), value :: h
            type(mio_compute_sdf_opts_t), intent(in) :: opts
            integer(c_int64_t), intent(out) :: dims(*)
            real(c_double), intent(out) :: origin(*), spacing(*)
            integer(c_int64_t), intent(out) :: max_depth, banded
            type(mio_surface_quality), intent(out) :: q
            type(c_ptr) :: r
        end function

        function c_mio_crop_predicate(h, array, compare, value, record_ids) &
                bind(c, name="mio_crop_predicate") result(r)
            import :: c_ptr, c_char, c_int, c_double
            type(c_ptr), value :: h
            character(kind=c_char), intent(in) :: array(*)
            integer(c_int), value :: compare, record_ids
            real(c_double), value :: value
            type(c_ptr) :: r
        end function

        function c_mio_surface_watertight_check(h, out) &
                bind(c, name="mio_surface_watertight_check") result(r)
            import :: c_ptr, c_int, mio_surface_quality
            type(c_ptr), value :: h
            type(mio_surface_quality), intent(out) :: out
            integer(c_int) :: r
        end function

        function c_mio_sample_distance(h, points, n_points, opts, out) &
                bind(c, name="mio_sample_distance") result(r)
            import :: c_ptr, c_int, c_int64_t, c_double, mio_sdf_opts_t
            type(c_ptr), value :: h
            real(c_double), intent(in) :: points(*)
            integer(c_int64_t), value :: n_points
            type(mio_sdf_opts_t), intent(in) :: opts
            real(c_double), intent(out) :: out(*)
            integer(c_int) :: r
        end function

        function c_mio_distance_to_surface(q, s, opts, banded, quality) &
                bind(c, name="mio_distance_to_surface") result(r)
            import :: c_ptr, c_int64_t, mio_sdf_opts_t, mio_surface_quality
            type(c_ptr), value :: q, s
            type(mio_sdf_opts_t), intent(in) :: opts
            integer(c_int64_t), intent(out) :: banded
            type(mio_surface_quality), intent(out) :: quality
            type(c_ptr) :: r
        end function

        subroutine c_mio_refine_opts_init(opts) bind(c, name="mio_refine_opts_init")
            import :: mio_refine_opts_t
            type(mio_refine_opts_t), intent(out) :: opts
        end subroutine

        function c_mio_refine_ex(h, opts) bind(c, name="mio_refine_ex") result(r)
            import :: c_ptr, mio_refine_opts_t
            type(c_ptr), value :: h
            type(mio_refine_opts_t), intent(in) :: opts
            type(c_ptr) :: r
        end function

        function c_mio_refine_result_take_mesh(r) &
                bind(c, name="mio_refine_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        function c_mio_refine_result_point_map(r, data, dtype, n) &
                bind(c, name="mio_refine_result_point_map") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: r
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int64_t), intent(out) :: n
            integer(c_int) :: s
        end function

        subroutine c_mio_refine_result_free(r) &
                bind(c, name="mio_refine_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_decimate(h, ratio, faces, max_error, placement, preserve_boundary, &
                                preserve_features, feature_angle) &
                bind(c, name="mio_decimate") result(r)
            import :: c_ptr, c_char, c_int, c_int64_t, c_double
            type(c_ptr), value :: h
            real(c_double), value :: ratio, max_error, feature_angle
            integer(c_int64_t), value :: faces
            character(kind=c_char), dimension(*), intent(in) :: placement
            integer(c_int), value :: preserve_boundary, preserve_features
            type(c_ptr) :: r
        end function

        function c_mio_decimate_result_take_mesh(r) &
                bind(c, name="mio_decimate_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        function c_mio_decimate_result_point_map(r, data, dtype, n) &
                bind(c, name="mio_decimate_result_point_map") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: r
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int64_t), intent(out) :: n
            integer(c_int) :: s
        end function

        function c_mio_decimate_result_faces_removed(r) &
                bind(c, name="mio_decimate_result_faces_removed") result(v)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: v
        end function

        function c_mio_decimate_result_points_removed(r) &
                bind(c, name="mio_decimate_result_points_removed") result(v)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: v
        end function

        function c_mio_decimate_result_collapses_rejected(r) &
                bind(c, name="mio_decimate_result_collapses_rejected") result(v)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: v
        end function

        function c_mio_decimate_result_max_error_applied(r) &
                bind(c, name="mio_decimate_result_max_error_applied") result(v)
            import :: c_ptr, c_double
            type(c_ptr), value :: r
            real(c_double) :: v
        end function

        subroutine c_mio_decimate_result_free(r) &
                bind(c, name="mio_decimate_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_partition(h, nparts, method, imbalance, mode, seed, record_ids, &
                                 ghost_layers, weights_key) &
                bind(c, name="mio_partition") result(r)
            import :: c_ptr, c_char, c_int, c_double
            type(c_ptr), value :: h
            integer(c_int), value :: nparts
            character(kind=c_char), dimension(*), intent(in) :: method
            real(c_double), value :: imbalance
            character(kind=c_char), dimension(*), intent(in) :: mode
            integer(c_int), value :: seed, record_ids, ghost_layers
            character(kind=c_char), dimension(*), intent(in) :: weights_key
            type(c_ptr) :: r
        end function

        function c_mio_partition_result_num_pieces(r) &
                bind(c, name="mio_partition_result_num_pieces") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: n
        end function

        function c_mio_partition_result_take_mesh(r, index) &
                bind(c, name="mio_partition_result_take_mesh") result(m)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            type(c_ptr) :: m
        end function

        subroutine c_mio_partition_result_free(r) &
                bind(c, name="mio_partition_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_partition_labels(h, nparts, method, imbalance, mode, seed, &
                                        weights_key, labels, labels_size) &
                bind(c, name="mio_partition_labels") result(s)
            import :: c_ptr, c_char, c_int, c_int64_t, c_double
            type(c_ptr), value :: h
            integer(c_int), value :: nparts
            character(kind=c_char), dimension(*), intent(in) :: method
            real(c_double), value :: imbalance
            character(kind=c_char), dimension(*), intent(in) :: mode
            integer(c_int), value :: seed
            character(kind=c_char), dimension(*), intent(in) :: weights_key
            integer(c_int64_t), dimension(*), intent(out) :: labels
            integer(c_int64_t), value :: labels_size
            integer(c_int) :: s
        end function

        function c_mio_stats(h, out) bind(c, name="mio_stats") result(s)
            import :: c_ptr, c_int, mio_stats_report
            type(c_ptr), value :: h
            type(mio_stats_report), intent(out) :: out
            integer(c_int) :: s
        end function

        ! -- data operations --
        ! Name lists cross as a C array of char* plus an explicit count; the
        ! `names` argument is a c_ptr to that array (built by c_str_array).

        function c_mio_data_drop(h, location, names, count, ignore_missing) &
                bind(c, name="mio_data_drop") result(m)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: h, names
            integer(c_int), value :: location, ignore_missing
            integer(c_int64_t), value :: count
            type(c_ptr) :: m
        end function

        function c_mio_data_keep(h, location, names, count, ignore_missing) &
                bind(c, name="mio_data_keep") result(m)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: h, names
            integer(c_int), value :: location, ignore_missing
            integer(c_int64_t), value :: count
            type(c_ptr) :: m
        end function

        function c_mio_data_rename(h, location, from_name, to_name) &
                bind(c, name="mio_data_rename") result(m)
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: h
            integer(c_int), value :: location
            character(kind=c_char), dimension(*), intent(in) :: from_name, to_name
            type(c_ptr) :: m
        end function

        function c_mio_data_point_to_cell(h, names, count, suffix) &
                bind(c, name="mio_data_point_to_cell") result(m)
            import :: c_ptr, c_int64_t, c_char
            type(c_ptr), value :: h, names
            integer(c_int64_t), value :: count
            character(kind=c_char), dimension(*), intent(in) :: suffix
            type(c_ptr) :: m
        end function

        function c_mio_data_cell_to_point(h, names, count, weight, suffix) &
                bind(c, name="mio_data_cell_to_point") result(m)
            import :: c_ptr, c_int, c_int64_t, c_char
            type(c_ptr), value :: h, names
            integer(c_int64_t), value :: count
            integer(c_int), value :: weight
            character(kind=c_char), dimension(*), intent(in) :: suffix
            type(c_ptr) :: m
        end function

        function c_mio_data_calc(h, expression, location, output_name, overwrite) &
                bind(c, name="mio_data_calc") result(m)
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: expression, output_name
            integer(c_int), value :: location, overwrite
            type(c_ptr) :: m
        end function

        function c_mio_data_condition(h, location, names, count, mode, lo, hi, scope, &
                                      nan_policy, nan_replacement, suffix) &
                bind(c, name="mio_data_condition") result(m)
            import :: c_ptr, c_int, c_int64_t, c_double, c_char
            type(c_ptr), value :: h, names
            integer(c_int), value :: location, mode, scope, nan_policy
            integer(c_int64_t), value :: count
            real(c_double), value :: lo, hi, nan_replacement
            character(kind=c_char), dimension(*), intent(in) :: suffix
            type(c_ptr) :: m
        end function

        function c_mio_data_info_create(h) bind(c, name="mio_data_info_create") result(r)
            import :: c_ptr
            type(c_ptr), value :: h
            type(c_ptr) :: r
        end function

        function c_mio_data_info_count(r) bind(c, name="mio_data_info_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: n
        end function

        function c_mio_data_info_name(r, index, buf, buflen) &
                bind(c, name="mio_data_info_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_data_info_entry(r, index, out) &
                bind(c, name="mio_data_info_entry") result(s)
            import :: c_ptr, c_int, c_int64_t, mio_data_array_info
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            type(mio_data_array_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_regions_create(h) bind(c, name="mio_regions_create") result(r)
            import :: c_ptr
            type(c_ptr), value :: h
            type(c_ptr) :: r
        end function

        function c_mio_regions_count(r) bind(c, name="mio_regions_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: n
        end function

        function c_mio_regions_name(r, index, buf, buflen) &
                bind(c, name="mio_regions_name") result(n)
            import :: c_ptr, c_int64_t, c_char
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), intent(inout) :: buf(*)
            integer(c_int64_t) :: n
        end function

        function c_mio_regions_info(r, index, out) &
                bind(c, name="mio_regions_info") result(s)
            import :: c_ptr, c_int64_t, c_int, mio_region_info
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            type(mio_region_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_regions_entries(r, index, count) &
                bind(c, name="mio_regions_entries") result(p)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            integer(c_int64_t), intent(out) :: count
            type(c_ptr) :: p
        end function

        subroutine c_mio_regions_free(r) bind(c, name="mio_regions_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_mesh_add_region(h, name, kind, dim, tag, entries, count) &
                bind(c, name="mio_mesh_add_region") result(s)
            import :: c_ptr, c_char, c_int, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), intent(in) :: name(*)
            integer(c_int), value :: kind, dim
            integer(c_int64_t), value :: tag, count
            integer(c_int64_t), intent(in) :: entries(*)
            integer(c_int) :: s
        end function

        subroutine c_mio_data_info_free(r) bind(c, name="mio_data_info_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_data_integrate_create(h, names, count) &
                bind(c, name="mio_data_integrate_create") result(r)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h, names
            integer(c_int64_t), value :: count
            type(c_ptr) :: r
        end function

        function c_mio_data_integrate_count(r) &
                bind(c, name="mio_data_integrate_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t) :: n
        end function

        function c_mio_data_integrate_name(r, index, buf, buflen) &
                bind(c, name="mio_data_integrate_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_data_integrate_entry(r, index, out) &
                bind(c, name="mio_data_integrate_entry") result(s)
            import :: c_ptr, c_int, c_int64_t, mio_field_integral_info
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            type(mio_field_integral_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_data_integrate_component(r, index, comp, total, mean, domain_measure, &
                num_nan) bind(c, name="mio_data_integrate_component") result(s)
            import :: c_ptr, c_int, c_int64_t, c_double
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, comp
            real(c_double), intent(out) :: total, mean, domain_measure
            integer(c_int64_t), intent(out) :: num_nan
            integer(c_int) :: s
        end function

        function c_mio_data_integrate_region_count(r, index) &
                bind(c, name="mio_data_integrate_region_count") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index
            integer(c_int64_t) :: n
        end function

        function c_mio_data_integrate_region_name(r, index, region, buf, buflen) &
                bind(c, name="mio_data_integrate_region_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, region, buflen
            character(kind=c_char), dimension(*), intent(out) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_data_integrate_region_entry(r, index, region, out) &
                bind(c, name="mio_data_integrate_region_entry") result(s)
            import :: c_ptr, c_int, c_int64_t, mio_field_integral_info
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, region
            type(mio_field_integral_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_data_integrate_region_component(r, index, region, comp, total, mean, &
                domain_measure, num_nan) &
                bind(c, name="mio_data_integrate_region_component") result(s)
            import :: c_ptr, c_int, c_int64_t, c_double
            type(c_ptr), value :: r
            integer(c_int64_t), value :: index, region, comp
            real(c_double), intent(out) :: total, mean, domain_measure
            integer(c_int64_t), intent(out) :: num_nan
            integer(c_int) :: s
        end function

        subroutine c_mio_data_integrate_free(r) bind(c, name="mio_data_integrate_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_reorder_result_take_mesh(r) &
                bind(c, name="mio_reorder_result_take_mesh") result(m)
            import :: c_ptr
            type(c_ptr), value :: r
            type(c_ptr) :: m
        end function

        function c_mio_reorder_result_node_perm(r, data, dtype, n) &
                bind(c, name="mio_reorder_result_node_perm") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: r
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int64_t), intent(out) :: n
            integer(c_int) :: s
        end function

        subroutine c_mio_reorder_result_free(r) bind(c, name="mio_reorder_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_compute_bandwidth(h) bind(c, name="mio_compute_bandwidth") result(bw)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: bw
        end function

        function c_mio_meshes_equal(a, b, atol, rtol, unordered, out_equal) &
                bind(c, name="mio_meshes_equal") result(s)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: a, b
            real(c_double), value :: atol, rtol
            integer(c_int), value :: unordered
            integer(c_int), intent(out) :: out_equal
            integer(c_int) :: s
        end function

        function c_mio_diff(a, b, atol, rtol, unordered, out) &
                bind(c, name="mio_diff") result(s)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: a, b
            real(c_double), value :: atol, rtol
            integer(c_int), value :: unordered
            type(c_ptr), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_diff_result_verdict(r) &
                bind(c, name="mio_diff_result_verdict") result(v)
            import :: c_ptr, c_int
            type(c_ptr), value :: r
            integer(c_int) :: v
        end function

        subroutine c_mio_diff_result_free(r) bind(c, name="mio_diff_result_free")
            import :: c_ptr
            type(c_ptr), value :: r
        end subroutine

        function c_mio_sniff_format(path, buf, buflen) &
                bind(c, name="mio_sniff_format") result(n)
            import :: c_char, c_int64_t
            character(kind=c_char), dimension(*), intent(in) :: path
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t), value :: buflen
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_set_points(h, dtype, num_points, dim, xyz) &
                bind(c, name="mio_mesh_set_points") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: h, xyz
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: num_points, dim
            integer(c_int) :: s
        end function

        function c_mio_mesh_add_cell_block(h, cell_type, num_cells, nodes_per_cell, dtype, conn) &
                bind(c, name="mio_mesh_add_cell_block") result(s)
            import :: c_ptr, c_char, c_int, c_int64_t
            type(c_ptr), value :: h, conn
            character(kind=c_char), dimension(*), intent(in) :: cell_type
            integer(c_int64_t), value :: num_cells, nodes_per_cell
            integer(c_int), value :: dtype
            integer(c_int) :: s
        end function

        function c_mio_mesh_add_point_data(h, name, dtype, ndim, shape, data) &
                bind(c, name="mio_mesh_add_point_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h, data
            character(kind=c_char), dimension(*), intent(in) :: name
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: ndim
            integer(c_int64_t), dimension(*), intent(in) :: shape
            integer(c_int) :: s
        end function

        function c_mio_mesh_append_cell_data(h, name, dtype, ndim, shape, data) &
                bind(c, name="mio_mesh_append_cell_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h, data
            character(kind=c_char), dimension(*), intent(in) :: name
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: ndim
            integer(c_int64_t), dimension(*), intent(in) :: shape
            integer(c_int) :: s
        end function

        function c_mio_mesh_add_field_data(h, name, dtype, ndim, shape, data) &
                bind(c, name="mio_mesh_add_field_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h, data
            character(kind=c_char), dimension(*), intent(in) :: name
            integer(c_int), value :: dtype
            integer(c_int32_t), value :: ndim
            integer(c_int64_t), dimension(*), intent(in) :: shape
            integer(c_int) :: s
        end function

        function c_mio_mesh_num_points(h) bind(c, name="mio_mesh_num_points") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_point_dim(h) bind(c, name="mio_mesh_point_dim") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_get_points(h, data, dtype) &
                bind(c, name="mio_mesh_get_points") result(s)
            import :: c_ptr, c_int
            type(c_ptr), value :: h
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int) :: s
        end function

        function c_mio_mesh_num_cell_blocks(h) bind(c, name="mio_mesh_num_cell_blocks") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_cell_block_info(h, block, num_cells, nodes_per_cell, is_ragged) &
                bind(c, name="mio_mesh_cell_block_info") result(s)
            import :: c_ptr, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: block
            integer(c_int64_t), intent(out) :: num_cells, nodes_per_cell
            integer(c_int32_t), intent(out) :: is_ragged
            integer(c_int) :: s
        end function

        function c_mio_mesh_cell_block_info_ex(h, block, out) &
                bind(c, name="mio_mesh_cell_block_info_ex") result(s)
            import :: c_ptr, c_int, c_int64_t, c_mio_cell_block_info
            type(c_ptr), value :: h
            integer(c_int64_t), value :: block
            type(c_mio_cell_block_info), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_mesh_add_polygon_block(h, ctype, num_cells, row_offsets, nodes, &
                                              num_nodes) &
                bind(c, name="mio_mesh_add_polygon_block") result(s)
            import :: c_ptr, c_char, c_int, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: ctype
            integer(c_int64_t), value :: num_cells, num_nodes
            integer(c_int64_t), dimension(*), intent(in) :: row_offsets, nodes
            integer(c_int) :: s
        end function

        function c_mio_mesh_add_polyhedron_block(h, ctype, num_cells, cell_offsets, num_faces, &
                                                 face_offsets, nodes, num_nodes) &
                bind(c, name="mio_mesh_add_polyhedron_block") result(s)
            import :: c_ptr, c_char, c_int, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: ctype
            integer(c_int64_t), value :: num_cells, num_faces, num_nodes
            integer(c_int64_t), dimension(*), intent(in) :: cell_offsets, face_offsets, nodes
            integer(c_int) :: s
        end function

        function c_mio_poly_conn_create(h, block) bind(c, name="mio_poly_conn_create") result(p)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: block
            type(c_ptr) :: p
        end function

        function c_mio_poly_conn_get_shape(p, out) &
                bind(c, name="mio_poly_conn_get_shape") result(s)
            import :: c_ptr, c_int, c_mio_poly_conn_shape
            type(c_ptr), value :: p
            type(c_mio_poly_conn_shape), intent(out) :: out
            integer(c_int) :: s
        end function

        function c_mio_poly_conn_nodes(p, count) bind(c, name="mio_poly_conn_nodes") result(q)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: p
            integer(c_int64_t), intent(out) :: count
            type(c_ptr) :: q
        end function

        function c_mio_poly_conn_face_offsets(p, count) &
                bind(c, name="mio_poly_conn_face_offsets") result(q)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: p
            integer(c_int64_t), intent(out) :: count
            type(c_ptr) :: q
        end function

        function c_mio_poly_conn_cell_offsets(p, count) &
                bind(c, name="mio_poly_conn_cell_offsets") result(q)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: p
            integer(c_int64_t), intent(out) :: count
            type(c_ptr) :: q
        end function

        subroutine c_mio_poly_conn_free(p) bind(c, name="mio_poly_conn_free")
            import :: c_ptr
            type(c_ptr), value :: p
        end subroutine

        function c_mio_mesh_cell_block_type(h, block, buf, buflen) &
                bind(c, name="mio_mesh_cell_block_type") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: block, buflen
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_cell_block_conn(h, block, conn, dtype) &
                bind(c, name="mio_mesh_cell_block_conn") result(s)
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: block
            type(c_ptr), intent(out) :: conn
            integer(c_int), intent(out) :: dtype
            integer(c_int) :: s
        end function

        function c_mio_mesh_num_point_data(h) bind(c, name="mio_mesh_num_point_data") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_point_data_name(h, index, buf, buflen) &
                bind(c, name="mio_mesh_point_data_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_get_point_data(h, name, data, dtype, ndim, shape) &
                bind(c, name="mio_mesh_get_point_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: name
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int32_t), intent(out) :: ndim
            integer(c_int64_t), dimension(*), intent(out) :: shape
            integer(c_int) :: s
        end function

        function c_mio_mesh_num_cell_data(h) bind(c, name="mio_mesh_num_cell_data") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_cell_data_name(h, index, buf, buflen) &
                bind(c, name="mio_mesh_cell_data_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_cell_data_num_blocks(h, name) &
                bind(c, name="mio_mesh_cell_data_num_blocks") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: name
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_get_cell_data(h, name, block, data, dtype, ndim, shape) &
                bind(c, name="mio_mesh_get_cell_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: name
            integer(c_int64_t), value :: block
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int32_t), intent(out) :: ndim
            integer(c_int64_t), dimension(*), intent(out) :: shape
            integer(c_int) :: s
        end function

        function c_mio_mesh_num_field_data(h) bind(c, name="mio_mesh_num_field_data") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_field_data_name(h, index, buf, buflen) &
                bind(c, name="mio_mesh_field_data_name") result(n)
            import :: c_ptr, c_char, c_int64_t
            type(c_ptr), value :: h
            integer(c_int64_t), value :: index, buflen
            character(kind=c_char), dimension(*), intent(inout) :: buf
            integer(c_int64_t) :: n
        end function

        function c_mio_mesh_get_field_data(h, name, data, dtype, ndim, shape) &
                bind(c, name="mio_mesh_get_field_data") result(s)
            import :: c_ptr, c_char, c_int, c_int32_t, c_int64_t
            type(c_ptr), value :: h
            character(kind=c_char), dimension(*), intent(in) :: name
            type(c_ptr), intent(out) :: data
            integer(c_int), intent(out) :: dtype
            integer(c_int32_t), intent(out) :: ndim
            integer(c_int64_t), dimension(*), intent(out) :: shape
            integer(c_int) :: s
        end function

        ! -- transient XDMF series --

        function c_mio_xdmf_series_create(path, data_format, gzip_level) &
                bind(c, name="mio_xdmf_series_create") result(h)
            import :: c_ptr, c_char, c_int32_t
            character(kind=c_char), dimension(*), intent(in) :: path, data_format
            integer(c_int32_t), value :: gzip_level
            type(c_ptr) :: h
        end function

        function c_mio_xdmf_series_write_points_cells(s, mesh) &
                bind(c, name="mio_xdmf_series_write_points_cells") result(st)
            import :: c_ptr, c_int
            type(c_ptr), value :: s, mesh
            integer(c_int) :: st
        end function

        function c_mio_xdmf_series_write_data(s, time, mesh) &
                bind(c, name="mio_xdmf_series_write_data") result(st)
            import :: c_ptr, c_int, c_double
            type(c_ptr), value :: s, mesh
            real(c_double), value :: time
            integer(c_int) :: st
        end function

        function c_mio_xdmf_series_finalize(s) &
                bind(c, name="mio_xdmf_series_finalize") result(st)
            import :: c_ptr, c_int
            type(c_ptr), value :: s
            integer(c_int) :: st
        end function

        function c_mio_xdmf_series_flush(s) &
                bind(c, name="mio_xdmf_series_flush") result(st)
            import :: c_ptr, c_int
            type(c_ptr), value :: s
            integer(c_int) :: st
        end function

        function c_mio_xdmf_series_finalized(s) &
                bind(c, name="mio_xdmf_series_finalized") result(f)
            import :: c_ptr, c_int32_t
            type(c_ptr), value :: s
            integer(c_int32_t) :: f
        end function

        function c_mio_xdmf_series_create_ex(path, opts) &
                bind(c, name="mio_xdmf_series_create_ex") result(h)
            import :: c_ptr, c_char, mio_xdmf_series_opts_t
            character(kind=c_char), dimension(*), intent(in) :: path
            type(mio_xdmf_series_opts_t), intent(in) :: opts
            type(c_ptr) :: h
        end function

        function c_mio_xdmf_series_num_steps(s) &
                bind(c, name="mio_xdmf_series_num_steps") result(n)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: s
            integer(c_int64_t) :: n
        end function

        subroutine c_mio_xdmf_series_free(s) bind(c, name="mio_xdmf_series_free")
            import :: c_ptr
            type(c_ptr), value :: s
        end subroutine

        function c_strlen(s) bind(c, name="strlen") result(n)
            import :: c_ptr, c_size_t
            type(c_ptr), value :: s
            integer(c_size_t) :: n
        end function

        ! Provenance (see doc/provenance.md). A scope cannot cross the ABI as
        ! RAII, so the C surface is a begin/end pair; Fortran keeps that shape
        ! explicitly, matching `mio_mesh`'s own explicit-free convention.
        function c_mio_provenance_scope_begin(mode) &
            bind(c, name="mio_provenance_scope_begin") result(r)
            import :: c_int
            integer(c_int), value :: mode
            integer(c_int) :: r
        end function

        subroutine c_mio_provenance_scope_end() bind(c, name="mio_provenance_scope_end")
        end subroutine

        subroutine c_mio_provenance_note(cat, det) bind(c, name="mio_provenance_note")
            import :: c_char
            character(kind=c_char), dimension(*), intent(in) :: cat, det
        end subroutine

        subroutine c_mio_provenance_set_source(path, fmt) &
            bind(c, name="mio_provenance_set_source")
            import :: c_char
            character(kind=c_char), dimension(*), intent(in) :: path, fmt
        end subroutine

        subroutine c_mio_provenance_set_target(fmt, enc, codec, ff) &
            bind(c, name="mio_provenance_set_target")
            import :: c_char
            character(kind=c_char), dimension(*), intent(in) :: fmt, enc, codec, ff
        end subroutine
    end interface

contains

    ! ------------------------------------------------------------------
    ! String / error helpers
    ! ------------------------------------------------------------------

    !> NUL-terminated copy for passing to C ("" stays "" + NUL, meaning
    !> "infer format" on the C side).
    pure function c_str(f) result(c)
        character(*), intent(in) :: f
        character(kind=c_char, len=:), allocatable :: c
        c = trim(f)//c_null_char
    end function

    !> Fortran string from a NUL-terminated C pointer (static storage).
    function c_ptr_to_string(p) result(s)
        type(c_ptr), intent(in) :: p
        character(:), allocatable :: s
        character(kind=c_char), pointer :: chars(:)
        integer :: n, i
        s = ''
        if (.not. c_associated(p)) return
        n = int(c_strlen(p))
        if (n <= 0) return
        call c_f_pointer(p, chars, [n])
        s = repeat(' ', n)  ! reallocation on assignment (s starts as '')
        do i = 1, n
            s(i:i) = chars(i)
        end do
    end function

    !> Build the `char**` a C name-list parameter expects.
    !>
    !> `storage` holds NUL-terminated copies of every name (one column each) and
    !> `cptrs` holds a c_ptr to each column. BOTH must stay in scope for the
    !> duration of the C call -- they are the actual backing memory, so the
    !> caller keeps them as local variables until the call returns.
    !> `names` may be zero-sized, in which case the returned pointer is NULL and
    !> the count is 0 ("every array at that location").
    subroutine c_str_array(names, storage, cptrs, arr, count)
        character(*), intent(in) :: names(:)
        character(kind=c_char), allocatable, target, intent(out) :: storage(:, :)
        type(c_ptr), allocatable, target, intent(out) :: cptrs(:)
        type(c_ptr), intent(out) :: arr
        integer(c_int64_t), intent(out) :: count
        integer :: i, j, n, width
        n = size(names)
        count = int(n, c_int64_t)
        if (n == 0) then
            allocate (storage(1, 1))
            allocate (cptrs(1))
            arr = c_null_ptr
            return
        end if
        width = len(names) + 1
        allocate (storage(width, n))
        allocate (cptrs(n))
        do i = 1, n
            do j = 1, len_trim(names(i))
                storage(j, i) = names(i) (j:j)
            end do
            storage(len_trim(names(i)) + 1, i) = c_null_char
            cptrs(i) = c_loc(storage(1, i))
        end do
        arr = c_loc(cptrs(1))
    end subroutine

    !> Fortran string from the first `n` chars of a C char buffer.
    function from_c_buf(buf, n) result(s)
        character(kind=c_char), intent(in) :: buf(*)
        integer, intent(in) :: n
        character(:), allocatable :: s
        integer :: i
        allocate (character(max(n, 0)) :: s)
        do i = 1, n
            s(i:i) = buf(i)
        end do
    end function

    !> The failure message of the most recent failed meshio++ call on this
    !> thread ('' if none).
    function mio_error_message() result(msg)
        character(:), allocatable :: msg
        msg = c_ptr_to_string(c_mio_last_error())
    end function

    !> Map a C status to the optional stat/errmsg pair; no stat + failure =
    !> print and error stop.
    subroutine handle_status(status, what, stat, errmsg)
        integer(c_int), intent(in) :: status
        character(*), intent(in) :: what
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (present(stat)) stat = int(status)
        if (present(errmsg)) errmsg = ''
        if (status /= 0_c_int) then
            if (present(errmsg)) errmsg = mio_error_message()
            if (.not. present(stat)) then
                write (error_unit, '(a)') 'meshio++ ('//what//'): '//mio_error_message()
                error stop 1
            end if
        end if
    end subroutine

    !> Report a Fortran-side failure (bad handle, shape mismatch, ...)
    !> through the same stat/errmsg protocol.
    subroutine handle_failure(what, msg, stat, errmsg)
        character(*), intent(in) :: what, msg
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (present(stat)) stat = 1
        if (present(errmsg)) errmsg = msg
        if (.not. present(stat)) then
            write (error_unit, '(a)') 'meshio++ ('//what//'): '//msg
            error stop 1
        end if
    end subroutine

    subroutine clear_status(stat, errmsg)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (present(stat)) stat = 0
        if (present(errmsg)) errmsg = ''
    end subroutine

    ! ------------------------------------------------------------------
    ! Module-level procedures
    ! ------------------------------------------------------------------

    !> The meshio++ version string, e.g. "6.1.0".
    function mio_version() result(v)
        character(:), allocatable :: v
        v = c_ptr_to_string(c_mio_version())
    end function

    !> The compile-time mesh backend: "meshio", "native", or "kratos".
    function mio_mesh_backend() result(b)
        character(:), allocatable :: b
        b = c_ptr_to_string(c_mio_mesh_backend())
    end function

    !> .true. if `format` (e.g. "gmsh", "vtu", "med") is readable in this build.
    function mio_format_readable(format) result(r)
        character(*), intent(in) :: format
        logical :: r
        r = c_mio_format_readable(c_str(format)) /= 0_c_int
    end function

    !> .true. if `format` is writable in this build.
    function mio_format_writable(format) result(r)
        character(*), intent(in) :: format
        logical :: r
        r = c_mio_format_writable(c_str(format)) /= 0_c_int
    end function

    !> Read `in_path` and immediately write it to `out_path` (the CLI's
    !> `convert`). Formats are inferred from the extensions unless given.
    subroutine mio_convert(in_path, out_path, in_format, out_format, stat, errmsg)
        character(*), intent(in) :: in_path, out_path
        character(*), intent(in), optional :: in_format, out_format
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: ifmt, ofmt
        ifmt = ''; if (present(in_format)) ifmt = in_format
        ofmt = ''; if (present(out_format)) ofmt = out_format
        call handle_status(c_mio_convert(c_str(in_path), c_str(ifmt), c_str(out_path), &
                                         c_str(ofmt)), 'convert', stat, errmsg)
    end subroutine

    !> Run a whole settings.json pipeline (read -> operation chain -> write;
    !> PascalCase vocabulary, see doc/pipeline.md). Needs a build with the
    !> JSON parser (-DMESHIOPLUSPLUS_WITH_JSON=ON); otherwise the error names
    !> the flag. mio_pipeline_has_json() reports which build this is.
    subroutine mio_pipeline_run_file(settings_path, stat, errmsg)
        character(*), intent(in) :: settings_path
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call handle_status(c_mio_pipeline_run_file(c_str(settings_path)), &
                           'pipeline_run_file', stat, errmsg)
    end subroutine

    !> Run a settings pipeline given as JSON text (see mio_pipeline_run_file).
    subroutine mio_pipeline_run_json(json_text, stat, errmsg)
        character(*), intent(in) :: json_text
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call handle_status(c_mio_pipeline_run_json(c_str(json_text)), &
                           'pipeline_run_json', stat, errmsg)
    end subroutine

    !> Whether this build carries the JSON pipeline parser.
    function mio_pipeline_has_json() result(has)
        logical :: has
        has = c_mio_pipeline_has_json() /= 0
    end function

    !> Guess a mesh file's format from its contents ('' if undetermined).
    function mio_sniff_format(path) result(fmt)
        character(*), intent(in) :: path
        character(:), allocatable :: fmt
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        fmt = ''
        n = c_mio_sniff_format(c_str(path), buf, int(STRBUF_LEN, c_int64_t))
        if (n > 0) fmt = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
    end function

    ! ------------------------------------------------------------------
    ! mio_mesh: lifecycle & file I/O
    ! ------------------------------------------------------------------

    !> Allocate an empty mesh (read() does this implicitly).
    subroutine mesh_create(self, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call mesh_free(self)
        self%handle = c_mio_mesh_create()
        if (.not. c_associated(self%handle)) then
            call handle_failure('create', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end subroutine

    !> Release the mesh and every pointer borrowed from it. Idempotent.
    subroutine mesh_free(self)
        class(mio_mesh), intent(inout) :: self
        if (c_associated(self%handle)) call c_mio_mesh_free(self%handle)
        self%handle = c_null_ptr
    end subroutine

    !> .true. between a successful create()/read() and free().
    logical function mesh_is_valid(self)
        class(mio_mesh), intent(in) :: self
        mesh_is_valid = c_associated(self%handle)
    end function

    !> Read a mesh file, replacing any previous content of this handle.
    !>
    !> `points_only` skips every data array; `arrays` keeps only the named ones
    !> (a zero-sized `arrays` keeps none, which is distinct from omitting the
    !> argument entirely -- that reads everything). Formats without a native
    !> selective path are read whole and filtered, so the result is the same
    !> either way; only the cost differs.
    subroutine mesh_read(self, path, format, points_only, arrays, time_step, lenient, &
                         stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: path
        character(*), intent(in), optional :: format
        logical, intent(in), optional :: points_only
        character(*), intent(in), optional :: arrays(:)
        !> Which step of a multi-step file to read: 0 (default) is the first,
        !> negative counts from the end. Out of range fails, never clamps.
        integer, intent(in), optional :: time_step
        !> Downgrade "this reader cannot represent construct X" to a warning and
        !> a skip. Not "ignore all errors": a malformed file still fails.
        logical, intent(in), optional :: lenient
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: fmt
        type(c_ptr) :: h
        type(mio_read_opts_t) :: opts
        ! NUL-terminated copies must outlive the call, so they are held here
        ! rather than built inline; c_loc needs them contiguous and TARGET.
        character(kind=c_char, len=STRBUF_LEN), allocatable, target :: bufs(:)
        type(c_ptr), allocatable, target :: ptrs(:)
        integer :: i, n

        fmt = ''; if (present(format)) fmt = format

        if (.not. present(points_only) .and. .not. present(arrays) &
            .and. .not. present(time_step) .and. .not. present(lenient)) then
            h = c_mio_read(c_str(path), c_str(fmt))
        else
            call c_mio_read_opts_init(opts)
            if (present(points_only)) then
                if (points_only) opts%points_only = 1
            end if
            if (present(time_step)) opts%time_step = int(time_step, c_int64_t)
            if (present(lenient)) then
                if (lenient) opts%lenient = 1
            end if
            if (present(arrays)) then
                n = size(arrays)
                allocate (bufs(max(n, 1)))
                allocate (ptrs(max(n, 1)))
                do i = 1, n
                    bufs(i) = trim(arrays(i))//c_null_char
                    ptrs(i) = c_loc(bufs(i)(1:1))
                end do
                ! Non-null pointer with count 0 means "no arrays" -- the C side
                ! distinguishes that from a null pointer ("every array").
                opts%arrays = c_loc(ptrs(1))
                opts%num_arrays = int(n, c_int64_t)
            end if
            h = c_mio_read_ex(c_str(path), c_str(fmt), opts)
        end if

        if (.not. c_associated(h)) then
            call handle_failure('read', mio_error_message(), stat, errmsg)
            return
        end if
        call mesh_free(self)
        self%handle = h
        call clear_status(stat, errmsg)
    end subroutine

    !> Summarize a mesh file without loading its heavy arrays.
    function mio_read_metadata(path, format, stat, errmsg) result(meta)
        character(*), intent(in) :: path
        character(*), intent(in), optional :: format
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_metadata) :: meta
        character(:), allocatable :: fmt
        type(c_ptr) :: h
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: nblocks, i, n, ncells, npc, nsteps, nregions
        integer(c_int) :: ragged, s
        type(mio_region_info) :: rinfo

        fmt = ''; if (present(format)) fmt = format
        h = c_mio_read_metadata_create(c_str(path), c_str(fmt))
        if (.not. c_associated(h)) then
            call handle_failure('read_metadata', mio_error_message(), stat, errmsg)
            return
        end if

        meta%num_points = c_mio_read_metadata_num_points(h)
        meta%point_dim = c_mio_read_metadata_point_dim(h)
        meta%num_cells = c_mio_read_metadata_num_cells(h)
        meta%fell_back_to_full_read = (c_mio_read_metadata_fell_back(h) == 1)

        nblocks = c_mio_read_metadata_num_cell_blocks(h)
        allocate (meta%cell_blocks(max(nblocks, 0_c_int64_t)))
        do i = 1, nblocks
            s = c_mio_read_metadata_cell_block(h, i - 1_c_int64_t, ncells, npc, ragged)
            if (s /= 0) cycle
            meta%cell_blocks(i)%num_cells = ncells
            meta%cell_blocks(i)%nodes_per_cell = npc
            meta%cell_blocks(i)%ragged = (ragged /= 0)
            n = c_mio_read_metadata_cell_block_type(h, i - 1_c_int64_t, buf, &
                                                    int(STRBUF_LEN, c_int64_t))
            if (n > 0) meta%cell_blocks(i)%cell_type = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        end do

        call metadata_read_names(h, MIO_DATA_POINT, meta%point_data_names)
        call metadata_read_names(h, MIO_DATA_CELL, meta%cell_data_names)
        call metadata_read_names(h, MIO_DATA_FIELD, meta%field_data_names)

        nsteps = c_mio_read_metadata_num_time_values(h)
        if (nsteps < 0) nsteps = 0
        allocate (meta%time_values(nsteps))
        if (nsteps > 0) n = c_mio_read_metadata_time_values(h, meta%time_values, nsteps)

        nregions = c_mio_read_metadata_num_regions(h)
        if (nregions < 0) nregions = 0
        allocate (meta%regions(nregions))
        do i = 1, nregions
            s = c_mio_read_metadata_region_info(h, i - 1_c_int64_t, rinfo)
            if (s /= 0) cycle
            meta%regions(i)%kind = rinfo%kind
            meta%regions(i)%dim = rinfo%dim
            meta%regions(i)%tag = rinfo%tag
            meta%regions(i)%num_entries = rinfo%num_entries
            n = c_mio_read_metadata_region_name(h, i - 1_c_int64_t, buf, &
                                                int(STRBUF_LEN, c_int64_t))
            if (n > 0) meta%regions(i)%name = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        end do

        call c_mio_read_metadata_free(h)
        call clear_status(stat, errmsg)
    end function

    !> Fill `names` with the data-array names at `location`.
    subroutine metadata_read_names(h, location, names)
        type(c_ptr), intent(in) :: h
        integer(c_int), intent(in) :: location
        character(len=STRBUF_LEN), allocatable, intent(out) :: names(:)
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: count, i, n
        count = c_mio_read_metadata_num_names(h, location)
        if (count < 0) count = 0
        allocate (names(count))
        do i = 1, count
            names(i) = ''
            n = c_mio_read_metadata_name(h, location, i - 1_c_int64_t, buf, &
                                          int(STRBUF_LEN, c_int64_t))
            if (n > 0) names(i) = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        end do
    end subroutine

    !> Write the mesh to a file.
    subroutine mesh_write(self, path, format, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: path
        character(*), intent(in), optional :: format
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: fmt
        fmt = ''; if (present(format)) fmt = format
        call handle_status(c_mio_write(c_str(path), self%handle, c_str(fmt)), 'write', &
                           stat, errmsg)
    end subroutine

    !> Extract the boundary of the mesh's highest-dimension cells as a new mesh
    !> (volume -> faces, surface -> edges).
    function mesh_extract_surface(self, record_parent_ids, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: record_parent_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: rec
        rec = 0
        if (present(record_parent_ids)) then
            if (record_parent_ids) rec = 1
        end if
        out%handle = c_mio_extract_surface(self%handle, rec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('extract_surface', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Extract the boundary skin of a volume mesh as a new surface mesh.
    function mesh_extract_skin(self, linearize, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: linearize
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: lin
        lin = 0
        if (present(linearize)) then
            if (linearize) lin = 1
        end if
        out%handle = c_mio_extract_skin(self%handle, lin)
        if (.not. c_associated(out%handle)) then
            call handle_failure('extract_skin', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> A copy of the mesh with per-cell quality metrics attached as cell_data.
    function mesh_attach_quality(self, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        out%handle = c_mio_attach_quality(self%handle)
        if (.not. c_associated(out%handle)) then
            call handle_failure('attach_quality', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Aggregate quality counts (total / inverted / degenerate cells).
    subroutine mesh_quality_counts(self, num_cells, num_inverted, num_degenerate, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        integer(int64), intent(out), optional :: num_cells, num_inverted, num_degenerate
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t) :: nc, ninv, ndeg
        if (c_mio_quality_counts(self%handle, nc, ninv, ndeg) /= 0_c_int) then
            call handle_failure('quality_counts', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_cells)) num_cells = nc
        if (present(num_inverted)) num_inverted = ninv
        if (present(num_degenerate)) num_degenerate = ndeg
        call clear_status(stat, errmsg)
    end subroutine

    !> Merge two or more meshes into one (concatenate, with optional welding of
    !> coincident nodes within `atol`). `data_policy` is "intersection" (default)
    !> or "fill". point_sets/cell_sets are not carried across the C ABI.
    function mio_merge(meshes, weld, atol, source_tag, data_policy, &
                       drop_duplicate_cells, stat, errmsg) result(out)
        type(mio_mesh), intent(in) :: meshes(:)
        logical, intent(in), optional :: weld, source_tag, drop_duplicate_cells
        real(real64), intent(in), optional :: atol
        character(*), intent(in), optional :: data_policy
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr), allocatable :: handles(:)
        integer(c_int) :: cweld, csrc, cpolicy, cdrop
        real(c_double) :: catol
        integer :: i, n
        n = size(meshes)
        allocate (handles(max(n, 1)))
        do i = 1, n
            handles(i) = meshes(i)%handle
        end do
        cweld = 0
        if (present(weld)) then
            if (weld) cweld = 1
        end if
        csrc = 1
        if (present(source_tag)) then
            if (.not. source_tag) csrc = 0
        end if
        cdrop = 0
        if (present(drop_duplicate_cells)) then
            if (drop_duplicate_cells) cdrop = 1
        end if
        cpolicy = 0
        if (present(data_policy)) then
            if (data_policy == 'fill') cpolicy = 1
        end if
        catol = 1.0e-8_c_double
        if (present(atol)) catol = real(atol, c_double)
        out%handle = c_mio_merge(handles, int(n, c_int64_t), cweld, catol, csrc, cpolicy, cdrop)
        if (.not. c_associated(out%handle)) then
            call handle_failure('merge', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Sample data arrays from `source` onto `target` (cross-mesh field
    !> transfer): source point_data at the target's points, source cell_data by
    !> nearest source-cell centroid regardless of the method. `method` is
    !> 'nearest' (default) or 'barycentric' (linear in a simplexified source;
    !> a target point outside the source domain receives `default_value`
    !> unless `extrapolate`). `arrays` omitted or zero-sized means every source
    !> point_data array; cell_data transfers only when named. `on_conflict` is
    !> 'error' (default), 'overwrite' or 'suffix' (name + '_interp').
    function mio_interpolate(source, target, method, arrays, extrapolate, &
                             default_value, on_conflict, stat, errmsg) result(out)
        type(mio_mesh), intent(in) :: source, target
        character(*), intent(in), optional :: method, on_conflict
        character(*), intent(in), optional :: arrays(:)
        logical, intent(in), optional :: extrapolate
        real(real64), intent(in), optional :: default_value
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        integer(c_int) :: cextrap
        real(c_double) :: cdefault
        character(:), allocatable :: cmethod, cconflict
        cmethod = 'nearest'
        if (present(method)) cmethod = trim(method)
        cconflict = 'error'
        if (present(on_conflict)) cconflict = trim(on_conflict)
        cextrap = 0
        if (present(extrapolate)) then
            if (extrapolate) cextrap = 1
        end if
        cdefault = 0.0_c_double
        if (present(default_value)) cdefault = real(default_value, c_double)
        if (present(arrays)) then
            call c_str_array(arrays, storage, cptrs, arr, count)
        else
            arr = c_null_ptr
            count = 0_c_int64_t
        end if
        out%handle = c_mio_interpolate(source%handle, target%handle, c_str(cmethod), arr, &
                                       count, cextrap, cdefault, c_str(cconflict))
        if (.not. c_associated(out%handle)) then
            call handle_failure('interpolate', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Mass-preserving cross-mesh field transfer: an exact overlap-measure
    !> weighted remap, so sum(target value * target measure) equals
    !> sum(source value * source measure) over the shared region -- the
    !> property mio_interpolate's 'barycentric' mode does not have. Both
    !> meshes are simplexified (accepting ragged/polyhedron blocks for free).
    !> `arrays` omitted or zero-sized means every source point_data AND
    !> cell_data array (one algorithm regardless of location, unlike
    !> mio_interpolate). `on_conflict` is 'error' (default), 'overwrite' or
    !> 'suffix' (name + '_interp'). Output arrays are always Float64.
    function mio_conservative_interpolate(source, target, arrays, default_value, &
                                          on_conflict, stat, errmsg) result(out)
        type(mio_mesh), intent(in) :: source, target
        character(*), intent(in), optional :: on_conflict
        character(*), intent(in), optional :: arrays(:)
        real(real64), intent(in), optional :: default_value
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        real(c_double) :: cdefault
        character(:), allocatable :: cconflict
        cconflict = 'error'
        if (present(on_conflict)) cconflict = trim(on_conflict)
        cdefault = 0.0_c_double
        if (present(default_value)) cdefault = real(default_value, c_double)
        if (present(arrays)) then
            call c_str_array(arrays, storage, cptrs, arr, count)
        else
            arr = c_null_ptr
            count = 0_c_int64_t
        end if
        out%handle = c_mio_conservative_interpolate(source%handle, target%handle, arr, &
                                                     count, cdefault, c_str(cconflict))
        if (.not. c_associated(out%handle)) then
            call handle_failure('conservative_interpolate', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Green-element undo: restore `fine`'s transitional (closure-only) cells
    !> back to their original parent, read verbatim from `coarse` -- a lookup,
    !> not a reconstruction, since refine() never renumbers or prunes points.
    !> `fine` must carry refine:cell_id/refine:parent_id/refine:level (i.e. it
    !> must come from a refine() call with record_hierarchy=.true.,
    !> record_levels=.true.); `coarse` must be the mesh that call was run on.
    !> The optional num_groups_undone/num_cells_removed report how many green
    !> sibling groups were substituted and how many cells that removed.
    function mio_undo_green(coarse, fine, num_groups_undone, num_cells_removed, &
                            stat, errmsg) result(out)
        type(mio_mesh), intent(in) :: coarse, fine
        integer(int64), intent(out), optional :: num_groups_undone, num_cells_removed
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int64_t) :: ngroups, nremoved
        out%handle = c_mio_undo_green(coarse%handle, fine%handle, ngroups, nremoved)
        if (.not. c_associated(out%handle)) then
            call handle_failure('undo_green', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_groups_undone)) num_groups_undone = int(ngroups, int64)
        if (present(num_cells_removed)) num_cells_removed = int(nremoved, int64)
        call clear_status(stat, errmsg)
    end function

    !> Renumber the mesh (method: "rcm", "morton", or "hilbert") as a pure
    !> permutation. Returns the renumbered mesh; the optional `node_perm`
    !> receives the node permutation (1-based old index -> 1-based new index).
    function mesh_reorder(self, method, node_perm, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: method
        integer(int64), allocatable, intent(out), optional :: node_perm(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res, cdata
        integer(c_int) :: s, dt
        integer(c_int64_t) :: nlen
        integer(c_int64_t), pointer :: fp(:)
        res = c_mio_reorder(self%handle, c_str(method))
        if (.not. c_associated(res)) then
            call handle_failure('reorder', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(node_perm)) then
            s = c_mio_reorder_result_node_perm(res, cdata, dt, nlen)
            if (s /= 0_c_int) then
                call c_mio_reorder_result_free(res)
                call handle_failure('reorder', mio_error_message(), stat, errmsg)
                return
            end if
            allocate (node_perm(nlen))
            if (nlen > 0) then
                call c_f_pointer(cdata, fp, [nlen])
                node_perm = int(fp, int64) + 1_int64  ! 0-based new id -> 1-based
            end if
        end if
        out%handle = c_mio_reorder_result_take_mesh(res)
        call c_mio_reorder_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('reorder', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Apply an affine transform to the mesh's point coordinates. `matrix` is a
    !> row-major 4x4 affine matrix flattened to 16 doubles (point p maps to
    !> M * [p, 1]). `rotate_vector_data` (default .false.) rotates vector/tensor
    !> point_data by the transform's linear part. Returns the transformed mesh.
    function mesh_transform(self, matrix, rotate_vector_data, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in) :: matrix(16)
        logical, intent(in), optional :: rotate_vector_data
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        real(c_double) :: cmat(16)
        integer(c_int) :: crot
        cmat = real(matrix, c_double)
        crot = 0
        if (present(rotate_vector_data)) then
            if (rotate_vector_data) crot = 1
        end if
        out%handle = c_mio_transform(self%handle, cmat, crot)
        if (.not. c_associated(out%handle)) then
            call handle_failure('transform', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Clean the mesh in one pass (weld / prune / de-dup). Each step is optional;
    !> defaults: no weld, remove orphans, drop degenerate, drop duplicate cells.
    !> The optional integer out-args receive the removal counts.
    function mesh_clean(self, weld, atol, remove_orphans, drop_degenerate, &
                        drop_duplicate_cells, points_welded, points_removed_orphan, &
                        cells_dropped_degenerate, cells_dropped_duplicate, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: weld, remove_orphans, drop_degenerate, &
                                         drop_duplicate_cells
        real(real64), intent(in), optional :: atol
        integer(int64), intent(out), optional :: points_welded, points_removed_orphan, &
                                                 cells_dropped_degenerate, cells_dropped_duplicate
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cweld, crm, cdeg, cdup
        real(c_double) :: catol
        integer(c_int64_t) :: nweld, norph, ndeg, ndup
        cweld = 0
        if (present(weld)) then
            if (weld) cweld = 1
        end if
        crm = 1
        if (present(remove_orphans)) then
            if (.not. remove_orphans) crm = 0
        end if
        cdeg = 1
        if (present(drop_degenerate)) then
            if (.not. drop_degenerate) cdeg = 0
        end if
        cdup = 1
        if (present(drop_duplicate_cells)) then
            if (.not. drop_duplicate_cells) cdup = 0
        end if
        catol = 1.0e-8_c_double
        if (present(atol)) catol = real(atol, c_double)
        out%handle = c_mio_clean(self%handle, cweld, catol, crm, cdeg, cdup, &
                                 nweld, norph, ndeg, ndup)
        if (.not. c_associated(out%handle)) then
            call handle_failure('clean', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(points_welded)) points_welded = int(nweld, int64)
        if (present(points_removed_orphan)) points_removed_orphan = int(norph, int64)
        if (present(cells_dropped_degenerate)) cells_dropped_degenerate = int(ndeg, int64)
        if (present(cells_dropped_duplicate)) cells_dropped_duplicate = int(ndup, int64)
        call clear_status(stat, errmsg)
    end function

    !> Smooth the mesh's point coordinates, leaving topology and data intact.
    !> `method` is "laplacian" or "taubin"; `iterations` is the pass count (for
    !> taubin one iteration is two passes). A negative `lambda` — the default —
    !> means "this method's own default" (0.5 laplacian, 0.33 taubin). The
    !> optional out-args receive the run summary. The caller pin mask (mFrozen)
    !> is not exposed across the C ABI.
    function mesh_smooth(self, method, iterations, lambda, mu, fix_boundary, preserve_features, &
                         feature_angle, guard_inversion, nodes_moved, max_displacement, &
                         skipped_inversion, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: method
        integer, intent(in) :: iterations
        real(real64), intent(in), optional :: lambda, mu, feature_angle
        logical, intent(in), optional :: fix_boundary, preserve_features, guard_inversion
        integer(int64), intent(out), optional :: nodes_moved, skipped_inversion
        real(real64), intent(out), optional :: max_displacement
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cfixb, cfeat, cguard
        real(c_double) :: clambda, cmu, cangle
        integer(c_int64_t) :: nmoved, nskip
        real(c_double) :: maxdisp
        clambda = -1.0_c_double  ! negative = this method's own default
        if (present(lambda)) clambda = real(lambda, c_double)
        cmu = -0.34_c_double
        if (present(mu)) cmu = real(mu, c_double)
        cangle = 30.0_c_double
        if (present(feature_angle)) cangle = real(feature_angle, c_double)
        cfixb = 1
        if (present(fix_boundary)) then
            if (.not. fix_boundary) cfixb = 0
        end if
        cfeat = 1
        if (present(preserve_features)) then
            if (.not. preserve_features) cfeat = 0
        end if
        cguard = 1
        if (present(guard_inversion)) then
            if (.not. guard_inversion) cguard = 0
        end if
        out%handle = c_mio_smooth(self%handle, c_str(method), int(iterations, c_int), &
                                  clambda, cmu, cfixb, cfeat, cangle, cguard, &
                                  nmoved, maxdisp, nskip)
        if (.not. c_associated(out%handle)) then
            call handle_failure('smooth', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(nodes_moved)) nodes_moved = int(nmoved, int64)
        if (present(max_displacement)) max_displacement = real(maxdisp, real64)
        if (present(skipped_inversion)) skipped_inversion = int(nskip, int64)
        call clear_status(stat, errmsg)
    end function

    !> ODT-remesh a tetrahedral mesh: relocate vertices AND flip connectivity
    !> (2-3/3-2, predicate-free). Tet-only. The point set is invariant, so
    !> point_data and named Point regions carry; cell_data and Cell/Side
    !> regions are dropped. See doc/optimize_volume.md.
    function mesh_optimize_volume(self, max_iterations, relocate, flip, preserve_boundary, &
                                  min_improvement, num_flips, num_23_flips, num_32_flips, &
                                  num_vertices_moved, num_tets, min_quality_before, &
                                  min_quality_after, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(in), optional :: max_iterations
        logical, intent(in), optional :: relocate, flip, preserve_boundary
        real(real64), intent(in), optional :: min_improvement
        integer(int64), intent(out), optional :: num_flips, num_23_flips, num_32_flips, &
                                                 num_vertices_moved, num_tets
        real(real64), intent(out), optional :: min_quality_before, min_quality_after
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cmaxiter, crel, cflip, cpresb
        real(c_double) :: cminimp
        integer(c_int64_t) :: nflips, n23, n32, nmoved, ntets
        real(c_double) :: qbefore, qafter
        cmaxiter = 10
        if (present(max_iterations)) cmaxiter = int(max_iterations, c_int)
        crel = 1
        if (present(relocate)) then
            if (.not. relocate) crel = 0
        end if
        cflip = 1
        if (present(flip)) then
            if (.not. flip) cflip = 0
        end if
        cpresb = 1
        if (present(preserve_boundary)) then
            if (.not. preserve_boundary) cpresb = 0
        end if
        cminimp = 1.0e-6_c_double
        if (present(min_improvement)) cminimp = real(min_improvement, c_double)
        out%handle = c_mio_optimize_volume(self%handle, cmaxiter, crel, cflip, cpresb, cminimp, &
                                           nflips, n23, n32, nmoved, ntets, qbefore, qafter)
        if (.not. c_associated(out%handle)) then
            call handle_failure('optimize_volume', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_flips)) num_flips = int(nflips, int64)
        if (present(num_23_flips)) num_23_flips = int(n23, int64)
        if (present(num_32_flips)) num_32_flips = int(n32, int64)
        if (present(num_vertices_moved)) num_vertices_moved = int(nmoved, int64)
        if (present(num_tets)) num_tets = int(ntets, int64)
        if (present(min_quality_before)) min_quality_before = real(qbefore, real64)
        if (present(min_quality_after)) min_quality_after = real(qafter, real64)
        call clear_status(stat, errmsg)
    end function

    !> Crop the mesh to an axis-aligned bounding box. `lo`/`hi` are the box
    !> corners (3 each). `mode` is "all" (default) or "any". Returns the crop.
    function mesh_crop_bbox(self, lo, hi, mode, record_ids, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in) :: lo(3), hi(3)
        character(*), intent(in), optional :: mode
        logical, intent(in), optional :: record_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cmode, crec
        cmode = 0
        if (present(mode)) then
            if (mode == 'any') cmode = 1
        end if
        crec = 0
        if (present(record_ids)) then
            if (record_ids) crec = 1
        end if
        out%handle = c_mio_crop_bbox(self%handle, real(lo, c_double), real(hi, c_double), &
                                     cmode, crec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('crop_bbox', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Crop the mesh to the half-space (p - point) . normal >= 0. `mode` is
    !> "all" (default) or "any". Returns the crop.
    function mesh_crop_plane(self, point, normal, mode, record_ids, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in) :: point(3), normal(3)
        character(*), intent(in), optional :: mode
        logical, intent(in), optional :: record_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cmode, crec
        cmode = 0
        if (present(mode)) then
            if (mode == 'any') cmode = 1
        end if
        crec = 0
        if (present(record_ids)) then
            if (record_ids) crec = 1
        end if
        out%handle = c_mio_crop_plane(self%handle, real(point, c_double), &
                                      real(normal, c_double), cmode, crec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('crop_plane', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Planar cross-section of the mesh: the intersection with the plane through
    !> `origin` with `normal`, one topological dimension below the cut cells (a
    !> volume mesh -> a triangle/quad surface, a 2D surface -> a line mesh). The
    !> input is simplexified first (marching tetrahedra); shared cut points are
    !> deduped so the section is watertight. `record_parent_ids` (default
    !> .false.) attaches a slice:parent_cell cell_data array.
    function mesh_slice(self, origin, normal, record_parent_ids, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in) :: origin(3), normal(3)
        logical, intent(in), optional :: record_parent_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: crec
        crec = 0
        if (present(record_parent_ids)) then
            if (record_parent_ids) crec = 1
        end if
        out%handle = c_mio_slice(self%handle, real(origin, c_double), &
                                 real(normal, c_double), crec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('slice', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Isosurfaces / contours: the level set of the scalar point_data array
    !> `array` at each of `isovalues`, one topological dimension below the cut
    !> cells (a volume mesh -> a triangle/quad surface, a 2D surface -> a line
    !> mesh). The data-driven sibling of `slice`, sharing its marching-tetrahedra
    !> cutter, so contours are watertight; faces are wound toward increasing
    !> field. `array` must be point_data -- cell_data is piecewise constant and
    !> has no level set (convert it with data_cell_to_point first). All contours
    !> land in the one result, tagged per cell with a Float64 iso:value and an
    !> Int64 iso:index (the ordinal, which is the integer tag `split` needs).
    !> `component` (default -1) picks a component of a multi-component array,
    !> negative meaning the row magnitude. `record_parent_ids` (default .false.)
    !> attaches an iso:parent_cell cell_data array.
    function mesh_isosurface(self, array, isovalues, component, record_parent_ids, &
                             stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        real(real64), intent(in) :: isovalues(:)
        integer, intent(in), optional :: component
        logical, intent(in), optional :: record_parent_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: ccomp, crec
        ccomp = -1_c_int
        if (present(component)) ccomp = int(component, c_int)
        crec = 0
        if (present(record_parent_ids)) then
            if (record_parent_ids) crec = 1
        end if
        out%handle = c_mio_isosurface(self%handle, c_str(array), &
                                      real(isovalues, c_double), &
                                      int(size(isovalues), c_int), ccomp, crec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('isosurface', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Differentiate a point_data field: its gradient, divergence or curl.
    !>
    !> `op` is "gradient" (default), "divergence" or "curl"; `method` is
    !> "green-gauss" (default) or "least-squares"; `location` is "cell"
    !> (default) or "point". The result is named `<array>:<op>` unless `output`
    !> overrides it. A gradient of an nc-component field has 3*nc components
    !> laid out [component][derivative]; divergence gives 1 and curl 3, both
    !> needing a 2- or 3-component field. `component` (default -1 = every
    !> component -- note this is the OPPOSITE of `isosurface`, where negative
    !> means the row magnitude) selects one component of the gradient.
    !>
    !> Cells that cannot be differentiated yield NaN and are reported in
    !> `num_skipped`; least-squares cells with a degenerate neighbourhood fall
    !> back to Green-Gauss and are reported in `num_fallback`.
    function mesh_gradient(self, array, op, method, location, output, component, &
                           overwrite, num_skipped, num_fallback, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        character(*), intent(in), optional :: op, method, location, output
        integer, intent(in), optional :: component
        logical, intent(in), optional :: overwrite
        integer(int64), intent(out), optional :: num_skipped, num_fallback
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: ccomp, cover
        integer(c_int64_t) :: nskip, nfall
        character(:), allocatable :: cop, cmethod, cloc, cout
        cop = ''
        if (present(op)) cop = op
        cmethod = ''
        if (present(method)) cmethod = method
        cloc = 'cell'
        if (present(location)) cloc = location
        cout = ''
        if (present(output)) cout = output
        ccomp = -1_c_int
        if (present(component)) ccomp = int(component, c_int)
        cover = 0
        if (present(overwrite)) then
            if (overwrite) cover = 1
        end if
        nskip = 0
        nfall = 0
        out%handle = c_mio_gradient(self%handle, c_str(array), c_str(cop), c_str(cmethod), &
                                    c_str(cloc), c_str(cout), ccomp, cover, nskip, nfall)
        if (.not. c_associated(out%handle)) then
            call handle_failure('gradient', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_skipped)) num_skipped = int(nskip, int64)
        if (present(num_fallback)) num_fallback = int(nfall, int64)
        call clear_status(stat, errmsg)
    end function

    !> The Hessian (second derivative) of a scalar point_data field --
    !> `gradient`'s companion one order further, for curvature-based adaptive
    !> refinement.
    !>
    !> A composition of TWO `gradient` calls, not a new numerical kernel: the
    !> field is differentiated once (point location), then that (n,3)
    !> gradient is differentiated again with the default "gradient" operator,
    !> producing (n,9) -- the flattened row-major 3x3 Hessian, H(i,j) at
    !> index i*3+j. `method` (default "green-gauss") is forwarded to BOTH
    !> internal passes. `output` overrides the default "<array>:hessian"
    !> name.
    !>
    !> A field that is at most LINEAR has an exactly zero Hessian everywhere
    !> -- the one mesh-shape-independent guarantee. For a genuinely
    !> quadratic field the composition is exact on a structured/symmetric
    !> mesh away from its own boundary and a good, standard, but genuinely
    !> approximate curvature estimate on an irregular mesh (see
    !> doc/hessian.md). Input must have exactly one component -- a vector
    !> field's Hessian is a separate quantity per component.
    !>
    !> Cells that cannot be evaluated yield NaN and are reported in
    !> `num_skipped`; least-squares cells with a degenerate neighbourhood in
    !> either internal pass fall back to Green-Gauss and are reported in
    !> `num_fallback` (summed over both passes).
    function mesh_hessian(self, array, method, location, output, overwrite, num_skipped, &
                          num_fallback, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        character(*), intent(in), optional :: method, location, output
        logical, intent(in), optional :: overwrite
        integer(int64), intent(out), optional :: num_skipped, num_fallback
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cover
        integer(c_int64_t) :: nskip, nfall
        character(:), allocatable :: cmethod, cloc, cout
        cmethod = ''
        if (present(method)) cmethod = method
        cloc = 'cell'
        if (present(location)) cloc = location
        cout = ''
        if (present(output)) cout = output
        cover = 0
        if (present(overwrite)) then
            if (overwrite) cover = 1
        end if
        nskip = 0
        nfall = 0
        out%handle = c_mio_hessian(self%handle, c_str(array), c_str(cmethod), c_str(cloc), &
                                   c_str(cout), cover, nskip, nfall)
        if (.not. c_associated(out%handle)) then
            call handle_failure('hessian', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_skipped)) num_skipped = int(nskip, int64)
        if (present(num_fallback)) num_fallback = int(nfall, int64)
        call clear_status(stat, errmsg)
    end function

    !> Estimate the per-cell recovered-gradient (Zienkiewicz-Zhu) error of a
    !> point_data field, and optionally mark cells for refinement.
    !>
    !> A composition of `gradient` (Green-Gauss, cell location) with the
    !> measure-weighted point<->cell averaging round trip: the indicator is
    !> `sqrt(|measure| * sum((recovered - raw)^2))` per cell, attached as
    !> `output` (default "error:zz"). `marking` is "none" (default), "absolute",
    !> "fraction", or "dorfler"; when not "none" a second Int64 0/1 array
    !> `marked` (default "error:marked") is attached too, so `refine`'s own
    !> `where` selector needs no change at all -- the intended use is
    !> `refine(where="error:marked > 0.5")`. `marking_value`'s meaning depends
    !> on `marking`: an absolute indicator threshold, a fraction in (0, 1] of
    !> cells, or the Doerfler bulk fraction theta in (0, 1].
    !>
    !> Cells that cannot be evaluated read NaN in the indicator array and 0
    !> (never NaN) in the marking array, and are reported in `num_skipped`
    !> (excluded from `global_error` and from `num_marked`).
    function mesh_estimate_error(self, array, method, marking, marking_value, output, &
                                 marked, overwrite, global_error, num_skipped, num_marked, &
                                 stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        character(*), intent(in), optional :: method, marking, output, marked
        real(real64), intent(in), optional :: marking_value
        logical, intent(in), optional :: overwrite
        real(real64), intent(out), optional :: global_error
        integer(int64), intent(out), optional :: num_skipped, num_marked
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cover
        real(c_double) :: cvalue, cerror
        integer(c_int64_t) :: nskip, nmark
        character(:), allocatable :: cmethod, cmarking, cout, cmarked
        cmethod = ''
        if (present(method)) cmethod = method
        cmarking = ''
        if (present(marking)) cmarking = marking
        cvalue = 0.0_c_double
        if (present(marking_value)) cvalue = real(marking_value, c_double)
        cout = ''
        if (present(output)) cout = output
        cmarked = ''
        if (present(marked)) cmarked = marked
        cover = 0
        if (present(overwrite)) then
            if (overwrite) cover = 1
        end if
        cerror = 0.0_c_double
        nskip = 0
        nmark = 0
        out%handle = c_mio_estimate_error(self%handle, c_str(array), c_str(cmethod), &
                                          c_str(cmarking), cvalue, c_str(cout), c_str(cmarked), &
                                          cover, cerror, nskip, nmark)
        if (.not. c_associated(out%handle)) then
            call handle_failure('estimate_error', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(global_error)) global_error = real(cerror, real64)
        if (present(num_skipped)) num_skipped = int(nskip, int64)
        if (present(num_marked)) num_marked = int(nmark, int64)
        call clear_status(stat, errmsg)
    end function

    !> Remesh a surface by approximated centroidal Voronoi diagram (ACVD)
    !> clustering: replace its triangulation with a new, near-uniformly-sized,
    !> well-shaped one at `num_clusters` vertices. Unlike every other
    !> resolution-changing procedure on this type, the output has NEW points
    !> and NEW connectivity with no correspondence to `self` -- point/cell
    !> data and named regions are dropped, field data is carried.
    !>
    !> `metric` is "isotropic" (default; area-weighted centroidal distance),
    !> "quadric" (Garland-Heckbert quadric error, preserves sharp edges and
    !> corners), or "anisotropic" (clusters shaped by a local curvature
    !> tensor -- see `max_anisotropy`). `subdivide` defaults to automatic
    !> (the smallest count reaching `subsample_ratio` items per cluster,
    !> capped at `max_subdivide`); pass 0 to disable subdivision.
    !>
    !> Goes through mio_remesh_ex/mio_remesh_opts rather than the flat
    !> mio_remesh -- the mio_refine_ex precedent, needed because mio_remesh
    !> is a flat C function with no room to grow (it already changed once).
    !> Per-vertex mean and Gaussian curvature of this surface, by the angle
    !> defect (K) and the cotangent Laplace-Beltrami operator (H) -- the
    !> signed distance's companion as a node feature.
    !>
    !> Writes `curvature:mean` and `curvature:gaussian` as point data,
    !> optionally `curvature:area` and the (n, 2) `curvature:principal`.
    !> Geometry, connectivity and existing data are carried through unchanged.
    !>
    !> `total_angle_defect` is the oracle: on a CLOSED surface it is
    !> `2*pi*chi` exactly -- `4*pi` for anything sphere-like -- whatever the
    !> tessellation and whichever `dual_area`, so a value that is not that
    !> means the input is not closed or the result is not sane.
    !>
    !> `H` is orientation-dependent and `K` is not, so check
    !> `inconsistent_pairs` before trusting a sign: a nonzero count means
    !> facets disagree about which side is out. This never repairs its input.
    function mesh_curvature(self, mean, gaussian, dual_area, include_boundary, &
                            record_area, record_principal, region, num_boundary, &
                            num_isolated, num_degenerate, total_angle_defect, &
                            boundary_edges, non_manifold_edges, inconsistent_pairs, &
                            degenerate_triangles, watertight, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: mean, gaussian, include_boundary
        logical, intent(in), optional :: record_area, record_principal
        character(*), intent(in), optional :: dual_area, region
        integer(int64), intent(out), optional :: num_boundary, num_isolated, num_degenerate
        real(real64), intent(out), optional :: total_angle_defect
        integer(int64), intent(out), optional :: boundary_edges, non_manifold_edges
        integer(int64), intent(out), optional :: inconsistent_pairs, degenerate_triangles
        logical, intent(out), optional :: watertight
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_curvature_opts_t) :: opts
        type(mio_curvature_report_t) :: report
        ! NUL-terminated copy must outlive the call, so it is held here rather
        ! than built inline; c_loc needs it contiguous and TARGET -- the
        ! region_buf idiom mesh_refine and mesh_remesh already use.
        character(kind=c_char, len=STRBUF_LEN), target :: region_buf

        call c_mio_curvature_opts_init(opts)
        if (present(mean)) then
            if (.not. mean) opts%mean = 0
        end if
        if (present(gaussian)) then
            if (.not. gaussian) opts%gaussian = 0
        end if
        if (present(dual_area)) then
            if (trim(dual_area) == 'barycentric') then
                opts%dual_area = 1
            else if (trim(dual_area) == 'mixed-voronoi') then
                opts%dual_area = 0
            else
                call handle_failure('curvature', &
                    "meshio++: curvature: unknown dual area '"//trim(dual_area)// &
                    "' (expected 'mixed-voronoi' or 'barycentric')", stat, errmsg)
                return
            end if
        end if
        if (present(include_boundary)) then
            if (include_boundary) opts%include_boundary = 1
        end if
        if (present(record_area)) then
            if (record_area) opts%record_area = 1
        end if
        if (present(record_principal)) then
            if (record_principal) opts%record_principal = 1
        end if
        if (present(region)) then
            region_buf = trim(region)//c_null_char
            opts%region = c_loc(region_buf(1:1))
        end if

        res = c_mio_compute_curvature(self%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('curvature', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_boundary)) num_boundary = int(report%num_boundary, int64)
        if (present(num_isolated)) num_isolated = int(report%num_isolated, int64)
        if (present(num_degenerate)) num_degenerate = int(report%num_degenerate, int64)
        if (present(total_angle_defect)) &
            total_angle_defect = real(report%total_angle_defect, real64)
        if (present(boundary_edges)) &
            boundary_edges = int(report%quality%boundary_edges, int64)
        if (present(non_manifold_edges)) &
            non_manifold_edges = int(report%quality%non_manifold_edges, int64)
        if (present(inconsistent_pairs)) &
            inconsistent_pairs = int(report%quality%inconsistent_pairs, int64)
        if (present(degenerate_triangles)) &
            degenerate_triangles = int(report%quality%degenerate_triangles, int64)
        if (present(watertight)) watertight = (report%quality%watertight /= 0)
        call clear_status(stat, errmsg)
    end function

    !> Repair this surface's orientation, holes and pinched vertices: weld
    !> (opt-in) -> triangulate (blocks 1:1) -> split bowties -> orient by the
    !> topological half-edge rule per component -> fan-fill boundary loops of
    !> at most `max_hole_edges` edges -> orient closed components outward.
    !> Lower-dimensional blocks ride along; fill triangles land in one
    !> trailing `triangle` block. Every counter is an optional out-argument.
    function mesh_repair(self, fix_orientation, orient_outward, fill_holes, &
                         split_non_manifold, max_hole_edges, weld_tolerance, &
                         record_provenance, num_flipped, num_components, &
                         num_holes_detected, num_holes_filled, num_holes_skipped, &
                         num_faces_added, num_points_added, num_vertices_split, &
                         num_oriented_outward, num_unorientable, points_welded, &
                         watertight_before, watertight_after, inconsistent_pairs_after, &
                         boundary_edges_after, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: fix_orientation, orient_outward, fill_holes
        logical, intent(in), optional :: split_non_manifold, record_provenance
        integer(int64), intent(in), optional :: max_hole_edges
        real(real64), intent(in), optional :: weld_tolerance
        integer(int64), intent(out), optional :: num_flipped, num_components
        integer(int64), intent(out), optional :: num_holes_detected, num_holes_filled
        integer(int64), intent(out), optional :: num_holes_skipped, num_faces_added
        integer(int64), intent(out), optional :: num_points_added, num_vertices_split
        integer(int64), intent(out), optional :: num_oriented_outward, num_unorientable
        integer(int64), intent(out), optional :: points_welded
        logical, intent(out), optional :: watertight_before, watertight_after
        integer(int64), intent(out), optional :: inconsistent_pairs_after, boundary_edges_after
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_repair_opts_t) :: opts
        type(mio_repair_report_t) :: report

        call c_mio_repair_opts_init(opts)
        if (present(fix_orientation)) then
            if (.not. fix_orientation) opts%fix_orientation = 0
        end if
        if (present(orient_outward)) then
            if (.not. orient_outward) opts%orient_outward = 0
        end if
        if (present(fill_holes)) then
            if (.not. fill_holes) opts%fill_holes = 0
        end if
        if (present(split_non_manifold)) then
            if (.not. split_non_manifold) opts%split_non_manifold = 0
        end if
        if (present(record_provenance)) then
            if (record_provenance) opts%record_provenance = 1
        end if
        if (present(max_hole_edges)) opts%max_hole_edges = int(max_hole_edges, c_int64_t)
        if (present(weld_tolerance)) opts%weld_tolerance = real(weld_tolerance, c_double)

        res = c_mio_repair(self%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('repair', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_flipped)) num_flipped = int(report%num_flipped, int64)
        if (present(num_components)) num_components = int(report%num_components, int64)
        if (present(num_holes_detected)) num_holes_detected = int(report%num_holes_detected, int64)
        if (present(num_holes_filled)) num_holes_filled = int(report%num_holes_filled, int64)
        if (present(num_holes_skipped)) num_holes_skipped = int(report%num_holes_skipped, int64)
        if (present(num_faces_added)) num_faces_added = int(report%num_faces_added, int64)
        if (present(num_points_added)) num_points_added = int(report%num_points_added, int64)
        if (present(num_vertices_split)) num_vertices_split = int(report%num_vertices_split, int64)
        if (present(num_oriented_outward)) &
            num_oriented_outward = int(report%num_oriented_outward, int64)
        if (present(num_unorientable)) num_unorientable = int(report%num_unorientable, int64)
        if (present(points_welded)) points_welded = int(report%points_welded, int64)
        if (present(watertight_before)) watertight_before = (report%quality_before%watertight /= 0)
        if (present(watertight_after)) watertight_after = (report%quality_after%watertight /= 0)
        if (present(inconsistent_pairs_after)) &
            inconsistent_pairs_after = int(report%quality_after%inconsistent_pairs, int64)
        if (present(boundary_edges_after)) &
            boundary_edges_after = int(report%quality_after%boundary_edges, int64)
        call clear_status(stat, errmsg)
    end function

    !> Sobolev (Helmholtz-filtered) deformation: solve `(M + l^2 K) u = M d`
    !> over this mesh's own P1 operators and move the points by `u`. `array`
    !> names the `(n, dim)` point-data displacement; `fixed_points_array` an
    !> optional point-data array whose nonzero entries pin their point. Every
    !> top-dimensional block must be a linear simplex. Non-convergence is
    !> reported through `converged`, never raised.
    function mesh_sobolev_deform(self, array, length_scale, fixed_points_array, &
                                 fix_boundary, record_filtered, max_iterations, &
                                 tolerance, num_iterations, residual, converged, &
                                 num_fixed, num_isolated, max_displacement, &
                                 stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        real(real64), intent(in) :: length_scale
        character(*), intent(in), optional :: fixed_points_array
        logical, intent(in), optional :: fix_boundary, record_filtered
        integer, intent(in), optional :: max_iterations
        real(real64), intent(in), optional :: tolerance
        integer(int64), intent(out), optional :: num_iterations, num_fixed, num_isolated
        real(real64), intent(out), optional :: residual, max_displacement
        logical, intent(out), optional :: converged
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_sobolev_opts_t) :: opts
        type(mio_sobolev_report_t) :: report
        character(kind=c_char, len=STRBUF_LEN), target :: array_buf, fixed_buf

        call c_mio_sobolev_opts_init(opts)
        array_buf = trim(array)//c_null_char
        opts%array = c_loc(array_buf(1:1))
        opts%length_scale = real(length_scale, c_double)
        if (present(fixed_points_array)) then
            fixed_buf = trim(fixed_points_array)//c_null_char
            opts%fixed_points_array = c_loc(fixed_buf(1:1))
        end if
        if (present(fix_boundary)) then
            if (fix_boundary) opts%fix_boundary = 1
        end if
        if (present(record_filtered)) then
            if (record_filtered) opts%record_filtered = 1
        end if
        if (present(max_iterations)) opts%max_iterations = int(max_iterations, c_int32_t)
        if (present(tolerance)) opts%tolerance = real(tolerance, c_double)

        res = c_mio_sobolev_deform(self%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('sobolev_deform', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_iterations)) num_iterations = int(report%num_iterations, int64)
        if (present(num_fixed)) num_fixed = int(report%num_fixed, int64)
        if (present(num_isolated)) num_isolated = int(report%num_isolated, int64)
        if (present(residual)) residual = real(report%residual, real64)
        if (present(max_displacement)) max_displacement = real(report%max_displacement, real64)
        if (present(converged)) converged = (report%converged /= 0)
        call clear_status(stat, errmsg)
    end function

    !> Project every (selected) point of `mesh` onto the surface of `target`:
    !> `x' = x + w (p + offset n - x)`, one projection, no iteration. Every
    !> point of the source moves whatever cells it carries; only the target
    !> must be a surface. `weights` names a point-data array on the source
    !> (nonzero selects, a float value blends); `normal_weight` is 'angle'
    !> (default) or 'area'. Module-level like `mio_interpolate`: two meshes in.
    function mio_shrinkwrap(mesh, target, offset, max_distance, weights, target_region, &
                            normal_weight, record_distance, record_closest_cell, &
                            num_projected, num_missed, num_skipped, max_displacement, &
                            target_watertight, stat, errmsg) result(out)
        type(mio_mesh), intent(in) :: mesh, target
        real(real64), intent(in), optional :: offset, max_distance
        character(*), intent(in), optional :: weights, target_region, normal_weight
        logical, intent(in), optional :: record_distance, record_closest_cell
        integer(int64), intent(out), optional :: num_projected, num_missed, num_skipped
        real(real64), intent(out), optional :: max_displacement
        logical, intent(out), optional :: target_watertight
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_shrinkwrap_opts_t) :: opts
        type(mio_shrinkwrap_report_t) :: report
        character(kind=c_char, len=STRBUF_LEN), target :: weights_buf, region_buf

        call c_mio_shrinkwrap_opts_init(opts)
        if (present(offset)) opts%offset = real(offset, c_double)
        if (present(max_distance)) opts%max_distance = real(max_distance, c_double)
        if (present(weights)) then
            weights_buf = trim(weights)//c_null_char
            opts%weights = c_loc(weights_buf(1:1))
        end if
        if (present(target_region)) then
            region_buf = trim(target_region)//c_null_char
            opts%target_region = c_loc(region_buf(1:1))
        end if
        if (present(normal_weight)) then
            if (trim(normal_weight) == 'area') then
                opts%normal_weight = 1
            else if (trim(normal_weight) == 'angle') then
                opts%normal_weight = 0
            else
                call handle_failure('shrinkwrap', &
                    "meshio++: shrinkwrap: unknown normal weight '"//trim(normal_weight)// &
                    "' (expected 'angle' or 'area')", stat, errmsg)
                return
            end if
        end if
        if (present(record_distance)) then
            if (record_distance) opts%record_distance = 1
        end if
        if (present(record_closest_cell)) then
            if (record_closest_cell) opts%record_closest_cell = 1
        end if

        res = c_mio_shrinkwrap(mesh%handle, target%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('shrinkwrap', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_projected)) num_projected = int(report%num_projected, int64)
        if (present(num_missed)) num_missed = int(report%num_missed, int64)
        if (present(num_skipped)) num_skipped = int(report%num_skipped, int64)
        if (present(max_displacement)) max_displacement = real(report%max_displacement, real64)
        if (present(target_watertight)) target_watertight = (report%quality%watertight /= 0)
        call clear_status(stat, errmsg)
    end function

    function mesh_remesh(self, num_clusters, subdivide, subsample_ratio, max_subdivide, &
                         max_iterations, max_repair_passes, metric, gradation, &
                         preserve_boundary, max_anisotropy, num_iterations, subdivide_applied, &
                         num_isolated_clusters, num_non_manifold_vertices, stat, errmsg) &
            result(out)
        class(mio_mesh), intent(in) :: self
        integer(int64), intent(in) :: num_clusters
        integer, intent(in), optional :: subdivide, max_subdivide, max_iterations
        integer, intent(in), optional :: max_repair_passes
        real(real64), intent(in), optional :: subsample_ratio
        character(*), intent(in), optional :: metric
        real(real64), intent(in), optional :: gradation
        logical, intent(in), optional :: preserve_boundary
        real(real64), intent(in), optional :: max_anisotropy
        integer(int64), intent(out), optional :: num_iterations, num_isolated_clusters
        integer, intent(out), optional :: subdivide_applied
        integer(int64), intent(out), optional :: num_non_manifold_vertices
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_remesh_opts_t) :: opts
        type(mio_remesh_report_t) :: report
        ! NUL-terminated copy must outlive the call, so it is held here
        ! rather than built inline; c_loc needs it contiguous and TARGET --
        ! the region_buf/array_buf idiom mesh_refine already uses.
        character(kind=c_char, len=STRBUF_LEN), target :: metric_buf

        call c_mio_remesh_opts_init(opts)
        opts%num_clusters = int(num_clusters, c_int64_t)
        if (present(subdivide)) opts%subdivide = int(subdivide, c_int32_t)
        if (present(subsample_ratio)) opts%subsample_ratio = real(subsample_ratio, c_double)
        if (present(max_subdivide)) opts%max_subdivide = int(max_subdivide, c_int32_t)
        if (present(max_iterations)) opts%max_iterations = int(max_iterations, c_int32_t)
        if (present(max_repair_passes)) opts%max_repair_passes = int(max_repair_passes, c_int32_t)
        if (present(metric)) then
            metric_buf = trim(metric)//c_null_char
            opts%metric = c_loc(metric_buf(1:1))
        end if
        if (present(gradation)) opts%gradation = real(gradation, c_double)
        if (present(preserve_boundary)) then
            if (.not. preserve_boundary) opts%preserve_boundary = 0
        end if
        if (present(max_anisotropy)) opts%max_anisotropy = real(max_anisotropy, c_double)

        res = c_mio_remesh_ex(self%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('remesh', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_iterations)) num_iterations = int(report%num_iterations, int64)
        if (present(subdivide_applied)) subdivide_applied = int(report%subdivide_applied)
        if (present(num_isolated_clusters)) &
            num_isolated_clusters = int(report%num_isolated_clusters, int64)
        if (present(num_non_manifold_vertices)) &
            num_non_manifold_vertices = int(report%num_non_manifold_vertices, int64)
        call clear_status(stat, errmsg)
    end function

    !> Retetrahedralize this mesh (a volume mesh, or a closed surface) at a
    !> caller-chosen resolution by isosurface stuffing -- remesh's volumetric
    !> sibling. Unlike `remesh`, `self` may be a VOLUME mesh directly (its
    !> boundary is extracted internally) as well as a closed surface. Same
    !> no-correspondence-with-`self` output contract: new points, new
    !> connectivity, point/cell data and named regions dropped, field data
    !> carried.
    !>
    !> Exactly one of `resolution`/`cell_size` must be given. `warp_fraction`
    !> (default 0.35) trades a small, measured chance of non-manifold
    !> boundary edges (reported in `num_non_manifold_edges`) for
    !> substantially better boundary tet quality; 0 disables warping. See
    !> doc/remesh_volume.md for the measured tradeoff.
    function mesh_remesh_volume(self, resolution, cell_size, bounds, padding, &
                                padding_relative, max_cells, max_tets, warp_fraction, sign, &
                                watertight_check, num_vertices_warped, num_tets_rejected, &
                                num_non_manifold_edges, quality, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(in), optional :: resolution(3)
        real(real64), intent(in), optional :: cell_size, bounds(6)
        real(real64), intent(in), optional :: padding, padding_relative, warp_fraction
        integer(int64), intent(in), optional :: max_cells, max_tets
        character(*), intent(in), optional :: sign, watertight_check
        integer(int64), intent(out), optional :: num_vertices_warped, num_tets_rejected
        integer(int64), intent(out), optional :: num_non_manifold_edges
        type(mio_surface_quality), intent(out), optional :: quality
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        type(mio_remesh_volume_opts_t) :: opts
        type(mio_remesh_volume_report_t) :: report
        ! The buffers the option pointers reference must outlive the call,
        ! the mesh_compute_sdf idiom above.
        integer(c_int64_t), target :: res_buf(3)
        real(c_double), target :: bounds_buf(6)
        integer(c_int32_t) :: code

        call c_mio_remesh_volume_opts_init(opts)
        if (present(resolution)) then
            res_buf = int(resolution, c_int64_t)
            opts%resolution = c_loc(res_buf(1))
        end if
        if (present(cell_size)) opts%cell_size = real(cell_size, c_double)
        if (present(bounds)) then
            bounds_buf = real(bounds, c_double)
            opts%bounds = c_loc(bounds_buf(1))
        end if
        if (present(padding)) opts%padding = real(padding, c_double)
        if (present(padding_relative)) opts%padding_relative = real(padding_relative, c_double)
        if (present(max_cells)) opts%max_cells = int(max_cells, c_int64_t)
        if (present(max_tets)) opts%max_tets = int(max_tets, c_int64_t)
        if (present(warp_fraction)) opts%warp_fraction = real(warp_fraction, c_double)
        if (present(sign)) then
            code = sdf_sign_code(sign)
            if (code < 0) then
                call handle_failure('remesh_volume', 'unknown sign '//trim(sign), stat, errmsg)
                return
            end if
            opts%distance%sign = code
        end if
        if (present(watertight_check)) then
            select case (trim(watertight_check))
            case ('off'); opts%distance%watertight_check = 0
            case ('', 'warn'); opts%distance%watertight_check = 1
            case ('error'); opts%distance%watertight_check = 2
            case default
                call handle_failure('remesh_volume', &
                                    'unknown watertight check '//trim(watertight_check), &
                                    stat, errmsg)
                return
            end select
        end if

        res = c_mio_remesh_volume_ex(self%handle, opts, report)
        if (.not. c_associated(res)) then
            call handle_failure('remesh_volume', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = res
        if (present(num_vertices_warped)) &
            num_vertices_warped = int(report%num_vertices_warped, int64)
        if (present(num_tets_rejected)) num_tets_rejected = int(report%num_tets_rejected, int64)
        if (present(num_non_manifold_edges)) &
            num_non_manifold_edges = int(report%num_non_manifold_edges, int64)
        if (present(quality)) quality = report%input_quality
        call clear_status(stat, errmsg)
    end function

    !> Convert the element representation of the mesh. `mode` is "linearize"
    !> (higher-order cells -> their linear base, pruning the orphaned nodes),
    !> "simplexify" (decompose into same-dimension simplices), or "elevate"
    !> (linear -> serendipity quadratic, adding one node per unique edge).
    !> `record_parent_ids` (default .false.) attaches a convert:parent_cell
    !> cell_data array. The optional `point_map` receives the input-point ->
    !> output-point mapping, 1-based, or 0 where the point was pruned.
    function mesh_convert_cells(self, mode, record_parent_ids, point_map, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: mode
        logical, intent(in), optional :: record_parent_ids
        integer(int64), allocatable, intent(out), optional :: point_map(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res, cdata
        integer(c_int) :: crec, s, dt
        integer(c_int64_t) :: nlen
        integer(c_int64_t), pointer :: fp(:)
        crec = 0
        if (present(record_parent_ids)) then
            if (record_parent_ids) crec = 1
        end if
        res = c_mio_convert_cells(self%handle, c_str(mode), crec)
        if (.not. c_associated(res)) then
            call handle_failure('convert_cells', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(point_map)) then
            s = c_mio_convert_cells_result_point_map(res, cdata, dt, nlen)
            if (s /= 0_c_int) then
                call c_mio_convert_cells_result_free(res)
                call handle_failure('convert_cells', mio_error_message(), stat, errmsg)
                return
            end if
            allocate (point_map(nlen))
            if (nlen > 0) then
                call c_f_pointer(cdata, fp, [nlen])
                point_map = int(fp, int64) + 1_int64  ! 0-based (-1 = pruned) -> 1-based (0 = pruned)
            end if
        end if
        out%handle = c_mio_convert_cells_result_take_mesh(res)
        call c_mio_convert_cells_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('convert_cells', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Polyhedrally refine the mesh: one polyhedral child per face of every
    !> eligible 3D cell, connected to a new interior point. Needs no per-type
    !> template table -- tabulated types (reduced to corners for a quadratic
    !> variant) and existing polyhedron blocks are handled uniformly.
    !> Automatically conforming, unlike `refine`. Non-3D blocks and the
    !> full-Lagrange family (no face table) pass through unchanged.
    !> `record_parent_ids` (default .false.) attaches a
    !> subdivide:parent_cell cell_data array. Unlike `convert_cells`, there is
    !> no point map to request -- subdivide never prunes or renumbers an
    !> original point.
    function mesh_subdivide(self, record_parent_ids, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        logical, intent(in), optional :: record_parent_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        integer(c_int) :: crec
        crec = 0
        if (present(record_parent_ids)) then
            if (record_parent_ids) crec = 1
        end if
        res = c_mio_subdivide(self%handle, crec)
        if (.not. c_associated(res)) then
            call handle_failure('subdivide', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = c_mio_subdivide_result_take_mesh(res)
        call c_mio_subdivide_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('subdivide', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Polyhedrally coarsen the mesh: merge groups of cells into single
    !> larger polyhedral cells via greedy seed-and-grow over the shared-face
    !> dual, absorbing face-adjacent neighbours until a group reaches
    !> `target_group_size` (default 8; a short group at a mesh boundary or
    !> pocket is expected, not an error). Non-volume blocks pass through
    !> unchanged; points are never pruned or renumbered (`clean` with
    !> `remove_orphans=.true.` is the follow-up for a minimal point set).
    !> `target_group_size=1` groups every cell by itself.
    function mesh_agglomerate(self, target_group_size, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer(int64), intent(in), optional :: target_group_size
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res
        integer(c_int64_t) :: tgs
        tgs = 8_c_int64_t
        if (present(target_group_size)) tgs = int(target_group_size, c_int64_t)
        res = c_mio_agglomerate(self%handle, tgs)
        if (.not. c_associated(res)) then
            call handle_failure('agglomerate', mio_error_message(), stat, errmsg)
            return
        end if
        out%handle = c_mio_agglomerate_result_take_mesh(res)
        call c_mio_agglomerate_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('agglomerate', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Uniformly refine the mesh, subdividing every cell into same-type children
    !> (line -> 2, triangle -> 4, quad -> 4, tetra -> 8, wedge -> 8,
    !> hexahedron -> 8). New nodes sit at edge / quad-face / body midpoints and
    !> are shared between neighbouring cells, so the result has no hanging
    !> nodes. `levels` (default 1) applies the templates repeatedly; 0 or less
    !> returns an unchanged copy. `record_parent_ids` (default .false.) attaches
    !> a refine:parent_cell cell_data array naming each output cell's ORIGINAL
    !> ancestor. The optional `point_map` receives the input-point ->
    !> output-point mapping, 1-based (the identity: refinement never prunes).
    !> Higher-order cells, pyramids and ragged blocks have no same-type
    !> subdivision and fail.
    !> The `mio_refine_compare` code for an operator spelling, or -1 if unknown.
    function refine_compare_code(op) result(code)
        character(*), intent(in) :: op
        integer(c_int32_t) :: code
        select case (trim(op))
        case ('<', 'lt'); code = 0
        case ('<=', 'le'); code = 1
        case ('>', 'gt'); code = 2
        case ('>=', 'ge'); code = 3
        case ('==', '=', 'eq'); code = 4
        case ('/=', '!=', 'ne'); code = 5
        case default; code = -1
        end select
    end function

    !> The `mio_sdf_sign` code for a sign name, or -1 if unknown.
    function sdf_sign_code(name) result(code)
        character(*), intent(in) :: name
        integer(c_int32_t) :: code
        select case (trim(name))
        case ('unsigned'); code = 0
        case ('', 'pseudonormal'); code = 1
        case ('winding-number', 'winding_number'); code = 2
        case default; code = -1
        end select
    end function

    !> The `mio_voxel_fill` code for a fill name, or -1 if unknown.
    function voxel_fill_code(name) result(code)
        character(*), intent(in) :: name
        integer(c_int32_t) :: code
        select case (trim(name))
        case ('', 'all'); code = 0
        case ('surface'); code = 1
        case ('inside'); code = 2
        case default; code = -1
        end select
    end function

    !> Build a regular hexahedron lattice from nothing -- the one entry point
    !> here that takes no input mesh. `dims` is the cell count per axis;
    !> `origin` (default 0) and `spacing` (default 1) place it. A zero count on
    !> any axis gives an empty mesh, which is a legal request rather than an
    !> error. `max_cells` (default 20000000) refuses a grid larger than that.
    function mio_grid(dims, origin, spacing, max_cells, stat, errmsg) result(out)
        integer, intent(in) :: dims(3)
        real(real64), intent(in), optional :: origin(3), spacing(3)
        integer(int64), intent(in), optional :: max_cells
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int64_t) :: cdims(3), cmax
        real(c_double) :: corigin(3), cspacing(3)
        cdims = int(dims, c_int64_t)
        corigin = 0.0_c_double
        cspacing = 1.0_c_double
        if (present(origin)) corigin = real(origin, c_double)
        if (present(spacing)) cspacing = real(spacing, c_double)
        cmax = 20000000_c_int64_t
        if (present(max_cells)) cmax = int(max_cells, c_int64_t)
        out%handle = c_mio_grid(cdims, corigin, cspacing, cmax)
        if (.not. c_associated(out%handle)) then
            call handle_failure('grid', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Build a regular grid around the mesh. Exactly one of `resolution` and
    !> `cell_size` must be given. `fill` is 'all' (the whole bounding box),
    !> 'surface' (only cells a triangle passes through) or 'inside' (only cells
    !> whose centre is inside the surface). The lattice geometry comes back
    !> through the optional `dims`/`origin`/`spacing`/`num_occupied` arguments.
    function mesh_voxelize(self, resolution, cell_size, bounds, padding, &
                           padding_relative, fill, sign, watertight_check, &
                           attach_occupancy, max_cells, &
                           dims, origin, spacing, num_occupied, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(in), optional :: resolution(3)
        real(real64), intent(in), optional :: cell_size, bounds(6)
        real(real64), intent(in), optional :: padding, padding_relative
        character(*), intent(in), optional :: fill, sign, watertight_check
        logical, intent(in), optional :: attach_occupancy
        integer(int64), intent(in), optional :: max_cells
        integer(int64), intent(out), optional :: dims(3), num_occupied
        real(real64), intent(out), optional :: origin(3), spacing(3)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(mio_voxel_opts_t) :: opts
        ! The buffers the option pointers reference must outlive the call, so
        ! they are locals here rather than temporaries.
        integer(c_int64_t), target :: res_buf(3)
        real(c_double), target :: bounds_buf(6)
        integer(c_int64_t) :: cdims(3), coccupied
        real(c_double) :: corigin(3), cspacing(3)
        integer(c_int32_t) :: code

        call c_mio_voxel_opts_init(opts)
        if (present(resolution)) then
            res_buf = int(resolution, c_int64_t)
            opts%resolution = c_loc(res_buf(1))
        end if
        if (present(cell_size)) opts%cell_size = real(cell_size, c_double)
        if (present(bounds)) then
            bounds_buf = real(bounds, c_double)
            opts%bounds = c_loc(bounds_buf(1))
        end if
        if (present(padding)) opts%padding = real(padding, c_double)
        if (present(padding_relative)) opts%padding_relative = real(padding_relative, c_double)
        if (present(attach_occupancy)) then
            if (attach_occupancy) opts%attach_occupancy = 1
        end if
        if (present(max_cells)) opts%max_cells = int(max_cells, c_int64_t)
        if (present(fill)) then
            code = voxel_fill_code(fill)
            if (code < 0) then
                call handle_failure('voxelize', 'unknown fill '//trim(fill), stat, errmsg)
                return
            end if
            opts%fill = code
        end if
        if (present(sign)) then
            code = sdf_sign_code(sign)
            if (code < 0) then
                call handle_failure('voxelize', 'unknown sign '//trim(sign), stat, errmsg)
                return
            end if
            opts%sign = code
        end if
        if (present(watertight_check)) then
            select case (trim(watertight_check))
            case ('off'); opts%watertight_check = 0
            case ('', 'warn'); opts%watertight_check = 1
            case ('error'); opts%watertight_check = 2
            case default
                call handle_failure('voxelize', &
                                    'unknown watertight check '//trim(watertight_check), &
                                    stat, errmsg)
                return
            end select
        end if

        out%handle = c_mio_voxelize(self%handle, opts, cdims, corigin, cspacing, coccupied)
        if (.not. c_associated(out%handle)) then
            call handle_failure('voxelize', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(dims)) dims = int(cdims, int64)
        if (present(origin)) origin = real(corigin, real64)
        if (present(spacing)) spacing = real(cspacing, real64)
        if (present(num_occupied)) num_occupied = int(coccupied, int64)
        call clear_status(stat, errmsg)
    end function

    !> Crop to the cells whose value in a scalar cell_data array satisfies a
    !> comparison. `compare` is one of '<', '<=', '>', '>=', '==', '!=' -- the
    !> SAME vocabulary `refine`'s `where_op` uses, evaluated by the same C++
    !> function, so the two cannot drift on the boundary cases. A NON-FINITE
    !> cell value never matches, whatever the comparison.
    !>
    !> There is deliberately no `mode`: `crop_bbox`/`crop_plane` test points and
    !> then need an all/any rule, whereas a cell_data predicate is already one
    !> value per cell and has nothing to reduce.
    !>
    !> Inside/outside a surface composes rather than being its own procedure:
    !> `distance_to_surface(..., location='center')` then this on 'sdf:distance'.
    function mesh_crop_predicate(self, array, compare, value, record_ids, stat, errmsg) &
            result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array
        character(*), intent(in), optional :: compare
        real(real64), intent(in), optional :: value
        logical, intent(in), optional :: record_ids
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: ccmp, crec
        real(c_double) :: cval
        ccmp = 0
        if (present(compare)) then
            ccmp = refine_compare_code(compare)
            if (ccmp < 0) then
                call handle_failure('crop_predicate', 'unknown comparison '//trim(compare), &
                                    stat, errmsg)
                return
            end if
        end if
        cval = 0.0_c_double
        if (present(value)) cval = real(value, c_double)
        crec = 0
        if (present(record_ids)) then
            if (record_ids) crec = 1
        end if
        out%handle = c_mio_crop_predicate(self%handle, c_str(array), ccmp, cval, crec)
        if (.not. c_associated(out%handle)) then
            call handle_failure('crop_predicate', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Generate a grid over this surface and fill it with signed distances.
    !>
    !> `structure` is 'voxel' (a dense lattice, the default) or 'octree' (refined
    !> near the surface, and therefore 1-irregular -- it has hanging nodes; see
    !> `refine`'s 'balanced' closure). `resolution`/`cell_size` size a voxel grid
    !> and are an ERROR with 'octree', whose finest cell is
    !> root_resolution/2**max_depth and is therefore already determined.
    !>
    !> `spacing` reports the FINEST cell size, `dims` the ROOT cell counts. The
    !> generated mesh also carries the numeric `sdf:*` field_data header; no file
    !> format persists arbitrary field_data, so write it as `.vti`, whose
    !> Origin/Spacing/WholeExtent attributes are the same information.
    function mesh_compute_sdf(self, structure, resolution, cell_size, bounds, padding, &
                              padding_relative, root_resolution, max_depth, band_cells, &
                              record_levels, max_cells, sign, weight, location, band, &
                              watertight_check, dims, origin, spacing, depth, num_banded, &
                              quality, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in), optional :: structure, sign, weight, location
        character(*), intent(in), optional :: watertight_check
        integer, intent(in), optional :: resolution(3)
        real(real64), intent(in), optional :: cell_size, bounds(6)
        real(real64), intent(in), optional :: padding, padding_relative, band_cells, band
        integer(int64), intent(in), optional :: root_resolution, max_depth, max_cells
        logical, intent(in), optional :: record_levels
        integer(int64), intent(out), optional :: dims(3), depth, num_banded
        real(real64), intent(out), optional :: origin(3), spacing(3)
        type(mio_surface_quality), intent(out), optional :: quality
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(mio_compute_sdf_opts_t) :: opts
        ! The buffers the option pointers reference must outlive the call, so
        ! they are locals here rather than temporaries.
        integer(c_int64_t), target :: res_buf(3)
        real(c_double), target :: bounds_buf(6)
        integer(c_int64_t) :: cdims(3), cdepth, cbanded
        real(c_double) :: corigin(3), cspacing(3)
        type(mio_surface_quality) :: q
        integer(c_int32_t) :: code

        call c_mio_compute_sdf_opts_init(opts)
        if (present(structure)) then
            select case (trim(structure))
            case ('', 'voxel'); opts%structure = 0
            case ('octree'); opts%structure = 1
            case default
                call handle_failure('compute_sdf', 'unknown structure '//trim(structure), &
                                    stat, errmsg)
                return
            end select
        end if
        if (present(resolution)) then
            res_buf = int(resolution, c_int64_t)
            opts%resolution = c_loc(res_buf(1))
        end if
        if (present(cell_size)) opts%cell_size = real(cell_size, c_double)
        if (present(bounds)) then
            bounds_buf = real(bounds, c_double)
            opts%bounds = c_loc(bounds_buf(1))
        end if
        if (present(padding)) opts%padding = real(padding, c_double)
        if (present(padding_relative)) opts%padding_relative = real(padding_relative, c_double)
        if (present(root_resolution)) opts%root_resolution = int(root_resolution, c_int64_t)
        if (present(max_depth)) opts%max_depth = int(max_depth, c_int64_t)
        if (present(band_cells)) opts%band_cells = real(band_cells, c_double)
        if (present(record_levels)) then
            opts%record_levels = 0
            if (record_levels) opts%record_levels = 1
        end if
        if (present(max_cells)) opts%max_cells = int(max_cells, c_int64_t)
        if (present(band)) opts%distance%band = real(band, c_double)
        if (present(sign)) then
            code = sdf_sign_code(sign)
            if (code < 0) then
                call handle_failure('compute_sdf', 'unknown sign '//trim(sign), stat, errmsg)
                return
            end if
            opts%distance%sign = code
        end if
        if (present(weight)) then
            select case (trim(weight))
            case ('', 'angle'); opts%distance%weight = 0
            case ('area'); opts%distance%weight = 1
            case default
                call handle_failure('compute_sdf', 'unknown weight '//trim(weight), stat, errmsg)
                return
            end select
        end if
        if (present(location)) then
            select case (trim(location))
            case ('', 'corner', 'point'); opts%distance%location = 0
            case ('center', 'centre', 'cell'); opts%distance%location = 1
            case default
                call handle_failure('compute_sdf', 'unknown location '//trim(location), &
                                    stat, errmsg)
                return
            end select
        end if
        if (present(watertight_check)) then
            select case (trim(watertight_check))
            case ('off'); opts%distance%watertight_check = 0
            case ('', 'warn'); opts%distance%watertight_check = 1
            case ('error'); opts%distance%watertight_check = 2
            case default
                call handle_failure('compute_sdf', &
                                    'unknown watertight check '//trim(watertight_check), &
                                    stat, errmsg)
                return
            end select
        end if

        out%handle = c_mio_compute_sdf(self%handle, opts, cdims, corigin, cspacing, &
                                       cdepth, cbanded, q)
        if (.not. c_associated(out%handle)) then
            call handle_failure('compute_sdf', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(dims)) dims = int(cdims, int64)
        if (present(origin)) origin = real(corigin, real64)
        if (present(spacing)) spacing = real(cspacing, real64)
        if (present(depth)) depth = int(cdepth, int64)
        if (present(num_banded)) num_banded = int(cbanded, int64)
        if (present(quality)) quality = q
        call clear_status(stat, errmsg)
    end function

    !> What is wrong with this surface, in numbers: boundary edges,
    !> non-manifold edges, inconsistently wound pairs and degenerate triangles.
    function mesh_watertight_check(self, stat, errmsg) result(q)
        class(mio_mesh), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_surface_quality) :: q
        integer(c_int) :: s
        s = c_mio_surface_watertight_check(self%handle, q)
        if (s /= 0_c_int) then
            call handle_failure('surface_watertight_check', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Signed distances from `points` (shaped (3, n), the column-major memory
    !> the C ABI reads as n rows of three) to this surface. Negative is inside.
    function mesh_sample_distance(self, points, sign, band, watertight_check, &
                                  stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in) :: points(:, :)
        character(*), intent(in), optional :: sign, watertight_check
        real(real64), intent(in), optional :: band
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        real(real64), allocatable :: out(:)
        type(mio_sdf_opts_t) :: opts
        real(c_double), allocatable :: buf(:), cpoints(:)
        integer(c_int) :: s
        integer(c_int32_t) :: code
        integer(int64) :: n

        n = size(points, 2)
        allocate (out(max(n, 1_int64)))
        out = 0.0_real64
        if (size(points, 1) /= 3) then
            call handle_failure('sample_distance', 'points must be shaped (3, n)', stat, errmsg)
            return
        end if

        call c_mio_sdf_opts_init(opts)
        if (present(band)) opts%band = real(band, c_double)
        if (present(sign)) then
            code = sdf_sign_code(sign)
            if (code < 0) then
                call handle_failure('sample_distance', 'unknown sign '//trim(sign), stat, errmsg)
                return
            end if
            opts%sign = code
        end if
        if (present(watertight_check)) then
            select case (trim(watertight_check))
            case ('off'); opts%watertight_check = 0
            case ('', 'warn'); opts%watertight_check = 1
            case ('error'); opts%watertight_check = 2
            case default
                call handle_failure('sample_distance', &
                                    'unknown watertight check '//trim(watertight_check), &
                                    stat, errmsg)
                return
            end select
        end if

        allocate (cpoints(max(3_int64*n, 1_int64)), buf(max(n, 1_int64)))
        cpoints = 0.0_c_double
        buf = 0.0_c_double
        if (n > 0) cpoints(1:3*n) = real(reshape(points, [3*n]), c_double)
        s = c_mio_sample_distance(self%handle, cpoints, int(n, c_int64_t), opts, buf)
        if (s /= 0_c_int) then
            call handle_failure('sample_distance', mio_error_message(), stat, errmsg)
            return
        end if
        if (n > 0) out(1:n) = real(buf(1:n), real64)
        call clear_status(stat, errmsg)
    end function

    !> Attach the signed distance from this mesh's points (or cell centres) to
    !> `surface`, as the `sdf:distance` array.
    function mesh_distance_to_surface(self, surface, sign, location, band, &
                                      record_inside, watertight_check, num_banded, &
                                      quality, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        type(mio_mesh), intent(in) :: surface
        character(*), intent(in), optional :: sign, location, watertight_check
        real(real64), intent(in), optional :: band
        logical, intent(in), optional :: record_inside
        integer(int64), intent(out), optional :: num_banded
        type(mio_surface_quality), intent(out), optional :: quality
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(mio_sdf_opts_t) :: opts
        type(mio_surface_quality) :: q
        integer(c_int64_t) :: banded
        integer(c_int32_t) :: code

        call c_mio_sdf_opts_init(opts)
        if (present(band)) opts%band = real(band, c_double)
        if (present(record_inside)) then
            if (record_inside) opts%record_inside = 1
        end if
        if (present(sign)) then
            code = sdf_sign_code(sign)
            if (code < 0) then
                call handle_failure('distance_to_surface', 'unknown sign '//trim(sign), &
                                    stat, errmsg)
                return
            end if
            opts%sign = code
        end if
        if (present(location)) then
            select case (trim(location))
            case ('', 'corner', 'point'); opts%location = 0
            case ('center', 'centre', 'cell'); opts%location = 1
            case default
                call handle_failure('distance_to_surface', &
                                    'unknown location '//trim(location), stat, errmsg)
                return
            end select
        end if
        if (present(watertight_check)) then
            select case (trim(watertight_check))
            case ('off'); opts%watertight_check = 0
            case ('', 'warn'); opts%watertight_check = 1
            case ('error'); opts%watertight_check = 2
            case default
                call handle_failure('distance_to_surface', &
                                    'unknown watertight check '//trim(watertight_check), &
                                    stat, errmsg)
                return
            end select
        end if

        out%handle = c_mio_distance_to_surface(self%handle, surface%handle, opts, banded, q)
        if (.not. c_associated(out%handle)) then
            call handle_failure('distance_to_surface', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(num_banded)) num_banded = int(banded, int64)
        if (present(quality)) quality = q
        call clear_status(stat, errmsg)
    end function

    !> The `mio_refine_closure` code for a closure name, or -1 if unknown.
    function refine_closure_code(name) result(code)
        character(*), intent(in) :: name
        integer(c_int32_t) :: code
        select case (trim(name))
        case ('', 'redgreen', 'red-green', 'green'); code = 0
        case ('propagate', 'red'); code = 1
        case ('balanced', '2:1'); code = 2
        case default; code = -1
        end select
    end function

    !> One-time ABI layout guard for mio_refine_opts_t, the Julia
    !> `_check_abi_layout()` precedent applied to Fortran. The C header pins
    !> `sizeof(mio_refine_opts) == 112` with a `static_assert`, and Julia
    !> checks its own mirror at load time, but Fortran has no compile-time
    !> equivalent -- without this, a field added to one side and not the
    !> other silently corrupts every call rather than failing loudly. Runs
    !> once per process (a SAVE'd flag), not on every refine() call.
    subroutine check_refine_opts_layout()
        logical, save :: checked = .false.
        type(mio_refine_opts_t) :: probe
        if (checked) return
        checked = .true.
        if (c_sizeof(probe) /= 112_c_size_t) then
            write (error_unit, '(a,i0,a)') &
                'meshio++: mio_refine_opts_t layout mismatch (', c_sizeof(probe), ' bytes)'
            error stop 1
        end if
    end subroutine

    function mesh_refine(self, levels, record_parent_ids, point_map, stat, errmsg, &
                         cells, region, where_array, where_op, where_value, closure, &
                         record_levels, record_hierarchy) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(in), optional :: levels
        logical, intent(in), optional :: record_parent_ids
        integer(int64), allocatable, intent(out), optional :: point_map(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        !> Global (block-major) 1-BASED indices of the cells to refine. Shifted
        !> to the C API's 0-based numbering inside.
        integer(int64), intent(in), optional :: cells(:)
        !> Name of a region to refine. A cell region selects its own cells; a
        !> point region selects every cell with ANY node in it; a side region is
        !> an error. At most one of cells/region/where_array may be given.
        character(*), intent(in), optional :: region
        !> Name of a scalar cell_data array to threshold, with `where_op` one of
        !> '<', '<=', '>', '>=', '==', '!='. A non-finite value never matches.
        character(*), intent(in), optional :: where_array
        character(*), intent(in), optional :: where_op
        real(real64), intent(in), optional :: where_value
        !> 'redgreen' (default, local, conforming), 'propagate' (conforming but
        !> spreads to the whole edge-connected component) or 'balanced' (keeps
        !> the hanging nodes and only enforces 2:1 balance -- NOT conforming).
        character(*), intent(in), optional :: closure
        !> Attach the refine:level cell_data array.
        logical, intent(in), optional :: record_levels
        !> Attach refine:cell_id/refine:parent_id -- the persistent parent/
        !> child hierarchy a multigrid caller resolves across the sequence of
        !> meshes it keeps. An input already carrying refine:cell_id is
        !> updated whatever this says. Also forces refine:entity to be
        !> attached even when the closure leaves no hanging node.
        logical, intent(in), optional :: record_hierarchy
        type(mio_mesh) :: out
        type(c_ptr) :: res, cdata
        integer(c_int) :: s, dt
        integer(c_int64_t) :: nlen
        integer(c_int64_t), pointer :: fp(:)
        type(mio_refine_opts_t) :: opts
        ! NUL-terminated copies must outlive the call, so they are held here
        ! rather than built inline; c_loc needs them contiguous and TARGET.
        character(kind=c_char, len=STRBUF_LEN), target :: region_buf, array_buf
        integer(c_int64_t), allocatable, target :: cell_ids(:)

        call check_refine_opts_layout()
        call c_mio_refine_opts_init(opts)
        if (present(levels)) opts%levels = int(levels, c_int32_t)
        if (present(record_parent_ids)) then
            if (record_parent_ids) opts%record_parent_ids = 1
        end if
        if (present(record_levels)) then
            if (record_levels) opts%record_levels = 1
        end if
        if (present(record_hierarchy)) then
            if (record_hierarchy) opts%record_hierarchy = 1_c_int64_t
        end if
        if (present(cells)) then
            allocate (cell_ids(max(size(cells), 1)))
            cell_ids = 0_c_int64_t
            if (size(cells) > 0) cell_ids(1:size(cells)) = int(cells, c_int64_t) - 1_c_int64_t
            opts%cells = c_loc(cell_ids(1))
            opts%num_cells = int(size(cells), c_int64_t)
        end if
        if (present(region)) then
            region_buf = trim(region)//c_null_char
            opts%region = c_loc(region_buf(1:1))
        end if
        if (present(where_array)) then
            array_buf = trim(where_array)//c_null_char
            opts%predicate_array = c_loc(array_buf(1:1))
        end if
        if (present(where_value)) opts%predicate_value = real(where_value, c_double)
        if (present(where_op)) then
            opts%predicate_op = refine_compare_code(where_op)
            if (opts%predicate_op < 0) then
                call handle_failure('refine', "unknown comparison '"//trim(where_op)//"'", &
                                    stat, errmsg)
                return
            end if
        end if
        if (present(closure)) then
            opts%closure = refine_closure_code(closure)
            if (opts%closure < 0) then
                call handle_failure('refine', "unknown closure '"//trim(closure)//"'", &
                                    stat, errmsg)
                return
            end if
        end if

        res = c_mio_refine_ex(self%handle, opts)
        if (.not. c_associated(res)) then
            call handle_failure('refine', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(point_map)) then
            s = c_mio_refine_result_point_map(res, cdata, dt, nlen)
            if (s /= 0_c_int) then
                call c_mio_refine_result_free(res)
                call handle_failure('refine', mio_error_message(), stat, errmsg)
                return
            end if
            allocate (point_map(nlen))
            if (nlen > 0) then
                call c_f_pointer(cdata, fp, [nlen])
                point_map = int(fp, int64) + 1_int64  ! 0-based -> 1-based
            end if
        end if
        out%handle = c_mio_refine_result_take_mesh(res)
        call c_mio_refine_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('refine', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Decimate a SURFACE mesh by quadric-error-metric edge collapse — the
    !> resolution-reducing inverse of refine. Exactly one of `ratio`,
    !> `target_faces`, `max_error` must be given. The output is all-triangle;
    !> boundary and feature vertices are pinned by default. The optional
    !> `point_map` receives, 1-based, each input point's surviving output index
    !> (0 when the survivor itself was pruned).
    function mesh_decimate(self, ratio, target_faces, max_error, placement, &
                           preserve_boundary, preserve_features, feature_angle, &
                           faces_removed, points_removed, collapses_rejected, &
                           max_error_applied, point_map, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        real(real64), intent(in), optional :: ratio
        integer(int64), intent(in), optional :: target_faces
        real(real64), intent(in), optional :: max_error
        character(*), intent(in), optional :: placement
        logical, intent(in), optional :: preserve_boundary
        logical, intent(in), optional :: preserve_features
        real(real64), intent(in), optional :: feature_angle
        integer(int64), intent(out), optional :: faces_removed
        integer(int64), intent(out), optional :: points_removed
        integer(int64), intent(out), optional :: collapses_rejected
        real(real64), intent(out), optional :: max_error_applied
        integer(int64), allocatable, intent(out), optional :: point_map(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        type(c_ptr) :: res, cdata
        real(c_double) :: cratio, cerror, cangle
        integer(c_int64_t) :: cfaces, nlen
        integer(c_int) :: cpb, cpf, s, dt
        integer(c_int64_t), pointer :: fp(:)
        character(:), allocatable :: cplacement
        cratio = -1.0_c_double
        if (present(ratio)) cratio = real(ratio, c_double)
        cfaces = -1_c_int64_t
        if (present(target_faces)) cfaces = int(target_faces, c_int64_t)
        cerror = -1.0_c_double
        if (present(max_error)) cerror = real(max_error, c_double)
        cplacement = 'optimal'
        if (present(placement)) cplacement = placement
        cpb = 1
        if (present(preserve_boundary)) then
            if (.not. preserve_boundary) cpb = 0
        end if
        cpf = 1
        if (present(preserve_features)) then
            if (.not. preserve_features) cpf = 0
        end if
        cangle = 30.0_c_double
        if (present(feature_angle)) cangle = real(feature_angle, c_double)
        res = c_mio_decimate(self%handle, cratio, cfaces, cerror, c_str(cplacement), &
                             cpb, cpf, cangle)
        if (.not. c_associated(res)) then
            call handle_failure('decimate', mio_error_message(), stat, errmsg)
            return
        end if
        if (present(faces_removed)) faces_removed = int(c_mio_decimate_result_faces_removed(res), int64)
        if (present(points_removed)) points_removed = int(c_mio_decimate_result_points_removed(res), int64)
        if (present(collapses_rejected)) then
            collapses_rejected = int(c_mio_decimate_result_collapses_rejected(res), int64)
        end if
        if (present(max_error_applied)) then
            max_error_applied = real(c_mio_decimate_result_max_error_applied(res), real64)
        end if
        if (present(point_map)) then
            s = c_mio_decimate_result_point_map(res, cdata, dt, nlen)
            if (s /= 0_c_int) then
                call c_mio_decimate_result_free(res)
                call handle_failure('decimate', mio_error_message(), stat, errmsg)
                return
            end if
            allocate (point_map(nlen))
            if (nlen > 0) then
                call c_f_pointer(cdata, fp, [nlen])
                point_map = int(fp, int64) + 1_int64  ! 0-based -> 1-based
            end if
        end if
        out%handle = c_mio_decimate_result_take_mesh(res)
        call c_mio_decimate_result_free(res)
        if (.not. c_associated(out%handle)) then
            call handle_failure('decimate', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Split the mesh into pieces by "type", "component", or "region"/"tag".
    !> Returns an array of meshes (one per piece); the optional `keys` array
    !> receives each piece's key. For "region"/"tag", `tag_name` selects the
    !> integer cell_data to split on (default: first integer cell_data).
    function mesh_split(self, by, tag_name, keys, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: by
        character(*), intent(in), optional :: tag_name
        character(len=STRBUF_LEN), allocatable, intent(out), optional :: keys(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh), allocatable :: out(:)
        type(c_ptr) :: res
        integer(c_int64_t) :: count, i, n
        character(kind=c_char) :: buf(STRBUF_LEN)
        character(len=:), allocatable :: tag
        tag = ''
        if (present(tag_name)) tag = tag_name
        res = c_mio_split(self%handle, c_str(by), c_str(tag))
        if (.not. c_associated(res)) then
            call handle_failure('split', mio_error_message(), stat, errmsg)
            allocate (out(0))
            return
        end if
        count = c_mio_split_result_count(res)
        allocate (out(count))
        if (present(keys)) allocate (keys(count))
        do i = 1, count
            out(i)%handle = c_mio_split_result_take_mesh(res, i - 1_c_int64_t)
            if (present(keys)) then
                n = c_mio_split_result_key(res, i - 1_c_int64_t, buf, int(STRBUF_LEN, c_int64_t))
                keys(i) = ''
                if (n > 0) keys(i) = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
            end if
        end do
        call c_mio_split_result_free(res)
        call clear_status(stat, errmsg)
    end function

    !> Decompose the mesh into exactly `nparts` balanced pieces (the
    !> count-driven complement to `split`). method: "sfc" (Hilbert curve cut,
    !> always available), "kahip" (needs a KaHIP-enabled build; fails by name
    !> otherwise), or "auto" (default: kahip when built, else sfc). Every piece
    !> keeps the input's cell-block structure 1:1, so the pieces recombine into
    !> the input. `weights_key` names a scalar cell_data array of per-cell
    !> weights. `ghost_layers` > 0 grows each piece by that many shared-node
    !> BFS layers of other parts' cells (a halo), tagged `partition:ghost`.
    function mesh_partition(self, nparts, method, imbalance, mode, seed, record_ids, &
                            ghost_layers, weights_key, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: nparts
        character(*), intent(in), optional :: method
        real(real64), intent(in), optional :: imbalance
        character(*), intent(in), optional :: mode
        integer, intent(in), optional :: seed
        logical, intent(in), optional :: record_ids
        integer, intent(in), optional :: ghost_layers
        character(*), intent(in), optional :: weights_key
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh), allocatable :: out(:)
        type(c_ptr) :: res
        integer(c_int64_t) :: count, i
        character(len=:), allocatable :: cmethod, cmode, cweights
        real(c_double) :: cimb
        integer(c_int) :: cseed, crec, cghost
        cmethod = 'auto'
        if (present(method)) cmethod = method
        cimb = 0.03_c_double
        if (present(imbalance)) cimb = real(imbalance, c_double)
        cmode = 'eco'
        if (present(mode)) cmode = mode
        cseed = 0_c_int
        if (present(seed)) cseed = int(seed, c_int)
        crec = 0_c_int
        if (present(record_ids)) then
            if (record_ids) crec = 1_c_int
        end if
        cghost = 0_c_int
        if (present(ghost_layers)) cghost = int(ghost_layers, c_int)
        cweights = ''
        if (present(weights_key)) cweights = weights_key
        res = c_mio_partition(self%handle, int(nparts, c_int), c_str(cmethod), cimb, &
                              c_str(cmode), cseed, crec, cghost, c_str(cweights))
        if (.not. c_associated(res)) then
            call handle_failure('partition', mio_error_message(), stat, errmsg)
            allocate (out(0))
            return
        end if
        count = c_mio_partition_result_num_pieces(res)
        allocate (out(count))
        do i = 1, count
            out(i)%handle = c_mio_partition_result_take_mesh(res, i - 1_c_int64_t)
        end do
        call c_mio_partition_result_free(res)
        call clear_status(stat, errmsg)
    end function

    !> The per-cell part assignment only: a flat block-major Int64 array (the
    !> mesh's cell blocks concatenated in order), values in [0, nparts). The
    !> values are part ids, not indices -- no 1-based shift applies.
    function mesh_partition_labels(self, nparts, method, imbalance, mode, seed, &
                                   weights_key, stat, errmsg) result(labels)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: nparts
        character(*), intent(in), optional :: method
        real(real64), intent(in), optional :: imbalance
        character(*), intent(in), optional :: mode
        integer, intent(in), optional :: seed
        character(*), intent(in), optional :: weights_key
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(int64), allocatable :: labels(:)
        character(len=:), allocatable :: cmethod, cmode, cweights
        real(c_double) :: cimb
        integer(c_int) :: cseed, s
        integer(c_int64_t) :: total, b
        integer(c_int64_t), allocatable :: buf(:)
        cmethod = 'auto'
        if (present(method)) cmethod = method
        cimb = 0.03_c_double
        if (present(imbalance)) cimb = real(imbalance, c_double)
        cmode = 'eco'
        if (present(mode)) cmode = mode
        cseed = 0_c_int
        if (present(seed)) cseed = int(seed, c_int)
        cweights = ''
        if (present(weights_key)) cweights = weights_key
        total = 0_c_int64_t
        do b = 1, self%num_cell_blocks()
            total = total + self%cell_block_num_cells(int(b))
        end do
        allocate (buf(max(total, 1_c_int64_t)))
        s = c_mio_partition_labels(self%handle, int(nparts, c_int), c_str(cmethod), cimb, &
                                   c_str(cmode), cseed, c_str(cweights), buf, total)
        if (s /= 0_c_int) then
            call handle_failure('partition_labels', mio_error_message(), stat, errmsg)
            allocate (labels(0))
            return
        end if
        allocate (labels(total))
        if (total > 0) labels = int(buf(1:total), int64)
        call clear_status(stat, errmsg)
    end function

    !> Geometric statistics of the mesh (bbox / centroid / area / volume /
    !> inverted count). Per-cell-type counts are not carried across the C ABI.
    function mesh_stats(self, stat, errmsg) result(rep)
        class(mio_mesh), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_stats_report) :: rep
        integer(c_int) :: s
        s = c_mio_stats(self%handle, rep)
        if (s /= 0_c_int) then
            call handle_failure('stats', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    ! ------------------------------------------------------------------
    ! Data operations
    !
    ! These act on the mesh's point/cell/field data arrays; the geometry is
    ! never modified. Each returns a NEW mesh which the caller must free.
    ! `location` is one of MIO_DATA_POINT / _CELL / _FIELD.
    ! ------------------------------------------------------------------

    !> Drop the named data arrays at `location`.
    function mesh_data_drop(self, location, names, ignore_missing, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer(c_int), intent(in) :: location
        character(*), intent(in) :: names(:)
        logical, intent(in), optional :: ignore_missing
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        integer(c_int) :: cignore
        cignore = 0
        if (present(ignore_missing)) then
            if (ignore_missing) cignore = 1
        end if
        call c_str_array(names, storage, cptrs, arr, count)
        out%handle = c_mio_data_drop(self%handle, location, arr, count, cignore)
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_drop', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Keep only the named data arrays at `location`, dropping the rest there.
    !> The other two locations are untouched; a zero-sized `names` drops all.
    function mesh_data_keep(self, location, names, ignore_missing, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer(c_int), intent(in) :: location
        character(*), intent(in) :: names(:)
        logical, intent(in), optional :: ignore_missing
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        integer(c_int) :: cignore
        cignore = 0
        if (present(ignore_missing)) then
            if (ignore_missing) cignore = 1
        end if
        call c_str_array(names, storage, cptrs, arr, count)
        out%handle = c_mio_data_keep(self%handle, location, arr, count, cignore)
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_keep', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Rename one data array, preserving its values, dtype and shape.
    function mesh_data_rename(self, location, from_name, to_name, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer(c_int), intent(in) :: location
        character(*), intent(in) :: from_name, to_name
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        out%handle = c_mio_data_rename(self%handle, location, c_str(from_name), c_str(to_name))
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_rename', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Average point_data onto the cells (mean over each cell's own nodes).
    !> The output is always double precision. Omit `names` to convert all.
    function mesh_data_point_to_cell(self, names, suffix, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in), optional :: names(:)
        character(*), intent(in), optional :: suffix
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        character(:), allocatable :: sfx
        character(len=1) :: empty(0)
        sfx = ''
        if (present(suffix)) sfx = suffix
        if (present(names)) then
            call c_str_array(names, storage, cptrs, arr, count)
        else
            call c_str_array(empty, storage, cptrs, arr, count)
        end if
        out%handle = c_mio_data_point_to_cell(self%handle, arr, count, c_str(sfx))
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_point_to_cell', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Average cell_data onto the points. `weight` is MIO_WEIGHT_UNIFORM
    !> (default) or MIO_WEIGHT_MEASURE. Points touched by no cell get NaN.
    function mesh_data_cell_to_point(self, names, weight, suffix, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in), optional :: names(:)
        integer(c_int), intent(in), optional :: weight
        character(*), intent(in), optional :: suffix
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        integer(c_int) :: cweight
        character(:), allocatable :: sfx
        character(len=1) :: empty(0)
        cweight = MIO_WEIGHT_UNIFORM
        if (present(weight)) cweight = weight
        sfx = ''
        if (present(suffix)) sfx = suffix
        if (present(names)) then
            call c_str_array(names, storage, cptrs, arr, count)
        else
            call c_str_array(empty, storage, cptrs, arr, count)
        end if
        out%handle = c_mio_data_cell_to_point(self%handle, arr, count, cweight, c_str(sfx))
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_cell_to_point', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Evaluate an elementwise expression over the arrays at `location` and
    !> store the result there as `output_name`. The grammar accepts + - * /,
    !> unary minus, parentheses, numbers, array names, and the functions abs,
    !> sqrt, min, max and norm -- nothing else is evaluated.
    function mesh_data_calc(self, expression, output_name, location, overwrite, &
                            stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: expression, output_name
        integer(c_int), intent(in), optional :: location
        logical, intent(in), optional :: overwrite
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        integer(c_int) :: cloc, cover
        cloc = MIO_DATA_POINT
        if (present(location)) cloc = location
        cover = 0
        if (present(overwrite)) then
            if (overwrite) cover = 1
        end if
        out%handle = c_mio_data_calc(self%handle, c_str(expression), cloc, &
                                     c_str(output_name), cover)
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_calc', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Condition the values of the selected arrays. `mode` is MIO_COND_CLAMP
    !> (default), _NORMALIZE or _STANDARDIZE; `scope` is MIO_SCOPE_COMPONENT
    !> (default) or _MAGNITUDE. For cell_data the statistics are computed
    !> jointly across all cell blocks.
    function mesh_data_condition(self, location, names, mode, lo, hi, scope, &
                                 nan_policy, nan_replacement, suffix, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        integer(c_int), intent(in) :: location
        character(*), intent(in), optional :: names(:)
        integer(c_int), intent(in), optional :: mode, scope, nan_policy
        real(real64), intent(in), optional :: lo, hi, nan_replacement
        character(*), intent(in), optional :: suffix
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: out
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        integer(c_int) :: cmode, cscope, cnan
        real(c_double) :: clo, chi, crep
        character(:), allocatable :: sfx
        character(len=1) :: empty(0)
        cmode = MIO_COND_CLAMP
        if (present(mode)) cmode = mode
        cscope = MIO_SCOPE_COMPONENT
        if (present(scope)) cscope = scope
        cnan = MIO_NAN_IGNORE
        if (present(nan_policy)) cnan = nan_policy
        clo = 0.0_c_double
        if (present(lo)) clo = real(lo, c_double)
        chi = 1.0_c_double
        if (present(hi)) chi = real(hi, c_double)
        crep = 0.0_c_double
        if (present(nan_replacement)) crep = real(nan_replacement, c_double)
        sfx = ''
        if (present(suffix)) sfx = suffix
        if (present(names)) then
            call c_str_array(names, storage, cptrs, arr, count)
        else
            call c_str_array(empty, storage, cptrs, arr, count)
        end if
        out%handle = c_mio_data_condition(self%handle, location, arr, count, cmode, clo, chi, &
                                          cscope, cnan, crep, c_str(sfx))
        if (.not. c_associated(out%handle)) then
            call handle_failure('data_condition', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Summarize every data array the mesh carries (read-only). Returns one
    !> mio_data_array_info per array; pass `keys` to also receive their names.
    function mesh_data_info(self, keys, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(len=STRBUF_LEN), allocatable, intent(out), optional :: keys(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_data_array_info), allocatable :: out(:)
        type(c_ptr) :: res
        integer(c_int64_t) :: count, i, n
        integer(c_int) :: s
        character(kind=c_char) :: buf(STRBUF_LEN)
        res = c_mio_data_info_create(self%handle)
        if (.not. c_associated(res)) then
            call handle_failure('data_info', mio_error_message(), stat, errmsg)
            allocate (out(0))
            return
        end if
        count = c_mio_data_info_count(res)
        if (count < 0) count = 0
        allocate (out(count))
        if (present(keys)) allocate (keys(count))
        do i = 1, count
            s = c_mio_data_info_entry(res, i - 1_c_int64_t, out(i))
            if (s /= 0_c_int) then
                call c_mio_data_info_free(res)
                call handle_failure('data_info', mio_error_message(), stat, errmsg)
                return
            end if
            if (present(keys)) then
                n = c_mio_data_info_name(res, i - 1_c_int64_t, buf, int(STRBUF_LEN, c_int64_t))
                keys(i) = ''
                if (n > 0) keys(i) = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
            end if
        end do
        call c_mio_data_info_free(res)
        call clear_status(stat, errmsg)
    end function

    !> Cell-measure-weighted total/mean of one or more cell_data arrays over
    !> the whole mesh -- gradient's integration counterpart. `arrays` absent
    !> means every cell_data array (sorted name order); a point_data-only name
    !> fails, naming data_point_to_cell as the fix. Returns one
    !> mio_field_integral_info per array; pass `keys` for their names and
    !> `totals`/`means`/`domain_measures`/`num_nans` for the flat,
    !> array-major, per-component buffers (length sum(out%num_components)).
    !> Per-region breakdown is a separate call: `data_integrate_region`.
    function mesh_data_integrate(self, arrays, keys, totals, means, domain_measures, num_nans, &
            stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in), optional :: arrays(:)
        character(len=STRBUF_LEN), allocatable, intent(out), optional :: keys(:)
        real(real64), allocatable, intent(out), optional :: totals(:), means(:), &
            domain_measures(:)
        integer(int64), allocatable, intent(out), optional :: num_nans(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_field_integral_info), allocatable :: out(:)
        type(c_ptr) :: res
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count, i, n, k, comp, off, total_comp
        integer(c_int) :: s
        character(kind=c_char) :: buf(STRBUF_LEN)
        real(c_double) :: ctotal, cmean, cdomain
        integer(c_int64_t) :: cnan

        if (present(arrays)) then
            call c_str_array(arrays, storage, cptrs, arr, count)
        else
            arr = c_null_ptr
            count = 0_c_int64_t
        end if
        res = c_mio_data_integrate_create(self%handle, arr, count)
        if (.not. c_associated(res)) then
            call handle_failure('data_integrate', mio_error_message(), stat, errmsg)
            allocate (out(0))
            if (present(keys)) allocate (keys(0))
            if (present(totals)) allocate (totals(0))
            if (present(means)) allocate (means(0))
            if (present(domain_measures)) allocate (domain_measures(0))
            if (present(num_nans)) allocate (num_nans(0))
            return
        end if

        n = c_mio_data_integrate_count(res)
        if (n < 0) n = 0
        allocate (out(n))
        if (present(keys)) allocate (keys(n))
        do i = 1, n
            s = c_mio_data_integrate_entry(res, i - 1_c_int64_t, out(i))
            if (s /= 0_c_int) then
                call c_mio_data_integrate_free(res)
                call handle_failure('data_integrate', mio_error_message(), stat, errmsg)
                return
            end if
            if (present(keys)) then
                k = c_mio_data_integrate_name(res, i - 1_c_int64_t, buf, int(STRBUF_LEN, c_int64_t))
                keys(i) = ''
                if (k > 0) keys(i) = from_c_buf(buf, min(int(k), STRBUF_LEN - 1))
            end if
        end do

        if (present(totals) .or. present(means) .or. present(domain_measures) .or. &
                present(num_nans)) then
            total_comp = 0
            do i = 1, n
                total_comp = total_comp + out(i)%num_components
            end do
            if (present(totals)) allocate (totals(total_comp))
            if (present(means)) allocate (means(total_comp))
            if (present(domain_measures)) allocate (domain_measures(total_comp))
            if (present(num_nans)) allocate (num_nans(total_comp))
            off = 0
            do i = 1, n
                do comp = 0, out(i)%num_components - 1
                    s = c_mio_data_integrate_component(res, i - 1_c_int64_t, comp, ctotal, cmean, &
                                                        cdomain, cnan)
                    if (s /= 0_c_int) then
                        call c_mio_data_integrate_free(res)
                        call handle_failure('data_integrate', mio_error_message(), stat, errmsg)
                        return
                    end if
                    off = off + 1
                    if (present(totals)) totals(off) = real(ctotal, real64)
                    if (present(means)) means(off) = real(cmean, real64)
                    if (present(domain_measures)) domain_measures(off) = real(cdomain, real64)
                    if (present(num_nans)) num_nans(off) = int(cnan, int64)
                end do
            end do
        end if

        call c_mio_data_integrate_free(res)
        call clear_status(stat, errmsg)
    end function

    !> Per-named-Cell-region breakdown of one array's field integral (see
    !> `data_integrate`, which this complements): one mio_field_integral_info
    !> per region present on the mesh, plus the same optional flat,
    !> region-major, per-component buffers.
    function mesh_data_integrate_region(self, array_name, keys, totals, means, domain_measures, &
            num_nans, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: array_name
        character(len=STRBUF_LEN), allocatable, intent(out), optional :: keys(:)
        real(real64), allocatable, intent(out), optional :: totals(:), means(:), &
            domain_measures(:)
        integer(int64), allocatable, intent(out), optional :: num_nans(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_field_integral_info), allocatable :: out(:)
        type(c_ptr) :: res
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count, n, k, comp, off, total_comp
        integer(c_int) :: s
        character(kind=c_char) :: buf(STRBUF_LEN)
        real(c_double) :: ctotal, cmean, cdomain
        integer(c_int64_t) :: cnan

        call c_str_array([array_name], storage, cptrs, arr, count)
        res = c_mio_data_integrate_create(self%handle, arr, count)
        if (.not. c_associated(res)) then
            call handle_failure('data_integrate_region', mio_error_message(), stat, errmsg)
            allocate (out(0))
            if (present(keys)) allocate (keys(0))
            if (present(totals)) allocate (totals(0))
            if (present(means)) allocate (means(0))
            if (present(domain_measures)) allocate (domain_measures(0))
            if (present(num_nans)) allocate (num_nans(0))
            return
        end if

        n = c_mio_data_integrate_region_count(res, 0_c_int64_t)
        if (n < 0) n = 0
        allocate (out(n))
        if (present(keys)) allocate (keys(n))
        do k = 1, n
            s = c_mio_data_integrate_region_entry(res, 0_c_int64_t, k - 1_c_int64_t, out(k))
            if (s /= 0_c_int) then
                call c_mio_data_integrate_free(res)
                call handle_failure('data_integrate_region', mio_error_message(), stat, errmsg)
                return
            end if
            if (present(keys)) then
                off = c_mio_data_integrate_region_name(res, 0_c_int64_t, k - 1_c_int64_t, buf, &
                                                        int(STRBUF_LEN, c_int64_t))
                keys(k) = ''
                if (off > 0) keys(k) = from_c_buf(buf, min(int(off), STRBUF_LEN - 1))
            end if
        end do

        if (present(totals) .or. present(means) .or. present(domain_measures) .or. &
                present(num_nans)) then
            total_comp = 0
            do k = 1, n
                total_comp = total_comp + out(k)%num_components
            end do
            if (present(totals)) allocate (totals(total_comp))
            if (present(means)) allocate (means(total_comp))
            if (present(domain_measures)) allocate (domain_measures(total_comp))
            if (present(num_nans)) allocate (num_nans(total_comp))
            off = 0
            do k = 1, n
                do comp = 0, out(k)%num_components - 1
                    s = c_mio_data_integrate_region_component(res, 0_c_int64_t, k - 1_c_int64_t, &
                                                               comp, ctotal, cmean, cdomain, cnan)
                    if (s /= 0_c_int) then
                        call c_mio_data_integrate_free(res)
                        call handle_failure('data_integrate_region', mio_error_message(), stat, &
                                             errmsg)
                        return
                    end if
                    off = off + 1
                    if (present(totals)) totals(off) = real(ctotal, real64)
                    if (present(means)) means(off) = real(cmean, real64)
                    if (present(domain_measures)) domain_measures(off) = real(cdomain, real64)
                    if (present(num_nans)) num_nans(off) = int(cnan, int64)
                end do
            end do
        end if

        call c_mio_data_integrate_free(res)
        call clear_status(stat, errmsg)
    end function

    !> Named regions: a group of points, cells or cell facets carried by a
    !> set-capable format (gmsh physical groups, Abaqus NSET / ELSET / SURFACE,
    !> ...). See doc/regions.md.
    !>
    !> `entries` is the flat int64 buffer, `sum(out%num_entries * out%stride)`
    !> long, with the groups laid out back to back in `out` order. Point and
    !> cell indices are shifted to Fortran's 1-based convention on the way out;
    !> the facet column of a side region is **not** shifted, matching the
    !> `partition_labels` rule that a value which is not an index stays as it is.
    function mesh_regions(self, keys, entries, stat, errmsg) result(out)
        class(mio_mesh), intent(in) :: self
        character(len=STRBUF_LEN), allocatable, intent(out), optional :: keys(:)
        integer(int64), allocatable, intent(out), optional :: entries(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_region_info), allocatable :: out(:)
        type(c_ptr) :: res, p
        integer(c_int64_t) :: count, i, n, nvals, total, off, k
        integer(c_int) :: s
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t), pointer :: vals(:)

        res = c_mio_regions_create(self%handle)
        if (.not. c_associated(res)) then
            call handle_failure('regions', mio_error_message(), stat, errmsg)
            allocate (out(0))
            if (present(keys)) allocate (keys(0))
            if (present(entries)) allocate (entries(0))
            return
        end if
        count = c_mio_regions_count(res)
        if (count < 0) count = 0
        allocate (out(count))
        if (present(keys)) allocate (keys(count))

        total = 0
        do i = 1, count
            s = c_mio_regions_info(res, i - 1_c_int64_t, out(i))
            if (s /= 0_c_int) then
                call c_mio_regions_free(res)
                call handle_failure('regions', mio_error_message(), stat, errmsg)
                return
            end if
            total = total + out(i)%num_entries * out(i)%stride
            if (present(keys)) then
                n = c_mio_regions_name(res, i - 1_c_int64_t, buf, int(STRBUF_LEN, c_int64_t))
                keys(i) = ''
                if (n > 0) keys(i) = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
            end if
        end do

        if (present(entries)) then
            allocate (entries(total))
            off = 0
            do i = 1, count
                p = c_mio_regions_entries(res, i - 1_c_int64_t, nvals)
                if (.not. c_associated(p) .or. nvals <= 0) cycle
                call c_f_pointer(p, vals, [nvals])
                if (out(i)%stride == 2) then
                    ! (cell, facet) pairs: shift only the cell column.
                    do k = 1, nvals, 2
                        entries(off + k) = vals(k) + 1_c_int64_t
                        entries(off + k + 1) = vals(k + 1)
                    end do
                else
                    do k = 1, nvals
                        entries(off + k) = vals(k) + 1_c_int64_t
                    end do
                end if
                off = off + nvals
            end do
        end if

        call c_mio_regions_free(res)
        call clear_status(stat, errmsg)
    end function

    !> Add a named region, replacing any with the same (kind, name, dim, tag).
    !> `entries` is 1-based for point/cell indices; a side region takes flat
    !> (cell, facet) pairs whose facet column is passed through unshifted.
    subroutine mesh_add_region(self, name, kind, entries, dim, tag, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(len=*), intent(in) :: name
        integer(c_int), intent(in) :: kind
        integer(int64), intent(in) :: entries(:)
        integer, intent(in), optional :: dim
        integer(int64), intent(in), optional :: tag
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t), allocatable :: buf(:)
        integer(c_int) :: s, c_dim
        integer(c_int64_t) :: c_tag, k

        c_dim = -1_c_int
        if (present(dim)) c_dim = int(dim, c_int)
        c_tag = -1_c_int64_t
        if (present(tag)) c_tag = int(tag, c_int64_t)

        allocate (buf(size(entries)))
        if (kind == MIO_REGION_SIDE) then
            do k = 1, int(size(entries), c_int64_t), 2
                buf(k) = entries(k) - 1_c_int64_t
                if (k + 1 <= size(entries)) buf(k + 1) = entries(k + 1)
            end do
        else
            do k = 1, int(size(entries), c_int64_t)
                buf(k) = entries(k) - 1_c_int64_t
            end do
        end if

        s = c_mio_mesh_add_region(self%handle, c_str(name), kind, c_dim, c_tag, buf, &
                                  int(size(entries), c_int64_t))
        if (s /= 0_c_int) then
            call handle_failure('add_region', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end subroutine

    !> Connectivity bandwidth: max over cells of (max - min) node index.
    function mesh_compute_bandwidth(self, stat, errmsg) result(bw)
        class(mio_mesh), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(int64) :: bw
        integer(c_int64_t) :: cbw
        cbw = c_mio_compute_bandwidth(self%handle)
        if (cbw < 0_c_int64_t) then
            call handle_failure('compute_bandwidth', mio_error_message(), stat, errmsg)
            bw = -1_int64
            return
        end if
        bw = int(cbw, int64)
        call clear_status(stat, errmsg)
    end function

    !> Are two meshes equal within tolerance (abs_err <= atol + rtol*|expected|)?
    !> `unordered` (default .false.) matches points by spatial proximity. Named
    !> point_sets/cell_sets are not compared.
    function mesh_equals(self, other, atol, rtol, unordered, stat, errmsg) result(eq)
        class(mio_mesh), intent(in) :: self
        type(mio_mesh), intent(in) :: other
        real(real64), intent(in), optional :: atol, rtol
        logical, intent(in), optional :: unordered
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        logical :: eq
        real(c_double) :: catol, crtol
        integer(c_int) :: cuno, ceq, s
        eq = .false.
        catol = 1.0e-12_c_double
        crtol = 1.0e-9_c_double
        cuno = 0_c_int
        if (present(atol)) catol = real(atol, c_double)
        if (present(rtol)) crtol = real(rtol, c_double)
        if (present(unordered)) then
            if (unordered) cuno = 1_c_int
        end if
        s = c_mio_meshes_equal(self%handle, other%handle, catol, crtol, cuno, ceq)
        if (s /= 0_c_int) then
            call handle_failure('equals', mio_error_message(), stat, errmsg)
            return
        end if
        eq = ceq /= 0_c_int
        call clear_status(stat, errmsg)
    end function

    !> Compare two meshes; returns the verdict (0 = identical, 1 = equal within
    !> tolerance, 2 = different). `unordered` matches points by proximity.
    function mesh_diff(self, other, atol, rtol, unordered, stat, errmsg) result(verdict)
        class(mio_mesh), intent(in) :: self
        type(mio_mesh), intent(in) :: other
        real(real64), intent(in), optional :: atol, rtol
        logical, intent(in), optional :: unordered
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer :: verdict
        real(c_double) :: catol, crtol
        integer(c_int) :: cuno, s
        type(c_ptr) :: res
        verdict = 2
        catol = 1.0e-12_c_double
        crtol = 1.0e-9_c_double
        cuno = 0_c_int
        if (present(atol)) catol = real(atol, c_double)
        if (present(rtol)) crtol = real(rtol, c_double)
        if (present(unordered)) then
            if (unordered) cuno = 1_c_int
        end if
        s = c_mio_diff(self%handle, other%handle, catol, crtol, cuno, res)
        if (s /= 0_c_int) then
            call handle_failure('diff', mio_error_message(), stat, errmsg)
            return
        end if
        verdict = int(c_mio_diff_result_verdict(res))
        call c_mio_diff_result_free(res)
        call clear_status(stat, errmsg)
    end function

    ! ------------------------------------------------------------------
    ! mio_mesh: building (setters copy; see module header for layout rules)
    ! ------------------------------------------------------------------

    !> Assign the point coordinates from a `points(dim, num_points)` array.
    subroutine mesh_set_points(self, points, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        real(real64), intent(in), contiguous, target :: points(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        call handle_status(c_mio_mesh_set_points(self%handle, MIO_FLOAT64, &
                                                 int(size(points, 2), c_int64_t), &
                                                 int(size(points, 1), c_int64_t), &
                                                 c_loc(points)), 'set_points', stat, errmsg)
    end subroutine

    !> Append one cell block from a 1-based `conn(nodes_per_cell, num_cells)`
    !> array (default-integer version).
    subroutine mesh_add_cell_block_i32(self, cell_type, conn, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: cell_type
        integer(int32), intent(in) :: conn(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t), allocatable, target :: shifted(:, :)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shifted = int(conn, c_int64_t) - 1_c_int64_t
        call handle_status(c_mio_mesh_add_cell_block(self%handle, c_str(cell_type), &
                                                     int(size(conn, 2), c_int64_t), &
                                                     int(size(conn, 1), c_int64_t), MIO_INT64, &
                                                     c_loc(shifted)), 'add_cell_block', &
                           stat, errmsg)
    end subroutine

    !> As above, for `integer(int64)` connectivity.
    subroutine mesh_add_cell_block_i64(self, cell_type, conn, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: cell_type
        integer(int64), intent(in) :: conn(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t), allocatable, target :: shifted(:, :)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shifted = int(conn, c_int64_t) - 1_c_int64_t
        call handle_status(c_mio_mesh_add_cell_block(self%handle, c_str(cell_type), &
                                                     int(size(conn, 2), c_int64_t), &
                                                     int(size(conn, 1), c_int64_t), MIO_INT64, &
                                                     c_loc(shifted)), 'add_cell_block', &
                           stat, errmsg)
    end subroutine

    !> Append a 1-level ragged (jagged polygon) cell block.
    !>
    !> `row_offsets(num_cells + 1)` and `nodes(:)` are BOTH 1-based on this
    !> side: `nodes(row_offsets(c) : row_offsets(c + 1) - 1)` is cell `c`'s node
    !> list. The offsets are shifted as well as the node ids because they index
    !> a Fortran array -- the same reasoning that makes MED's own INN 1-based.
    subroutine mesh_add_polygon_block(self, cell_type, row_offsets, nodes, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: cell_type
        integer(int64), intent(in) :: row_offsets(:), nodes(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t), allocatable :: c_off(:), c_nodes(:)
        if (size(row_offsets) < 1) then
            call handle_failure('add_polygon_block', &
                                'row_offsets must have num_cells + 1 entries', stat, errmsg)
            return
        end if
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        c_off = int(row_offsets, c_int64_t) - 1_c_int64_t
        c_nodes = int(nodes, c_int64_t) - 1_c_int64_t
        call handle_status(c_mio_mesh_add_polygon_block(self%handle, c_str(cell_type), &
                                                        int(size(row_offsets) - 1, c_int64_t), &
                                                        c_off, c_nodes, &
                                                        int(size(nodes), c_int64_t)), &
                           'add_polygon_block', stat, errmsg)
    end subroutine

    !> Append a 2-level ragged (polyhedron) cell block: each cell is a list of
    !> faces, each face a list of node ids.
    !>
    !> `cell_offsets(num_cells + 1)` indexes the FACE list,
    !> `face_offsets(num_faces + 1)` indexes `nodes`, and all three are 1-based
    !> here, so face `f` of cell `c` is
    !> `nodes(face_offsets(j) : face_offsets(j + 1) - 1)` with
    !> `j = cell_offsets(c) + f - 1`.
    subroutine mesh_add_polyhedron_block(self, cell_type, cell_offsets, face_offsets, nodes, &
                                         stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: cell_type
        integer(int64), intent(in) :: cell_offsets(:), face_offsets(:), nodes(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t), allocatable :: c_cells(:), c_faces(:), c_nodes(:)
        if (size(cell_offsets) < 1 .or. size(face_offsets) < 1) then
            call handle_failure('add_polyhedron_block', &
                                'cell_offsets and face_offsets must each have one more entry '// &
                                'than the count they describe', stat, errmsg)
            return
        end if
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        c_cells = int(cell_offsets, c_int64_t) - 1_c_int64_t
        c_faces = int(face_offsets, c_int64_t) - 1_c_int64_t
        c_nodes = int(nodes, c_int64_t) - 1_c_int64_t
        call handle_status(c_mio_mesh_add_polyhedron_block( &
                           self%handle, c_str(cell_type), &
                           int(size(cell_offsets) - 1, c_int64_t), c_cells, &
                           int(size(face_offsets) - 1, c_int64_t), c_faces, c_nodes, &
                           int(size(nodes), c_int64_t)), 'add_polyhedron_block', stat, errmsg)
    end subroutine

    !> Attach a scalar per-point field: `data(num_points)`.
    subroutine mesh_add_point_data_r1(self, name, data, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: name
        real(real64), intent(in), contiguous, target :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t) :: shape(1)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shape(1) = int(size(data), c_int64_t)
        call handle_status(c_mio_mesh_add_point_data(self%handle, c_str(name), MIO_FLOAT64, &
                                                     1_c_int32_t, shape, c_loc(data)), &
                           'add_point_data', stat, errmsg)
    end subroutine

    !> Attach a vector per-point field: `data(num_components, num_points)`
    !> (same memory as the C API's row-major `(num_points, num_components)`).
    subroutine mesh_add_point_data_r2(self, name, data, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: name
        real(real64), intent(in), contiguous, target :: data(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t) :: shape(2)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shape(1) = int(size(data, 2), c_int64_t)  ! C shape is the reverse of the Fortran one
        shape(2) = int(size(data, 1), c_int64_t)
        call handle_status(c_mio_mesh_add_point_data(self%handle, c_str(name), MIO_FLOAT64, &
                                                     2_c_int32_t, shape, c_loc(data)), &
                           'add_point_data', stat, errmsg)
    end subroutine

    !> Append the named cell-data field's array for the next cell block (call
    !> once per block, in block order): `data(num_cells_in_block)`.
    subroutine mesh_add_cell_data_r1(self, name, data, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: name
        real(real64), intent(in), contiguous, target :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t) :: shape(1)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shape(1) = int(size(data), c_int64_t)
        call handle_status(c_mio_mesh_append_cell_data(self%handle, c_str(name), MIO_FLOAT64, &
                                                       1_c_int32_t, shape, c_loc(data)), &
                           'add_cell_data', stat, errmsg)
    end subroutine

    !> Attach a named mesh-level (field) data array.
    subroutine mesh_add_field_data_r1(self, name, data, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        character(*), intent(in) :: name
        real(real64), intent(in), contiguous, target :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        integer(c_int64_t) :: shape(1)
        call ensure_handle(self, stat, errmsg)
        if (.not. c_associated(self%handle)) return
        shape(1) = int(size(data), c_int64_t)
        call handle_status(c_mio_mesh_add_field_data(self%handle, c_str(name), MIO_FLOAT64, &
                                                     1_c_int32_t, shape, c_loc(data)), &
                           'add_field_data', stat, errmsg)
    end subroutine

    ! ------------------------------------------------------------------
    ! mio_mesh: inspection
    ! ------------------------------------------------------------------

    integer(int64) function mesh_num_points(self)
        class(mio_mesh), intent(in) :: self
        mesh_num_points = c_mio_mesh_num_points(self%handle)
    end function

    integer(int64) function mesh_point_dim(self)
        class(mio_mesh), intent(in) :: self
        mesh_point_dim = c_mio_mesh_point_dim(self%handle)
    end function

    integer(int64) function mesh_num_cell_blocks(self)
        class(mio_mesh), intent(in) :: self
        mesh_num_cell_blocks = c_mio_mesh_num_cell_blocks(self%handle)
    end function

    !> Copy the coordinates into `points(point_dim, num_points)` (allocated
    !> here), converting to real64 if the mesh stores single precision.
    subroutine mesh_get_points(self, points, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        real(real64), allocatable, intent(out) :: points(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: p
        integer(c_int) :: dtype, status
        integer(int64) :: n, dim
        real(c_double), pointer :: d64(:)
        real(c_float), pointer :: d32(:)
        status = c_mio_mesh_get_points(self%handle, p, dtype)
        if (status /= 0_c_int) then
            call handle_status(status, 'get_points', stat, errmsg)
            return
        end if
        n = c_mio_mesh_num_points(self%handle)
        dim = c_mio_mesh_point_dim(self%handle)
        allocate (points(dim, n))
        if (n*dim == 0) then
            call clear_status(stat, errmsg)
            return
        end if
        select case (dtype)
        case (MIO_FLOAT64)
            call c_f_pointer(p, d64, [n*dim])
            points = reshape(d64, [dim, n])
        case (MIO_FLOAT32)
            call c_f_pointer(p, d32, [n*dim])
            points = reshape(real(d32, real64), [dim, n])
        case default
            call handle_failure('get_points', 'unexpected points dtype', stat, errmsg)
            return
        end select
        call clear_status(stat, errmsg)
    end subroutine

    !> Zero-copy borrow of the coordinates as `ptr(point_dim, num_points)`.
    !> Valid until the next mutating call on this mesh or free(); fails if
    !> the mesh stores anything but real64 (then use get_points).
    subroutine mesh_points_ptr(self, ptr, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        real(real64), pointer, intent(out) :: ptr(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: p
        integer(c_int) :: dtype, status
        ptr => null()
        status = c_mio_mesh_get_points(self%handle, p, dtype)
        if (status /= 0_c_int) then
            call handle_status(status, 'points_ptr', stat, errmsg)
            return
        end if
        if (dtype /= MIO_FLOAT64) then
            call handle_failure('points_ptr', 'points are not real64; use get_points', &
                                stat, errmsg)
            return
        end if
        call c_f_pointer(p, ptr, [c_mio_mesh_point_dim(self%handle), &
                                  c_mio_mesh_num_points(self%handle)])
        call clear_status(stat, errmsg)
    end subroutine

    !> The meshio type name of 1-based cell block `block` (e.g. "tetra10").
    function mesh_cell_block_type(self, block, stat, errmsg) result(type_name)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: type_name
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        type_name = ''
        n = c_mio_mesh_cell_block_type(self%handle, int(block - 1, c_int64_t), buf, &
                                       int(STRBUF_LEN, c_int64_t))
        if (n < 0) then
            call handle_failure('cell_block_type', mio_error_message(), stat, errmsg)
            return
        end if
        type_name = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        call clear_status(stat, errmsg)
    end function

    integer(int64) function mesh_cell_block_num_cells(self, block)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(c_int64_t) :: nc, npc
        integer(c_int32_t) :: ragged
        mesh_cell_block_num_cells = -1
        if (c_mio_mesh_cell_block_info(self%handle, int(block - 1, c_int64_t), nc, npc, &
                                       ragged) == 0_c_int) mesh_cell_block_num_cells = nc
    end function

    integer(int64) function mesh_cell_block_nodes_per_cell(self, block)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(c_int64_t) :: nc, npc
        integer(c_int32_t) :: ragged
        mesh_cell_block_nodes_per_cell = -1
        if (c_mio_mesh_cell_block_info(self%handle, int(block - 1, c_int64_t), nc, npc, &
                                       ragged) == 0_c_int) mesh_cell_block_nodes_per_cell = npc
    end function

    logical function mesh_cell_block_is_ragged(self, block)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(c_int64_t) :: nc, npc
        integer(c_int32_t) :: ragged
        mesh_cell_block_is_ragged = .false.
        if (c_mio_mesh_cell_block_info(self%handle, int(block - 1, c_int64_t), nc, npc, &
                                       ragged) == 0_c_int) mesh_cell_block_is_ragged = ragged /= 0
    end function

    !> Whether 1-based cell block `block` is 2-level ragged (a polyhedron block:
    !> each cell a list of faces). Implies `cell_block_is_ragged`.
    logical function mesh_cell_block_is_polyhedron(self, block)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        type(c_mio_cell_block_info) :: info
        mesh_cell_block_is_polyhedron = .false.
        if (c_mio_mesh_cell_block_info_ex(self%handle, int(block - 1, c_int64_t), info) &
            == 0_c_int) mesh_cell_block_is_polyhedron = info%is_polyhedron /= 0
    end function

    !> Copy 1-based ragged cell block `block` into a 1-based CSR pair,
    !> `row_offsets(num_cells + 1)` and `nodes(:)`, both allocated here.
    !> Cell `c`'s node list is `nodes(row_offsets(c) : row_offsets(c + 1) - 1)`.
    !>
    !> Fails on a rectangular block (use `get_cell_block`) and on a polyhedron
    !> block (use `get_polyhedron_block`).
    subroutine mesh_get_polygon_block(self, block, row_offsets, nodes, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(int64), allocatable, intent(out) :: row_offsets(:), nodes(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: poly, p
        type(c_mio_poly_conn_shape) :: shape
        integer(c_int64_t) :: n
        integer(c_int64_t), pointer :: vals(:)

        allocate (row_offsets(0), nodes(0))
        poly = c_mio_poly_conn_create(self%handle, int(block - 1, c_int64_t))
        if (.not. c_associated(poly)) then
            call handle_failure('get_polygon_block', mio_error_message(), stat, errmsg)
            return
        end if
        if (c_mio_poly_conn_get_shape(poly, shape) /= 0_c_int) then
            call c_mio_poly_conn_free(poly)
            call handle_failure('get_polygon_block', mio_error_message(), stat, errmsg)
            return
        end if
        if (shape%is_polyhedron /= 0) then
            call c_mio_poly_conn_free(poly)
            call handle_failure('get_polygon_block', &
                                'cell block is a polyhedron block; use get_polyhedron_block', &
                                stat, errmsg)
            return
        end if
        p = c_mio_poly_conn_face_offsets(poly, n)
        deallocate (row_offsets)
        allocate (row_offsets(max(n, 0_c_int64_t)))
        if (c_associated(p) .and. n > 0) then
            call c_f_pointer(p, vals, [n])
            row_offsets = vals + 1_int64
        end if
        p = c_mio_poly_conn_nodes(poly, n)
        deallocate (nodes)
        allocate (nodes(max(n, 0_c_int64_t)))
        if (c_associated(p) .and. n > 0) then
            call c_f_pointer(p, vals, [n])
            nodes = vals + 1_int64
        end if
        call c_mio_poly_conn_free(poly)
        call clear_status(stat, errmsg)
    end subroutine

    !> Copy 1-based polyhedron cell block `block` into a 1-based CSR triple,
    !> all allocated here: `cell_offsets(num_cells + 1)` indexes the FACE list,
    !> `face_offsets(num_faces + 1)` indexes `nodes`. Face `f` of cell `c` is
    !> `nodes(face_offsets(j) : face_offsets(j + 1) - 1)` with
    !> `j = cell_offsets(c) + f - 1`.
    subroutine mesh_get_polyhedron_block(self, block, cell_offsets, face_offsets, nodes, &
                                         stat, errmsg)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(int64), allocatable, intent(out) :: cell_offsets(:), face_offsets(:), nodes(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: poly, p
        type(c_mio_poly_conn_shape) :: shape
        integer(c_int64_t) :: n
        integer(c_int64_t), pointer :: vals(:)

        allocate (cell_offsets(0), face_offsets(0), nodes(0))
        poly = c_mio_poly_conn_create(self%handle, int(block - 1, c_int64_t))
        if (.not. c_associated(poly)) then
            call handle_failure('get_polyhedron_block', mio_error_message(), stat, errmsg)
            return
        end if
        if (c_mio_poly_conn_get_shape(poly, shape) /= 0_c_int) then
            call c_mio_poly_conn_free(poly)
            call handle_failure('get_polyhedron_block', mio_error_message(), stat, errmsg)
            return
        end if
        if (shape%is_polyhedron == 0) then
            call c_mio_poly_conn_free(poly)
            call handle_failure('get_polyhedron_block', &
                                'cell block is a 1-level polygon block; use get_polygon_block', &
                                stat, errmsg)
            return
        end if
        p = c_mio_poly_conn_cell_offsets(poly, n)
        deallocate (cell_offsets)
        allocate (cell_offsets(max(n, 0_c_int64_t)))
        if (c_associated(p) .and. n > 0) then
            call c_f_pointer(p, vals, [n])
            cell_offsets = vals + 1_int64
        end if
        p = c_mio_poly_conn_face_offsets(poly, n)
        deallocate (face_offsets)
        allocate (face_offsets(max(n, 0_c_int64_t)))
        if (c_associated(p) .and. n > 0) then
            call c_f_pointer(p, vals, [n])
            face_offsets = vals + 1_int64
        end if
        p = c_mio_poly_conn_nodes(poly, n)
        deallocate (nodes)
        allocate (nodes(max(n, 0_c_int64_t)))
        if (c_associated(p) .and. n > 0) then
            call c_f_pointer(p, vals, [n])
            nodes = vals + 1_int64
        end if
        call c_mio_poly_conn_free(poly)
        call clear_status(stat, errmsg)
    end subroutine

    !> Copy 1-based cell block `block` into `conn(nodes_per_cell, num_cells)`
    !> (allocated here), shifting the 0-based core indices to 1-based.
    subroutine mesh_get_cell_block(self, block, conn, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: block
        integer(int64), allocatable, intent(out) :: conn(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: p
        integer(c_int) :: dtype, status
        integer(c_int64_t) :: nc, npc
        integer(c_int32_t) :: ragged
        integer(c_int64_t), pointer :: i64(:)
        integer(c_int32_t), pointer :: i32(:)
        status = c_mio_mesh_cell_block_info(self%handle, int(block - 1, c_int64_t), nc, npc, &
                                            ragged)
        if (status == 0_c_int) status = c_mio_mesh_cell_block_conn(self%handle, &
                                                                   int(block - 1, c_int64_t), &
                                                                   p, dtype)
        if (status /= 0_c_int) then
            call handle_status(status, 'get_cell_block', stat, errmsg)
            return
        end if
        allocate (conn(npc, nc))
        if (nc*npc == 0) then
            call clear_status(stat, errmsg)
            return
        end if
        select case (dtype)
        case (MIO_INT64)
            call c_f_pointer(p, i64, [nc*npc])
            conn = reshape(i64, [npc, nc]) + 1_int64
        case (MIO_INT32)
            call c_f_pointer(p, i32, [nc*npc])
            conn = reshape(int(i32, int64), [npc, nc]) + 1_int64
        case default
            call handle_failure('get_cell_block', 'unexpected connectivity dtype', stat, errmsg)
            return
        end select
        call clear_status(stat, errmsg)
    end subroutine

    integer(int64) function mesh_num_point_data(self)
        class(mio_mesh), intent(in) :: self
        mesh_num_point_data = c_mio_mesh_num_point_data(self%handle)
    end function

    !> The 1-based `index`-th point-data name (ascending lexicographic order).
    function mesh_point_data_name(self, index, stat, errmsg) result(name)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: name
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        name = ''
        n = c_mio_mesh_point_data_name(self%handle, int(index - 1, c_int64_t), buf, &
                                       int(STRBUF_LEN, c_int64_t))
        if (n < 0) then
            call handle_failure('point_data_name', mio_error_message(), stat, errmsg)
            return
        end if
        name = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        call clear_status(stat, errmsg)
    end function

    !> Copy the named scalar point-data array into `data(:)` (allocated here).
    subroutine mesh_get_point_data_r1(self, name, data, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: name
        real(real64), allocatable, intent(out) :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call get_named_r1(self, 'point_data', name, -1_int64, data, stat, errmsg)
    end subroutine

    !> Copy the named vector point-data array into
    !> `data(num_components, num_points)` (allocated here).
    subroutine mesh_get_point_data_r2(self, name, data, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: name
        real(real64), allocatable, intent(out) :: data(:, :)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: p
        integer(c_int) :: dtype, status
        integer(c_int32_t) :: ndim
        integer(c_int64_t) :: shape(MIO_MAX_NDIM)
        status = c_mio_mesh_get_point_data(self%handle, c_str(name), p, dtype, ndim, shape)
        if (status /= 0_c_int) then
            call handle_status(status, 'get_point_data', stat, errmsg)
            return
        end if
        if (ndim /= 2_c_int32_t) then
            call handle_failure('get_point_data', 'point_data "'//trim(name)// &
                                '" is not rank-2', stat, errmsg)
            return
        end if
        call copy_out_r2(p, dtype, shape(1), shape(2), data, 'get_point_data', stat, errmsg)
    end subroutine

    integer(int64) function mesh_num_cell_data(self)
        class(mio_mesh), intent(in) :: self
        mesh_num_cell_data = c_mio_mesh_num_cell_data(self%handle)
    end function

    !> The 1-based `index`-th cell-data name (sorted, as above).
    function mesh_cell_data_name(self, index, stat, errmsg) result(name)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: name
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        name = ''
        n = c_mio_mesh_cell_data_name(self%handle, int(index - 1, c_int64_t), buf, &
                                      int(STRBUF_LEN, c_int64_t))
        if (n < 0) then
            call handle_failure('cell_data_name', mio_error_message(), stat, errmsg)
            return
        end if
        name = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        call clear_status(stat, errmsg)
    end function

    integer(int64) function mesh_cell_data_num_blocks(self, name)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: name
        mesh_cell_data_num_blocks = c_mio_mesh_cell_data_num_blocks(self%handle, c_str(name))
    end function

    !> Copy the named cell-data field's array for 1-based cell block `block`
    !> into `data(:)` (allocated here).
    subroutine mesh_get_cell_data_r1(self, name, block, data, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: name
        integer, intent(in) :: block
        real(real64), allocatable, intent(out) :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call get_named_r1(self, 'cell_data', name, int(block - 1, int64), data, stat, errmsg)
    end subroutine

    integer(int64) function mesh_num_field_data(self)
        class(mio_mesh), intent(in) :: self
        mesh_num_field_data = c_mio_mesh_num_field_data(self%handle)
    end function

    !> The 1-based `index`-th field-data name (sorted, as above).
    function mesh_field_data_name(self, index, stat, errmsg) result(name)
        class(mio_mesh), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: name
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        name = ''
        n = c_mio_mesh_field_data_name(self%handle, int(index - 1, c_int64_t), buf, &
                                       int(STRBUF_LEN, c_int64_t))
        if (n < 0) then
            call handle_failure('field_data_name', mio_error_message(), stat, errmsg)
            return
        end if
        name = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
        call clear_status(stat, errmsg)
    end function

    !> Copy the named field-data array into `data(:)` (allocated here).
    subroutine mesh_get_field_data_r1(self, name, data, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: name
        real(real64), allocatable, intent(out) :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call get_named_r1(self, 'field_data', name, -2_int64, data, stat, errmsg)
    end subroutine

    ! ------------------------------------------------------------------
    ! Private implementation helpers
    ! ------------------------------------------------------------------

    !> Setters on a never-created handle allocate one on the fly, so
    !> `type(mio_mesh) :: m; call m%set_points(...)` just works.
    subroutine ensure_handle(self, stat, errmsg)
        class(mio_mesh), intent(inout) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (.not. c_associated(self%handle)) call mesh_create(self, stat, errmsg)
    end subroutine

    !> Shared rank-1 copy-getter over the three named-data families
    !> (block >= 0: cell_data block; -1: point_data; -2: field_data).
    subroutine get_named_r1(self, family, name, block, data, stat, errmsg)
        class(mio_mesh), intent(in) :: self
        character(*), intent(in) :: family, name
        integer(int64), intent(in) :: block
        real(real64), allocatable, intent(out) :: data(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(c_ptr) :: p
        integer(c_int) :: dtype, status
        integer(c_int32_t) :: ndim
        integer(c_int64_t) :: shape(MIO_MAX_NDIM)
        select case (family)
        case ('point_data')
            status = c_mio_mesh_get_point_data(self%handle, c_str(name), p, dtype, ndim, shape)
        case ('cell_data')
            status = c_mio_mesh_get_cell_data(self%handle, c_str(name), block, p, dtype, ndim, &
                                              shape)
        case default
            status = c_mio_mesh_get_field_data(self%handle, c_str(name), p, dtype, ndim, shape)
        end select
        if (status /= 0_c_int) then
            call handle_status(status, 'get_'//family, stat, errmsg)
            return
        end if
        if (ndim /= 1_c_int32_t) then
            call handle_failure('get_'//family, family//' "'//trim(name)// &
                                '" is not rank-1 (use the rank-2 getter)', stat, errmsg)
            return
        end if
        call copy_out_r1(p, dtype, shape(1), data, 'get_'//family, stat, errmsg)
    end subroutine

    !> Copy a C array of `n` elements of any supported dtype into a real64
    !> rank-1 allocatable.
    subroutine copy_out_r1(p, dtype, n, data, what, stat, errmsg)
        type(c_ptr), intent(in) :: p
        integer(c_int), intent(in) :: dtype
        integer(c_int64_t), intent(in) :: n
        real(real64), allocatable, intent(out) :: data(:)
        character(*), intent(in) :: what
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        real(c_double), pointer :: d64(:)
        real(c_float), pointer :: d32(:)
        integer(c_int64_t), pointer :: i64(:)
        integer(c_int32_t), pointer :: i32(:)
        allocate (data(n))
        if (n == 0) then
            call clear_status(stat, errmsg)
            return
        end if
        select case (dtype)
        case (MIO_FLOAT64)
            call c_f_pointer(p, d64, [n]); data = d64
        case (MIO_FLOAT32)
            call c_f_pointer(p, d32, [n]); data = real(d32, real64)
        case (MIO_INT64)
            call c_f_pointer(p, i64, [n]); data = real(i64, real64)
        case (MIO_INT32)
            call c_f_pointer(p, i32, [n]); data = real(i32, real64)
        case default
            call handle_failure(what, 'unsupported dtype for real64 copy', stat, errmsg)
            return
        end select
        call clear_status(stat, errmsg)
    end subroutine

    !> As copy_out_r1 for a C row-major `(n0, n1)` array, delivered as the
    !> Fortran `(n1, n0)` view of the same memory order.
    subroutine copy_out_r2(p, dtype, n0, n1, data, what, stat, errmsg)
        type(c_ptr), intent(in) :: p
        integer(c_int), intent(in) :: dtype
        integer(c_int64_t), intent(in) :: n0, n1
        real(real64), allocatable, intent(out) :: data(:, :)
        character(*), intent(in) :: what
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        real(c_double), pointer :: d64(:)
        real(c_float), pointer :: d32(:)
        integer(c_int64_t), pointer :: i64(:)
        integer(c_int32_t), pointer :: i32(:)
        allocate (data(n1, n0))
        if (n0*n1 == 0) then
            call clear_status(stat, errmsg)
            return
        end if
        select case (dtype)
        case (MIO_FLOAT64)
            call c_f_pointer(p, d64, [n0*n1]); data = reshape(d64, [n1, n0])
        case (MIO_FLOAT32)
            call c_f_pointer(p, d32, [n0*n1]); data = reshape(real(d32, real64), [n1, n0])
        case (MIO_INT64)
            call c_f_pointer(p, i64, [n0*n1]); data = reshape(real(i64, real64), [n1, n0])
        case (MIO_INT32)
            call c_f_pointer(p, i32, [n0*n1]); data = reshape(real(i32, real64), [n1, n0])
        case default
            call handle_failure(what, 'unsupported dtype for real64 copy', stat, errmsg)
            return
        end select
        call clear_status(stat, errmsg)
    end subroutine

    ! ------------------------------------------------------------------
    ! mio_xdmf_series: transient (time-series) XDMF writing
    !
    ! Same lifecycle rules as mio_mesh: create() replaces whatever the handle
    ! held, free() is idempotent, and nothing is freed implicitly.
    ! ------------------------------------------------------------------

    !> Open a transient XDMF series for writing. Nothing is written yet.
    !>
    !> `data_format` is `"HDF"` (the default; needs an HDF5-enabled build),
    !> `"XML"` (inline) or `"Binary"`. `gzip_level` applies to `"HDF"` datasets
    !> only; negative (the default) means no compression. An unknown format, or
    !> `"HDF"` without HDF5 support, fails through `stat`/`errmsg`.
    subroutine xdmf_series_create(self, path, data_format, gzip_level, mode, auto_flush, &
                                  stat, errmsg)
        class(mio_xdmf_series), intent(inout) :: self
        character(*), intent(in) :: path
        character(*), intent(in), optional :: data_format
        integer, intent(in), optional :: gzip_level
        !> 'truncate' (default) or 'append'.
        character(*), intent(in), optional :: mode
        !> Flush the light data after every write_data (default .false.).
        logical, intent(in), optional :: auto_flush
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: fmt, md
        character(len=:, kind=c_char), allocatable, target :: fmt_buf
        type(mio_xdmf_series_opts_t) :: opts
        integer(c_int32_t) :: level
        call xdmf_series_free(self)
        fmt = 'HDF'; if (present(data_format)) fmt = data_format
        md = 'truncate'; if (present(mode)) md = mode
        level = -1_c_int32_t
        if (present(gzip_level)) level = int(gzip_level, c_int32_t)

        if (md /= 'truncate' .and. md /= 'append') then
            call handle_failure('xdmf_series create', &
                                "mode must be 'truncate' or 'append', got '"//md//"'", stat, errmsg)
            return
        end if

        if (.not. present(mode) .and. .not. present(auto_flush)) then
            self%handle = c_mio_xdmf_series_create(c_str(path), c_str(fmt), level)
        else
            ! The NUL-terminated format string must outlive the call, so it is
            ! held in a TARGET local rather than built inline.
            fmt_buf = fmt//c_null_char
            opts%data_format = c_loc(fmt_buf(1:1))
            opts%gzip_level = level
            if (md == 'append') opts%mode = 1_c_int32_t
            if (present(auto_flush)) then
                if (auto_flush) opts%auto_flush = 1_c_int32_t
            end if
            self%handle = c_mio_xdmf_series_create_ex(c_str(path), opts)
        end if
        if (.not. c_associated(self%handle)) then
            call handle_failure('xdmf_series create', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end subroutine

    !> Write the `.xdmf` as it currently stands, without finalizing, so a run
    !> that is killed or still going leaves a readable file covering every
    !> flushed step. Safe to call repeatedly; a no-op once finalized.
    subroutine xdmf_series_flush(self, stat, errmsg)
        class(mio_xdmf_series), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (.not. c_associated(self%handle)) then
            call handle_failure('xdmf_series flush', 'series handle is not open', stat, errmsg)
            return
        end if
        call handle_status(c_mio_xdmf_series_flush(self%handle), 'xdmf_series flush', stat, errmsg)
    end subroutine

    !> .true. once finalize() has run for this series (also on a closed handle).
    function xdmf_series_finalized(self) result(f)
        class(mio_xdmf_series), intent(in) :: self
        logical :: f
        f = .true.
        if (.not. c_associated(self%handle)) return
        f = c_mio_xdmf_series_finalized(self%handle) == 1_c_int32_t
    end function

    !> Finalize (if it has not happened yet) and release the series. Idempotent.
    !> A write failure during the implicit finalize is swallowed here -- call
    !> `finalize()` explicitly to see it.
    subroutine xdmf_series_free(self)
        class(mio_xdmf_series), intent(inout) :: self
        if (c_associated(self%handle)) call c_mio_xdmf_series_free(self%handle)
        self%handle = c_null_ptr
    end subroutine

    !> .true. between a successful create() and free().
    logical function xdmf_series_is_valid(self)
        class(mio_xdmf_series), intent(in) :: self
        xdmf_series_is_valid = c_associated(self%handle)
    end function

    !> Write the static grid every step shares. Call once, before the first
    !> `write_data`. Only the mesh's points and cells are used.
    subroutine xdmf_series_write_points_cells(self, mesh, stat, errmsg)
        class(mio_xdmf_series), intent(in) :: self
        class(mio_mesh), intent(in) :: mesh
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (.not. c_associated(self%handle)) then
            call handle_failure('xdmf_series write_points_cells', &
                                'series handle is not open', stat, errmsg)
            return
        end if
        call handle_status(c_mio_xdmf_series_write_points_cells(self%handle, mesh%handle), &
                           'xdmf_series write_points_cells', stat, errmsg)
    end subroutine

    !> Write one time step's point_data and cell_data at time `time`. The mesh's
    !> geometry is ignored, so a solver can pass the same object it updates in
    !> place; its cell blocks must match those of the static grid.
    subroutine xdmf_series_write_data(self, time, mesh, stat, errmsg)
        class(mio_xdmf_series), intent(in) :: self
        real(real64), intent(in) :: time
        class(mio_mesh), intent(in) :: mesh
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (.not. c_associated(self%handle)) then
            call handle_failure('xdmf_series write_data', 'series handle is not open', &
                                stat, errmsg)
            return
        end if
        call handle_status(c_mio_xdmf_series_write_data(self%handle, &
                                                        real(time, c_double), mesh%handle), &
                           'xdmf_series write_data', stat, errmsg)
    end subroutine

    !> Write the `.xdmf` and close the heavy-data container. Idempotent, and
    !> `free()` does it too -- call this explicitly so a write failure surfaces
    !> through `stat` instead of being swallowed.
    subroutine xdmf_series_finalize(self, stat, errmsg)
        class(mio_xdmf_series), intent(in) :: self
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        if (.not. c_associated(self%handle)) then
            call handle_failure('xdmf_series finalize', 'series handle is not open', &
                                stat, errmsg)
            return
        end if
        call handle_status(c_mio_xdmf_series_finalize(self%handle), &
                           'xdmf_series finalize', stat, errmsg)
    end subroutine

    !> Number of steps written so far (0 on a closed handle).
    function xdmf_series_num_steps(self) result(n)
        class(mio_xdmf_series), intent(in) :: self
        integer(int64) :: n
        n = 0_int64
        if (.not. c_associated(self%handle)) return
        n = int(c_mio_xdmf_series_num_steps(self%handle), int64)
        if (n < 0_int64) n = 0_int64
    end function

    ! ----------------------------------------------------------------------
    ! Sequences (multi-file / transient datasets). See doc/sequences.md.
    !
    ! Indices are 1-based on this side, like every other Fortran accessor;
    ! the C ABI's 0-based index is produced by the `- 1` inside each wrapper.
    ! ----------------------------------------------------------------------

    !> Open a sequence from a glob pattern (`*` and `?` only; the directory
    !> part is taken literally). A pattern matching nothing is an error, never
    !> an empty sequence. Replaces any sequence this handle already held.
    subroutine sequence_open(self, pattern, stat, errmsg)
        class(mio_sequence), intent(inout) :: self
        character(*), intent(in) :: pattern
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call sequence_free(self)
        self%handle = c_mio_sequence_open(c_str(pattern))
        if (.not. c_associated(self%handle)) then
            call handle_failure('sequence open', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end subroutine

    !> Open a sequence from an explicit, ordered path list. The order is the
    !> caller's and is kept -- a stated order is not second-guessed.
    subroutine sequence_open_list(self, paths, stat, errmsg)
        class(mio_sequence), intent(inout) :: self
        character(*), intent(in) :: paths(:)
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        ! `storage`/`cptrs` are the actual backing memory for the C string
        ! array and must stay in scope for the whole call -- see c_str_array.
        character(kind=c_char), allocatable, target :: storage(:, :)
        type(c_ptr), allocatable, target :: cptrs(:)
        type(c_ptr) :: arr
        integer(c_int64_t) :: count
        call sequence_free(self)
        if (size(paths) <= 0) then
            call handle_failure('sequence open_list', 'the path list is empty', stat, errmsg)
            return
        end if
        call c_str_array(paths, storage, cptrs, arr, count)
        self%handle = c_mio_sequence_open_list(cptrs, count)
        if (.not. c_associated(self%handle)) then
            call handle_failure('sequence open_list', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end subroutine

    !> Release the sequence handle. Meshes it produced are unaffected.
    subroutine sequence_free(self)
        class(mio_sequence), intent(inout) :: self
        if (c_associated(self%handle)) call c_mio_sequence_free(self%handle)
        self%handle = c_null_ptr
    end subroutine

    !> Whether this handle currently holds a sequence.
    function sequence_is_valid(self) result(v)
        class(mio_sequence), intent(in) :: self
        logical :: v
        v = c_associated(self%handle)
    end function

    !> Number of steps (0 on a closed handle).
    function sequence_count(self) result(n)
        class(mio_sequence), intent(in) :: self
        integer(int64) :: n
        n = 0_int64
        if (.not. c_associated(self%handle)) return
        n = int(c_mio_sequence_count(self%handle), int64)
        if (n < 0_int64) n = 0_int64
    end function

    !> Entry `index`'s file path (1-based; empty on error).
    function sequence_path(self, index) result(p)
        class(mio_sequence), intent(in) :: self
        integer, intent(in) :: index
        character(:), allocatable :: p
        character(kind=c_char) :: buf(STRBUF_LEN)
        integer(c_int64_t) :: n
        p = ''
        if (.not. c_associated(self%handle)) return
        n = c_mio_sequence_path(self%handle, int(index - 1, c_int64_t), buf, &
                                int(STRBUF_LEN, c_int64_t))
        if (n > 0) p = from_c_buf(buf, min(int(n), STRBUF_LEN - 1))
    end function

    !> Entry `index`'s step WITHIN its own file (0 for a single-step file).
    function sequence_step(self, index) result(k)
        class(mio_sequence), intent(in) :: self
        integer, intent(in) :: index
        integer(int64) :: k
        k = -1_int64
        if (.not. c_associated(self%handle)) return
        k = int(c_mio_sequence_step(self%handle, int(index - 1, c_int64_t)), int64)
    end function

    !> Entry `index`'s time value.
    function sequence_time(self, index) result(t)
        class(mio_sequence), intent(in) :: self
        integer, intent(in) :: index
        real(real64) :: t
        real(c_double) :: v
        t = 0.0_real64
        if (.not. c_associated(self%handle)) return
        if (c_mio_sequence_time(self%handle, int(index - 1, c_int64_t), v) == 0) &
            t = real(v, real64)
    end function

    !> Where entry `index`'s time came from: 0 explicit, 1 file, 2 filename,
    !> 3 index (the fallback). Reported so a caller can tell "the file said
    !> 0.25" from "nothing said anything, so this is position 3".
    function sequence_time_source(self, index) result(src)
        class(mio_sequence), intent(in) :: self
        integer, intent(in) :: index
        integer :: src
        src = -1
        if (.not. c_associated(self%handle)) return
        src = int(c_mio_sequence_time_source(self%handle, int(index - 1, c_int64_t)))
    end function

    !> Read entry `index` (1-based). The result is OWNED: free it with
    !> `%free()`. The sequence caches nothing, which is what keeps a long
    !> dataset traversable.
    function sequence_read_step(self, index, stat, errmsg) result(m)
        class(mio_sequence), intent(in) :: self
        integer, intent(in) :: index
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        type(mio_mesh) :: m
        if (.not. c_associated(self%handle)) then
            call handle_failure('sequence read_step', 'sequence handle is not open', &
                                stat, errmsg)
            return
        end if
        m%handle = c_mio_sequence_read(self%handle, int(index - 1, c_int64_t))
        if (.not. c_associated(m%handle)) then
            call handle_failure('sequence read_step', mio_error_message(), stat, errmsg)
            return
        end if
        call clear_status(stat, errmsg)
    end function

    !> Fan-in: write every step into one multi-step file. Streams -- one mesh
    !> alive at a time. A format that cannot hold a series fails by name.
    subroutine sequence_to_timeseries(self, out_path, out_format, stat, errmsg)
        class(mio_sequence), intent(in) :: self
        character(*), intent(in) :: out_path
        character(*), intent(in), optional :: out_format
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: fmt
        fmt = ''
        if (present(out_format)) fmt = out_format
        if (.not. c_associated(self%handle)) then
            call handle_failure('sequence to_timeseries', 'sequence handle is not open', &
                                stat, errmsg)
            return
        end if
        call handle_status(c_mio_sequence_to_timeseries(self%handle, c_str(out_path), &
                                                        c_str(fmt)), &
                           'sequence to_timeseries', stat, errmsg)
    end subroutine

    !> Fan-out: write each step of a multi-step file to `out_pattern`, which
    !> must contain '{step}' or '{index}'.
    subroutine mio_timeseries_to_sequence(in_path, out_pattern, in_format, out_format, &
                                          stat, errmsg)
        character(*), intent(in) :: in_path
        character(*), intent(in) :: out_pattern
        character(*), intent(in), optional :: in_format
        character(*), intent(in), optional :: out_format
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        character(:), allocatable :: ifmt, ofmt
        ifmt = ''
        ofmt = ''
        if (present(in_format)) ifmt = in_format
        if (present(out_format)) ofmt = out_format
        call handle_status(c_mio_timeseries_to_sequence(c_str(in_path), c_str(ifmt), &
                                                        c_str(out_pattern), c_str(ofmt)), &
                           'timeseries_to_sequence', stat, errmsg)
    end subroutine

    !> Run a sequence settings document (the pipeline schema plus Mode /
    !> Input.Pattern / Input.Paths / Input.Times / Input.TimeFrom). A document
    !> using none of those behaves exactly as `mio_pipeline_run_file`.
    subroutine mio_sequence_pipeline_run_file(settings_path, stat, errmsg)
        character(*), intent(in) :: settings_path
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call handle_status(c_mio_sequence_pipeline_run_file(c_str(settings_path)), &
                           'sequence_pipeline_run_file', stat, errmsg)
    end subroutine

    !> `mio_sequence_pipeline_run_file` over JSON text.
    subroutine mio_sequence_pipeline_run_json(json_text, stat, errmsg)
        character(*), intent(in) :: json_text
        integer, intent(out), optional :: stat
        character(:), allocatable, intent(out), optional :: errmsg
        call handle_status(c_mio_sequence_pipeline_run_json(c_str(json_text)), &
                           'sequence_pipeline_run_json', stat, errmsg)
    end subroutine

    !> Open a provenance scope for this thread. Every mio write made before the
    !> matching mio_provenance_end records the richer block; with no scope open
    !> a writer emits only the one-line credit. Explicit begin/end rather than a
    !> block construct, matching this module's own explicit m%free() convention.
    subroutine mio_provenance_begin(mode, stat, errmsg)
        integer, intent(in), optional :: mode
        integer, intent(out), optional :: stat
        character(len=:), allocatable, intent(out), optional :: errmsg
        integer(c_int) :: m, rc
        m = MIO_PROVENANCE_BEST_EFFORT
        if (present(mode)) m = int(mode, c_int)
        rc = c_mio_provenance_scope_begin(m)
        call handle_status(rc, "mio_provenance_begin", stat, errmsg)
    end subroutine

    !> Close the most recently opened provenance scope. A stray call with none
    !> open is a no-op.
    subroutine mio_provenance_end()
        call c_mio_provenance_scope_end()
    end subroutine

    !> Record one conversion assumption. A no-op outside a scope, so it is safe
    !> to call unconditionally.
    subroutine mio_provenance_note(category, detail)
        character(*), intent(in) :: category, detail
        call c_mio_provenance_note(c_str(category), c_str(detail))
    end subroutine

    !> Record where the mesh came from. A no-op outside a scope.
    subroutine mio_provenance_source(path, format)
        character(*), intent(in) :: path, format
        call c_mio_provenance_set_source(c_str(path), c_str(format))
    end subroutine

    !> Record what was actually written (not what was requested). A no-op
    !> outside a scope.
    subroutine mio_provenance_target(format, encoding, codec, float_format)
        character(*), intent(in) :: format
        character(*), intent(in), optional :: encoding, codec, float_format
        character(len=:), allocatable :: e, c, f
        e = ""; c = ""; f = ""
        if (present(encoding)) e = encoding
        if (present(codec)) c = codec
        if (present(float_format)) f = float_format
        call c_mio_provenance_set_target(c_str(format), c_str(e), c_str(c), c_str(f))
    end subroutine

end module meshioplusplus
