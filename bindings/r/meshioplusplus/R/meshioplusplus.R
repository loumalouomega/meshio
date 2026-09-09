## R interface to the meshio++ C library.
##
## Two facts govern the whole file, and both are documented rather than hidden:
##
##  * COLUMN-MAJOR. R matrices are column-major and the C core row-major, so an
##    R (dim x num_points) matrix is the SAME layout as the C API's row-major
##    (num_points, dim). Nothing is transposed anywhere.
##  * COPY-ONLY. R vectors are R-managed, so the C API's zero-copy borrow does
##    not survive into R without ALTREP machinery that is out of scope. EVERY
##    accessor here copies. There is deliberately no `_ptr` accessor, unlike the
##    Julia and Fortran bindings; the 0-based reader is named
##    `mio_connectivity_raw` so nobody reads it as a borrow.
##
## R also has no native 64-bit integer type, so the int64 arrays the C API
## returns (connectivity, region entries, index maps, permutations) come back as
## `double`. That is exact to 2^53 -- far beyond any mesh -- and avoids a hard
## dependency on `bit64` for a theoretical case.

# --- small helpers ------------------------------------------------------------

.mio_location <- function(location) {
  switch(location,
    point = 0L, cell = 1L, field = 2L,
    stop("`location` must be \"point\", \"cell\" or \"field\", not \"", location, "\"")
  )
}

.mio_mode <- function(mode) {
  switch(mode,
    all = 0L, any = 1L,
    stop("`mode` must be \"all\" or \"any\", not \"", mode, "\"")
  )
}

.mio_mmap <- function(mmap) {
  switch(mmap,
    auto = 0L, on = 1L, off = 2L,
    stop("`mmap` must be \"auto\", \"on\" or \"off\", not \"", mmap, "\"")
  )
}

.mio_block <- function(block) {
  block <- as.numeric(block)
  if (length(block) != 1L || is.na(block) || block < 1) {
    stop("`block` must be a single 1-based cell-block index")
  }
  block - 1  # 1-based in R, 0-based at the ABI
}

# --- introspection ------------------------------------------------------------

#' meshio++ build information
#'
#' @return `mio_version()` and `mio_mesh_backend()` return a single string;
#'   `mio_last_error()` returns the message of the most recent failed call on
#'   this thread, or `""`.
#' @export
mio_version <- function() .Call(R_mio_version)

#' @rdname mio_version
#' @export
mio_mesh_backend <- function() .Call(R_mio_mesh_backend)

#' @rdname mio_version
#' @export
mio_last_error <- function() .Call(R_mio_last_error)

#' Format support in this build
#'
#' @param format A format name, e.g. `"gmsh"`, `"vtu"`, `"med"`.
#' @return A single logical: whether the format can be read / written by the
#'   installed library. Formats backed by optional dependencies (HDF5, netCDF)
#'   exist only in builds configured with them.
#' @export
mio_format_readable <- function(format) .Call(R_mio_format_readable, as.character(format))

#' @rdname mio_format_readable
#' @export
mio_format_writable <- function(format) .Call(R_mio_format_writable, as.character(format))

#' Guess a mesh file's format from its contents
#'
#' @param path Path to an existing, readable file.
#' @return The format name, or `""` when the leading bytes are not a confident
#'   match. Ambiguous magics are never guessed at.
#' @export
mio_sniff_format <- function(path) .Call(R_mio_sniff_format, as.character(path))

#' Cell-type metadata
#'
#' @param name A meshio cell-type name, e.g. `"triangle"`, `"tetra10"`.
#' @return `mio_cell_type_num_nodes()` gives the fixed nodes per cell, or `-1`
#'   for the variable-size types (polygon, polyhedron, VTK Lagrange) and for
#'   unknown names. `mio_cell_type_dimension()` gives the topological dimension
#'   0-3, or `-1` for an unknown name.
#' @export
mio_cell_type_num_nodes <- function(name) {
  .Call(R_mio_cell_type_num_nodes, as.character(name))
}

#' @rdname mio_cell_type_num_nodes
#' @export
mio_cell_type_dimension <- function(name) {
  .Call(R_mio_cell_type_dimension, as.character(name))
}

#' @rdname mio_read
#' @export
mio_reader_supports_options <- function(format) {
  .Call(R_mio_reader_supports_options, as.character(format))
}

# --- lifecycle ----------------------------------------------------------------

#' A meshio++ mesh handle
#'
#' `mio_mesh()` creates an empty mesh. The handle is an external pointer with a
#' registered finalizer, so it is released automatically when garbage-collected;
#' `mio_release()` frees it immediately and is idempotent.
#'
#' @param mesh A `mio_mesh` object.
#' @param x A `mio_mesh` object.
#' @param ... Ignored.
#' @return `mio_mesh()` returns a new `mio_mesh`. `mio_release()` returns
#'   `NULL` invisibly. `mio_is_open()` returns a single logical.
#'   `print.mio_mesh()` returns its argument invisibly.
#' @examples
#' m <- mio_mesh()
#' mio_is_open(m)
#' mio_release(m)
#' @export
mio_mesh <- function() .Call(R_mio_mesh_create)

#' @rdname mio_mesh
#' @export
mio_release <- function(mesh) invisible(.Call(R_mio_mesh_release, mesh))

#' @rdname mio_mesh
#' @export
mio_is_open <- function(mesh) .Call(R_mio_mesh_is_open, mesh)

#' @rdname mio_mesh
#' @export
print.mio_mesh <- function(x, ...) {
  if (!mio_is_open(x)) {
    cat("<mio_mesh: released>\n")
  } else {
    cat(sprintf(
      "<mio_mesh: %g points, %g cells in %g block(s)>\n",
      mio_num_points(x), mio_num_cells(x), mio_num_cell_blocks(x)
    ))
  }
  invisible(x)
}

# --- file I/O -----------------------------------------------------------------

#' Read a mesh file
#'
#' @param path Path to the file.
#' @param format Explicit format name, or `NULL` to infer it from the extension.
#' @param points_only Read geometry only, skipping every data array.
#' @param metadata_only Read the header/summary only.
#' @param arrays `NULL` keeps every data array; a character vector keeps only
#'   the named ones. `character(0)` means *no* arrays, which is deliberately a
#'   different request from `NULL`.
#' @param mmap One of `"auto"`, `"on"`, `"off"`.
#' @param time_step Which step of a multi-step file to materialize. `0` (the
#'   default) is the first, preserving the historical behaviour; negative counts
#'   from the end. Out of range is an error naming the available count, never a
#'   silent clamp. Honoured by formats carrying a time series (currently
#'   `exodus`); `mio_read_metadata()$time_values` reports how many there are.
#' @param lenient Downgrade "this reader cannot represent construct X" errors to
#'   a warning plus a skip -- currently `mdpa`'s `Table`, `Geometries`, `Mesh`
#'   and `Constraints` blocks. This is *not* "ignore all errors": a malformed
#'   file, a truncated block or a bad node reference still fails, because
#'   continuing past those would return a mesh that is quietly wrong.
#' @return A `mio_mesh` object.
#' @examples
#' \dontrun{
#' m <- mio_read("bracket.msh")
#' m <- mio_read("bracket.vtu", points_only = TRUE)
#' m <- mio_read("run.exo", time_step = -1) # the last step
#' m <- mio_read("model.mdpa", lenient = TRUE) # skip unsupported blocks
#' }
#' @export
mio_read <- function(path, format = NULL, points_only = FALSE, metadata_only = FALSE,
                     arrays = NULL, mmap = "auto", time_step = 0, lenient = FALSE) {
  .Call(
    R_mio_read, as.character(path), format, isTRUE(points_only),
    isTRUE(metadata_only), if (is.null(arrays)) NULL else as.character(arrays),
    .mio_mmap(mmap), as.integer(time_step), isTRUE(lenient)
  )
}

#' Write a mesh file
#'
#' @param mesh A `mio_mesh` object.
#' @param path Destination path.
#' @param format Explicit format name, or `NULL` to infer it from the extension.
#' @return `NULL`, invisibly.
#' @export
mio_write <- function(mesh, path, format = NULL) {
  invisible(.Call(R_mio_write, mesh, as.character(path), format))
}

#' Convert a mesh file without materializing it
#'
#' Reads `in_path` and immediately writes `out_path`, the CLI's `convert` verb.
#'
#' @param in_path,out_path Source and destination paths.
#' @param in_format,out_format Explicit format names, or `NULL` to infer them.
#' @return `NULL`, invisibly.
#' @export
mio_convert <- function(in_path, out_path, in_format = NULL, out_format = NULL) {
  invisible(.Call(
    R_mio_convert, as.character(in_path), in_format,
    as.character(out_path), out_format
  ))
}

#' Run a settings.json operation pipeline
#'
#' Runs a whole settings pipeline -- read `Input.Path`, apply `Operations` in
#' order, write `Output.Path` -- described by a `settings.json` document
#' (PascalCase ops and keys; see `doc/pipeline.md`). Needs a library built
#' with the JSON parser (`-DMESHIOPLUSPLUS_WITH_JSON=ON`, the default when the
#' `src/cpp/third_party/json` submodule is checked out); otherwise the raised
#' error names the flag -- `mio_pipeline_has_json()` reports which build the
#' loaded library is.
#'
#' @param settings_path Path of the `settings.json` to run.
#' @return `NULL`, invisibly.
#' @export
mio_pipeline_run_file <- function(settings_path) {
  invisible(.Call(R_mio_pipeline_run_file, as.character(settings_path)))
}

#' @rdname mio_pipeline_run_file
#' @param json_text The settings document as JSON text.
#' @export
mio_pipeline_run_json <- function(json_text) {
  invisible(.Call(R_mio_pipeline_run_json, as.character(json_text)))
}

#' @rdname mio_pipeline_run_file
#' @export
mio_pipeline_has_json <- function() {
  .Call(R_mio_pipeline_has_json)
}

#' Summarize a mesh file without loading its arrays
#'
#' @param path Path to the file.
#' @param format Explicit format name, or `NULL` to infer it.
#' @return A named list with `num_points`, `point_dim`, `num_cells`,
#'   `num_cell_blocks`, the per-block vectors `cell_block_types`,
#'   `cell_block_num_cells`, `cell_block_nodes_per_cell`,
#'   `cell_block_is_ragged`, the three data-name vectors, `bbox` (a 3x2 matrix,
#'   or `NULL` -- absent is the normal case for a native summary, since
#'   computing it would mean decoding the coordinates), `fell_back`
#'   (`TRUE` when the whole file had to be read because the format has no
#'   header-only path), `time_values` (the file's recorded time-series
#'   values; length 0 for a format with no time concept -- this is the count
#'   `mio_read(time_step = ...)` may name), and `regions` (a list of
#'   `name`/`kind`/`dim`/`tag`/`num_entries` -- the `mio_regions()` shape minus
#'   the entries themselves; empty on a native metadata path, since none of
#'   those formats currently map regions).
#' @export
mio_read_metadata <- function(path, format = NULL) {
  .Call(R_mio_read_metadata, as.character(path), format)
}

# --- inspecting ---------------------------------------------------------------

#' Mesh size and cell-block structure
#'
#' @param mesh A `mio_mesh` object.
#' @param block A **1-based** cell-block index.
#' @return `mio_num_points()`, `mio_point_dim()`, `mio_num_cells()` and
#'   `mio_num_cell_blocks()` return a single number. `mio_cell_block_info()`
#'   returns a list with `num_cells`, `nodes_per_cell` and `is_ragged`.
#'   `mio_cell_block_type()` returns the meshio type name;
#'   `mio_cell_block_types()` returns one per block.
#' @export
mio_num_points <- function(mesh) .Call(R_mio_num_points, mesh)

#' @rdname mio_num_points
#' @export
mio_point_dim <- function(mesh) .Call(R_mio_point_dim, mesh)

#' @rdname mio_num_points
#' @export
mio_num_cell_blocks <- function(mesh) .Call(R_mio_num_cell_blocks, mesh)

#' @rdname mio_num_points
#' @export
mio_num_cells <- function(mesh) {
  # The C API has no single total-cell accessor (a documented gap), so this is
  # summed over the blocks.
  n <- mio_num_cell_blocks(mesh)
  if (n < 1) {
    return(0)
  }
  sum(vapply(seq_len(n), function(i) mio_cell_block_info(mesh, i)$num_cells, numeric(1)))
}

#' @rdname mio_num_points
#' @export
mio_cell_block_info <- function(mesh, block) {
  .Call(R_mio_cell_block_info, mesh, .mio_block(block))
}

#' @rdname mio_num_points
#' @export
mio_cell_block_type <- function(mesh, block) {
  .Call(R_mio_cell_block_type, mesh, .mio_block(block))
}

#' @rdname mio_num_points
#' @export
mio_cell_block_types <- function(mesh) {
  n <- mio_num_cell_blocks(mesh)
  if (n < 1) {
    return(character(0))
  }
  vapply(seq_len(n), function(i) mio_cell_block_type(mesh, i), character(1))
}

#' Point coordinates and cell connectivity
#'
#' `mio_points()` returns a `dim x num_points` matrix. Because R matrices are
#' column-major and the C core is row-major, that is the *same layout* as the C
#' API's `(num_points, dim)` -- nothing is transposed.
#'
#' `mio_connectivity()` returns **1-based** node indices;
#' `mio_connectivity_raw()` returns the ABI's own **0-based** ones. Both copy:
#' unlike the Julia and Fortran bindings, R has no zero-copy borrow, so there is
#' deliberately no `_ptr` accessor to imply one.
#'
#' Values arrive as `double` because R has no native 64-bit integer type; that
#' is exact to 2^53.
#'
#' @param mesh A `mio_mesh` object.
#' @param block A **1-based** cell-block index.
#' @return A numeric matrix: `dim x num_points` for `mio_points()`,
#'   `nodes_per_cell x num_cells` for the connectivity accessors.
#' @export
mio_points <- function(mesh) .Call(R_mio_points, mesh)

#' @rdname mio_points
#' @export
mio_connectivity <- function(mesh, block = 1) {
  .Call(R_mio_connectivity, mesh, .mio_block(block))
}

#' @rdname mio_points
#' @export
mio_connectivity_raw <- function(mesh, block = 1) {
  .Call(R_mio_connectivity_raw, mesh, .mio_block(block))
}

#' Ragged (polygon and polyhedron) cell blocks
#'
#' A ragged block holds cells of varying size, so it has no rectangular
#' connectivity matrix and `mio_connectivity()` raises for it. The C ABI carries
#' these as flat CSR arrays, but R has lists, so that is what these return --
#' the same reasoning that gives Fortran a CSR triple (no ragged array type) and
#' Julia nested vectors. Node indices are **1-based**, like every other copying
#' accessor here.
#'
#' `mio_polygon_block()` handles 1-level ragged blocks (jagged polygons): one
#' numeric vector per cell. `mio_polyhedron_block()` handles 2-level ones: one
#' list of faces per cell, each face a numeric vector of node indices. Use
#' `mio_cell_block_info()$is_polyhedron` to tell them apart; each accessor
#' raises by name if handed the other kind.
#'
#' Faces *should* be wound so their right-hand normal points out of the cell,
#' but meshio++ does not require it -- the geometric kernels repair an
#' inconsistent winding rather than rejecting it.
#'
#' @param mesh A `mio_mesh` object.
#' @param block A **1-based** cell-block index.
#' @param cell_type A meshio type name, e.g. `"polygon"` or `"polyhedron"`.
#' @param rows A list of numeric vectors, one per cell (1-based node indices).
#' @param cells A list of cells, each a list of numeric face vectors.
#' @return `mio_polygon_block()` a list of numeric vectors;
#'   `mio_polyhedron_block()` a list of lists of numeric vectors. The setters
#'   return the mesh invisibly.
#' @export
mio_polygon_block <- function(mesh, block = 1) {
  .Call(R_mio_polygon_block, mesh, .mio_block(block))
}

#' @rdname mio_polygon_block
#' @export
mio_polyhedron_block <- function(mesh, block = 1) {
  .Call(R_mio_polyhedron_block, mesh, .mio_block(block))
}

#' @rdname mio_polygon_block
#' @export
mio_add_polygon_block <- function(mesh, cell_type, rows) {
  .Call(R_mio_add_polygon_block, mesh, as.character(cell_type), rows)
  invisible(mesh)
}

#' @rdname mio_polygon_block
#' @export
mio_add_polyhedron_block <- function(mesh, cell_type, cells) {
  .Call(R_mio_add_polyhedron_block, mesh, as.character(cell_type), cells)
  invisible(mesh)
}

#' Named data arrays
#'
#' A data array shaped `(rows, components...)` in the C API comes back with the
#' dimensions reversed, so a scalar per-point field is a plain vector and a
#' 3-component vector field is a `3 x num_points` matrix -- the same memory,
#' read column-major. The dtype it was stored as is attached as a `"dtype"`
#' attribute; the values themselves are always `double`.
#'
#' @param mesh A `mio_mesh` object.
#' @param name The array name.
#' @param block A **1-based** cell-block index.
#' @return The `*_names()` functions return a character vector in ascending
#'   lexicographic order (identical on every mesh backend).
#'   `mio_cell_data_num_blocks()` returns a single number. The accessors return
#'   a numeric vector or array.
#' @export
mio_point_data_names <- function(mesh) .Call(R_mio_point_data_names, mesh)

#' @rdname mio_point_data_names
#' @export
mio_cell_data_names <- function(mesh) .Call(R_mio_cell_data_names, mesh)

#' @rdname mio_point_data_names
#' @export
mio_field_data_names <- function(mesh) .Call(R_mio_field_data_names, mesh)

#' @rdname mio_point_data_names
#' @export
mio_cell_data_num_blocks <- function(mesh, name) {
  .Call(R_mio_cell_data_num_blocks, mesh, as.character(name))
}

#' @rdname mio_point_data_names
#' @export
mio_point_data <- function(mesh, name) .Call(R_mio_point_data, mesh, as.character(name))

#' @rdname mio_point_data_names
#' @export
mio_cell_data <- function(mesh, name, block = 1) {
  .Call(R_mio_cell_data, mesh, as.character(name), .mio_block(block))
}

#' @rdname mio_point_data_names
#' @export
mio_field_data <- function(mesh, name) .Call(R_mio_field_data, mesh, as.character(name))

# --- building -----------------------------------------------------------------

#' Build a mesh from raw arrays
#'
#' All of these **copy** the caller's memory, so the R objects can be reclaimed
#' as soon as the call returns.
#'
#' `mio_set_points()` takes a `dim x num_points` matrix and
#' `mio_add_cell_block()` a `nodes_per_cell x num_cells` matrix of **1-based**
#' node indices; the shift to the core's 0-based indexing happens in the copy.
#'
#' `mio_append_cell_data()` must be called once per cell block, in block order,
#' after the blocks have been added.
#'
#' @param mesh A `mio_mesh` object.
#' @param points A `dim x num_points` numeric matrix.
#' @param cell_type A meshio cell-type name, e.g. `"tetra"`.
#' @param conn A `nodes_per_cell x num_cells` matrix of 1-based node indices.
#' @param name The array name.
#' @param data A numeric vector or array shaped `(components..., rows)`.
#' @return `NULL`, invisibly.
#' @examples
#' m <- mio_mesh()
#' mio_set_points(m, matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0), nrow = 3))
#' mio_add_cell_block(m, "triangle", matrix(c(1, 2, 3), nrow = 3))
#' mio_num_cells(m)
#' mio_release(m)
#' @export
mio_set_points <- function(mesh, points) {
  invisible(.Call(R_mio_set_points, mesh, as.matrix(points)))
}

#' @rdname mio_set_points
#' @export
mio_add_cell_block <- function(mesh, cell_type, conn) {
  invisible(.Call(R_mio_add_cell_block, mesh, as.character(cell_type), as.matrix(conn)))
}

#' @rdname mio_set_points
#' @export
mio_add_point_data <- function(mesh, name, data) {
  invisible(.Call(R_mio_add_point_data, mesh, as.character(name), data))
}

#' @rdname mio_set_points
#' @export
mio_append_cell_data <- function(mesh, name, data) {
  invisible(.Call(R_mio_append_cell_data, mesh, as.character(name), data))
}

#' @rdname mio_set_points
#' @export
mio_add_field_data <- function(mesh, name, data) {
  invisible(.Call(R_mio_add_field_data, mesh, as.character(name), data))
}

# --- named regions ------------------------------------------------------------

#' Named regions
#'
#' A region is a named group of mesh entities: a Gmsh physical group, an Exodus
#' block or node/side set, an Abaqus `*NSET`/`*ELSET`/`*SURFACE`, a MED family,
#' a Kratos SubModelPart.
#'
#' Entries are **1-based** here, with one exception: for a `"side"` region the
#' second row is a *facet ordinal within the cell type*, not a mesh index, so it
#' is passed through unshifted. Cell indices are **global and block-major**
#' (block 1's cells first, then block 2's).
#'
#' @param mesh A `mio_mesh` object.
#' @param name The region name.
#' @param kind One of `"point"`, `"cell"`, `"side"`.
#' @param entries A numeric vector of indices, or a `stride x n` matrix
#'   (`stride` is 2 for `"side"`, 1 otherwise).
#' @param dim Topological dimension, or `-1` when unspecified.
#' @param tag Format-native integer id, or `-1` when there is none.
#' @return `mio_regions()` returns a list of lists with `name`, `kind`, `dim`,
#'   `tag` and an `entries` matrix. `mio_add_region()` returns `NULL`
#'   invisibly.
#' @export
mio_regions <- function(mesh) .Call(R_mio_regions, mesh)

#' @rdname mio_regions
#' @export
mio_add_region <- function(mesh, name, kind, entries, dim = -1L, tag = -1L) {
  invisible(.Call(
    R_mio_add_region, mesh, as.character(name), as.character(kind),
    entries, as.integer(dim), as.numeric(tag)
  ))
}

# --- operations ---------------------------------------------------------------

#' Mesh operations
#'
#' Computations on a mesh rather than file formats. Each returns a new mesh (or
#' a list containing one); the input is never modified.
#'
#' @param mesh,source,target,a,b `mio_mesh` objects.
#' @param meshes A list of `mio_mesh` objects.
#' @param record_parent_ids Attach a cell-data array naming each output cell's
#'   source cell.
#' @param record_ids Attach `crop:`/`partition:` original-id arrays.
#' @param linearize Emit corner nodes only (triangle/quad output).
#' @param matrix A 4x4 affine matrix; point `p` maps to `M %*% c(p, 1)`.
#' @param rotate_vector_data Also rotate vector (trailing dim 3) and tensor
#'   (trailing dim 9) point data by the transform's 3x3 linear block.
#' @param weld Fuse coincident points within `atol`.
#' @param atol Absolute coincidence tolerance.
#' @param remove_orphans Drop points referenced by no cell.
#' @param drop_degenerate Drop degenerate cells.
#' @param drop_duplicate_cells Drop exact-duplicate cells.
#' @param method For `mio_smooth()`, `"taubin"`, `"laplacian"` or `"odt"`
#'   (optimal-Delaunay-triangulation smoothing -- tet-only, moves each free
#'   interior vertex toward the volume-weighted average of its incident
#'   tets' circumcenters); for `mio_reorder()`, `"rcm"`, `"morton"` or
#'   `"hilbert"`; for `mio_partition()`, `"sfc"`, `"kahip"` or `"auto"`; for
#'   `mio_interpolate()`, `"nearest"` or `"barycentric"`.
#' @param iterations Smoothing iterations.
#' @param lambda Relaxation factor in (0, 1). **Negative means "this method's
#'   own default"**: 0.5 for laplacian, 0.33 for taubin, 0.9 for odt.
#' @param mu Taubin's un-shrinking factor, which must satisfy `mu < -lambda < 0`.
#'   Silently ignored by laplacian and odt.
#' @param fix_boundary Pin every node on a boundary facet.
#' @param preserve_features Pin boundary nodes whose incident facets meet above
#'   `feature_angle`.
#' @param preserve_boundary Pin boundary vertices during decimation.
#' @param feature_angle Feature angle in degrees (30 is the `vtkFeatureEdges`
#'   convention).
#' @param guard_inversion Reject any move that would flip a cell's signed
#'   measure.
#' @param lo,hi Bounding-box corners (3 numbers each).
#' @param point,normal,origin Plane definition (3 numbers each).
#' @param mode `"all"` (every node inside) or `"any"` (at least one).
#' @param array The point-data array to contour.
#' @param isovalues One or more level values.
#' @param component Component of a multi-component array. For `mio_isosurface`
#'   a negative value means the row magnitude; for `mio_gradient` it means
#'   **every** component -- deliberately the opposite sentinel.
#' @param op For `mio_gradient`: `"gradient"`, `"divergence"` or `"curl"`.
#' @param method For `mio_gradient`: `"green-gauss"` or `"least-squares"`. For
#'   `mio_estimate_error`: only `"zz"` exists today; `""` selects it.
#' @param location For `mio_gradient`: `"cell"` or `"point"`.
#' @param output Output array name; `""` uses `<array>:<op>` for
#'   `mio_gradient`, `"error:zz"` for `mio_estimate_error`.
#' @param marking For `mio_estimate_error`: `"none"` (default), `"absolute"`,
#'   `"fraction"`, or `"dorfler"`.
#' @param marking_value For `mio_estimate_error`: meaning depends on
#'   `marking` -- an absolute indicator threshold, a fraction in `(0, 1]` of
#'   cells, or the Doerfler bulk fraction theta in `(0, 1]`. Ignored for
#'   `"none"`.
#' @param marked For `mio_estimate_error`: the marking array name; `""` uses
#'   `"error:marked"`. Ignored when `marking` is `"none"`.
#' @param dims Cell counts `c(nx, ny, nz)` for `mio_grid()`.
#' @param origin Lattice lo corner.
#' @param spacing Cell size per axis.
#' @param max_cells Refuse a grid larger than this, by name.
#' @param resolution Cell counts for `mio_voxelize()`; give exactly one of
#'   this and `cell_size`.
#' @param cell_size Cubic cell size; `0` means unset.
#' @param bounds Explicit `c(xlo, ylo, zlo, xhi, yhi, zhi)`.
#' @param padding Grow the box by this on every side.
#' @param padding_relative Grow the box by this fraction of its diagonal.
#' @param fill `"all"`, `"surface"` or `"inside"`.
#' @param sign `"pseudonormal"`, `"winding-number"` or `"unsigned"`.
#' @param attach_occupancy Attach the `voxel:occupancy` array.
#' @param watertight_check `"off"`, `"warn"` or `"error"`.
#' @param points A `3 x n` matrix of query coordinates.
#' @param band Clamp distances beyond this; `0` is the full field.
#' @param record_inside Attach the `sdf:inside` array.
#' @param surface The surface to measure distance to.
#' @param overwrite Replace an existing array of the output name instead of
#'   failing.
#' @param source_tag Add an integer `source_mesh_id` cell-data array.
#' @param data_policy `"intersection"` (keep keys present in every input) or
#'   `"fill"` (union, NaN for missing rows).
#' @param arrays Source array names to transfer; `NULL` means every source
#'   point-data array.
#' @param extrapolate Give a target point outside the source domain the nearest
#'   source value instead of `default_value`.
#' @param default_value Fill value outside the source domain.
#' @param on_conflict `"error"`, `"overwrite"` or `"suffix"`.
#' @param rtol Relative tolerance.
#' @param unordered Match points by spatial proximity instead of by index.
#' @param by `"type"`, `"component"`, or `"region"`/`"tag"`.
#' @param tag_name For `by = "region"`, the integer cell-data array name.
#' @param convert_mode `"linearize"`, `"simplexify"` or `"elevate"`.
#' @param target_group_size Approximate member cells per `mio_agglomerate()`
#'   output group; must be at least 1 (`1` groups every cell by itself).
#' @param levels How many times to apply the refinement templates.
#' @param ratio Fraction of faces to KEEP, in (0, 1]; negative = unset.
#' @param target_faces Absolute face count to stop at; negative = unset.
#' @param max_error Quadric-error ceiling; negative = unset.
#' @param placement `"optimal"`, `"midpoint"` or `"endpoint"`.
#' @param nparts Number of partitions.
#' @param imbalance KaHIP only: allowed imbalance fraction in (0, 1).
#' @param partition_mode KaHIP only: `"fast"`, `"eco"` or `"strong"`.
#' @param seed KaHIP only: random seed.
#' @param ghost_layers Grow each piece by this many shared-node BFS layers of
#'   other parts' cells (a halo), tagged `partition:ghost` (0 = owned).
#' @param weights Name of a scalar numeric cell-data array of per-cell weights.
#' @return A `mio_mesh`, or a named list containing one plus the operation's
#'   maps and counters. `mio_quality_counts()`, `mio_stats()` and `mio_diff()`
#'   return named lists; `mio_meshes_equal()` a single logical;
#'   `mio_compute_bandwidth()` a single number; `mio_split()` and
#'   `mio_partition()` lists of pieces; `mio_partition_labels()` a numeric
#'   vector of part ids.
#' @examples
#' \dontrun{
#' m <- mio_read("bracket.msh")
#' surf <- mio_extract_surface(m)
#' mio_write(surf, "bracket_surface.vtu")
#' }
#' @export
mio_extract_surface <- function(mesh, record_parent_ids = FALSE) {
  .Call(R_mio_extract_surface, mesh, isTRUE(record_parent_ids))
}

#' @rdname mio_extract_surface
#' @export
mio_extract_skin <- function(mesh, linearize = FALSE) {
  .Call(R_mio_extract_skin, mesh, isTRUE(linearize))
}

#' @rdname mio_extract_surface
#' @export
mio_attach_quality <- function(mesh) .Call(R_mio_attach_quality, mesh)

#' @rdname mio_extract_surface
#' @export
mio_quality_counts <- function(mesh) .Call(R_mio_quality_counts, mesh)

#' @rdname mio_extract_surface
#' @export
mio_transform <- function(mesh, matrix, rotate_vector_data = FALSE) {
  .Call(R_mio_transform, mesh, as.numeric(matrix), isTRUE(rotate_vector_data))
}

#' @rdname mio_extract_surface
#' @export
mio_clean <- function(mesh, weld = FALSE, atol = 1e-12, remove_orphans = TRUE,
                      drop_degenerate = TRUE, drop_duplicate_cells = TRUE) {
  .Call(
    R_mio_clean, mesh, isTRUE(weld), as.numeric(atol), isTRUE(remove_orphans),
    isTRUE(drop_degenerate), isTRUE(drop_duplicate_cells)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_smooth <- function(mesh, method = "taubin", iterations = 10L, lambda = -1,
                       mu = -0.34, fix_boundary = TRUE, preserve_features = TRUE,
                       feature_angle = 30, guard_inversion = TRUE) {
  .Call(
    R_mio_smooth, mesh, as.character(method), as.integer(iterations),
    as.numeric(lambda), as.numeric(mu), isTRUE(fix_boundary),
    isTRUE(preserve_features), as.numeric(feature_angle), isTRUE(guard_inversion)
  )
}

#' ODT-remesh a tetrahedral mesh
#'
#' Raise a tetrahedral mesh's worst element quality by relocating vertices AND
#' flipping connectivity (2-3/3-2, predicate-free). The genuine "ODT remeshing"
#' sibling of [mio_remesh_volume()] (which generates a fresh lattice mesh) and
#' of `mio_smooth(method = "odt")` (which only moves points on fixed
#' connectivity). Tet-only. The point set is invariant, so point data and named
#' Point regions carry; cell data and Cell/Side regions are dropped. With
#' `preserve_boundary` the boundary surface is exactly preserved. See
#' `doc/optimize_volume.md`.
#'
#' @param mesh a `mio_mesh` external pointer (tetra-only volume mesh).
#' @param max_iterations optimisation sweeps; stops early at a fixed point.
#' @param relocate run the ODT vertex-relocation half of each sweep.
#' @param flip run the topological-flip half of each sweep.
#' @param preserve_boundary pin boundary vertices during relocation.
#' @param min_improvement strict scaled-Jacobian gain a flip must deliver.
#' @return a named list with `mesh` and the counters `num_flips`,
#'   `num_23_flips`, `num_32_flips`, `num_vertices_moved`, `num_tets`,
#'   `min_quality_before`, `min_quality_after`.
#' @export
mio_optimize_volume <- function(mesh, max_iterations = 10L, relocate = TRUE, flip = TRUE,
                                preserve_boundary = TRUE, min_improvement = 1e-6) {
  .Call(
    R_mio_optimize_volume, mesh, as.integer(max_iterations), isTRUE(relocate),
    isTRUE(flip), isTRUE(preserve_boundary), as.numeric(min_improvement)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_crop_bbox <- function(mesh, lo, hi, mode = "all", record_ids = FALSE) {
  .Call(
    R_mio_crop_bbox, mesh, as.numeric(lo), as.numeric(hi), .mio_mode(mode),
    isTRUE(record_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_crop_plane <- function(mesh, point, normal, mode = "all", record_ids = FALSE) {
  .Call(
    R_mio_crop_plane, mesh, as.numeric(point), as.numeric(normal), .mio_mode(mode),
    isTRUE(record_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_slice <- function(mesh, origin, normal, record_parent_ids = FALSE) {
  .Call(
    R_mio_slice, mesh, as.numeric(origin), as.numeric(normal),
    isTRUE(record_parent_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_isosurface <- function(mesh, array, isovalues, component = -1L,
                           record_parent_ids = FALSE) {
  .Call(
    R_mio_isosurface, mesh, as.character(array), as.numeric(isovalues),
    as.integer(component), isTRUE(record_parent_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_gradient <- function(mesh, array, op = "gradient", method = "green-gauss",
                         location = "cell", output = "", component = -1L,
                         overwrite = FALSE) {
  .Call(
    R_mio_gradient, mesh, as.character(array), as.character(op),
    as.character(method), as.character(location), as.character(output),
    as.integer(component), isTRUE(overwrite)
  )
}

#' Hessian (second derivative) of a scalar field
#'
#' The Hessian of a **scalar point-data** field -- [mio_gradient()]'s
#' companion one order further, for curvature-based adaptive refinement.
#'
#' A composition of TWO [mio_gradient()] calls, not a new numerical kernel:
#' the field is differentiated once (point location), then that `(n, 3)`
#' gradient is differentiated again with the default `"gradient"` operator,
#' producing `(n, 9)` -- the flattened row-major 3x3 Hessian, `H[i,j]` at
#' index `i*3+j`. `method` is forwarded to BOTH internal passes. The result
#' is named `"<array>:hessian"` unless `output` overrides it.
#'
#' A field that is at most LINEAR has an exactly zero Hessian everywhere --
#' the one mesh-shape-independent guarantee. For a genuinely quadratic field
#' the composition is exact on a structured/symmetric mesh away from its own
#' boundary and a good, standard, but genuinely approximate curvature
#' estimate on an irregular mesh (see `doc/hessian.md`). Input must have
#' exactly one component -- a vector field's Hessian is a separate quantity
#' per component.
#'
#' Cells that cannot be evaluated yield `NaN` and are counted in
#' `num_skipped`; least-squares cells with a degenerate neighbourhood in
#' either internal pass fall back to Green-Gauss and are counted in
#' `num_fallback` (summed over both passes). A curvature-driven refinement
#' indicator needs no new function: `norm(...)` in [mio_data_calc()] on the
#' 9-component output is exactly its Frobenius norm, ready for
#' [mio_refine()]'s `where` selector.
#'
#' @param mesh A `mio_mesh`.
#' @param array Name of the scalar `point_data` array to differentiate
#'   twice.
#' @param method `"green-gauss"` (default) or `"least-squares"`, forwarded
#'   to both internal `mio_gradient` passes.
#' @param location `"cell"` (default) or `"point"` for the result.
#' @param output Output array name; empty selects `"<array>:hessian"`.
#' @param overwrite Allow replacing an existing array of the output name.
#' @return A list of `mesh`, `num_skipped` and `num_fallback`.
#' @export
mio_hessian <- function(mesh, array, method = "green-gauss", location = "cell",
                        output = "", overwrite = FALSE) {
  .Call(
    R_mio_hessian, mesh, as.character(array), as.character(method),
    as.character(location), as.character(output), isTRUE(overwrite)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_estimate_error <- function(mesh, array, method = "zz", marking = "none",
                               marking_value = 0.0, output = "", marked = "",
                               overwrite = FALSE) {
  .Call(
    R_mio_estimate_error, mesh, as.character(array), as.character(method),
    as.character(marking), as.numeric(marking_value), as.character(output),
    as.character(marked), isTRUE(overwrite)
  )
}

#' Remesh a surface by approximated centroidal Voronoi diagram (ACVD)
#' clustering
#'
#' Replace a surface mesh's triangulation with a new, near-uniformly-sized,
#' well-shaped one at `num_clusters` vertices. Unlike every other
#' resolution-changing operation, the output has NEW points and NEW
#' connectivity with no correspondence to the input -- point/cell data and
#' named regions are dropped, field data is carried.
#'
#' `metric` is `"isotropic"` (default; area-weighted centroidal distance,
#' fast, rounds sharp features), `"quadric"` (Garland-Heckbert quadric
#' error, preserves sharp edges/corners at extra cost per candidate move),
#' or `"anisotropic"` (clusters shaped by a local curvature tensor --
#' elongated along low-curvature directions -- see `max_anisotropy`).
#' `subdivide` defaults to automatic (`-1`): the smallest count of uniform
#' `refine` passes reaching `subsample_ratio` items per cluster, capped at
#' `max_subdivide`; `0` disables subdivision. `num_isolated_clusters` and
#' `num_non_manifold_vertices` in the result are two distinct causes of
#' non-manifold output (disconnected clusters vs. "bowtie" vertices) that
#' repair could not fully fix; check both rather than assuming.
#'
#' Goes through `mio_remesh_ex`/`mio_remesh_opts` rather than the flat
#' `mio_remesh` -- the `mio_refine_ex` precedent, needed because
#' `mio_remesh` is a flat C function with no room to grow (it already
#' changed once).
#'
#' Attribution: the isotropic clustering engine is derived from
#' \href{https://github.com/pyvista/pyacvd}{pyacvd} (MIT, (c) 2017-2024 The
#' PyVista Developers), itself an independent implementation of the
#' published research of S. Valette and J.-M. Chassery, not of the
#' CeCILL-B licensed ACVD project. See `doc/remesh.md`.
#'
#' @param mesh A `mio_mesh` (surface only).
#' @param num_clusters Number of clusters, i.e. output vertices aimed for;
#'   must be >= 4 and <= the subdivided input's vertex count.
#' @param subdivide Uniform refine passes before clustering; negative (the
#'   default) picks the smallest count reaching `subsample_ratio`
#'   items/cluster, capped at `max_subdivide`; `0` disables subdivision.
#' @param subsample_ratio Items per cluster targeted by automatic
#'   subdivision.
#' @param max_subdivide Ceiling on automatic subdivision (ignored when
#'   `subdivide` is set explicitly).
#' @param max_iterations Maximum energy-minimisation sweeps per pass.
#' @param max_repair_passes "Split disconnected clusters, minimise again"
#'   passes; `0` skips repair.
#' @param metric `"isotropic"` (default), `"quadric"` or `"anisotropic"`.
#' @param gradation Curvature-gradation exponent `gamma` in the item weight
#'   `area * kappa^gamma`; `0.0` (default) disables gradation entirely and
#'   reproduces plain area weighting.
#' @param preserve_boundary Detect the input's open boundary (if any), seed
#'   it before the interior, and emit a `line` dual cell along boundary
#'   edges whose endpoints land in different clusters. A no-op on a closed
#'   mesh (`TRUE` by default).
#' @param max_anisotropy Under `metric = "anisotropic"`, the maximum ratio
#'   between the two in-plane target edge lengths a per-vertex curvature
#'   tensor may request; `1.0` recovers the isotropic shape exactly. Must be
#'   at least `1.0`. An error to set away from its default (`4.0`, a
#'   measured value -- see `kRemeshDefaultMaxAnisotropy`'s doc comment in
#'   `remesh.hpp`) under any other metric.
#' @return A list of `mesh`, `num_clusters`, `num_iterations`,
#'   `subdivide_applied`, `num_isolated_clusters` and
#'   `num_non_manifold_vertices`.
#' @export
mio_remesh <- function(mesh, num_clusters, subdivide = -1L, subsample_ratio = 10.0,
                       max_subdivide = 4L, max_iterations = 100L,
                       max_repair_passes = 10L, metric = "isotropic", gradation = 0.0,
                       preserve_boundary = TRUE, max_anisotropy = 4.0) {
  .Call(
    R_mio_remesh, mesh, as.integer(num_clusters), as.integer(subdivide),
    as.numeric(subsample_ratio), as.integer(max_subdivide),
    as.integer(max_iterations), as.integer(max_repair_passes),
    as.character(metric), as.numeric(gradation), isTRUE(preserve_boundary),
    as.numeric(max_anisotropy)
  )
}


#' Per-vertex mean and Gaussian curvature of a surface mesh
#'
#' `K` (Gaussian curvature) by the angle defect, `H` (mean curvature) by the
#' cotangent Laplace-Beltrami operator -- the estimators the
#' discrete-differential-geometry convergence results are about, and NVIDIA
#' PhysicsNeMo's own `gaussian_curvature_vertices`/`mean_curvature_vertices`
#' use. The signed distance's natural companion as a node feature: `sdf` says
#' how far a point is from the surface, this says how the surface bends
#' there.
#'
#' Writes `curvature:mean` and `curvature:gaussian` as point data, optionally
#' `curvature:area` (the dual area each was divided by) and
#' `curvature:principal` (an `(n, 2)` matrix, `k1 >= k2`). Geometry,
#' connectivity and existing data are carried through unchanged. Triangles
#' come from the same fan `convert_cells(simplexify)` uses, so a quad mesh's
#' curvature is the curvature of its canonical triangulation; a volume or
#' polyhedron block is refused by name pointing at `extract_surface`, a
#' higher-order one pointing at `linearize`.
#'
#' `total_angle_defect` is the oracle: on a CLOSED surface it is `2*pi*chi`
#' exactly -- `4*pi` for anything sphere-like -- whatever the tessellation and
#' whichever `dual_area`, so a value that is not that means the input is not
#' closed or the result is not sane.
#'
#' `H` is orientation-dependent and `K` is not, so check
#' `quality$inconsistent_pairs` before trusting a sign: a nonzero count means
#' facets disagree about which side is out. This never repairs its input.
#'
#' @param mesh A `mio_mesh` (surface only).
#' @param mean Attach `curvature:mean` (`TRUE` by default).
#' @param gaussian Attach `curvature:gaussian` (`TRUE` by default).
#' @param dual_area `"mixed-voronoi"` (default, converges better on an
#'   irregular tessellation) or `"barycentric"` (cruder but branch-free).
#' @param include_boundary Compute a biased value at boundary vertices
#'   instead of leaving them `NaN`. Isolated vertices are `NaN` either way.
#' @param record_area Also attach `curvature:area`.
#' @param record_principal Also attach `curvature:principal`.
#' @param region Restrict to this named cell region; `""` (default) takes
#'   every surface cell.
#' @return A list of `mesh`, `quality` (a list of `boundary_edges`,
#'   `non_manifold_edges`, `inconsistent_pairs`, `degenerate_triangles`,
#'   `watertight`), `num_boundary`, `num_isolated`, `num_degenerate` and
#'   `total_angle_defect`.
#' @export
mio_compute_curvature <- function(mesh, mean = TRUE, gaussian = TRUE,
                                  dual_area = "mixed-voronoi",
                                  include_boundary = FALSE, record_area = FALSE,
                                  record_principal = FALSE, region = "") {
  .Call(
    R_mio_compute_curvature, mesh, isTRUE(mean), isTRUE(gaussian),
    as.character(dual_area), isTRUE(include_boundary), isTRUE(record_area),
    isTRUE(record_principal), as.character(region)
  )
}

#' Repair a surface mesh's orientation, holes and pinched vertices
#'
#' Surface repair beyond `mio_clean()`: weld (opt-in) -> triangulate (blocks
#' 1:1) -> split bowties (a vertex whose triangle star is edge-disconnected is
#' duplicated, geometry unchanged) -> orient by the topological half-edge rule
#' per connected component (two triangles sharing an edge agree iff they
#' traverse it in opposite directions -- exact, where a normal test fails
#' across a crease) -> fan-fill boundary loops of at most `max_hole_edges`
#' edges, wound to agree with the surrounding surface -> orient closed
#' components outward. Adapted from NVIDIA PhysicsNeMo's `mesh.repair`.
#'
#' Lower-dimensional blocks ride along; fill triangles land in one trailing
#' `triangle` block (only when there is one). Non-manifold *edges* are counted,
#' never split or crossed. The C++ index maps are not exposed on the flat ABI.
#'
#' @param mesh A `mio_mesh` (surface only; a volume block is refused naming
#'   `extract_surface`, a higher-order one naming `linearize`).
#' @param fix_orientation Rewind triangles so neighbours agree (`TRUE`).
#' @param orient_outward Flip closed components with a negative volume (`TRUE`).
#' @param fill_holes Fan-fill boundary loops (`TRUE`).
#' @param split_non_manifold Duplicate bowtie vertices (`TRUE`).
#' @param max_hole_edges Longest loop still filled (`10`); `<= 0` means no limit.
#' @param weld_tolerance Weld coincident points within this distance first;
#'   `0` (the default) skips the weld.
#' @param record_provenance Attach `repair:parent_point` and `repair:hole`.
#' @return A list of `mesh`, `quality_before`, `quality_after` (each a list of
#'   `boundary_edges`, `non_manifold_edges`, `inconsistent_pairs`,
#'   `degenerate_triangles`, `watertight`) and the counters `num_flipped`,
#'   `num_components`, `largest_component`, `num_oriented_outward`,
#'   `num_unorientable`, `num_vertices_split`, `num_holes_detected`,
#'   `num_holes_filled`, `num_holes_skipped`, `num_faces_added`,
#'   `num_points_added`, `points_welded`.
#' @export
mio_repair <- function(mesh, fix_orientation = TRUE, orient_outward = TRUE,
                       fill_holes = TRUE, split_non_manifold = TRUE,
                       max_hole_edges = 10, weld_tolerance = 0.0,
                       record_provenance = FALSE) {
  .Call(
    R_mio_repair, mesh, isTRUE(fix_orientation), isTRUE(orient_outward),
    isTRUE(fill_holes), isTRUE(split_non_manifold), as.numeric(max_hole_edges),
    as.numeric(weld_tolerance), isTRUE(record_provenance)
  )
}

#' Project a mesh's points onto a target surface (shrinkwrap)
#'
#' `x' = x + w (p + offset n - x)` with `p` the closest point on the target
#' surface and `n` the unit pseudonormal of the hit FEATURE (the bisector at a
#' crease, not the selected triangle's normal) -- one projection, no
#' iteration, no self-intersection guard, as in NVIDIA PhysicsNeMo's
#' `mesh.shrinkwrap`. Every point of `mesh` moves whatever cells it carries;
#' only `target` must be a surface (quads and polygons are fanned). A point
#' farther than `max_distance` from the target is left alone and counted.
#' Connectivity, data, regions and property sets pass through verbatim.
#'
#' @param mesh The `mio_mesh` whose points move.
#' @param target The surface `mio_mesh` to project onto.
#' @param offset Signed offset along the feature pseudonormal.
#' @param max_distance Leave points farther than this alone; `<= 0` means unlimited.
#' @param weights Name of a `(n,)` point-data array on the source: nonzero
#'   selects, a fractional value blends. `""` (default) moves every point.
#' @param target_region Restrict the target to this named cell region.
#' @param normal_weight `"angle"` (default) or `"area"` vertex-pseudonormal weighting.
#' @param record_distance Attach `shrinkwrap:distance`.
#' @param record_closest_cell Attach `shrinkwrap:closest_cell`.
#' @return A list of `mesh`, `quality` (the target's), `num_projected`,
#'   `num_missed`, `num_skipped` and `max_displacement`.
#' @export
mio_shrinkwrap <- function(mesh, target, offset = 0.0, max_distance = 0.0,
                           weights = "", target_region = "",
                           normal_weight = "angle", record_distance = FALSE,
                           record_closest_cell = FALSE) {
  .Call(
    R_mio_shrinkwrap, mesh, target, as.numeric(offset), as.numeric(max_distance),
    as.character(weights), as.character(target_region), as.character(normal_weight),
    isTRUE(record_distance), isTRUE(record_closest_cell)
  )
}

#' Sobolev (Helmholtz-filtered) deformation
#'
#' Solves `(M + l^2 K) u = M d` over the mesh's own P1 finite-element
#' operators -- `K` the stiffness matrix from the simplex edge Gram matrix,
#' `M` the uniform mean lumped vertex mass, matrix-free Jacobi-preconditioned
#' conjugate gradients -- and moves the points by `u`: a screened-Poisson
#' low-pass filter of the raw displacement `d` whose cutoff wavelength is
#' `length_scale`, as in NVIDIA PhysicsNeMo's `mesh.sobolev_deform`. Every
#' top-dimensional block must be a linear simplex (`line`, `triangle`,
#' `tetra`); lower-dimensional blocks ride along. Nothing is pinned by default
#' (a constant displacement is preserved exactly). Non-convergence is
#' reported, never raised: the last iterate is returned.
#'
#' @param mesh A `mio_mesh`.
#' @param array Name of the `(n, dim)` point-data displacement array.
#' @param length_scale The smoothing length; `0` applies the raw field at free points.
#' @param fixed_points_array Name of a point-data array whose nonzero entries
#'   pin their point (`""` pins nothing this way). Note R writes point data as
#'   `Float64`, which this accepts.
#' @param fix_boundary Also pin every point on a boundary facet.
#' @param record_filtered Attach `sobolev:displacement`.
#' @param max_iterations Conjugate-gradient cap (`128`).
#' @param tolerance Relative residual tolerance (`1e-10`).
#' @return A list of `mesh`, `num_iterations`, `residual`, `converged`,
#'   `num_fixed`, `num_isolated` and `max_displacement`.
#' @export
mio_sobolev_deform <- function(mesh, array, length_scale, fixed_points_array = "",
                               fix_boundary = FALSE, record_filtered = FALSE,
                               max_iterations = 128L, tolerance = 1e-10) {
  .Call(
    R_mio_sobolev_deform, mesh, as.character(array), as.numeric(length_scale),
    as.character(fixed_points_array), isTRUE(fix_boundary), isTRUE(record_filtered),
    as.integer(max_iterations), as.numeric(tolerance)
  )
}

#' Retetrahedralize a volume mesh (or a closed surface) by isosurface stuffing
#'
#' The volumetric sibling of [mio_remesh()], generating an entirely new tet
#' mesh (no point/cell maps, `point_data`/`cell_data`/named regions dropped,
#' `field_data` carried) rather than working on the input's own cells. Unlike
#' `mio_remesh`, `mesh` may be a VOLUME mesh directly (its boundary is
#' extracted internally) as well as a closed surface.
#'
#' Exactly one of `resolution`/`cell_size` must be given, sizing a
#' body-centered cubic (BCC) lattice whose uncut tets have dihedral angles
#' from a fixed, mesh-size-independent set. `warp_fraction` (default `0.35`)
#' moves lattice vertices near the surface onto it, trading a small,
#' measured chance of non-manifold boundary edges (reported as
#' `num_non_manifold_edges`) for substantially better boundary tet quality;
#' `0` disables warping and gives an exactly watertight but lower-quality
#' boundary. See `doc/remesh_volume.md` for the measured tradeoff.
#'
#' Attribution: implemented from the PUBLISHED DESCRIPTION of Labelle &
#' Shewchuk, "Isosurface Stuffing" (SIGGRAPH 2007) only -- neither the
#' paper's own reference implementation (Stellar) nor TetGen (AGPL-3.0) is
#' read or vendored here. See `doc/remesh_volume.md`.
#'
#' @param mesh A `mio_mesh` (volume mesh, or closed surface).
#' @param resolution Cell counts `c(nx, ny, nz)` of the root lattice; exactly
#'   one of `resolution`/`cell_size` must be given.
#' @param cell_size Cubic cell size of the root lattice.
#' @param bounds Explicit `c(xlo, ylo, zlo, xhi, yhi, zhi)`; `NULL` uses the
#'   surface's own bounding box.
#' @param padding Padding added to every side, in world units.
#' @param padding_relative Padding added to every side, as a fraction of the
#'   bounding-box diagonal (default `0.1`).
#' @param max_cells Refuse to generate a root lattice above this many cells.
#' @param max_tets Refuse an output with more tets than this (checked after
#'   cutting; unlike `max_cells`, no "lifts the limit" value).
#' @param warp_fraction Fraction of a lattice vertex's own shortest incident
#'   edge length within which it may be warped onto the surface; `0`
#'   disables warping (default `0.35`).
#' @param sign How lattice vertices are classified inside/outside:
#'   `"pseudonormal"` (default) or `"winding-number"`.
#' @param watertight_check What to do about an input surface that is not
#'   watertight: `"off"`, `"warn"` (default) or `"error"`.
#' @param surface_region Restrict the surface to a named `Cell` region;
#'   `""` (default) is all of it.
#' @param grid_cell_size Bucket size of the search grid; `0` (default)
#'   derives one from the triangle sizes.
#' @param max_winding_work Refuse a `"winding-number"` query above this cost.
#' @return A list of `mesh`, `num_tets`, `num_vertices_warped`,
#'   `num_tets_rejected`, `num_non_manifold_edges` and `input_quality` (the
#'   verdict for the INPUT surface, not the output).
#' @export
mio_remesh_volume <- function(mesh, resolution = NULL, cell_size = 0, bounds = NULL,
                              padding = 0, padding_relative = 0.1, max_cells = 20000000,
                              max_tets = 20000000, warp_fraction = 0.35,
                              sign = "pseudonormal", watertight_check = "warn",
                              surface_region = "", grid_cell_size = 0,
                              max_winding_work = 2e9) {
  signs <- c(unsigned = 0L, pseudonormal = 1L, `winding-number` = 2L)
  checks <- c(off = 0L, warn = 1L, error = 2L)
  if (!sign %in% names(signs)) stop("unknown sign '", sign, "'")
  if (!watertight_check %in% names(checks)) {
    stop("unknown watertight check '", watertight_check, "'")
  }
  .Call(
    R_mio_remesh_volume, mesh,
    if (is.null(resolution)) numeric(0) else as.numeric(resolution),
    as.numeric(cell_size),
    if (is.null(bounds)) numeric(0) else as.numeric(bounds),
    as.numeric(padding), as.numeric(padding_relative), as.numeric(max_cells),
    as.numeric(max_tets), as.numeric(warp_fraction), signs[[sign]],
    checks[[watertight_check]], as.character(surface_region),
    as.numeric(grid_cell_size), as.numeric(max_winding_work)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_merge <- function(meshes, weld = FALSE, atol = 1e-12, source_tag = TRUE,
                      data_policy = "intersection", drop_duplicate_cells = FALSE) {
  policy <- switch(data_policy,
    intersection = 0L, fill = 1L,
    stop("`data_policy` must be \"intersection\" or \"fill\"")
  )
  .Call(
    R_mio_merge, meshes, isTRUE(weld), as.numeric(atol), isTRUE(source_tag),
    policy, isTRUE(drop_duplicate_cells)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_interpolate <- function(source, target, method = "nearest", arrays = NULL,
                            extrapolate = FALSE, default_value = 0,
                            on_conflict = "error") {
  .Call(
    R_mio_interpolate, source, target, as.character(method),
    if (is.null(arrays)) NULL else as.character(arrays), isTRUE(extrapolate),
    as.numeric(default_value), as.character(on_conflict)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_conservative_interpolate <- function(source, target, arrays = NULL,
                                         default_value = 0, on_conflict = "error") {
  .Call(
    R_mio_conservative_interpolate, source, target,
    if (is.null(arrays)) NULL else as.character(arrays),
    as.numeric(default_value), as.character(on_conflict)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_undo_green <- function(coarse, fine) {
  .Call(R_mio_undo_green, coarse, fine)
}

#' @rdname mio_extract_surface
#' @export
mio_meshes_equal <- function(a, b, atol = 0, rtol = 0, unordered = FALSE) {
  .Call(R_mio_meshes_equal, a, b, as.numeric(atol), as.numeric(rtol), isTRUE(unordered))
}

#' @rdname mio_extract_surface
#' @export
mio_diff <- function(a, b, atol = 0, rtol = 0, unordered = FALSE) {
  .Call(R_mio_diff, a, b, as.numeric(atol), as.numeric(rtol), isTRUE(unordered))
}

#' @rdname mio_extract_surface
#' @export
mio_stats <- function(mesh) .Call(R_mio_stats, mesh)

#' @rdname mio_extract_surface
#' @export
mio_compute_bandwidth <- function(mesh) .Call(R_mio_compute_bandwidth, mesh)

#' @rdname mio_extract_surface
#' @export
mio_reorder <- function(mesh, method = "rcm") {
  .Call(R_mio_reorder, mesh, as.character(method))
}

#' @rdname mio_extract_surface
#' @export
mio_split <- function(mesh, by = "type", tag_name = NULL) {
  .Call(R_mio_split, mesh, as.character(by), tag_name)
}

#' @rdname mio_extract_surface
#' @export
mio_convert_cells <- function(mesh, convert_mode, record_parent_ids = FALSE) {
  .Call(
    R_mio_convert_cells, mesh, as.character(convert_mode),
    isTRUE(record_parent_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_subdivide <- function(mesh, record_parent_ids = FALSE) {
  .Call(R_mio_subdivide, mesh, isTRUE(record_parent_ids))
}

#' @rdname mio_extract_surface
#' @export
mio_agglomerate <- function(mesh, target_group_size = 8) {
  .Call(R_mio_agglomerate, mesh, as.numeric(target_group_size))
}

#' @rdname mio_extract_surface
#' @param cells global (block-major) **1-based** cell indices to refine. At most
#'   one of `cells`, `region` and `where_array` may be given; with none, every
#'   cell is refined.
#' @param region name of a region to refine. A cell region selects its own cells;
#'   a point region selects every cell with any node in it; a side region is an
#'   error.
#' @param where_array name of a scalar `cell_data` array to threshold, with
#'   `where_op` one of `"<"`, `"<="`, `">"`, `">="`, `"=="`, `"!="` and
#'   `where_value` the right-hand side. A non-finite value never matches.
#' @param where_op,where_value the predicate's comparison and threshold.
#' @param closure how to resolve the hanging nodes a partial refinement leaves:
#'   `"redgreen"` (default) keeps the extra refinement local, `"propagate"` is
#'   defined for every cell type but converges to uniform refinement of the whole
#'   edge-connected component, and `"balanced"` keeps the hanging nodes and only
#'   enforces 2:1 balance (the output is then **not** conforming; the constrained
#'   nodes come back in `refine:hanging`).
#' @param record_levels attach the `refine:level` `cell_data` array.
#' @param record_hierarchy attach the `refine:cell_id`/`refine:parent_id`
#'   `cell_data` arrays -- the persistent parent/child hierarchy a multigrid
#'   caller resolves across the sequence of meshes it keeps ("a link between
#'   two meshes, not a tree inside one"): an unsplit cell keeps its id and is
#'   its own parent; a split cell's children each get a fresh id and carry the
#'   parent's id. An input already carrying `refine:cell_id` is updated
#'   whatever this says. Also forces `refine:entity` to be attached even when
#'   the closure leaves no hanging node, since it already records the coarse
#'   corners each new fine node is the mean of -- the multigrid prolongation
#'   weights, which `"redgreen"`/`"propagate"` would otherwise never expose.
#' @export
mio_refine <- function(mesh, levels = 1L, record_parent_ids = FALSE,
                       cells = NULL, region = NULL, where_array = NULL,
                       where_op = "<", where_value = 0,
                       closure = "redgreen", record_levels = FALSE,
                       record_hierarchy = FALSE) {
  .Call(
    R_mio_refine, mesh, as.integer(levels), isTRUE(record_parent_ids),
    cells, region, where_array, where_op, as.numeric(where_value),
    closure, isTRUE(record_levels), isTRUE(record_hierarchy)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_decimate <- function(mesh, ratio = -1, target_faces = -1, max_error = -1,
                         placement = "optimal", preserve_boundary = TRUE,
                         preserve_features = TRUE, feature_angle = 30) {
  .Call(
    R_mio_decimate, mesh, as.numeric(ratio), as.numeric(target_faces),
    as.numeric(max_error), as.character(placement), isTRUE(preserve_boundary),
    isTRUE(preserve_features), as.numeric(feature_angle)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_partition <- function(mesh, nparts, method = "auto", imbalance = 0.03,
                          partition_mode = "eco", seed = 0L, record_ids = FALSE,
                          ghost_layers = 0L, weights = NULL) {
  .Call(
    R_mio_partition, mesh, as.integer(nparts), as.character(method),
    as.numeric(imbalance), as.character(partition_mode), as.integer(seed),
    isTRUE(record_ids), as.integer(ghost_layers), weights
  )
}

#' @rdname mio_extract_surface
#' @export
mio_partition_labels <- function(mesh, nparts, method = "auto", imbalance = 0.03,
                                 partition_mode = "eco", seed = 0L, weights = NULL) {
  .Call(
    R_mio_partition_labels, mesh, as.integer(nparts), as.character(method),
    as.numeric(imbalance), as.character(partition_mode), as.integer(seed),
    weights, mio_num_cells(mesh)
  )
}

# --- data operations ----------------------------------------------------------

#' Data operations
#'
#' These act on a mesh's point, cell and field data arrays and **never touch
#' geometry**: points, connectivity and block order come through bit-identical.
#'
#' The combined `data_manage` of the C++ API is not exposed across the C ABI (a
#' documented gap); `mio_data_drop()`, `mio_data_keep()` and
#' `mio_data_rename()` compose to the same effect.
#'
#' @param mesh A `mio_mesh` object.
#' @param location One of `"point"`, `"cell"`, `"field"`.
#' @param names Array names. `NULL` or `character(0)` means "every array at that
#'   location" for the averaging and conditioning entry points, and "nothing"
#'   for `mio_data_drop()`.
#' @param ignore_missing Skip names that do not exist.
#' @param from,to Old and new array names.
#' @param suffix Write to `name + suffix` instead of in place.
#' @param weight `"uniform"` or `"measure"` (weight by cell measure).
#' @param expression An elementwise expression over the arrays at `location`.
#'   The grammar accepts `+ - * /`, unary minus, parentheses, numeric literals,
#'   array names and the functions `abs`, `sqrt`, `min`, `max` and `norm`.
#'   Nothing else is evaluated: there is no arbitrary-code path.
#' @param output_name Name of the array to create.
#' @param overwrite Allow replacing an existing array.
#' @param cond_mode `"clamp"`, `"normalize"` or `"standardize"`.
#' @param lo,hi Clamp bounds, or the normalize target range.
#' @param scope `"component"` (each trailing component independently) or
#'   `"magnitude"` (rescale whole rows, preserving direction).
#' @param nan_policy `"ignore"`, `"replace"` or `"fail"`. Non-finite values are
#'   always excluded from reductions regardless.
#' @param nan_replacement Used when `nan_policy` is `"replace"`.
#' @return A new `mio_mesh`, except `mio_data_info()`, which returns a list of
#'   per-array summaries.
#' @export
mio_data_drop <- function(mesh, location, names, ignore_missing = FALSE) {
  .Call(
    R_mio_data_drop, mesh, .mio_location(location),
    if (is.null(names)) NULL else as.character(names), isTRUE(ignore_missing)
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_keep <- function(mesh, location, names, ignore_missing = FALSE) {
  .Call(
    R_mio_data_keep, mesh, .mio_location(location),
    if (is.null(names)) NULL else as.character(names), isTRUE(ignore_missing)
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_rename <- function(mesh, location, from, to) {
  .Call(
    R_mio_data_rename, mesh, .mio_location(location), as.character(from),
    as.character(to)
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_point_to_cell <- function(mesh, names = NULL, suffix = NULL) {
  .Call(
    R_mio_data_point_to_cell, mesh,
    if (is.null(names)) NULL else as.character(names), suffix
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_cell_to_point <- function(mesh, names = NULL, weight = "uniform",
                                   suffix = NULL) {
  w <- switch(weight,
    uniform = 0L, measure = 1L,
    stop("`weight` must be \"uniform\" or \"measure\"")
  )
  .Call(
    R_mio_data_cell_to_point, mesh,
    if (is.null(names)) NULL else as.character(names), w, suffix
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_calc <- function(mesh, expression, output_name, location = "point",
                          overwrite = FALSE) {
  .Call(
    R_mio_data_calc, mesh, as.character(expression), .mio_location(location),
    as.character(output_name), isTRUE(overwrite)
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_condition <- function(mesh, location, names = NULL, cond_mode = "clamp",
                               lo = 0, hi = 1, scope = "component",
                               nan_policy = "ignore", nan_replacement = 0,
                               suffix = NULL) {
  md <- switch(cond_mode,
    clamp = 0L, normalize = 1L, standardize = 2L,
    stop("`cond_mode` must be \"clamp\", \"normalize\" or \"standardize\"")
  )
  sc <- switch(scope,
    component = 0L, magnitude = 1L,
    stop("`scope` must be \"component\" or \"magnitude\"")
  )
  np <- switch(nan_policy,
    ignore = 0L, replace = 1L, fail = 2L,
    stop("`nan_policy` must be \"ignore\", \"replace\" or \"fail\"")
  )
  .Call(
    R_mio_data_condition, mesh, .mio_location(location),
    if (is.null(names)) NULL else as.character(names), md, as.numeric(lo),
    as.numeric(hi), sc, np, as.numeric(nan_replacement), suffix
  )
}

#' @rdname mio_data_drop
#' @export
mio_data_info <- function(mesh) .Call(R_mio_data_info, mesh)

#' Cell-measure-weighted field integration
#'
#' Total/mean of one or more `cell_data` arrays, weighted by each cell's own
#' length/area/volume -- [mio_gradient()]'s integration counterpart
#' (`mio_gradient` differentiates a field, this integrates one). A cell whose
#' measure is not computable (ragged, unsupported type, or degenerate), or
#' whose value is non-finite in a given component, is excluded from that
#' component's numerator *and* denominator, never given a fallback weight.
#' Reported for the whole mesh (`domain`) and independently for every named
#' Cell region (`regions`) -- a cell in two regions contributes fully to
#' both, one in none contributes to neither. A `point_data`-only name fails,
#' naming [mio_data_point_to_cell()] as the fix. See
#' `doc/field_integration.md`.
#'
#' @param mesh A `mio_mesh`.
#' @param names `cell_data` array names to integrate, or `NULL` for every
#'   `cell_data` array.
#' @return A list of per-array summaries, each with `name`, `num_components`,
#'   `domain` (`num_cells`, `num_skipped`, and a `num_components x 4`
#'   `total`/`mean`/`domain_measure`/`num_nan` `components` matrix) and
#'   `regions` (a list of the same shape, one per named Cell region present).
#' @export
mio_data_integrate <- function(mesh, names = NULL) {
  .Call(R_mio_data_integrate, mesh, if (is.null(names)) NULL else as.character(names))
}

# --- transient (time-series) XDMF writing -------------------------------------

#' Write a transient (time-series) XDMF file
#'
#' The write half of what `mio_read(time_step = )` and
#' `mio_read_metadata()$time_values` expose on the read side, and the one writer
#' `mio_write()` cannot express: a series is a **stateful** multi-call object.
#' The grid is written once with `mio_xdmf_series_write_points_cells()` and each
#' solve appends a cheap step with `mio_xdmf_series_write_data()`.
#'
#' The `.xdmf` light data is **buffered until the series is finalized**, so the
#' file is only readable after `mio_xdmf_series_flush()`,
#' `mio_xdmf_series_finalize()` or
#' `mio_xdmf_series_release()`. Heavy data for `data_format = "HDF"` goes to a
#' `<path minus extension>.h5` *sibling* of the `.xdmf`.
#'
#' The handle is an external pointer with a registered finalizer, exactly like
#' [mio_mesh()], so it is released (and finalized) on garbage collection;
#' `mio_xdmf_series_release()` does it immediately and is idempotent. A write
#' failure during that implicit finalize cannot be reported from a finalizer,
#' which is why `mio_xdmf_series_finalize()` exists as an explicit call.
#'
#' @param path Path of the `.xdmf`/`.xmf` light-data file to write.
#' @param data_format `"HDF"` (the default; needs an HDF5-enabled library),
#'   `"XML"` (everything inline in the `.xdmf`) or `"Binary"`. An unknown
#'   format, or `"HDF"` against a library built without HDF5, is an error.
#' @param gzip_level Gzip level for `"HDF"` datasets; negative (the default)
#'   means no compression. Ignored by the other formats.
#' @param series A `mio_xdmf_series` object.
#' @param mesh A `mio_mesh`. `mio_xdmf_series_write_points_cells()` uses only
#'   its points and cells; `mio_xdmf_series_write_data()` uses only its
#'   `point_data`/`cell_data`, so a solver can pass the very object it updates
#'   in place. Its cell blocks must match those of the static grid.
#' @param time The step's time value.
#' @param x A `mio_xdmf_series` object.
#' @param mode `"truncate"` (default) starts a fresh series; `"append"`
#'   continues the one already at `path`, if any. A path with no file yet is
#'   simply a fresh series, so a restartable script can always pass `"append"`.
#' @param auto_flush Flush the light data after every
#'   `mio_xdmf_series_write_data()` (default `FALSE`). Off by default because a
#'   flush re-serializes the whole document, making per-step flushing quadratic
#'   in the step count; call `mio_xdmf_series_flush()` on your own cadence
#'   instead.
#' @param ... Ignored.
#' @return `mio_xdmf_series()` returns a new `mio_xdmf_series`.
#'   `mio_xdmf_series_num_steps()` returns the number of steps written so far
#'   (a `double`; R has no native 64-bit integer). The write, finalize and
#'   release functions return `NULL` invisibly.
#'   `mio_xdmf_series_is_open()` returns a single logical.
#' @examples
#' \dontrun{
#' s <- mio_xdmf_series("simulation.xdmf", data_format = "XML")
#' mio_xdmf_series_write_points_cells(s, mesh)
#' for (k in 0:9) {
#'   mio_xdmf_series_write_data(s, k * 0.1, mesh)
#' }
#' mio_xdmf_series_finalize(s)
#' mio_xdmf_series_release(s)
#' }
#' @export
mio_xdmf_series <- function(path, data_format = "HDF", gzip_level = -1L,
                            mode = c("truncate", "append"), auto_flush = FALSE) {
  mode <- match.arg(mode)
  .Call(
    R_mio_xdmf_series_create, as.character(path),
    if (is.null(data_format)) NULL else as.character(data_format),
    as.integer(gzip_level), mode, isTRUE(auto_flush)
  )
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_write_points_cells <- function(series, mesh) {
  invisible(.Call(R_mio_xdmf_series_write_points_cells, series, mesh))
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_write_data <- function(series, time, mesh) {
  invisible(.Call(R_mio_xdmf_series_write_data, series, as.numeric(time), mesh))
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_flush <- function(series) {
  invisible(.Call(R_mio_xdmf_series_flush, series))
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_finalized <- function(series) .Call(R_mio_xdmf_series_finalized, series)

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_finalize <- function(series) {
  invisible(.Call(R_mio_xdmf_series_finalize, series))
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_num_steps <- function(series) .Call(R_mio_xdmf_series_num_steps, series)

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_release <- function(series) {
  invisible(.Call(R_mio_xdmf_series_release, series))
}

#' @rdname mio_xdmf_series
#' @export
mio_xdmf_series_is_open <- function(series) .Call(R_mio_xdmf_series_is_open, series)

#' @rdname mio_xdmf_series
#' @export
print.mio_xdmf_series <- function(x, ...) {
  if (!mio_xdmf_series_is_open(x)) {
    cat("<mio_xdmf_series: released>\n")
  } else {
    cat(sprintf("<mio_xdmf_series: %g step(s) written>\n", mio_xdmf_series_num_steps(x)))
  }
  invisible(x)
}

# ---------------------------------------------------------------------------
# Sequences: multi-file / transient datasets. See doc/sequences.md.
# ---------------------------------------------------------------------------

#' Open a multi-file / transient sequence
#'
#' An ordered plan over a set of files (or the steps inside one multi-step
#' file), read one step at a time. The handle stores paths, per-file step
#' indices and time values, never a mesh -- which is why
#' \code{mio_sequence_read()} hands back an independent mesh rather than a
#' borrow the sequence would have to cache. That is what keeps a 500-step
#' dataset traversable.
#'
#' Ordering is \strong{natural-numeric}, so \code{out_9.vtu} precedes
#' \code{out_10.vtu}; a lexicographic sort gets that backwards.
#'
#' @param pattern a glob (\code{*} and \code{?} only -- no \code{**}, no
#'   \code{[set]}; the directory part is taken literally).
#' @param format forced input format, or NULL to resolve per file.
#' @param times explicit per-entry times; the count must match.
#' @param time_from one of "auto", "file", "filename", "index".
#' @return an external pointer of class \code{mio_sequence}.
#' @export
mio_sequence <- function(pattern, format = NULL, times = NULL, time_from = "auto") {
  .Call(R_mio_sequence_open, as.character(pattern), format, times,
        as.character(time_from), FALSE)
}

#' Open a sequence from an explicit, ordered path list
#'
#' The order is yours and is kept unless \code{sort = TRUE}.
#' @param paths character vector of file paths.
#' @param format forced input format, or NULL.
#' @param times explicit per-entry times.
#' @param time_from one of "auto", "file", "filename", "index".
#' @param sort natural-numeric sort the list too.
#' @return an external pointer of class \code{mio_sequence}.
#' @export
mio_sequence_list <- function(paths, format = NULL, times = NULL,
                              time_from = "auto", sort = FALSE) {
  .Call(R_mio_sequence_open_list, as.character(paths), format, times,
        as.character(time_from), isTRUE(sort))
}

#' Number of steps in a sequence
#' @param seq a \code{mio_sequence}.
#' @return the step count.
#' @export
mio_sequence_count <- function(seq) .Call(R_mio_sequence_count, seq)

#' Entry i's file path (1-based)
#' @param seq a \code{mio_sequence}.
#' @param index 1-based entry index.
#' @return the path.
#' @export
mio_sequence_path <- function(seq, index) .Call(R_mio_sequence_path, seq, as.integer(index))

#' Entry i's step index within its own file (0 for a single-step file)
#' @param seq a \code{mio_sequence}.
#' @param index 1-based entry index.
#' @return the step index.
#' @export
mio_sequence_step <- function(seq, index) .Call(R_mio_sequence_step, seq, as.integer(index))

#' Entry i's time value
#' @param seq a \code{mio_sequence}.
#' @param index 1-based entry index.
#' @return the time.
#' @export
mio_sequence_time <- function(seq, index) .Call(R_mio_sequence_time, seq, as.integer(index))

#' Where entry i's time came from
#'
#' One of "explicit", "file", "filename" or "index" (the fallback). Reported
#' rather than left to be guessed: "the file said 0.25" and "nothing said
#' anything, so this is position 3" are different facts.
#' @param seq a \code{mio_sequence}.
#' @param index 1-based entry index.
#' @return the source name.
#' @export
mio_sequence_time_source <- function(seq, index) {
  .Call(R_mio_sequence_time_source, seq, as.integer(index))
}

#' Read one step of a sequence
#'
#' The returned mesh is independent of the sequence, which caches nothing.
#' @param seq a \code{mio_sequence}.
#' @param index 1-based entry index.
#' @return a \code{mio_mesh}.
#' @export
mio_sequence_read <- function(seq, index) .Call(R_mio_sequence_read, seq, as.integer(index))

#' Release a sequence handle
#' @param seq a \code{mio_sequence}.
#' @return NULL, invisibly.
#' @export
mio_sequence_free <- function(seq) invisible(.Call(R_mio_sequence_free, seq))

#' Fan-in: write every step of a sequence into one multi-step file
#'
#' Streams -- one mesh alive at a time. A format that cannot hold a series
#' fails naming itself and pointing at \code{\{step\}}, never a silent
#' truncation to the first step.
#' @param seq a \code{mio_sequence}.
#' @param out_path the output file.
#' @param out_format forced output format, or NULL.
#' @param ascii if TRUE, selects XDMF's "XML" data format (everything inline,
#'   no HDF5 needed) instead of the default "HDF".
#' @return NULL, invisibly.
#' @export
mio_sequence_to_timeseries <- function(seq, out_path, out_format = NULL, ascii = FALSE) {
  invisible(.Call(R_mio_sequence_to_timeseries, seq, as.character(out_path), out_format,
                  isTRUE(ascii)))
}

#' Fan-out: write each step of a multi-step file to a pattern
#'
#' @param in_path the multi-step input.
#' @param out_pattern must contain \code{\{step\}} or \code{\{index\}}.
#' @param in_format forced input format, or NULL.
#' @param out_format forced output format, or NULL.
#' @return NULL, invisibly.
#' @export
mio_timeseries_to_sequence <- function(in_path, out_pattern, in_format = NULL,
                                       out_format = NULL) {
  invisible(.Call(R_mio_timeseries_to_sequence, as.character(in_path), in_format,
                  as.character(out_pattern), out_format))
}

#' Run a sequence settings document
#'
#' The pipeline schema plus \code{Mode}, \code{Input.Pattern},
#' \code{Input.Paths}, \code{Input.Times} and \code{Input.TimeFrom}. A document
#' using none of those behaves exactly as \code{mio_pipeline_run_file()}.
#' @param settings_path path of the settings.json.
#' @return NULL, invisibly.
#' @export
mio_sequence_pipeline_run_file <- function(settings_path) {
  invisible(.Call(R_mio_sequence_pipeline_run_file, as.character(settings_path)))
}

#' Run a sequence settings document given as JSON text
#' @param json_text the document.
#' @return NULL, invisibly.
#' @export
mio_sequence_pipeline_run_json <- function(json_text) {
  invisible(.Call(R_mio_sequence_pipeline_run_json, as.character(json_text)))
}

#' @rdname mio_extract_surface
#' @export
mio_grid <- function(dims, origin = c(0, 0, 0), spacing = c(1, 1, 1),
                     max_cells = 20000000) {
  .Call(
    R_mio_grid, as.numeric(dims), as.numeric(origin), as.numeric(spacing),
    as.numeric(max_cells)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_voxelize <- function(mesh, resolution = NULL, cell_size = 0, bounds = NULL,
                         padding = 0, padding_relative = 0, fill = "all",
                         sign = "pseudonormal", attach_occupancy = FALSE,
                         max_cells = 20000000, watertight_check = "warn") {
  fills <- c(all = 0L, surface = 1L, inside = 2L)
  signs <- c(unsigned = 0L, pseudonormal = 1L, `winding-number` = 2L)
  checks <- c(off = 0L, warn = 1L, error = 2L)
  if (!fill %in% names(fills)) stop("unknown fill '", fill, "'")
  if (!sign %in% names(signs)) stop("unknown sign '", sign, "'")
  if (!watertight_check %in% names(checks)) {
    stop("unknown watertight check '", watertight_check, "'")
  }
  .Call(
    R_mio_voxelize, mesh,
    if (is.null(resolution)) numeric(0) else as.numeric(resolution),
    as.numeric(cell_size),
    if (is.null(bounds)) numeric(0) else as.numeric(bounds),
    as.numeric(padding), as.numeric(padding_relative),
    fills[[fill]], signs[[sign]], isTRUE(attach_occupancy),
    as.numeric(max_cells), checks[[watertight_check]]
  )
}

#' @rdname mio_extract_surface
#' @export
mio_crop_predicate <- function(mesh, array, compare = "<", value = 0,
                               record_ids = FALSE) {
  comparisons <- c(
    `<` = 0L, `<=` = 1L, `>` = 2L, `>=` = 3L, `==` = 4L, `!=` = 5L
  )
  if (!compare %in% names(comparisons)) stop("unknown comparison '", compare, "'")
  .Call(
    R_mio_crop_predicate, mesh, as.character(array), comparisons[[compare]],
    as.numeric(value), isTRUE(record_ids)
  )
}

#' @rdname mio_extract_surface
#' @export
mio_compute_sdf <- function(mesh, structure = "voxel", resolution = NULL,
                            cell_size = 0, bounds = NULL, padding = 0,
                            padding_relative = 0.1, root_resolution = 8,
                            max_depth = 4, band_cells = 1, record_levels = TRUE,
                            max_cells = 20000000, sign = "pseudonormal",
                            location = "corner", band = 0,
                            watertight_check = "warn") {
  structures <- c(voxel = 0L, octree = 1L)
  signs <- c(unsigned = 0L, pseudonormal = 1L, `winding-number` = 2L)
  # The SDF locations, NOT `.mio_location`'s point/cell/field data locations.
  locations <- c(corner = 0L, point = 0L, center = 1L, centre = 1L, cell = 1L)
  checks <- c(off = 0L, warn = 1L, error = 2L)
  if (!structure %in% names(structures)) stop("unknown structure '", structure, "'")
  if (!sign %in% names(signs)) stop("unknown sign '", sign, "'")
  if (!location %in% names(locations)) stop("unknown location '", location, "'")
  if (!watertight_check %in% names(checks)) {
    stop("unknown watertight check '", watertight_check, "'")
  }
  .Call(
    R_mio_compute_sdf, mesh, structures[[structure]],
    if (is.null(resolution)) numeric(0) else as.numeric(resolution),
    as.numeric(cell_size),
    if (is.null(bounds)) numeric(0) else as.numeric(bounds),
    as.numeric(padding), as.numeric(padding_relative),
    as.numeric(root_resolution), as.numeric(max_depth), as.numeric(band_cells),
    isTRUE(record_levels), as.numeric(max_cells), signs[[sign]],
    locations[[location]], as.numeric(band), checks[[watertight_check]]
  )
}

#' @rdname mio_extract_surface
#' @export
mio_surface_watertight_check <- function(mesh) {
  .Call(R_mio_surface_watertight_check, mesh)
}

#' @rdname mio_extract_surface
#' @export
mio_sample_distance <- function(mesh, points, sign = "pseudonormal", band = 0,
                                watertight_check = "warn") {
  signs <- c(unsigned = 0L, pseudonormal = 1L, `winding-number` = 2L)
  checks <- c(off = 0L, warn = 1L, error = 2L)
  if (!sign %in% names(signs)) stop("unknown sign '", sign, "'")
  if (!watertight_check %in% names(checks)) {
    stop("unknown watertight check '", watertight_check, "'")
  }
  .Call(
    R_mio_sample_distance, mesh, as.matrix(points), signs[[sign]],
    as.numeric(band), checks[[watertight_check]]
  )
}

#' @rdname mio_extract_surface
#' @export
mio_distance_to_surface <- function(mesh, surface, sign = "pseudonormal",
                                    location = "corner", band = 0,
                                    record_inside = FALSE, watertight_check = "warn") {
  signs <- c(unsigned = 0L, pseudonormal = 1L, `winding-number` = 2L)
  locations <- c(corner = 0L, point = 0L, center = 1L, centre = 1L, cell = 1L)
  checks <- c(off = 0L, warn = 1L, error = 2L)
  if (!sign %in% names(signs)) stop("unknown sign '", sign, "'")
  if (!location %in% names(locations)) stop("unknown location '", location, "'")
  if (!watertight_check %in% names(checks)) {
    stop("unknown watertight check '", watertight_check, "'")
  }
  .Call(
    R_mio_distance_to_surface, mesh, surface, signs[[sign]], locations[[location]],
    as.numeric(band), isTRUE(record_inside), checks[[watertight_check]]
  )
}

#' Run an expression with a provenance scope open
#'
#' Any `mio_write()` inside `expr` records the richer provenance block --
#' source, target, operation chain, conversion assumptions, timestamp -- into
#' the file's header slot, where the format has one. With no scope open a
#' writer emits only the one-line credit, which is the default everywhere.
#'
#' The scope is closed on exit even if `expr` throws: R has no RAII, so this
#' wrapper pairs the underlying begin/end C calls with `on.exit()`, the same
#' shape `mio_xdmf_series()` uses to give a C handle an R-shaped lifetime.
#'
#' @param expr Expression to evaluate with the scope open.
#' @param mode One of `"off"`, `"best_effort"` (default) or `"required"`.
#'   `"required"` errors for a format whose header slot cannot hold a block
#'   at all; it does **not** error merely because a slot is too small for the
#'   full block, since degrading to the credit line is that slot's honest
#'   maximum. See `doc/provenance.md`.
#' @return The value of `expr`.
#' @examples
#' \dontrun{
#' mio_with_provenance({
#'   mio_provenance_source("bracket.msh", "gmsh")
#'   mio_provenance_note("regions-dropped", "Side regions have no VTU equivalent")
#'   mio_write(mesh, "out.vtu")
#' })
#' }
#' @export
mio_with_provenance <- function(expr, mode = c("best_effort", "off", "required")) {
  mode <- match.arg(mode)
  code <- switch(mode, off = 0L, best_effort = 1L, required = 2L)
  .Call("R_mio_provenance_begin", code, PACKAGE = "meshioplusplus")
  on.exit(.Call("R_mio_provenance_end", PACKAGE = "meshioplusplus"), add = TRUE)
  expr
}

#' Record one conversion assumption against the active provenance scope
#'
#' A no-op outside a scope, so it is safe to call unconditionally.
#'
#' @param category Short stable slug, e.g. `"regions-dropped"` or `"dtype"`.
#' @param detail Human-readable specifics.
#' @return `NULL`, invisibly.
#' @export
mio_provenance_note <- function(category, detail) {
  .Call("R_mio_provenance_note", as.character(category), as.character(detail),
        PACKAGE = "meshioplusplus")
  invisible(NULL)
}

#' Record where a mesh came from, on the active provenance scope
#'
#' A no-op outside a scope.
#'
#' @param path Source file path.
#' @param format Source format name.
#' @return `NULL`, invisibly.
#' @export
mio_provenance_source <- function(path, format) {
  .Call("R_mio_provenance_set_source", as.character(path), as.character(format),
        PACKAGE = "meshioplusplus")
  invisible(NULL)
}

#' Record what was actually written, on the active provenance scope
#'
#' Records the encoding/codec/float format actually used, as opposed to what
#' was requested. A no-op outside a scope.
#'
#' @param format Target format name.
#' @param encoding,codec,float_format Optional; empty when not applicable.
#' @return `NULL`, invisibly.
#' @export
mio_provenance_target <- function(format, encoding = "", codec = "", float_format = "") {
  .Call("R_mio_provenance_set_target", as.character(format), as.character(encoding),
        as.character(codec), as.character(float_format), PACKAGE = "meshioplusplus")
  invisible(NULL)
}
