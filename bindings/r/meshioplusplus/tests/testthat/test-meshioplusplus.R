test_that("build information is available", {
  expect_true(nzchar(mio_version()))
  expect_true(mio_mesh_backend() %in% c("meshio", "native", "kratos"))
  expect_true(mio_format_readable("vtu"))
  expect_true(mio_format_writable("vtu"))
  expect_true(mio_format_writable("openfoam")) # writable since v9.20.0
  expect_false(mio_format_readable("nonexistent"))
  expect_equal(mio_cell_type_num_nodes("tetra10"), 10L)
  expect_equal(mio_cell_type_dimension("triangle"), 2L)
  expect_equal(mio_cell_type_num_nodes("polygon"), -1L) # variable-size
})

test_that("a mesh can be built and inspected", {
  m <- fixture()
  on.exit(mio_release(m))

  expect_equal(mio_num_points(m), 5)
  expect_equal(mio_point_dim(m), 3)
  expect_equal(mio_num_cell_blocks(m), 1)
  expect_equal(mio_num_cells(m), 2)
  expect_equal(mio_cell_block_type(m, 1), "tetra")
  expect_equal(mio_cell_block_types(m), "tetra")

  info <- mio_cell_block_info(m, 1)
  expect_equal(info$num_cells, 2)
  expect_equal(info$nodes_per_cell, 4)
  expect_false(info$is_ragged)

  # Names come back sorted, on every mesh backend.
  expect_equal(mio_point_data_names(m), c("displacement", "temperature"))
  expect_equal(mio_cell_data_names(m), "material")
  expect_equal(mio_cell_data_num_blocks(m, "material"), 1)
})

test_that("the column-major shape identity holds", {
  m <- fixture()
  on.exit(mio_release(m))

  # An R (dim x num_points) matrix is the SAME layout as the C API's row-major
  # (num_points, dim) -- never transposed. With this non-square 3x5 fixture a
  # transpose would show up immediately as a 5x3 result or scrambled values.
  p <- mio_points(m)
  expect_equal(dim(p), c(3L, 5L))
  expect_equal(p, POINTS)
  expect_equal(p[, 2], c(1.1, 0.2, 0.3)) # point 2's coordinates
  expect_equal(p[1, ], c(0.0, 1.1, 0.4, 0.6, 1.4)) # every point's x

  # A 3-component vector field is (components x num_points).
  d <- mio_point_data(m, "displacement")
  expect_equal(dim(d), c(3L, 5L))
  expect_equal(as.vector(d), as.vector(VEC))

  # A scalar field stays a plain vector.
  expect_equal(as.vector(mio_point_data(m, "temperature")), TEMPERATURE)
  expect_equal(as.vector(mio_cell_data(m, "material", 1)), c(7, 9))
})

test_that("mio_connectivity is 1-based and mio_connectivity_raw is 0-based", {
  m <- fixture()
  on.exit(mio_release(m))

  c1 <- mio_connectivity(m, 1)
  c0 <- mio_connectivity_raw(m, 1)
  expect_equal(dim(c1), c(4L, 2L))
  expect_equal(c1, CONN)
  # Exactly the documented relationship between the two accessors.
  expect_equal(c1, c0 + 1)
})

test_that("ragged polygon and polyhedron blocks round-trip as lists", {
  # Before meshio++ 9.15 the C ABI could neither build nor read one of these.
  m <- mio_mesh()
  on.exit(mio_release(m))
  mio_set_points(m, matrix(c(
    0, 0, 0,
    1, 0, 0,
    1, 1, 0,
    0, 1, 0,
    2, 0.5, 0
  ), nrow = 3))

  # A quad then a triangle -- the point of a jagged block.
  rows <- list(c(1, 2, 3, 4), c(2, 5, 3))
  mio_add_polygon_block(m, "polygon", rows)
  # A 4-face tetrahedron then a 3-face sliver: different face counts, so the
  # cell offsets carry real information.
  cells <- list(
    list(c(1, 2, 3), c(1, 4, 2), c(2, 4, 3), c(3, 4, 1)),
    list(c(2, 3, 5), c(3, 4, 5), c(4, 2, 5))
  )
  mio_add_polyhedron_block(m, "polyhedron", cells)

  expect_equal(mio_num_cell_blocks(m), 2)
  i1 <- mio_cell_block_info(m, 1)
  expect_true(i1$is_ragged)
  expect_false(i1$is_polyhedron)
  expect_equal(i1$num_cells, 2)
  expect_equal(i1$nodes_per_cell, 0)
  expect_equal(i1$num_faces, 2)
  expect_equal(i1$num_nodes, 7)
  i2 <- mio_cell_block_info(m, 2)
  expect_true(i2$is_polyhedron)
  expect_equal(i2$num_faces, 7)
  expect_equal(i2$num_nodes, 21)

  expect_equal(mio_polygon_block(m, 1), rows)
  expect_equal(mio_polyhedron_block(m, 2), cells)

  # Each accessor refuses the wrong shape by name rather than returning
  # something plausible-but-wrong.
  expect_error(mio_polyhedron_block(m, 1), "polygon")
  expect_error(mio_polygon_block(m, 2), "polyhedron")
  expect_error(mio_connectivity(m, 1), "ragged")
})

test_that("data arrays carry their stored dtype as an attribute", {
  m <- fixture()
  on.exit(mio_release(m))
  # R has no native int64, so values arrive as double; the dtype the core
  # actually stored is reported separately rather than silently lost.
  expect_equal(attr(mio_point_data(m, "temperature"), "dtype"), "float64")
})

test_that("meshes round-trip through files", {
  m <- fixture()
  on.exit(mio_release(m))
  dir <- tempfile()
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)

  for (ext in c("vtu", "vtk")) {
    path <- file.path(dir, paste0("mesh.", ext))
    mio_write(m, path)
    expect_true(file.exists(path))

    r <- mio_read(path)
    expect_equal(mio_num_points(r), 5)
    expect_equal(mio_point_dim(r), 3)
    expect_equal(mio_cell_block_type(r, 1), "tetra")
    expect_equal(mio_points(r), POINTS, tolerance = 1e-10)
    expect_equal(mio_connectivity(r, 1), CONN)
    expect_true(mio_meshes_equal(m, r, atol = 1e-10, rtol = 1e-10))
    mio_release(r)
  }

  src <- file.path(dir, "mesh.vtu")
  dst <- file.path(dir, "converted.vtk")
  mio_convert(src, dst)
  expect_true(file.exists(dst))

  # sniff_format reads the leading bytes, not the extension.
  expect_equal(mio_sniff_format(src), "vtu")
})

test_that("selective reads and metadata work", {
  m <- fixture()
  dir <- tempfile()
  dir.create(dir)
  on.exit({
    mio_release(m)
    unlink(dir, recursive = TRUE)
  })
  path <- file.path(dir, "mesh.vtu")
  mio_write(m, path)

  r <- mio_read(path, points_only = TRUE)
  expect_equal(mio_num_points(r), 5)
  expect_length(mio_point_data_names(r), 0)
  mio_release(r)

  # `arrays = character(0)` is NOT the same request as `arrays = NULL`:
  # the first means "no arrays", the second "every array".
  r2 <- mio_read(path, arrays = "temperature")
  expect_equal(mio_point_data_names(r2), "temperature")
  mio_release(r2)

  r3 <- mio_read(path, arrays = character(0))
  expect_length(mio_point_data_names(r3), 0)
  mio_release(r3)

  meta <- mio_read_metadata(path)
  expect_equal(meta$num_points, 5)
  expect_equal(meta$point_dim, 3)
  expect_equal(meta$num_cells, 2)
  expect_equal(meta$num_cell_blocks, 1)
  expect_equal(meta$cell_block_types, "tetra")
  expect_false(meta$cell_block_is_ragged[1])
  expect_true("temperature" %in% meta$point_data_names)
  expect_true(mio_reader_supports_options("vtu"))
})

test_that("regions round-trip in memory", {
  m <- fixture()
  on.exit(mio_release(m))

  mio_add_region(m, "inlet", "point", c(1, 3, 5))
  mio_add_region(m, "solid", "cell", c(1, 2), dim = 3L, tag = 17)
  # (cell, facet) pairs: column-major, so 2 rows x N columns.
  mio_add_region(m, "wall", "side", matrix(c(1, 0, 2, 2), nrow = 2))

  rs <- mio_regions(m)
  expect_length(rs, 3)
  by_name <- rs
  names(by_name) <- vapply(rs, function(r) r$name, character(1))

  inlet <- by_name[["inlet"]]
  expect_equal(inlet$kind, "point")
  expect_equal(as.vector(inlet$entries), c(1, 3, 5)) # 1-based, as given

  solid <- by_name[["solid"]]
  expect_equal(solid$kind, "cell")
  expect_equal(solid$dim, 3L)
  expect_equal(solid$tag, 17)
  expect_equal(as.vector(solid$entries), c(1, 2))

  wall <- by_name[["wall"]]
  expect_equal(wall$kind, "side")
  expect_equal(dim(wall$entries), c(2L, 2L))
  # Canonicalized by the core. The CELL row is 1-based here; the FACET row is
  # an ordinal within the cell type, not a mesh index, so it is NOT shifted --
  # the same rule the Fortran and Julia bindings follow.
  expect_equal(wall$entries[1, ], c(1, 2)) # cells, 1-based
  expect_equal(wall$entries[2, ], c(0, 2)) # facets, unshifted

  expect_error(mio_add_region(m, "bad", "point", c(0, 1)), "1-based")
  expect_error(mio_add_region(m, "bad", "side", c(1, 2)), "value")
})

test_that("regions survive a file round-trip", {
  dir <- tempfile()
  dir.create(dir)
  m <- mio_mesh()
  on.exit({
    mio_release(m)
    unlink(dir, recursive = TRUE)
  })
  mio_set_points(m, POINTS)
  mio_add_cell_block(m, "triangle", matrix(c(1, 2, 3, 2, 3, 4), nrow = 3))
  mio_add_region(m, "physical", "cell", c(1, 2), dim = 2L, tag = 5)

  # Abaqus is one of the two Phase-1 region formats. Its *ELSET has no notion
  # of a dimension or an integer tag, so those come back unspecified (-1); the
  # group itself is what survives.
  path <- file.path(dir, "mesh.inp")
  mio_write(m, path)
  r <- mio_read(path)
  names <- vapply(mio_regions(r), function(x) x$name, character(1))
  expect_true("physical" %in% names)
  got <- mio_regions(r)[[which(names == "physical")]]
  expect_equal(got$kind, "cell")
  expect_equal(as.vector(got$entries), c(1, 2))
  mio_release(r)
})

test_that("geometry operations work", {
  m <- fixture()
  on.exit(mio_release(m))

  surf <- mio_extract_surface(m)
  expect_gt(mio_num_points(surf), 0)
  expect_equal(mio_cell_block_type(surf, 1), "triangle")
  mio_release(surf)

  surf2 <- mio_extract_surface(m, record_parent_ids = TRUE)
  expect_true("surface:parent_cell" %in% mio_cell_data_names(surf2))
  mio_release(surf2)

  skin <- mio_extract_skin(m, linearize = TRUE)
  expect_gt(mio_num_cells(skin), 0)
  mio_release(skin)

  q <- mio_attach_quality(m)
  expect_true(any(startsWith(mio_cell_data_names(q), "quality:")))
  mio_release(q)

  expect_equal(mio_quality_counts(m)$num_cells, 2)

  st <- mio_stats(m)
  expect_equal(st$num_points, 5)
  expect_equal(st$num_cells, 2)
  expect_equal(st$bbox_min[1], 0.0)
  expect_equal(st$bbox_max[1], 1.4)
  expect_gte(mio_compute_bandwidth(m), 0)
})

test_that("transform moves points and leaves topology alone", {
  m <- fixture()
  on.exit(mio_release(m))
  M <- rbind(
    c(1, 0, 0, 2),
    c(0, 1, 0, 3),
    c(0, 0, 1, 4),
    c(0, 0, 0, 1)
  )
  t <- mio_transform(m, M)
  p <- mio_points(t)
  expect_equal(p[, 1], c(2, 3, 4)) # the origin moved by the translation
  expect_equal(p[, 2], POINTS[, 2] + c(2, 3, 4))
  expect_equal(mio_connectivity(t, 1), CONN) # topology untouched
  mio_release(t)
})

test_that("refine reports 1-based maps with a 0 sentinel", {
  m <- fixture()
  on.exit(mio_release(m))

  r <- mio_refine(m, levels = 1L)
  expect_equal(mio_num_cells(r$mesh), 16) # 2 tetra -> 8 children each
  # Refinement never prunes, so the point map is the identity -- 1-based here,
  # like every copied index array.
  expect_equal(r$point_map, as.numeric(1:5))
  expect_length(r$cell_maps, 1)
  expect_equal(r$cell_maps[[1]], c(1, 9)) # each parent's first child
  mio_release(r$mesh)

  # convert_cells CAN prune, and a pruned point is reported as 0 -- never a
  # valid 1-based index, so the sentinel stays unambiguous.
  cc <- mio_convert_cells(m, "simplexify")
  expect_equal(mio_num_cells(cc$mesh), 2) # a tetra is already a simplex
  expect_true(all(cc$point_map >= 0))
  mio_release(cc$mesh)
})

test_that("subdivide polyhedrally refines with no point map", {
  m <- fixture()
  on.exit(mio_release(m))

  s <- mio_subdivide(m, record_parent_ids = TRUE)
  expect_equal(mio_num_cell_blocks(s$mesh), 1)
  expect_equal(mio_cell_block_type(s$mesh, 1), "polyhedron")
  info <- mio_cell_block_info(s$mesh, 1)
  expect_true(info$is_polyhedron)
  expect_equal(info$num_cells, 8) # 2 tetra, 4 faces each -> 8 children
  expect_equal(mio_num_points(s$mesh), mio_num_points(m) + 2) # one apex each

  # No point_map (subdivide never prunes/renumbers a point); cell_maps is
  # per-block, first-child, 1-based.
  expect_null(s$point_map)
  expect_length(s$cell_maps, 1)
  expect_equal(s$cell_maps[[1]], c(1, 5))
  expect_true("subdivide:parent_cell" %in% mio_cell_data_names(s$mesh))
  mio_release(s$mesh)
})

test_that("agglomerate polyhedrally coarsens with a flat cell map", {
  m <- fixture()  # 2 tetra sharing one face
  on.exit(mio_release(m))

  a <- mio_agglomerate(m, target_group_size = 2)
  expect_equal(mio_num_cell_blocks(a$mesh), 1)
  expect_equal(mio_cell_block_type(a$mesh, 1), "polyhedron")
  info <- mio_cell_block_info(a$mesh, 1)
  expect_true(info$is_polyhedron)
  expect_equal(info$num_cells, 1) # both tetra merge into one cell
  expect_equal(mio_num_points(a$mesh), mio_num_points(m)) # never pruned

  # cell_map is a single FLAT array (unlike subdivide's per-block
  # cell_maps): both input cells land in the one merged output cell.
  expect_length(a$cell_map, 2)
  expect_equal(a$cell_map[1], a$cell_map[2])
  mio_release(a$mesh)
})

test_that("selective refine closes up conformingly", {
  m <- fixture()
  on.exit(mio_release(m))

  # `cells` is 1-based here, like every index array this binding takes, and is
  # shifted to the C API's 0-based numbering inside.
  sel <- mio_refine(m, cells = c(1), record_levels = TRUE)
  expect_gt(mio_num_cells(sel$mesh), mio_num_cells(m))
  expect_lt(mio_num_cells(sel$mesh), 16) # not the uniform 8-per-cell
  expect_true("refine:level" %in% mio_cell_data_names(sel$mesh))
  mio_release(sel$mesh)

  # Propagation is the always-works baseline: on a connected mesh it reaches
  # every cell, which is exactly the uniform refinement.
  prop <- mio_refine(m, cells = c(1), closure = "propagate")
  expect_equal(mio_num_cells(prop$mesh), 16)
  mio_release(prop$mesh)

  # At most one selector, and both enum names are validated.
  expect_error(mio_refine(m, cells = c(1), region = "anything"))
  expect_error(mio_refine(m, closure = "blue"))
  expect_error(mio_refine(m, where_array = "q", where_op = "~="))
})

test_that("refine's persistent parent/child hierarchy is opt-in and maintained", {
  m <- fixture()
  on.exit(mio_release(m))

  plain <- mio_refine(m, cells = c(1))
  expect_false("refine:cell_id" %in% mio_cell_data_names(plain$mesh))
  mio_release(plain$mesh)

  hier <- mio_refine(m, cells = c(1), record_hierarchy = TRUE)
  expect_true("refine:cell_id" %in% mio_cell_data_names(hier$mesh))
  expect_true("refine:parent_id" %in% mio_cell_data_names(hier$mesh))
  ids <- mio_cell_data(hier$mesh, "refine:cell_id", 1)
  parents <- mio_cell_data(hier$mesh, "refine:parent_id", 1)
  # Ids/parent-ids are stable IDENTIFIERS, not the 1-based index maps this
  # binding otherwise shifts -- they ride the raw C-side numbering unchanged,
  # like mio_partition_labels' part ids.
  expect_equal(length(unique(ids)), length(ids))
  # The coarse mesh (m) has 2 cells, implicitly numbered 0 and 1; every
  # parent_id must resolve to one of them.
  expect_true(all(parents %in% c(0, 1)))
  # Also proves the multigrid-stencil fix: this call used the default
  # redgreen closure, which leaves no hanging nodes, so refine:entity would
  # normally never be attached at all.
  expect_true("refine:entity" %in% mio_point_data_names(hier$mesh))

  # An existing hierarchy is maintained without the flag.
  again <- mio_refine(hier$mesh, cells = c(1))
  expect_true("refine:cell_id" %in% mio_cell_data_names(again$mesh))
  mio_release(again$mesh)
  mio_release(hier$mesh)
})

test_that("undo_green restores the coarse parent verbatim", {
  m <- fixture()  # 2 tetrahedra sharing a whole face
  on.exit(mio_release(m))

  fine <- mio_refine(m, cells = c(1), record_hierarchy = TRUE, record_levels = TRUE)
  fine_n <- mio_cell_block_info(fine$mesh, 1)$num_cells
  coarse_n <- mio_cell_block_info(m, 1)$num_cells

  undone <- mio_undo_green(m, fine$mesh)
  expect_gt(undone$num_groups_undone, 0)
  expect_gt(undone$num_cells_removed, 0)
  undone_n <- mio_cell_block_info(undone$mesh, 1)$num_cells
  expect_lt(undone_n, fine_n)
  expect_gt(undone_n, coarse_n)
  expect_false("refine:cell_id" %in% mio_cell_data_names(undone$mesh))
  expect_false("refine:entity" %in% mio_point_data_names(undone$mesh))

  # Fails by name rather than guessing: no hierarchy at all on "fine" here.
  expect_error(mio_undo_green(m, m))

  mio_release(undone$mesh)
  mio_release(fine$mesh)
})

test_that("split and partition pieces own their handles", {
  m <- fixture()
  on.exit(mio_release(m))

  pieces <- mio_split(m, by = "type")
  expect_length(pieces, 1)
  expect_equal(names(pieces), "tetra")
  # The piece must still work after the C result handle is gone -- which it
  # does, because every piece takes ownership rather than borrowing.
  expect_equal(mio_num_cells(pieces[[1]]), 2)
  expect_equal(mio_points(pieces[[1]]), POINTS)
  mio_release(pieces[[1]])

  parts <- mio_partition(m, 2L, method = "sfc")
  expect_length(parts, 2)
  expect_equal(vapply(parts, function(p) p$part_id, integer(1)), c(0L, 1L))
  total <- sum(vapply(parts, function(p) mio_num_cells(p$mesh), numeric(1)))
  expect_equal(total, 2) # partition of unity
  for (p in parts) {
    expect_length(p$cell_maps, 1) # block structure kept 1:1
    mio_release(p$mesh)
  }

  labels <- mio_partition_labels(m, 2L, method = "sfc")
  expect_length(labels, 2)
  # Part IDs, not indices: deliberately NOT shifted to 1-based.
  expect_equal(sort(labels), c(0, 1))
})

test_that("reorder returns 1-based permutations", {
  m <- fixture()
  on.exit(mio_release(m))
  r <- mio_reorder(m, "rcm")
  expect_equal(mio_num_points(r$mesh), 5)
  expect_equal(sort(r$node_perm), as.numeric(1:5))
  expect_length(r$cell_perms, 1)
  expect_equal(sort(r$cell_perms[[1]]), as.numeric(1:2))
  mio_release(r$mesh)
})

test_that("clean, smooth and crop work", {
  m <- fixture()
  on.exit(mio_release(m))

  c1 <- mio_clean(m)
  expect_equal(mio_num_points(c1$mesh), 5) # nothing to remove here
  expect_equal(c1$points_welded, 0)
  mio_release(c1$mesh)

  s <- mio_smooth(m, iterations = 2L)
  expect_equal(mio_num_points(s$mesh), 5) # a pure coordinate move
  expect_equal(mio_num_cells(s$mesh), 2)
  expect_gte(s$max_displacement, 0)
  mio_release(s$mesh)

  # method="odt": tet-only, C++-core only. `m`'s own block is "tetra".
  so <- mio_smooth(m, method = "odt", iterations = 1L, fix_boundary = FALSE,
                   preserve_features = FALSE)
  expect_equal(mio_num_points(so$mesh), 5)
  expect_equal(mio_num_cells(so$mesh), 2)
  expect_gte(so$num_nodes_moved, 0)
  mio_release(so$mesh)

  # optimize_volume: ODT remeshing (relocate + flip connectivity), tet-only.
  ov <- mio_optimize_volume(m, max_iterations = 5L)
  expect_equal(mio_num_points(ov$mesh), 5) # the point set is invariant
  expect_gte(ov$num_flips, 0)
  expect_gte(ov$min_quality_after, ov$min_quality_before - 1e-12)
  mio_release(ov$mesh)

  b <- mio_crop_bbox(m, c(-1, -1, -1), c(10, 10, 10))
  expect_equal(mio_num_cells(b), 2) # the box holds everything
  mio_release(b)

  b2 <- mio_crop_bbox(m, c(100, 100, 100), c(101, 101, 101))
  expect_equal(mio_num_cells(b2), 0)
  mio_release(b2)

  p <- mio_crop_plane(m, c(0, 0, 0), c(1, 0, 0), mode = "any")
  expect_gte(mio_num_cells(p), 0)
  mio_release(p)
})

test_that("slice and isosurface work", {
  m <- fixture()
  on.exit(mio_release(m))

  sl <- mio_slice(m, c(0.5, 0.5, 0.5), c(0, 0, 1))
  expect_gte(mio_num_points(sl), 0) # may be empty; never an error
  mio_release(sl)

  iso <- mio_isosurface(m, "temperature", 3)
  expect_gte(mio_num_points(iso), 0)
  if (mio_num_cells(iso) > 0) {
    expect_true("iso:value" %in% mio_cell_data_names(iso))
    expect_true("iso:index" %in% mio_cell_data_names(iso))
  }
  mio_release(iso)

  # A cell_data array has no level set: that must fail by name.
  expect_error(mio_isosurface(m, "material", 8))
})

test_that("gradient differentiates a point-data field", {
  m <- fixture()
  on.exit(mio_release(m))

  g <- mio_gradient(m, "temperature")
  expect_equal(g$num_skipped, 0)
  expect_equal(g$num_fallback, 0)
  expect_true("temperature:gradient" %in% mio_cell_data_names(g$mesh))
  mio_release(g$mesh)

  l <- mio_gradient(m, "temperature", method = "least-squares")
  expect_gte(l$num_fallback, 0)
  mio_release(l$mesh)

  p <- mio_gradient(m, "temperature", location = "point", output = "dT")
  expect_true("dT" %in% mio_point_data_names(p$mesh))
  mio_release(p$mesh)

  # A cell_data array is piecewise constant and has no derivative; a scalar has
  # no divergence. Both must fail by name.
  expect_error(mio_gradient(m, "material"))
  expect_error(mio_gradient(m, "temperature", op = "divergence"))
})

test_that("hessian computes the second derivative of a scalar field", {
  m <- fixture()
  on.exit(mio_release(m))

  h <- mio_hessian(m, "temperature")
  expect_equal(h$num_skipped, 0)
  expect_true("temperature:hessian" %in% mio_cell_data_names(h$mesh))
  mio_release(h$mesh)

  l <- mio_hessian(m, "temperature", method = "least-squares")
  expect_gte(l$num_fallback, 0)
  mio_release(l$mesh)

  p <- mio_hessian(m, "temperature", location = "point", output = "H2")
  expect_true("H2" %in% mio_point_data_names(p$mesh))
  mio_release(p$mesh)

  # A cell_data array has no derivative, and a vector field is scalar-only.
  # Both must fail by name.
  expect_error(mio_hessian(m, "material"))
  expect_error(mio_hessian(m, "displacement"))
})

test_that("estimate_error estimates a recovery-based indicator and marks", {
  m <- fixture()
  on.exit(mio_release(m))

  e <- mio_estimate_error(m, "temperature")
  expect_equal(e$num_skipped, 0)
  expect_gte(e$global_error, 0)
  expect_equal(e$num_marked, 0)
  expect_true("error:zz" %in% mio_cell_data_names(e$mesh))
  expect_false("error:marked" %in% mio_cell_data_names(e$mesh))
  mio_release(e$mesh)

  f <- mio_estimate_error(m, "temperature", marking = "absolute", marking_value = 1e-9,
                          output = "ind", marked = "flag")
  expect_true("ind" %in% mio_cell_data_names(f$mesh))
  expect_true("flag" %in% mio_cell_data_names(f$mesh))
  expect_gte(f$num_marked, 0)
  mio_release(f$mesh)

  # A cell_data array has no derivative to recover, and an out-of-range
  # marking_value for "fraction" is rejected. Both must fail by name.
  expect_error(mio_estimate_error(m, "material"))
  expect_error(mio_estimate_error(m, "temperature", marking = "fraction", marking_value = 1.5))
})

test_that("merge and interpolate work", {
  a <- fixture()
  b <- fixture()
  on.exit({
    mio_release(a)
    mio_release(b)
  })

  merged <- mio_merge(list(a, b))
  expect_equal(mio_num_points(merged), 10)
  expect_equal(mio_num_cells(merged), 4)
  expect_true("source_mesh_id" %in% mio_cell_data_names(merged))
  mio_release(merged)

  merged2 <- mio_merge(list(a, b), weld = TRUE, atol = 1e-9, source_tag = FALSE)
  expect_equal(mio_num_points(merged2), 5) # the two copies weld together
  mio_release(merged2)

  # The target already carries "temperature", so the default on_conflict
  # ("error") must refuse rather than silently overwrite.
  target <- fixture()
  expect_error(mio_interpolate(a, target, arrays = "temperature"))
  it <- mio_interpolate(a, target, arrays = "temperature", on_conflict = "suffix")
  expect_true("temperature_interp" %in% mio_point_data_names(it))
  mio_release(it)
  mio_release(target)
})

test_that("diff reports structured results", {
  a <- fixture()
  b <- fixture()
  on.exit({
    mio_release(a)
    mio_release(b)
  })

  rep <- mio_diff(a, b)
  expect_equal(rep$verdict, "identical")
  expect_equal(rep$points$num_exceeding, 0)

  moved <- POINTS
  moved[1, 1] <- 5.0
  mio_set_points(b, moved)
  rep2 <- mio_diff(a, b)
  expect_equal(rep2$verdict, "different")
  expect_equal(rep2$points$max_abs_error, 5.0)
  expect_false(mio_meshes_equal(a, b))
  expect_true(mio_meshes_equal(a, b, atol = 10)) # inside a loose tolerance
})

test_that("data operations never touch geometry", {
  m <- fixture()
  on.exit(mio_release(m))
  before_points <- mio_points(m)
  before_conn <- mio_connectivity(m, 1)

  d <- mio_data_drop(m, "point", "temperature")
  expect_equal(mio_point_data_names(d), "displacement")
  expect_equal(mio_points(d), before_points) # geometry bit-identical
  expect_equal(mio_connectivity(d, 1), before_conn)
  mio_release(d)

  k <- mio_data_keep(m, "point", "temperature")
  expect_equal(mio_point_data_names(k), "temperature")
  mio_release(k)

  r <- mio_data_rename(m, "point", "temperature", "T")
  expect_true("T" %in% mio_point_data_names(r))
  expect_false("temperature" %in% mio_point_data_names(r))
  mio_release(r)

  p2c <- mio_data_point_to_cell(m, "temperature")
  expect_true("temperature" %in% mio_cell_data_names(p2c))
  mio_release(p2c)

  c2p <- mio_data_cell_to_point(m, "material")
  expect_true("material" %in% mio_point_data_names(c2p))
  mio_release(c2p)

  calc <- mio_data_calc(m, "temperature * 2.0", "doubled")
  expect_equal(as.vector(mio_point_data(calc, "doubled")), c(2, 4, 6, 8, 10))
  mio_release(calc)

  cond <- mio_data_condition(m, "point", "temperature", cond_mode = "normalize")
  t <- as.vector(mio_point_data(cond, "temperature"))
  expect_equal(min(t), 0)
  expect_equal(max(t), 1)
  mio_release(cond)

  info <- mio_data_info(m)
  expect_length(info, 3)
  names <- vapply(info, function(x) x$name, character(1))
  temp <- info[[which(names == "temperature")]]
  expect_equal(temp$location, "point")
  expect_equal(temp$num_entries, 5)
  expect_equal(temp$num_components, 1)
  expect_equal(temp$min, 1)
  expect_equal(temp$max, 5)
  expect_equal(temp$mean, 3)
  expect_equal(temp$num_nan, 0)
})

test_that("field integration (mio_data_integrate) totals/means over cells", {
  m <- fixture()
  on.exit(mio_release(m))
  mio_add_region(m, "solid", "cell", 1L, dim = 3L, tag = 17)

  report <- mio_data_integrate(m, "material")
  expect_length(report, 1)
  arr <- report[[1]]
  expect_equal(arr$name, "material")
  expect_equal(arr$num_components, 1)
  expect_equal(arr$domain$num_cells, 2)
  expect_equal(arr$domain$num_skipped, 0)
  dcomp <- arr$domain$components
  expect_gt(unname(dcomp[1, "domain_measure"]), 0)
  expect_equal(
    unname(dcomp[1, "mean"]), unname(dcomp[1, "total"] / dcomp[1, "domain_measure"])
  )
  expect_equal(unname(dcomp[1, "num_nan"]), 0)

  expect_length(arr$regions, 1)
  region <- arr$regions[[1]]
  expect_equal(region$name, "solid")
  expect_equal(region$num_cells, 1)
  expect_lt(unname(region$components[1, "total"]), unname(dcomp[1, "total"])) # one cell < both

  # NULL names means every cell_data array (there's exactly one: material).
  expect_length(mio_data_integrate(m), 1)

  # A point_data-only name fails, naming the fix.
  expect_error(mio_data_integrate(m, "temperature"), "point_data_to_cell_data")
})

test_that("the settings pipeline runs (or fails naming the flag)", {
  # Behaviour follows the build: with the JSON parser a bad document fails
  # naming the offending op; without it every entry point fails naming
  # -DMESHIOPLUSPLUS_WITH_JSON=ON. Never a missing symbol either way.
  bad <- paste0(
    '{"Input": {"Path": "a"}, "Output": {"Path": "b"},',
    ' "Operations": [{"Op": "Nope"}]}'
  )
  if (mio_pipeline_has_json()) {
    expect_error(mio_pipeline_run_json(bad), "Nope")
    dir <- tempfile("mio_pipeline")
    dir.create(dir)
    inp <- file.path(dir, "in.vtu")
    out <- file.path(dir, "out.vtu")
    m <- fixture()
    mio_write(m, inp)
    mio_release(m)
    settings <- file.path(dir, "settings.json")
    writeLines(sprintf(
      '{"Input": {"Path": "%s"}, "Operations": [{"Op": "Quality"}], "Output": {"Path": "%s"}}',
      inp, out
    ), settings)
    mio_pipeline_run_file(settings)
    back <- mio_read(out)
    expect_true("quality:scaled_jacobian" %in% mio_cell_data_names(back))
    mio_release(back)
    unlink(dir, recursive = TRUE)
  } else {
    expect_error(mio_pipeline_run_json(bad), "MESHIOPLUSPLUS_WITH_JSON")
  }
})

test_that("errors carry the C API's own message", {
  # Every failure becomes an R condition carrying mio_last_error(); a status
  # code never reaches R.
  err <- tryCatch(mio_read("/definitely/not/a/mesh/file.vtu"), error = function(e) e)
  expect_s3_class(err, "error")
  expect_true(nzchar(conditionMessage(err)))
  expect_equal(conditionMessage(err), mio_last_error())

  m <- fixture()
  expect_error(mio_point_data(m, "no_such_array"))
  expect_error(mio_cell_block_type(m, 99))
  expect_error(mio_data_rename(m, "point", "missing", "x"))

  # Using a released mesh is an error, not a crash.
  mio_release(m)
  expect_error(mio_num_points(m), "released")
  mio_release(m) # idempotent
  expect_false(mio_is_open(m))

  # A foreign external pointer is rejected rather than dereferenced.
  expect_error(mio_num_points(list()), "mio_mesh")
})

test_that("a transient XDMF series round-trips", {
  # The one writer mio_write() cannot express: the grid goes out once and each
  # step is appended. "XML" keeps everything in the single .xdmf, so this runs
  # against a library built without HDF5 too.
  dir <- tempfile("mio_series")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  path <- file.path(dir, "series.xdmf")

  m <- fixture()
  s <- mio_xdmf_series(path, data_format = "XML")
  expect_s3_class(s, "mio_xdmf_series")
  expect_true(mio_xdmf_series_is_open(s))
  expect_equal(mio_xdmf_series_num_steps(s), 0)

  mio_xdmf_series_write_points_cells(s, m)

  # temperature = t + node id, so no two steps share a value and a step
  # mix-up cannot pass.
  times <- c(0, 0.5, 1)
  step <- mio_mesh()
  mio_set_points(step, POINTS)
  mio_add_cell_block(step, "tetra", CONN)
  for (t in times) {
    mio_add_point_data(step, "temperature", t + seq_len(5))
    mio_xdmf_series_write_data(s, t, step)
  }
  expect_equal(mio_xdmf_series_num_steps(s), 3)
  expect_output(print(s), "3 step")

  # The .xdmf is buffered until finalize; nothing is readable before it.
  mio_xdmf_series_finalize(s)
  mio_xdmf_series_finalize(s) # idempotent
  mio_xdmf_series_release(s)
  expect_false(mio_xdmf_series_is_open(s))
  mio_xdmf_series_release(s) # idempotent
  expect_error(mio_xdmf_series_num_steps(s), "released")
  expect_output(print(s), "released")

  meta <- mio_read_metadata(path)
  expect_equal(meta$time_values, times)

  for (k in seq_along(times)) {
    back <- mio_read(path, time_step = k - 1L)
    expect_equal(mio_num_points(back), 5)
    expect_equal(as.vector(mio_point_data(back, "temperature")), times[k] + seq_len(5))
  }

  # An unknown data format is an error carrying the C API's own message.
  expect_error(mio_xdmf_series(file.path(dir, "bad.xdmf"), data_format = "NoSuchFormat"))

  # A mesh and a series are separately tagged: neither is accepted for the
  # other, and a foreign pointer is rejected rather than dereferenced.
  expect_error(mio_xdmf_series_num_steps(m), "mio_xdmf_series")
  other <- mio_xdmf_series(file.path(dir, "b.xdmf"), data_format = "XML")
  expect_error(mio_num_points(other), "mio_mesh")
  expect_error(mio_xdmf_series_num_steps(list()), "mio_xdmf_series")
  # Released before the tempdir goes away: a finalizer running afterwards
  # would try to write the .xdmf into a directory that no longer exists.
  mio_xdmf_series_release(other)
})

test_that("grid builds a lattice from nothing", {
  g <- mio_grid(c(2, 2, 2))
  on.exit(mio_release(g))
  expect_equal(mio_num_points(g), 27)
  expect_equal(mio_num_cell_blocks(g), 1)

  # An empty lattice is a legal request, not an error.
  e <- mio_grid(c(0, 0, 0))
  expect_equal(mio_num_points(e), 0)
  mio_release(e)

  expect_error(mio_grid(c(100, 100, 100), max_cells = 1000))
})

test_that("voxelize builds a grid around a surface", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  q <- mio_surface_watertight_check(cube)
  expect_true(q$watertight)
  expect_equal(q$boundary_edges, 0)

  v <- mio_voxelize(cube, resolution = c(4, 4, 4))
  expect_equal(v$num_occupied, 64)
  expect_equal(v$dims, c(4, 4, 4))
  expect_equal(v$spacing[[1]], 0.25)
  mio_release(v$mesh)

  inside <- mio_voxelize(cube,
    resolution = c(5, 5, 5), bounds = c(-0.5, -0.5, -0.5, 1.5, 1.5, 1.5),
    fill = "inside", watertight_check = "off"
  )
  expect_equal(inside$num_occupied, 27)
  mio_release(inside$mesh)

  # An unknown fill fails by name rather than silently defaulting.
  expect_error(mio_voxelize(cube, resolution = c(2, 2, 2), fill = "solid"))
})

test_that("signed distance matches the cube's closed form", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  # The centre is 0.5 inside; the other two are 1.0 outside.
  pts <- matrix(c(0.5, 0.5, 0.5, 2, 0.5, 0.5, -1, 0.5, 0.5), nrow = 3)
  d <- mio_sample_distance(cube, pts, watertight_check = "off")
  expect_equal(length(d), 3)
  expect_equal(d[[1]], -0.5, tolerance = 1e-12)
  expect_equal(d[[2]], 1.0, tolerance = 1e-12)
  expect_equal(d[[3]], 1.0, tolerance = 1e-12)

  g <- mio_grid(c(2, 2, 2), origin = c(-0.5, -0.5, -0.5))
  f <- mio_distance_to_surface(g, cube, watertight_check = "off", record_inside = TRUE)
  expect_true(f$quality$watertight)
  expect_equal(f$num_banded, 0)
  expect_true("sdf:distance" %in% mio_point_data_names(f$mesh))
  expect_true("sdf:inside" %in% mio_point_data_names(f$mesh))
  mio_release(f$mesh)
  mio_release(g)

  expect_error(mio_sample_distance(cube, pts, sign = "magic"))
})

test_that("compute_sdf generates the grid and the field in one call", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  sdf <- mio_compute_sdf(cube, resolution = c(4, 4, 4), watertight_check = "off")
  expect_equal(sdf$dims, c(4, 4, 4))
  expect_equal(sdf$max_depth, 0)
  expect_true("sdf:distance" %in% mio_point_data_names(sdf$mesh))
  expect_equal(mio_cell_block_info(sdf$mesh, 1)$num_cells, 64)
  mio_release(sdf$mesh)

  # The octree refines only near the surface: more than the root, far less than
  # the uniform grid of the same finest resolution.
  tree <- mio_compute_sdf(cube,
    structure = "octree", root_resolution = 4, max_depth = 2,
    watertight_check = "off"
  )
  expect_equal(tree$max_depth, 2)
  n <- mio_cell_block_info(tree$mesh, 1)$num_cells
  expect_true(n > 64 && n < 4096)
  mio_release(tree$mesh)

  # resolution/cell_size size a voxel grid; an octree's finest cell is already
  # determined by root_resolution and max_depth.
  expect_error(mio_compute_sdf(cube, structure = "octree", resolution = c(4, 4, 4)))
  expect_error(mio_compute_sdf(cube, structure = "quadtree"))
})

test_that("crop_predicate keeps the cells a data comparison selects", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  dom <- mio_grid(c(4, 4, 4), origin = c(-0.5, -0.5, -0.5), spacing = c(0.5, 0.5, 0.5))
  on.exit(mio_release(dom), add = TRUE)
  f <- mio_distance_to_surface(dom, cube, location = "center", watertight_check = "off")
  on.exit(mio_release(f$mesh), add = TRUE)

  kept <- mio_crop_predicate(f$mesh, "sdf:distance", compare = "<", value = 0)
  expect_equal(mio_cell_block_info(kept, 1)$num_cells, 8)
  mio_release(kept)

  # There is deliberately no `mode`: a cell_data predicate is already one value
  # per cell and has nothing for an all/any rule to reduce.
  expect_error(mio_crop_predicate(f$mesh, "sdf:distance", compare = "~"))
  expect_error(mio_crop_predicate(f$mesh, "nope"))
})

test_that("remesh replaces a surface's triangulation by ACVD clustering", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  r <- mio_remesh(cube, 10)
  on.exit(mio_release(r$mesh), add = TRUE)
  expect_equal(mio_cell_block_type(r$mesh, 1), "triangle")
  expect_equal(mio_num_points(r$mesh), r$num_clusters)
  expect_gte(r$num_isolated_clusters, 0)
  expect_gte(r$num_non_manifold_vertices, 0)

  # gradation/preserve_boundary are accepted and reach the C API; a closed
  # mesh's boundary handling is a no-op either way.
  q <- mio_remesh(cube, 10, metric = "quadric", gradation = 1.5, preserve_boundary = FALSE)
  on.exit(mio_release(q$mesh), add = TRUE)
  expect_gt(mio_num_cells(q$mesh), 0)

  # The anisotropic metric + max_anisotropy go through mio_remesh_ex.
  a <- mio_remesh(cube, 10, metric = "anisotropic", max_anisotropy = 3.0)
  on.exit(mio_release(a$mesh), add = TRUE)
  expect_gt(mio_num_cells(a$mesh), 0)
  expect_gte(a$num_non_manifold_vertices, 0)

  # max_anisotropy away from the default under a non-anisotropic metric is
  # guarded, not silently ignored.
  expect_error(mio_remesh(cube, 10, metric = "isotropic", max_anisotropy = 2.0))

  expect_error(mio_remesh(cube, 3))
})

test_that("remesh_volume retetrahedralizes by isosurface stuffing", {
  cube <- cube_surface()
  on.exit(mio_release(cube))

  # Unlike remesh, remesh_volume accepts the closed surface directly and
  # returns a tetra mesh.
  r <- mio_remesh_volume(cube, cell_size = 0.4, watertight_check = "off")
  on.exit(mio_release(r$mesh), add = TRUE)
  expect_equal(mio_cell_block_type(r$mesh, 1), "tetra")
  expect_gt(r$num_tets, 0)
  expect_equal(r$num_tets, mio_num_cells(r$mesh))
  expect_gte(r$num_vertices_warped, 0)
  expect_gte(r$num_tets_rejected, 0)
  expect_gte(r$num_non_manifold_edges, 0)
  expect_true(r$input_quality$watertight)

  # Neither resolution nor cell_size given is rejected by name.
  expect_error(mio_remesh_volume(cube))
})

test_that("repair rewinds, fills and splits a surface", {
  # The cube surface with two facets flipped: exactly those two are rewound
  # and the output is watertight with no point added.
  conn <- matrix(c(
    1, 3, 2, 1, 4, 3, 5, 6, 7, 5, 7, 8,
    1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6,
    3, 4, 8, 3, 8, 7, 4, 1, 5, 4, 5, 8
  ), nrow = 3)
  broken <- conn
  broken[2:3, 1] <- broken[3:2, 1]
  broken[2:3, 9] <- broken[3:2, 9]
  m <- mio_mesh()
  on.exit(mio_release(m))
  mio_set_points(m, matrix(c(
    0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1
  ), nrow = 3))
  mio_add_cell_block(m, "triangle", broken)
  r <- mio_repair(m)
  on.exit(mio_release(r$mesh), add = TRUE)
  expect_gt(r$quality_before$inconsistent_pairs, 0)
  expect_true(r$quality_after$watertight)
  expect_equal(r$num_flipped, 2)
  expect_equal(r$num_components, 1)
  expect_equal(r$num_holes_detected, 0)
  expect_equal(mio_num_points(r$mesh), 8)

  # Minus one facet: the hole is filled by a centroid fan in a trailing block.
  h <- mio_mesh()
  on.exit(mio_release(h), add = TRUE)
  mio_set_points(h, mio_points(m))
  mio_add_cell_block(h, "triangle", conn[, -1])
  f <- mio_repair(h)
  on.exit(mio_release(f$mesh), add = TRUE)
  expect_equal(f$num_holes_filled, 1)
  expect_equal(f$num_faces_added, 3)
  expect_equal(f$num_points_added, 1)
  expect_true(f$quality_after$watertight)
  expect_equal(mio_num_cell_blocks(f$mesh), 2)
  g <- mio_repair(h, fill_holes = FALSE)
  on.exit(mio_release(g$mesh), add = TRUE)
  expect_equal(g$num_holes_detected, 0)
  expect_equal(g$quality_after$boundary_edges, 3)
})

test_that("shrinkwrap projects points onto a target surface", {
  target <- mio_mesh()
  on.exit(mio_release(target))
  mio_set_points(target, matrix(c(0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0), nrow = 3))
  mio_add_cell_block(target, "triangle", matrix(c(1, 2, 3, 1, 3, 4), nrow = 3))
  cloud <- mio_mesh()
  on.exit(mio_release(cloud), add = TRUE)
  mio_set_points(cloud, matrix(c(1, 1, 0.5, 2, 2, -0.7, 3, 1, 1.5), nrow = 3))
  mio_add_cell_block(cloud, "vertex", matrix(c(1, 2, 3), nrow = 1))
  mio_add_point_data(cloud, "w", c(1, 0.5, 1))
  w <- mio_shrinkwrap(cloud, target, offset = 0.25, weights = "w", record_distance = TRUE)
  on.exit(mio_release(w$mesh), add = TRUE)
  expect_equal(w$num_projected, 3)
  expect_equal(w$num_skipped, 0)
  expect_equal(w$quality$boundary_edges, 4)
  p <- mio_points(w$mesh)
  expect_equal(p[3, 1], 0.25, tolerance = 1e-14)
  expect_equal(p[3, 2], -0.7 + 0.5 * (0.25 + 0.7), tolerance = 1e-14)
  expect_true("shrinkwrap:distance" %in% mio_point_data_names(w$mesh))
  expect_error(mio_shrinkwrap(cloud, target, normal_weight = "nope"))
})

test_that("sobolev_deform filters a displacement field", {
  n <- 4
  pts <- matrix(0, nrow = 3, ncol = n * n)
  disp <- matrix(0, nrow = 3, ncol = n * n)
  for (j in 0:(n - 1)) for (i in 0:(n - 1)) {
    pts[, j * n + i + 1] <- c(i, j, 0)
    disp[3, j * n + i + 1] <- if ((i + j) %% 2 == 1) 0.1 else -0.1
  }
  conn <- c()
  for (j in 0:(n - 2)) for (i in 0:(n - 2)) {
    a <- j * n + i + 1
    conn <- c(conn, a, a + 1, a + n + 1, a, a + n + 1, a + n)
  }
  m <- mio_mesh()
  on.exit(mio_release(m))
  mio_set_points(m, pts)
  mio_add_cell_block(m, "triangle", matrix(conn, nrow = 3))
  mio_add_point_data(m, "d", disp)
  pin <- numeric(n * n)
  pin[1] <- 1
  mio_add_point_data(m, "pin", pin)
  s <- mio_sobolev_deform(m, "d", 2.0, fixed_points_array = "pin", record_filtered = TRUE)
  on.exit(mio_release(s$mesh), add = TRUE)
  expect_true(s$converged)
  expect_gt(s$num_iterations, 0)
  expect_equal(s$num_fixed, 1)
  expect_equal(s$num_isolated, 0)
  expect_true(s$max_displacement > 0 && s$max_displacement < 0.1)
  expect_equal(mio_points(s$mesh)[3, 1], 0)
  expect_true("sobolev:displacement" %in% mio_point_data_names(s$mesh))
  # A constant field with nothing pinned is preserved exactly, zero iterations.
  mio_add_point_data(m, "c", matrix(0.4, nrow = 3, ncol = n * n))
  k <- mio_sobolev_deform(m, "c", 2.0)
  on.exit(mio_release(k$mesh), add = TRUE)
  expect_equal(k$num_iterations, 0)
  expect_equal(mio_points(k$mesh)[1, ], pts[1, ] + 0.4)
  expect_error(mio_sobolev_deform(m, "missing", 1.0))
})
