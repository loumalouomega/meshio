# Test suite for the meshio++ Julia binding.
#
# The fixture mirrors tests/fortran/test_fortran_api.f90 deliberately: 5 points
# x 3 dims, 2 tetra x 4 nodes, 3-component vector data, and coordinates that
# are asymmetric in every axis. Nothing about it is square, so a transposed
# array mapping or a missed 1-based shift cannot cancel out and pass anyway.

using Test
using MeshioPlusPlus
import MeshioPlusPlus as mio

# The same numbers the Fortran test uses, in Julia's (dim, num_points) shape.
const POINTS = Float64[0.0 1.1 0.4 0.6 1.4
                       0.0 0.2 1.2 0.7 1.5
                       0.0 0.3 0.5 1.3 1.6]
# (nodes_per_cell, num_cells), 1-based.
const CONN = Int64[1 2
                   2 3
                   3 4
                   4 5]
const VEC = Float64[10.0 20.0 30.0 40.0 50.0
                    11.0 21.0 31.0 41.0 51.0
                    12.0 22.0 32.0 42.0 52.0]

"""Build the reference mesh from raw arrays."""
function fixture()
    m = Mesh()
    set_points!(m, POINTS)
    add_cell_block!(m, "tetra", CONN)
    add_point_data!(m, "displacement", VEC)
    add_point_data!(m, "temperature", Float64[1.0, 2.0, 3.0, 4.0, 5.0])
    append_cell_data!(m, "material", Int64[7, 9])
    m
end

@testset "MeshioPlusPlus" begin

@testset "introspection" begin
    @test !isempty(version())
    @test mesh_backend() in ("meshio", "native", "kratos")
    @test !isempty(library_path())
    @test format_readable("vtu")
    @test format_writable("vtu")
    @test format_writable("openfoam")       # writable since v9.20.0
    @test !format_readable("nonexistent")
    @test cell_type_num_nodes("tetra10") == 10
    @test cell_type_dimension("triangle") == 2
    @test cell_type_num_nodes("polygon") == -1   # variable-size
end

@testset "build and inspect" begin
    m = fixture()
    @test num_points(m) == 5
    @test point_dim(m) == 3
    @test num_cell_blocks(m) == 1
    @test num_cells(m) == 2
    @test cell_block_type(m, 1) == "tetra"
    @test cell_block_types(m) == ["tetra"]

    info = cell_block_info(m, 1)
    @test info.num_cells == 2
    @test info.nodes_per_cell == 4
    @test !info.is_ragged

    # Data names come back sorted, on every backend.
    @test point_data_names(m) == ["displacement", "temperature"]
    @test cell_data_names(m) == ["material"]
    @test num_point_data(m) == 2
    @test cell_data_num_blocks(m, "material") == 1
    close(m)
end

@testset "column-major shape identity" begin
    m = fixture()
    p = points(m)
    # (dim, num_points) in Julia == the C API's row-major (num_points, dim):
    # the SAME memory, never transposed. If a transpose had crept in, this
    # non-square 3x5 fixture would come back 5x3 or scrambled.
    @test size(p) == (3, 5)
    @test p == POINTS
    @test p[:, 2] == [1.1, 0.2, 0.3]      # point 2's coordinates
    @test p[1, :] == [0.0, 1.1, 0.4, 0.6, 1.4]  # every point's x

    # A 3-component vector field is (components, num_points) here.
    d = point_data(m, "displacement")
    @test size(d) == (3, 5)
    @test d == VEC
    # A scalar field stays a plain vector.
    @test point_data(m, "temperature") == [1.0, 2.0, 3.0, 4.0, 5.0]
    @test cell_data(m, "material", 1) == Int64[7, 9]
    close(m)
end

@testset "1-based copying vs 0-based borrowing" begin
    m = fixture()
    c1 = connectivity(m, 1)             # copy, 1-based
    c0 = parent(connectivity_ptr(m, 1)) # borrow, 0-based (the ABI's own)
    @test size(c1) == (4, 2)
    @test c1 == CONN
    @test c1 == c0 .+ 1                 # exactly the documented relationship
    @test eltype(c0) == Int64
    close(m)
end

@testset "ragged (polygon / polyhedron) connectivity" begin
    # Before meshio++ 9.15 the C ABI could neither build nor read one of these,
    # so this whole testset is new ground rather than a regression guard.
    m = Mesh()
    set_points!(m, Float64[0.0 1.0 1.0 0.0 2.0
                           0.0 0.0 1.0 1.0 0.5
                           0.0 0.0 0.0 0.0 0.0])
    # A quad then a triangle -- the point of a jagged block.
    rows = [[1, 2, 3, 4], [2, 5, 3]]
    add_polygon_block!(m, "polygon", rows)
    # A 4-face tetrahedron then a 3-face sliver: different face counts, so the
    # cell offsets carry real information.
    cells = [[[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]],
             [[2, 3, 5], [3, 4, 5], [4, 2, 5]]]
    add_polyhedron_block!(m, "polyhedron", cells)

    @test num_cell_blocks(m) == 2
    i1 = cell_block_info(m, 1)
    @test i1.is_ragged && !i1.is_polyhedron
    @test i1.num_cells == 2 && i1.nodes_per_cell == 0
    @test i1.num_faces == 2 && i1.num_nodes == 7
    i2 = cell_block_info(m, 2)
    @test i2.is_ragged && i2.is_polyhedron
    @test i2.num_cells == 2 && i2.num_faces == 7 && i2.num_nodes == 21

    @test polygon_block(m, 1) == rows
    @test polyhedron_block(m, 2) == cells

    # Each accessor refuses the wrong shape by name rather than returning
    # something plausible-but-wrong.
    @test_throws ArgumentError polyhedron_block(m, 1)
    @test_throws ArgumentError polygon_block(m, 2)
    @test_throws MeshioPlusPlus.MeshioError connectivity(m, 1)
    @test_throws Exception polygon_block(m, 3)   # out of range
    close(m)
end

@testset "zero-copy borrow really aliases" begin
    m = fixture()
    b = points_ptr(m)
    @test size(b) == (3, 5)
    @test b[1, 2] == 1.1
    # Writing through the borrow must be visible to the mesh, or it is not a
    # borrow at all.
    b[1, 2] = 99.0
    @test points(m)[1, 2] == 99.0
    close(m)
end

@testset "borrow window is enforced" begin
    m = fixture()
    b = points_ptr(m)
    @test b[1, 2] == 1.1                     # readable inside the window

    # A MUTATING call ends the window. The C API says a borrowed pointer stays
    # valid only until the next mutating mio_mesh_* call; rather than let a
    # stale read succeed by luck, using the borrow afterwards must throw.
    add_point_data!(m, "extra", Float64[1.0, 2.0, 3.0, 4.0, 5.0])
    @test_throws BorrowError b[1, 2]
    @test_throws BorrowError size(b)

    # A fresh borrow works again.
    b2 = points_ptr(m)
    @test b2[1, 2] == 1.1

    # Closing the mesh also ends every window.
    close(m)
    @test_throws BorrowError b2[1, 2]

    # A read-only accessor does NOT end the window (C API rule 3).
    m2 = fixture()
    b3 = points_ptr(m2)
    _ = num_points(m2); _ = points(m2); _ = connectivity(m2, 1)
    @test b3[1, 2] == 1.1
    close(m2)
end

@testset "file round-trip" begin
    mktempdir() do dir
        m = fixture()
        for ext in ("vtu", "vtk")
            path = joinpath(dir, "mesh.$ext")
            mio.write(m, path)
            @test isfile(path)
            r = mio.read(path)
            @test num_points(r) == 5
            @test point_dim(r) == 3
            @test cell_block_type(r, 1) == "tetra"
            @test points(r) ≈ POINTS
            @test connectivity(r, 1) == CONN
            @test point_data(r, "displacement") ≈ VEC
            @test meshes_equal(m, r; atol=1e-10, rtol=1e-10)
            close(r)
        end

        # convert() never materializes a mesh for the caller.
        src = joinpath(dir, "mesh.vtu")
        dst = joinpath(dir, "converted.vtk")
        mio.convert(src, dst)
        @test isfile(dst)

        # sniff_format reads the leading bytes, not the extension.
        @test mio.sniff_format(src) == "vtu"
        close(m)
    end
end

@testset "selective read and metadata" begin
    mktempdir() do dir
        path = joinpath(dir, "mesh.vtu")
        m = fixture()
        mio.write(m, path)
        close(m)

        r = mio.read(path; options=ReadOptions(points_only=true))
        @test num_points(r) == 5
        @test isempty(point_data_names(r))    # every array skipped
        close(r)

        # `arrays=[...]` keeps only what is named; the empty vector is NOT the
        # same request as `nothing` (which keeps everything).
        r2 = mio.read(path; options=ReadOptions(arrays=["temperature"]))
        @test point_data_names(r2) == ["temperature"]
        close(r2)
        r3 = mio.read(path; options=ReadOptions(arrays=String[]))
        @test isempty(point_data_names(r3))
        close(r3)

        meta = read_metadata(path)
        @test meta.num_points == 5
        @test meta.point_dim == 3
        @test meta.num_cells == 2
        @test meta.num_cell_blocks == 1
        @test meta.cell_blocks[1].type == "tetra"
        @test meta.cell_blocks[1].num_cells == 2
        @test !meta.cell_blocks[1].is_ragged
        @test "temperature" in meta.point_data_names
        @test reader_supports_options("vtu")
    end
end

@testset "regions round-trip" begin
    m = fixture()
    add_region!(m, "inlet", :point, [1, 3, 5])
    add_region!(m, "solid", :cell, [1, 2]; dim=3, tag=17)
    # (cell, facet) pairs: column-major, so 2 rows x N columns.
    add_region!(m, "wall", :side, Int64[1 2; 0 2])

    rs = regions(m)
    @test length(rs) == 3
    by_name = Dict(r.name => r for r in rs)

    inlet = by_name["inlet"]
    @test inlet.kind === :point
    @test vec(inlet.entries) == [1, 3, 5]     # 1-based, as given

    solid = by_name["solid"]
    @test solid.kind === :cell
    @test solid.dim == 3
    @test solid.tag == 17
    @test vec(solid.entries) == [1, 2]

    wall = by_name["wall"]
    @test wall.kind === :side
    @test size(wall.entries) == (2, 2)
    # Canonicalized (sorted lexicographically on the pair) by the core. The
    # CELL column is 1-based here; the FACET column is an ordinal within the
    # cell type, not a mesh index, so it is NOT shifted -- the same rule the
    # Fortran module follows.
    @test wall.entries[1, :] == [1, 2]        # cells, 1-based
    @test wall.entries[2, :] == [0, 2]        # facets, unshifted

    @test_throws ArgumentError add_region!(m, "bad", :point, [0, 1])   # 0 is not 1-based
    @test_throws ArgumentError add_region!(m, "bad", :side, [1, 2])    # needs pairs
    close(m)
end

@testset "regions survive a file round-trip" begin
    mktempdir() do dir
        m = Mesh()
        set_points!(m, POINTS)
        add_cell_block!(m, "triangle", Int64[1 2; 2 3; 3 4])
        add_region!(m, "physical", :cell, [1, 2]; dim=2, tag=5)

        # Abaqus is one of the two Phase-1 region formats. Its *ELSET has no
        # notion of a dimension or an integer tag, so those come back
        # unspecified (-1) -- the group itself is what survives.
        path = joinpath(dir, "mesh.inp")
        mio.write(m, path)
        r = mio.read(path)
        got = only(filter(x -> x.name == "physical", regions(r)))
        @test got.kind === :cell
        @test vec(got.entries) == [1, 2]
        close(r); close(m)
    end
end

@testset "operations: geometry" begin
    m = fixture()

    surf = extract_surface(m)
    @test num_points(surf) > 0
    @test cell_block_type(surf, 1) == "triangle"
    close(surf)

    surf2 = extract_surface(m; record_parent_ids=true)
    @test "surface:parent_cell" in cell_data_names(surf2)
    close(surf2)

    skin = extract_skin(m; linearize=true)
    @test num_cells(skin) > 0
    close(skin)

    q = attach_quality(m)
    @test any(startswith(n, "quality:") for n in cell_data_names(q))
    close(q)

    qc = quality_counts(m)
    @test qc.num_cells == 2

    st = stats(m)
    @test st.num_points == 5
    @test st.num_cells == 2
    @test st.bbox_min[1] ≈ 0.0
    @test st.bbox_max[1] ≈ 1.4
    @test st.extent[1] ≈ 1.4

    @test compute_bandwidth(m) >= 0
    close(m)
end

@testset "operations: transform" begin
    m = fixture()
    M = [1.0 0 0 2.0
         0 1.0 0 3.0
         0 0 1.0 4.0
         0 0 0 1.0]
    t = mio.transform(m, M)
    p = points(t)
    @test p[:, 1] ≈ [2.0, 3.0, 4.0]          # the origin moved by the translation
    @test p[:, 2] ≈ POINTS[:, 2] .+ [2.0, 3.0, 4.0]
    @test connectivity(t, 1) == CONN         # topology untouched
    close(t); close(m)
    @test_throws ArgumentError mio.transform(fixture(), [1.0 0; 0 1.0])
end

@testset "operations: refine and the map sentinel" begin
    m = fixture()
    r = refine(m; levels=1)
    @test num_cells(r.mesh) == 16            # 2 tetra -> 8 children each
    # Refinement never prunes, so the point map is the identity -- and it is
    # 1-based here, like every copied index array.
    @test r.point_map == collect(1:5)
    @test length(r.cell_maps) == 1
    @test r.cell_maps[1] == [1, 9]           # each parent's first child, 1-based
    close(r.mesh)

    # convert_cells CAN prune, and a pruned point is reported as 0 -- never a
    # valid 1-based index, so the sentinel stays unambiguous (Fortran's rule).
    c = convert_cells(m, "simplexify")
    @test num_cells(c.mesh) == 2             # a tetra is already a simplex
    @test all(>=(0), c.point_map)
    close(c.mesh)
    close(m)
end

@testset "operations: subdivide (polyhedral refinement)" begin
    m = fixture()
    s = subdivide(m; record_parent_ids=true)
    @test num_cell_blocks(s.mesh) == 1
    @test cell_block_type(s.mesh, 1) == "polyhedron"
    info = cell_block_info(s.mesh, 1)
    @test info.is_polyhedron
    @test info.num_cells == 8                # 2 tetra, 4 faces each -> 8 children
    @test num_points(s.mesh) == num_points(m) + 2  # one interior apex per parent

    # No point map (subdivide never prunes/renumbers a point); cell_maps is
    # per-block, first-child, 1-based -- the second parent's children start
    # right after the first parent's 4.
    @test length(s.cell_maps) == 1
    @test s.cell_maps[1] == [1, 5]
    @test "subdivide:parent_cell" in cell_data_names(s.mesh)
    close(s.mesh)
    close(m)
end

@testset "operations: agglomerate (polyhedral coarsening)" begin
    m = fixture()  # 2 tetra sharing one face
    a = agglomerate(m; target_group_size=2)
    @test num_cell_blocks(a.mesh) == 1
    @test cell_block_type(a.mesh, 1) == "polyhedron"
    info = cell_block_info(a.mesh, 1)
    @test info.is_polyhedron
    @test info.num_cells == 1                # both tetra merge into one cell
    @test num_points(a.mesh) == num_points(m)  # points never pruned/renumbered

    # cell_map is a single FLAT array (unlike subdivide's per-block
    # cell_maps): both input cells land in the one merged output cell.
    @test length(a.cell_map) == 2
    @test a.cell_map[1] == a.cell_map[2] == 1
    close(a.mesh)
    close(m)
end

@testset "operations: selective refine with a conforming closure" begin
    m = fixture()

    # One cell only. `cells` is 1-based here, like every index array this
    # binding takes, and is shifted to the C API's 0-based numbering inside.
    sel = refine(m; cells=[1], record_levels=true)
    @test num_cells(sel.mesh) > num_cells(m)
    @test num_cells(sel.mesh) < 16           # not the uniform 8-per-cell
    @test "refine:level" in cell_data_names(sel.mesh)
    close(sel.mesh)

    # Propagation is the always-works baseline: on a connected mesh it reaches
    # every cell, which is exactly the uniform refinement.
    prop = refine(m; cells=[1], closure="propagate")
    @test num_cells(prop.mesh) == 16
    close(prop.mesh)

    # At most one selector, and the closure name is validated before the call.
    @test_throws Exception refine(m; cells=[1], region="anything")
    @test_throws ArgumentError refine(m; closure="blue")
    @test_throws ArgumentError refine(m; where_array="q", where_op="~=")
    close(m)
end

@testset "operations: refine's persistent hierarchy" begin
    m = fixture()

    # Opt-in.
    plain = refine(m; cells=[1])
    @test !("refine:cell_id" in cell_data_names(plain.mesh))
    close(plain.mesh)

    hier = refine(m; cells=[1], record_hierarchy=true)
    @test "refine:cell_id" in cell_data_names(hier.mesh)
    @test "refine:parent_id" in cell_data_names(hier.mesh)
    ids = vec(cell_data(hier.mesh, "refine:cell_id", 1))
    parents = vec(cell_data(hier.mesh, "refine:parent_id", 1))
    # Ids/parent-ids are stable IDENTIFIERS, not index maps into a Julia
    # array -- the partition_labels precedent (returns part *ids*,
    # deliberately unshifted). They ride the raw C-side numbering unchanged.
    @test length(unique(ids)) == length(ids)  # every id unique
    # The coarse mesh (m) has 2 cells, implicitly numbered 0 and 1; every
    # parent_id must resolve to one of them.
    @test all(p -> p in (0, 1), parents)
    # Also proves the multigrid-stencil fix: this call used the redgreen
    # closure, which leaves no hanging nodes, so refine:entity would
    # normally never be attached at all.
    @test "refine:entity" in point_data_names(hier.mesh)

    # An existing hierarchy is maintained without the flag.
    again = refine(hier.mesh; cells=[1])
    @test "refine:cell_id" in cell_data_names(again.mesh)
    close(again.mesh)
    close(hier.mesh)
    close(m)
end

@testset "operations: undo_green restores the coarse parent" begin
    m = fixture()  # 2 tetrahedra sharing a whole face

    fine = refine(m; cells=[1], record_hierarchy=true, record_levels=true)
    fine_n = num_cells(fine.mesh)
    coarse_n = num_cells(m)

    undone = undo_green(m, fine.mesh)
    @test undone.num_groups_undone > 0
    @test undone.num_cells_removed > 0
    @test num_cells(undone.mesh) < fine_n
    @test num_cells(undone.mesh) > coarse_n
    # The reserved refine:* arrays are dropped entirely.
    @test !("refine:cell_id" in cell_data_names(undone.mesh))
    @test !("refine:entity" in point_data_names(undone.mesh))

    # Fails by name rather than guessing: no hierarchy on the "fine" argument.
    @test_throws MeshioError undo_green(m, m)

    close(undone.mesh)
    close(fine.mesh)
    close(m)
end

@testset "operations: split and partition own their meshes" begin
    m = fixture()

    pieces = mio.split(m; by="type")
    @test length(pieces) == 1
    key, piece = pieces[1]
    @test key == "tetra"
    # The piece must still be usable after the C result handle is gone --
    # which it is, because every piece takes ownership rather than borrowing.
    @test num_cells(piece) == 2
    @test points(piece) == POINTS
    close(piece)

    parts = partition(m, 2; method="sfc")
    @test length(parts) == 2
    @test [p.part_id for p in parts] == [0, 1]
    @test sum(num_cells(p.mesh) for p in parts) == 2   # partition of unity
    for p in parts
        @test length(p.cell_maps) == 1                 # block structure kept 1:1
        close(p.mesh)
    end

    labels = partition_labels(m, 2; method="sfc")
    @test length(labels) == 2
    # Part IDs, not indices: deliberately NOT shifted to 1-based.
    @test all(l -> 0 <= l < 2, labels)
    @test sort(labels) == [0, 1]
    close(m)
end

@testset "operations: reorder" begin
    m = fixture()
    r = reorder(m, "rcm")
    @test num_points(r.mesh) == 5
    @test sort(r.node_perm) == collect(1:5)   # a permutation, 1-based
    @test length(r.cell_perms) == 1
    @test sort(r.cell_perms[1]) == collect(1:2)
    close(r.mesh)
    close(m)
end

@testset "operations: clean, smooth, crop" begin
    m = fixture()

    c = clean(m)
    @test num_points(c.mesh) == 5             # nothing to remove in the fixture
    @test c.points_welded == 0
    close(c.mesh)

    s = smooth(m; iterations=2)
    @test num_points(s.mesh) == 5             # a pure coordinate move
    @test num_cells(s.mesh) == 2
    @test s.max_displacement >= 0.0
    close(s.mesh)

    # method="odt": tet-only, C++-core only. `m`'s own block is "tetra".
    so = smooth(m; method="odt", iterations=1, fix_boundary=false, preserve_features=false)
    @test num_points(so.mesh) == 5
    @test num_cells(so.mesh) == 2
    @test so.num_nodes_moved >= 0
    close(so.mesh)

    # optimize_volume: ODT remeshing (relocate + flip connectivity), tet-only.
    ov = optimize_volume(m; max_iterations=5)
    @test num_points(ov.mesh) == 5                       # the point set is invariant
    @test ov.num_flips >= 0
    @test ov.min_quality_after >= ov.min_quality_before - 1e-12
    close(ov.mesh)

    b = crop_bbox(m, [-1.0, -1.0, -1.0], [10.0, 10.0, 10.0])
    @test num_cells(b) == 2                   # the box holds everything
    close(b)
    b2 = crop_bbox(m, [100.0, 100.0, 100.0], [101.0, 101.0, 101.0])
    @test num_cells(b2) == 0
    close(b2)

    p = crop_plane(m, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]; mode=:any)
    @test num_cells(p) >= 0
    close(p)
    close(m)
end

@testset "operations: slice and isosurface" begin
    m = fixture()

    sl = mio.slice(m, [0.5, 0.5, 0.5], [0.0, 0.0, 1.0])
    @test num_points(sl) >= 0                 # may be empty; never an error
    close(sl)

    iso = isosurface(m, "temperature", 3.0)
    @test num_points(iso) >= 0
    if num_cells(iso) > 0
        @test "iso:value" in cell_data_names(iso)
        @test "iso:index" in cell_data_names(iso)
    end
    close(iso)

    # A cell_data array has no level set: that must fail by name.
    @test_throws MeshioError isosurface(m, "material", 8.0)
    close(m)
end

@testset "operations: gradient" begin
    m = fixture()

    g = gradient(m, "temperature")
    @test g.num_skipped == 0
    @test "temperature:gradient" in cell_data_names(g.mesh)
    close(g.mesh)

    # The underscore spelling must reach the C API's hyphenated name.
    l = gradient(m, "temperature"; method=:least_squares)
    @test l.num_fallback >= 0
    close(l.mesh)

    p = gradient(m, "temperature"; location=:point, output="dT")
    @test "dT" in point_data_names(p.mesh)
    close(p.mesh)

    # A cell_data array is piecewise constant and has no derivative, and a
    # scalar has no divergence: both must fail by name.
    @test_throws MeshioError gradient(m, "material")
    @test_throws MeshioError gradient(m, "temperature"; operator=:divergence)
    close(m)
end

@testset "operations: hessian" begin
    m = fixture()

    h = hessian(m, "temperature")
    @test h.num_skipped == 0
    @test "temperature:hessian" in cell_data_names(h.mesh)
    close(h.mesh)

    l = hessian(m, "temperature"; method=:least_squares)
    @test l.num_fallback >= 0
    close(l.mesh)

    p = hessian(m, "temperature"; location=:point, output="H2")
    @test "H2" in point_data_names(p.mesh)
    close(p.mesh)

    # A cell_data array has no derivative, and a vector field is scalar-only:
    # both must fail by name.
    @test_throws MeshioError hessian(m, "material")
    @test_throws MeshioError hessian(m, "displacement")
    close(m)
end

@testset "operations: estimate_error" begin
    m = fixture()

    e = estimate_error(m, "temperature")
    @test e.num_skipped == 0
    @test e.global_error >= 0.0
    @test e.num_marked == 0
    @test "error:zz" in cell_data_names(e.mesh)
    @test !("error:marked" in cell_data_names(e.mesh))
    close(e.mesh)

    f = estimate_error(m, "temperature"; marking=:absolute, marking_value=1e-9,
                       output="ind", marked="flag")
    @test "ind" in cell_data_names(f.mesh)
    @test "flag" in cell_data_names(f.mesh)
    @test f.num_marked >= 0
    close(f.mesh)

    # A cell_data array has no derivative to recover, and an out-of-range
    # marking_value for "fraction" is rejected: both must fail by name.
    @test_throws MeshioError estimate_error(m, "material")
    @test_throws MeshioError estimate_error(m, "temperature"; marking=:fraction,
                                            marking_value=1.5)
    close(m)
end

@testset "operations: remesh" begin
    # A regular octahedron: the smallest closed 2-manifold triangle mesh with
    # no coplanar faces.
    octa = Mesh()
    set_points!(octa, Float64[1 -1 0 0 0 0; 0 0 1 -1 0 0; 0 0 0 0 1 -1])
    add_cell_block!(octa, "triangle",
                    Int64[1 3 2 4 3 2 4 1; 3 2 4 1 1 3 2 4; 5 5 5 5 6 6 6 6])

    r = remesh(octa, 30)
    @test r.num_clusters == 30
    @test num_points(r.mesh) == 30
    @test r.subdivide_applied > 0  # 6 vertices cannot support 30 clusters unsubdivided
    @test r.num_iterations >= 0
    @test r.num_isolated_clusters >= 0
    close(r.mesh)

    # The quadric ("feature-preserving") metric is accepted too.
    q = remesh(octa, 30; metric=:quadric)
    @test q.num_clusters == 30
    close(q.mesh)

    # The anisotropic metric + max_anisotropy go through mio_remesh_ex.
    a = remesh(octa, 30; metric=:anisotropic, max_anisotropy=3.0)
    @test a.num_clusters > 0
    @test a.num_non_manifold_vertices >= 0
    close(a.mesh)

    # max_anisotropy away from the default under a non-anisotropic metric is
    # guarded, not silently ignored.
    @test_throws MeshioError remesh(octa, 30; metric=:isotropic, max_anisotropy=2.0)

    # Too few clusters is rejected by name.
    @test_throws MeshioError remesh(octa, 3)

    close(octa)
end

@testset "operations: remesh_volume" begin
    octa = Mesh()
    set_points!(octa, Float64[1 -1 0 0 0 0; 0 0 1 -1 0 0; 0 0 0 0 1 -1])
    add_cell_block!(octa, "triangle",
                    Int64[1 3 2 4 3 2 4 1; 3 2 4 1 1 3 2 4; 5 5 5 5 6 6 6 6])

    # Unlike remesh, remesh_volume accepts the closed surface directly and
    # returns a tetra mesh.
    r = remesh_volume(octa; cell_size=0.4, watertight_check=:off)
    @test num_points(r.mesh) > 0
    @test num_cell_blocks(r.mesh) == 1
    @test cell_block_type(r.mesh, 1) == "tetra"
    @test r.num_tets > 0
    @test r.num_tets == num_cells(r.mesh)
    @test r.num_vertices_warped >= 0
    @test r.num_tets_rejected >= 0
    @test r.num_non_manifold_edges >= 0
    @test r.input_quality.watertight
    close(r.mesh)

    # Neither resolution nor cell_size given is rejected by name.
    @test_throws MeshioError remesh_volume(octa)

    close(octa)
end

@testset "operations: merge and interpolate" begin
    a = fixture()
    b = fixture()
    merged = mio.merge([a, b])
    @test num_points(merged) == 10
    @test num_cells(merged) == 4
    @test "source_mesh_id" in cell_data_names(merged)
    close(merged)

    merged2 = mio.merge([a, b]; weld=true, atol=1e-9, source_tag=false)
    @test num_points(merged2) == 5            # the two copies weld together
    close(merged2)

    # The target already carries "temperature", so the default on_conflict
    # ("error") must refuse rather than silently overwrite.
    target = fixture()
    @test_throws MeshioError interpolate(a, target; arrays=["temperature"])

    it = interpolate(a, target; arrays=["temperature"], on_conflict="suffix")
    @test "temperature_interp" in point_data_names(it)
    @test point_data(it, "temperature_interp") ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
    close(it)

    # Onto a target with no data of its own, the plain call works.
    bare = Mesh()
    set_points!(bare, POINTS)
    add_cell_block!(bare, "tetra", CONN)
    it2 = interpolate(a, bare; arrays=["temperature"])
    @test point_data(it2, "temperature") ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
    close(it2); close(bare); close(target); close(a); close(b)
end

@testset "operations: diff" begin
    a = fixture()
    b = fixture()
    rep = mio.diff(a, b)
    @test rep.verdict === :identical
    @test rep.points.num_exceeding == 0

    pb = points_ptr(b)
    pb[1, 1] = 5.0
    rep2 = mio.diff(a, b)
    @test rep2.verdict === :different
    @test rep2.points.max_abs_error ≈ 5.0
    @test !meshes_equal(a, b)
    @test meshes_equal(a, b; atol=10.0)       # inside a loose tolerance
    close(a); close(b)
end

@testset "data operations never touch geometry" begin
    m = fixture()
    before_points = points(m)
    before_conn = connectivity(m, 1)

    d = data_drop(m, :point, ["temperature"])
    @test point_data_names(d) == ["displacement"]
    @test points(d) == before_points          # geometry bit-identical
    @test connectivity(d, 1) == before_conn
    close(d)

    k = data_keep(m, :point, ["temperature"])
    @test point_data_names(k) == ["temperature"]
    close(k)

    r = data_rename(m, :point, "temperature", "T")
    @test "T" in point_data_names(r)
    @test !("temperature" in point_data_names(r))
    close(r)

    p2c = data_point_to_cell(m, ["temperature"])
    @test "temperature" in cell_data_names(p2c)
    close(p2c)

    c2p = data_cell_to_point(m, ["material"]; weight=:uniform)
    @test "material" in point_data_names(c2p)
    close(c2p)

    calc = data_calc(m, "temperature * 2.0", "doubled")
    @test point_data(calc, "doubled") ≈ [2.0, 4.0, 6.0, 8.0, 10.0]
    close(calc)

    cond = data_condition(m, :point, ["temperature"]; mode=:normalize, lo=0.0, hi=1.0)
    t = point_data(cond, "temperature")
    @test minimum(t) ≈ 0.0
    @test maximum(t) ≈ 1.0
    close(cond)

    info = data_info(m)
    @test length(info) == 3
    temp = only(filter(x -> x.name == "temperature", info))
    @test temp.location === :point
    @test temp.num_entries == 5
    @test temp.num_components == 1
    @test temp.min ≈ 1.0
    @test temp.max ≈ 5.0
    @test temp.mean ≈ 3.0
    @test temp.num_nan == 0
    @test length(temp.components) == 1
    close(m)
end

@testset "field integration (data_integrate)" begin
    m = fixture()
    add_region!(m, "solid", :cell, [1]; dim=3, tag=17)

    report = data_integrate(m, ["material"])
    @test length(report) == 1
    arr = only(report)
    @test arr.name == "material"
    @test arr.num_components == 1
    @test arr.domain.num_cells == 2
    @test arr.domain.num_skipped == 0
    dc = only(arr.domain.components)
    @test dc.domain_measure > 0
    @test dc.mean ≈ dc.total / dc.domain_measure
    @test dc.num_nan == 0

    @test length(arr.regions) == 1
    reg = only(arr.regions)
    @test reg.name == "solid"
    @test reg.num_cells == 1
    rc = only(reg.components)
    @test rc.total < dc.total   # one cell contributes less than both

    # Empty names means every cell_data array (there's exactly one).
    @test length(data_integrate(m)) == 1

    # A point_data-only name throws naming the fix.
    @test_throws MeshioError data_integrate(m, ["temperature"])

    close(m)
end

@testset "transient XDMF series" begin
    # The one writer `write` cannot express: the grid goes out once and each
    # step is appended. "XML" keeps everything in the single .xdmf, so this
    # runs against a library built without HDF5 too.
    mktempdir() do dir
        path = joinpath(dir, "series.xdmf")
        m = fixture()

        s = XdmfSeries(path; data_format="XML")
        @test isopen(s)
        @test num_steps(s) == 0
        write_points_cells!(s, m)

        # temperature = t + node id, so no two steps share a value and a
        # step mix-up cannot pass.
        step = Mesh()
        set_points!(step, POINTS)
        add_cell_block!(step, "tetra", CONN)
        times = [0.0, 0.5, 1.0]
        for t in times
            add_point_data!(step, "temperature", Float64[t + i for i in 1:5])
            write_data!(s, t, step)
        end
        @test num_steps(s) == 3
        @test occursin("3 steps", sprint(show, s))

        # The .xdmf is buffered until finalize; nothing is readable before it.
        finalize!(s)
        finalize!(s)                       # idempotent
        close(s)
        @test !isopen(s)
        close(s)                           # idempotent
        @test_throws MeshioError num_steps(s)
        close(step)

        meta = read_metadata(path)
        @test meta.time_values ≈ times

        for (k, t) in enumerate(times)
            back = mio.read(path; options=ReadOptions(time_step=k - 1))
            @test num_points(back) == 5
            @test point_data(back, "temperature") ≈ Float64[t + i for i in 1:5]
            close(back)
        end

        # The do-block form closes even when the body throws.
        outer = nothing
        @test_throws ErrorException XdmfSeries(joinpath(dir, "boom.xdmf");
                                               data_format="XML") do inner
            outer = inner
            write_points_cells!(inner, m)
            error("boom")
        end
        @test !isopen(outer)

        # An unknown data format throws from the constructor, carrying the C
        # API's own message.
        @test_throws MeshioError XdmfSeries(joinpath(dir, "bad.xdmf");
                                            data_format="NoSuchFormat")
        close(m)
    end
end

@testset "settings pipeline" begin
    # Behaviour follows the build: with the JSON parser a bad document fails
    # naming the offending op; without it every entry point fails naming
    # -DMESHIOPLUSPLUS_WITH_JSON=ON. Never a missing symbol either way.
    bad = """{"Input": {"Path": "a"}, "Output": {"Path": "b"},
              "Operations": [{"Op": "Nope"}]}"""
    err = try
        run_pipeline_json(bad)
        nothing
    catch e
        e
    end
    @test err isa MeshioError
    if pipeline_has_json()
        @test occursin("Nope", err.msg)
        # ... and a good document runs end to end.
        mktempdir() do dir
            inp = joinpath(dir, "in.vtu")
            out = joinpath(dir, "out.vtu")
            m = fixture()
            MeshioPlusPlus.write(m, inp)
            close(m)
            settings = joinpath(dir, "settings.json")
            open(settings, "w") do io
                print(io, """{"Input": {"Path": "$(inp)"},
                              "Operations": [{"Op": "Quality"}],
                              "Output": {"Path": "$(out)"}}""")
            end
            run_pipeline_file(settings)
            back = MeshioPlusPlus.read(out)
            @test "quality:scaled_jacobian" in cell_data_names(back)
            close(back)
        end
    else
        @test occursin("MESHIOPLUSPLUS_WITH_JSON", err.msg)
    end
end

@testset "errors carry mio_last_error()" begin
    # Every failure surfaces as a MeshioError carrying the C API's own
    # thread-local message; a status code never reaches the caller.
    err = try
        mio.read("/definitely/not/a/mesh/file.vtu")
        nothing
    catch e
        e
    end
    @test err isa MeshioError
    @test !isempty(err.msg)
    @test err.msg == last_error()             # straight from mio_last_error()
    @test occursin("MeshioError", sprint(showerror, err))

    m = fixture()
    @test_throws MeshioError point_data(m, "no_such_array")
    @test_throws MeshioError cell_block_type(m, 99)
    @test_throws MeshioError data_rename(m, :point, "missing", "x")

    # Using a closed mesh is an error, not a crash.
    close(m)
    @test_throws MeshioError num_points(m)
    close(m)                                   # idempotent
    @test !isopen(m)
end

@testset "grids and signed distance" begin
    # A lattice from nothing: the one constructor that takes no input mesh.
    g = grid([2, 2, 2])
    @test num_points(g) == 27
    @test num_cell_blocks(g) == 1
    close(g)

    # An empty lattice is a legal request, not an error.
    e = grid([0, 0, 0])
    @test num_points(e) == 0
    close(e)

    # The unit cube as a closed, outward-wound triangle surface.
    cube = Mesh()
    set_points!(cube, Float64[0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1])
    add_cell_block!(cube, "triangle",
                   Int64[1 1 5 5 1 1 2 2 3 3 4 4;
                         3 4 6 7 2 6 3 7 4 8 1 5;
                         2 3 7 8 6 5 7 6 8 7 5 8])

    q = surface_watertight_check(cube)
    @test q.watertight
    @test q.boundary_edges == 0

    v = voxelize(cube; resolution=[4, 4, 4])
    @test v.num_occupied == 64
    @test v.dims == (4, 4, 4)
    @test v.spacing[1] ≈ 0.25
    close(v.mesh)

    inside = voxelize(cube; resolution=[5, 5, 5],
                      bounds=[-0.5, -0.5, -0.5, 1.5, 1.5, 1.5],
                      fill=:inside, watertight_check=:off)
    @test inside.num_occupied == 27
    close(inside.mesh)

    # An unknown fill fails by name rather than silently defaulting.
    @test_throws ArgumentError voxelize(cube; resolution=[2, 2, 2], fill=:solid)

    # The cube's exact SDF is known: the centre is 0.5 in, the others 1.0 out.
    d = sample_distance(cube, [0.5 2.0 -1.0; 0.5 0.5 0.5; 0.5 0.5 0.5];
                        watertight_check=:off)
    @test length(d) == 3
    @test d[1] ≈ -0.5 atol = 1e-12
    @test d[2] ≈ 1.0 atol = 1e-12
    @test d[3] ≈ 1.0 atol = 1e-12

    qgrid = grid([2, 2, 2]; origin=(-0.5, -0.5, -0.5), spacing=(1.0, 1.0, 1.0))
    f = distance_to_surface(qgrid, cube; watertight_check=:off, record_inside=true)
    @test f.quality.watertight
    @test f.num_banded == 0
    @test num_point_data(f.mesh) == 2   # sdf:distance and sdf:inside
    close(f.mesh)
    close(qgrid)

    # compute_sdf: the grid and the field in one call.
    sdf = compute_sdf(cube; resolution=[4, 4, 4], watertight_check=:off)
    @test sdf.dims == (4, 4, 4)
    @test sdf.max_depth == 0
    @test num_point_data(sdf.mesh) == 1
    close(sdf.mesh)

    # The octree refines only near the surface.
    tree = compute_sdf(cube; structure=:octree, root_resolution=4, max_depth=2,
                       watertight_check=:off)
    @test tree.max_depth == 2
    @test cell_block_info(tree.mesh, 1).num_cells > 64
    @test cell_block_info(tree.mesh, 1).num_cells < 4096
    close(tree.mesh)

    # resolution/cell_size size a voxel grid; an octree's finest cell is already
    # determined by root_resolution and max_depth.
    @test_throws MeshioError compute_sdf(cube; structure=:octree, resolution=[4, 4, 4])
    @test_throws ArgumentError compute_sdf(cube; structure=:quadtree)

    # The predicate crop, composed with a cell-centred distance field.
    dom = grid([4, 4, 4]; origin=(-0.5, -0.5, -0.5), spacing=(0.5, 0.5, 0.5))
    field = distance_to_surface(dom, cube; location=:center, watertight_check=:off)
    kept = crop_predicate(field.mesh, "sdf:distance"; compare="<", value=0.0)
    @test cell_block_info(kept, 1).num_cells == 8
    close(kept)
    @test_throws ArgumentError crop_predicate(field.mesh, "sdf:distance"; compare="~")
    @test_throws MeshioError crop_predicate(field.mesh, "nope")
    close(field.mesh)
    close(dom)
    close(cube)
end

@testset "operations: repair, shrinkwrap, sobolev_deform" begin
    # The unit cube surface with two facets flipped: repair rewinds exactly
    # those and reports a watertight output; minus one facet, the hole is
    # filled by a centroid fan in a trailing block.
    conn = Int64[1 1 5 5 1 1 2 2 3 3 4 4;
                 3 4 6 7 2 6 3 7 4 8 1 5;
                 2 3 7 8 6 5 7 6 8 7 5 8]
    broken = copy(conn)
    broken[2, 1], broken[3, 1] = broken[3, 1], broken[2, 1]
    broken[2, 9], broken[3, 9] = broken[3, 9], broken[2, 9]
    cube = Mesh()
    set_points!(cube, Float64[0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1])
    add_cell_block!(cube, "triangle", broken)
    r = repair(cube)
    @test r.quality_before.inconsistent_pairs > 0
    @test r.quality_after.watertight
    @test r.num_flipped == 2
    @test r.num_components == 1
    @test r.num_holes_detected == 0
    @test num_points(r.mesh) == 8
    close(r.mesh)
    close(cube)

    holed = Mesh()
    set_points!(holed, Float64[0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1])
    add_cell_block!(holed, "triangle", conn[:, 2:end])
    h = repair(holed)
    @test h.num_holes_filled == 1
    @test h.num_faces_added == 3
    @test h.num_points_added == 1
    @test h.quality_after.watertight
    @test num_cell_blocks(h.mesh) == 2
    close(h.mesh)
    h2 = repair(holed; fill_holes=false)
    @test h2.num_holes_detected == 0
    @test h2.quality_after.boundary_edges == 3
    close(h2.mesh)
    close(holed)

    # Shrinkwrap three points onto the z = 0 square with an offset; a float
    # weights array blends the second one halfway.
    target = Mesh()
    set_points!(target, Float64[0 4 4 0; 0 0 4 4; 0 0 0 0])
    add_cell_block!(target, "triangle", Int64[1 1; 2 3; 3 4])
    cloud = Mesh()
    set_points!(cloud, Float64[1 2 3; 1 2 1; 0.5 -0.7 1.5])
    add_cell_block!(cloud, "vertex", Int64[1 2 3])
    add_point_data!(cloud, "w", Float64[1.0, 0.5, 1.0])
    w = shrinkwrap(cloud, target; offset=0.25, weights="w", record_distance=true)
    @test w.num_projected == 3
    @test w.num_skipped == 0
    @test w.quality.boundary_edges == 4
    p = points(w.mesh)
    @test p[3, 1] ≈ 0.25 atol=1e-15
    @test p[3, 2] ≈ -0.7 + 0.5 * (0.25 + 0.7) atol=1e-15
    @test p[3, 3] ≈ 0.25 atol=1e-15
    close(w.mesh)
    @test_throws ArgumentError shrinkwrap(cloud, target; normal_weight=:nope)
    close(cloud)
    close(target)

    # Sobolev: a checkerboard on a triangle grid is damped, a pinned corner
    # stays, and a constant field is preserved exactly in zero iterations.
    n = 4
    gp = zeros(3, n * n)
    d = zeros(3, n * n)
    for j in 0:n-1, i in 0:n-1
        gp[:, j*n+i+1] = [i, j, 0.0]
        d[3, j*n+i+1] = isodd(i + j) ? 0.1 : -0.1
    end
    gc = Int64[]
    for j in 0:n-2, i in 0:n-2
        a = j * n + i + 1
        append!(gc, [a, a + 1, a + n + 1, a, a + n + 1, a + n])
    end
    gridm = Mesh()
    set_points!(gridm, gp)
    add_cell_block!(gridm, "triangle", reshape(gc, 3, :))
    add_point_data!(gridm, "d", d)
    pin = zeros(n * n); pin[1] = 1.0
    add_point_data!(gridm, "pin", pin)
    s = sobolev_deform(gridm, "d", 2.0; fixed_points_array="pin", record_filtered=true)
    @test s.converged
    @test s.num_iterations > 0
    @test s.num_fixed == 1
    @test s.num_isolated == 0
    @test 0.0 < s.max_displacement < 0.1
    sp = points(s.mesh)
    @test sp[3, 1] == 0.0
    @test num_point_data(s.mesh) == 3
    close(s.mesh)
    add_point_data!(gridm, "c", fill(0.4, 3, n * n))
    c = sobolev_deform(gridm, "c", 2.0)
    @test c.num_iterations == 0
    @test all(points(c.mesh)[1, :] .== gp[1, :] .+ 0.4)
    close(c.mesh)
    @test_throws MeshioError sobolev_deform(gridm, "missing", 1.0)
    close(gridm)
end

end # testset
