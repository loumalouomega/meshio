!  License:         MIT License
!                   meshio++ default license: LICENSE
!
!  Main authors:    Vicente Mataix Ferrandiz
!
! Test program for the meshio++ Fortran module (registered as the `fortran_api`
! ctest). Plain checks, no framework: exit code 0 iff every check passed.
! argv(1) is a path PREFIX for the files the test writes (e.g.
! "<builddir>/fortran_test_out" -> "<builddir>/fortran_test_out_mesh.vtu").
!
! The mesh is deliberately non-square everywhere (5 points x 3 dims, 2 cells x
! 4 nodes, 3-component vector data) with asymmetric coordinates, so a
! transposed array mapping or a missed 1-based shift cannot cancel out.
program test_fortran_api
    use, intrinsic :: iso_fortran_env, only: real64, int64, error_unit
    use meshioplusplus
    implicit none

    type(mio_mesh) :: m, r, c, s, q, ro, seq_step
    type(mio_sequence) :: seq
    integer(int64) :: qcells, qinv, qdeg, bw
    integer(int64), allocatable :: node_perm(:)
    logical :: perm_ok, eq
    integer :: verdict
    real(real64) :: points(3, 5), vec(3, 5)
    real(real64), allocatable :: rpoints(:, :), rdata(:), rvec(:, :)
    real(real64), pointer :: pview(:, :)
    integer(int64) :: conn(4, 2)
    integer(int64), allocatable :: rconn(:, :)
    character(:), allocatable :: prefix, vtu_path, vtk_path, msg, name
    integer :: fails, ierr, nargs, arglen, i, j

    fails = 0
    nargs = command_argument_count()
    if (nargs < 1) then
        write (error_unit, '(a)') 'usage: test_fortran_api <output-path-prefix>'
        error stop 2
    end if
    call get_command_argument(1, length=arglen)
    allocate (character(arglen) :: prefix)
    call get_command_argument(1, value=prefix)
    vtu_path = prefix//'_mesh.vtu'
    vtk_path = prefix//'_mesh.vtk'

    ! ---- module-level introspection ------------------------------------
    call check(len(mio_version()) > 0, 'mio_version() is non-empty')
    call check(len(mio_mesh_backend()) > 0, 'mio_mesh_backend() is non-empty')
    call check(mio_format_readable('vtu'), 'vtu is readable')
    call check(mio_format_writable('vtu'), 'vtu is writable')
    call check(mio_format_writable('openfoam'), 'openfoam is writable since v9.20.0')
    call check(.not. mio_format_readable('nonexistent'), 'unknown format is not readable')

    ! ---- build a small tet mesh from arrays ----------------------------
    points = reshape([0.0_real64, 0.0_real64, 0.0_real64, &
                      1.1_real64, 0.2_real64, 0.3_real64, &
                      0.4_real64, 1.2_real64, 0.5_real64, &
                      0.6_real64, 0.7_real64, 1.3_real64, &
                      1.4_real64, 1.5_real64, 1.6_real64], [3, 5])
    conn = reshape([1_int64, 2_int64, 3_int64, 4_int64, &
                    2_int64, 3_int64, 4_int64, 5_int64], [4, 2])
    do i = 1, 3
        do j = 1, 5
            vec(i, j) = 10.0_real64*j + i
        end do
    end do

    call m%set_points(points)
    call m%add_cell_block('tetra', conn)
    call m%add_point_data('temperature', [1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, &
                                          5.0_real64])
    call m%add_point_data('velocity', vec)
    call m%add_cell_data('quality', [0.5_real64, 0.75_real64])

    call check(m%num_points() == 5_int64, 'num_points before write')
    call check(m%point_dim() == 3_int64, 'point_dim before write')
    call check(m%num_cell_blocks() == 1_int64, 'num_cell_blocks before write')

    ! ---- mesh operations ------------------------------------------------
    s = m%extract_surface()
    call check(s%num_cell_blocks() >= 1_int64, 'extract_surface produces cells')
    call check(s%cell_block_type(1) == 'triangle', 'extract_surface gives triangles')
    call s%free()

    q = m%attach_quality()
    call check(q%num_cell_data() >= 1_int64, 'attach_quality adds cell_data')
    call q%free()

    call m%quality_counts(num_cells=qcells, num_inverted=qinv, num_degenerate=qdeg)
    call check(qcells == 2_int64, 'quality_counts reports two cells')

    ! ---- reorder + bandwidth -------------------------------------------
    bw = m%compute_bandwidth()
    call check(bw >= 0_int64, 'compute_bandwidth is non-negative')
    ro = m%reorder('rcm', node_perm=node_perm)
    call check(ro%num_points() == 5_int64, 'reorder preserves point count')
    call check(ro%num_cell_blocks() == 1_int64, 'reorder preserves cell blocks')
    call check(allocated(node_perm), 'reorder returns a node permutation')
    call check(size(node_perm) == 5, 'node permutation has one entry per node')
    ! 1-based bijection over [1, 5]
    perm_ok = .true.
    do i = 1, 5
        if (count(node_perm == int(i, int64)) /= 1) perm_ok = .false.
    end do
    call check(perm_ok, 'node permutation is a 1-based bijection')
    call ro%free()

    ! ---- write, read back, verify everything ---------------------------
    call m%write(vtu_path)
    call r%read(vtu_path)

    call check(r%num_points() == 5_int64, 'num_points after round-trip')
    call check(r%point_dim() == 3_int64, 'point_dim after round-trip')
    call check(r%num_cell_blocks() == 1_int64, 'num_cell_blocks after round-trip')
    call check(r%cell_block_type(1) == 'tetra', 'cell block type after round-trip')
    call check(r%cell_block_num_cells(1) == 2_int64, 'cell block num_cells')
    call check(r%cell_block_nodes_per_cell(1) == 4_int64, 'cell block nodes_per_cell')
    call check(.not. r%cell_block_is_ragged(1), 'cell block is not ragged')

    ! ---- diff / equals -------------------------------------------------
    eq = m%equals(r, atol=1.0e-8_real64)
    call check(eq, 'equals: round-trip mesh equals the original within tolerance')
    verdict = m%diff(r, atol=1.0e-8_real64)
    call check(verdict <= 1, 'diff: round-trip verdict is identical or within tolerance')

    call r%get_points(rpoints)
    call check(size(rpoints, 1) == 3 .and. size(rpoints, 2) == 5, 'points shape (dim, n)')
    call check(maxval(abs(rpoints - points)) < 1.0e-12_real64, 'point coordinates round-trip')

    call r%points_ptr(pview)
    call check(associated(pview), 'points_ptr is associated')
    call check(maxval(abs(pview - points)) < 1.0e-12_real64, 'points_ptr zero-copy view')

    call r%get_cell_block(1, rconn)
    call check(size(rconn, 1) == 4 .and. size(rconn, 2) == 2, 'connectivity shape (npc, ncells)')
    call check(all(rconn == conn), '1-based connectivity round-trip')

    call check(r%num_point_data() == 2_int64, 'num_point_data after round-trip')
    name = r%point_data_name(1)  ! sorted: "temperature" < "velocity"
    call check(name == 'temperature', 'first sorted point_data name')
    call r%get_point_data('temperature', rdata)
    call check(size(rdata) == 5, 'scalar point_data size')
    call check(maxval(abs(rdata - [1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, &
                                   5.0_real64])) < 1.0e-12_real64, 'scalar point_data values')
    call r%get_point_data('velocity', rvec)
    call check(size(rvec, 1) == 3 .and. size(rvec, 2) == 5, 'vector point_data shape (comp, n)')
    call check(maxval(abs(rvec - vec)) < 1.0e-12_real64, 'vector point_data values')

    call check(r%num_cell_data() == 1_int64, 'num_cell_data after round-trip')
    call check(r%cell_data_num_blocks('quality') == 1_int64, 'cell_data num_blocks')
    call r%get_cell_data('quality', 1, rdata)
    call check(maxval(abs(rdata - [0.5_real64, 0.75_real64])) < 1.0e-12_real64, &
               'cell_data values')

    ! ---- convert + read the result -------------------------------------
    call mio_convert(vtu_path, vtk_path)
    call c%read(vtk_path)
    call check(c%num_points() == 5_int64, 'num_points after convert to vtk')
    call check(c%num_cell_blocks() == 1_int64, 'num_cell_blocks after convert to vtk')

    ! ---- second-format round-trip (gmsh) exercises another writer/reader ----
    ! VTU is XML/binary; gmsh is a wholly different text writer/reader, so this
    ! shakes out format-specific bugs the single VTU path can't.
    block
        type(mio_mesh) :: g
        real(real64), allocatable :: gpoints(:, :)
        integer(int64), allocatable :: gconn(:, :)
        character(:), allocatable :: gmsh_path

        gmsh_path = prefix//'_mesh.msh'
        call m%write(gmsh_path)
        call g%read(gmsh_path)
        call check(g%num_points() == 5_int64, 'gmsh: num_points round-trip')
        call check(g%num_cell_blocks() == 1_int64, 'gmsh: num_cell_blocks round-trip')
        call check(g%cell_block_type(1) == 'tetra', 'gmsh: cell block type round-trip')
        call g%get_points(gpoints)
        call check(maxval(abs(gpoints - points)) < 1.0e-12_real64, &
                   'gmsh: point coordinates round-trip')
        call g%get_cell_block(1, gconn)
        call check(all(gconn == conn), 'gmsh: 1-based connectivity round-trip')
        call g%free()
    end block

    ! ---- error paths ----------------------------------------------------
    ierr = 0
    call r%read(prefix//'_does_not_exist.vtu', stat=ierr, errmsg=msg)
    call check(ierr /= 0, 'reading a nonexistent file sets stat')
    call check(len(msg) > 0, 'reading a nonexistent file sets errmsg')
    call check(len(mio_error_message()) > 0, 'mio_error_message() is populated')
    call check(r%num_points() == 5_int64, 'failed read leaves the previous mesh intact')

    call m%write(prefix//'_mesh.nonsense_extension', stat=ierr, errmsg=msg)
    call check(ierr /= 0, 'unknown extension sets stat')

    ! An explicit but unknown format name is rejected via stat/errmsg too.
    ierr = 0
    msg = ''
    call m%write(vtu_path, format='not-a-real-format', stat=ierr, errmsg=msg)
    call check(ierr /= 0, 'unknown explicit format sets stat')
    call check(len(msg) > 0, 'unknown explicit format sets errmsg')

    ! ---- data operations -----------------------------------------------
    ! These act on the data arrays only; the geometry must come through
    ! untouched. `m` carries point_data temperature/velocity and cell_data
    ! quality.
    block
        type(mio_mesh) :: d
        type(mio_data_array_info), allocatable :: arrays(:)
        character(len=STRBUF_LEN), allocatable :: dkeys(:)
        integer :: k
        logical :: saw

        ! drop / keep / rename
        d = m%data_drop(MIO_DATA_POINT, ['temperature'], stat=ierr)
        call check(ierr == 0, 'data_drop succeeds')
        call check(d%num_points() == 5_int64, 'data_drop leaves geometry intact')
        call check(d%num_point_data() == 1_int64, 'data_drop removed one array')
        call d%free()

        d = m%data_keep(MIO_DATA_POINT, ['velocity   '], stat=ierr)
        call check(ierr == 0, 'data_keep succeeds')
        call check(d%num_point_data() == 1_int64, 'data_keep retained one array')
        call d%free()

        d = m%data_rename(MIO_DATA_POINT, 'temperature', 'T', stat=ierr)
        call check(ierr == 0, 'data_rename succeeds')
        call check(d%num_point_data() == 2_int64, 'data_rename preserves the count')
        call d%free()

        ! An unknown key fails through stat rather than aborting.
        ierr = 0
        d = m%data_drop(MIO_DATA_POINT, ['nope'], stat=ierr, errmsg=msg)
        call check(ierr /= 0, 'data_drop on an unknown key sets stat')
        call check(len(msg) > 0, 'data_drop on an unknown key sets errmsg')

        ! averaging
        d = m%data_point_to_cell(stat=ierr)
        call check(ierr == 0, 'data_point_to_cell succeeds')
        call check(d%num_points() == 5_int64, 'data_point_to_cell keeps geometry')
        call d%free()

        d = m%data_cell_to_point(['quality'], weight=MIO_WEIGHT_MEASURE, stat=ierr)
        call check(ierr == 0, 'data_cell_to_point (measure-weighted) succeeds')
        call check(d%num_point_data() == 3_int64, 'data_cell_to_point added an array')
        call d%free()

        ! calc
        d = m%data_calc('norm(velocity)', 'speed', stat=ierr)
        call check(ierr == 0, 'data_calc succeeds')
        call check(d%num_point_data() == 3_int64, 'data_calc added an array')
        call d%free()

        ierr = 0
        d = m%data_calc('log(temperature)', 'bad', stat=ierr, errmsg=msg)
        call check(ierr /= 0, 'data_calc rejects an unknown function')
        call check(len(msg) > 0, 'data_calc failure sets errmsg')

        ! conditioning
        d = m%data_condition(MIO_DATA_POINT, ['temperature'], mode=MIO_COND_NORMALIZE, &
                             stat=ierr)
        call check(ierr == 0, 'data_condition normalize succeeds')
        call check(d%num_points() == 5_int64, 'data_condition keeps geometry')
        call d%free()

        ! read-only summary
        arrays = m%data_info(keys=dkeys, stat=ierr)
        call check(ierr == 0, 'data_info succeeds')
        call check(size(arrays) == 3, 'data_info reports every array')
        saw = .false.
        do k = 1, size(arrays)
            if (trim(dkeys(k)) == 'temperature') then
                saw = .true.
                call check(arrays(k)%location == MIO_DATA_POINT, 'temperature is point_data')
                call check(arrays(k)%num_entries == 5_int64, 'temperature has 5 entries')
                call check(arrays(k)%num_components == 1_int64, 'temperature is scalar')
                call check(arrays(k)%num_finite == 5_int64, 'temperature is all finite')
                call check(abs(arrays(k)%min - 1.0_real64) < 1.0e-12_real64, &
                           'temperature min is 1')
                call check(abs(arrays(k)%max - 5.0_real64) < 1.0e-12_real64, &
                           'temperature max is 5')
                call check(abs(arrays(k)%mean - 3.0_real64) < 1.0e-12_real64, &
                           'temperature mean is 3')
            end if
        end do
        call check(saw, 'data_info named the temperature array')
    end block

    ! field integration -- gradient's counterpart (total/mean over cells)
    block
        type(mio_field_integral_info), allocatable :: fi(:)
        character(len=STRBUF_LEN), allocatable :: fikeys(:)
        real(real64), allocatable :: fitotals(:), fimeans(:), fidm(:)
        integer(int64), allocatable :: finan(:)

        fi = m%data_integrate(arrays=['quality'], keys=fikeys, totals=fitotals, means=fimeans, &
                              domain_measures=fidm, num_nans=finan, stat=ierr)
        call check(ierr == 0, 'data_integrate succeeds')
        call check(size(fi) == 1, 'data_integrate reports one array')
        call check(trim(fikeys(1)) == 'quality', 'data_integrate names the array')
        call check(fi(1)%num_components == 1_int64, 'quality is scalar')
        call check(fi(1)%num_cells == 2_int64, 'both tets have a computable measure')
        call check(fi(1)%num_skipped == 0_int64, 'no cells skipped')
        call check(fidm(1) > 0.0_real64, 'domain measure is positive')
        call check(abs(fimeans(1) - fitotals(1) / fidm(1)) < 1.0e-9_real64, &
                   'mean is total / domain_measure')
        call check(finan(1) == 0_int64, 'no NaN values')

        ! No array filter means every cell_data array (there is exactly one).
        fi = m%data_integrate(keys=fikeys, stat=ierr)
        call check(ierr == 0, 'data_integrate with no array filter succeeds')
        call check(size(fi) == 1, 'no filter still reports the one cell_data array')

        ! A point_data-only name fails, naming the fix.
        ierr = 0
        fi = m%data_integrate(arrays=['temperature'], stat=ierr, errmsg=msg)
        call check(ierr /= 0, 'data_integrate rejects a point_data array')
        call check(len(msg) > 0, 'data_integrate failure sets errmsg')
    end block

    ! -- selective reads and read_metadata ---------------------------------
    block
        type(mio_mesh) :: sel
        type(mio_metadata) :: meta
        integer :: st

        call m%write('fortran_selective.vtu', stat=st)
        call check(st == 0, 'write for selective read')

        ! points_only keeps geometry, drops every data array.
        call sel%read('fortran_selective.vtu', points_only=.true., stat=st)
        call check(st == 0, 'selective read succeeded')
        call check(sel%num_points() > 0, 'points_only kept the points')
        call sel%free()

        ! A zero-sized `arrays` means "no arrays", not "every array".
        block
            character(len=16) :: wanted(0)
            call sel%read('fortran_selective.vtu', arrays=wanted, stat=st)
            call check(st == 0, 'empty-arrays read succeeded')
            call sel%free()
        end block

        meta = mio_read_metadata('fortran_selective.vtu', stat=st)
        call check(st == 0, 'read_metadata succeeded')
        call check(meta%num_points > 0, 'metadata reports points')
        call check(allocated(meta%cell_blocks), 'metadata reports cell blocks')
        call check(size(meta%cell_blocks) > 0, 'metadata has >= 1 cell block')
        call check(meta%num_cells > 0, 'metadata reports cells')
        ! vtu has a native metadata path, so this was genuinely cheap.
        call check(.not. meta%fell_back_to_full_read, 'vtu metadata is not a fallback')
    end block

    ! -- convert_cells: elevate then linearize returns the original topology --
    block
        type(mio_mesh) :: up, down
        integer(int64), allocatable :: pmap(:)
        integer :: st

        up = m%convert_cells('elevate', stat=st)
        call check(st == 0, 'convert_cells elevate succeeded')
        call check(up%num_points() > m%num_points(), 'elevate added mid-edge nodes')

        down = up%convert_cells('linearize', point_map=pmap, stat=st)
        call check(st == 0, 'convert_cells linearize succeeded')
        call check(down%num_points() == m%num_points(), 'linearize pruned the mid nodes')
        call check(allocated(pmap), 'convert_cells returned a point map')
        call check(size(pmap) == up%num_points(), 'point map covers the input points')

        call up%free()
        call down%free()

        ! An unknown mode fails through stat, never by aborting.
        up = m%convert_cells('nope', stat=st)
        call check(st /= 0, 'convert_cells rejects an unknown mode')
    end block

    ! -- subdivide: one polyhedral child per face, no per-type table needed --
    block
        type(mio_mesh) :: sub
        integer :: st

        sub = m%subdivide(record_parent_ids=.true., stat=st)
        call check(st == 0, 'subdivide succeeded')
        call check(sub%num_cell_blocks() == 1, 'subdivide kept one block')
        call check(sub%cell_block_type(1) == 'polyhedron', 'subdivide produced polyhedron cells')
        call check(sub%cell_block_is_polyhedron(1), 'subdivide block is 2-level ragged')
        ! Two tetra parents, 4 faces each -> 8 children.
        call check(sub%cell_block_num_cells(1) == 8_int64, 'subdivide made 4 children per tetra')
        call check(sub%num_points() == m%num_points() + 2_int64, &
                   'subdivide added one interior point per parent cell')

        call sub%free()
    end block

    ! -- agglomerate: merge face-adjacent cells into one polyhedron --------
    block
        type(mio_mesh) :: agg
        integer :: st

        ! The two tetra share a face (nodes 2,3,4 1-based), so target=2
        ! merges them into one polyhedron.
        agg = m%agglomerate(target_group_size=2_int64, stat=st)
        call check(st == 0, 'agglomerate succeeded')
        call check(agg%num_cell_blocks() == 1, 'agglomerate kept one block')
        call check(agg%cell_block_type(1) == 'polyhedron', 'agglomerate produced polyhedron cells')
        call check(agg%cell_block_num_cells(1) == 1_int64, 'agglomerate merged both tetra')
        call check(agg%num_points() == m%num_points(), 'agglomerate never prunes points')

        call agg%free()
    end block

    ! -- smooth: relax coordinates, leaving topology and data intact --
    block
        type(mio_mesh) :: relaxed, bad
        integer(int64) :: moved, skipped, n0, nc0
        real(real64) :: max_disp
        integer :: st

        n0 = m%num_points()
        nc0 = m%cell_block_num_cells(1)

        relaxed = m%smooth('taubin', 5, nodes_moved=moved, max_displacement=max_disp, &
                           skipped_inversion=skipped, stat=st)
        call check(st == 0, 'smooth succeeded')
        call check(relaxed%num_points() == n0, 'smooth preserves the point count')
        call check(relaxed%cell_block_num_cells(1) == nc0, 'smooth preserves the cell count')
        call check(moved >= 0_int64, 'smooth reported a node-moved count')
        call check(max_disp >= 0.0_real64, 'smooth reported a max displacement')
        call check(skipped >= 0_int64, 'smooth reported a skipped-inversion count')

        ! The optional arguments really are optional.
        call relaxed%free()
        relaxed = m%smooth('laplacian', 3, stat=st)
        call check(st == 0, 'smooth works without the optional out-args')

        ! An unknown method name fails through stat rather than aborting.
        bad = m%smooth('bogus', 1, stat=st)
        call check(st /= 0, 'smooth rejects an unknown method')

        call relaxed%free()

        ! ODT (optimal-Delaunay-triangulation smoothing): tet-only, C++-core
        ! only, moves free vertices toward incident-tet circumcenters. `m`'s
        ! own cell block is 'tetra' (set up above), so this is in scope.
        relaxed = m%smooth('odt', 1, fix_boundary=.false., preserve_features=.false., &
                           nodes_moved=moved, max_displacement=max_disp, stat=st)
        call check(st == 0, 'smooth accepts method=odt on a tet mesh')
        call check(relaxed%num_points() == n0, 'odt smooth preserves the point count')
        call check(moved >= 0_int64, 'odt smooth reported a node-moved count')
        call relaxed%free()
    end block

    ! -- optimize_volume: ODT remeshing (relocate + flip connectivity) --
    block
        type(mio_mesh) :: opt
        integer(int64) :: nflips, n23, n32, nmoved, ntets, n0
        real(real64) :: qb, qa
        integer :: st

        n0 = m%num_points()
        opt = m%optimize_volume(max_iterations=5, num_flips=nflips, num_23_flips=n23, &
                                num_32_flips=n32, num_vertices_moved=nmoved, num_tets=ntets, &
                                min_quality_before=qb, min_quality_after=qa, stat=st)
        call check(st == 0, 'optimize_volume succeeded on a tet mesh')
        call check(opt%num_points() == n0, 'optimize_volume preserves the point set')
        call check(nflips >= 0_int64, 'optimize_volume reported a flip count')
        call check(qa >= qb - 1.0e-12_real64, 'optimize_volume never lowers worst quality')
        call opt%free()
    end block

    ! -- interpolate: cross-mesh field transfer --
    block
        type(mio_mesh) :: sampled, bad
        real(real64), allocatable :: tdata(:)
        integer :: st

        ! Sampling the mesh onto itself: nearest reproduces the field exactly.
        ! 'temperature' already exists on the target, so resolve by suffix.
        sampled = mio_interpolate(m, m, method='nearest', &
                                  arrays=[character(len=16) :: 'temperature'], &
                                  on_conflict='suffix', stat=st)
        call check(st == 0, 'interpolate succeeded')
        call check(sampled%num_points() == m%num_points(), &
                   'interpolate preserves the target point count')
        call sampled%get_point_data('temperature_interp', tdata)
        call check(size(tdata) == 5, 'interpolated array has one row per target point')
        call check(maxval(abs(tdata - [1.0_real64, 2.0_real64, 3.0_real64, 4.0_real64, &
                                       5.0_real64])) < 1.0e-12_real64, &
                   'self-interpolation reproduces the field')
        call sampled%free()

        ! Every source point_data name collides under the default 'error'.
        bad = mio_interpolate(m, m, stat=st)
        call check(st /= 0, 'interpolate rejects a name collision under error')

        ! An unknown method fails through stat, never by aborting.
        bad = mio_interpolate(m, m, method='bogus', on_conflict='overwrite', stat=st)
        call check(st /= 0, 'interpolate rejects an unknown method')
    end block

    ! -- conservative_interpolate: mass-preserving cross-mesh field transfer --
    block
        type(mio_mesh) :: sampled, bad
        real(real64), allocatable :: cdata(:)
        integer :: st

        ! Self-remap of cell_data 'quality' onto the identical mesh exactly
        ! reproduces every value (100% coverage, no clipping loss).
        sampled = mio_conservative_interpolate(m, m, &
                                               arrays=[character(len=16) :: 'quality'], &
                                               on_conflict='suffix', stat=st)
        call check(st == 0, 'conservative_interpolate succeeded')
        call sampled%get_cell_data('quality_interp', 1, cdata)
        call check(size(cdata) == 2, 'conservative_interpolate has one row per target cell')
        call check(maxval(abs(cdata - [0.5_real64, 0.75_real64])) < 1.0e-9_real64, &
                   'self-remap reproduces the field exactly')
        call sampled%free()

        ! A name collision under the default 'error' fails through stat.
        bad = mio_conservative_interpolate(m, m, stat=st)
        call check(st /= 0, 'conservative_interpolate rejects a name collision under error')
    end block

    ! -- refine: uniform subdivision into same-type children --
    block
        type(mio_mesh) :: one, two, quadratic, bad
        integer(int64), allocatable :: pmap(:)
        integer(int64) :: n0
        integer :: st

        n0 = m%cell_block_num_cells(1)

        one = m%refine(1, point_map=pmap, stat=st)
        call check(st == 0, 'refine succeeded')
        call check(one%cell_block_num_cells(1) == n0*8, 'refine split each tetra into 8')
        call check(one%num_points() > m%num_points(), 'refine added mid-edge nodes')
        call check(allocated(pmap), 'refine returned a point map')
        call check(size(pmap) == m%num_points(), 'point map covers the input points')

        ! levels=2 matches applying one level twice.
        two = m%refine(2, stat=st)
        call check(st == 0, 'refine levels=2 succeeded')
        call check(two%cell_block_num_cells(1) == n0*64, &
                   'two levels split each tetra into 64')

        ! A cell type with no same-type subdivision fails through stat.
        quadratic = m%convert_cells('elevate', stat=st)
        call check(st == 0, 'built a tetra10 mesh to reject')
        bad = quadratic%refine(1, stat=st)
        call check(st /= 0, 'refine rejects higher-order cells')

        call one%free()
        call two%free()
        call quadratic%free()
    end block

    ! -- refine: selective, with a conforming closure --
    block
        type(mio_mesh) :: sel, prop, oops
        integer(int64) :: n0
        integer :: st

        n0 = m%cell_block_num_cells(1)

        ! One cell only (1-based here, shifted to the C API's 0-based inside).
        sel = m%refine(cells=[1_int64], record_levels=.true., stat=st)
        call check(st == 0, 'selective refine succeeded')
        call check(sel%cell_block_num_cells(1) > n0, 'selective refine added cells')
        call check(sel%cell_block_num_cells(1) < n0*8, &
                   'selective refine did not refine everything')

        ! Propagation is the always-works baseline: on a connected mesh it
        ! reaches every cell, which is exactly a uniform refinement.
        prop = m%refine(cells=[1_int64], closure='propagate', stat=st)
        call check(st == 0, 'propagate closure succeeded')
        call check(prop%cell_block_num_cells(1) == n0*8, &
                   'propagate reproduces the uniform refinement')

        ! Two selectors at once is an error reported through stat.
        oops = m%refine(cells=[1_int64], region='anything', stat=st)
        call check(st /= 0, 'refine rejects two selectors at once')

        call sel%free()
        call prop%free()
    end block

    ! -- refine: the persistent parent/child hierarchy --
    block
        type(mio_mesh) :: hier
        real(real64), allocatable :: entity(:, :)
        integer :: st

        ! record_hierarchy is opt-in ('quality' was attached to m earlier and
        ! is carried through regardless -- only the two new reserved names are
        ! gated on the flag).
        hier = m%refine(cells=[1_int64], stat=st)
        call check(st == 0, 'plain selective refine succeeded')
        call check(hier%cell_data_num_blocks('refine:cell_id') < 0_int64, &
                   'refine:cell_id is not recorded unless asked')
        call hier%free()

        hier = m%refine(cells=[1_int64], record_hierarchy=.true., stat=st)
        call check(st == 0, 'record_hierarchy refine succeeded')
        call check(hier%cell_data_num_blocks('refine:cell_id') == 1_int64, &
                   'refine:cell_id attached')
        call check(hier%cell_data_num_blocks('refine:parent_id') == 1_int64, &
                   'refine:parent_id attached')
        ! Both arrays are (n, 1)-shaped (one component per cell), and the
        ! rank-1 get_cell_data getter refuses anything but a true rank-1
        ! array -- a pre-existing Fortran-binding limitation unrelated to this
        ! feature (there is no rank-2 cell_data getter, unlike point_data), so
        ! presence/coverage is what this binding can check; value-level parity
        ! is covered by the C++ and Python test suites.
        !
        ! Also proves the multigrid-stencil fix: this call used the redgreen
        ! closure, which leaves no hanging nodes, so refine:entity would
        ! normally never be attached at all.
        call hier%get_point_data('refine:entity', entity)
        call check(size(entity, 1) == 4, 'refine:entity is 4 values per point')
        call hier%free()
    end block

    ! -- undo_green: restore a transitional cell to its coarse parent --
    block
        type(mio_mesh) :: fine, undone, bad
        integer(int64) :: ngroups, nremoved, fine_n, undone_n, coarse_n
        integer :: st

        ! m's two tetrahedra share a whole face; refining cell 1 forces cell
        ! 2's shared-face edges to close up as one green group.
        fine = m%refine(cells=[1_int64], record_hierarchy=.true., record_levels=.true., stat=st)
        call check(st == 0, 'refine for undo_green succeeded')
        fine_n = fine%cell_block_num_cells(1)
        coarse_n = m%cell_block_num_cells(1)

        undone = mio_undo_green(m, fine, num_groups_undone=ngroups, &
                                num_cells_removed=nremoved, stat=st)
        call check(st == 0, 'undo_green succeeded')
        call check(ngroups > 0_int64, 'undo_green undid at least one green group')
        call check(nremoved > 0_int64, 'undo_green removed at least one cell')
        undone_n = undone%cell_block_num_cells(1)
        call check(undone_n < fine_n, 'undo_green produced fewer cells than fine')
        call check(undone_n > coarse_n, 'undo_green kept the genuine (red) refinement')
        call check(undone%cell_data_num_blocks('refine:cell_id') < 0_int64, &
                   'the reserved refine:* arrays are dropped')

        ! Fails by name rather than guessing: "fine" here has no hierarchy at all.
        bad = mio_undo_green(m, m, stat=st)
        call check(st /= 0, 'undo_green rejects a mesh with no hierarchy')

        call undone%free()
        call fine%free()
    end block

    ! -- decimate: QEM edge collapse of a surface mesh --
    block
        type(mio_mesh) :: fan, coarse, bad
        integer(int64), allocatable :: pmap(:)
        integer(int64) :: nfaces, npts, nrej
        real(real64) :: err_applied
        integer :: st
        real(real64) :: fan_points(3, 5)
        integer(int64) :: fan_conn(3, 4)

        ! A 4-triangle fan around a centre vertex: the centre collapses into
        ! the pinned boundary, leaving 2 triangles.
        fan_points = reshape([0.0_real64, 0.0_real64, 0.0_real64, &
                              1.0_real64, 0.0_real64, 0.0_real64, &
                              1.0_real64, 1.0_real64, 0.0_real64, &
                              0.0_real64, 1.0_real64, 0.0_real64, &
                              0.5_real64, 0.5_real64, 0.0_real64], [3, 5])
        fan_conn = reshape([1_int64, 2_int64, 5_int64, &
                            2_int64, 3_int64, 5_int64, &
                            3_int64, 4_int64, 5_int64, &
                            4_int64, 1_int64, 5_int64], [3, 4])
        call fan%create()
        call fan%set_points(fan_points)
        call fan%add_cell_block('triangle', fan_conn)

        coarse = fan%decimate(target_faces=1_int64, faces_removed=nfaces, &
                              points_removed=npts, collapses_rejected=nrej, &
                              max_error_applied=err_applied, point_map=pmap, stat=st)
        call check(st == 0, 'decimate succeeded')
        call check(coarse%cell_block_num_cells(1) == 2_int64, &
                   'decimate collapsed the fan to 2 triangles')
        call check(nfaces == 2_int64, 'decimate reported 2 faces removed')
        call check(npts == 1_int64, 'decimate reported 1 point removed')
        call check(nrej >= 0_int64, 'decimate reported a rejection count')
        call check(err_applied >= 0.0_real64, 'decimate reported a max error')
        call check(allocated(pmap) .and. size(pmap) == 5, &
                   'decimate returned a full point map')
        call check(all(pmap >= 1_int64), 'every point maps to a live survivor')

        ! Exactly one stopping criterion is required.
        bad = fan%decimate(stat=st)
        call check(st /= 0, 'decimate rejects a missing criterion')
        ! A volume mesh is out of scope by name.
        bad = m%decimate(ratio=0.5_real64, stat=st)
        call check(st /= 0, 'decimate rejects a volume mesh')

        call coarse%free()
        call fan%free()
    end block

    ! -- remesh: ACVD surface remeshing (a brand-new mesh, no maps) --
    block
        type(mio_mesh) :: octa, out, out_q, out_a, bad
        integer(int64) :: niter, niso, nnm
        integer :: subapp, st
        real(real64) :: octa_points(3, 6)
        integer(int64) :: octa_conn(3, 8)

        ! A regular octahedron: the smallest closed 2-manifold triangle mesh
        ! with no coplanar faces.
        octa_points = reshape([1.0_real64, 0.0_real64, 0.0_real64, &
                               -1.0_real64, 0.0_real64, 0.0_real64, &
                               0.0_real64, 1.0_real64, 0.0_real64, &
                               0.0_real64, -1.0_real64, 0.0_real64, &
                               0.0_real64, 0.0_real64, 1.0_real64, &
                               0.0_real64, 0.0_real64, -1.0_real64], [3, 6])
        octa_conn = reshape([1_int64, 3_int64, 5_int64, 3_int64, 2_int64, 5_int64, &
                             2_int64, 4_int64, 5_int64, 4_int64, 1_int64, 5_int64, &
                             3_int64, 1_int64, 6_int64, 2_int64, 3_int64, 6_int64, &
                             4_int64, 2_int64, 6_int64, 1_int64, 4_int64, 6_int64], [3, 8])
        call octa%create()
        call octa%set_points(octa_points)
        call octa%add_cell_block('triangle', octa_conn)

        out = octa%remesh(30_int64, num_iterations=niter, subdivide_applied=subapp, &
                          num_isolated_clusters=niso, stat=st)
        call check(st == 0, 'remesh succeeded')
        call check(out%num_points() == 30_int64, 'remesh produced 30 clusters')
        call check(subapp > 0, 'remesh auto-subdivided (6 vertices cannot support 30 clusters)')
        call check(niter >= 0_int64, 'remesh reported an iteration count')
        call check(niso >= 0_int64, 'remesh reported an isolated-cluster count')

        ! The quadric ("feature-preserving") metric is accepted too.
        out_q = octa%remesh(30_int64, metric='quadric', stat=st)
        call check(st == 0, 'remesh accepts the quadric metric')

        ! The anisotropic metric + max_anisotropy go through mio_remesh_ex.
        out_a = octa%remesh(30_int64, metric='anisotropic', max_anisotropy=3.0_real64, &
                            num_non_manifold_vertices=nnm, stat=st)
        call check(st == 0, 'remesh accepts the anisotropic metric')
        call check(out_a%num_points() > 0_int64, 'anisotropic remesh produced clusters')
        call check(nnm >= 0_int64, 'remesh reported a non-manifold-vertex count')

        ! Too few clusters is rejected by name.
        bad = octa%remesh(3_int64, stat=st)
        call check(st /= 0, 'remesh rejects a too-small cluster count')

        call out%free()
        call out_q%free()
        call out_a%free()

        ! -- remesh_volume: isosurface stuffing, remesh's volumetric sibling --
        block
            type(mio_mesh) :: rv_out, rv_bad
            integer(int64) :: warped, rejected, nonmanifold
            type(mio_surface_quality) :: q
            integer :: rv_st

            ! Unlike remesh, remesh_volume accepts the closed surface (octa)
            ! directly and returns a tetra mesh.
            rv_out = octa%remesh_volume(cell_size=0.4_real64, watertight_check='off', &
                                        num_vertices_warped=warped, num_tets_rejected=rejected, &
                                        num_non_manifold_edges=nonmanifold, quality=q, stat=rv_st)
            call check(rv_st == 0, 'remesh_volume succeeded')
            call check(rv_out%num_points() > 0_int64, 'remesh_volume produced points')
            call check(rv_out%num_cell_blocks() == 1, 'remesh_volume produced one cell block')
            call check(rv_out%cell_block_type(1) == 'tetra', 'remesh_volume produced tets')
            call check(warped >= 0_int64, 'remesh_volume reported a warped-vertex count')
            call check(rejected >= 0_int64, 'remesh_volume reported a rejected-tet count')
            call check(nonmanifold >= 0_int64, 'remesh_volume reported a non-manifold-edge count')
            call check(q%watertight /= 0, 'remesh_volume reported the input as watertight')

            ! Neither resolution nor cell_size given is rejected by name.
            rv_bad = octa%remesh_volume(stat=rv_st)
            call check(rv_st /= 0, 'remesh_volume rejects a missing resolution/cell_size')

            call rv_out%free()
        end block

        call octa%free()
    end block

    ! -- partition: N balanced pieces + flat labels --
    block
        type(mio_mesh), allocatable :: pieces(:)
        integer(int64), allocatable :: labels(:)
        integer(int64) :: total_in, total_out, i
        integer :: st, p

        total_in = 0
        do p = 1, int(m%num_cell_blocks())
            total_in = total_in + m%cell_block_num_cells(p)
        end do

        pieces = m%partition(2, method='sfc', stat=st)
        call check(st == 0, 'partition succeeded')
        call check(size(pieces) == 2, 'partition returned exactly nparts pieces')
        total_out = 0
        do p = 1, size(pieces)
            call check(pieces(p)%num_cell_blocks() == m%num_cell_blocks(), &
                       'piece keeps the input block structure')
            total_out = total_out + pieces(p)%cell_block_num_cells(1)
        end do
        call check(total_out == total_in, 'pieces cover every cell exactly once')

        labels = m%partition_labels(2, method='sfc', stat=st)
        call check(st == 0, 'partition_labels succeeded')
        call check(size(labels, kind=int64) == total_in, 'labels cover every cell')
        call check(all(labels >= 0) .and. all(labels < 2), 'labels lie in [0, nparts)')

        do p = 1, size(pieces)
            call pieces(p)%free()
        end do
        deallocate (pieces)

        ! A NEGATIVE ghost_layers must fail through stat, not abort.
        pieces = m%partition(2, ghost_layers=-1, stat=st)
        call check(st /= 0, 'partition rejects a negative ghost_layers')
        do i = 1, size(pieces)
            call pieces(i)%free()
        end do

        ! A positive one is a supported request since v9.0.0: it grows each
        ! piece by a shared-node halo, so no piece can shrink.
        pieces = m%partition(2, ghost_layers=1, stat=st)
        call check(st == 0, 'partition accepts a positive ghost_layers')
        call check(size(pieces) == 2, 'ghosted partition still returns nparts pieces')
        do i = 1, size(pieces)
            call check(pieces(i)%num_points() > 0, 'ghosted piece is non-empty')
            call pieces(i)%free()
        end do
    end block

    ! ---- isosurface (the level set of a point_data field) ----------------
    block
        type(mio_mesh) :: iso
        integer :: st

        ! temperature runs 1..5 over the 5 nodes, so the 3.0 level set cuts
        ! both tetrahedra and yields a non-empty contour, tagged per cell.
        iso = m%isosurface('temperature', [3.0_real64], record_parent_ids=.true., stat=st)
        call check(st == 0, 'isosurface succeeded')
        call check(iso%num_cell_blocks() >= 1_int64, 'isosurface produces contour cells')
        call check(iso%num_points() > 0_int64, 'isosurface produces contour points')
        call check(iso%cell_data_num_blocks('iso:value') >= 1_int64, 'iso:value tag attached')
        call check(iso%cell_data_num_blocks('iso:index') >= 1_int64, 'iso:index tag attached')
        call check(iso%cell_data_num_blocks('iso:parent_cell') >= 1_int64, &
                   'iso:parent_cell attached')
        call iso%free()

        ! An isovalue outside the field's range is an empty contour, not a failure.
        iso = m%isosurface('temperature', [99.0_real64], stat=st)
        call check(st == 0, 'an out-of-range isovalue is not an error')
        call check(iso%num_cell_blocks() == 0_int64, 'an out-of-range isovalue is empty')
        call iso%free()

        ! A cell_data name has no level set: it must fail through stat, not abort.
        iso = m%isosurface('quality', [0.5_real64], stat=st)
        call check(st /= 0, 'isosurface rejects a cell_data field')
    end block

    ! ---- gradient (field differential operators) --------------------------
    block
        type(mio_mesh) :: g
        integer(int64) :: nskip, nfall
        integer :: st

        g = m%gradient('temperature', stat=st, num_skipped=nskip, num_fallback=nfall)
        call check(st == 0, 'gradient succeeded')
        call check(g%cell_data_num_blocks('temperature:gradient') >= 1_int64, &
                   'gradient attaches temperature:gradient')
        call check(nfall == 0_int64, 'green-gauss never falls back')
        call g%free()

        ! Least squares on this small mesh may fall back per cell, which is a
        ! reported outcome rather than a failure.
        g = m%gradient('temperature', method='least-squares', stat=st, num_fallback=nfall)
        call check(st == 0, 'least-squares gradient succeeded')
        call check(nfall >= 0_int64, 'the fallback counter is reported')
        call g%free()

        ! The point location moves the result into point_data.
        g = m%gradient('temperature', location='point', output='dT', stat=st)
        call check(st == 0, 'point-located gradient succeeded')
        call check(g%num_point_data() > m%num_point_data(), &
                   'the point-located result adds a point_data array')
        call g%free()

        ! A cell_data field is piecewise constant and has no derivative: it must
        ! fail through stat, never abort.
        g = m%gradient('quality', stat=st)
        call check(st /= 0, 'gradient rejects a cell_data field')
        ! A scalar has no divergence.
        g = m%gradient('temperature', op='divergence', stat=st)
        call check(st /= 0, 'gradient rejects a scalar divergence')
    end block

    ! ---- hessian (second derivative, gradient's companion) ----------------
    block
        type(mio_mesh) :: h
        integer(int64) :: nskip, nfall
        integer :: st

        h = m%hessian('temperature', stat=st, num_skipped=nskip, num_fallback=nfall)
        call check(st == 0, 'hessian succeeded')
        call check(h%cell_data_num_blocks('temperature:hessian') >= 1_int64, &
                   'hessian attaches temperature:hessian')
        call check(nskip == 0_int64, 'no cell skipped on this mesh')
        call h%free()

        ! The point location moves the result into point_data.
        h = m%hessian('temperature', location='point', output='H2', stat=st)
        call check(st == 0, 'point-located hessian succeeded')
        call check(h%num_point_data() > m%num_point_data(), &
                   'the point-located result adds a point_data array')
        call h%free()

        ! A cell_data field is piecewise constant and has no derivative.
        h = m%hessian('quality', stat=st)
        call check(st /= 0, 'hessian rejects a cell_data field')
        ! Hessian is scalar-only: a multi-component field is rejected by name.
        h = m%hessian('velocity', stat=st)
        call check(st /= 0, 'hessian rejects a multi-component field')
    end block

    ! ---- estimate_error (ZZ recovery-based error indicator) ---------------
    block
        type(mio_mesh) :: e
        real(real64) :: gerr
        integer(int64) :: nskip, nmark
        integer :: st

        e = m%estimate_error('temperature', stat=st, global_error=gerr, num_skipped=nskip)
        call check(st == 0, 'estimate_error succeeded')
        call check(e%cell_data_num_blocks('error:zz') >= 1_int64, &
                   'estimate_error attaches error:zz')
        call check(e%cell_data_num_blocks('error:marked') <= 0_int64, &
                   'no marking requested by default')
        call check(gerr >= 0.0_real64, 'global_error is reported')
        call e%free()

        e = m%estimate_error('temperature', marking='absolute', marking_value=1.0d-9, &
                             output='ind', marked='flag', stat=st, num_marked=nmark)
        call check(st == 0, 'estimate_error with marking succeeded')
        call check(e%cell_data_num_blocks('ind') >= 1_int64, 'custom indicator name honoured')
        call check(e%cell_data_num_blocks('flag') >= 1_int64, 'custom marked name honoured')
        call check(nmark >= 0_int64, 'num_marked is reported')
        call e%free()

        ! A cell_data field has no derivative to recover: fails through stat.
        e = m%estimate_error('quality', stat=st)
        call check(st /= 0, 'estimate_error rejects a cell_data field')
        ! An out-of-range marking_value for "fraction" fails through stat too.
        e = m%estimate_error('temperature', marking='fraction', marking_value=1.5d0, stat=st)
        call check(st /= 0, 'estimate_error rejects an out-of-range marking_value')
    end block

    ! --- named regions (doc/regions.md) ---------------------------------------
    ! The first named groups this API can carry at all; before meshio++ 8.1 they
    ! never left the Python layer.
    block
        type(mio_region_info), allocatable :: regs(:)
        character(len=STRBUF_LEN), allocatable :: rkeys(:)
        integer(int64), allocatable :: rentries(:)
        integer :: st

        call m%add_region('fixed', MIO_REGION_POINT, [1_int64, 4_int64], stat=st)
        call check(st == 0, 'add_region point')
        call m%add_region('solid', MIO_REGION_CELL, [1_int64], dim=3, tag=42_int64, stat=st)
        call check(st == 0, 'add_region cell')

        regs = m%regions(keys=rkeys, entries=rentries, stat=st)
        call check(st == 0, 'regions status')
        call check(size(regs) == 2, 'two regions')
        ! Order is (kind, name, dim, tag): point before cell.
        call check(trim(rkeys(1)) == 'fixed', 'first region name')
        call check(regs(1)%kind == MIO_REGION_POINT, 'first region kind')
        call check(regs(1)%stride == 1, 'point region stride')
        call check(regs(2)%tag == 42_int64, 'cell region tag survives')
        call check(regs(2)%dim == 3, 'cell region dim survives')
        ! Point/cell indices come back 1-based, as everywhere else in this API.
        call check(size(rentries) == 3, 'entry buffer length')
        call check(rentries(1) == 1_int64 .and. rentries(2) == 4_int64, 'point entries 1-based')
    end block

    ! field integration's per-region breakdown, now that `m` carries the
    ! 'solid' Cell region added just above.
    block
        type(mio_field_integral_info), allocatable :: fir(:)
        character(len=STRBUF_LEN), allocatable :: firkeys(:)
        real(real64), allocatable :: firtotals(:)
        integer :: st

        fir = m%data_integrate_region('quality', keys=firkeys, totals=firtotals, stat=st)
        call check(st == 0, 'data_integrate_region succeeds')
        call check(size(fir) == 1, 'one named Cell region reported')
        call check(trim(firkeys(1)) == 'solid', 'data_integrate_region names the region')
        call check(fir(1)%num_cells == 1_int64, 'the region has one cell')
    end block

    ! ---- transient (time-series) XDMF writing ---------------------------
    ! The one writer m%write() cannot express: the grid goes out once and each
    ! solve appends a cheap step. "XML" keeps the whole series in the single
    ! .xdmf file, so this runs in a build without HDF5 too.
    block
        type(mio_xdmf_series) :: series
        type(mio_mesh) :: step, back
        type(mio_metadata) :: meta
        real(real64) :: t, values(5)
        real(real64), allocatable :: got(:)
        character(:), allocatable :: series_path
        integer :: st, k

        series_path = prefix//'_series.xdmf'

        call series%create(series_path, data_format='XML', stat=st)
        call check(st == 0, 'xdmf series create')
        call check(series%is_valid(), 'xdmf series handle is valid')
        call check(series%num_steps() == 0_int64, 'a fresh series has no steps')

        call series%write_points_cells(m, stat=st)
        call check(st == 0, 'xdmf series write_points_cells')

        ! Three steps at t = 0.0, 0.5, 1.0, whose 'temperature' is t + node id
        ! so a step mix-up cannot pass: every step's values differ.
        call step%create()
        call step%set_points(points)
        call step%add_cell_block('tetra', conn)
        do k = 0, 2
            t = 0.5_real64*real(k, real64)
            do i = 1, 5
                values(i) = t + real(i, real64)
            end do
            call step%add_point_data('temperature', values, stat=st)
            call check(st == 0, 'step point_data set')
            call series%write_data(t, step, stat=st)
            call check(st == 0, 'xdmf series write_data')
        end do
        call check(series%num_steps() == 3_int64, 'three steps written')

        ! The .xdmf light data is buffered until finalize, so nothing is
        ! readable before this call.
        call series%finalize(stat=st)
        call check(st == 0, 'xdmf series finalize')
        call series%finalize(stat=st)
        call check(st == 0, 'xdmf series finalize is idempotent')
        call series%free()
        call check(.not. series%is_valid(), 'series handle invalid after free')
        call step%free()

        ! Read the series back through the ordinary reader: metadata reports
        ! every step's <Time Value> without touching a payload, and time_step
        ! selects which step's attributes are materialized.
        meta = mio_read_metadata(series_path, stat=st)
        call check(st == 0, 'series metadata read')
        call check(allocated(meta%time_values), 'metadata carries time values')
        if (allocated(meta%time_values)) then
            call check(size(meta%time_values) == 3, 'three recorded time values')
            if (size(meta%time_values) == 3) then
                write (*, '(a,3(1x,f6.3))') 'fortran xdmf series time values:', &
                    meta%time_values(1), meta%time_values(2), meta%time_values(3)
                call check(abs(meta%time_values(1) - 0.0_real64) < 1.0e-12_real64, &
                           'time value 0')
                call check(abs(meta%time_values(2) - 0.5_real64) < 1.0e-12_real64, &
                           'time value 1')
                call check(abs(meta%time_values(3) - 1.0_real64) < 1.0e-12_real64, &
                           'time value 2')
            end if
        end if

        do k = 0, 2
            call back%read(series_path, time_step=k, stat=st)
            call check(st == 0, 'read series step')
            if (st /= 0) cycle
            call check(back%num_points() == 5_int64, 'series step has the shared grid')
            call back%get_point_data('temperature', got, stat=st)
            call check(st == 0, 'series step point_data')
            if (st /= 0) cycle
            write (*, '(a,i0,a,5(1x,f6.3))') 'fortran xdmf series step ', k, &
                ' temperature:', got(1), got(2), got(3), got(4), got(5)
            do i = 1, 5
                call check(abs(got(i) - (0.5_real64*real(k, real64) + real(i, real64))) &
                           < 1.0e-9_real64, 'series step values')
            end do
        end do
        call back%free()

        ! An unknown data format fails through stat rather than aborting.
        call series%create(prefix//'_bad.xdmf', data_format='NoSuchFormat', stat=st)
        call check(st /= 0, 'xdmf series rejects an unknown data format')
        call check(.not. series%is_valid(), 'a failed create leaves no handle')
    end block

    ! ------------------------------------------------------------------
    ! Settings pipeline (v9.11.0): behaviour follows the build -- with the
    ! JSON parser a bad document fails naming the offender; without it every
    ! entry point fails naming -DMESHIOPLUSPLUS_WITH_JSON=ON.
    ! ------------------------------------------------------------------
    block
        character(:), allocatable :: msg
        integer :: st
        call mio_pipeline_run_json('{"Input": {"Path": "a"}, "Output": {"Path": "b"}, '// &
                                   '"Operations": [{"Op": "Nope"}]}', stat=st, errmsg=msg)
        call check(st /= 0, 'pipeline rejects a bad document through stat')
        if (mio_pipeline_has_json()) then
            call check(index(msg, 'Nope') > 0, 'pipeline schema error names the op')
        else
            call check(index(msg, 'MESHIOPLUSPLUS_WITH_JSON') > 0, &
                       'compiled-out pipeline names the flag')
        end if
    end block

    call m%free()
    call r%free()
    call c%free()
    call check(.not. m%is_valid(), 'handle invalid after free')

    ! ------------------------------------------------------------------
    ! Sequences (multi-file / transient datasets). The three files below are
    ! deliberately named _1/_2/_10 so the natural-numeric ordering rule is
    ! exercised: a lexicographic sort would put _10 second.
    ! ------------------------------------------------------------------
    call m%create()
    call m%set_points(points)
    call m%add_cell_block('tetra', conn)
    call m%write(prefix//'_seq_1.vtu')
    call m%write(prefix//'_seq_2.vtu')
    call m%write(prefix//'_seq_10.vtu')
    call m%free()

    call seq%open(prefix//'_seq_*.vtu', stat=ierr)
    call check(ierr == 0, 'sequence open')
    call check(seq%count() == 3_int64, 'sequence count')
    call check(index(seq%path(2), '_seq_2.vtu') > 0, 'natural order puts _2 before _10')
    call check(index(seq%path(3), '_seq_10.vtu') > 0, 'natural order puts _10 last')
    call check(seq%time(3) == 10.0_real64, 'time parsed from the filename')
    call check(seq%time_source(3) == 2, 'time source is filename')
    call check(seq%step(1) == 0_int64, 'single-step file has step 0')

    ! read_step hands back an OWNED mesh: freeing it must be right, and the
    ! sequence must stay usable (it caches nothing).
    seq_step = seq%read_step(1, stat=ierr)
    call check(ierr == 0, 'sequence read_step')
    call check(seq_step%num_points() == 5_int64, 'read step has the right points')
    call seq_step%free()
    call check(seq%count() == 3_int64, 'sequence usable after a step was freed')

    call seq%to_timeseries(prefix//'_seq_series.xdmf', stat=ierr)
    call check(ierr == 0, 'sequence fan-in')

    ! A format that cannot hold a series must fail by name, not truncate.
    call seq%to_timeseries(prefix//'_seq_bad.vtu', stat=ierr, errmsg=msg)
    call check(ierr /= 0, 'fan-in to a non-series format fails')
    call check(index(msg, '{step}') > 0, 'and the message names the remedy')
    call seq%free()
    call check(.not. seq%is_valid(), 'sequence handle invalid after free')

    ! NOTE the fan-out stem: it must NOT match the '_seq_*.vtu' input pattern
    ! above, or a second run in the same build directory would glob its own
    ! previous output back in and the count check would fail.
    call mio_timeseries_to_sequence(prefix//'_seq_series.xdmf', &
                                    prefix//'_fanout_{step}.vtu', stat=ierr)
    call check(ierr == 0, 'sequence fan-out')
    call r%read(prefix//'_fanout_0002.vtu', stat=ierr)
    call check(ierr == 0, 'fan-out wrote the third step')
    call r%free()

    ! ---- ragged (polygon / polyhedron) connectivity ---------------------
    call check_ragged()

    ! ---- regular grids and signed distance -------------------------------
    call check_grids_and_distance()

    ! ---- per-vertex curvature --------------------------------------------
    call check_curvature()
    call check_repair_shrinkwrap_sobolev()

    if (fails /= 0) then
        write (error_unit, '(a,i0,a)') 'test_fortran_api: ', fails, ' check(s) FAILED'
        error stop 1
    end if
    write (*, '(a)') 'test_fortran_api: all checks passed'

contains

    !> Build a jagged polygon block and a polyhedron block through the module,
    !> read both back, and assert the documented 1-based slice identity. Before
    !> meshio++ 9.15 the C ABI could do neither.
    subroutine check_ragged()
        type(mio_mesh) :: pm
        real(real64) :: pts(3, 5)
        integer(int64) :: row_off(3), poly_nodes(7)
        integer(int64) :: cell_off(3), face_off(8), face_nodes(21)
        integer(int64), allocatable :: g_row(:), g_nodes(:), g_cell(:), g_face(:)
        integer :: ierr2, j

        pts = reshape([0.0_real64, 0.0_real64, 0.0_real64, &
                       1.0_real64, 0.0_real64, 0.0_real64, &
                       1.0_real64, 1.0_real64, 0.0_real64, &
                       0.0_real64, 1.0_real64, 0.0_real64, &
                       2.0_real64, 0.5_real64, 0.0_real64], [3, 5])
        call pm%create()
        call pm%set_points(pts)

        ! A quad then a triangle: 1-based offsets, so cell c spans
        ! nodes(row_off(c) : row_off(c + 1) - 1).
        row_off = [1_int64, 5_int64, 8_int64]
        poly_nodes = [1_int64, 2_int64, 3_int64, 4_int64, 2_int64, 5_int64, 3_int64]
        call pm%add_polygon_block('polygon', row_off, poly_nodes, stat=ierr2)
        call check(ierr2 == 0, 'add_polygon_block succeeds')

        ! A 4-face tetrahedron then a 3-face sliver -- deliberately different
        ! face counts, so cell_offsets carries real information.
        cell_off = [1_int64, 5_int64, 8_int64]
        face_off = [1_int64, 4_int64, 7_int64, 10_int64, 13_int64, 16_int64, 19_int64, 22_int64]
        face_nodes = [1_int64, 2_int64, 3_int64, 1_int64, 4_int64, 2_int64, &
                      2_int64, 4_int64, 3_int64, 3_int64, 4_int64, 1_int64, &
                      2_int64, 3_int64, 5_int64, 3_int64, 4_int64, 5_int64, &
                      4_int64, 2_int64, 5_int64]
        call pm%add_polyhedron_block('polyhedron', cell_off, face_off, face_nodes, stat=ierr2)
        call check(ierr2 == 0, 'add_polyhedron_block succeeds')

        call check(pm%num_cell_blocks() == 2_int64, 'ragged mesh has two blocks')
        call check(pm%cell_block_is_ragged(1), 'polygon block reports ragged')
        call check(.not. pm%cell_block_is_polyhedron(1), 'polygon block is not 2-level')
        call check(pm%cell_block_is_polyhedron(2), 'polyhedron block is 2-level')
        call check(pm%cell_block_num_cells(1) == 2_int64, 'polygon block cell count')
        call check(pm%cell_block_nodes_per_cell(1) == 0_int64, 'ragged nodes_per_cell is 0')

        call pm%get_polygon_block(1, g_row, g_nodes, stat=ierr2)
        call check(ierr2 == 0, 'get_polygon_block succeeds')
        call check(size(g_row) == 3, 'polygon row_offsets has num_cells + 1 entries')
        call check(all(g_row == row_off), 'polygon row_offsets round-trip (1-based)')
        call check(all(g_nodes == poly_nodes), 'polygon nodes round-trip (1-based)')
        ! The documented slice identity, checked rather than described.
        call check(size(g_nodes(g_row(1):g_row(2) - 1)) == 4, 'polygon cell 1 has 4 nodes')
        call check(all(g_nodes(g_row(2):g_row(3) - 1) == [2_int64, 5_int64, 3_int64]), &
                   'polygon cell 2 slices to its own node list')

        call pm%get_polyhedron_block(2, g_cell, g_face, g_nodes, stat=ierr2)
        call check(ierr2 == 0, 'get_polyhedron_block succeeds')
        call check(all(g_cell == cell_off), 'polyhedron cell_offsets round-trip')
        call check(all(g_face == face_off), 'polyhedron face_offsets round-trip')
        call check(all(g_nodes == face_nodes), 'polyhedron nodes round-trip')
        ! Face 1 of cell 2 -- j = cell_offsets(c) + f - 1.
        j = int(g_cell(2))
        call check(all(g_nodes(g_face(j):g_face(j + 1) - 1) == [2_int64, 3_int64, 5_int64]), &
                   'polyhedron cell 2 face 1 slices to its own node list')

        ! Each accessor refuses the other kind by name rather than returning
        ! something wrong.
        call pm%get_polyhedron_block(1, g_cell, g_face, g_nodes, stat=ierr2)
        call check(ierr2 /= 0, 'get_polyhedron_block rejects a 1-level block')
        call pm%get_polygon_block(2, g_row, g_nodes, stat=ierr2)
        call check(ierr2 /= 0, 'get_polygon_block rejects a 2-level block')

        call pm%free()
    end subroutine

    !> grid / voxelize / watertight / sample_distance / distance_to_surface.
    !> The cube's exact SDF is known in closed form, so the distances are
    !> asserted against it rather than against another run of the same code.
    subroutine check_grids_and_distance()
        type(mio_mesh) :: g, cube, vox, field, sdf, kept
        type(mio_surface_quality) :: q
        real(real64) :: pts(3, 8), query(3, 3), d(3)
        real(real64) :: origin(3), spacing(3)
        integer(int64) :: tri(3, 12), dims(3), occupied, depth
        integer :: ierr

        ! A lattice from nothing: 2 x 2 x 2 cells means 27 points.
        g = mio_grid([2, 2, 2], stat=ierr)
        call check(ierr == 0, 'grid built')
        call check(g%num_points() == 27_int64, 'grid has 27 points')
        call check(g%num_cell_blocks() == 1_int64, 'grid has one block')
        call g%free()

        ! An empty lattice is a legal request, not an error.
        g = mio_grid([0, 0, 0], stat=ierr)
        call check(ierr == 0, 'an empty grid is not an error')
        call check(g%num_points() == 0_int64, 'an empty grid has no points')
        call g%free()

        ! The unit cube as a closed, outward-wound triangle surface.
        pts = reshape([0d0, 0d0, 0d0, 1d0, 0d0, 0d0, 1d0, 1d0, 0d0, 0d0, 1d0, 0d0, &
                       0d0, 0d0, 1d0, 1d0, 0d0, 1d0, 1d0, 1d0, 1d0, 0d0, 1d0, 1d0], [3, 8])
        tri = reshape([1_int64, 3_int64, 2_int64, 1_int64, 4_int64, 3_int64, &
                       5_int64, 6_int64, 7_int64, 5_int64, 7_int64, 8_int64, &
                       1_int64, 2_int64, 6_int64, 1_int64, 6_int64, 5_int64, &
                       2_int64, 3_int64, 7_int64, 2_int64, 7_int64, 6_int64, &
                       3_int64, 4_int64, 8_int64, 3_int64, 8_int64, 7_int64, &
                       4_int64, 1_int64, 5_int64, 4_int64, 5_int64, 8_int64], [3, 12])
        call cube%create()
        call cube%set_points(pts, stat=ierr)
        call check(ierr == 0, 'cube points set')
        call cube%add_cell_block('triangle', tri, stat=ierr)
        call check(ierr == 0, 'cube triangles set')

        q = cube%watertight_check(stat=ierr)
        call check(ierr == 0, 'watertight check ran')
        call check(q%watertight /= 0, 'the cube is watertight')
        call check(q%boundary_edges == 0_int64, 'the cube has no boundary edges')

        vox = cube%voxelize(resolution=[4, 4, 4], dims=dims, origin=origin, &
                            spacing=spacing, num_occupied=occupied, stat=ierr)
        call check(ierr == 0, 'voxelize ran')
        call check(occupied == 64_int64, 'voxelize all kept 64 cells')
        call check(dims(1) == 4_int64, 'voxelize reported the cell counts')
        call check(abs(spacing(1) - 0.25d0) < 1d-12, 'voxelize reported the cell size')
        call check(abs(origin(1)) < 1d-12, 'voxelize reported the origin')
        call vox%free()

        vox = cube%voxelize(resolution=[5, 5, 5], &
                            bounds=[-0.5d0, -0.5d0, -0.5d0, 1.5d0, 1.5d0, 1.5d0], &
                            fill='inside', watertight_check='off', &
                            num_occupied=occupied, stat=ierr)
        call check(ierr == 0, 'voxelize inside ran')
        call check(occupied == 27_int64, 'voxelize inside kept the 27 interior cells')
        call vox%free()

        ! An unknown fill fails by name rather than silently defaulting.
        vox = cube%voxelize(resolution=[2, 2, 2], fill='solid', stat=ierr)
        call check(ierr /= 0, 'an unknown fill is refused')

        ! The centre is 0.5 inside; the other two are 1.0 outside.
        query = reshape([0.5d0, 0.5d0, 0.5d0, 2d0, 0.5d0, 0.5d0, -1d0, 0.5d0, 0.5d0], [3, 3])
        d = cube%sample_distance(query, watertight_check='off', stat=ierr)
        call check(ierr == 0, 'sample_distance ran')
        call check(abs(d(1) + 0.5d0) < 1d-12, 'the cube centre is 0.5 inside')
        call check(abs(d(2) - 1d0) < 1d-12, 'a point 1 unit out reads +1')
        call check(abs(d(3) - 1d0) < 1d-12, 'the other side reads +1 too')

        g = mio_grid([2, 2, 2], origin=[-0.5d0, -0.5d0, -0.5d0], &
                     spacing=[1d0, 1d0, 1d0], stat=ierr)
        call check(ierr == 0, 'query grid built')
        field = g%distance_to_surface(cube, watertight_check='off', stat=ierr)
        call check(ierr == 0, 'distance_to_surface ran')
        call check(field%num_point_data() == 1_int64, 'sdf:distance was attached')
        call field%free()
        call g%free()

        ! compute_sdf: the grid and the field in one call.
        sdf = cube%compute_sdf(resolution=[4, 4, 4], watertight_check='off', &
                               dims=dims, origin=origin, spacing=spacing, depth=depth, &
                               stat=ierr)
        call check(ierr == 0, 'compute_sdf voxel ran')
        call check(dims(1) == 4_int64, 'compute_sdf reported the root cell counts')
        call check(depth == 0_int64, 'a voxel grid has no octree depth')
        call check(sdf%num_point_data() == 1_int64, 'sdf:distance was attached')
        call check(sdf%cell_block_num_cells(1) == 64_int64, 'compute_sdf built 4^3 cells')
        call sdf%free()

        ! The octree refines only near the surface: more than the root, far less
        ! than the uniform grid of the same finest resolution.
        sdf = cube%compute_sdf(structure='octree', root_resolution=4_int64, &
                               max_depth=2_int64, watertight_check='off', &
                               depth=depth, stat=ierr)
        call check(ierr == 0, 'compute_sdf octree ran')
        call check(depth == 2_int64, 'the octree ran two passes')
        call check(sdf%cell_block_num_cells(1) > 64_int64, 'the octree refined')
        call check(sdf%cell_block_num_cells(1) < 4096_int64, 'the octree is not uniform')
        call sdf%free()

        ! resolution/cell_size size a voxel grid; an octree's finest cell is
        ! already determined, so passing one is an error, not a preference.
        sdf = cube%compute_sdf(structure='octree', resolution=[4, 4, 4], stat=ierr)
        call check(ierr /= 0, 'octree refuses a voxel sizing')

        ! The predicate crop, composed with the distance field: the inside.
        g = mio_grid([4, 4, 4], origin=[-0.5d0, -0.5d0, -0.5d0], &
                     spacing=[0.5d0, 0.5d0, 0.5d0], stat=ierr)
        field = g%distance_to_surface(cube, location='center', watertight_check='off', &
                                      stat=ierr)
        call check(ierr == 0, 'cell-centred distance ran')
        kept = field%crop_predicate('sdf:distance', compare='<', value=0.0d0, stat=ierr)
        call check(ierr == 0, 'crop_predicate ran')
        call check(kept%cell_block_num_cells(1) == 8_int64, &
                   'the eight cells inside the unit cube were kept')
        call kept%free()
        ! An unknown array fails by name rather than keeping everything.
        kept = field%crop_predicate('nope', stat=ierr)
        call check(ierr /= 0, 'crop_predicate refuses an unknown array')
        call field%free()
        call g%free()
        call cube%free()
    end subroutine

    subroutine check_curvature()
        type(mio_mesh) :: oct, curv, bad
        integer(int64) :: nb, ni, nd, incons
        real(real64) :: defect
        logical :: wt
        integer :: ierr
        real(real64) :: oct_points(3, 6)
        integer(int64) :: oct_conn(3, 8)

        ! A regular octahedron: the smallest closed, consistently wound
        ! triangle surface, so Gauss-Bonnet applies exactly.
        oct_points = reshape([ 1.0_real64,  0.0_real64,  0.0_real64, &
                              -1.0_real64,  0.0_real64,  0.0_real64, &
                               0.0_real64,  1.0_real64,  0.0_real64, &
                               0.0_real64, -1.0_real64,  0.0_real64, &
                               0.0_real64,  0.0_real64,  1.0_real64, &
                               0.0_real64,  0.0_real64, -1.0_real64], [3, 6])
        oct_conn = reshape([1_int64, 3_int64, 5_int64, &
                            3_int64, 2_int64, 5_int64, &
                            2_int64, 4_int64, 5_int64, &
                            4_int64, 1_int64, 5_int64, &
                            3_int64, 1_int64, 6_int64, &
                            2_int64, 3_int64, 6_int64, &
                            4_int64, 2_int64, 6_int64, &
                            1_int64, 4_int64, 6_int64], [3, 8])
        call oct%create()
        call oct%set_points(oct_points)
        call oct%add_cell_block('triangle', oct_conn)

        curv = oct%curvature(record_principal=.true., num_boundary=nb, &
                             num_isolated=ni, num_degenerate=nd, &
                             total_angle_defect=defect, inconsistent_pairs=incons, &
                             watertight=wt, stat=ierr)
        call check(ierr == 0, 'curvature succeeded')
        ! The tessellation-independent oracle: on a closed surface the angle
        ! defects sum to 2*pi*chi, which is 4*pi for anything sphere-like.
        call check(abs(defect - 4.0_real64*3.141592653589793_real64) < 1.0e-12_real64, &
                   'curvature satisfies Gauss-Bonnet on a closed surface')
        call check(nb == 0_int64, 'a closed surface has no boundary vertices')
        call check(ni == 0_int64, 'no isolated vertices')
        call check(nd == 0_int64, 'no degenerate triangles')
        call check(incons == 0_int64, 'the octahedron is consistently wound')
        call check(wt, 'the octahedron is watertight')
        call check(curv%num_points() == 6_int64, 'curvature is a pure data step')
        call check(curv%num_point_data() >= 3_int64, 'curvature attached its arrays')
        call curv%free()

        ! An unknown dual area is refused by name rather than silently defaulted.
        bad = oct%curvature(dual_area='nope', stat=ierr)
        call check(ierr /= 0, 'curvature rejects an unknown dual area')
        call oct%free()
    end subroutine

    subroutine check_repair_shrinkwrap_sobolev()
        type(mio_mesh) :: oct, fixed, holed, filled, target, cloud, wrapped, grid, moved
        integer(int64) :: nflip, nholes, nfaces, npts, nproj, nskip, nit, nfixed
        integer(int64) :: incons
        logical :: wt, conv
        integer :: ierr, i, j
        real(real64) :: oct_points(3, 6), cloud_points(3, 3), grid_points(3, 16)
        real(real64) :: disp(3, 16), pin(16), pick(3), maxd
        real(real64), allocatable :: out_points(:, :)
        integer(int64) :: oct_conn(3, 8), seven(3, 7), vconn(1, 3), gconn(3, 18)

        oct_points = reshape([ 1.0_real64,  0.0_real64,  0.0_real64, &
                              -1.0_real64,  0.0_real64,  0.0_real64, &
                               0.0_real64,  1.0_real64,  0.0_real64, &
                               0.0_real64, -1.0_real64,  0.0_real64, &
                               0.0_real64,  0.0_real64,  1.0_real64, &
                               0.0_real64,  0.0_real64, -1.0_real64], [3, 6])
        oct_conn = reshape([1_int64, 3_int64, 5_int64, &
                            3_int64, 2_int64, 5_int64, &
                            2_int64, 4_int64, 5_int64, &
                            4_int64, 1_int64, 5_int64, &
                            3_int64, 1_int64, 6_int64, &
                            2_int64, 3_int64, 6_int64, &
                            4_int64, 2_int64, 6_int64, &
                            1_int64, 4_int64, 6_int64], [3, 8])
        ! Flip facet 0 (swap its second and third corners).
        oct_conn(2, 1) = 5_int64
        oct_conn(3, 1) = 3_int64
        call oct%create()
        call oct%set_points(oct_points)
        call oct%add_cell_block('triangle', oct_conn)
        fixed = oct%repair(num_flipped=nflip, inconsistent_pairs_after=incons, &
                           watertight_after=wt, stat=ierr)
        call check(ierr == 0, 'repair succeeded')
        call check(nflip == 1_int64, 'repair rewound exactly the flipped facet')
        call check(incons == 0_int64, 'repair left no inconsistent pair')
        call check(wt, 'repair output is watertight')
        call check(fixed%num_points() == 6_int64, 'repair added no point to a closed surface')
        call fixed%free()
        call oct%free()

        ! Drop one facet: the hole is filled by three triangles and one point.
        seven = oct_conn(:, 2:8)
        call holed%create()
        call holed%set_points(oct_points)
        call holed%add_cell_block('triangle', seven)
        filled = holed%repair(num_holes_filled=nholes, num_faces_added=nfaces, &
                              num_points_added=npts, watertight_after=wt, stat=ierr)
        call check(ierr == 0, 'repair of a holed surface succeeded')
        call check(nholes == 1_int64 .and. nfaces == 3_int64 .and. npts == 1_int64, &
                   'repair filled the hole with a centroid fan')
        call check(wt, 'the filled surface is watertight')
        call check(filled%num_cell_blocks() == 2_int64, 'fill triangles land in a trailing block')
        call filled%free()
        ! fill_holes off leaves it open.
        filled = holed%repair(fill_holes=.false., num_holes_filled=nholes, stat=ierr)
        call check(ierr == 0 .and. nholes == 0_int64, 'repair honours fill_holes=.false.')
        call filled%free()
        call holed%free()

        ! Shrinkwrap three points onto a z = 0 square, offset 0.25, one point
        ! deselected by an integer-valued weights array.
        call target%create()
        call target%set_points(reshape([0.0_real64, 0.0_real64, 0.0_real64, &
                                        4.0_real64, 0.0_real64, 0.0_real64, &
                                        4.0_real64, 4.0_real64, 0.0_real64, &
                                        0.0_real64, 4.0_real64, 0.0_real64], [3, 4]))
        call target%add_cell_block('triangle', reshape([1_int64, 2_int64, 3_int64, &
                                                        1_int64, 3_int64, 4_int64], [3, 2]))
        cloud_points = reshape([1.0_real64, 1.0_real64, 0.5_real64, &
                                2.0_real64, 2.0_real64, -0.7_real64, &
                                3.0_real64, 1.0_real64, 1.5_real64], [3, 3])
        vconn = reshape([1_int64, 2_int64, 3_int64], [1, 3])
        pick = [1.0_real64, 0.0_real64, 1.0_real64]
        call cloud%create()
        call cloud%set_points(cloud_points)
        call cloud%add_cell_block('vertex', vconn)
        call cloud%add_point_data('pick', pick)
        wrapped = mio_shrinkwrap(cloud, target, offset=0.25_real64, weights='pick', &
                                 num_projected=nproj, num_skipped=nskip, &
                                 max_displacement=maxd, target_watertight=wt, stat=ierr)
        call check(ierr == 0, 'shrinkwrap succeeded')
        call check(nproj == 2_int64 .and. nskip == 1_int64, 'shrinkwrap honoured the selection')
        call check(abs(maxd - 1.25_real64) < 1.0e-12_real64, 'shrinkwrap max displacement')
        call wrapped%get_points(out_points)
        call check(abs(out_points(3, 1) - 0.25_real64) < 1.0e-12_real64, &
                   'shrinkwrap put the first point at the offset height')
        call check(abs(out_points(3, 2) + 0.7_real64) < 1.0e-12_real64, &
                   'shrinkwrap left the deselected point alone')
        call wrapped%free()
        ! An unknown normal weight is refused by name.
        wrapped = mio_shrinkwrap(cloud, target, normal_weight='nope', stat=ierr)
        call check(ierr /= 0, 'shrinkwrap rejects an unknown normal weight')
        call cloud%free()
        call target%free()

        ! Sobolev deformation of a checkerboard on a 3x3 triangle grid with one
        ! corner pinned by array: converges, damps, pins.
        do j = 0, 3
            do i = 0, 3
                grid_points(:, j*4 + i + 1) = [real(i, real64), real(j, real64), 0.0_real64]
                disp(:, j*4 + i + 1) = [0.0_real64, 0.0_real64, &
                                        merge(0.1_real64, -0.1_real64, mod(i + j, 2) == 1)]
            end do
        end do
        do j = 0, 2
            do i = 0, 2
                gconn(:, (j*3 + i)*2 + 1) = [j*4 + i + 1, j*4 + i + 2, j*4 + i + 6]
                gconn(:, (j*3 + i)*2 + 2) = [j*4 + i + 1, j*4 + i + 6, j*4 + i + 5]
            end do
        end do
        pin = 0.0_real64
        pin(1) = 1.0_real64
        call grid%create()
        call grid%set_points(grid_points)
        call grid%add_cell_block('triangle', gconn)
        call grid%add_point_data('d', disp)
        call grid%add_point_data('pin', pin)
        moved = grid%sobolev_deform('d', 2.0_real64, fixed_points_array='pin', &
                                    record_filtered=.true., num_iterations=nit, &
                                    converged=conv, num_fixed=nfixed, &
                                    max_displacement=maxd, stat=ierr)
        call check(ierr == 0, 'sobolev_deform succeeded')
        call check(conv .and. nit > 0_int64, 'sobolev_deform converged')
        call check(nfixed == 1_int64, 'sobolev_deform pinned by array')
        call check(maxd > 0.0_real64 .and. maxd < 0.1_real64, 'sobolev_deform damped the field')
        call moved%get_points(out_points)
        call check(abs(out_points(3, 1)) < 1.0e-15_real64, 'the pinned point did not move')
        call moved%free()
        ! A missing array is refused.
        moved = grid%sobolev_deform('missing', 1.0_real64, stat=ierr)
        call check(ierr /= 0, 'sobolev_deform rejects a missing array')
        call grid%free()
    end subroutine

    subroutine check(ok, what)
        logical, intent(in) :: ok
        character(*), intent(in) :: what
        if (.not. ok) then
            fails = fails + 1
            write (error_unit, '(a)') 'FAIL: '//what
        end if
    end subroutine

end program test_fortran_api
