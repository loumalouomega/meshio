"""Curved-cell tessellation: `tessellate` / `Tessellation`.

Every discriminating oracle below is sabotage-verified where the task calls
for it: a helper temporarily corrupts one piece of the implementation,
asserts the relevant assertion then fails, and restores it in a ``finally``
block. ``_CellLattice._cache`` must be cleared around any sabotage that
patches basis-function code, since the per-``(curved_type, levels)`` lattice
cache would otherwise silently keep serving the pre-sabotage answer.
"""

from __future__ import annotations

import meshioplusplus as mio
import numpy as np
import pytest
from meshioplusplus import _tessellation as T
from meshioplusplus._regions import Region
from meshioplusplus._tessellation import (
    _NODE_COORDS,
    CURVED_TYPES,
    SOURCE_CELL_NAME,
    SOURCE_POINT_NAME,
    STENCIL_NAME,
    SUB_INDEX_NAME,
    WEIGHTS_NAME,
    Tessellation,
    _CellLattice,
    _corner_weights,
    tessellate,
)


def _clear_cache():
    _CellLattice._cache.clear()


# --------------------------------------------------------------------------- #
# fixture builders                                                            #
# --------------------------------------------------------------------------- #
def _affine_curved_cell(curved_type, matrix, offset):
    """One cell of ``curved_type`` whose every node sits at ``matrix @ xi +
    offset`` for its own reference coordinate ``xi`` -- a genuinely AFFINE
    isoparametric map (never merely trilinear), for which the isoparametric
    basis reproduces the map exactly everywhere, not only at its own nodes.
    """
    coords = _NODE_COORDS[curved_type]
    dim = coords.shape[1]
    matrix = np.asarray(matrix, dtype=np.float64)[:dim, :dim]
    offset = np.asarray(offset, dtype=np.float64)[:dim]
    real = coords @ matrix.T + offset
    return mio.Mesh(real, [(curved_type, np.arange(coords.shape[0]).reshape(1, -1))])


_ASYMMETRIC_MATRIX_3D = np.array([[1.7, 0.3, -0.2], [0.1, 2.1, 0.4], [-0.3, 0.2, 1.3]])
_ASYMMETRIC_MATRIX_2D = np.array([[1.6, 0.4], [-0.3, 2.2]])


def _random_asymmetric_hex27():
    """A generically-asymmetric (non-affine, genuinely trilinear) hexahedron27
    whose 27 nodes are at the exact trilinear reduction of 8 perturbed
    corners -- the fixture the module docstring's affine-degeneracy oracle
    needs to be non-trivial (a symmetric cube cannot discriminate several
    wrong node permutations, since several leave the point SET unchanged)."""
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.3, 0.1],
            [3.2, 2.7, -0.2],
            [0.1, 2.5, 0.3],
            [0.2, -0.1, 3.1],
            [3.3, 0.2, 3.4],
            [3.1, 2.6, 2.9],
            [-0.1, 2.4, 3.2],
        ]
    )
    coords = _NODE_COORDS["hexahedron27"]
    wc = _corner_weights("hexahedron", coords)
    real_nodes = wc @ corners
    mesh = mio.Mesh(real_nodes, [("hexahedron27", np.arange(27).reshape(1, 27))])
    return mesh, corners


def _symmetric_cube_hex27():
    """A perfectly symmetric unit cube's own hexahedron27 -- the fixture the
    plan explicitly requires be checked and shown INADEQUATE (several wrong
    node permutations leave the point set unchanged on this fixture)."""
    coords = _NODE_COORDS["hexahedron27"]
    return mio.Mesh(coords.copy(), [("hexahedron27", np.arange(27).reshape(1, 27))])


def _two_hex27_sharing_a_face(scramble_b=True):
    """Two axis-aligned unit cubes sharing the x=1 face, each elevated to
    hexahedron27 via an exactly AFFINE per-cell map. Cell B's own local
    corner order is a cyclic permutation of cell A's convention (rotated
    about the shared face's own normal), so their local face-traversal
    orders for the SHARED face genuinely differ -- the scenario the
    "smallest id, two steps away" diagonal rule exists to handle.
    """
    # Cell A: standard corners of [0,1]^3.
    a_corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    b_corners = a_corners + np.array([1.0, 0.0, 0.0])
    if scramble_b:
        # A cyclic relabelling of the standard corner order (rotate the
        # bottom ring and the top ring together by one step) -- still a
        # valid, positively oriented hexahedron corner order (edges/faces
        # are simply relabelled), but its LOCAL face traversal for the
        # shared x=1 face differs from cell A's.
        perm = [1, 2, 3, 0, 5, 6, 7, 4]
        b_corners = b_corners[perm]

    def elevate(corners):
        wc = _corner_weights("hexahedron", _NODE_COORDS["hexahedron27"])
        return wc @ corners

    a_nodes = elevate(a_corners)
    b_nodes = elevate(b_corners)

    # Shared points: A's corners {1,2,5,6} (the x=1 face) must be the SAME
    # global points as B's corners {whichever local indices are at x=1 in
    # B's own -- possibly permuted -- order}. Build a merged point table by
    # matching on coordinates.
    all_a = a_nodes
    all_b = b_nodes
    points = [p for p in all_a]
    b_to_global = []
    tol = 1e-9
    for p in all_b:
        found = None
        for i, q in enumerate(points):
            if np.linalg.norm(p - q) < tol:
                found = i
                break
        if found is None:
            found = len(points)
            points.append(p)
        b_to_global.append(found)
    points = np.asarray(points)
    a_conn = np.arange(27)
    b_conn = np.asarray(b_to_global, dtype=np.int64)
    mesh = mio.Mesh(
        points,
        [("hexahedron27", np.stack([a_conn, b_conn]))],
    )
    return mesh


# --------------------------------------------------------------------------- #
# basis-level sanity (not itself one of the plan's oracles, but what the      #
# integration oracles below build on)                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curved_type", CURVED_TYPES)
def test_shape_functions_partition_of_unity_and_kronecker_delta(curved_type):
    coords = _NODE_COORDS[curved_type]
    n = T._shape_functions(curved_type, coords)
    assert np.allclose(n.sum(axis=1), 1.0)
    assert np.allclose(n, np.eye(coords.shape[0]), atol=1e-9)


# --------------------------------------------------------------------------- #
# oracle: affine degeneracy on a generic (asymmetric) cell                   #
# --------------------------------------------------------------------------- #
def test_affine_degeneracy_hexahedron27():
    mesh, corners = _random_asymmetric_hex27()
    rng = np.random.RandomState(0)
    xi = rng.uniform(-1.0, 1.0, size=(50, 3))
    direct = _corner_weights("hexahedron", xi) @ corners
    via_basis = T._shape_functions("hexahedron27", xi) @ mesh.points
    assert np.allclose(via_basis, direct, atol=1e-12)


def test_symmetric_cube_is_an_inadequate_fixture_documented_finding():
    """On a symmetric cube, permuting two face-centre nodes (20/22) leaves
    the reconstructed point SET unchanged for a specific set of query
    points -- confirming the plan's own note that a symmetric fixture
    cannot discriminate several wrong permutations. The asymmetric fixture
    above is what actually catches such a defect (see the sabotage test)."""
    mesh = _symmetric_cube_hex27()
    xi = _NODE_COORDS["hexahedron27"]
    saved = T._NODE_COORDS["hexahedron27"].copy()
    bad = saved.copy()
    bad[[20, 22]] = bad[[22, 20]]
    T._NODE_COORDS["hexahedron27"] = bad
    try:
        via_basis = T._shape_functions("hexahedron27", xi) @ mesh.points
    finally:
        T._NODE_COORDS["hexahedron27"] = saved
    # The 27-point SET is unchanged (still exactly the cube's own 27 nodes,
    # just permuted) even though two specific nodes were mislabelled.
    for row in xi:
        d = np.linalg.norm(via_basis - row, axis=1)
        assert d.min() < 1e-9


def test_affine_degeneracy_sabotage_is_caught_on_the_asymmetric_fixture():
    mesh, corners = _random_asymmetric_hex27()
    rng = np.random.RandomState(1)
    xi = rng.uniform(-1.0, 1.0, size=(30, 3))
    direct = _corner_weights("hexahedron", xi) @ corners

    src = T.__file__
    text = open(src).read()
    old = "out[:, k] *= _lagrange1d(xi[:, a], coords[k, a])"
    assert old in text, "sabotage target text not found in _tessellation.py"

    saved = T._NODE_COORDS["hexahedron27"].copy()
    bad = saved.copy()
    bad[[12, 13]] = bad[[13, 12]]
    T._NODE_COORDS["hexahedron27"] = bad
    try:
        via_basis = T._shape_functions("hexahedron27", xi) @ mesh.points
        assert not np.allclose(via_basis, direct, atol=1e-9)
    finally:
        T._NODE_COORDS["hexahedron27"] = saved
    via_basis_restored = T._shape_functions("hexahedron27", xi) @ mesh.points
    assert np.allclose(via_basis_restored, direct, atol=1e-12)


def test_affine_degeneracy_tetra10_and_triangle6():
    corners = np.array([[0, 0, 0.0], [2, 0.1, 0.0], [0.1, 2.1, 0.0], [0.05, 0.05, 2.2]])
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    nodes = np.array(
        list(corners) + [0.5 * (corners[a] + corners[b]) for a, b in edges]
    )
    mesh = mio.Mesh(nodes, [("tetra10", np.arange(10).reshape(1, 10))])
    tess = tessellate(mesh, levels=2)
    # A perfectly affine tetra10: total tessellated volume must equal the
    # linear tetra's own volume to machine precision.
    v = np.abs(np.linalg.det(corners[1:] - corners[0])) / 6.0
    assert abs(tess.measures().sum() - v) < 1e-10


# --------------------------------------------------------------------------- #
# oracle: partition of unity (weak on its own, paired above)                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curved_type", CURVED_TYPES)
def test_partition_of_unity_at_generic_points(curved_type):
    rng = np.random.RandomState(2)
    base = T._BASE_OF[curved_type]
    dim = _NODE_COORDS[curved_type].shape[1]
    xi = rng.uniform(-0.8, 0.8, size=(20, dim))
    if base in ("triangle", "tetra"):
        # keep inside the reference simplex
        xi = np.abs(xi) * 0.3
    n = T._shape_functions(curved_type, xi)
    assert np.allclose(n.sum(axis=1), 1.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# oracle: face restriction                                                    #
# --------------------------------------------------------------------------- #
def test_face_restriction_hexahedron27_binds_to_cell_faces_table():
    """N_i vanishes off any face not containing node i, and restricted to a
    face containing it, reduces to that face's own 2-D (quad9) basis."""
    from meshioplusplus._skin import _CELL_FACES

    face_rows = _CELL_FACES["hexahedron27"]
    coords = _NODE_COORDS["hexahedron27"]
    rng = np.random.RandomState(3)
    for _ctype, _ncorners, row in face_rows:
        # Sample points ON this face (one coordinate fixed at its shared
        # value, the other two free) and check that only THIS face's own 9
        # nodes have nonzero weight there.
        face_local = list(row)
        face_coords = coords[face_local]
        fixed_axis = None
        fixed_val = None
        for axis in range(3):
            vals = np.unique(face_coords[:4, axis])
            if len(vals) == 1:
                fixed_axis = axis
                fixed_val = vals[0]
                break
        assert fixed_axis is not None
        free_axes = [a for a in range(3) if a != fixed_axis]
        u = rng.uniform(-1, 1, size=5)
        v = rng.uniform(-1, 1, size=5)
        xi = np.zeros((5, 3))
        xi[:, fixed_axis] = fixed_val
        xi[:, free_axes[0]] = u
        xi[:, free_axes[1]] = v
        n = T._shape_functions("hexahedron27", xi)
        off_face = [i for i in range(27) if i not in face_local]
        assert np.allclose(n[:, off_face], 0.0, atol=1e-10)


def test_face_restriction_sabotage_swapping_two_node_coords():
    """The same swap that breaks affine reproduction (see
    ``test_affine_degeneracy_sabotage_is_caught_on_the_asymmetric_fixture``)
    also breaks face restriction: on at least one of the six faces, an
    off-face node's weight is no longer exactly zero."""
    from meshioplusplus._skin import _CELL_FACES

    coords = _NODE_COORDS["hexahedron27"]

    def off_face_max(coords_table):
        worst = 0.0
        for _ctype, _n, row in _CELL_FACES["hexahedron27"]:
            face = list(row)
            fixed_axis = None
            for axis in range(3):
                vals = np.unique(coords_table[face[:4], axis])
                if len(vals) == 1:
                    fixed_axis = axis
                    fixed_val = vals[0]
                    break
            assert fixed_axis is not None
            free = [a for a in range(3) if a != fixed_axis]
            rng = np.random.RandomState(hash((tuple(row), 0)) % (2**31))
            xi = np.zeros((8, 3))
            xi[:, free[0]] = rng.uniform(-1, 1, size=8)
            xi[:, free[1]] = rng.uniform(-1, 1, size=8)
            xi[:, fixed_axis] = fixed_val
            n = T._shape_functions("hexahedron27", xi)
            off_face = [i for i in range(27) if i not in face]
            worst = max(worst, float(np.abs(n[:, off_face]).max()))
        return worst

    baseline_worst = off_face_max(coords)
    assert baseline_worst < 1e-10

    saved = T._NODE_COORDS["hexahedron27"].copy()
    bad = saved.copy()
    bad[[12, 13]] = bad[[13, 12]]
    T._NODE_COORDS["hexahedron27"] = bad
    try:
        sabotaged_worst = off_face_max(T._NODE_COORDS["hexahedron27"])
    finally:
        T._NODE_COORDS["hexahedron27"] = saved
    assert sabotaged_worst > 1e-6
    assert off_face_max(T._NODE_COORDS["hexahedron27"]) < 1e-10


# --------------------------------------------------------------------------- #
# oracle: curved volume convergence                                          #
# --------------------------------------------------------------------------- #
def _bumped_hex27(bump):
    coords = _NODE_COORDS["hexahedron27"].copy()
    real = coords.copy()
    # Perturb the six face-centre nodes (20-25) and body node (26) outward
    # along their own axis, making the cell genuinely curved.
    real[20:26] *= 1.0 + bump
    real[26] *= 1.0 + bump
    return mio.Mesh(real, [("hexahedron27", np.arange(27).reshape(1, 27))])


def _gauss_reference_volume(bump, n=12):
    """An independent Gauss-Legendre quadrature of the SAME curved cell's
    volume, computed directly from the isoparametric map's Jacobian --
    never via this module's own tessellation."""
    coords = _NODE_COORDS["hexahedron27"].copy()
    real = coords.copy()
    real[20:26] *= 1.0 + bump
    real[26] *= 1.0 + bump
    gp, gw = np.polynomial.legendre.leggauss(n)
    total = 0.0
    h = 1e-6
    for i in range(n):
        for j in range(n):
            for k in range(n):
                xi0 = np.array([gp[i], gp[j], gp[k]])
                # numerical Jacobian of x(xi) = N(xi) @ real
                jac = np.empty((3, 3))
                for a in range(3):
                    d = np.zeros(3)
                    d[a] = h
                    xp = T._shape_functions("hexahedron27", (xi0 + d)[None, :]) @ real
                    xm = T._shape_functions("hexahedron27", (xi0 - d)[None, :]) @ real
                    jac[:, a] = (xp[0] - xm[0]) / (2 * h)
                total += gw[i] * gw[j] * gw[k] * abs(np.linalg.det(jac))
    return total


def test_curved_volume_convergence():
    bump = 0.35
    ref = _gauss_reference_volume(bump)
    errors = []
    for levels in (1, 2, 3):
        mesh = _bumped_hex27(bump)
        _clear_cache()
        tess = tessellate(mesh, levels=levels, curved=True)
        errors.append(abs(tess.measures().sum() - ref))
    # error should shrink substantially with each added level
    assert errors[0] > errors[1] * 2.0
    assert errors[1] > errors[2] * 1.5


def test_curved_volume_convergence_sabotage_curved_false_plateaus():
    bump = 0.35
    ref = _gauss_reference_volume(bump)
    errors = []
    for levels in (1, 2, 3):
        mesh = _bumped_hex27(bump)
        _clear_cache()
        tess = tessellate(mesh, levels=levels, curved=False)
        errors.append(abs(tess.measures().sum() - ref))
    # curved=False treats the cell as pass-through (unchanged, still typed
    # hexahedron27 -- a curved type `_data_average._cell_measures` does not
    # know how to measure), so `measures()` is NaN identically regardless
    # of `levels`: no shrinking convergence, since nothing was tessellated.
    assert np.isnan(errors).all()


# --------------------------------------------------------------------------- #
# oracle: straight-cell measure exactness                                    #
# --------------------------------------------------------------------------- #
def test_straight_cell_measures_match_stats_on_a_linear_mesh():
    mesh = mio.grid([3, 2, 2])
    tess = tessellate(mesh)
    stats = mio.compute_stats(mesh)
    assert abs(tess.measures().sum() - stats["unsigned_volume"]) < 1e-8


# --------------------------------------------------------------------------- #
# oracle: gather / scatter round trip                                        #
# --------------------------------------------------------------------------- #
def test_gather_scatter_round_trip_at_a_shared_corner():
    """Four tetra10 cells sharing one corner: a value gathered onto the
    tessellated points and scattered back must reproduce the original
    value exactly at that shared corner, regardless of which of the four
    incident cells contributed the stencil."""
    # A small tetra10 mesh: split a cube into 5 tets, all sharing one
    # corner point (index 0), each elevated with linear (affine-per-cell)
    # mid-edge nodes so gather()/scatter() have no interpolation error.
    cube = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    tets = [
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (1, 3, 4, 6),
        (3, 4, 6, 7),
        (1, 4, 5, 6),
    ]
    edges = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    points = list(cube)
    cells = []
    for tet in tets:
        corner_pts = cube[list(tet)]
        mid_ids = []
        for a, b in edges:
            mid = 0.5 * (corner_pts[a] + corner_pts[b])
            idx = None
            for i, p in enumerate(points):
                if np.linalg.norm(p - mid) < 1e-9:
                    idx = i
                    break
            if idx is None:
                idx = len(points)
                points.append(mid)
            mid_ids.append(idx)
        cells.append(list(tet) + mid_ids)
    points = np.asarray(points)
    field = np.arange(len(points), dtype=float) * 2.0 + 1.0
    mesh = mio.Mesh(
        points,
        [("tetra10", np.asarray(cells, dtype=np.int64))],
        point_data={"f": field},
    )
    tess = tessellate(mesh, levels=2, fields=True)
    gathered = tess.gather(field)
    back = tess.scatter(gathered)
    assert abs(back[0] - field[0]) < 1e-10


# --------------------------------------------------------------------------- #
# oracle: cell aggregation                                                    #
# --------------------------------------------------------------------------- #
def test_aggregate_matches_plain_group_by_mean():
    mesh, _ = _random_asymmetric_hex27()
    tess = tessellate(mesh, levels=2)
    n = tess.source_cell.shape[0]
    rng = np.random.RandomState(5)
    values = rng.uniform(size=n)
    uniq, agg = tess.aggregate(values, reduction="mean")
    assert list(uniq) == [0]
    assert abs(agg[0] - values.mean()) < 1e-12

    uniq_w, agg_w = tess.aggregate(values, reduction="weighted_mean")
    w = np.abs(tess.measures())
    expect = np.average(values, weights=w)
    assert abs(agg_w[0] - expect) < 1e-10

    uniq_f, agg_f = tess.aggregate(values, reduction="first")
    order = np.argsort(tess.sub_index)
    assert agg_f[0] == values[order[0]]


# --------------------------------------------------------------------------- #
# oracle: watertightness                                                     #
# --------------------------------------------------------------------------- #
def test_watertight_two_hexahedra_sharing_a_scrambled_face():
    mesh = _two_hex27_sharing_a_face(scramble_b=True)
    tess = tessellate(mesh, levels=2)
    w = tess.schema["watertight"]
    assert w["num_facets_with_bad_count"] == 0
    n = len(tess.mesh.points)
    assert np.unique(tess.mesh.points, axis=0).shape[0] == n


def test_watertight_sabotage_dropping_canonical_rotation():
    """Two cells describing the SAME physical face in opposite traversal
    directions (the natural outward-normal convention: A's own face and
    B's matching face are mirror images of each other) must compute the
    SAME canonical order via `_canonical_cycle` -- and hence the SAME
    ``(u', v')`` key for the same physical point. Dropping canonicalization
    (using the raw, direction-dependent order) does not."""
    a_order = [11, 22, 33, 44]
    b_order = [44, 33, 22, 11]  # the same face, reversed traversal
    canon_a = T._canonical_cycle(a_order)
    canon_b = T._canonical_cycle(b_order)
    assert canon_a == canon_b

    def uncanonicalized(gids):
        return list(gids)

    assert uncanonicalized(a_order) != uncanonicalized(b_order)


def test_watertight_sabotage_fixed_diagonal_breaks_facet_pairing():
    """`_quad_diagonal_split`'s "smallest id, two steps away" rule gives
    the SAME diagonal pair regardless of which direction (forward or
    reversed) a face's corners are listed in -- proven mathematically in
    the module docstring (opposite-by-2-steps is invariant under reversal
    for a 4-cycle) and confirmed here. A naive FIXED-position (0, 2)
    diagonal -- "the fallback straight decomposition" the plan's sabotage
    names -- does NOT: it picks a genuinely different pair for the reversed
    listing, which is exactly the defect that would tear a mesh at a
    shared hexahedron face."""
    a_order = [11, 22, 33, 44]
    b_order = [44, 33, 22, 11]  # the same face, reversed traversal

    def diagonal_of(split):
        tri0, tri1 = split
        return tuple(sorted(set(tri0) & set(tri1)))

    diag_a = diagonal_of(T._quad_diagonal_split(a_order))
    diag_b = diagonal_of(T._quad_diagonal_split(b_order))
    assert diag_a == diag_b

    def fixed_diagonal_split(gids):
        a, b, c, d = gids[0], gids[1], gids[2], gids[3]
        return [(a, b, c), (a, c, d)]

    bad_diag_a = diagonal_of(fixed_diagonal_split(a_order))
    bad_diag_b = diagonal_of(fixed_diagonal_split(b_order))
    assert bad_diag_a != bad_diag_b


# --------------------------------------------------------------------------- #
# oracle: determinism / block-order independence                             #
# --------------------------------------------------------------------------- #
def test_determinism_independent_of_block_order():
    mesh, _ = _random_asymmetric_hex27()
    corners2 = np.array(
        [
            [5.0, 0.0, 0.0],
            [8.0, 0.2, 0.1],
            [8.1, 2.6, -0.1],
            [5.2, 2.4, 0.2],
            [5.1, -0.2, 3.0],
            [8.2, 0.1, 3.3],
            [8.0, 2.5, 2.8],
            [4.9, 2.3, 3.1],
        ]
    )
    coords = _NODE_COORDS["hexahedron27"]
    wc = _corner_weights("hexahedron", coords)
    nodes2 = wc @ corners2
    two_block_mesh = mio.Mesh(
        np.concatenate([mesh.points, nodes2]),
        [
            ("hexahedron27", np.arange(27).reshape(1, 27)),
            ("hexahedron27", (np.arange(27) + 27).reshape(1, 27)),
        ],
    )
    reordered_mesh = mio.Mesh(
        np.concatenate([nodes2, mesh.points]),
        [
            ("hexahedron27", np.arange(27).reshape(1, 27)),
            ("hexahedron27", (np.arange(27) + 27).reshape(1, 27)),
        ],
    )
    _clear_cache()
    t1 = tessellate(two_block_mesh, levels=1)
    _clear_cache()
    t2 = tessellate(reordered_mesh, levels=1)
    assert np.allclose(sorted(t1.mesh.points.tolist()), sorted(t2.mesh.points.tolist()))


# --------------------------------------------------------------------------- #
# oracle: mixed-mesh merge                                                    #
# --------------------------------------------------------------------------- #
def test_mixed_curved_and_linear_blocks_share_one_point_table():
    curved_mesh, _ = _random_asymmetric_hex27()
    linear = mio.Mesh(
        np.array(
            [
                [10.0, 0, 0],
                [11, 0, 0],
                [11, 1, 0],
                [10, 1, 0],
                [10, 0, 1],
                [11, 0, 1],
                [11, 1, 1],
                [10, 1, 1],
            ]
        ),
        [("hexahedron", np.arange(8).reshape(1, 8))],
    )
    mesh = mio.Mesh(
        np.concatenate([curved_mesh.points, linear.points]),
        [
            ("hexahedron27", np.arange(27).reshape(1, 27)),
            ("hexahedron", (np.arange(8) + 27).reshape(1, 8)),
        ],
    )
    tess = tessellate(mesh)
    types = {cb.type for cb in tess.mesh.cells}
    assert "hexahedron" in types  # pass-through kept
    assert "tetra" in types  # curved cell tessellated
    # a single coherent point table
    assert tess.mesh.points.shape[0] == len(tess.source_point)


# --------------------------------------------------------------------------- #
# oracle: no-op on a linear mesh                                             #
# --------------------------------------------------------------------------- #
def test_noop_on_a_purely_linear_mesh():
    mesh = mio.grid([2, 2, 2])
    tess = tessellate(mesh)
    assert np.array_equal(tess.mesh.points, mesh.points)
    assert np.array_equal(tess.source_point, np.arange(len(mesh.points)))
    assert (tess.source_point >= 0).all()
    for cb, orig in zip(tess.mesh.cells, mesh.cells):
        assert cb.type == orig.type
        assert np.array_equal(cb.data, orig.data)


# --------------------------------------------------------------------------- #
# oracle: staleness detection                                                 #
# --------------------------------------------------------------------------- #
def test_staleness_detection_after_transform():
    mesh, _ = _random_asymmetric_hex27()
    tess = tessellate(mesh, levels=2, record_stencil=True)
    moved = mio.transform(mesh, translate=(5.0, 0.0, 0.0))
    with pytest.warns(UserWarning, match="no longer reproduce"):
        reconstructed = Tessellation.from_mesh(tess.mesh, moved)
    # Degraded: only kept (corner) points still have a usable stencil.
    kept = reconstructed.source_point >= 0
    assert np.all(reconstructed.weights[kept].sum(axis=1) == 1.0)


def test_from_mesh_reconstructs_a_fresh_tessellation_exactly():
    mesh, _ = _random_asymmetric_hex27()
    tess = tessellate(mesh, levels=2, record_stencil=True)
    reconstructed = Tessellation.from_mesh(tess.mesh, mesh)
    assert np.array_equal(reconstructed.source_point, tess.source_point)
    g1 = tess.gather(mesh.points)
    g2 = reconstructed.gather(mesh.points)
    assert np.allclose(g1, g2, atol=1e-10)


# --------------------------------------------------------------------------- #
# scope: hexahedron20 is deliberately excluded (documented, not just implied) #
# --------------------------------------------------------------------------- #
def test_hexahedron20_is_not_a_curved_type():
    assert "hexahedron20" not in CURVED_TYPES
    coords = _NODE_COORDS["hexahedron27"]
    conn20 = np.arange(20).reshape(1, 20)
    mesh = mio.Mesh(coords[:20].copy(), [("hexahedron20", conn20)])
    tess = tessellate(mesh)
    assert tess.mesh.cells[0].type == "hexahedron20"
    assert np.array_equal(tess.mesh.cells[0].data, conn20)


# --------------------------------------------------------------------------- #
# record_stencil / fields options and CLI/MCP-relevant array names            #
# --------------------------------------------------------------------------- #
def test_record_stencil_attaches_arrays_only_when_requested():
    mesh, _ = _random_asymmetric_hex27()
    tess_off = tessellate(mesh, levels=1, record_stencil=False)
    assert STENCIL_NAME not in tess_off.mesh.point_data
    assert WEIGHTS_NAME not in tess_off.mesh.point_data
    assert SOURCE_POINT_NAME in tess_off.mesh.point_data
    assert SOURCE_CELL_NAME in tess_off.mesh.cell_data
    assert SUB_INDEX_NAME in tess_off.mesh.cell_data

    tess_on = tessellate(mesh, levels=1, record_stencil=True)
    assert STENCIL_NAME in tess_on.mesh.point_data
    assert WEIGHTS_NAME in tess_on.mesh.point_data
    assert tess_on.mesh.point_data[STENCIL_NAME].shape == (len(tess_on.mesh.points), 27)


def test_fields_false_skips_point_and_cell_data():
    mesh, _ = _random_asymmetric_hex27()
    mesh.point_data["f"] = np.arange(27, dtype=float)
    mesh.cell_data["c"] = [np.array([1.0])]
    tess = tessellate(mesh, fields=False)
    assert "f" not in tess.mesh.point_data
    assert "c" not in tess.mesh.cell_data


def test_to_dict_is_json_safe():
    import json

    mesh, _ = _random_asymmetric_hex27()
    tess = tessellate(mesh, levels=1)
    json.dumps(tess.to_dict())


def test_levels_must_be_non_negative():
    mesh, _ = _random_asymmetric_hex27()
    with pytest.raises(ValueError):
        tessellate(mesh, levels=-1)


def test_curved_false_is_a_full_noop():
    mesh, _ = _random_asymmetric_hex27()
    tess = tessellate(mesh, curved=False)
    assert tess.mesh.cells[0].type == "hexahedron27"
    assert np.array_equal(tess.mesh.points, mesh.points)


def test_regions_are_carried_for_pass_through_points():
    mesh = mio.grid([2, 2, 2])
    mesh.regions.append(Region("corner", "point", np.array([0]), dim=0, tag=1))
    tess = tessellate(mesh)
    assert tess.mesh.regions
