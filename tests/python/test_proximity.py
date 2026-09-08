"""Tests for proximity graphs (radius / kNN / periodic / bistride).

The oracles here are deliberately independent of the code under test:

* the bucket-grid search is checked against the exact ``O(N^2)`` brute-force
  path, which shares no code with it beyond the minimum-image one-liner;
* the periodic case is checked against a *hand-computed* short vector, because
  a brute force that forgot the wrap would agree with a search that forgot it
  too;
* the bistride hierarchy is checked against the published BSMS reference for a
  16-node path graph -- shapes and surviving ids, both computed by hand.

Everything is pure numpy, so the whole file runs in the default CI matrix with
no optional dependency installed.
"""

import numpy as np
import pytest

import meshioplusplus as mio
from meshioplusplus import _proximity as px
from meshioplusplus._ml import _canonical_edges


# --------------------------------------------------------------------------- #
# fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _cloud(n=5000, scale=10.0, seed=0, dim=3):
    """A uniform cloud, large enough to force the bucket-grid path."""
    return np.random.RandomState(seed).rand(n, dim) * scale


def _brute(points, box=None, radius=None, k=None):
    """The exact answer, by the route that shares no machinery with the grid."""
    if radius is not None:
        a, b = px._brute_force_radius(points, radius, box)
    else:
        a, b = px._brute_force_knn(points, k, box)
    return _canonical_edges(a, b, True)


def _path_graph(n=16):
    pos = np.stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)], axis=1)
    a = np.arange(n - 1)
    return _canonical_edges(a, a + 1, True), pos


# --------------------------------------------------------------------------- #
# the radius graph                                                            #
# --------------------------------------------------------------------------- #
def test_a_unit_square_links_its_sides_and_not_its_diagonals():
    square = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]])
    edges = mio.proximity_graph(square, radius=1.0, undirected=False)
    # The diagonals are sqrt(2) away; the sides are exactly at the cutoff, so
    # this also pins the cutoff as inclusive (d <= r, never d < r).
    assert edges.T.tolist() == [[0, 1], [0, 3], [1, 2], [2, 3]]


def test_a_pair_exactly_at_the_cutoff_is_linked():
    pair = np.array([[0.0, 0, 0], [0.25, 0, 0]])
    assert mio.proximity_graph(pair, radius=0.25).shape[1] == 2
    assert mio.proximity_graph(pair, radius=0.249).shape[1] == 0


def test_the_bucket_search_also_treats_the_cutoff_as_inclusive():
    # The two tests above sit under _BRUTE_FORCE_MAX and so only exercise the
    # O(N^2) route; this one carries the same exact-distance pair on a cloud
    # large enough to force the grid, where an exclusive comparison would drop
    # it. Both coordinates are powers of two, so the distance is exact.
    radius = 0.5
    points = np.concatenate(
        [
            _cloud(3000, seed=13),
            np.array([[2.0, 2.0, 2.0], [2.5, 2.0, 2.0]]),
        ]
    )
    assert len(points) > px._BRUTE_FORCE_MAX
    edges = mio.proximity_graph(points, radius=radius, undirected=False)
    assert any(set(e) == {3000, 3001} for e in edges.T.tolist())


@pytest.mark.parametrize("radius", [0.4, 0.9])
def test_the_bucket_search_matches_brute_force(radius):
    points = _cloud()
    assert len(points) > px._BRUTE_FORCE_MAX  # the grid path really is running
    assert np.array_equal(
        _brute(points, radius=radius), mio.proximity_graph(points, radius=radius)
    )


def test_a_pair_straddling_a_bucket_face_is_linked():
    # Both points sit just either side of a bucket boundary at x = radius,
    # closer to each other than the cutoff: found only if the neighbouring
    # bucket is searched at all.
    radius = 0.5
    points = np.concatenate(
        [
            _cloud(3000, seed=4),
            np.array([[radius - 1e-9, 5.0, 5.0], [radius + 0.499, 5.0, 5.0]]),
        ]
    )
    edges = mio.proximity_graph(points, radius=radius, undirected=False)
    pair = {3000, 3001}
    assert any(set(e) == pair for e in edges.T.tolist())


def test_the_lattice_is_an_accelerator_and_never_part_of_the_answer():
    points = _cloud(4000, seed=6)
    reference = mio.proximity_graph(points, method="knn", max_neighbors=12)
    original = px._KNN_FILL
    try:
        for fill in (0.25, 1.0, 4.0):
            px._KNN_FILL = fill
            assert np.array_equal(
                reference,
                mio.proximity_graph(points, method="knn", max_neighbors=12),
            )
    finally:
        px._KNN_FILL = original


# --------------------------------------------------------------------------- #
# the k-nearest-neighbour graph                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [4, 16])
def test_the_knn_search_matches_brute_force(k):
    points = _cloud()
    assert np.array_equal(
        _brute(points, k=k),
        mio.proximity_graph(points, method="knn", max_neighbors=k),
    )


def test_knn_is_symmetrized():
    # p3 is far away: its nearest neighbour is p0, but p0's own two nearest are
    # p1 and p2. The edge exists in both directions all the same.
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [40.0, 0, 0]])
    edges = mio.proximity_graph(points, method="knn", max_neighbors=2).T.tolist()
    assert [0, 3] in edges and [3, 0] in edges


def test_knn_clamps_the_budget_to_the_cloud():
    points = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    assert mio.proximity_graph(points, method="knn", max_neighbors=9).shape[1] == 2


def test_knn_survives_a_cloud_of_duplicates():
    # Only 60 distinct positions among 3600 points: every bucket overflows and
    # the search has to fall through to exhaustion rather than loop forever.
    points = np.repeat(np.random.RandomState(7).rand(60, 3), 60, axis=0)
    assert np.array_equal(
        _brute(points, k=5),
        mio.proximity_graph(points, method="knn", max_neighbors=5),
    )


def test_a_collinear_cloud_has_neighbours():
    line = np.stack([np.linspace(0, 1, 3000), np.zeros(3000), np.zeros(3000)], axis=1)
    assert np.array_equal(
        _brute(line, k=4),
        mio.proximity_graph(line, method="knn", max_neighbors=4),
    )


# --------------------------------------------------------------------------- #
# periodic boxes                                                              #
# --------------------------------------------------------------------------- #
def test_a_pair_across_the_boundary_is_linked_by_its_short_image():
    box = 1.0
    points = np.array([[0.01, 0.5, 0.5], [box - 0.01, 0.5, 0.5]])
    edges = mio.proximity_graph(points, radius=0.05, box_size=box, undirected=False)
    assert edges.shape[1] == 1, "the periodic image was not searched"
    vectors = mio.edge_vectors(points, edges, box_size=box)
    # Hand-computed: 0.01 + 0.01 across the seam, never 0.98 through the box.
    assert vectors[0, 0] == pytest.approx(0.02)
    assert vectors[0, 3] == pytest.approx(0.02)


def test_without_a_box_the_same_pair_is_not_linked():
    points = np.array([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]])
    assert mio.proximity_graph(points, radius=0.05).shape[1] == 0


@pytest.mark.parametrize("box", [[2.0, 3.0, 1.5], [1.0, 1.0, 1.0], [0.31, 0.77, 1.03]])
@pytest.mark.parametrize("radius", [0.05, 0.15])
def test_the_periodic_bucket_search_matches_brute_force(box, radius):
    # The box sides are deliberately not multiples of the radius: the seam cell
    # is then a partial one, and a lattice sized at exactly the radius would
    # miss pairs across it.
    box = np.array(box)
    points = np.mod(_cloud(4000, seed=3, scale=1.0) * box, box)
    assert np.array_equal(
        _brute(points, box=box, radius=radius),
        mio.proximity_graph(points, radius=radius, box_size=box),
    )


def test_the_periodic_knn_search_matches_brute_force():
    box = np.array([2.0, 3.0, 1.5])
    points = np.mod(_cloud(4000, seed=9, scale=1.0) * box, box)
    assert np.array_equal(
        _brute(points, box=box, k=8),
        mio.proximity_graph(points, method="knn", max_neighbors=8, box_size=box),
    )


def test_unwrapped_positions_are_taken_modulo_the_box():
    box = 1.0
    inside = np.array([[0.01, 0.5, 0.5], [0.99, 0.5, 0.5]])
    outside = inside + np.array([[3.0, -2.0, 5.0], [0.0, 0.0, 0.0]])
    assert np.array_equal(
        mio.proximity_graph(inside, radius=0.05, box_size=box),
        mio.proximity_graph(outside, radius=0.05, box_size=box),
    )


def test_a_radius_past_half_the_box_is_refused():
    with pytest.raises(ValueError, match="minimum image is ambiguous"):
        mio.proximity_graph(_cloud(10, scale=1.0), radius=0.6, box_size=1.0)


# --------------------------------------------------------------------------- #
# the shape of the answer                                                     #
# --------------------------------------------------------------------------- #
def test_the_output_is_canonical_and_deterministic():
    points = _cloud(3000, seed=2)
    first = mio.proximity_graph(points, radius=0.5)
    assert first.dtype == np.int64 and first.flags["C_CONTIGUOUS"]
    assert first.tobytes() == mio.proximity_graph(points, radius=0.5).tobytes()
    src, dst = first
    assert np.all(src != dst), "a self edge survived"
    assert np.array_equal(np.lexsort((dst, src)), np.arange(src.size))
    # Both directions, and each exactly once.
    directed = {(int(a), int(b)) for a, b in first.T}
    assert all((b, a) in directed for a, b in directed)
    assert len(directed) == src.size
    half = mio.proximity_graph(points, radius=0.5, undirected=False)
    assert half.shape[1] * 2 == first.shape[1]
    assert np.all(half[0] < half[1])


def test_a_mesh_can_supply_the_positions():
    mesh = mio.grid([4, 4, 4])
    from_mesh = mio.proximity_graph(mesh, radius=1.1)
    from_points = mio.proximity_graph(np.asarray(mesh.points), radius=1.1)
    assert np.array_equal(from_mesh, from_points)
    # Cell centroids are a different, coarser point set.
    assert mio.proximity_graph(mesh, radius=1.1, kind="cell").shape[1] == 288


def test_a_two_dimensional_cloud_keeps_two_components():
    points = _cloud(3000, seed=8, dim=2)
    edges = mio.proximity_graph(points, radius=0.5)
    assert mio.edge_vectors(points, edges).shape == (edges.shape[1], 3)


def test_degenerate_inputs_are_empty_rather_than_an_error():
    assert mio.proximity_graph(np.zeros((1, 3)), radius=1.0).shape == (2, 0)
    assert mio.proximity_graph(np.zeros((0, 3)), radius=1.0).shape == (2, 0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(radius=1.0, max_neighbors=3), "max_neighbors belongs to"),
        (dict(method="knn", radius=1.0), "radius belongs to"),
        (dict(), "needs a positive radius"),
        (dict(radius=-1.0), "needs a positive radius"),
        (dict(method="knn"), "needs max_neighbors"),
        (dict(method="tree", radius=1.0), "unknown method"),
        (dict(radius=1.0, kind="edge"), "kind must be"),
        (dict(radius=1.0, box_size=[1.0, 2.0]), "box_size must be one value"),
        (dict(radius=1.0, box_size=-1.0), "must be positive"),
    ],
)
def test_bad_requests_are_refused_by_name(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mio.proximity_graph(_cloud(20, scale=1.0), **kwargs)


def test_a_non_finite_coordinate_is_refused():
    points = _cloud(20, scale=1.0)
    points[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite coordinate"):
        mio.proximity_graph(points, radius=0.5)


# --------------------------------------------------------------------------- #
# edge vectors                                                                #
# --------------------------------------------------------------------------- #
def test_edge_vectors_are_source_minus_destination_then_the_norm():
    points = np.array([[0.0, 0, 0], [3.0, 4.0, 0.0]])
    edges = np.array([[1], [0]], dtype=np.int64)
    assert mio.edge_vectors(points, edges).tolist() == [[3.0, 4.0, 0.0, 5.0]]


def test_edge_vectors_agree_with_the_graph_sample_convention():
    from meshioplusplus.physicsnemo import _edge_attributes

    points = _cloud(500, seed=12)
    edges = mio.proximity_graph(points, radius=1.5)
    assert np.array_equal(
        mio.edge_vectors(points, edges), _edge_attributes(points, edges)
    )


def test_edge_vectors_refuse_an_out_of_range_index():
    with pytest.raises(ValueError, match="outside"):
        mio.edge_vectors(np.zeros((2, 3)), np.array([[0], [5]]))


# --------------------------------------------------------------------------- #
# the bistride hierarchy                                                      #
# --------------------------------------------------------------------------- #
def test_a_path_graph_reproduces_the_published_hierarchy():
    edges, positions = _path_graph(16)
    hierarchy = mio.bistride_hierarchy(edges, positions, num_levels=2)
    # Hand-computed: the BFS is seeded at node 7 (nearest the centroid 7.5),
    # so the two colour classes are the odd and the even indices; the tie goes
    # to the class holding the seed, and each level halves a path.
    assert [e.shape for e in hierarchy.edges] == [(2, 30), (2, 14), (2, 6)]
    assert hierarchy.ids[0].tolist() == [1, 3, 5, 7, 9, 11, 13, 15]
    assert hierarchy.ids[1].tolist() == [1, 3, 5, 7]
    assert hierarchy.schema["num_nodes"] == [16, 8, 4]
    assert hierarchy.num_levels == 2 and len(hierarchy) == 3


def _reachable_within_two(adjacency, kept, self_loops):
    """The survivors' coarse edges, straight from the definition: the squared
    adjacency (with or without self-loops), restricted and renumbered."""
    n = len(adjacency)
    base = adjacency | np.eye(n, dtype=bool) if self_loops else adjacency
    reach = base @ base
    reach[np.diag_indices(n)] = False
    rows = sorted(kept)
    return {
        (i, j)
        for i, a in enumerate(rows)
        for j, b in enumerate(rows)
        if i < j and reach[a, b]
    }


def test_a_coarse_level_is_the_squared_adjacency_with_self_loops():
    # The one subtle step, pinned against the definition rather than against
    # itself: a survivor must inherit the reach of the neighbours that were
    # dropped, which is what squaring A + I gives and squaring A alone does
    # not. The fixture is chosen so the two disagree -- survivors that were
    # already adjacent are exactly the pairs A @ A cannot see.
    positions = np.random.RandomState(3).rand(20, 3)
    edges = mio.proximity_graph(positions, method="knn", max_neighbors=3)
    hierarchy = mio.bistride_hierarchy(edges, positions, num_levels=1)
    kept = hierarchy.ids[0].tolist()

    adjacency = np.zeros((20, 20), dtype=bool)
    adjacency[edges[0], edges[1]] = True
    with_loops = _reachable_within_two(adjacency, kept, True)
    without = _reachable_within_two(adjacency, kept, False)
    assert with_loops - without, "the fixture cannot tell the two apart"

    got = {(int(a), int(b)) for a, b in hierarchy.edges[1].T if a < b}
    assert got == with_loops


def test_every_level_is_a_clean_undirected_graph():
    edges, positions = _path_graph(32)
    hierarchy = mio.bistride_hierarchy(edges, positions, num_levels=3)
    for level, (edge_set, size) in enumerate(
        zip(hierarchy.edges, hierarchy.schema["num_nodes"])
    ):
        src, dst = edge_set
        assert np.all(src != dst), f"self loop at level {level}"
        assert src.size == 0 or src.max() < size
        directed = {(int(a), int(b)) for a, b in edge_set.T}
        assert all((b, a) in directed for a, b in directed)


def test_a_grid_graph_halves_each_level():
    mesh = mio.grid([8, 8, 1])
    edges = mio.edge_index(mesh)
    hierarchy = mio.bistride_hierarchy(edges, np.asarray(mesh.points), num_levels=2)
    sizes = hierarchy.schema["num_nodes"]
    assert sizes[1] < sizes[0] and sizes[2] < sizes[1]


def test_each_component_keeps_a_representative():
    left, positions = _path_graph(8)
    edges = _canonical_edges(
        np.concatenate([left[0], left[0] + 8]),
        np.concatenate([left[1], left[1] + 8]),
        True,
    )
    positions = np.concatenate([positions, positions + [100.0, 0, 0]])
    hierarchy = mio.bistride_hierarchy(edges, positions, num_levels=1)
    kept = hierarchy.ids[0]
    assert np.any(kept < 8) and np.any(kept >= 8)


def test_an_isolated_node_survives():
    edges = _canonical_edges(np.array([0, 1]), np.array([1, 2]), True)
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [50.0, 0, 0]])
    hierarchy = mio.bistride_hierarchy(edges, positions, num_levels=1)
    assert 3 in hierarchy.ids[0].tolist()


def test_asking_for_more_levels_than_the_graph_has_is_refused():
    edges = _canonical_edges(np.array([0]), np.array([1]), True)
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="ask for fewer levels"):
        mio.bistride_hierarchy(edges, positions, num_levels=2)


def test_bistride_refuses_a_meaningless_level_count():
    edges, positions = _path_graph(8)
    with pytest.raises(ValueError, match="at least 1"):
        mio.bistride_hierarchy(edges, positions, num_levels=0)


def test_bistride_refuses_an_edge_naming_a_missing_node():
    edges = _canonical_edges(np.array([0]), np.array([9]), True)
    with pytest.raises(ValueError, match="outside"):
        mio.bistride_hierarchy(edges, np.zeros((4, 3)), num_levels=1)


def test_the_public_api_is_exported():
    for name in (
        "proximity_graph",
        "edge_vectors",
        "bistride_hierarchy",
        "BistrideHierarchy",
    ):
        assert name in mio.__all__ and hasattr(mio, name)
