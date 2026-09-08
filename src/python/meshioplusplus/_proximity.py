"""Proximity graphs (``meshioplusplus.proximity_graph``).

:func:`meshioplusplus.edge_index` builds a graph out of a mesh's *connectivity*
-- the cell edges, or the facet-sharing cell dual. For a particle method there
is no connectivity to build one from: a smoothed-particle or discrete-element
state is a cloud of positions, and what makes two particles interact is the
*interaction radius*, not a shared element. A graph network trained on such a
state needs neighbourhoods recomputed from geometry at every step, which is
what this module is.

It also covers the two structures a mesh graph needs but connectivity cannot
supply. **World edges** are a proximity graph laid alongside the mesh edges, so
a model can see two surfaces that touch without being connected -- contact,
self-collision, a fluid meeting a wall. A **bistride hierarchy** is the
multiscale pooling a deep GNN needs so information crosses the mesh in a few
message-passing steps instead of one element per step.

Everything here is pure numpy over a bucket grid, the same spatial hash
:func:`meshioplusplus.interpolate` and :func:`meshioplusplus.merge` already
use, vectorized here because a proximity graph queries every point rather than
a few. No SciPy, no KD-tree, no cell-list library: the search is exact, its
output is a function of the input alone, and the bucket size is an
implementation detail that cannot change the answer.

Periodic boxes
--------------
``box_size`` turns on the **minimum image** convention: a pair is linked by the
shortest of its images across the box's faces, so two particles either side of
a boundary are neighbours and the edge vector between them is short. This is
the difference between a periodic simulation's graph and a wrong one, and it is
one line -- ``d -= box * round(d / box)`` -- applied consistently to the search
and to :func:`edge_vectors`.

A radius larger than half the smallest box side is refused rather than
approximated: past that, a pair has two images at the same distance and
"the" minimum image does not exist.

What it costs
-------------
Both searches are ``O(N + E)`` in the bucket grid, and ``E`` is what dominates:
a radius graph's edge count grows as the cube of the radius, so the honest unit
is edges per second rather than points. Measured on one core over a uniform
cloud at unit density:

===========  ==========  ============  ========
points       method      edges         time (s)
===========  ==========  ============  ========
50 000       radius      3.3 M         1.3
50 000       knn (k=16)  0.9 M         4.5
200 000      radius      13.4 M        6.4
200 000      knn (k=16)  3.6 M         23.0
===========  ==========  ============  ========

``knn`` is dearer per edge because it must *rank* every candidate rather than
threshold it. Both are exact; neither approximates to go faster. At or below
``_BRUTE_FORCE_MAX`` points the plain ``O(N^2)`` route runs instead -- the same
answer by a shorter road, and the oracle the bucket path is tested against.

Memory follows the same rule: a pair only exists while it is being tested, and
expansions are chunked, so peak memory is set by the *output* rather than by
the cloud. A radius a few mean spacings wide is the intended operating point;
one ten times that asks for a thousand times the edges.

What comes back
---------------
``(2, E)`` int64 in the layout PyTorch Geometric and DGL expect, canonical:
both directions of every edge, lexicographically sorted, de-duplicated, no self
edges -- byte-identical to what :func:`meshioplusplus.edge_index` produces for
the same pair set, because both go through the same canonicalization.

Public API:

* :func:`proximity_graph` -- points -> a radius or k-nearest-neighbour graph.
* :func:`edge_vectors` -- the MeshGraphNet edge-feature convention, periodic-aware.
* :func:`bistride_hierarchy` / :class:`BistrideHierarchy` -- multiscale pooling.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "PROXIMITY_GRAPH_VERSION",
    "proximity_graph",
    "edge_vectors",
    "BistrideHierarchy",
    "bistride_hierarchy",
]

#: Bumped when the recorded schema changes meaning.
PROXIMITY_GRAPH_VERSION = 1

_METHODS = ("radius", "knn")

#: At or below this many points the exact ``O(N^2)`` path runs. It is not a
#: fallback for a failure -- it is the same answer by a shorter route, and it
#: is what the bucket path is tested against.
_BRUTE_FORCE_MAX = 2048

#: Soft ceiling on how many candidate pairs one expansion materializes at
#: once. Bounds peak memory; cannot change the result.
_PAIR_CHUNK = 2_000_000

#: The mixed-radix bucket code must fit in an int64 with room to spare.
_MAX_CODE = 1 << 62

#: How many points a k-nearest-neighbour bucket is sized to hold, as a
#: fraction of ``k``. A ball of the radius holding ``k`` points fits inside a
#: cube of about ``0.24 k`` points, so at this factor one 3^d block already
#: settles almost every query while carrying half the candidates a
#: ``k``-per-bucket lattice would. Measured on 200k uniform points at k=16:
#: 33.6 s at 1.0, 20.5 s at 0.5, 21.2 s at 0.3, 26.6 s at 0.25 -- below the
#: knee the queries that need a second, 5^d block cost more than the
#: candidates saved. The output is identical at every factor, which is the
#: point: the lattice is an accelerator, never part of the answer.
_KNN_FILL = 0.5


# --------------------------------------------------------------------------- #
# shared input handling                                                       #
# --------------------------------------------------------------------------- #
def _points_nd(points, prefix):
    """The coordinates as a contiguous ``(N, d)`` float64 array, ``d`` in 1..3.

    Unlike ``_interpolate._coords3`` nothing is padded to three columns: a
    2-D cloud keeps two columns, so its bucket grid has 9 neighbour offsets
    rather than 27 and its edge vectors have two components.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (1, 2, 3):
        raise ValueError(
            f"{prefix}points must be an (N, d) array with d in 1..3, got shape "
            f"{p.shape}"
        )
    return np.ascontiguousarray(p)


def _graph_positions(mesh, kind):
    """Graph-vertex positions: mesh points for the node graph, block-major cell
    centroids (the ``partition`` numbering) for the cell dual.

    The single owner of that rule; ``meshioplusplus.physicsnemo`` imports it
    rather than keeping a second copy.
    """
    if kind not in ("node", "cell"):
        raise ValueError(
            f"meshio++: proximity_graph: kind must be 'node' or 'cell', not {kind!r}"
        )
    points = np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float64))
    if kind == "node":
        return points
    from ._partition import _centroids
    from ._regions import block_bases

    total = block_bases(mesh.cells)[-1] if mesh.cells else 0
    dim = min(points.shape[1] if points.ndim == 2 else 0, 3)
    return np.ascontiguousarray(_centroids(mesh, total)[:, :dim])


def _box(box_size, dim, prefix):
    """``box_size`` validated into a ``(dim,)`` float64 array, or ``None``."""
    if box_size is None:
        return None
    b = np.asarray(box_size, dtype=np.float64).reshape(-1)
    if b.size == 1:
        b = np.repeat(b, dim)
    if b.size != dim:
        raise ValueError(
            f"{prefix}box_size must be one value or {dim} (the point dimension), "
            f"got {b.size}"
        )
    if not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError(
            f"{prefix}box_size must be positive and finite, got {b.tolist()}"
        )
    return np.ascontiguousarray(b)


def _min_image(disp, box):
    """The shortest image of a displacement across a periodic box, in place."""
    if box is None:
        return disp
    return disp - box * np.round(disp / box)


# --------------------------------------------------------------------------- #
# edge features                                                               #
# --------------------------------------------------------------------------- #
def edge_vectors(points, edge_index, *, box_size=None):
    """The MeshGraphNet edge-feature convention: displacement, then its norm.

    ``(E, d + 1)`` float64: ``points[row] - points[col]`` (source minus
    destination -- the stable upstream convention, and the one
    :func:`meshioplusplus.physicsnemo.graph_sample` already emits) followed by
    the Euclidean length. With ``box_size`` the displacement is the minimum
    image, so an edge across a periodic boundary is short rather than
    box-length.
    """
    prefix = "meshio++: edge_vectors: "
    pos = _points_nd(points, prefix)
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"{prefix}edge_index must be (2, E), got shape {ei.shape}")
    n = len(pos)
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise ValueError(f"{prefix}edge_index refers to a point outside 0..{n - 1}")
    box = _box(box_size, pos.shape[1], prefix)
    row, col = ei
    disp = _min_image(pos[row] - pos[col], box)
    norm = np.linalg.norm(disp, axis=-1, keepdims=True)
    return np.ascontiguousarray(np.concatenate((disp, norm), axis=-1))


# --------------------------------------------------------------------------- #
# the bucket grid                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Buckets:
    """Points bucketed onto a lattice of side ``h``, sorted by bucket code."""

    ijk: np.ndarray  # (N, d) int64 bucket coordinates, already wrapped
    nb: np.ndarray  # (d,) int64 per-axis bucket count (the mixed radix)
    side: np.ndarray  # (d,) float64 actual cell side, >= the requested h
    order: np.ndarray  # (N,) point ids sorted by code
    ucodes: np.ndarray  # (B,) ascending occupied codes
    starts: np.ndarray  # (B,) first slot of each occupied bucket in `order`
    counts: np.ndarray  # (B,) how many points it holds
    periodic: bool


def _code(ijk, nb):
    """Mixed-radix bucket code, x fastest (the lattice convention repo-wide)."""
    out = ijk[..., 0].astype(np.int64, copy=True)
    stride = np.int64(1)
    for axis in range(1, ijk.shape[-1]):
        stride = stride * np.int64(nb[axis - 1])
        out = out + stride * ijk[..., axis]
    return out


def _build_buckets(points, h, box, prefix):
    d = points.shape[1]
    if box is None:
        lo = points.min(axis=0)
        side = np.full(d, float(h))
        ijk = np.floor((points - lo) / h).astype(np.int64)
        ijk -= ijk.min(axis=0)
        nb = ijk.max(axis=0) + 1
        periodic = False
    else:
        # A periodic axis is divided into a WHOLE number of cells, each at
        # least `h` wide. Sizing them at exactly `h` instead would leave a
        # short partial cell at the seam, and two points either side of it
        # could then be within `h` of each other while sitting two buckets
        # apart -- which the single-offset neighbourhood does not look at.
        nb = np.maximum(np.floor(box / h).astype(np.int64), 1)
        side = box / nb
        ijk = np.floor(points / side).astype(np.int64) % nb
        periodic = True
    span = 1
    for axis in range(d):
        span *= int(nb[axis])
        if span > _MAX_CODE:
            raise ValueError(
                f"{prefix}the search lattice would need more than 2^62 cells "
                f"(the cloud spans {int(nb[0])} or more cells on one axis); the "
                "radius is tiny relative to the cloud's extent"
            )
    code = _code(ijk, nb)
    order = np.argsort(code, kind="stable")
    scode = code[order]
    starts = np.flatnonzero(np.concatenate(([True], scode[1:] != scode[:-1])))
    ucodes = scode[starts]
    counts = np.diff(np.concatenate((starts, [scode.size]))).astype(np.int64)
    return _Buckets(ijk, nb, side, order, ucodes, starts, counts, periodic)


def _offsets(d, radius=1):
    """Every integer offset within Chebyshev distance ``radius``, ascending."""
    rng = range(-radius, radius + 1)
    return np.array(list(itertools.product(*([rng] * d)))[::1], dtype=np.int64)


def _pairs_for_offset(buckets, qijk, offset, chunk=_PAIR_CHUNK):
    """Yield ``(qi, j)`` chunks: query row ``qi`` paired with every point ``j``
    in the bucket at ``offset`` from that query's own bucket."""
    nk = qijk + offset
    if buckets.periodic:
        nk = nk % buckets.nb
        valid = np.ones(len(nk), dtype=bool)
    else:
        valid = np.all((nk >= 0) & (nk < buckets.nb), axis=1)
        if not valid.any():
            return
        nk = np.where(valid[:, None], nk, 0)
    ncode = _code(nk, buckets.nb)
    pos = np.searchsorted(buckets.ucodes, ncode)
    hit = valid & (pos < buckets.ucodes.size)
    pos = np.where(hit, pos, 0)
    hit &= buckets.ucodes[pos] == ncode
    qi = np.flatnonzero(hit)
    if qi.size == 0:
        return
    cnt = buckets.counts[pos[qi]]
    st = buckets.starts[pos[qi]]
    csum = np.cumsum(cnt)
    lo = 0
    while lo < qi.size:
        base = int(csum[lo - 1]) if lo else 0
        hi = int(np.searchsorted(csum, base + chunk, side="left")) + 1
        hi = min(max(hi, lo + 1), qi.size)
        c, s = cnt[lo:hi], st[lo:hi]
        total = int(c.sum())
        off = np.repeat(np.cumsum(c) - c, c)
        t = np.arange(total, dtype=np.int64) - off
        yield np.repeat(qi[lo:hi], c), buckets.order[np.repeat(s, c) + t]
        lo = hi


# --------------------------------------------------------------------------- #
# radius                                                                      #
# --------------------------------------------------------------------------- #
def _brute_force_radius(points, radius, box):
    disp = _min_image(points[None, :, :] - points[:, None, :], box)
    d2 = np.einsum("ijk,ijk->ij", disp, disp)
    keep = d2 <= radius * radius
    keep &= np.triu(np.ones(keep.shape, dtype=bool), k=1)
    return np.nonzero(keep)


def _radius_pairs(points, radius, box):
    """Every pair within ``radius``, as ``(i, j)`` with ``i < j``."""
    n = len(points)
    if n <= _BRUTE_FORCE_MAX:
        return _brute_force_radius(points, radius, box)
    prefix = "meshio++: proximity_graph: "
    buckets = _build_buckets(points, radius, box, prefix)
    r2 = radius * radius
    lo_out, hi_out = [], []
    for offset in _offsets(points.shape[1]):
        for qi, j in _pairs_for_offset(buckets, buckets.ijk, offset):
            keep = qi < j
            if not keep.any():
                continue
            qi, j = qi[keep], j[keep]
            disp = _min_image(points[qi] - points[j], box)
            d2 = np.einsum("ij,ij->i", disp, disp)
            keep = d2 <= r2
            if keep.any():
                lo_out.append(qi[keep])
                hi_out.append(j[keep])
    if not lo_out:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lo_out), np.concatenate(hi_out)


# --------------------------------------------------------------------------- #
# k nearest neighbours                                                        #
# --------------------------------------------------------------------------- #
def _brute_force_knn(points, k, box):
    disp = _min_image(points[None, :, :] - points[:, None, :], box)
    d2 = np.einsum("ijk,ijk->ij", disp, disp)
    np.fill_diagonal(d2, np.inf)
    j = np.argsort(d2, axis=1, kind="stable")[:, :k]
    q = np.repeat(np.arange(len(points), dtype=np.int64), k)
    return q, np.ascontiguousarray(j).reshape(-1)


def _knn_cell_size(points, k, prefix):
    """A bucket side holding about ``k`` points, so one 3^d block answers most
    queries -- ``_point_budget._grid_representatives``' sizing rule."""
    extent = points.max(axis=0) - points.min(axis=0)
    live = extent > 0.0
    if not np.any(live):
        raise ValueError(
            f"{prefix}every point sits at the same position; there are no "
            "distinct neighbours to find"
        )
    measure = float(np.prod(extent[live]))
    h = (measure / len(points) * max(k, 1) * _KNN_FILL) ** (1.0 / int(live.sum()))
    return max(h, float(extent[live].max()) * 1e-9)


def _knn_pairs(points, k, box):
    """The ``k`` nearest neighbours of every point, as directed ``(q, j)``.

    Exact: a query stops only once its k-th neighbour is closer than the
    block radius already searched, so nothing outside the block could have
    been nearer.
    """
    n = len(points)
    if n <= _BRUTE_FORCE_MAX:
        return _brute_force_knn(points, k, box)
    prefix = "meshio++: proximity_graph: "
    d = points.shape[1]
    h = _knn_cell_size(points, k, prefix)
    buckets = _build_buckets(points, h, box, prefix)
    pending = np.arange(n, dtype=np.int64)
    out_q, out_j = [], []
    block = 1
    while pending.size:
        offsets = _offsets(d, block)
        covered = block * float(buckets.side.min())
        again = []
        # Chunk the queries so one block's expansion stays bounded.
        per_query = max(1, int(buckets.counts.mean()) * len(offsets))
        step = max(1, _PAIR_CHUNK // per_query)
        for lo in range(0, pending.size, step):
            q_ids = pending[lo : lo + step]
            qijk = buckets.ijk[q_ids]
            qs, js = [], []
            for offset in offsets:
                for qi, j in _pairs_for_offset(buckets, qijk, offset):
                    qs.append(qi)
                    js.append(j)
            if not qs:
                again.append(q_ids)
                continue
            qi = np.concatenate(qs)
            j = np.concatenate(js)
            keep = q_ids[qi] != j
            qi, j = qi[keep], j[keep]
            disp = _min_image(points[q_ids[qi]] - points[j], box)
            d2 = np.einsum("ij,ij->i", disp, disp)
            order = np.lexsort((j, d2, qi))
            qi, j, d2 = qi[order], j[order], d2[order]
            if qi.size:
                uniq = np.concatenate(([True], (qi[1:] != qi[:-1]) | (j[1:] != j[:-1])))
                qi, j, d2 = qi[uniq], j[uniq], d2[uniq]
            starts = np.flatnonzero(np.concatenate(([True], qi[1:] != qi[:-1])))
            counts = np.diff(np.concatenate((starts, [qi.size])))
            rank = np.arange(qi.size, dtype=np.int64) - np.repeat(starts, counts)
            # A query is answered when it has k candidates and the k-th sits
            # inside the block already searched (so nothing outside can beat it).
            kth = np.full(q_ids.size, np.inf)
            enough = counts >= k
            kth[qi[starts[enough] + (k - 1)]] = d2[starts[enough] + (k - 1)]
            settled = kth <= covered * covered
            keep = settled[qi] & (rank < k)
            out_q.append(q_ids[qi[keep]])
            out_j.append(j[keep])
            unsettled = np.flatnonzero(~settled)
            if unsettled.size:
                again.append(q_ids[unsettled])
        pending = np.concatenate(again) if again else np.zeros(0, dtype=np.int64)
        if pending.size and np.all(block >= buckets.nb.max()):
            # The block already covers the occupied lattice: whatever was found
            # is the whole cloud, so the answer is exact by exhaustion.
            for lo in range(0, pending.size, 4096):
                q_ids = pending[lo : lo + 4096]
                disp = _min_image(points[q_ids][:, None, :] - points[None, :, :], box)
                d2 = np.einsum("ijk,ijk->ij", disp, disp)
                d2[np.arange(q_ids.size), q_ids] = np.inf
                j = np.argsort(d2, axis=1, kind="stable")[:, :k]
                out_q.append(np.repeat(q_ids, k))
                out_j.append(np.ascontiguousarray(j).reshape(-1))
            break
        block *= 2
    if not out_q:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(out_q), np.concatenate(out_j)


# --------------------------------------------------------------------------- #
# proximity_graph                                                             #
# --------------------------------------------------------------------------- #
def proximity_graph(
    mesh_or_points,
    *,
    method: str = "radius",
    radius: Optional[float] = None,
    max_neighbors: Optional[int] = None,
    box_size=None,
    kind: str = "node",
    undirected: bool = True,
):
    """A graph over positions rather than connectivity.

    Parameters
    ----------
    mesh_or_points :
        a :class:`Mesh` (whose points, or with ``kind="cell"`` whose cell
        centroids, are the vertices) or a plain ``(N, d)`` array, ``d`` in 1..3.
    method :
        ``"radius"`` -- every pair closer than ``radius``; the physical rule
        for an interaction cutoff, and the one whose degree varies with local
        density. ``"knn"`` -- each point's ``max_neighbors`` nearest, then
        symmetrized, so degree is near-constant and a sparse region still has
        edges.
    radius :
        required by ``method="radius"``, refused by ``"knn"``.
    max_neighbors :
        required by ``method="knn"``, refused by ``"radius"``. Clamped to
        ``N - 1``.
    box_size :
        one value or one per axis: the periodic box. Pairs are linked by their
        minimum image. ``method="radius"`` refuses a radius above half the
        smallest side, where the minimum image is ambiguous.
    kind :
        ``"node"`` (mesh points) or ``"cell"`` (block-major cell centroids),
        as for :func:`meshioplusplus.edge_index`.
    undirected :
        ``True`` (default) emits both directions, lexsorted by (source,
        target); ``False`` emits each edge once with ``source < target``.

    Returns
    -------
    numpy.ndarray
        ``(2, E)`` C-contiguous int64.
    """
    from ._ml import _canonical_edges

    prefix = "meshio++: proximity_graph: "
    if method not in _METHODS:
        raise ValueError(
            f"{prefix}unknown method {method!r} (expected one of "
            f"{', '.join(_METHODS)})"
        )
    if kind not in ("node", "cell"):
        raise ValueError(f"{prefix}kind must be 'node' or 'cell', not {kind!r}")
    if hasattr(mesh_or_points, "points"):
        points = _points_nd(_graph_positions(mesh_or_points, kind), prefix)
    else:
        points = _points_nd(mesh_or_points, prefix)
    n = len(points)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{prefix}the positions contain a non-finite coordinate")
    box = _box(box_size, points.shape[1], prefix)
    if box is not None:
        points = np.ascontiguousarray(np.mod(points, box))

    if method == "radius":
        if max_neighbors is not None:
            raise ValueError(
                f"{prefix}max_neighbors belongs to method='knn'; "
                "method='radius' takes radius"
            )
        if radius is None or not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{prefix}method='radius' needs a positive radius")
        radius = float(radius)
        if box is not None and radius > 0.5 * float(box.min()):
            raise ValueError(
                f"{prefix}radius {radius} exceeds half the smallest box side "
                f"({0.5 * float(box.min())}); a pair then has two images at the "
                "same distance and the minimum image is ambiguous"
            )
        a, b = (
            _radius_pairs(points, radius, box)
            if n
            else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        )
    else:
        if radius is not None:
            raise ValueError(
                f"{prefix}radius belongs to method='radius'; method='knn' takes "
                "max_neighbors"
            )
        if max_neighbors is None or int(max_neighbors) < 1:
            raise ValueError(f"{prefix}method='knn' needs max_neighbors >= 1")
        k = min(int(max_neighbors), max(n - 1, 0))
        a, b = (
            _knn_pairs(points, k, box)
            if k
            else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        )
    return _canonical_edges(
        np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), undirected
    )


# --------------------------------------------------------------------------- #
# bistride multiscale hierarchy                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BistrideHierarchy:
    """A graph and its coarsenings, in the layout a BSMS model consumes.

    ``edges`` holds ``num_levels + 1`` edge sets, each ``(2, E_i)`` **in its own
    level's numbering**; ``ids`` holds ``num_levels`` arrays, ``ids[i]`` naming
    the rows of level ``i`` that survive into level ``i + 1`` (ascending). The
    two lengths differ by one deliberately -- that is the model's contract:
    every level has edges, every *transition* has a pooling map.
    """

    edges: tuple
    ids: tuple
    schema: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.edges)

    @property
    def num_levels(self) -> int:
        """How many coarsenings; ``len(self.edges) - 1``."""
        return len(self.edges) - 1


def _adjacency(edge_index, n):
    """CSR row pointers plus the (sorted) neighbour column array."""
    src, dst = edge_index
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = np.searchsorted(src, np.arange(n + 1, dtype=np.int64))
    return indptr, dst


def _components(indptr, adjncy, n):
    """Connected-component labels, discovered from the lowest unvisited node."""
    label = np.full(n, -1, dtype=np.int64)
    current = 0
    for seed in range(n):
        if label[seed] >= 0:
            continue
        label[seed] = current
        frontier = np.array([seed], dtype=np.int64)
        while frontier.size:
            counts = indptr[frontier + 1] - indptr[frontier]
            total = int(counts.sum())
            if total == 0:
                break
            off = np.repeat(np.cumsum(counts) - counts, counts)
            t = np.arange(total, dtype=np.int64) - off
            nxt = adjncy[np.repeat(indptr[frontier], counts) + t]
            nxt = np.unique(nxt[label[nxt] < 0])
            label[nxt] = current
            frontier = nxt
        current += 1
    return label, current


def _parity_selection(indptr, adjncy, positions, rows):
    """The smaller of a component's two BFS parity classes (tie -> even).

    The BFS is seeded at the node nearest the component's own centroid, so the
    two classes are the geometric checkerboard rather than an artefact of the
    node numbering.
    """
    centre = positions[rows].mean(axis=0)
    seed = rows[int(np.argmin(np.linalg.norm(positions[rows] - centre, axis=1)))]
    depth = {int(seed): 0}
    frontier = np.array([seed], dtype=np.int64)
    level = 0
    while frontier.size:
        level += 1
        counts = indptr[frontier + 1] - indptr[frontier]
        total = int(counts.sum())
        if total == 0:
            break
        off = np.repeat(np.cumsum(counts) - counts, counts)
        t = np.arange(total, dtype=np.int64) - off
        nxt = adjncy[np.repeat(indptr[frontier], counts) + t]
        nxt = np.unique(nxt)
        nxt = nxt[[int(v) not in depth for v in nxt]] if nxt.size else nxt
        for v in nxt:
            depth[int(v)] = level
        frontier = nxt
    even = [v for v, lv in depth.items() if lv % 2 == 0]
    odd = [v for v, lv in depth.items() if lv % 2 == 1]
    if not odd:
        return even
    return even if len(even) <= len(odd) else odd


def _two_hop(edge_index, indptr, adjncy, n):
    """One- and two-hop reachability, the squared adjacency with self-loops.

    Self-loops before squaring are load-bearing: without them a kept node loses
    the neighbours that were dropped rather than inheriting their reach, and a
    path graph falls apart instead of coarsening.
    """
    src, dst = edge_index
    counts = indptr[dst + 1] - indptr[dst]
    total = int(counts.sum())
    if total:
        off = np.repeat(np.cumsum(counts) - counts, counts)
        t = np.arange(total, dtype=np.int64) - off
        far = adjncy[np.repeat(indptr[dst], counts) + t]
        near = np.repeat(src, counts)
        return np.concatenate((src, near)), np.concatenate((dst, far))
    return src, dst


def bistride_hierarchy(edge_index, positions, *, num_levels: int = 1):
    """Coarsen a graph by keeping every other BFS level (Cao et al.'s BSMS).

    Each level two-colours every connected component by BFS depth from the node
    nearest its centroid, keeps the smaller colour class, and joins the
    survivors wherever they were within two hops -- so a path of ``N`` nodes
    becomes a path of about ``N / 2``, and a message crosses the mesh in
    logarithmically many steps instead of linearly many.

    Returns a :class:`BistrideHierarchy`. Raises by name when a level would
    collapse below two nodes: a coarsening that leaves nothing to connect is a
    request for too many levels, not a graph to hand a model.
    """
    prefix = "meshio++: bistride_hierarchy: "
    num_levels = int(num_levels)
    if num_levels < 1:
        raise ValueError(f"{prefix}num_levels must be at least 1, got {num_levels}")
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2:
        raise ValueError(f"{prefix}positions must be (N, d), got shape {pos.shape}")
    n = len(pos)
    ei = np.ascontiguousarray(np.asarray(edge_index, dtype=np.int64))
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"{prefix}edge_index must be (2, E), got shape {ei.shape}")
    if ei.size and (ei.min() < 0 or ei.max() >= n):
        raise ValueError(
            f"{prefix}edge_index refers to a node outside 0..{n - 1}; positions "
            f"has {n} rows"
        )

    edges = [ei]
    ids = []
    sizes = [n]
    for level in range(num_levels):
        if n < 2:
            raise ValueError(
                f"{prefix}level {level} has {n} node(s); there is nothing left to "
                "coarsen -- ask for fewer levels"
            )
        indptr, adjncy = _adjacency(ei, n)
        label, ncomp = _components(indptr, adjncy, n)
        kept = []
        for c in range(ncomp):
            rows = np.flatnonzero(label == c)
            kept.extend(_parity_selection(indptr, adjncy, pos, rows))
        kept = np.unique(np.asarray(kept, dtype=np.int64))
        if kept.size < 2:
            raise ValueError(
                f"{prefix}coarsening level {level} leaves {kept.size} node(s), "
                "too few to connect -- ask for fewer levels"
            )
        src, dst = _two_hop(ei, indptr, adjncy, n)
        renumber = np.full(n, -1, dtype=np.int64)
        renumber[kept] = np.arange(kept.size, dtype=np.int64)
        a, b = renumber[src], renumber[dst]
        keep = (a >= 0) & (b >= 0) & (a != b)
        from ._ml import _canonical_edges

        ei = _canonical_edges(a[keep], b[keep], True)
        pos = pos[kept]
        n = kept.size
        ids.append(kept)
        edges.append(ei)
        sizes.append(n)
    schema = {
        "version": PROXIMITY_GRAPH_VERSION,
        "num_levels": num_levels,
        "num_nodes": [int(s) for s in sizes],
    }
    return BistrideHierarchy(tuple(edges), tuple(ids), schema)
