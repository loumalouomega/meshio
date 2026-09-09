"""Curved-cell tessellation (``meshioplusplus.tessellate``).

Roadmap section 1's last open bullet: isoparametric subdivision of the five
serendipity/Lagrange quadratic cell types onto a regular refinement lattice --
a curved mesh cell has no single "position", so a value on it (a model
prediction, a training target) can only be attached to a *linearized*
approximation, and nothing in this repo built that approximation with a
recorded map back to where each piece came from. ``convert_cells(simplexify)``
already linearizes and splits conformingly, but drops the mid-side/face/body
nodes' curvature entirely (it keeps corners only) and records no per-simplex
provenance; this module keeps the curvature (through the cell's own
isoparametric shape functions) and records exactly which original point or
cell every tessellated point/cell descends from.

Everything here is pure numpy over existing machinery (``_convert_cells``'s
tables, ``_data_average``'s cell measures, ``_regions.block_bases``) -- no
C++, ABI, WASM or binding change, matching every other module in this
section (``_point_budget.py``, ``_proximity.py``, ``_guard.py``).

Scope: five curved types
-------------------------
``CURVED_TYPES`` is exactly ``("triangle6", "quad8", "quad9", "tetra10",
"hexahedron27")`` -- the serendipity/full-Lagrange quadratics this repo's own
tables (``_convert_cells._ELEVATE``, ``_refine_templates.QUAD_FACES``) already
describe a basis for. **``hexahedron20`` is deliberately excluded**: its
faces are ``quad8`` while ``hexahedron27``'s are ``quad9`` -- they disagree on
the face *interior* (quad9 has a face-centre node, quad8 does not), and
curving both would let two neighbouring cells' watertightness keys silently
merge two different face-interior positions into one point. Every other
cell (linear types, ``hexahedron20``, ``wedge15``/``18``, ``pyramid13``/``14``,
the VTK-Lagrange family, ragged/polyhedron blocks) is a **no-op**: it passes
through into the output mesh completely unchanged -- same points (by index),
same connectivity, same ``cell_data`` row -- which is what makes
``tessellate(linear_mesh)`` an exact identity (the "no-op on a linear mesh"
oracle). An earlier design considered simplexifying every cell (curved or
not) for one uniform simplex output mesh; the plan explicitly leaves that
choice open, but it was rejected here because it would make the no-op oracle
false by construction (an all-linear mesh would still change cell types) for
no benefit the roadmap bullet asked for.

The lattice, and why it needs no external library
--------------------------------------------------
``levels`` is **divisions per axis**, not applications of a doubling
operator (unlike :func:`meshioplusplus.refine`, whose own "levels" *is*
``2**levels`` divisions) -- picked because a curved cell's own tessellation
resolution is naturally small (2-4) and an exponential blow-up per level
would make the parameter surprising. Two lattice families, chosen per base
shape:

* **Tensor bases** (``quad``, ``hexahedron``): a plain ``(n+1)``-point-per-axis
  grid at ``t = linspace(-1, 1, n+1)``, whose endpoints are *exactly*
  ``-1.0``/``1.0`` in floating point -- what makes the boundary
  classification below exact rather than tolerance-based.
* **Simplex bases** (``triangle``, ``tetra``): built by recursively applying
  the *same* fixed same-type subdivision template :func:`meshioplusplus.refine`
  already documents for a single reference cell -- triangle -> 4 children
  (three corner triangles plus one central one, unambiguous, no diagonal
  choice exists for a triangle) and tetra -> 8 children (four corner
  tetrahedra plus the central octahedron split along the fixed *interior*
  diagonal between the midpoints of edges (0,1) and (2,3), in the exact ring
  order CLAUDE.md records for :func:`meshioplusplus.refine`'s own template:
  mid(0,2) -> mid(0,3) -> mid(1,3) -> mid(1,2)). Every point this produces is
  an exact dyadic rational with denominator dividing ``2**levels``, so a
  simple ``round(coord * 2**levels)`` integer key deduplicates points created
  by sibling recursive calls with no floating-point tolerance anywhere.

Every corner/edge-midpoint/face-centre/body-centre node's *parametric*
position is **derived**, never hardcoded, from three tables this repo
already owns and tests: ``_REFERENCE_CORNERS`` (four standard reference
elements, chosen so the barycentric solve below is an exact identity matrix
-- see ``_barycentric``), ``_convert_cells._ELEVATE`` (which edge is mid-node
``k``, the exact contract :func:`meshioplusplus.convert_cells`'s own
``"elevate"`` mode already commits to) and ``_refine_templates.QUAD_FACES``
(which face is centre-node ``k``, for ``quad9``/``hexahedron27``). No
permutation constant is written down anywhere in this module; a wrong
convention in either upstream table would show up here as a wrong node
position, not as a silent local relabelling.

Isoparametric shape functions
------------------------------
Evaluated at each lattice point in **reference space**, then applied to the
curved cell's own *real* (possibly curved) node positions -- the standard
isoparametric map ``x(xi) = sum_i N_i(xi) * X_i``. Three families:

* ``quad9``/``hexahedron27`` -- full tensor-product Lagrange (a genuine
  bubble/body node exists, so this is a plain product of per-axis 1-D
  quadratic factors, no serendipity correction needed).
* ``quad8`` -- serendipity (8 nodes, no bubble): the standard closed-form
  corner/edge-mid functions.
* ``triangle6``/``tetra10`` -- barycentric quadratics
  (``N_i = lambda_i (2 lambda_i - 1)`` at a corner, ``N_ij = 4 lambda_i
  lambda_j`` at the midpoint of edge ``(i, j)``), with ``lambda`` solved
  *generically* from ``_REFERENCE_CORNERS`` (an identity solve for this
  module's own reference corners, but written so a different reference
  choice would still work).

Watertightness: point identity across cells
---------------------------------------------
A lattice point's identity is a function of **which original entity it lies
on**, classified from its own **corner weights** (the same weights the
isoparametric map already computes) rather than from coordinate comparisons:
count how many of a point's corner weights are exactly nonzero. All nonzero
means interior; one nonzero corner means the point *is* that corner (a
kept, original mesh point); two nonzero corners means an edge point; and
for ``tetra10`` three nonzero (one zero) means a triangular-face point, for
``hexahedron27`` four nonzero out of eight means a quad-face point. This
needs no lookup table for "which edge/face is this" at all -- the nonzero
corners *are* the edge/face, addressed by their own global point ids.

The resulting key is what deduplicates a point across every cell that
touches it:

* an original point -- its own global id (no key needed, ``source_point`` is
  set directly).
* ``(edge, min(ga, gb), max(ga, gb), t')`` for an edge point, ``t'`` the
  quantized fractional distance from the *smaller*-id endpoint -- a fraction
  is intrinsic to the pair, not to which of the two cells (or which
  direction) computed it.
* ``(face, (c0, c1, c2[, c3]), u', v')`` for a ``tetra10`` triangular face
  point or a ``hexahedron27`` quad face point, where ``(c0, c1, ...)`` is a
  **canonical traversal** of the face's corner ids: start at the globally
  smallest id, then continue toward whichever of its two cyclic neighbours
  has the smaller id. That rule is a property of the abstract cycle of ids
  alone (the "which neighbour is smaller" question has one answer
  regardless of which cell, or which direction, listed the face), so two
  cells describing the same physical face always compute the same
  ``(c0, c1, c2, c3)`` -- proved, not merely tested, by the fact that
  "neighbour in a cycle" is a symmetric relation. ``u'``/``v'`` are then
  ``w(c1)`` and ``w(c2)`` (triangular face) or ``w(c1) + w(c2)`` and
  ``w(c2) + w(c3)`` (quad face), where ``w(g)`` is *this cell's own* local
  weight for the corner whose global id is ``g`` -- reindexing the SAME
  weights by global id rather than by local position is what makes the pair
  invariant under the cell's own arbitrary local numbering (the two
  opposite corners of a bilinear patch are preserved by every rotation and
  reflection of a face's own cyclic order, so summing the pair keyed this
  way gives the identical value from either side). ``u'``/``v'`` are
  quantized to integers (``round(u' * n)``) before hashing, exact because
  they are themselves dyadic rationals of denominator ``n``.
* ``(interior, global_cell, ...)`` for an interior point -- always unique to
  one cell (no neighbour ever needs to agree with it), so no
  canonicalization is needed at all.

**Output cell shape**: a curved cell's linearized *base* determines its
output. ``triangle6``/``tetra10`` refine directly into more triangles/tetra
(the recursive template above never produces anything else). ``quad8``/
``quad9`` refine into a quad lattice, then split into two triangles per
sub-quad by the same "smallest id, then diagonal to the corner two steps
away" rule -- provably direction-independent for a 4-cycle (2 steps is its
own inverse), so no consistency issue exists for a 2-D mesh anyway (two 2-D
cells only ever share an *edge*, never a face, so this diagonal never has to
agree with a neighbour's; it is used purely for a deterministic single
output). ``hexahedron27`` refines into a hex lattice; each sub-hexahedron
becomes 12 tetrahedra via one new interior "apex" point (this sub-cell's own
centroid -- an interior key, never shared) plus, for each of its 6 quad
faces, the identical "smallest id, diagonal two steps away" split into 2
triangles, each combined with the apex into one tetrahedron. This is the
one place a genuine cross-cell consistency requirement exists (two
hexahedra can share a whole quad *face*), and it is why the diagonal rule is
keyed by **global** ids rather than a cell's own local numbering (which a
merely fixed local diagonal -- the ``convert_cells(simplexify)`` precedent
-- does not generally guarantee agrees between two independently-numbered
neighbours): see the sabotage test in ``tests/python/test_tessellation.py``
for the two-hexahedra fixture this specifically targets.

Fields
------
``fields=True`` (default) interpolates every input ``point_data`` array onto
the output mesh's points (component-wise, using each output point's own
weights against its parent cell's real node values -- exactly
:meth:`Tessellation.gather`'s own formula) and replicates every input
``cell_data`` row onto its cell's children (a plain gather by
``source_cell``, since ``cell_data`` is piecewise-constant and has no
interpolant to speak of). ``field_data`` copies through unchanged.
``fields=False`` skips all of this, leaving only the ``tessellate:*``
bookkeeping arrays -- useful when a caller wants to attach a *different*
array later (a model's own prediction) via :meth:`Tessellation.scatter`/
:meth:`Tessellation.aggregate` rather than the input's own fields.

Public API:

* :func:`tessellate` -- mesh -> :class:`Tessellation`.
* :class:`Tessellation` -- the mesh plus its provenance arrays, with
  ``gather``/``scatter``/``aggregate``/``measures`` and ``from_mesh`` to
  reconstruct one after a file round trip.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from ._convert_cells import _ELEVATE, _LINEAR_BASE, _NUM_CORNERS
from ._data_average import _cell_measures
from ._interop import _emit
from ._mesh import Mesh
from ._refine_templates import QUAD_FACES
from ._regions import Region, block_bases

__all__ = [
    "TESSELLATION_VERSION",
    "SOURCE_POINT_NAME",
    "SOURCE_CELL_NAME",
    "SUB_INDEX_NAME",
    "STENCIL_NAME",
    "WEIGHTS_NAME",
    "CURVED_TYPES",
    "Tessellation",
    "tessellate",
]

#: Bumped when the recorded schema or array contract changes meaning.
TESSELLATION_VERSION = 1

#: Int64 ``point_data``: the original mesh point index a tessellated point
#: was kept from, or ``-1`` for a synthetic (mid-side/face/body/lattice) one.
SOURCE_POINT_NAME = "tessellate:source_point"
#: Int64 ``cell_data``: the GLOBAL block-major index (see
#: ``_regions.block_bases``) of the original cell a tessellated cell
#: descends from.
SOURCE_CELL_NAME = "tessellate:source_cell"
#: Int64 ``cell_data``: a tessellated cell's index among its own source
#: cell's children (0 for a pass-through, non-curved cell).
SUB_INDEX_NAME = "tessellate:sub_index"
#: Int64 ``(P, K)`` ``point_data``, opt-in (``record_stencil=True``): every
#: tessellated point's interpolation stencil into the *source* mesh's own
#: points, padded with ``-1`` past however many entries a point actually
#: uses.
STENCIL_NAME = "tessellate:stencil"
#: Float64 ``(P, K)`` ``point_data``, opt-in: the matching interpolation
#: weights (padding entries carry weight ``0.0``).
WEIGHTS_NAME = "tessellate:weights"

#: The five curved types this module knows a basis for -- see the module
#: docstring for why ``hexahedron20`` is deliberately not among them.
CURVED_TYPES = ("triangle6", "quad8", "quad9", "tetra10", "hexahedron27")

#: The maximum stencil width across every curved type (hexahedron27's own 27
#: nodes) -- the fixed ``K`` of the ``(P, K)`` stencil/weights arrays.
_MAX_STENCIL = 27

# --------------------------------------------------------------------------- #
# reference elements -- hand-written, never derived from anything else       #
# --------------------------------------------------------------------------- #
#: Standard reference elements, chosen so ``_barycentric``'s solve below is
#: an exact identity matrix for the two simplex bases (no rounding at all
#: beyond the input coordinates' own).
_REFERENCE_CORNERS = {
    "triangle": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    "quad": np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
    "tetra": np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ),
    "hexahedron": np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    ),
}

#: curved type -> its linear base (a narrowed view of ``_LINEAR_BASE``).
_BASE_OF = {t: _LINEAR_BASE[t] for t in CURVED_TYPES}
_TENSOR_BASES = ("quad", "hexahedron")
_SIMPLEX_BASES = ("triangle", "tetra")
#: curved types whose node layout includes ``QUAD_FACES``' face-centre nodes.
_INCLUDES_FACES = ("quad9", "hexahedron27")
_INCLUDES_BODY = ("hexahedron27",)
_NUM_NODES = {"triangle6": 6, "quad8": 8, "quad9": 9, "tetra10": 10, "hexahedron27": 27}


def _node_coords(curved_type):
    """Parametric coordinate of every node of ``curved_type``, corners first
    then edge-midpoints (``_convert_cells._ELEVATE``'s own contract) then,
    for ``quad9``/``hexahedron27``, face-centres (``QUAD_FACES``' rows) and a
    body centre. Derived, never hardcoded -- see the module docstring."""
    base = _BASE_OF[curved_type]
    corners = _REFERENCE_CORNERS[base]
    _, _, edges = _ELEVATE[base]
    coords = [corners[i] for i in range(len(corners))]
    for a, b in edges:
        coords.append(0.5 * (corners[a] + corners[b]))
    if curved_type in _INCLUDES_FACES:
        for face in QUAD_FACES[base]:
            coords.append(np.mean(corners[list(face)], axis=0))
    if curved_type in _INCLUDES_BODY:
        coords.append(np.mean(corners, axis=0))
    out = np.asarray(coords, dtype=np.float64)
    assert out.shape[0] == _NUM_NODES[curved_type]
    return out


_NODE_COORDS = {t: _node_coords(t) for t in CURVED_TYPES}


# --------------------------------------------------------------------------- #
# isoparametric shape functions                                              #
# --------------------------------------------------------------------------- #
def _barycentric(corners, xi):
    """``xi`` -> barycentric weights against ``corners`` (a simplex).

    Generic (never hardcoded to a specific numeric layout): solves
    ``sum_i L_i corners[i] = xi``, ``sum_i L_i = 1``. An identity solve for
    this module's own reference corners, but correct for any simplex.
    """
    corners = np.asarray(corners, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    origin = corners[0]
    t = (corners[1:] - origin).T
    tinv = np.linalg.inv(t)
    rel = xi - origin
    lam_rest = rel @ tinv.T
    lam0 = 1.0 - lam_rest.sum(axis=1, keepdims=True)
    return np.concatenate([lam0, lam_rest], axis=1)


def _corner_weights(base, xi):
    """``xi`` (``(M, dim)``) -> ``(M, ncorners)`` weight of each of the
    base's own corners -- barycentric for a simplex, the standard
    bilinear/trilinear tensor-product weight for a quad/hexahedron. Used
    both for the isoparametric map (corners are the lowest-order term) and,
    more importantly, for classifying a lattice point by which corners have
    a nonzero weight (see the module docstring's "Watertightness" section).
    """
    if base in _SIMPLEX_BASES:
        return _barycentric(_REFERENCE_CORNERS[base], xi)
    corners = _REFERENCE_CORNERS[base]
    xi = np.asarray(xi, dtype=np.float64)
    w = np.ones((xi.shape[0], corners.shape[0]), dtype=np.float64)
    for c in range(corners.shape[0]):
        for a in range(corners.shape[1]):
            if corners[c, a] > 0:
                w[:, c] *= (1.0 + xi[:, a]) * 0.5
            else:
                w[:, c] *= (1.0 - xi[:, a]) * 0.5
    return w


def _lagrange1d(t, xi_val):
    """The 1-D quadratic Lagrange factor at nodal position ``xi_val`` in
    ``{-1, 0, 1}``, evaluated at ``t`` (array-like, reference coordinate in
    ``[-1, 1]``)."""
    if xi_val < -0.5:
        return 0.5 * t * (t - 1.0)
    if xi_val > 0.5:
        return 0.5 * t * (t + 1.0)
    return 1.0 - t * t


def _tensor_basis(curved_type, xi):
    """Full tensor-product Lagrange basis (``quad9``/``hexahedron27``): a
    genuine bubble/body node exists at every axis combination, so no
    serendipity correction is needed -- just the product of each axis's
    own factor."""
    coords = _NODE_COORDS[curved_type]
    xi = np.asarray(xi, dtype=np.float64)
    out = np.ones((xi.shape[0], coords.shape[0]), dtype=np.float64)
    for k in range(coords.shape[0]):
        for a in range(coords.shape[1]):
            out[:, k] *= _lagrange1d(xi[:, a], coords[k, a])
    return out


def _serendipity_quad8_basis(xi):
    """The standard 8-node serendipity quad basis (no interior/9th node)."""
    coords = _NODE_COORDS["quad8"]
    xi = np.asarray(xi, dtype=np.float64)
    x, y = xi[:, 0], xi[:, 1]
    out = np.empty((xi.shape[0], 8), dtype=np.float64)
    for k in range(8):
        xk, yk = coords[k]
        if abs(xk) < 0.5:
            out[:, k] = 0.5 * (1.0 - x * x) * (1.0 + y * yk)
        elif abs(yk) < 0.5:
            out[:, k] = 0.5 * (1.0 + x * xk) * (1.0 - y * y)
        else:
            out[:, k] = 0.25 * (1.0 + x * xk) * (1.0 + y * yk) * (x * xk + y * yk - 1.0)
    return out


def _simplex_basis(curved_type, xi):
    """Barycentric quadratic basis (``triangle6``/``tetra10``): corner
    ``N_i = lambda_i (2 lambda_i - 1)``, edge-midpoint ``N_ij = 4 lambda_i
    lambda_j``."""
    base = _BASE_OF[curved_type]
    corners = _REFERENCE_CORNERS[base]
    lam = _barycentric(corners, xi)
    ncorners = corners.shape[0]
    _, _, edges = _ELEVATE[base]
    out = np.empty((xi.shape[0], ncorners + len(edges)), dtype=np.float64)
    for i in range(ncorners):
        out[:, i] = lam[:, i] * (2.0 * lam[:, i] - 1.0)
    for k, (a, b) in enumerate(edges):
        out[:, ncorners + k] = 4.0 * lam[:, a] * lam[:, b]
    return out


def _shape_functions(curved_type, xi):
    """``xi`` (``(M, dim)``) -> ``(M, nnodes)`` shape function values."""
    if curved_type == "quad8":
        return _serendipity_quad8_basis(xi)
    if curved_type in ("quad9", "hexahedron27"):
        return _tensor_basis(curved_type, xi)
    return _simplex_basis(curved_type, xi)


# --------------------------------------------------------------------------- #
# reference lattices                                                          #
# --------------------------------------------------------------------------- #
def _triangle_full_split(c):
    """One triangle's fixed same-type split into 4 (``refine``'s own
    template): three corner triangles plus one central one. No diagonal
    ambiguity exists for a triangle at all."""
    m01 = 0.5 * (c[0] + c[1])
    m12 = 0.5 * (c[1] + c[2])
    m20 = 0.5 * (c[2] + c[0])
    return [
        (c[0], m01, m20),
        (c[1], m12, m01),
        (c[2], m20, m12),
        (m01, m12, m20),
    ]


def _tetra_full_split(c):
    """One tetrahedron's fixed same-type split into 8 (``refine``'s own
    template): four corner tetrahedra plus the central octahedron split
    along the fixed interior diagonal mid(0,1)-mid(2,3), fanned in the ring
    order CLAUDE.md records (mid(0,2) -> mid(0,3) -> mid(1,3) -> mid(1,2)).
    Strictly interior to the parent, so this choice never has to agree with
    a neighbour."""
    m01 = 0.5 * (c[0] + c[1])
    m12 = 0.5 * (c[1] + c[2])
    m02 = 0.5 * (c[0] + c[2])
    m03 = 0.5 * (c[0] + c[3])
    m13 = 0.5 * (c[1] + c[3])
    m23 = 0.5 * (c[2] + c[3])
    return [
        (c[0], m01, m02, m03),
        (c[1], m01, m12, m13),
        (c[2], m02, m12, m23),
        (c[3], m03, m13, m23),
        (m01, m23, m02, m03),
        (m01, m23, m03, m13),
        (m01, m23, m13, m12),
        (m01, m23, m12, m02),
    ]


def _recursive_lattice(base, levels):
    """``levels`` applications of a simplex base's own same-type split,
    starting from the reference corners. Points dedup exactly: every
    coordinate is a dyadic rational of denominator dividing ``2**levels``.
    """
    splitter = _triangle_full_split if base == "triangle" else _tetra_full_split
    scale = 2 ** max(levels, 0)
    points = []
    index_of = {}

    def get_index(p):
        key = tuple(int(round(v * scale)) for v in p)
        idx = index_of.get(key)
        if idx is None:
            idx = len(points)
            index_of[key] = idx
            points.append(np.asarray(p, dtype=np.float64))
        return idx

    corners = _REFERENCE_CORNERS[base]
    cells = [tuple(get_index(corners[i]) for i in range(len(corners)))]
    for _ in range(levels):
        new_cells = []
        for cell in cells:
            corner_pts = [points[i] for i in cell]
            for child in splitter(corner_pts):
                new_cells.append(tuple(get_index(c) for c in child))
        cells = new_cells
    return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int64)


def _tensor_lattice(base, levels):
    """A plain ``(n+1)``-point-per-axis grid, ``n = levels``, for a tensor
    base (quad/hexahedron); still-hexahedron/quad-shaped sub-cells."""
    n = max(levels, 1)
    t = np.linspace(-1.0, 1.0, n + 1)
    if base == "quad":
        grids = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")

        def idx(i, j):
            return i + (n + 1) * j

        pts = np.stack([t[grids[0].ravel()], t[grids[1].ravel()]], axis=1)
        cells = []
        for i in range(n):
            for j in range(n):
                cells.append(
                    [idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                )
        return pts, np.asarray(cells, dtype=np.int64)

    grids = np.meshgrid(
        np.arange(n + 1), np.arange(n + 1), np.arange(n + 1), indexing="ij"
    )

    def idx3(i, j, k):
        return i + (n + 1) * (j + (n + 1) * k)

    pts = np.stack(
        [t[grids[0].ravel()], t[grids[1].ravel()], t[grids[2].ravel()]], axis=1
    )
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                cells.append(
                    [
                        idx3(i, j, k),
                        idx3(i + 1, j, k),
                        idx3(i + 1, j + 1, k),
                        idx3(i, j + 1, k),
                        idx3(i, j, k + 1),
                        idx3(i + 1, j, k + 1),
                        idx3(i + 1, j + 1, k + 1),
                        idx3(i, j + 1, k + 1),
                    ]
                )
    return pts, np.asarray(cells, dtype=np.int64)


def _reference_lattice(curved_type, levels):
    """``(xi_points (M, dim), cells (K, ncorners))`` in reference space --
    still same-base-shaped output cells (triangles/tets stay so; quads and
    hexahedra are simplexified separately, one level up, since only the
    hexahedron case needs the cross-cell diagonal rule)."""
    base = _BASE_OF[curved_type]
    if base in _SIMPLEX_BASES:
        return _recursive_lattice(base, levels)
    return _tensor_lattice(base, levels)


# --------------------------------------------------------------------------- #
# the smallest-id-then-two-steps-away diagonal, and cyclic canonicalization  #
# --------------------------------------------------------------------------- #
def _quad_diagonal_split(gids):
    """A quad's 4 (global-id) corners -> 2 triangles, split by the diagonal
    from the smallest id to the corner two cyclic steps away -- direction-
    independent for a 4-cycle (2 steps is its own inverse under reversal),
    so this needs no orientation reasoning at all."""
    p = int(np.argmin(gids))
    a, b, c, d = (gids[(p + k) % 4] for k in range(4))
    return [(a, b, c), (a, c, d)]


def _canonical_cycle(gids):
    """Canonicalize a cyclic id sequence: start at the smallest id, then walk
    toward whichever neighbour has the smaller id. A property of the
    abstract cycle alone, so any two cells listing the same physical
    face/ring -- in any rotation or direction -- compute the same result.
    Returns ``gids`` reordered canonically (a new list)."""
    n = len(gids)
    start = int(np.argmin(gids))
    nxt = gids[(start + 1) % n]
    prv = gids[(start - 1) % n]
    step = 1 if nxt < prv else -1
    return [gids[(start + step * k) % n] for k in range(n)]


# --------------------------------------------------------------------------- #
# Tessellation                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tessellation:
    """A curved mesh's isoparametric tessellation, plus its provenance.

    ``mesh`` is the tessellated (linear) output. ``source_point``/
    ``source_cell``/``sub_index`` mirror the ``tessellate:*`` arrays already
    attached to ``mesh``'s ``point_data``/``cell_data`` (kept as separate
    fields too, since :meth:`gather`/:meth:`aggregate` read them far more
    than any file writer does). ``stencil``/``weights`` are always populated
    in memory (needed by :meth:`gather`) regardless of whether
    ``tessellate``'s ``record_stencil`` also *persisted* them onto ``mesh``.
    """

    #: the tessellated mesh.
    mesh: Mesh
    #: ``(P,)`` int64: original point index, or -1 for a synthetic point.
    source_point: np.ndarray
    #: ``(C,)`` int64: the GLOBAL (block-major) source cell of each output
    #: cell, in the SAME block-major order as ``mesh.cells``.
    source_cell: np.ndarray
    #: ``(C,)`` int64: a cell's index among its own source cell's children.
    sub_index: np.ndarray
    #: ``(P, K)`` int64: each output point's interpolation stencil into the
    #: SOURCE mesh's own points (-1 padding).
    stencil: np.ndarray
    #: ``(P, K)`` float64: matching interpolation weights (0.0 padding).
    weights: np.ndarray
    #: JSON-serializable: version, levels, curved, num_source_points/cells,
    #: per-type counts, watertightness diagnostics.
    schema: dict = field(default_factory=dict)

    def __post_init__(self):
        source_point = np.ascontiguousarray(
            np.asarray(self.source_point, dtype=np.int64).reshape(-1)
        )
        source_cell = np.ascontiguousarray(
            np.asarray(self.source_cell, dtype=np.int64).reshape(-1)
        )
        sub_index = np.ascontiguousarray(
            np.asarray(self.sub_index, dtype=np.int64).reshape(-1)
        )
        stencil = np.ascontiguousarray(np.asarray(self.stencil, dtype=np.int64))
        weights = np.ascontiguousarray(np.asarray(self.weights, dtype=np.float64))
        npoints = len(self.mesh.points)
        ncells = sum(len(cb) for cb in self.mesh.cells)
        if source_point.shape != (npoints,):
            raise ValueError(
                f"meshio++: Tessellation: source_point has {source_point.shape[0]} "
                f"entries but the mesh has {npoints} points"
            )
        if source_cell.shape != (ncells,) or sub_index.shape != (ncells,):
            raise ValueError(
                "meshio++: Tessellation: source_cell/sub_index must have one "
                f"entry per cell ({ncells}), got {source_cell.shape[0]} and "
                f"{sub_index.shape[0]}"
            )
        if stencil.shape[0] != npoints or weights.shape != stencil.shape:
            raise ValueError(
                "meshio++: Tessellation: stencil/weights must be (num_points, K) "
                f"and agree with each other, got {stencil.shape} vs {weights.shape} "
                f"against {npoints} points"
            )
        for name, arr in (
            ("source_point", source_point),
            ("source_cell", source_cell),
            ("sub_index", sub_index),
            ("stencil", stencil),
            ("weights", weights),
        ):
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "schema", dict(self.schema))

    # ------------------------------------------------------------------ #
    def gather(self, values) -> np.ndarray:
        """Gather a per-source-point array onto the tessellated points.

        One expression once ``values`` is reshaped to ``(n, C)``:
        ``(values[stencil] * weights[..., None]).sum(1)``. Padding entries
        (``weight == 0``) never let a padded, possibly non-finite row
        contribute -- their product is forced to exactly 0 rather than
        trusted to already be, since ``0 * nan`` is ``nan``.
        """
        values = np.asarray(values)
        n = int(self.schema.get("num_source_points", -1))
        if n >= 0 and values.shape[0] != n:
            raise ValueError(
                f"meshio++: Tessellation.gather: expected {n} source-point rows, "
                f"got {values.shape[0]}"
            )
        flat = values.reshape(values.shape[0], -1)
        safe = np.where(self.stencil < 0, 0, self.stencil)
        contribution = flat[safe] * self.weights[..., None]
        contribution = np.where(self.weights[..., None] == 0.0, 0.0, contribution)
        out = contribution.sum(axis=1)
        return out.reshape((len(self.mesh.points),) + values.shape[1:])

    def scatter(self, values) -> np.ndarray:
        """The reverse of :meth:`gather`: write per-tessellated-point values
        back onto the SOURCE mesh's own points via ``source_point`` (a
        synthetic point, ``source_point == -1``, is dropped). Where several
        tessellated points share one source point (an original corner
        touched by several cells), their values are averaged -- a stated,
        pinned rule, since a corner receives one contribution per incident
        cell and there is no reason to prefer any one of them. A source
        point touched by nothing (should not normally happen -- every
        source point is at least a corner of some cell) comes back NaN.
        """
        values = np.asarray(values)
        n = int(self.schema.get("num_source_points", -1))
        if n < 0:
            raise ValueError(
                "meshio++: Tessellation.scatter: schema carries no "
                "num_source_points -- was this reconstructed from a stale mesh?"
            )
        if values.shape[0] != len(self.mesh.points):
            raise ValueError(
                "meshio++: Tessellation.scatter: expected one row per "
                f"tessellated point ({len(self.mesh.points)}), got {values.shape[0]}"
            )
        flat = values.reshape(values.shape[0], -1)
        kept = self.source_point >= 0
        total = np.zeros((n,) + flat.shape[1:], dtype=np.float64)
        count = np.zeros(n, dtype=np.float64)
        np.add.at(total, self.source_point[kept], flat[kept])
        np.add.at(count, self.source_point[kept], 1.0)
        with np.errstate(invalid="ignore"):
            out = np.where(
                count[:, None] > 0, total / np.maximum(count, 1)[:, None], np.nan
            )
        return out.reshape((n,) + values.shape[1:])

    def aggregate(self, values, reduction: str = "mean"):
        """Per-source-cell aggregation of a per-tessellated-cell array.

        ``values`` is one row per OUTPUT cell (block-major, matching
        ``source_cell``). ``reduction`` is ``"mean"`` (plain average over a
        source cell's children), ``"weighted_mean"`` (weighted by each
        child's own :meth:`measures` -- the natural choice for a physically
        additive quantity) or ``"first"`` (the first child in
        ``sub_index`` order, e.g. for something that does not vary across
        children). Returns ``(unique_source_cells, aggregated_values)``.
        """
        if reduction not in ("mean", "weighted_mean", "first"):
            raise ValueError(
                "meshio++: Tessellation.aggregate: reduction must be one of "
                f"'mean', 'weighted_mean', 'first', got {reduction!r}"
            )
        values = np.asarray(values)
        if values.shape[0] != self.source_cell.shape[0]:
            raise ValueError(
                "meshio++: Tessellation.aggregate: expected one row per "
                f"tessellated cell ({self.source_cell.shape[0]}), got "
                f"{values.shape[0]}"
            )
        flat = values.reshape(values.shape[0], -1)
        unique = np.unique(self.source_cell)
        remap = {int(g): i for i, g in enumerate(unique)}
        groups = np.asarray([remap[int(g)] for g in self.source_cell], dtype=np.int64)
        if reduction == "first":
            out = np.zeros((len(unique),) + flat.shape[1:], dtype=flat.dtype)
            order = np.argsort(self.sub_index, kind="stable")
            seen = np.zeros(len(unique), dtype=bool)
            for i in order:
                g = groups[i]
                if not seen[g]:
                    out[g] = flat[i]
                    seen[g] = True
        else:
            weight = (
                np.ones(flat.shape[0], dtype=np.float64)
                if reduction == "mean"
                else np.abs(self.measures())
            )
            total = np.zeros((len(unique),) + flat.shape[1:], dtype=np.float64)
            wsum = np.zeros(len(unique), dtype=np.float64)
            np.add.at(total, groups, flat * weight[:, None])
            np.add.at(wsum, groups, weight)
            with np.errstate(invalid="ignore"):
                out = np.where(
                    wsum[:, None] > 0,
                    total / np.maximum(wsum, 1e-300)[:, None],
                    np.nan,
                )
        return unique, out.reshape((len(unique),) + values.shape[1:])

    def measures(self) -> np.ndarray:
        """``|measure|`` (length/area/volume by topological dimension) of
        every cell of the tessellated mesh, block-major -- one row per
        entry of ``source_cell``. Reuses ``_data_average._cell_measures``,
        the same primitive :func:`meshioplusplus.data_integrate` weights
        totals with."""
        out = []
        for cb in self.mesh.cells:
            m = _cell_measures(self.mesh, cb)
            out.append(np.full(len(cb), np.nan) if m is None else m)
        return np.concatenate(out) if out else np.empty(0)

    def to_dict(self) -> dict:
        """snake_case, for a model card; the mesh itself is not included
        (a caller writes it separately -- this is the provenance record)."""
        doc = {"version": TESSELLATION_VERSION}
        doc.update(self.schema)
        return doc

    @classmethod
    def from_mesh(cls, tessellated: Mesh, source: Mesh) -> "Tessellation":
        """Reconstruct a :class:`Tessellation` from a mesh already carrying
        the ``tessellate:*`` arrays (e.g. after a file round trip), against
        the source mesh it was built from.

        Mirrors ``_refine.py``'s ``refine:entity`` stale-key guard: if
        ``STENCIL_NAME``/``WEIGHTS_NAME`` are present, every stencil is
        checked to still reproduce its own point's coordinates (a weighted
        sum against ``source``'s CURRENT points) -- a mesh moved or
        renumbered since the stencil was written fails this, and the
        stencil is warned about and dropped (degrading to an identity
        stencil for kept points, and an all-zero one for synthetic points,
        which :meth:`gather` then answers with 0 rather than a wrong,
        stale value) rather than trusted.
        """
        n = len(tessellated.points)
        source_point = tessellated.point_data.get(SOURCE_POINT_NAME)
        if source_point is None:
            raise ValueError(
                f"meshio++: Tessellation.from_mesh: the mesh carries no "
                f"{SOURCE_POINT_NAME!r} -- it was not produced by tessellate()"
            )
        source_point = np.asarray(source_point, dtype=np.int64).reshape(-1)

        def _concat_cell_data(name):
            blocks = tessellated.cell_data.get(name)
            if not blocks:
                return np.zeros(0, dtype=np.int64)
            return np.concatenate(
                [np.asarray(v, dtype=np.int64) for v in blocks if v is not None]
            )

        source_cell = _concat_cell_data(SOURCE_CELL_NAME)
        sub_index = _concat_cell_data(SUB_INDEX_NAME)

        stencil_raw = tessellated.point_data.get(STENCIL_NAME)
        weights_raw = tessellated.point_data.get(WEIGHTS_NAME)
        num_source = len(source.points)
        stencil = weights = None
        if stencil_raw is not None and weights_raw is not None:
            stencil = np.asarray(stencil_raw, dtype=np.int64).reshape(n, -1)
            weights = np.asarray(weights_raw, dtype=np.float64).reshape(n, -1)
            safe = np.where(stencil < 0, 0, stencil)
            src_pts = np.asarray(source.points, dtype=np.float64)
            contribution = src_pts[safe] * weights[..., None]
            contribution = np.where(weights[..., None] == 0.0, 0.0, contribution)
            recon = contribution.sum(axis=1)
            want = np.asarray(tessellated.points, dtype=np.float64)
            if not np.allclose(recon, want, atol=1e-8, rtol=1e-6):
                warnings.warn(
                    f"tessellate: ignoring {STENCIL_NAME!r}/{WEIGHTS_NAME!r}: they "
                    "no longer reproduce the mesh's own point positions, so the "
                    "source mesh was moved or renumbered since they were written.",
                    stacklevel=2,
                )
                stencil = weights = None
        if stencil is None:
            stencil = np.where(source_point >= 0, source_point, 0).reshape(n, 1)
            weights = np.where(source_point >= 0, 1.0, 0.0).reshape(n, 1)

        schema = {
            "version": TESSELLATION_VERSION,
            "num_source_points": num_source,
            "num_points": n,
            "stencil_width": int(stencil.shape[1]),
            "reconstructed": True,
        }
        return cls(
            tessellated, source_point, source_cell, sub_index, stencil, weights, schema
        )


# --------------------------------------------------------------------------- #
# building one curved cell's tessellation                                   #
# --------------------------------------------------------------------------- #
def _classify(base, weights_row, tol=1e-9):
    """A lattice point's own corner weights -> ``(kind, nonzero positions)``
    where ``kind`` is ``"corner"``, ``"edge"``, ``"face"`` or ``"interior"``.
    """
    nz = np.flatnonzero(np.abs(weights_row) > tol)
    if len(nz) == 1:
        return "corner", nz
    if len(nz) == 2:
        return "edge", nz
    if base == "hexahedron" and len(nz) == 4:
        return "face", nz
    if base == "tetra" and len(nz) == 3:
        return "face", nz
    return "interior", nz


class _CellLattice:
    """One curved cell's reference lattice + isoparametric evaluation,
    shared across all instances of a given ``(curved_type, levels)`` pair
    (the reference-space computation never depends on the specific cell)."""

    _cache = {}

    def __init__(self, curved_type, levels):
        self.curved_type = curved_type
        self.base = _BASE_OF[curved_type]
        self.xi, self.cells = _reference_lattice(curved_type, levels)
        self.corner_w = _corner_weights(self.base, self.xi)
        self.node_w = _shape_functions(curved_type, self.xi)
        self.n = 2 ** max(levels, 0) if self.base in _SIMPLEX_BASES else max(levels, 1)

    @classmethod
    def get(cls, curved_type, levels):
        key = (curved_type, levels)
        out = cls._cache.get(key)
        if out is None:
            out = cls(curved_type, levels)
            cls._cache[key] = out
        return out


def _resolve_point(
    lattice,
    local_idx,
    base_gids,
    node_ids,
    cell_global_id,
    point_dict,
    points_out,
    real_nodes,
):
    """One reference-lattice point of one curved cell -> its OUTPUT point
    index, deduplicated across every cell sharing its entity via the key
    scheme described in the module docstring. ``node_ids`` are the curved
    cell's own real mesh point ids (all ``curved_type``'s nodes, corners
    first) -- the stencil is built against these, never against
    ``base_gids`` (corners only), since a mid-side/face/body node's weight
    can be nonzero there too."""
    corner_w = lattice.corner_w[local_idx]
    kind, nz = _classify(lattice.base, corner_w)
    stencil = lattice.node_w[local_idx]
    keep = np.flatnonzero(np.abs(stencil) > 1e-12)
    n = lattice.n

    # Every key is a tuple of PLAIN INTEGERS, top-level-tagged 0/1/2/3
    # (point/edge/face/interior) -- deliberately never a string or a mix of
    # types at one tuple position, so the whole key space sorts directly
    # and totally under Python's default tuple order (no custom sort key
    # needed, and no risk of "point" gids being compared as strings, which
    # would break the natural ascending point order a "kept" point must
    # keep for the no-op-on-a-linear-mesh guarantee).
    if kind == "corner":
        key = (0, int(base_gids[nz[0]]))
    elif kind == "edge":
        ga, gb = int(base_gids[nz[0]]), int(base_gids[nz[1]])
        wa, wb = float(corner_w[nz[0]]), float(corner_w[nz[1]])
        if ga > gb:
            ga, gb, wb = gb, ga, wa
        key = (1, ga, gb, int(round(wb * n)))
    elif kind == "face":
        gids = [int(base_gids[i]) for i in nz]
        wmap = {int(base_gids[i]): float(corner_w[i]) for i in nz}
        canon = _canonical_cycle(gids)
        if len(canon) == 3:
            u = wmap[canon[1]]
            v = wmap[canon[2]]
        else:
            u = wmap[canon[1]] + wmap[canon[2]]
            v = wmap[canon[2]] + wmap[canon[3]]
        key = (2, len(canon)) + tuple(canon) + (int(round(u * n)), int(round(v * n)))
    else:
        key = (3, 0, int(cell_global_id)) + tuple(
            int(round(x * n)) for x in lattice.xi[local_idx]
        )

    out_idx = point_dict.get(key)
    if out_idx is None:
        physical = lattice.node_w[local_idx] @ real_nodes
        out_idx = len(points_out)
        point_dict[key] = out_idx
        points_out.append((physical, node_ids[keep].copy(), stencil[keep].copy()))
    return out_idx


def _tessellate_curved_cell(
    base_gids,
    node_ids,
    real_nodes,
    curved_type,
    levels,
    cell_global_id,
    point_dict,
    points_out,
):
    """One curved cell -> its output simplices (local reference indices
    resolved through ``point_dict``/``points_out``, the whole tessellation's
    shared dedup state). Returns ``(cells, arity)`` where ``arity`` is 3
    (triangle output) or 4 (tetra output)."""
    lattice = _CellLattice.get(curved_type, levels)
    base = lattice.base

    resolved = [
        _resolve_point(
            lattice,
            i,
            base_gids,
            node_ids,
            cell_global_id,
            point_dict,
            points_out,
            real_nodes,
        )
        for i in range(lattice.xi.shape[0])
    ]

    out_cells = []
    if base in _SIMPLEX_BASES:
        arity = 3 if base == "triangle" else 4
        for cell in lattice.cells:
            out_cells.append(tuple(resolved[i] for i in cell))
        return out_cells, arity

    if base == "quad":
        for cell in lattice.cells:
            gids = [resolved[i] for i in cell]
            out_cells.extend(_quad_diagonal_split(gids))
        return out_cells, 3

    # hexahedron: 12 tetrahedra per sub-hex via a fresh interior apex point.
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    for cell_i, cell in enumerate(lattice.cells):
        local_ids = list(cell)
        out_ids = [resolved[i] for i in local_ids]
        apex_xi = lattice.xi[local_ids].mean(axis=0)
        apex_stencil = lattice.node_w[local_ids].mean(axis=0)
        # Sub-kind 1 (vs. the plain interior point's 0) so the two never
        # collide despite sharing the top-level "interior" tag 3 -- see the
        # key-scheme note in `_resolve_point`.
        apex_key = (3, 1, int(cell_global_id), cell_i) + tuple(
            int(round(x * lattice.n)) for x in apex_xi
        )
        apex_idx = point_dict.get(apex_key)
        if apex_idx is None:
            keep = np.flatnonzero(np.abs(apex_stencil) > 1e-12)
            apex_phys = apex_stencil @ real_nodes
            apex_idx = len(points_out)
            point_dict[apex_key] = apex_idx
            points_out.append(
                (apex_phys, node_ids[keep].copy(), apex_stencil[keep].copy())
            )
        for face in faces:
            face_gids = [out_ids[i] for i in face]
            for tri in _quad_diagonal_split(face_gids):
                out_cells.append(tri + (apex_idx,))
    return out_cells, 4


def _cell_data_row(blocks, bases, global_cell):
    """The row of a (mesh-wide) ``cell_data`` array at a GLOBAL cell index,
    or ``None`` if that block's own entry is absent."""
    b = int(np.searchsorted(bases, global_cell, side="right") - 1)
    if b < 0 or b >= len(blocks) or blocks[b] is None:
        return None
    local = global_cell - int(bases[b])
    return np.asarray(blocks[b])[local]


def _watertight_report(out_cells):
    """Facet-use counts over the tetrahedral part of the OUTPUT mesh: an
    internal triangular facet must be used by exactly two tetrahedra, a
    boundary one by exactly one -- the diagnostic that would catch a wrong
    (non-canonical) diagonal choice at a shared hexahedron face.
    """
    facet_count = {}
    faces = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]
    for ctype, conn in out_cells:
        if ctype != "tetra":
            continue
        conn = np.asarray(conn)
        for row in conn:
            for f in faces:
                key = tuple(sorted(int(row[i]) for i in f))
                facet_count[key] = facet_count.get(key, 0) + 1
    boundary = sum(1 for v in facet_count.values() if v == 1)
    interior = sum(1 for v in facet_count.values() if v == 2)
    bad = sum(1 for v in facet_count.values() if v not in (1, 2))
    return {
        "num_boundary_facets": int(boundary),
        "num_interior_facets": int(interior),
        "num_facets_with_bad_count": int(bad),
    }


def _remap_regions_for_tessellate(mesh, point_dict, remap, out_cell_source):
    """Carry the input mesh's regions through, as far as they honestly can
    go: **Point** regions remap fully, via the point-identity dedup keys
    every kept point (a pass-through cell's node, or a curved cell's own
    corner) already goes through -- a global point never loses its
    identity regardless of which cell(s) reference it. **Cell** regions
    remap only for cells that stayed **pass-through** (a genuine 1:1
    correspondence); an entry naming a curved (tessellated) cell has no
    single output cell to become, so it is dropped -- silently would be
    wrong, so it is warned about, matching every other operation's
    ``warn_regions_dropped`` convention. **Side** regions are always
    dropped (a tessellated cell's facets have no correspondence with the
    original's at all).
    """
    if not getattr(mesh, "regions", None):
        return []

    point_gid_to_out = {}
    for key, old_idx in point_dict.items():
        if key[0] == 0:
            point_gid_to_out[key[1]] = int(remap[old_idx])

    # A cell's GLOBAL source id maps to itself as an output cell only for
    # pass-through blocks; curved-cell children carry the SAME source id
    # for every child, so more than one output cell shares it and there is
    # no 1:1 mapping to use.
    counts = {}
    global_to_single_output = {}
    out_cell_index = 0
    for src_ids in out_cell_source:
        for gid in src_ids:
            gid = int(gid)
            counts[gid] = counts.get(gid, 0) + 1
            global_to_single_output[gid] = out_cell_index
            out_cell_index += 1
    pass_through_cells = {gid for gid, c in counts.items() if c == 1}

    out_regions = []
    dropped = []
    for region in mesh.regions:
        if region.kind == "point":
            mapped = np.asarray(
                [
                    point_gid_to_out[int(p)]
                    for p in region.entries
                    if int(p) in point_gid_to_out
                ],
                dtype=np.int64,
            )
            out_regions.append(
                Region(region.name, "point", mapped, region.dim, region.tag)
            )
        elif region.kind == "cell":
            kept = [
                global_to_single_output[int(c)]
                for c in region.entries
                if int(c) in pass_through_cells
            ]
            if len(kept) != len(region.entries):
                dropped.append(f"{region.name} (cell, entries on a tessellated cell)")
            out_regions.append(
                Region(
                    region.name,
                    "cell",
                    np.asarray(kept, dtype=np.int64),
                    region.dim,
                    region.tag,
                )
            )
        else:
            dropped.append(f"{region.name} (side)")
    if dropped:
        _emit("tessellate", [f"dropped region entries {dropped}"])
    return out_regions


def tessellate(
    mesh,
    *,
    levels: int = 2,
    curved: bool = True,
    fields: bool = True,
    record_stencil: bool = False,
) -> Tessellation:
    """Isoparametric tessellation of a mesh's curved cells.

    Every cell whose type is in :data:`CURVED_TYPES` is subdivided onto a
    ``levels``-division-per-axis reference lattice mapped through its own
    isoparametric shape functions; every other cell (including linear types
    and the non-curved higher-order family -- ``hexahedron20``,
    ``wedge15``/``18``, ``pyramid13``/``14``, VTK-Lagrange, ragged and
    polyhedron blocks) passes through **unchanged**. ``curved=False``
    disables curved handling entirely, making the whole mesh a no-op.

    ``fields=True`` (default) interpolates every input ``point_data`` array
    onto the output via :meth:`Tessellation.gather`'s own formula, and
    broadcasts every ``cell_data`` row onto its cell's children.
    ``record_stencil=True`` additionally attaches ``tessellate:stencil``/
    ``tessellate:weights`` as real ``point_data`` (off by default: a
    ``(P, 27)`` array is expensive to persist), so a caller can later
    reconstruct a full :class:`Tessellation` (including :meth:`gather`)
    from a written-and-reread file via :meth:`Tessellation.from_mesh`.

    See the module docstring for the isoparametric bases, the lattice
    construction, and the watertightness key scheme.
    """
    levels = int(levels)
    if levels < 0:
        raise ValueError(f"meshio++: tessellate: levels must be >= 0, got {levels}")
    npoints = len(mesh.points)
    dim = mesh.points.shape[1] if npoints else 3
    bases = block_bases(mesh.cells)

    point_dict = {}
    points_out = []  # (physical_xyz, source_gids, source_weights)

    def resolve_kept(gid):
        key = (0, int(gid))
        idx = point_dict.get(key)
        if idx is None:
            idx = len(points_out)
            point_dict[key] = idx
            points_out.append(
                (
                    np.asarray(mesh.points[gid], dtype=np.float64),
                    np.array([int(gid)], dtype=np.int64),
                    np.array([1.0]),
                )
            )
        return idx

    pass_through_blocks = []  # (type, local_conn, src_ids, sub_index)
    tri_cells, tri_source, tri_sub = [], [], []
    tet_cells, tet_source, tet_sub = [], [], []
    num_curved = 0
    num_pass_through = 0

    for b, cb in enumerate(mesh.cells):
        ctype = cb.type
        if curved and ctype in CURVED_TYPES:
            num_curved += len(cb)
            base = _BASE_OF[ctype]
            ncorn = _NUM_CORNERS[base]
            conn = np.asarray(cb.data)
            for local_c in range(len(cb)):
                gid_global = int(bases[b] + local_c)
                node_ids = conn[local_c]
                real_nodes = np.asarray(mesh.points[node_ids], dtype=np.float64)
                base_gids = node_ids[:ncorn]
                cells, arity = _tessellate_curved_cell(
                    base_gids,
                    node_ids,
                    real_nodes,
                    ctype,
                    levels,
                    gid_global,
                    point_dict,
                    points_out,
                )
                dest_cells = tri_cells if arity == 3 else tet_cells
                dest_source = tri_source if arity == 3 else tet_source
                dest_sub = tri_sub if arity == 3 else tet_sub
                for local_child, cell in enumerate(cells):
                    dest_cells.append(cell)
                    dest_source.append(gid_global)
                    dest_sub.append(local_child)
        else:
            num_pass_through += len(cb)
            data = cb.data
            remapped = []
            if isinstance(data, list):
                for row in data:
                    remapped.append([resolve_kept(int(g)) for g in row])
            else:
                conn = np.asarray(data)
                for local_c in range(len(cb)):
                    remapped.append([resolve_kept(int(g)) for g in conn[local_c]])
                remapped = np.asarray(remapped, dtype=np.int64)
            src_ids = (bases[b] + np.arange(len(cb))).astype(np.int64)
            pass_through_blocks.append(
                (ctype, remapped, src_ids, np.zeros(len(cb), dtype=np.int64))
            )

    # -- assemble the output point table in a fully deterministic (sorted
    # key) order, independent of block traversal order. Every key is a
    # plain tuple of ints (see `_resolve_point`'s comment), so the default
    # tuple order sorts it directly -- and, load-bearing for the no-op
    # case, sorts "point" (kind 0) keys by ascending original gid exactly,
    # never as a string (which would corrupt a kept point's natural order,
    # e.g. placing gid 10 before gid 2). --
    ordered_keys = sorted(point_dict.keys())
    remap = np.full(len(points_out), -1, dtype=np.int64)
    out_points = np.empty((len(points_out), dim), dtype=np.float64)
    out_source_point = np.full(len(points_out), -1, dtype=np.int64)
    for new_idx, key in enumerate(ordered_keys):
        old_idx = point_dict[key]
        remap[old_idx] = new_idx
        out_points[new_idx] = points_out[old_idx][0]
        if key[0] == 0:
            out_source_point[new_idx] = key[1]

    stencil_arr = np.full((len(points_out), _MAX_STENCIL), -1, dtype=np.int64)
    weight_arr = np.zeros((len(points_out), _MAX_STENCIL), dtype=np.float64)
    for old_idx in range(len(points_out)):
        new_idx = remap[old_idx]
        gids, w = points_out[old_idx][1], points_out[old_idx][2]
        k = min(len(gids), _MAX_STENCIL)
        stencil_arr[new_idx, :k] = gids[:k]
        weight_arr[new_idx, :k] = w[:k]

    out_cells = []
    out_cell_source = []
    out_cell_sub = []
    for ctype, conn, src_ids, sub in pass_through_blocks:
        remapped_conn = (
            remap[conn]
            if not isinstance(conn, list)
            else [remap[np.asarray(r)] for r in conn]
        )
        out_cells.append((ctype, remapped_conn))
        out_cell_source.append(src_ids)
        out_cell_sub.append(sub)
    if tri_cells:
        conn = remap[np.asarray(tri_cells, dtype=np.int64)]
        out_cells.append(("triangle", conn))
        out_cell_source.append(np.asarray(tri_source, dtype=np.int64))
        out_cell_sub.append(np.asarray(tri_sub, dtype=np.int64))
    if tet_cells:
        conn = remap[np.asarray(tet_cells, dtype=np.int64)]
        out_cells.append(("tetra", conn))
        out_cell_source.append(np.asarray(tet_source, dtype=np.int64))
        out_cell_sub.append(np.asarray(tet_sub, dtype=np.int64))

    point_data = {SOURCE_POINT_NAME: out_source_point}
    cell_data = {
        SOURCE_CELL_NAME: list(out_cell_source),
        SUB_INDEX_NAME: list(out_cell_sub),
    }
    if record_stencil:
        point_data[STENCIL_NAME] = stencil_arr.copy()
        point_data[WEIGHTS_NAME] = weight_arr.copy()

    field_data = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in mesh.field_data.items()
    }

    if fields:
        safe_stencil = np.where(stencil_arr < 0, 0, stencil_arr)
        for name, value in mesh.point_data.items():
            value = np.asarray(value)
            flat = value.reshape(value.shape[0], -1)
            contribution = flat[safe_stencil] * weight_arr[..., None]
            contribution = np.where(weight_arr[..., None] == 0.0, 0.0, contribution)
            gathered = contribution.sum(axis=1).reshape(
                (out_points.shape[0],) + value.shape[1:]
            )
            point_data[name] = gathered
        for name, blocks in mesh.cell_data.items():
            new_blocks = []
            for src_ids in out_cell_source:
                rows = [_cell_data_row(blocks, bases, int(g)) for g in src_ids]
                new_blocks.append(
                    None if any(r is None for r in rows) else np.asarray(rows)
                )
            cell_data[name] = new_blocks

    regions = _remap_regions_for_tessellate(mesh, point_dict, remap, out_cell_source)

    out_mesh = Mesh(
        out_points,
        out_cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        regions=regions,
    )

    all_source_cell = (
        np.concatenate(out_cell_source)
        if out_cell_source
        else np.empty(0, dtype=np.int64)
    )
    all_sub_index = (
        np.concatenate(out_cell_sub) if out_cell_sub else np.empty(0, dtype=np.int64)
    )

    schema = {
        "version": TESSELLATION_VERSION,
        "levels": levels,
        "curved": bool(curved),
        "num_source_points": npoints,
        "num_source_cells": int(bases[-1]),
        "num_curved_source_cells": int(num_curved),
        "num_pass_through_source_cells": int(num_pass_through),
        "num_points": int(out_points.shape[0]),
        "num_cells": int(all_source_cell.shape[0]),
        "stencil_width": _MAX_STENCIL,
        "watertight": _watertight_report(out_cells),
    }
    return Tessellation(
        out_mesh,
        out_source_point,
        all_source_cell,
        all_sub_index,
        stencil_arr,
        weight_arr,
        schema,
    )
