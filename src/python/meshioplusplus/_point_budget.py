"""Point-cloud budgets (``meshioplusplus.select_points`` / ``subsample_points``).

A transformer-shaped model -- Transolver, FLARE, DoMINO, any architecture whose
cost is quadratic or worse in the number of tokens -- does not take a mesh
either. It takes a *fixed number of points*, and a real surface mesh has a few
hundred thousand of them where the model wants a few thousand. Reducing a cloud
to a budget is the whole of the data path such a model needs before
:func:`meshioplusplus.feature_matrix` can build its token table, and it is what
this module is: a selection as a value, three ways to make one, and a mesh
carrying only the selected points.

Everything here is pure numpy over machinery that already exists. The
quantization behind the ``"grid"`` method is :class:`meshioplusplus.GridSpec`'s,
the region model is the mesh's own, and the recommended way to get the token
table is composition rather than a duplicate of ``feature_matrix``::

    budget = mio.select_points(mesh, 4096)
    tokens = mio.feature_matrix(mesh).matrix[budget.indices]    # (4096, F)

which preserves that function's column contract for free.

Selection order is information
------------------------------
A :class:`PointBudget`'s ``indices`` are kept **in selection order**, not sorted.
For ``"farthest"`` every prefix of the selection is itself the farthest-point
sample of that smaller budget, so one selection at 8192 serves 4096 and 2048 as
well by slicing. ``np.sort(budget.indices)`` gives index order when a consumer
wants it.

The three methods, and what they cost
-------------------------------------
``"farthest"`` is exact greedy farthest-point sampling: the first point is
``start``, every later one is the candidate farthest from everything selected so
far. It gives the most uniform coverage of any budget and it is ``O(N * count)``.
Measured, one core: 50k points / 1024 in 0.10 s, 200k / 2048 in 1.1 s, 1M / 4096
in 22.6 s. Fine to a few hundred thousand points; above that it is not wrong,
merely slow, and the table is here so the wait is not a surprise.

``"grid"`` is the scalable one. The cloud is quantized onto a coarse lattice
(sized so that the occupied cells outnumber the budget a few times over), each
occupied cell contributes the point nearest its centre, and farthest-point
sampling then runs over those representatives only. The quantization is
``O(N)`` and the sampling ``O(count^2)``: the same three sizes take 0.04 s,
0.15 s and 0.8 s. Coverage is near-uniform rather than optimal, since within one
cell the representative is the closest-to-centre point rather than the farthest
one -- on the 56k-point Stanford bunny at 2048 points the largest gap left
uncovered is 3.2 length units against ``"farthest"``'s 2.7.

``"random"`` is the baseline, ``O(count)``, and it is here so the other two can
be measured against it: the same draw from the bunny leaves a gap of 13.8, five
times either of the others. A uniform random draw from a surface cloud leaves
visible holes at every budget the other two fill.

Seeding follows the library's one convention: a keyword-only ``seed`` fed to a
locally constructed ``random.Random``, applied to a deterministically ordered
input. ``"farthest"`` and ``"grid"`` are deterministic given ``start``; ``seed``
reaches them only through ``start=None``, which draws the first point.

Public API:

* :class:`PointBudget` -- a selection as a value.
* :func:`select_points` -- mesh -> budget.
* :func:`subsample_points` -- mesh -> a mesh of only the selected points.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._grid import DEFAULT_MAX_CELLS
from ._grid_transfer import GridSpec
from ._interop import _emit
from ._mesh import Mesh
from ._regions import Region

__all__ = [
    "POINT_BUDGET_VERSION",
    "BUDGET_ID_NAME",
    "PointBudget",
    "select_points",
    "subsample_points",
]

#: Bumped when the recorded schema changes meaning.
POINT_BUDGET_VERSION = 1

#: The ``point_data`` array ``subsample_points(record_ids=True)`` attaches: each
#: selected point's index in the mesh it came from.
BUDGET_ID_NAME = "budget:original_point_id"

_METHODS = ("farthest", "grid", "random")

#: The ``"grid"`` method refines its lattice until this many occupied cells per
#: requested point exist, so the farthest-point pass has a choice to make.
_GRID_OVERSAMPLE = 2

#: ... and gives up refining after this many doublings (each one multiplies the
#: cell count by eight), which is when a cloud simply has too few distinct
#: positions for the budget.
_GRID_MAX_REFINEMENTS = 12


# --------------------------------------------------------------------------- #
# the selection as a value                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PointBudget:
    """Which points of a mesh made the budget, in the order they were picked.

    ``indices`` are int64 indices into the source mesh's points, **in selection
    order** (see the module docstring for why that order is kept), unique, and
    exactly ``count`` long. ``schema`` records how the selection was made --
    method, seed, start point, bounds, the lattice the ``"grid"`` method used --
    so a budget written into a model card can be reproduced.
    """

    indices: np.ndarray
    method: str
    count: int
    schema: dict = field(default_factory=dict)

    def __post_init__(self):
        idx = np.ascontiguousarray(np.asarray(self.indices, dtype=np.int64).reshape(-1))
        if self.method not in _METHODS:
            raise ValueError(
                f"meshio++: PointBudget: unknown method {self.method!r} "
                f"(expected one of {', '.join(_METHODS)})"
            )
        if int(self.count) != idx.size:
            raise ValueError(
                f"meshio++: PointBudget: count is {int(self.count)} but there are "
                f"{idx.size} indices"
            )
        if idx.size and idx.min() < 0:
            raise ValueError("meshio++: PointBudget: indices must be non-negative")
        if np.unique(idx).size != idx.size:
            raise ValueError("meshio++: PointBudget: indices must be unique")
        idx.flags.writeable = False
        object.__setattr__(self, "indices", idx)
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "schema", dict(self.schema))

    def __len__(self) -> int:
        return self.count

    def __eq__(self, other) -> bool:
        if not isinstance(other, PointBudget):
            return NotImplemented
        return (
            self.method == other.method
            and self.count == other.count
            and bool(np.array_equal(self.indices, other.indices))
            and self.schema == other.schema
        )

    def __hash__(self):
        return hash((self.method, self.count, self.indices.tobytes()))

    @property
    def sorted_indices(self) -> np.ndarray:
        """The same selection in ascending index order (a copy)."""
        return np.sort(self.indices)

    def take(self, values):
        """Gather the selected rows of a per-point array.

        ``budget.take(feature_matrix(mesh).matrix)`` is the token table; the
        rows come back in selection order, like ``indices``.
        """
        values = np.asarray(values)
        n = int(self.schema.get("num_points", -1))
        if n >= 0 and values.shape[:1] != (n,):
            raise ValueError(
                f"meshio++: PointBudget.take: expected a per-point array with {n} "
                f"rows, got shape {values.shape}"
            )
        return values[self.indices]

    def prefix(self, count: int) -> "PointBudget":
        """The first ``count`` selected points, as a budget of their own.

        Meaningful for ``"farthest"`` -- and for ``"grid"``, whose second stage
        is farthest-point sampling -- where a prefix of the selection *is* the
        selection at that smaller budget. For ``"random"`` it is merely another
        uniform draw.
        """
        count = int(count)
        if not 1 <= count <= self.count:
            raise ValueError(
                f"meshio++: PointBudget.prefix: count must be between 1 and "
                f"{self.count}, got {count}"
            )
        schema = dict(self.schema)
        schema["count"] = count
        schema["prefix_of"] = self.count
        return PointBudget(self.indices[:count], self.method, count, schema)

    def to_dict(self) -> dict:
        """snake_case, for a model card or a JSON report; ``indices`` included."""
        doc = {"version": POINT_BUDGET_VERSION, "method": self.method}
        doc.update(self.schema)
        doc["version"] = POINT_BUDGET_VERSION
        doc["count"] = self.count
        doc["indices"] = [int(i) for i in self.indices]
        return doc

    @classmethod
    def from_dict(cls, doc: dict) -> "PointBudget":
        try:
            indices = doc["indices"]
            method = doc["method"]
        except KeyError as exc:
            raise ValueError(
                f"meshio++: PointBudget.from_dict: missing key {exc.args[0]!r} "
                "(indices and method are required)"
            ) from None
        schema = {k: v for k, v in doc.items() if k != "indices"}
        return cls(indices, method, len(indices), schema)


# --------------------------------------------------------------------------- #
# the three selectors (all over a local (M, 3) candidate array)               #
# --------------------------------------------------------------------------- #
def _fps(points, count, first, prefix):
    """Exact greedy farthest-point sampling; ``O(len(points) * count)``.

    Ties break on the lowest index (``argmax``'s rule), so the selection is a
    function of the input order alone. Raises when the cloud runs out of
    *distinct* positions before the budget is met -- selecting a point at
    distance zero would be selecting one already chosen.

    The coordinates are split into three contiguous 1-D arrays and every step
    is an in-place ufunc over them: the same arithmetic as
    ``np.sum((points - points[nxt]) ** 2, axis=1)`` (identical results, checked)
    at a sixth of the cost, since the axis-1 reduction over an ``(N, 3)`` array
    is what dominated a row-wise version.
    """
    n = len(points)
    order = np.empty(count, dtype=np.int64)
    order[0] = first
    x = np.ascontiguousarray(points[:, 0])
    y = np.ascontiguousarray(points[:, 1])
    z = np.ascontiguousarray(points[:, 2])
    d2 = (x - x[first]) ** 2 + (y - y[first]) ** 2 + (z - z[first]) ** 2
    bx = np.empty(n, dtype=np.float64)
    by = np.empty(n, dtype=np.float64)
    bz = np.empty(n, dtype=np.float64)
    for i in range(1, count):
        nxt = int(np.argmax(d2))
        if d2[nxt] <= 0.0:
            raise ValueError(
                f"{prefix}the cloud has only {i} distinct positions among its "
                f"{n} candidates, fewer than the requested {count}; lower the count "
                "or weld the duplicates first"
            )
        order[i] = nxt
        np.subtract(x, x[nxt], out=bx)
        np.multiply(bx, bx, out=bx)
        np.subtract(y, y[nxt], out=by)
        np.multiply(by, by, out=by)
        np.subtract(z, z[nxt], out=bz)
        np.multiply(bz, bz, out=bz)
        np.add(bx, by, out=bx)
        np.add(bx, bz, out=bx)
        np.minimum(d2, bx, out=d2)
    return order


def _grid_representatives(points, count, prefix, max_cells):
    """One point per occupied lattice cell (the nearest to the cell's centre).

    The lattice starts at about ``_GRID_OVERSAMPLE * count`` cells over the
    cloud's box and is refined by doubling until the *occupied* cells outnumber
    the budget by that factor -- a surface cloud occupies far fewer cells than
    its box holds, so the first guess is usually too coarse. Returns the local
    indices of the representatives, their cell ids, and the spec.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = hi - lo
    live = extent > 0.0
    if not np.any(live):
        raise ValueError(
            f"{prefix}every candidate sits at the same position; there is nothing "
            "to select among"
        )
    # Isotropic cells sized so the box holds about the target cell count, over
    # the axes that have any extent; a degenerate axis gets a single cell.
    target = float(_GRID_OVERSAMPLE * count)
    measure = float(np.prod(extent[live]))
    h = (measure / target) ** (1.0 / int(live.sum()))
    h = max(h, float(extent[live].max()) * 1e-12)
    for _ in range(_GRID_MAX_REFINEMENTS + 1):
        dims = np.where(live, np.ceil(extent / h), 1).astype(np.int64)
        dims = np.maximum(dims, 1)
        if int(np.prod(dims)) > max_cells:
            break
        spacing = np.where(live, h, 1.0)
        spec = GridSpec(origin=lo, spacing=spacing, dims=dims)
        ijk = np.floor((points - lo) / spacing).astype(np.int64)
        ijk = np.minimum(np.maximum(ijk, 0), dims - 1)
        cell = ijk[:, 0] + dims[0] * (ijk[:, 1] + dims[1] * ijk[:, 2])
        centre = lo + (ijk.astype(np.float64) + 0.5) * spacing
        d2 = np.sum((points - centre) ** 2, axis=1)
        order = np.lexsort((d2, cell))
        sorted_cells = cell[order]
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_cells[1:] != sorted_cells[:-1]))
        )
        reps = order[starts]
        if len(reps) >= _GRID_OVERSAMPLE * count or len(reps) == len(points):
            return reps, cell, spec
        h *= 0.5
    if len(reps) >= count:
        return reps, cell, spec
    raise ValueError(
        f"{prefix}the cloud occupies only {len(reps)} distinct lattice cells at "
        f"the finest lattice tried ({int(dims[0])}x{int(dims[1])}x{int(dims[2])}), "
        f"fewer than the requested {count}; lower the count, raise max_cells, or "
        "use method='farthest', which works on the points directly"
    )


def _points_3d(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (1, 2, 3):
        raise ValueError(
            "meshio++: select_points: points must be an (N, d) array with d in "
            f"1..3, got shape {p.shape}"
        )
    if p.shape[1] < 3:
        p = np.column_stack([p, np.zeros((len(p), 3 - p.shape[1]))])
    return np.ascontiguousarray(p)


# --------------------------------------------------------------------------- #
# select_points                                                               #
# --------------------------------------------------------------------------- #
def select_points(
    mesh,
    count: int,
    *,
    method: str = "farthest",
    seed: int = 0,
    start: Optional[int] = 0,
    bounds=None,
    max_cells: int = DEFAULT_MAX_CELLS,
) -> PointBudget:
    """Choose ``count`` of a mesh's points under a token budget.

    Parameters
    ----------
    mesh :
        a :class:`Mesh`, or a plain ``(N, d)`` point array (``d`` in 1..3).
    count :
        how many points to select; at most the number of candidates.
    method :
        ``"farthest"`` (exact farthest-point sampling, ``O(N * count)``),
        ``"grid"`` (lattice representatives then farthest-point sampling over
        them, ``O(N + count^2)``) or ``"random"`` (``O(count)``). See the module
        docstring for what each costs and gives.
    seed :
        drives ``"random"``, and ``start=None``.
    start :
        the index (into the mesh's points) of the first selected point for
        ``"farthest"`` and ``"grid"``; ``None`` draws it from ``seed``. Ignored
        by ``"random"``.
    bounds :
        ``(xlo, ylo, zlo, xhi, yhi, zhi)``: only points inside this box are
        candidates. The returned indices are still into the *whole* mesh.
    max_cells :
        the ``"grid"`` method's lattice ceiling, as for ``voxelize``.

    Returns
    -------
    PointBudget
        ``count`` unique indices in selection order, plus the schema.
    """
    prefix = "meshio++: select_points: "
    points = _points_3d(mesh.points if hasattr(mesh, "points") else mesh)
    n = len(points)
    if method not in _METHODS:
        raise ValueError(
            f"{prefix}unknown method {method!r} (expected one of "
            f"{', '.join(_METHODS)})"
        )
    count = int(count)
    if count < 1:
        raise ValueError(f"{prefix}count must be at least 1, got {count}")

    bounds_doc = None
    if bounds is None:
        candidates = np.arange(n, dtype=np.int64)
    else:
        b = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if b.size != 6:
            raise ValueError(
                f"{prefix}bounds must be six numbers (xlo, ylo, zlo, xhi, yhi, "
                f"zhi), got {b.size}"
            )
        lo, hi = b[:3], b[3:]
        if np.any(hi < lo):
            raise ValueError(f"{prefix}bounds have hi < lo on some axis: {b.tolist()}")
        inside = np.all((points >= lo) & (points <= hi), axis=1)
        candidates = np.flatnonzero(inside).astype(np.int64)
        bounds_doc = [float(v) for v in b]
    m = len(candidates)
    if count > m:
        where = "inside the bounds" if bounds is not None else "in the mesh"
        raise ValueError(
            f"{prefix}count is {count} but only {m} points are {where}; lower the "
            "count" + (" or widen the bounds" if bounds is not None else "")
        )
    local = points[candidates]

    # The first point, as a local index into the candidates.
    rng = random.Random(seed)
    first_local = None
    if method in ("farthest", "grid"):
        if start is None:
            first_local = rng.randrange(m)
        else:
            start = int(start)
            pos = int(np.searchsorted(candidates, start))
            if not (0 <= start < n) or pos >= m or candidates[pos] != start:
                where = "a candidate inside the bounds" if bounds else "a point index"
                raise ValueError(
                    f"{prefix}start={start} is not {where} (the mesh has {n} " "points)"
                )
            first_local = pos

    schema = {
        "version": POINT_BUDGET_VERSION,
        "method": method,
        "count": count,
        "seed": int(seed),
        "start": None,
        "bounds": bounds_doc,
        "num_points": n,
        "num_candidates": m,
    }
    if method == "farthest":
        order = _fps(local, count, first_local, prefix)
        schema["start"] = int(candidates[first_local])
    elif method == "grid":
        reps, cell, spec = _grid_representatives(local, count, prefix, max_cells)
        # The first point is the representative of the cell `start` lands in.
        rep_of_cell = {int(cell[r]): i for i, r in enumerate(reps)}
        first_rep = rep_of_cell[int(cell[first_local])]
        order = reps[_fps(local[reps], count, first_rep, prefix)]
        schema["start"] = int(candidates[first_local])
        schema["grid"] = spec.to_dict()
        schema["num_representatives"] = int(len(reps))
    else:
        order = np.asarray(rng.sample(range(m), count), dtype=np.int64)
    return PointBudget(candidates[order], method, count, schema)


# --------------------------------------------------------------------------- #
# subsample_points                                                            #
# --------------------------------------------------------------------------- #
def subsample_points(mesh, budget, *, record_ids: bool = False, **select_kwargs):
    """A mesh carrying only the budgeted points, as a point cloud.

    ``budget`` is a :class:`PointBudget` or a count, in which case
    :func:`select_points` is called with the remaining keyword arguments.

    What survives: the selected points (in selection order), every per-point
    ``point_data`` array gathered to them, ``field_data``, and every **Point**
    region remapped. What does not: the cells, ``cell_data`` and Cell/Side
    regions -- a point cloud has no cells, and that is the operation rather
    than a limitation, so each drop is one warning naming what went. The
    output gets a single ``vertex`` block so every writer accepts it.
    ``record_ids=True`` attaches ``budget:original_point_id``.
    """
    prefix = "meshio++: subsample_points: "
    if isinstance(budget, PointBudget):
        if select_kwargs:
            raise ValueError(
                f"{prefix}selection keywords {sorted(select_kwargs)} were given "
                "alongside an already-made budget; pass a count to select here"
            )
    else:
        budget = select_points(mesh, budget, **select_kwargs)
    n = len(mesh.points)
    idx = budget.indices
    if idx.size and int(idx.max()) >= n:
        raise ValueError(
            f"{prefix}the budget refers to point {int(idx.max())} but the mesh "
            f"has {n} points; it was made for a different mesh"
        )
    k = int(idx.size)

    points = np.asarray(mesh.points)[idx]
    point_data = {}
    for name, value in mesh.point_data.items():
        value = np.asarray(value)
        if value.shape[:1] == (n,):
            point_data[name] = value[idx]
        else:
            point_data[name] = value.copy()
    if record_ids:
        point_data[BUDGET_ID_NAME] = idx.astype(np.int64)

    notes = []
    num_cells = sum(len(cb) for cb in mesh.cells)
    if num_cells:
        notes.append(
            f"dropped {num_cells} cells in {len(mesh.cells)} blocks (a point cloud "
            "has no cells)"
        )
    if mesh.cell_data:
        notes.append(f"dropped cell_data {sorted(mesh.cell_data)} with the cells")

    inverse = np.full(n, -1, dtype=np.int64)
    inverse[idx] = np.arange(k, dtype=np.int64)
    regions = []
    dropped = []
    for region in getattr(mesh, "regions", []):
        if region.kind != "point":
            dropped.append(f"{region.name} ({region.kind})")
            continue
        mapped = inverse[region.entries]
        regions.append(
            Region(region.name, "point", mapped[mapped >= 0], region.dim, region.tag)
        )
    if dropped:
        notes.append(f"dropped cell/side regions {dropped} with the cells")
    _emit("subsample_points", notes)

    field_data = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in mesh.field_data.items()
    }
    return Mesh(
        points,
        [("vertex", np.arange(k, dtype=np.int64).reshape(-1, 1))],
        point_data=point_data,
        field_data=field_data,
        regions=regions,
    )
