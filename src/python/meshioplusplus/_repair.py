"""Surface repair beyond ``clean``: consistent orientation, hole filling and
non-manifold (pinched-vertex) splitting -- the three defects ``SurfaceQuality``
counts and ``clean`` does not touch.

A dependency-free mesh *operation* (not a file format). Adapted from NVIDIA
PhysicsNeMo's ``physicsnemo.mesh.repair`` (2.2: ``fix_orientation``,
``fill_holes`` and the generator's private ``split_pinched_vertices``);
algorithms only, rewritten over meshio++'s own machinery. Two deliberate
divergences from upstream: orientation is propagated by the topological
half-edge rule (two triangles sharing an edge agree iff they traverse it in
opposite directions), which is exact where upstream's normal dot test
mis-orients across a crease sharper than 90 degrees; and a hole's fan is wound
to agree with the surrounding surface, where upstream winds it from the loop's
own traversal.

**This operation is C++-core only, with no numpy fallback at all** -- the
``subdivide``/``agglomerate`` precedent. The outward-orientation pass decides
on the sign of a rounded enclosed volume, a discrete branch a second
implementation could land on the other side of for a near-degenerate
component, and it is on by default; rather than ship a twin that excludes the
default path, this module raises when ``_core`` cannot be used.

Public API:

* :func:`repair` -- repair a surface mesh.
"""

from __future__ import annotations

# Kept for parity with every other operation shim's import shape.
import numpy as np  # noqa: F401

__all__ = ["repair", "PARENT_POINT_NAME", "HOLE_NAME"]

#: Point data (opt-in): the input point each output point came from, -1 for a
#: hole centroid. Twins ``kRepairParentPointName``.
PARENT_POINT_NAME = "repair:parent_point"
#: Cell data (opt-in): -1 for an input triangle, the hole ordinal for a fill
#: triangle. Twins ``kRepairHoleName``.
HOLE_NAME = "repair:hole"


def repair(
    mesh,
    fix_orientation: bool = True,
    orient_outward: bool = True,
    fill_holes: bool = True,
    split_non_manifold: bool = True,
    max_hole_edges: int = 10,
    weld_tolerance: float = 0.0,
    record_provenance: bool = False,
    return_report: bool = False,
):
    """Repair a surface mesh's orientation, holes and pinched vertices.

    Passes run in this order: weld (opt-in, through :func:`clean`) ->
    triangulate (quads and polygons fan exactly as
    ``convert_cells(mode="simplexify")`` does, blocks staying 1:1) -> split
    bowties -> orient (a BFS per connected component over manifold edges,
    fewest flips winning ties) -> fill holes (one centroid point and one
    triangle per loop edge, for every traceable boundary loop of at most
    ``max_hole_edges`` edges) -> orient outward (every closed component whose
    signed volume is negative is flipped whole).

    The output is all-triangle at the surface with blocks 1:1 with the input,
    lower-dimensional blocks carried verbatim, plus one trailing ``triangle``
    block holding the fill triangles when there are any. Points are the
    originals, then the split copies, then the hole centroids; ``point_data``
    copies inherit their source row and centroids the mean of their loop's
    rows; the fill block's ``cell_data`` is NaN for float and 0 for integer
    arrays. Point and Cell regions (and so ``point_sets``/``cell_sets``)
    survive -- a copy joins its source's regions; Side regions are dropped by
    name, since a flip permutes a triangle's edge numbering. Non-manifold
    *edges* (three or more triangles) are neither split nor crossed, only
    counted; nested cavities are not detected.

    :param mesh: a surface mesh (never modified). A 3-D or polyhedron block is
        refused naming ``extract_surface``, a higher-order surface block naming
        ``linearize``.
    :param fix_orientation: rewind triangles so neighbours agree.
    :param orient_outward: after filling, flip closed components with a
        negative volume. Ignored when ``fix_orientation`` is off.
    :param fill_holes: fan-fill boundary loops.
    :param split_non_manifold: duplicate bowtie vertices.
    :param max_hole_edges: the longest loop still filled; ``<= 0`` means no
        limit.
    :param weld_tolerance: weld coincident points within this distance first;
        ``0`` (the default) skips the weld.
    :param record_provenance: attach ``repair:parent_point`` (point data) and
        ``repair:hole`` (cell data).
    :param return_report: also return the run summary dict (the counters,
        ``quality_before``/``quality_after``, ``point_map`` and ``cell_maps``).
    :returns: the repaired mesh, or ``(mesh, report)`` if ``return_report``.
    :raises ValueError: on an out-of-scope input (naming the fix).
    :raises NotImplementedError: when the compiled C++ core is unavailable --
        this operation has no pure-Python fallback.
    """
    try:
        from . import _core
    except ImportError:
        _core = None

    if _core is None:
        raise NotImplementedError(
            "meshio++: repair: this operation is C++-core only and has no "
            "pure-numpy fallback (the outward-orientation pass is a branch on "
            "the sign of a rounded volume a second implementation could "
            "disagree with) -- install a build with the compiled "
            "meshioplusplus._core extension"
        )

    res = _core.repair(
        mesh,
        bool(fix_orientation),
        bool(orient_outward),
        bool(fill_holes),
        bool(split_non_manifold),
        int(max_hole_edges),
        float(weld_tolerance),
        bool(record_provenance),
    )
    out = res.pop("mesh")
    return (out, res) if return_report else out
