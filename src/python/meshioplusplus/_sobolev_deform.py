"""Sobolev (Helmholtz-filtered) deformation: smooth a raw per-point
displacement field through the mesh's own P1 finite-element operators, then
move the points by the smoothed field.

A dependency-free mesh *operation* (not a file format). Adapted from NVIDIA
PhysicsNeMo's ``physicsnemo.mesh.sobolev_deform`` (2.2): solve, per ambient
component, ``(M + l^2 K) u = M d`` and set ``x' = x + u``, with ``K`` the P1
stiffness matrix assembled dimension-generically from the simplex edge Gram
matrix, ``M`` the uniform mean lumped vertex mass (upstream's choice, kept so
a parity test against the reference is a valid oracle) and ``l`` the length
scale -- a screened-Poisson low-pass filter whose cutoff wavelength is ``l``.

**This operation is C++-core only, with no numpy fallback at all.** The
conjugate-gradient stopping test is a branch on a rounded reduction and the
iterate depends on how many iterations ran, so a numpy transcription could
stop one iteration early or late and disagree macroscopically rather than in
the last ulp -- the ``subdivide``/``agglomerate`` precedent applied to a
solver. This module raises when ``_core`` cannot be used.

Public API:

* :func:`sobolev_deform` -- filter a displacement field and apply it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["sobolev_deform", "DISPLACEMENT_NAME"]

#: Point data (opt-in): the filtered displacement. Twins
#: ``kSobolevDisplacementName``.
DISPLACEMENT_NAME = "sobolev:displacement"


def sobolev_deform(
    mesh,
    array: str,
    length_scale: float,
    fixed_points=None,
    fix_boundary: bool = False,
    record_filtered: bool = False,
    max_iterations: int = 128,
    tolerance: float = 1e-10,
    return_report: bool = False,
):
    """Move a mesh's points by the Sobolev-filtered version of a raw
    displacement field.

    Every cell block at the mesh's top topological dimension must be a linear
    simplex (``line``, ``triangle`` or ``tetra``); a quadratic block is refused
    naming ``linearize``, any other type naming
    ``convert_cells(mode="simplexify")``. Lower-dimensional blocks ride along.
    A point in no top-dimensional cell receives its raw displacement. Nothing
    is pinned by default (the natural Neumann condition, under which a
    constant displacement is preserved exactly); ``fixed_points`` and
    ``fix_boundary`` impose zero-Dirichlet rows. Non-convergence within
    ``max_iterations`` is a warning plus ``report["converged"] == False`` --
    the last iterate is still returned.

    This is a pure coordinate move: connectivity, every data array, regions
    and property sets pass through, and the points keep their dtype.

    :param mesh: the mesh (never modified).
    :param array: the ``point_data`` array holding the raw displacement,
        ``(n, dim)`` (or ``(n, 3)`` on a 2-D mesh, z ignored with a warning).
    :param length_scale: the smoothing length ``l`` in mesh units; ``0``
        applies the raw displacement at the free points.
    :param fixed_points: points to pin: a sequence of point ids, the name of
        one of ``mesh.point_sets``, or the name of an integer/bool
        ``point_data`` array (nonzero pins).
    :param fix_boundary: also pin every point on a boundary facet of the
        top-dimensional cells.
    :param record_filtered: attach ``sobolev:displacement``.
    :param max_iterations: the conjugate-gradient iteration cap.
    :param tolerance: relative residual tolerance ``||r|| <= tol * ||b||``.
    :param return_report: also return the run summary dict.
    :returns: the moved mesh, or ``(mesh, report)`` if ``return_report``.
    :raises ValueError: on a missing/mis-shaped array, an out-of-scope block
        (naming the fix), a degenerate cell, or a mis-sized pin list.
    :raises NotImplementedError: when the compiled C++ core is unavailable --
        this operation has no pure-Python fallback.
    """
    ids = None
    fixed_array = ""
    if isinstance(fixed_points, str):
        sets = getattr(mesh, "point_sets", None) or {}
        if fixed_points in sets:
            ids = np.asarray(sets[fixed_points], dtype=np.int64)
        elif fixed_points in mesh.point_data:
            fixed_array = fixed_points
        else:
            raise ValueError(
                f"meshio++: sobolev_deform: no point_set or point_data named "
                f"{fixed_points!r} (point_sets: {sorted(sets)}; point_data: "
                f"{sorted(mesh.point_data)})"
            )
    elif fixed_points is not None:
        ids = np.asarray(fixed_points, dtype=np.int64).ravel()

    try:
        from . import _core
    except ImportError:
        _core = None

    if _core is None:
        raise NotImplementedError(
            "meshio++: sobolev_deform: this operation is C++-core only and has "
            "no pure-numpy fallback (the conjugate-gradient stopping test is a "
            "branch on a rounded reduction, so a second implementation could "
            "stop an iteration early or late) -- install a build with the "
            "compiled meshioplusplus._core extension"
        )

    res = _core.sobolev_deform(
        mesh,
        str(array),
        float(length_scale),
        ids,
        fixed_array,
        bool(fix_boundary),
        bool(record_filtered),
        int(max_iterations),
        float(tolerance),
    )
    out = res.pop("mesh")
    if not res["converged"]:
        from ._common import warn

        warn(
            f"sobolev_deform: conjugate gradients did not converge in "
            f"{res['num_iterations']} iteration(s) (relative residual "
            f"{res['residual']:.3g}); the last iterate is returned -- raise "
            "max_iterations or lower length_scale"
        )
    return (out, res) if return_report else out
