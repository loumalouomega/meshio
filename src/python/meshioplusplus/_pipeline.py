"""The declarative operation pipeline: read a mesh, apply a chain of
operations, write the result — described by a ``settings.json`` document (or
the equivalent dict) and exposed as ``meshioplusplus.run_pipeline`` and the
``pipeline`` CLI verb.

This is the **pure-Python twin** of the C++ engine in
``operations/pipeline.cpp`` (reachable as ``_core.run_pipeline_file`` /
``_core.run_pipeline_json`` when the build carries the vendored nlohmann/json).
It deliberately dispatches over the *public Python API* rather than calling the
C++ engine, for the same reason every shim in this repo has a Python path: the
Python ops inherit each operation's C++/numpy parity contract **and** the
per-format Python fallbacks, so a pipeline whose input needs one (a gmsh
``$Periodic`` file, say) still runs — which ``_core.run_pipeline_file``, being
C++ end to end, could never do. It also makes ``run_pipeline`` fully
functional on an sdist install without the submodule.

Vocabulary (PascalCase ops and keys, lowercase enum values, ``Compare`` — not
``Op`` — for refine's comparator) and the report shape
``{"steps": [{"op", ...counters}], "warnings": [...]}`` are the C++ engine's,
transcribed; ``tests/python/test_pipeline.py`` pins the op/key table against
``_core.pipeline_op_table()`` and the two engines' outputs against each other,
so the twins cannot drift silently. See ``doc/pipeline.md`` for the schema.
"""

import json
import os
import pathlib

from ._agglomerate import agglomerate
from ._clean import clean
from ._convert_cells import convert_cells
from ._crop import crop
from ._curvature import compute_curvature
from ._data_average import cell_data_to_point_data, point_data_to_cell_data
from ._data_calc import data_calc
from ._data_condition import data_condition
from ._data_manage import data_drop, data_keep, data_rename
from ._decimate import decimate
from ._decimate_volume import decimate_volume
from ._error import estimate_error
from ._gradient import gradient
from ._helpers import _filetypes_from_path, read, write
from ._hessian import hessian
from ._isosurface import isosurface
from ._optimize_volume import optimize_volume
from ._partition import partition_labels
from ._provenance import add_operation as _prov_add_operation
from ._provenance import set_source as _prov_set_source
from ._provenance import set_target as _prov_set_target
from ._quality import attach_quality
from ._refine import refine
from ._remesh import remesh
from ._remesh_volume import remesh_volume
from ._reorder import reorder
from ._sdf import compute_sdf
from ._skin import extract_skin
from ._slice import slice as _slice
from ._smooth import smooth
from ._subdivide import subdivide
from ._surface import extract_surface
from ._transform import transform
from ._voxelize import voxelize

__all__ = ["run_pipeline"]

# The step vocabulary: op name -> allowed parameter keys. A transcription of
# pipe_op_table() in src/cpp/src/operations/pipeline.cpp, pinned against
# _core.pipeline_op_table() by tests/python/test_pipeline.py -- edit both.
_OP_TABLE = {
    "Quality": (),
    "Clean": ("Weld", "Atol", "RemoveOrphans", "DropDegenerate", "DropDuplicateCells"),
    "Smooth": (
        "Method",
        "Iterations",
        "Lambda",
        "Mu",
        "FixBoundary",
        "PreserveFeatures",
        "FeatureAngle",
        "GuardInversion",
    ),
    "Refine": (
        "Levels",
        "Cells",
        "Region",
        "Array",
        "Compare",
        "Value",
        "Closure",
        "RecordLevels",
        "RecordHierarchy",
    ),
    "Decimate": (
        "Ratio",
        "TargetFaces",
        "MaxError",
        "Placement",
        "PreserveBoundary",
        "PreserveFeatures",
        "FeatureAngle",
    ),
    "DecimateVolume": (
        "Ratio",
        "TargetCells",
        "MaxError",
        "Placement",
        "PreserveBoundary",
        "PreserveFeatures",
        "FeatureAngle",
    ),
    "Partition": ("Nparts", "Method", "Imbalance", "Mode", "Seed", "WeightsKey"),
    "Slice": ("Point", "Normal", "RecordParentIds"),
    "Section": ("Point", "Normal", "RecordParentIds"),  # alias of Slice
    "Gradient": ("Array", "Operator", "Method", "Location", "Output", "Component"),
    "Hessian": ("Array", "Method", "Location", "Output"),
    "Curvature": (
        "Mean",
        "Gaussian",
        "DualArea",
        "IncludeBoundary",
        "RecordArea",
        "RecordPrincipal",
        "Region",
    ),
    "EstimateError": ("Array", "Method", "Marking", "MarkingValue", "Output", "Marked"),
    "Remesh": (
        "NumClusters",
        "Subdivide",
        "SubsampleRatio",
        "MaxSubdivide",
        "MaxIterations",
        "MaxRepairPasses",
        "Metric",
        "Gradation",
        "PreserveBoundary",
        "MaxAnisotropy",
    ),
    "RemeshVolume": (
        "Resolution",
        "CellSize",
        "Bounds",
        "Padding",
        "PaddingRelative",
        "MaxCells",
        "MaxTets",
        "WarpFraction",
        "Sign",
        "WatertightCheck",
    ),
    "OptimizeVolume": (
        "MaxIterations",
        "Relocate",
        "Flip",
        "PreserveBoundary",
        "MinImprovement",
    ),
    "Isosurface": ("Array", "Isovalue", "Isovalues", "Component", "RecordParentIds"),
    "Voxelize": (
        "Resolution",
        "CellSize",
        "Bounds",
        "Padding",
        "PaddingRelative",
        "Fill",
        "AttachOccupancy",
        "MaxCells",
        "Sign",
    ),
    "ComputeSdf": (
        "Structure",
        "Resolution",
        "CellSize",
        "Bounds",
        "Padding",
        "PaddingRelative",
        "RootResolution",
        "MaxDepth",
        "BandCells",
        "RecordLevels",
        "MaxCells",
        "Sign",
        "Location",
        "Band",
    ),
    "Transform": (
        "Translate",
        "Scale",
        "RotateAxis",
        "RotateDegrees",
        "Matrix",
        "ScaleUnits",
        "RotateData",
    ),
    "ConvertCells": ("Mode", "RecordParentIds"),
    "Subdivide": ("RecordParentIds",),
    "Agglomerate": ("TargetGroupSize",),
    "Crop": (
        "Bbox",
        "Point",
        "Normal",
        "Where",
        "Compare",
        "Value",
        "Mode",
        "RecordIds",
    ),
    "ExtractSurface": ("RecordParentIds",),
    "ExtractSkin": ("Linearize",),
    "Reorder": ("Method",),
    "DataDrop": ("Point", "Cell", "Field", "IgnoreMissing"),
    "DataKeep": ("Point", "Cell", "Field"),
    "DataRename": ("Point", "Cell", "Field"),
    "DataCalc": ("Expr", "Location", "Overwrite"),
    "DataCondition": (
        "Mode",
        "Location",
        "Names",
        "Scope",
        "Lo",
        "Hi",
        "NanPolicy",
        "NanReplacement",
        "Suffix",
    ),
    "ToCell": ("Names",),
    "ToPoint": ("Names", "Weight"),
}

# Ops that exist in meshio++ but cannot be a single-mesh pipeline step; the
# error names where to go instead (the C++ engine's exact rule).
_EXCLUDED_OPS = {
    "Merge": "'Merge' needs several input meshes and is not a pipeline step; "
    "use the `merge` CLI verb",
    "Interpolate": "'Interpolate' needs a second (source) mesh and is not a "
    "pipeline step; use the `interpolate` CLI verb",
    "Split": "'Split' produces several output meshes and is not a pipeline "
    "step; use the `split` CLI verb",
    "Diff": "'Diff' compares two meshes and is not a pipeline step; use the "
    "`diff` CLI verb",
    "UndoGreen": "'UndoGreen' needs a second (coarse) mesh and is not a "
    "pipeline step; use the `undo-green` CLI verb",
    "ConservativeInterpolate": "'ConservativeInterpolate' needs a second "
    "(source) mesh and is not a pipeline step; use the "
    "`conservative-interpolate` CLI verb",
}


def _err(op, what):
    return ValueError(f"meshio++: pipeline: step '{op}': {what}")


def _number(step, key, fallback):
    v = step.get(key)
    if v is None:
        return fallback
    # bool is an int subclass; a boolean here is a type error, not a number.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _err(step["Op"], f"parameter '{key}' must be a number")
    return float(v)


def _flag(step, key, fallback):
    v = step.get(key)
    if v is None:
        return fallback
    if not isinstance(v, bool):
        raise _err(step["Op"], f"parameter '{key}' must be a boolean")
    return v


def _text(step, key, fallback):
    v = step.get(key)
    if v is None:
        return fallback
    if not isinstance(v, str):
        raise _err(step["Op"], f"parameter '{key}' must be a string")
    return v


def _dvec(step, key):
    v = step.get(key)
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or any(
        isinstance(e, bool) or not isinstance(e, (int, float)) for e in v
    ):
        raise _err(step["Op"], f"parameter '{key}' must be an array of numbers")
    return [float(e) for e in v]


def _svec(step, key):
    v = step.get(key)
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or any(not isinstance(e, str) for e in v):
        raise _err(step["Op"], f"parameter '{key}' must be an array of strings")
    return list(v)


def _vec3(step, key):
    v = _dvec(step, key)
    if v is None or len(v) != 3:
        raise _err(step["Op"], f"parameter '{key}' must be an array of 3 numbers")
    return v


def _total_cells(mesh):
    return int(sum(len(block.data) for block in mesh.cells))


def _validate_step(step):
    if not isinstance(step, dict):
        raise ValueError("meshio++: pipeline: every operation must be an object")
    op = step.get("Op")
    if not isinstance(op, str) or not op:
        raise ValueError("meshio++: pipeline: every operation needs a string 'Op'")
    if op not in _OP_TABLE:
        if op in _EXCLUDED_OPS:
            raise ValueError(f"meshio++: pipeline: {_EXCLUDED_OPS[op]}")
        known = ", ".join(_OP_TABLE)
        raise ValueError(
            f"meshio++: pipeline: unknown operation '{op}' (known: {known})"
        )
    allowed = _OP_TABLE[op]
    for key in step:
        if key != "Op" and key not in allowed:
            keys = ", ".join(allowed)
            hint = f" (known: {keys})" if keys else " (the op takes no parameters)"
            raise _err(op, f"unknown parameter '{key}'{hint}")


def _apply_data_manage(mesh, op, step):
    locations = (("Point", "point"), ("Cell", "cell"), ("Field", "field"))
    if op == "DataDrop":
        ignore_missing = _flag(step, "IgnoreMissing", False)
        for key, location in locations:
            if key in step:
                mesh = data_drop(mesh, location, _svec(step, key) or [], ignore_missing)
    elif op == "DataKeep":
        for key, location in locations:
            if key in step:
                mesh = data_keep(mesh, location, _svec(step, key) or [])
    else:  # DataRename
        for key, location in locations:
            for entry in _svec(step, key) or []:
                # "OLD:NEW" splits on the LAST colon (names carry colons:
                # `gmsh:physical`) -- the CLI's documented rule.
                old, sep, new = entry.rpartition(":")
                if not sep or not old or not new:
                    raise _err(op, f"rename entry '{entry}' must be 'OLD:NEW'")
                mesh = data_rename(mesh, location, old, new)
    return mesh


def _apply_transform(mesh, step):
    kwargs = {}
    if "Translate" in step:
        kwargs["translate"] = _vec3(step, "Translate")
    if "Scale" in step:
        v = step["Scale"]
        kwargs["scale"] = (
            _dvec(step, "Scale")
            if isinstance(v, (list, tuple))
            else _number(step, "Scale", 1.0)
        )
    if "RotateAxis" in step or "RotateDegrees" in step:
        axis = _vec3(step, "RotateAxis")
        if "RotateDegrees" not in step:
            raise _err("Transform", "'RotateAxis' requires 'RotateDegrees'")
        kwargs["rotate"] = (axis, _number(step, "RotateDegrees", 0.0))
    if "Matrix" in step:
        matrix = _dvec(step, "Matrix")
        if matrix is None or len(matrix) != 16:
            raise _err("Transform", "'Matrix' expects 16 values (row-major 4x4)")
        kwargs["matrix"] = matrix
    if "ScaleUnits" in step:
        kwargs["scale_units"] = _number(step, "ScaleUnits", 1.0)
    if len(kwargs) != 1:
        raise _err(
            "Transform",
            "give exactly one of 'Translate'/'Scale'/'RotateAxis'+'RotateDegrees'/"
            "'Matrix'/'ScaleUnits'",
        )
    return transform(
        mesh, rotate_vector_data=_flag(step, "RotateData", False), **kwargs
    )


def _render_op(step):
    """Renders one step as ``Op(Key=value, Key=value, ...)`` for the
    provenance operation chain -- the Python twin of `pipe_render_op` in
    pipeline.cpp. Keys are sorted (the C++ side's std::map iteration order
    is already sorted, so this is what keeps the two textually identical)."""
    parts = []
    for key in sorted(k for k in step if k != "Op"):
        value = step[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return f"{step['Op']}(" + ", ".join(parts) + ")"


def _apply_step(mesh, step, steps, warnings):
    """One validated step. Counters and warning strings are the C++ engine's,
    transcribed -- keep the two in lockstep."""
    op = step["Op"]
    entry = {"op": op}

    if op == "Quality":
        mesh = attach_quality(mesh)
    elif op == "Clean":
        mesh, report = clean(
            mesh,
            weld=_flag(step, "Weld", False),
            atol=_number(step, "Atol", 1e-8),
            remove_orphans=_flag(step, "RemoveOrphans", True),
            drop_degenerate=_flag(step, "DropDegenerate", True),
            drop_duplicate_cells=_flag(step, "DropDuplicateCells", True),
            return_report=True,
        )
        entry["PointsWelded"] = report["points_welded"]
        entry["PointsRemovedOrphan"] = report["points_removed_orphan"]
        entry["CellsDroppedDegenerate"] = report["cells_dropped_degenerate"]
        entry["CellsDroppedDuplicate"] = report["cells_dropped_duplicate"]
    elif op == "Smooth":
        mesh, report = smooth(
            mesh,
            method=_text(step, "Method", "taubin"),
            iterations=int(_number(step, "Iterations", 10)),
            lambda_=_number(step, "Lambda", -1.0),
            mu=_number(step, "Mu", -0.34),
            fix_boundary=_flag(step, "FixBoundary", True),
            preserve_features=_flag(step, "PreserveFeatures", True),
            feature_angle=_number(step, "FeatureAngle", 30.0),
            guard_inversion=_flag(step, "GuardInversion", True),
            return_report=True,
        )
        entry["NumNodesMoved"] = report["num_nodes_moved"]
        entry["MaxDisplacement"] = report["max_displacement"]
        entry["NumSkippedInversion"] = report["num_skipped_inversion"]
    elif op == "Refine":
        where = None
        array = _text(step, "Array", "")
        if array:
            # `Compare`, not `Op`: `Op` is the step's own discriminant.
            compare = _text(step, "Compare", "<")
            where = f"{array} {compare} {_number(step, 'Value', 0.0)}"
        cells = _dvec(step, "Cells")
        mesh = refine(
            mesh,
            levels=int(_number(step, "Levels", 1)),
            cells=[int(c) for c in cells] if cells is not None else None,
            region=_text(step, "Region", "") or None,
            where=where,
            closure=_text(step, "Closure", "redgreen"),
            record_levels=_flag(step, "RecordLevels", False),
            record_hierarchy=_flag(step, "RecordHierarchy", False),
        )
    elif op == "Decimate":
        ratio = _number(step, "Ratio", -1.0)
        target_faces = _number(step, "TargetFaces", -1.0)
        max_error = _number(step, "MaxError", -1.0)
        if ratio < 0.0 and target_faces < 0.0 and max_error < 0.0:
            ratio = 0.5  # a usable pipeline default (the C++ engine's rule)
        mesh, report = decimate(
            mesh,
            ratio=ratio if ratio >= 0.0 else None,
            target_faces=int(target_faces) if target_faces >= 0.0 else None,
            max_error=max_error if max_error >= 0.0 else None,
            placement=_text(step, "Placement", "optimal"),
            preserve_boundary=_flag(step, "PreserveBoundary", True),
            preserve_features=_flag(step, "PreserveFeatures", True),
            feature_angle=_number(step, "FeatureAngle", 30.0),
            return_report=True,
        )
        entry["FacesRemoved"] = report["faces_removed"]
        entry["CollapsesRejected"] = report["collapses_rejected"]
    elif op == "DecimateVolume":
        ratio = _number(step, "Ratio", -1.0)
        target_cells = _number(step, "TargetCells", -1.0)
        max_error = _number(step, "MaxError", -1.0)
        if ratio < 0.0 and target_cells < 0.0 and max_error < 0.0:
            ratio = 0.5  # a usable pipeline default (the C++ engine's rule)
        mesh, report = decimate_volume(
            mesh,
            ratio=ratio if ratio >= 0.0 else None,
            target_cells=int(target_cells) if target_cells >= 0.0 else None,
            max_error=max_error if max_error >= 0.0 else None,
            placement=_text(step, "Placement", "optimal"),
            preserve_boundary=_flag(step, "PreserveBoundary", False),
            preserve_features=_flag(step, "PreserveFeatures", True),
            feature_angle=_number(step, "FeatureAngle", 30.0),
            return_report=True,
        )
        entry["TetsRemoved"] = report["tets_removed"]
        entry["CollapsesRejected"] = report["collapses_rejected"]
    elif op == "Partition":
        nparts = int(_number(step, "Nparts", 2))
        labels = partition_labels(
            mesh,
            nparts,
            method=_text(step, "Method", "auto"),
            imbalance=_number(step, "Imbalance", 0.03),
            mode=_text(step, "Mode", "eco"),
            seed=int(_number(step, "Seed", 0)),
            weights=_text(step, "WeightsKey", "") or None,
        )
        mesh.cell_data["partition:part"] = labels
        entry["Nparts"] = float(nparts)
    elif op in ("Slice", "Section"):
        mesh = _slice(
            mesh,
            origin=_vec3(step, "Point"),
            normal=_vec3(step, "Normal"),
            record_parent_ids=_flag(step, "RecordParentIds", False),
        )
        entry["SectionFaces"] = float(_total_cells(mesh))
        if _total_cells(mesh) == 0:
            warnings.append(
                "the section is empty; the plane misses the mesh -- move or flip it"
            )
    elif op == "Gradient":
        component = int(_number(step, "Component", -1.0))
        mesh, report = gradient(
            mesh,
            _text(step, "Array", ""),
            operator=_text(step, "Operator", "gradient"),
            method=_text(step, "Method", "green-gauss"),
            location=_text(step, "Location", "cell"),
            output=_text(step, "Output", "") or None,
            component=component if component >= 0 else None,
            overwrite=True,
            return_report=True,
        )
        entry["NumSkipped"] = report["num_skipped"]
        entry["NumFallback"] = report["num_fallback"]
        if report["num_skipped"] > 0:
            warnings.append(
                f"gradient: {report['num_skipped']} cell(s) could not be "
                "differentiated and are NaN"
            )
    elif op == "Hessian":
        mesh, report = hessian(
            mesh,
            _text(step, "Array", ""),
            method=_text(step, "Method", "green-gauss"),
            location=_text(step, "Location", "cell"),
            output=_text(step, "Output", "") or None,
            overwrite=True,
            return_report=True,
        )
        entry["NumSkipped"] = report["num_skipped"]
        entry["NumFallback"] = report["num_fallback"]
        if report["num_skipped"] > 0:
            warnings.append(
                f"hessian: {report['num_skipped']} cell(s) could not be "
                "evaluated and are NaN"
            )
    elif op == "Curvature":
        mesh, report = compute_curvature(
            mesh,
            mean=_flag(step, "Mean", True),
            gaussian=_flag(step, "Gaussian", True),
            dual_area=_text(step, "DualArea", "mixed-voronoi"),
            include_boundary=_flag(step, "IncludeBoundary", False),
            record_area=_flag(step, "RecordArea", False),
            record_principal=_flag(step, "RecordPrincipal", False),
            region=_text(step, "Region", ""),
            return_report=True,
        )
        entry["NumBoundary"] = report["num_boundary"]
        entry["NumIsolated"] = report["num_isolated"]
        entry["NumDegenerate"] = report["num_degenerate"]
        entry["TotalAngleDefect"] = report["total_angle_defect"]
        # H's sign comes from the surface's own winding, so a mesh whose facets
        # disagree about which side is out yields sign-flipped patches with no
        # error raised. Say so rather than letting it pass silently.
        if report["quality"]["inconsistent_pairs"] > 0:
            warnings.append(
                f"curvature: {report['quality']['inconsistent_pairs']} edge "
                "pair(s) wind the same way, so the sign of 'curvature:mean' is "
                "not trustworthy"
            )
    elif op == "EstimateError":
        mesh, report = estimate_error(
            mesh,
            _text(step, "Array", ""),
            method=_text(step, "Method", "zz"),
            marking=_text(step, "Marking", "none"),
            marking_value=_number(step, "MarkingValue", 0.0),
            output=_text(step, "Output", "") or None,
            marked_name=_text(step, "Marked", "") or None,
            overwrite=True,
            return_report=True,
        )
        entry["GlobalError"] = report["global_error"]
        entry["NumSkipped"] = report["num_skipped"]
        entry["NumMarked"] = report["num_marked"]
        if report["num_skipped"] > 0:
            warnings.append(
                f"estimate_error: {report['num_skipped']} cell(s) could not be "
                "evaluated and are NaN"
            )
    elif op == "Remesh":
        # Unlike every other step, the output has NO correspondence to the
        # input: new points, new connectivity. point_data/cell_data/regions
        # are dropped (remesh's own contract), field_data carries through.
        n_subdivide = _number(step, "Subdivide", -1)
        mesh, report = remesh(
            mesh,
            int(_number(step, "NumClusters", 0)),
            subdivide=None if n_subdivide < 0 else int(n_subdivide),
            subsample_ratio=_number(step, "SubsampleRatio", 10.0),
            max_subdivide=int(_number(step, "MaxSubdivide", 4)),
            max_iterations=int(_number(step, "MaxIterations", 100)),
            max_repair_passes=int(_number(step, "MaxRepairPasses", 10)),
            metric=_text(step, "Metric", "isotropic"),
            gradation=_number(step, "Gradation", 0.0),
            preserve_boundary=_flag(step, "PreserveBoundary", True),
            max_anisotropy=_number(step, "MaxAnisotropy", 4.0),
            return_report=True,
        )
        entry["NumClusters"] = float(report["num_clusters"])
        entry["NumIterations"] = float(report["num_iterations"])
        entry["SubdivideApplied"] = float(report["subdivide_applied"])
        entry["NumIsolatedClusters"] = float(report["num_isolated_clusters"])
        entry["NumNonManifoldVertices"] = float(report["num_non_manifold_vertices"])
        if (
            report["num_isolated_clusters"] > 0
            or report["num_non_manifold_vertices"] > 0
        ):
            warnings.append(
                f"remesh: {report['num_isolated_clusters']} isolated cluster(s), "
                f"{report['num_non_manifold_vertices']} non-manifold vertex/vertices could "
                "not be repaired; output may be non-manifold near them"
            )
    elif op == "RemeshVolume":
        # Remesh's volumetric sibling: same no-correspondence-with-the-input
        # contract, same lattice-sizing vocabulary as Voxelize/ComputeSdf.
        resolution = _dvec(step, "Resolution")
        bounds = _dvec(step, "Bounds")
        mesh, report = remesh_volume(
            mesh,
            resolution=None if not resolution else [int(v) for v in resolution],
            cell_size=_number(step, "CellSize", 0.0) if "CellSize" in step else None,
            bounds=None if not bounds else list(bounds),
            padding=_number(step, "Padding", 0.0),
            padding_relative=_number(step, "PaddingRelative", 0.1),
            max_cells=int(_number(step, "MaxCells", 20000000.0)),
            max_tets=int(_number(step, "MaxTets", 20000000.0)),
            warp_fraction=_number(step, "WarpFraction", 0.35),
            sign=_text(step, "Sign", "pseudonormal"),
            watertight_check=_text(step, "WatertightCheck", "warn"),
            return_report=True,
        )
        entry["NumTets"] = float(report["num_tets"])
        entry["NumVerticesWarped"] = float(report["num_vertices_warped"])
        entry["NumTetsRejected"] = float(report["num_tets_rejected"])
        entry["NumNonManifoldEdges"] = float(report["num_non_manifold_edges"])
        if report["num_non_manifold_edges"] > 0:
            warnings.append(
                f"remesh_volume: {report['num_non_manifold_edges']} non-manifold boundary "
                "edge(s), from warping (see doc/remesh_volume.md)"
            )
    elif op == "OptimizeVolume":
        # ODT remeshing: relocate vertices AND flip connectivity (2-3/3-2). The
        # point set is invariant, so point_data/field_data and Point regions
        # carry; cell_data + Cell/Side regions are dropped.
        mesh, report = optimize_volume(
            mesh,
            max_iterations=int(_number(step, "MaxIterations", 10.0)),
            relocate=_flag(step, "Relocate", True),
            flip=_flag(step, "Flip", True),
            preserve_boundary=_flag(step, "PreserveBoundary", True),
            min_improvement=_number(step, "MinImprovement", 1e-6),
            return_report=True,
        )
        entry["NumFlips"] = float(report["num_flips"])
        entry["Num23Flips"] = float(report["num_23_flips"])
        entry["Num32Flips"] = float(report["num_32_flips"])
        entry["NumVerticesMoved"] = float(report["num_vertices_moved"])
        entry["NumTets"] = float(report["num_tets"])
        entry["MinQualityBefore"] = float(report["min_quality_before"])
        entry["MinQualityAfter"] = float(report["min_quality_after"])
    elif op == "Voxelize":
        resolution = _dvec(step, "Resolution")
        bounds = _dvec(step, "Bounds")
        mesh, rep = voxelize(
            mesh,
            resolution=None if not resolution else [int(v) for v in resolution],
            cell_size=_number(step, "CellSize", 0.0) if "CellSize" in step else None,
            bounds=None if not bounds else list(bounds),
            padding=_number(step, "Padding", 0.0),
            padding_relative=_number(step, "PaddingRelative", 0.0),
            fill=_text(step, "Fill", "all"),
            attach_occupancy=_flag(step, "AttachOccupancy", False),
            max_cells=int(_number(step, "MaxCells", 20000000.0)),
            sign=_text(step, "Sign", "pseudonormal"),
            watertight_check="off",
            return_report=True,
        )
        entry["NumOccupied"] = float(rep["num_occupied"])

    elif op == "ComputeSdf":
        # Voxelize's sibling: it likewise REPLACES the input's geometry, taking a
        # surface in and handing a filled grid back.
        resolution = _dvec(step, "Resolution")
        bounds = _dvec(step, "Bounds")
        mesh, rep = compute_sdf(
            mesh,
            structure=_text(step, "Structure", "voxel"),
            resolution=None if not resolution else [int(v) for v in resolution],
            cell_size=_number(step, "CellSize", 0.0) if "CellSize" in step else None,
            bounds=None if not bounds else list(bounds),
            padding=_number(step, "Padding", 0.0),
            padding_relative=_number(step, "PaddingRelative", 0.1),
            root_resolution=int(_number(step, "RootResolution", 8.0)),
            max_depth=int(_number(step, "MaxDepth", 4.0)),
            band_cells=_number(step, "BandCells", 1.0),
            record_levels=_flag(step, "RecordLevels", True),
            max_cells=int(_number(step, "MaxCells", 20000000.0)),
            sign=_text(step, "Sign", "pseudonormal"),
            location=_text(step, "Location", "corner"),
            band=_number(step, "Band", 0.0),
            watertight_check="off",
            return_report=True,
        )
        entry["MaxDepth"] = float(rep["max_depth"])
        entry["NumBanded"] = float(rep["num_banded"])

    elif op == "Isosurface":
        isovalues = _dvec(step, "Isovalues")
        if not isovalues:
            isovalues = [_number(step, "Isovalue", 0.0)]
        component = int(_number(step, "Component", -1.0))
        mesh = isosurface(
            mesh,
            _text(step, "Array", ""),
            isovalues,
            component=component if component >= 0 else None,
            record_parent_ids=_flag(step, "RecordParentIds", False),
        )
        entry["ContourCells"] = float(_total_cells(mesh))
        if _total_cells(mesh) == 0:
            warnings.append(
                "the contour is empty; the isovalue lies outside the field's range"
            )
    elif op == "Transform":
        mesh = _apply_transform(mesh, step)
    elif op == "ConvertCells":
        mesh = convert_cells(
            mesh,
            mode=_text(step, "Mode", "linearize"),
            record_parent_ids=_flag(step, "RecordParentIds", False),
        )
    elif op == "Subdivide":
        mesh = subdivide(
            mesh,
            record_parent_ids=_flag(step, "RecordParentIds", False),
        )
    elif op == "Agglomerate":
        mesh = agglomerate(
            mesh,
            target_group_size=int(_number(step, "TargetGroupSize", 8.0)),
        )
    elif op == "Crop":
        has_bbox = "Bbox" in step
        has_plane = "Point" in step or "Normal" in step
        has_where = "Where" in step
        if has_bbox + has_plane + has_where != 1:
            raise _err(op, "give one of 'Bbox', 'Point'+'Normal' or 'Where'")
        if has_where and "Mode" in step:
            raise _err(
                op,
                "'Mode' applies to 'Bbox' and 'Point'+'Normal', which test points; "
                "a 'Where' predicate is already one value per cell and has nothing "
                "to reduce",
            )
        if has_bbox:
            kwargs = {"bbox": _dvec(step, "Bbox")}
            if kwargs["bbox"] is None or len(kwargs["bbox"]) != 6:
                raise _err(op, "'Bbox' must be [xmin, ymin, zmin, xmax, ymax, zmax]")
        elif has_plane:
            kwargs = {"plane": (_vec3(step, "Point"), _vec3(step, "Normal"))}
        else:
            kwargs = {
                "where": (
                    _text(step, "Where", ""),
                    _text(step, "Compare", "<"),
                    _number(step, "Value", 0.0),
                )
            }
        mesh = crop(
            mesh,
            record_ids=_flag(step, "RecordIds", False),
            **({} if has_where else {"mode": _text(step, "Mode", "all")}),
            **kwargs,
        )
        entry["CellsKept"] = float(_total_cells(mesh))
    elif op == "ExtractSurface":
        mesh = extract_surface(
            mesh, record_parent_ids=_flag(step, "RecordParentIds", False)
        )
    elif op == "ExtractSkin":
        mesh = extract_skin(mesh, linearize=_flag(step, "Linearize", False))
    elif op == "Reorder":
        mesh = reorder(mesh, method=_text(step, "Method", "rcm"))
    elif op in ("DataDrop", "DataKeep", "DataRename"):
        mesh = _apply_data_manage(mesh, op, step)
    elif op == "DataCalc":
        spec = _text(step, "Expr", "")
        # "NAME = EXPR" splits on the FIRST '=' (the CLI's documented rule).
        name, sep, expr = spec.partition("=")
        if not sep:
            raise _err(op, "'Expr' must be 'NAME = EXPRESSION'")
        name = name.strip()
        if not name:
            raise _err(op, "'Expr' has an empty output name")
        mesh = data_calc(
            mesh,
            expr.strip(),
            location=_text(step, "Location", "point"),
            output=name,
            overwrite=_flag(step, "Overwrite", False),
        )
    elif op == "DataCondition":
        mesh = data_condition(
            mesh,
            location=_text(step, "Location", "point"),
            keys=_svec(step, "Names"),
            mode=_text(step, "Mode", "clamp"),
            scope=_text(step, "Scope", "component"),
            lo=_number(step, "Lo", 0.0),
            hi=_number(step, "Hi", 1.0),
            nan_policy=_text(step, "NanPolicy", "ignore"),
            nan_replacement=_number(step, "NanReplacement", 0.0),
            suffix=_text(step, "Suffix", ""),
        )
    elif op == "ToCell":
        mesh = point_data_to_cell_data(mesh, keys=_svec(step, "Names"))
    elif op == "ToPoint":
        mesh = cell_data_to_point_data(
            mesh,
            keys=_svec(step, "Names"),
            weighted=_text(step, "Weight", "uniform") == "measure",
        )
    else:  # unreachable after _validate_step
        raise ValueError(f"meshio++: pipeline: unknown operation '{op}'")

    steps.append(entry)
    # A no-op when no scope is open (the common case); mirrors what
    # pipe_push_step does on the C++ side, from the step's own INPUT
    # parameters (not the result entry above, which holds counters).
    _prov_add_operation(_render_op(step))
    return mesh


def _check_keys(obj, where, allowed):
    for key in obj:
        if key not in allowed:
            raise ValueError(
                f"meshio++: pipeline: unknown key '{key}' in {where} "
                f"(known: {', '.join(allowed)})"
            )


def _read_kwargs_from(inp, warnings):
    """``Input.Options`` -> ``read()`` kwargs, validating strictly.

    Shared with ``_sequence.py`` rather than transcribed: two copies of this
    mapping would drift on the next option added, and a sequence document's
    ``Input.Options`` must mean exactly what a single-file one's does.
    """
    read_kwargs = {}
    options = inp.get("Options") or {}
    if not isinstance(options, dict):
        raise ValueError("meshio++: pipeline: Input.Options must be an object")
    _check_keys(
        options,
        "Input.Options",
        ("PointsOnly", "DataArrays", "TimeStep", "Lenient", "Mmap"),
    )
    if options.get("PointsOnly"):
        read_kwargs["points_only"] = True
    if "DataArrays" in options:
        read_kwargs["arrays"] = list(options["DataArrays"])
    if "TimeStep" in options:
        read_kwargs["time_step"] = int(options["TimeStep"])
    if options.get("Mmap") not in (None, "auto", "on", "off"):
        raise ValueError(
            "meshio++: pipeline: Input.Options.Mmap must be 'auto', 'on' or 'off'"
        )
    # Mmap is advisory by contract (a preference, never a demand), so the
    # Python runner -- whose readers do not memory-map -- may ignore it.
    if options.get("Lenient"):
        # Lenient exists to make the *C++* readers tolerant; this runner reads
        # through the Python API, whose per-format fallbacks already tolerate
        # those constructs. Say so rather than silently dropping the flag.
        warnings.append(
            "Input.Options.Lenient has no effect in the Python runner (its "
            "per-format fallbacks already read what lenient would skip)"
        )
    return read_kwargs


def _write_kwargs_from(out, out_path):
    """``Output`` -> ``write()`` kwargs. Shared with ``_sequence.py``.

    ``out_path`` is passed separately because a sequence's output is a pattern
    whose *expanded* name is what carries the extension.
    """
    encoding = out.get("Encoding", "default")
    if encoding not in ("default", "ascii", "binary"):
        raise ValueError(
            "meshio++: pipeline: Output.Encoding must be 'ascii' or 'binary'"
        )
    codec = out.get("Codec")
    if codec is not None and codec not in ("none", "zlib", "lz4", "zstd"):
        raise ValueError(
            "meshio++: pipeline: Output.Codec must be 'none', 'zlib', 'lz4' or 'zstd'"
        )
    write_kwargs = {}
    if encoding != "default" or codec is not None:
        out_fmt = out.get("Format")
        if not out_fmt:
            candidates = _filetypes_from_path(pathlib.Path(str(out_path)))
            out_fmt = candidates[0] if candidates else None
        if out_fmt is None:
            raise ValueError(
                "meshio++: pipeline: cannot infer the output format needed to "
                "apply Encoding/Codec; set Output.Format"
            )
        # The MCP convert tool's writer-kwargs mapping is the single Python
        # owner of "this format's ascii/binary/codec spelling"; imported
        # lazily -- mcp/_tools.py imports the package, so a module-scope
        # import here would be a cycle. (It never imports the mcp SDK.)
        from .mcp._tools import _variant_kwargs

        write_kwargs = _variant_kwargs(
            out_fmt, "auto" if encoding == "default" else encoding, codec
        )
    if out.get("FloatFormat"):
        write_kwargs["float_fmt"] = out["FloatFormat"]
    return write_kwargs


def _resolve_settings(settings):
    """A dict is used as-is; a string/PathLike is JSON text when it starts
    with '{' (after whitespace), else a settings-file path."""
    if isinstance(settings, dict):
        return settings
    if isinstance(settings, os.PathLike):
        settings = os.fspath(settings)
    if not isinstance(settings, str):
        raise TypeError(
            "meshio++: pipeline: settings must be a dict, JSON text or a path"
        )
    if settings.lstrip().startswith("{"):
        return json.loads(settings)
    with open(settings, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_pipeline(settings, input_path=None, output_path=None):
    """Run a whole settings pipeline: read ``Input.Path``, apply
    ``Operations`` in order, write ``Output.Path``.

    :param settings: the settings document -- a dict, the JSON text itself, or
        the path of a ``settings.json`` (see ``doc/pipeline.md`` for the
        schema; ops and keys are PascalCase, enum values lowercase).
    :param input_path: overrides ``Input.Path`` (the CLI's ``--input``).
    :param output_path: overrides ``Output.Path`` (the CLI's ``--output``).
    :returns: the run report ``{"steps": [{"op", ...counters}],
        "warnings": [...]}`` -- counter keys are PascalCase, matching the C++
        engine's report exactly.
    :raises ValueError: on any schema error -- an unknown op, an unknown key
        anywhere, a mis-typed parameter -- always naming the offender, never
        silently ignoring it.
    """
    doc = _resolve_settings(settings)
    if not isinstance(doc, dict):
        raise ValueError("meshio++: pipeline: the settings document must be an object")

    # A transient document (a glob/list input, several steps, a {step} output)
    # is routed to the sequence driver rather than rejected, so one entry point
    # serves both and nobody has to know which kind of document they hold. A
    # plain single-file document never reaches that code path at all -- its
    # behaviour below is physically unchanged.
    from ._sequence import _is_sequence_document, run_sequence_pipeline

    if _is_sequence_document(doc, input_path, output_path):
        return run_sequence_pipeline(
            doc, input_path=input_path, output_path=output_path
        )

    _check_keys(
        doc, "the settings document", ("Version", "Input", "Operations", "Output")
    )
    version = doc.get("Version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(
            f"meshio++: pipeline: unsupported Version {version!r} (this build knows 1)"
        )

    if "Input" not in doc or not isinstance(doc["Input"], dict):
        raise ValueError("meshio++: pipeline: Input is required and must be an object")
    if "Output" not in doc or not isinstance(doc["Output"], dict):
        raise ValueError("meshio++: pipeline: Output is required and must be an object")
    inp, out = doc["Input"], doc["Output"]
    _check_keys(inp, "Input", ("Path", "Format", "Options"))
    _check_keys(out, "Output", ("Path", "Format", "Encoding", "Codec", "FloatFormat"))

    in_path = input_path if input_path is not None else inp.get("Path")
    out_path = output_path if output_path is not None else out.get("Path")
    if not in_path:
        raise ValueError("meshio++: pipeline: Input.Path is required")
    if not out_path:
        raise ValueError("meshio++: pipeline: Output.Path is required")

    warnings = []
    read_kwargs = _read_kwargs_from(inp, warnings)

    steps_spec = doc.get("Operations", [])
    if not isinstance(steps_spec, list):
        raise ValueError("meshio++: pipeline: Operations must be an array of steps")
    for step in steps_spec:
        _validate_step(step)

    write_kwargs = _write_kwargs_from(out, out_path)

    # A no-op unless the caller opened a provenance scope around this call
    # (see meshioplusplus._provenance) -- the pipeline never opens one
    # itself, matching the C++ engine's own posture (opt-in everywhere).
    in_fmt = (
        inp.get("Format")
        or (_filetypes_from_path(pathlib.Path(str(in_path))) or [""])[0]
    )
    _prov_set_source(str(in_path), in_fmt)

    mesh = read(in_path, file_format=inp.get("Format") or None, **read_kwargs)

    steps = []
    for step in steps_spec:
        mesh = _apply_step(mesh, step, steps, warnings)

    out_fmt = (
        out.get("Format")
        or (_filetypes_from_path(pathlib.Path(str(out_path))) or [""])[0]
    )
    out_encoding = out.get("Encoding", "default")
    _prov_set_target(
        out_fmt,
        "" if out_encoding in (None, "default") else out_encoding,
        out.get("Codec") or "",
        out.get("FloatFormat") or "",
    )

    write(out_path, mesh, file_format=out.get("Format") or None, **write_kwargs)
    return {"steps": steps, "warnings": warnings}
