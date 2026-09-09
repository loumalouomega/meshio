"""The pure payload layer of the MCP server: one function per tool.

Everything an MCP tool does lives here, and nothing here imports the ``mcp``
SDK — this module runs on every Python the package supports (>= 3.8) and is
tested in the default CI matrix with the ``[mcp]`` extra absent, exactly the
way :mod:`meshioplusplus._interop`'s payload layer is tested without PyVista.
The FastMCP registration in :mod:`meshioplusplus.mcp._server` is a thin,
typed wrapper around the functions below.

Design rules (the contracts the tests pin):

* **Stateless and file-path based.** Every tool takes input path(s), operates
  through the public :mod:`meshioplusplus` API, writes output path(s), and
  returns a JSON-safe ``dict`` report. No mesh ever lives between calls.
* **Strict JSON.** Every report survives ``json.dumps(..., allow_nan=False)``:
  :func:`_json_safe` converts numpy scalars/arrays, replaces non-finite
  values with ``None`` (counted in ``non_finite_replaced``), and truncates
  arrays longer than ``_MAX_ARRAY`` to a ``{"truncated": ...}`` wrapper.
* **Optional path sandbox.** With a root configured (``--root`` /
  ``MESHIOPLUSPLUS_MCP_ROOT``), every path is realpath-resolved and must stay
  inside it; violations raise a clean ``ValueError``, never a traceback the
  client cannot act on. Without a root, paths resolve against the CWD.
* **``TOOL_REGISTRY`` is the single source of truth.** ``_server`` registers
  from it and ``tests/python/test_mcp.py``'s parity guard asserts every public
  operation in ``meshioplusplus.__all__`` is claimed by some tool's ``wraps``.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from collections import OrderedDict

import numpy as np

from .. import (
    GeometryGuard,
    GridArray,
    GridSpec,
    agglomerate,
    attach_quality,
    cell_data_to_point_data,
    clean,
    compute_bandwidth,
    compute_curvature,
    compute_quality,
    compute_sdf,
    compute_stats,
    conservative_interpolate,
    convert_cells,
    crop,
    data_calc,
    data_condition,
    data_info,
    data_integrate,
    data_manage,
    decimate,
    decimate_volume,
    diff,
    distance_to_surface,
    estimate_error,
    extract_skin,
    extract_surface,
    geometry_descriptors,
    gradient,
    grid,
    hessian,
    interpolate,
    isosurface,
    merge,
    optimize_volume,
    partition,
    point_data_to_cell_data,
    power_spectrum,
    proximity_graph,
    read,
    read_metadata,
    refine,
    remesh,
    remesh_volume,
    reorder,
    repair,
    resample_grid,
    run_pipeline,
    sample_distance,
    sample_grid,
    scatter_grid,
)
from .. import screenshot as _screenshot_fn
from .. import (
    shrinkwrap,
)
from .. import slice as _slice_op
from .. import (
    smooth,
    sniff_format,
    sobolev_deform,
    split,
    subdivide,
    subsample_points,
    surface_watertight_check,
    transform,
    undo_green,
    voxelize,
    write,
)
from .. import write_parquet as _write_parquet_fn
from .._helpers import _filetypes_from_path, formats

# --------------------------------------------------------------------------- #
# Path sandbox                                                                #
# --------------------------------------------------------------------------- #
_ROOT = os.environ.get("MESHIOPLUSPLUS_MCP_ROOT") or None

# Report-size caps: arrays longer than _MAX_ARRAY become a truncation wrapper
# carrying the first _PREVIEW values; region entry previews reuse _PREVIEW.
_MAX_ARRAY = 1000
_PREVIEW = 50


def set_root(root):
    """Confine every tool's paths to ``root`` (``None`` lifts the sandbox)."""
    global _ROOT
    _ROOT = os.path.realpath(str(root)) if root else None


def get_root():
    return _ROOT


# Where training jobs land (doc/dashboard.md): `<root or cwd>/runs` unless the
# server's --runs-dir says otherwise. Job ids are validated by the manager, so
# a job path can never leave this directory.
_RUNS_DIR = None
_WEBHOOK = None
_MANAGERS = {}


def set_runs_dir(path):
    global _RUNS_DIR
    _RUNS_DIR = os.path.realpath(str(path)) if path else None
    _MANAGERS.clear()


def set_webhook(url):
    """The URL a terminal job is POSTed to (the server's ``--webhook``).

    Deliberately a server-side setting rather than a spec key or a tool
    parameter: a URL supplied by a client and fetched by the server is
    server-side request forgery by design.
    """
    global _WEBHOOK
    _WEBHOOK = str(url) if url else None
    _MANAGERS.clear()


def get_webhook():
    return _WEBHOOK


def get_runs_dir():
    return _RUNS_DIR or os.path.join(
        _ROOT if _ROOT is not None else os.getcwd(), "runs"
    )


def _jobs_manager():
    from ._jobs import JobManager

    runs = get_runs_dir()
    manager = _MANAGERS.get(runs)
    if manager is None:
        manager = _MANAGERS[runs] = JobManager(runs, webhook=_WEBHOOK)
    return manager


def _resolve(path, must_exist=False, for_write=False):
    """Resolve a client-supplied path, enforcing the sandbox when configured."""
    raw = os.path.expanduser(str(path))
    base = _ROOT if _ROOT is not None else os.getcwd()
    resolved = os.path.realpath(os.path.join(base, raw))
    if _ROOT is not None:
        root = os.path.realpath(_ROOT)
        if resolved != root and not resolved.startswith(root + os.sep):
            raise ValueError(
                f"meshio++: mcp: path '{path}' resolves outside the configured "
                f"root '{root}'"
            )
    # A directory counts as existing: `pmsh` and `zarr` inputs ARE directories
    # (the `openfoam` shape). A directory that is not one of those still fails,
    # just later and with the format's own error rather than "not found".
    if must_exist and not (os.path.isfile(resolved) or os.path.isdir(resolved)):
        raise ValueError(f"meshio++: mcp: input file not found: '{resolved}'")
    if for_write:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return resolved


# --------------------------------------------------------------------------- #
# Strict-JSON sanitizer                                                       #
# --------------------------------------------------------------------------- #
def _sanitize(obj, max_array, preview, stats):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            stats["replaced"] += 1
            return None
        return obj
    if isinstance(obj, np.generic):
        return _sanitize(obj.item(), max_array, preview, stats)
    if isinstance(obj, np.ndarray):
        if obj.size > max_array:
            head = obj.reshape(-1)[:preview]
            return {
                "truncated": True,
                "size": int(obj.size),
                "shape": [int(s) for s in obj.shape],
                "dtype": str(obj.dtype),
                "preview": _sanitize(head.tolist(), max_array, preview, stats),
            }
        return _sanitize(obj.tolist(), max_array, preview, stats)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, max_array, preview, stats) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, max_array, preview, stats) for v in obj]
    return str(obj)


def _json_safe(obj, max_array=_MAX_ARRAY, preview=_PREVIEW):
    """Return a strict-JSON copy of ``obj`` (numpy converted, NaN/Inf -> None).

    A ``non_finite_replaced`` count is added to dict results when any value
    had to be replaced, so lossy sanitization is visible rather than silent.
    """
    stats = {"replaced": 0}
    out = _sanitize(obj, int(max_array), int(preview), stats)
    if stats["replaced"] and isinstance(out, dict):
        out["non_finite_replaced"] = stats["replaced"]
    return out


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _load(input_path, input_format=None, **read_kwargs):
    return read(
        _resolve(input_path, must_exist=True), file_format=input_format, **read_kwargs
    )


def _store(mesh, output_path, output_format=None, **write_kwargs):
    resolved = _resolve(output_path, for_write=True)
    write(resolved, mesh, file_format=output_format, **write_kwargs)
    return resolved


def _mesh_summary(mesh):
    return {
        "num_points": int(len(mesh.points)),
        "num_cells": int(sum(len(b) for b in mesh.cells)),
        "cell_blocks": [{"type": b.type, "num_cells": int(len(b))} for b in mesh.cells],
        "point_data": sorted(mesh.point_data),
        "cell_data": sorted(mesh.cell_data),
        "field_data": sorted(mesh.field_data),
        "num_regions": len(mesh.regions),
    }


def _result(output_path, mesh, **extra):
    report = {"output_path": output_path}
    report.update(_mesh_summary(mesh))
    report.update(extra)
    return _json_safe(report)


def _safe_key(key):
    return re.sub(r"[^\w.+-]", "_", str(key)) or "_"


# --------------------------------------------------------------------------- #
# Inspection tools (read-only)                                                #
# --------------------------------------------------------------------------- #
def tool_formats():
    """List readable/writable formats and the extension -> formats map."""
    return _json_safe(formats_payload())


def formats_payload():
    """The public ``meshioplusplus.formats()`` answer, verbatim."""
    return formats()


def tool_sniff(input_path):
    """Identify a mesh file's format from its leading bytes and extension."""
    resolved = _resolve(input_path, must_exist=True)
    sniffed = sniff_format(resolved)
    try:
        from_extension = _filetypes_from_path(pathlib.Path(resolved))
    except Exception:
        from_extension = []
    return _json_safe(
        {"format": sniffed or None, "from_extension": list(from_extension)}
    )


def tool_info(input_path, file_format=None):
    """Summarize a mesh file without loading its heavy arrays (fast)."""
    meta = read_metadata(_resolve(input_path, must_exist=True), file_format=file_format)
    return _json_safe(meta)


def tool_stats(input_path, file_format=None, time_step=0):
    """Geometric statistics: bbox, centroid, areas/volumes, inverted cells."""
    mesh = _load(input_path, file_format, time_step=time_step)
    return _json_safe(compute_stats(mesh))


def tool_quality(input_path, file_format=None, output_path=None, output_format=None):
    """Per-metric mesh-quality summaries; optionally write the annotated mesh.

    The per-cell metric arrays are omitted from the report (they scale with the
    mesh); pass ``output_path`` to write a copy of the mesh carrying them as
    ``quality:<metric>`` cell_data instead.
    """
    mesh = _load(input_path, file_format)
    report = compute_quality(mesh)
    report.pop("cell_arrays", None)
    report["cell_arrays"] = (
        "omitted; pass output_path to write a mesh with quality:<metric> cell_data"
    )
    if output_path is not None:
        annotated = attach_quality(mesh)
        report["output_path"] = _store(annotated, output_path, output_format)
    return _json_safe(report)


def tool_data_info(input_path, file_format=None):
    """Describe every data array: dtype, shape, ranges, NaN/Inf counts."""
    mesh = _load(input_path, file_format)
    return _json_safe({"arrays": data_info(mesh)})


def tool_data_integrate(input_path, file_format=None, arrays=None):
    """Cell-measure-weighted total/mean of cell_data arrays, whole mesh and
    per named Cell region."""
    mesh = _load(input_path, file_format)
    return _json_safe({"arrays": data_integrate(mesh, arrays=arrays)})


def tool_regions(input_path, file_format=None):
    """List the mesh's named regions (point/cell/side groups)."""
    mesh = _load(input_path, file_format)
    regions = [
        {
            "name": r.name,
            "kind": r.kind,
            "dim": r.dim,
            "tag": r.tag,
            "num_entries": len(r),
            "entries_preview": np.asarray(r.entries).reshape(len(r), -1)[:_PREVIEW],
        }
        for r in mesh.regions
    ]
    return _json_safe({"regions": regions})


def tool_bandwidth(input_path, file_format=None):
    """Node-numbering bandwidth (max per-cell node-index spread)."""
    mesh = _load(input_path, file_format)
    return _json_safe({"bandwidth": compute_bandwidth(mesh)})


def tool_data_preview(
    input_path, array, location="point", offset=0, limit=100, file_format=None
):
    """Return a bounded window of one data array's values."""
    mesh = _load(input_path, file_format)
    if location == "point":
        table = mesh.point_data
    elif location == "cell":
        table = mesh.cell_data
    elif location == "field":
        table = mesh.field_data
    else:
        raise ValueError(
            f"meshio++: mcp: unknown location '{location}' "
            "(expected 'point', 'cell' or 'field')"
        )
    if array not in table:
        raise ValueError(
            f"meshio++: mcp: no {location} data array named '{array}' "
            f"(available: {sorted(table)})"
        )
    if location == "cell":
        blocks = [np.asarray(a) for a in table[array]]
        values = (
            np.concatenate([b.reshape(len(b), -1) for b in blocks], axis=0)
            if blocks
            else np.empty((0,))
        )
    else:
        values = np.asarray(table[array])
    offset = max(int(offset), 0)
    limit = max(int(limit), 0)
    window = values[offset : offset + limit]
    return _json_safe(
        {
            "location": location,
            "name": array,
            "dtype": str(values.dtype),
            "shape": [int(s) for s in values.shape],
            "num_rows": int(len(values)),
            "offset": offset,
            "values": _json_safe(window, max_array=window.size + 1),
        }
    )


def tool_diff(
    path_a,
    path_b,
    format_a=None,
    format_b=None,
    atol=1e-12,
    rtol=1e-9,
    unordered=False,
    max_reported=10,
):
    """Compare two mesh files; the report carries a verdict and per-section detail."""
    mesh_a = _load(path_a, format_a)
    mesh_b = _load(path_b, format_b)
    report = diff(
        mesh_a,
        mesh_b,
        atol=atol,
        rtol=rtol,
        unordered=unordered,
        max_reported=max_reported,
    )
    report["equal"] = report.get("verdict") != "different"
    return _json_safe(report)


# --------------------------------------------------------------------------- #
# Conversion                                                                  #
# --------------------------------------------------------------------------- #
# The ascii/binary/compression writer-kwargs mirror the CLI's `ascii`,
# `binary`, `compress` and `decompress` verbs (src/python/meshioplusplus/_cli/)
# so both surfaces stay in step on which formats have which variant.
_ASCII_VARIANT = {
    "ansys": {"binary": False},
    "flac3d": {"binary": False},
    "gmsh": {"binary": False},
    "mdpa": {"binary": False},
    "ply": {"binary": False},
    "stl": {"binary": False},
    "vtk": {"binary": False},
    "vtu": {"binary": False},
    "xdmf": {"data_format": "XML"},
}
_BINARY_VARIANT = {
    "ansys": {"binary": True},
    "flac3d": {"binary": True},
    "gmsh": {"binary": True},
    "mdpa": {"binary": True},
    "ply": {"binary": True},
    "stl": {"binary": True},
    "vtk": {"binary": True},
    "vtu": {"binary": True},
    "xdmf": {"data_format": "HDF"},
}
_BLOCK_CODECS = ("zlib", "lz4", "zstd", "lzma")


def _variant_kwargs(out_fmt, mode, compression):
    kwargs = {}
    if mode not in ("auto", "ascii", "binary"):
        raise ValueError(
            f"meshio++: mcp: unknown mode '{mode}' "
            "(expected 'auto', 'ascii' or 'binary')"
        )
    if mode == "ascii" and compression is not None and compression != "none":
        raise ValueError(
            "meshio++: mcp: mode='ascii' cannot be combined with compression"
        )
    if mode != "auto":
        table = _ASCII_VARIANT if mode == "ascii" else _BINARY_VARIANT
        if out_fmt not in table:
            raise ValueError(
                f"meshio++: mcp: format '{out_fmt}' has no {mode} variant "
                f"(supported: {sorted(table)})"
            )
        kwargs.update(table[out_fmt])
    if compression is None:
        return kwargs
    if compression in _BLOCK_CODECS:
        if out_fmt not in ("vti", "vtu", "vtp"):
            raise ValueError(
                f"meshio++: mcp: compression '{compression}' selects the VTK XML "
                "block codec and only vti/vtu/vtp have one"
            )
        kwargs.update({"binary": True, "compression": compression})
    elif compression == "gzip":
        if out_fmt in ("cgns", "h5m"):
            kwargs.update({"compression": "gzip", "compression_opts": 4})
        elif out_fmt == "xdmf":
            kwargs.update(
                {"data_format": "HDF", "compression": "gzip", "compression_opts": 4}
            )
        else:
            raise ValueError(
                f"meshio++: mcp: gzip compression applies to cgns/h5m/xdmf, "
                f"not '{out_fmt}'"
            )
    elif compression == "none":
        if out_fmt in ("cgns", "h5m"):
            kwargs.update({"compression": None})
        elif out_fmt == "vtu":
            kwargs.update({"binary": True, "compression": None})
        elif out_fmt == "xdmf":
            kwargs.update({"data_format": "HDF", "compression": None})
        else:
            raise ValueError(
                f"meshio++: mcp: compression='none' applies to cgns/h5m/vtu/xdmf, "
                f"not '{out_fmt}'"
            )
    else:
        raise ValueError(
            f"meshio++: mcp: unknown compression '{compression}' (expected one of "
            f"{list(_BLOCK_CODECS)}, 'gzip' or 'none')"
        )
    return kwargs


def tool_convert(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    points_only=False,
    arrays=None,
    time_step=0,
    mode="auto",
    compression=None,
):
    """Convert between mesh formats, optionally selecting variant/compression."""
    mesh = _load(
        input_path,
        input_format,
        points_only=points_only,
        arrays=arrays,
        time_step=time_step,
    )
    out_fmt = output_format
    if out_fmt is None:
        try:
            candidates = _filetypes_from_path(pathlib.Path(str(output_path)))
            out_fmt = candidates[0] if candidates else None
        except Exception:
            out_fmt = None
    write_kwargs = {}
    if mode != "auto" or compression is not None:
        if out_fmt is None:
            raise ValueError(
                "meshio++: mcp: cannot infer the output format needed to apply "
                "mode/compression; pass output_format explicitly"
            )
        write_kwargs = _variant_kwargs(out_fmt, mode, compression)
    resolved = _store(mesh, output_path, output_format, **write_kwargs)
    return _result(resolved, mesh, output_format=out_fmt)


def _resolve_pattern(pattern):
    """Resolve a glob pattern inside the sandbox and return its matched files.

    ``_resolve(..., must_exist=True)`` uses ``os.path.isfile`` and so rejects a
    pattern outright, and a pattern's *directory* component is the obvious
    escape hatch (``../*.vtu``, ``/etc/*``) -- so the directory is resolved and
    containment-checked first, and every matched file then goes back through
    ``_resolve`` individually.
    """
    from .._sequence import glob_match, is_sample_path

    raw = os.path.expanduser(str(pattern))
    # os.path.split, not a manual os.sep rpartition: on Windows os.sep is
    # '\\' alone, while ntpath (and every caller here) also accepts '/' --
    # a manual rpartition(os.sep) would treat the whole of "../*.vtu" as the
    # filename part, silently skipping the containment check on the '..'
    # component instead of rejecting it.
    head, base = os.path.split(raw)
    if "*" in head or "?" in head:
        raise ValueError(
            "meshio++: mcp: the directory part of a pattern is taken literally, "
            f"so '{head}' cannot contain '*' or '?'"
        )
    # Resolve the directory through the same containment check as any path.
    directory = _resolve(head or ".")
    if not os.path.isdir(directory):
        raise ValueError(f"meshio++: mcp: pattern directory not found: '{directory}'")
    matched = sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if glob_match(base, name) and is_sample_path(os.path.join(directory, name))
    )
    if not matched:
        raise ValueError(f"meshio++: mcp: pattern '{pattern}' matched no files")
    return [_resolve(path, must_exist=True) for path in matched]


def tool_sequence(
    input_pattern=None,
    input_paths=None,
    output_path=None,
    mode=None,
    times=None,
    time_from="auto",
    input_format=None,
    output_format=None,
):
    """Run a multi-file / transient sequence: fan-in, fan-out or N->N."""
    from .. import run_sequence_pipeline, sequence_entries
    from .._sequence import pattern_has_token

    if (input_pattern is None) == (input_paths is None):
        raise ValueError(
            "meshio++: mcp: give exactly one of input_pattern or input_paths"
        )
    if not output_path:
        raise ValueError("meshio++: mcp: output_path is required")

    if input_pattern is not None:
        resolved_in = _resolve_pattern(input_pattern)
    else:
        resolved_in = [_resolve(p, must_exist=True) for p in input_paths]

    # The output is a pattern for a fan-out; _resolve its directory the same way
    # so a '{step}' path cannot escape the sandbox either.
    if pattern_has_token(output_path):
        probe = str(output_path).replace("{step}", "0").replace("{index}", "0")
        resolved_probe = _resolve(probe, for_write=True)
        resolved_out = os.path.join(
            os.path.dirname(resolved_probe), os.path.basename(str(output_path))
        )
    else:
        resolved_out = _resolve(output_path, for_write=True)

    doc = {
        "Version": 1,
        "Input": {"Paths": resolved_in},
        "Output": {"Path": resolved_out},
    }
    if mode:
        doc["Mode"] = mode
    if times is not None:
        doc["Input"]["Times"] = [float(t) for t in times]
    if time_from and time_from != "auto":
        doc["Input"]["TimeFrom"] = time_from
    if input_format:
        doc["Input"]["Format"] = input_format
    if output_format:
        doc["Output"]["Format"] = output_format

    entries = sequence_entries(
        resolved_in,
        file_format=input_format,
        times=[float(t) for t in times] if times is not None else None,
        time_from=time_from,
    )
    report = run_sequence_pipeline(doc)
    report["steps_plan"] = [
        {"path": e["path"], "time": e["time"], "time_source": e["time_source"]}
        for e in entries
    ]
    report["output_path"] = resolved_out
    return _json_safe(report)


def tool_pipeline(settings_path, input_path=None, output_path=None):
    """Run a settings.json operation pipeline (read -> ops chain -> write)."""
    with open(_resolve(settings_path, must_exist=True), "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError("meshio++: mcp: the settings document must be an object")
    # Sandbox the paths INSIDE the settings too, not just the settings file --
    # a document naming /etc/passwd must fail the same way a path argument
    # would. Overrides take precedence exactly like the CLI's --input/--output.
    raw_in = input_path if input_path is not None else doc.get("Input", {}).get("Path")
    raw_out = (
        output_path if output_path is not None else doc.get("Output", {}).get("Path")
    )
    if not raw_in:
        raise ValueError("meshio++: pipeline: Input.Path is required")
    if not raw_out:
        raise ValueError("meshio++: pipeline: Output.Path is required")
    resolved_in = _resolve(raw_in, must_exist=True)
    resolved_out = _resolve(raw_out, for_write=True)
    report = run_pipeline(doc, input_path=resolved_in, output_path=resolved_out)
    report["output_path"] = resolved_out
    return _json_safe(report)


# --------------------------------------------------------------------------- #
# Mesh operations (path in -> path out + report)                              #
# --------------------------------------------------------------------------- #
def tool_extract_surface(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    record_parent_ids=False,
):
    """Extract the boundary surface (3D -> faces, 2D -> edges)."""
    mesh = _load(input_path, input_format)
    out = extract_surface(mesh, record_parent_ids=record_parent_ids)
    return _result(_store(out, output_path, output_format), out)


def tool_extract_skin(
    input_path, output_path, input_format=None, output_format=None, linearize=False
):
    """Extract the outer skin of a volume mesh."""
    mesh = _load(input_path, input_format)
    out = extract_skin(mesh, linearize=linearize)
    return _result(_store(out, output_path, output_format), out)


def tool_reorder(
    input_path, output_path, input_format=None, output_format=None, method="rcm"
):
    """Renumber nodes/cells (rcm | morton | hilbert); reports bandwidth change."""
    mesh = _load(input_path, input_format)
    before = compute_bandwidth(mesh)
    out = reorder(mesh, method=method)
    after = compute_bandwidth(out)
    return _result(
        _store(out, output_path, output_format),
        out,
        method=method,
        bandwidth_before=before,
        bandwidth_after=after,
    )


def tool_clean(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    weld=False,
    atol=1e-8,
    remove_orphans=True,
    drop_degenerate=True,
    drop_duplicate_cells=True,
):
    """Weld/prune/de-duplicate a mesh; reports what was removed."""
    mesh = _load(input_path, input_format)
    out, report = clean(
        mesh,
        weld=weld,
        atol=atol,
        remove_orphans=remove_orphans,
        drop_degenerate=drop_degenerate,
        drop_duplicate_cells=drop_duplicate_cells,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_crop(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    bbox=None,
    plane_origin=None,
    plane_normal=None,
    where_array=None,
    where_compare="<",
    where_value=0.0,
    mode="all",
    record_ids=False,
):
    """Crop to a bounding box, a half-space, or a cell_data comparison."""
    mesh = _load(input_path, input_format)
    plane = None
    if plane_origin is not None or plane_normal is not None:
        if plane_origin is None or plane_normal is None:
            raise ValueError(
                "meshio++: mcp: crop needs both plane_origin and plane_normal"
            )
        plane = (plane_origin, plane_normal)
    where = None
    if where_array is not None:
        where = (where_array, where_compare, where_value)
    out = crop(
        mesh,
        bbox=bbox,
        plane=plane,
        where=where,
        mode=mode,
        record_ids=record_ids,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_slice(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    origin=(0.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
    record_parent_ids=False,
):
    """Planar cross-section (one topological dimension below the input)."""
    mesh = _load(input_path, input_format)
    out = _slice_op(
        mesh, origin=origin, normal=normal, record_parent_ids=record_parent_ids
    )
    return _result(_store(out, output_path, output_format), out)


def tool_isosurface(
    input_path,
    output_path,
    array,
    isovalues,
    input_format=None,
    output_format=None,
    component=None,
    record_parent_ids=False,
):
    """Contour a point_data scalar at one or more isovalues."""
    mesh = _load(input_path, input_format)
    out = isosurface(
        mesh,
        array,
        isovalues,
        component=component,
        record_parent_ids=record_parent_ids,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_grid(
    output_path,
    dims,
    output_format=None,
    origin=None,
    spacing=None,
    max_cells=20000000,
):
    """Generate a regular hexahedron lattice; the only tool that reads no input."""
    out = grid(
        dims,
        origin=(0.0, 0.0, 0.0) if origin is None else origin,
        spacing=(1.0, 1.0, 1.0) if spacing is None else spacing,
        max_cells=max_cells,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_voxelize(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.0,
    fill="all",
    attach_occupancy=False,
    max_cells=20000000,
    sign="pseudonormal",
):
    """Build a regular grid around a mesh; optionally keep only its surface or interior."""
    mesh = _load(input_path, input_format)
    out, report = voxelize(
        mesh,
        resolution=resolution,
        cell_size=cell_size,
        bounds=bounds,
        padding=padding,
        padding_relative=padding_relative,
        fill=fill,
        attach_occupancy=attach_occupancy,
        max_cells=max_cells,
        sign=sign,
        watertight_check="off",
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def _grid_spec_report(spec):
    return {
        "origin": [float(v) for v in spec.origin],
        "spacing": [float(v) for v in spec.spacing],
        "dims": [int(v) for v in spec.dims],
        "shape": list(spec.shape),
        "layout": "channels_first_zyx",
    }


def tool_grid_sample(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.0,
    fields=None,
    extrapolate=False,
    fill_value=0.0,
    max_cells=20000000,
):
    """Sample a mesh's point data onto a regular grid, written as a lattice mesh."""
    mesh = _load(input_path, input_format)
    spec = GridSpec.from_mesh(
        mesh,
        resolution=resolution,
        cell_size=cell_size,
        bounds=bounds,
        padding=padding,
        padding_relative=padding_relative,
        max_cells=max_cells,
    )
    array = sample_grid(
        mesh,
        spec,
        fields=fields,
        extrapolate=extrapolate,
        fill_value=fill_value,
    )
    out = array.to_mesh()
    return _result(
        _store(out, output_path, output_format),
        out,
        channels=list(array.channels),
        coverage=array.coverage,
        grid=_grid_spec_report(spec),
    )


def tool_grid_scatter(
    grid_path,
    target_path,
    output_path,
    grid_format=None,
    target_format=None,
    output_format=None,
    fields=None,
    on_conflict="error",
):
    """Write a grid's fields back onto a mesh's points by trilinear interpolation."""
    array = GridArray.from_mesh(_load(grid_path, grid_format), fields=fields)
    target = _load(target_path, target_format)
    out = scatter_grid(array, target, on_conflict=on_conflict)
    return _result(
        _store(out, output_path, output_format),
        out,
        channels=list(array.channels),
        grid=_grid_spec_report(array.spec),
    )


def tool_grid_resample(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    factor=None,
    resolution=None,
    fields=None,
):
    """Resample a grid onto a finer or coarser one - the trilinear upsampling baseline."""
    array = GridArray.from_mesh(_load(input_path, input_format), fields=fields)
    if (factor is None) == (resolution is None):
        raise ValueError(
            "meshio++: grid_resample: give exactly one of factor and resolution"
        )
    if factor is not None:
        target = array.spec.upscale(factor)
    else:
        lo, hi = array.spec.bounds
        dims = np.asarray(resolution, dtype=np.int64).reshape(-1)
        if dims.size != 3:
            raise ValueError(
                "meshio++: grid_resample: resolution must be three cell counts"
            )
        target = GridSpec(origin=lo, spacing=(hi - lo) / dims.astype(float), dims=dims)
    values = resample_grid(array.values, array.spec, target)
    out = GridArray(values, target, array.channels, dict(array.schema)).to_mesh()
    return _result(
        _store(out, output_path, output_format),
        out,
        channels=list(array.channels),
        source_grid=_grid_spec_report(array.spec),
        grid=_grid_spec_report(target),
        scaling_factor=array.spec.scaling_factor(target),
    )


def tool_grid_power_spectrum(input_path, field, input_format=None, max_bins=256):
    """The azimuthally averaged power spectrum of one field on a regular grid."""
    array = GridArray.from_mesh(_load(input_path, input_format))
    names = [c for c in array.channels if c == field or c.startswith(field + "_")]
    if not names:
        raise ValueError(
            f"meshio++: grid_power_spectrum: no channel named {field!r} "
            f"(have {list(array.channels)})"
        )
    values = np.stack([array.channel(n) for n in names], axis=0)
    ps = power_spectrum(values, array.spec)
    keep = (
        min(int(max_bins), len(ps.power))
        if max_bins and max_bins > 0
        else len(ps.power)
    )
    return _json_safe(
        {
            "field": field,
            "channels": names,
            "units": ps.units,
            "wavenumber": ps.wavenumber[:keep],
            "power": ps.power[:keep],
            "counts": ps.counts[:keep],
            "num_bins": int(len(ps.power)),
            "total_power": float(ps.power.sum()),
            "grid": _grid_spec_report(array.spec),
        }
    )


def tool_subsample(
    input_path,
    output_path,
    count,
    input_format=None,
    output_format=None,
    method="farthest",
    seed=0,
    start=0,
    bounds=None,
    record_ids=False,
):
    """Reduce a mesh to a point cloud of exactly `count` points under a token budget."""
    mesh = _load(input_path, input_format)
    out = subsample_points(
        mesh,
        int(count),
        method=method,
        seed=seed,
        start=start,
        bounds=bounds,
        record_ids=record_ids,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        method=method,
        count=int(len(out.points)),
        num_source_points=int(len(mesh.points)),
    )


def tool_proximity_graph(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    method="radius",
    radius=None,
    max_neighbors=None,
    box_size=None,
    kind="node",
):
    """Build a radius or k-nearest-neighbour graph over a mesh's points or cell centroids."""
    import numpy as np

    from .._mesh import Mesh
    from .._proximity import _graph_positions

    mesh = _load(input_path, input_format)
    edges = proximity_graph(
        mesh,
        method=method,
        radius=radius,
        max_neighbors=max_neighbors,
        box_size=box_size,
        kind=kind,
    )
    points = _graph_positions(mesh, kind)
    half = edges[:, edges[0] < edges[1]]
    degree = np.bincount(edges[0], minlength=len(points)).astype(np.int64)
    out = Mesh(
        points,
        [("line", np.ascontiguousarray(half.T))],
        point_data={"degree": degree},
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        method=method,
        num_vertices=int(len(points)),
        num_edges=int(half.shape[1]),
        degree_min=int(degree.min()) if degree.size else 0,
        degree_mean=float(degree.mean()) if degree.size else 0.0,
        degree_max=int(degree.max()) if degree.size else 0,
        num_isolated=int((degree == 0).sum()),
    )


def tool_distance_to_surface(
    input_path,
    surface_path,
    output_path,
    input_format=None,
    surface_format=None,
    output_format=None,
    sign="pseudonormal",
    location="corner",
    band=0.0,
    record_inside=False,
    record_closest_cell=False,
):
    """Attach the signed distance from a mesh's points (or cell centres) to a surface."""
    mesh = _load(input_path, input_format)
    surface = _load(surface_path, surface_format)
    out, report = distance_to_surface(
        mesh,
        surface,
        sign=sign,
        location=location,
        band=band,
        record_inside=record_inside,
        record_closest_cell=record_closest_cell,
        watertight_check="off",
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_banded=report["num_banded"],
        **report["quality"],
    )


def tool_compute_sdf(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    structure="voxel",
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.1,
    root_resolution=8,
    max_depth=4,
    band_cells=1.0,
    max_cells=20000000,
    sign="pseudonormal",
    location="corner",
    band=0.0,
):
    """Generate a grid over a surface and fill it with signed distances."""
    surface = _load(input_path, input_format)
    out, report = compute_sdf(
        surface,
        structure=structure,
        resolution=resolution,
        cell_size=cell_size,
        bounds=bounds,
        padding=padding,
        padding_relative=padding_relative,
        root_resolution=root_resolution,
        max_depth=max_depth,
        band_cells=band_cells,
        max_cells=max_cells,
        sign=sign,
        location=location,
        band=band,
        watertight_check="off",
        return_report=True,
    )
    quality = report.pop("quality")
    return _result(_store(out, output_path, output_format), out, **report, **quality)


def tool_surface_watertight_check(input_path, input_format=None):
    """Report a surface's boundary, non-manifold, inconsistent and degenerate counts."""
    return _json_safe(dict(surface_watertight_check(_load(input_path, input_format))))


def tool_sample_distance(
    input_path,
    points,
    input_format=None,
    sign="pseudonormal",
    band=0.0,
):
    """Signed distances from a list of [x, y, z] points to a surface."""
    surface = _load(input_path, input_format)
    values = sample_distance(
        surface, points, sign=sign, band=band, watertight_check="off"
    )
    return _json_safe({"distances": values})


def tool_gradient(
    input_path,
    output_path,
    array,
    input_format=None,
    output_format=None,
    operator="gradient",
    method="green-gauss",
    location="cell",
    output=None,
    component=None,
    overwrite=False,
):
    """Gradient, divergence or curl of a point_data field."""
    mesh = _load(input_path, input_format)
    out, report = gradient(
        mesh,
        array,
        operator=operator,
        method=method,
        location=location,
        output=output,
        component=component,
        overwrite=overwrite,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_skipped=int(report["num_skipped"]),
        num_fallback=int(report["num_fallback"]),
    )


def tool_hessian(
    input_path,
    output_path,
    array,
    input_format=None,
    output_format=None,
    method="green-gauss",
    location="cell",
    output=None,
    overwrite=False,
):
    """Hessian (second derivative) of a scalar point_data field --
    gradient's companion one order further, a composition of two gradient
    calls."""
    mesh = _load(input_path, input_format)
    out, report = hessian(
        mesh,
        array,
        method=method,
        location=location,
        output=output,
        overwrite=overwrite,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_skipped=int(report["num_skipped"]),
        num_fallback=int(report["num_fallback"]),
    )


def tool_curvature(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    mean=True,
    gaussian=True,
    dual_area="mixed-voronoi",
    include_boundary=False,
    record_area=False,
    record_principal=False,
    region="",
):
    """Per-vertex mean and Gaussian curvature of a surface, by the angle defect
    (K) and the cotangent Laplace-Beltrami operator (H) -- the signed
    distance's companion as a node feature. Reports total_angle_defect, which
    is 2*pi*chi exactly for a closed surface (4*pi for a sphere) whatever the
    tessellation: the cheapest check that a result is sane."""
    mesh = _load(input_path, input_format)
    out, report = compute_curvature(
        mesh,
        mean=mean,
        gaussian=gaussian,
        dual_area=dual_area,
        include_boundary=include_boundary,
        record_area=record_area,
        record_principal=record_principal,
        region=region,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_boundary=int(report["num_boundary"]),
        num_isolated=int(report["num_isolated"]),
        num_degenerate=int(report["num_degenerate"]),
        total_angle_defect=float(report["total_angle_defect"]),
        quality=report["quality"],
    )


def tool_repair(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    fix_orientation=True,
    orient_outward=True,
    fill_holes=True,
    split_non_manifold=True,
    max_hole_edges=10,
    weld_tolerance=0.0,
    record_provenance=False,
):
    """Repair a surface mesh's orientation, holes and pinched vertices -- the
    three defects `clean` does not touch. Triangles are rewound so neighbours
    agree (by the topological half-edge rule, exact across any crease),
    boundary loops of at most max_hole_edges edges are fan-filled consistently
    with the surrounding surface, bowtie vertices are duplicated, and closed
    components are oriented outward. Reports both surfaces' defect counts, so
    what was fixed and what remains are both visible."""
    mesh = _load(input_path, input_format)
    out, report = repair(
        mesh,
        fix_orientation=fix_orientation,
        orient_outward=orient_outward,
        fill_holes=fill_holes,
        split_non_manifold=split_non_manifold,
        max_hole_edges=max_hole_edges,
        weld_tolerance=weld_tolerance,
        record_provenance=record_provenance,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        quality_before=report["quality_before"],
        quality_after=report["quality_after"],
        num_flipped=int(report["num_flipped"]),
        num_components=int(report["num_components"]),
        largest_component=int(report["largest_component"]),
        num_oriented_outward=int(report["num_oriented_outward"]),
        num_unorientable=int(report["num_unorientable"]),
        num_vertices_split=int(report["num_vertices_split"]),
        num_holes_detected=int(report["num_holes_detected"]),
        num_holes_filled=int(report["num_holes_filled"]),
        num_holes_skipped=int(report["num_holes_skipped"]),
        num_faces_added=int(report["num_faces_added"]),
        num_points_added=int(report["num_points_added"]),
        points_welded=int(report["points_welded"]),
    )


def tool_shrinkwrap(
    input_path,
    target_path,
    output_path,
    input_format=None,
    target_format=None,
    output_format=None,
    offset=0.0,
    max_distance=0.0,
    weights="",
    target_region="",
    normal_weight="angle",
    record_distance=False,
    record_closest_cell=False,
):
    """Project every (selected) point of the input mesh onto the surface of
    the target: one projection, no iteration, optionally offset along the hit
    feature's pseudonormal. Every point of the source moves whatever cells it
    carries; only the target must be a surface. A point farther than
    max_distance is left alone and counted."""
    mesh = _load(input_path, input_format)
    target = _load(target_path, target_format)
    out, report = shrinkwrap(
        mesh,
        target,
        offset=offset,
        max_distance=max_distance,
        weights=weights or None,
        target_region=target_region,
        normal_weight=normal_weight,
        record_distance=record_distance,
        record_closest_cell=record_closest_cell,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_projected=int(report["num_projected"]),
        num_missed=int(report["num_missed"]),
        num_skipped=int(report["num_skipped"]),
        max_displacement=float(report["max_displacement"]),
        quality=report["quality"],
    )


def tool_sobolev_deform(
    input_path,
    output_path,
    array,
    length_scale,
    input_format=None,
    output_format=None,
    fixed_points_array="",
    fix_boundary=False,
    record_filtered=False,
    max_iterations=128,
    tolerance=1e-10,
):
    """Sobolev (Helmholtz-filtered) deformation: smooth a raw per-point
    displacement field through the mesh's own P1 finite-element operators and
    move the points by the result -- a low-pass filter whose cutoff wavelength
    is length_scale, which turns a jagged displacement into one a mesh can
    follow without tangling. Every top-dimensional block must be a linear
    simplex. Non-convergence is reported, never raised."""
    mesh = _load(input_path, input_format)
    out, report = sobolev_deform(
        mesh,
        array,
        length_scale,
        fixed_points=fixed_points_array or None,
        fix_boundary=fix_boundary,
        record_filtered=record_filtered,
        max_iterations=max_iterations,
        tolerance=tolerance,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        num_iterations=int(report["num_iterations"]),
        residual=float(report["residual"]),
        converged=bool(report["converged"]),
        num_fixed=int(report["num_fixed"]),
        num_isolated=int(report["num_isolated"]),
        max_displacement=float(report["max_displacement"]),
    )


def tool_estimate_error(
    input_path,
    output_path,
    array,
    input_format=None,
    output_format=None,
    method="zz",
    marking="none",
    marking_value=0.0,
    output=None,
    marked=None,
    overwrite=False,
):
    """ZZ recovery-based error indicator of a point_data field, plus optional
    marking (absolute/fraction/dorfler) of cells for refinement."""
    mesh = _load(input_path, input_format)
    out, report = estimate_error(
        mesh,
        array,
        method=method,
        marking=marking,
        marking_value=marking_value,
        output=output,
        marked_name=marked,
        overwrite=overwrite,
        return_report=True,
    )
    return _result(
        _store(out, output_path, output_format),
        out,
        global_error=float(report["global_error"]),
        num_skipped=int(report["num_skipped"]),
        num_marked=int(report["num_marked"]),
    )


def tool_transform(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    translate=None,
    scale=None,
    rotate_axis=None,
    rotate_degrees=None,
    matrix=None,
    scale_units=None,
    rotate_vector_data=False,
):
    """Affine-transform point coordinates (translate/scale/rotate/matrix)."""
    mesh = _load(input_path, input_format)
    rotate = None
    if rotate_axis is not None or rotate_degrees is not None:
        if rotate_axis is None or rotate_degrees is None:
            raise ValueError(
                "meshio++: mcp: transform needs both rotate_axis and rotate_degrees"
            )
        rotate = (rotate_axis, rotate_degrees)
    out = transform(
        mesh,
        translate=translate,
        scale=scale,
        rotate=rotate,
        matrix=matrix,
        scale_units=scale_units,
        rotate_vector_data=rotate_vector_data,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_convert_cells(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    mode="linearize",
    record_parent_ids=False,
):
    """Convert the element representation (linearize | simplexify | elevate)."""
    mesh = _load(input_path, input_format)
    out = convert_cells(mesh, mode=mode, record_parent_ids=record_parent_ids)
    return _result(_store(out, output_path, output_format), out)


def tool_subdivide(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    record_parent_ids=False,
):
    """Polyhedrally refine: one polyhedral child per 3D cell face."""
    mesh = _load(input_path, input_format)
    out = subdivide(mesh, record_parent_ids=record_parent_ids)
    return _result(_store(out, output_path, output_format), out)


def tool_agglomerate(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    target_group_size=8,
):
    """Polyhedrally coarsen: merge groups of cells into larger polyhedra."""
    mesh = _load(input_path, input_format)
    out = agglomerate(mesh, target_group_size=target_group_size)
    return _result(_store(out, output_path, output_format), out)


def tool_refine(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    levels=1,
    record_parent_ids=False,
    cells=None,
    region=None,
    where=None,
    closure="redgreen",
    record_levels=False,
    record_hierarchy=False,
):
    """Subdivide cells into same-type children, all or a selected subset."""
    mesh = _load(input_path, input_format)
    out = refine(
        mesh,
        levels=levels,
        record_parent_ids=record_parent_ids,
        cells=cells,
        region=region,
        where=where,
        closure=closure,
        record_levels=record_levels,
        record_hierarchy=record_hierarchy,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_decimate(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    ratio=None,
    target_faces=None,
    max_error=None,
    placement="optimal",
    preserve_boundary=True,
    preserve_features=True,
    feature_angle=30.0,
):
    """Reduce a surface mesh's face count (exactly one stopping criterion)."""
    mesh = _load(input_path, input_format)
    out, report = decimate(
        mesh,
        ratio=ratio,
        target_faces=target_faces,
        max_error=max_error,
        placement=placement,
        preserve_boundary=preserve_boundary,
        preserve_features=preserve_features,
        feature_angle=feature_angle,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_decimate_volume(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    ratio=None,
    target_cells=None,
    max_error=None,
    placement="optimal",
    preserve_boundary=False,
    preserve_features=True,
    feature_angle=30.0,
):
    """Reduce a tet mesh's cell count (exactly one stopping criterion)."""
    mesh = _load(input_path, input_format)
    out, report = decimate_volume(
        mesh,
        ratio=ratio,
        target_cells=target_cells,
        max_error=max_error,
        placement=placement,
        preserve_boundary=preserve_boundary,
        preserve_features=preserve_features,
        feature_angle=feature_angle,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_remesh(
    input_path,
    output_path,
    num_clusters,
    input_format=None,
    output_format=None,
    subdivide=None,
    subsample_ratio=10.0,
    max_subdivide=4,
    max_iterations=100,
    max_repair_passes=10,
    metric="isotropic",
    gradation=0.0,
    preserve_boundary=True,
    max_anisotropy=4.0,
):
    """Replace a surface's triangulation with a new, well-shaped one (ACVD
    clustering). The output has NO correspondence to the input -- point/cell
    data and named regions are dropped, field data is carried."""
    mesh = _load(input_path, input_format)
    out, report = remesh(
        mesh,
        num_clusters,
        subdivide=subdivide,
        subsample_ratio=subsample_ratio,
        max_subdivide=max_subdivide,
        max_iterations=max_iterations,
        max_repair_passes=max_repair_passes,
        metric=metric,
        gradation=gradation,
        preserve_boundary=preserve_boundary,
        max_anisotropy=max_anisotropy,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_remesh_volume(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.1,
    max_cells=20000000,
    max_tets=20000000,
    warp_fraction=0.35,
    watertight_check="warn",
):
    """Retetrahedralize a volume mesh (or closed surface) at a chosen
    resolution by isosurface stuffing -- the volumetric sibling of `remesh`.
    The output has NO correspondence to the input -- point/cell data and
    named regions are dropped, field data is carried. Exactly one of
    `resolution`/`cell_size` must be given."""
    mesh = _load(input_path, input_format)
    out, report = remesh_volume(
        mesh,
        resolution=resolution,
        cell_size=cell_size,
        bounds=bounds,
        padding=padding,
        padding_relative=padding_relative,
        max_cells=max_cells,
        max_tets=max_tets,
        warp_fraction=warp_fraction,
        watertight_check=watertight_check,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_optimize_volume(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    max_iterations=10,
    relocate=True,
    flip=True,
    preserve_boundary=True,
    min_improvement=1e-6,
):
    """ODT-remesh a tetrahedral mesh: raise its worst element quality by
    relocating vertices AND flipping connectivity (2-3/3-2, predicate-free).
    Tet-only and C++-core-only (no pure-Python fallback). The point set is
    invariant, so point data and named Point regions carry; cell data and
    Cell/Side regions are dropped (a flip has no cell correspondence)."""
    mesh = _load(input_path, input_format)
    out, report = optimize_volume(
        mesh,
        max_iterations=max_iterations,
        relocate=relocate,
        flip=flip,
        preserve_boundary=preserve_boundary,
        min_improvement=min_improvement,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_smooth(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    method="taubin",
    iterations=10,
    lambda_factor=-1.0,
    mu=-0.34,
    fix_boundary=True,
    preserve_features=True,
    feature_angle=30.0,
    guard_inversion=True,
):
    """Smooth point coordinates (taubin | laplacian | odt); geometry-only
    move. odt is tet-only and C++-core-only (no pure-Python fallback)."""
    mesh = _load(input_path, input_format)
    out, report = smooth(
        mesh,
        method=method,
        iterations=iterations,
        lambda_=lambda_factor,
        mu=mu,
        fix_boundary=fix_boundary,
        preserve_features=preserve_features,
        feature_angle=feature_angle,
        guard_inversion=guard_inversion,
        return_report=True,
    )
    return _result(_store(out, output_path, output_format), out, **report)


def tool_merge(
    input_paths,
    output_path,
    output_format=None,
    weld=False,
    atol=1e-8,
    source_tag=True,
    data_policy="intersection",
    drop_duplicate_cells=False,
):
    """Merge several mesh files into one (optionally welding coincident points)."""
    if not input_paths or len(input_paths) < 2:
        raise ValueError("meshio++: mcp: merge needs at least two input_paths")
    meshes = [_load(p) for p in input_paths]
    out = merge(
        meshes,
        weld=weld,
        atol=atol,
        source_tag=source_tag,
        data_policy=data_policy,
        drop_duplicate_cells=drop_duplicate_cells,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_split(
    input_path,
    input_format=None,
    by="type",
    tag=None,
    output_dir=None,
    name_template="{stem}_{key}.vtu",
):
    """Split into submeshes by type/region/component/tag; writes one file per piece."""
    resolved_in = _resolve(input_path, must_exist=True)
    mesh = read(resolved_in, file_format=input_format)
    pieces = split(mesh, by=by, tag=tag)
    stem = os.path.splitext(os.path.basename(resolved_in))[0]
    out_dir = output_dir if output_dir is not None else os.path.dirname(resolved_in)
    written = OrderedDict()
    summaries = {}
    for key, piece in pieces.items():
        name = name_template.format(stem=stem, key=_safe_key(key))
        path = _resolve(os.path.join(out_dir, name), for_write=True)
        write(path, piece)
        written[str(key)] = path
        summaries[str(key)] = _mesh_summary(piece)
    return _json_safe({"by": by, "pieces": written, "summaries": summaries})


def tool_partition(
    input_path,
    nparts,
    input_format=None,
    method="auto",
    imbalance=0.03,
    mode="eco",
    seed=0,
    record_ids=False,
    ghost_layers=0,
    weights=None,
    output_dir=None,
    name_template="{stem}_part{part}.vtu",
):
    """Partition into exactly nparts balanced pieces; writes one file per part."""
    resolved_in = _resolve(input_path, must_exist=True)
    mesh = read(resolved_in, file_format=input_format)
    pieces = partition(
        mesh,
        nparts,
        method=method,
        imbalance=imbalance,
        mode=mode,
        seed=seed,
        record_ids=record_ids,
        ghost_layers=ghost_layers,
        weights=weights,
    )
    stem = os.path.splitext(os.path.basename(resolved_in))[0]
    out_dir = output_dir if output_dir is not None else os.path.dirname(resolved_in)
    written = []
    summaries = []
    for i, piece in enumerate(pieces):
        name = name_template.format(stem=stem, part=i)
        path = _resolve(os.path.join(out_dir, name), for_write=True)
        write(path, piece)
        written.append(path)
        summaries.append(_mesh_summary(piece))
    return _json_safe(
        {
            "nparts": int(nparts),
            "method": method,
            "parts": written,
            "summaries": summaries,
        }
    )


def tool_interpolate(
    source_path,
    target_path,
    output_path,
    source_format=None,
    target_format=None,
    output_format=None,
    method="nearest",
    arrays=None,
    extrapolate=False,
    default_value=0.0,
    on_conflict="error",
):
    """Sample the source mesh's data arrays onto the target mesh's geometry."""
    source = _load(source_path, source_format)
    target = _load(target_path, target_format)
    out = interpolate(
        source,
        target,
        method=method,
        arrays=arrays,
        extrapolate=extrapolate,
        default_value=default_value,
        on_conflict=on_conflict,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_conservative_interpolate(
    source_path,
    target_path,
    output_path,
    source_format=None,
    target_format=None,
    output_format=None,
    arrays=None,
    default_value=0.0,
    on_conflict="error",
):
    """Mass-preservingly (overlap-measure weighted) sample the source mesh's
    data arrays onto the target mesh's geometry -- unlike interpolate, this
    conserves sum(value * measure) over the region the two meshes share."""
    source = _load(source_path, source_format)
    target = _load(target_path, target_format)
    out = conservative_interpolate(
        source,
        target,
        arrays=arrays,
        default_value=default_value,
        on_conflict=on_conflict,
    )
    return _result(_store(out, output_path, output_format), out)


def tool_undo_green(
    coarse_path,
    fine_path,
    output_path,
    coarse_format=None,
    fine_format=None,
    output_format=None,
):
    """Restore the fine mesh's transitional (green) cells to their original
    parent, read verbatim from the coarse mesh (a prior refine() input).
    Returns num_groups_undone/num_cells_removed alongside the mesh summary."""
    coarse = _load(coarse_path, coarse_format)
    fine = _load(fine_path, fine_format)
    out, report = undo_green(coarse, fine, return_report=True)
    return _result(_store(out, output_path, output_format), out, **report)


# --------------------------------------------------------------------------- #
# Data operations                                                             #
# --------------------------------------------------------------------------- #
def tool_data_manage(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    keep=None,
    drop=None,
    rename=None,
    ignore_missing=False,
):
    """Keep/drop/rename data arrays.

    ``keep``/``drop`` are lists of ``[location, name]`` pairs, ``rename`` a
    list of ``[location, old, new]`` triples (location: point/cell/field).
    """
    if not (keep or drop or rename):
        raise ValueError(
            "meshio++: mcp: data_manage needs at least one of keep/drop/rename"
        )
    mesh = _load(input_path, input_format)
    result = data_manage(
        mesh, keep=keep, drop=drop, rename=rename, ignore_missing=ignore_missing
    )
    out = result["mesh"]
    return _result(
        _store(out, output_path, output_format),
        out,
        dropped=result["dropped"],
        renamed=[list(pair) for pair in result["renamed"]],
    )


def tool_data_convert(
    input_path,
    output_path,
    direction,
    input_format=None,
    output_format=None,
    arrays=None,
    weighted=False,
    nan_policy="ignore",
    nan_replacement=0.0,
):
    """Average data between locations: point_to_cell or cell_to_point."""
    mesh = _load(input_path, input_format)
    if direction == "point_to_cell":
        out = point_data_to_cell_data(
            mesh, keys=arrays, nan_policy=nan_policy, nan_replacement=nan_replacement
        )
    elif direction == "cell_to_point":
        out = cell_data_to_point_data(
            mesh,
            keys=arrays,
            weighted=weighted,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement,
        )
    else:
        raise ValueError(
            f"meshio++: mcp: unknown direction '{direction}' "
            "(expected 'point_to_cell' or 'cell_to_point')"
        )
    return _result(_store(out, output_path, output_format), out)


def tool_data_calc(
    input_path,
    output_path,
    expression,
    input_format=None,
    output_format=None,
    location="point",
    output_name="",
    overwrite=False,
):
    """Evaluate an arithmetic expression over data arrays into a new array.

    Accepts the CLI's ``"NAME = EXPR"`` spelling (split on the first ``=``,
    the documented colon rule) when ``output_name`` is not given separately.
    """
    if not output_name and "=" in expression:
        output_name, expression = (s.strip() for s in expression.split("=", 1))
    mesh = _load(input_path, input_format)
    out = data_calc(
        mesh, expression, location=location, output=output_name, overwrite=overwrite
    )
    return _result(_store(out, output_path, output_format), out)


def tool_data_condition(
    input_path,
    output_path,
    input_format=None,
    output_format=None,
    operation="clamp",
    location="point",
    arrays=None,
    scope="component",
    lo=0.0,
    hi=1.0,
    nan_policy="ignore",
    nan_replacement=0.0,
    suffix="",
    preserve_dtype=True,
):
    """Condition data values: clamp, normalize or standardize."""
    mesh = _load(input_path, input_format)
    out = data_condition(
        mesh,
        location=location,
        keys=arrays,
        mode=operation,
        scope=scope,
        lo=lo,
        hi=hi,
        nan_policy=nan_policy,
        nan_replacement=nan_replacement,
        suffix=suffix,
        preserve_dtype=preserve_dtype,
    )
    return _result(_store(out, output_path, output_format), out)


# --------------------------------------------------------------------------- #
# Dataset manifests (doc/datasets.md) — pure stdlib, ungated                  #
# --------------------------------------------------------------------------- #
def _load_manifest(manifest_path, must_exist=True):
    from .._dataset import DatasetManifest

    resolved = _resolve(manifest_path, must_exist=must_exist, for_write=not must_exist)
    if os.path.isfile(resolved):
        return DatasetManifest.load(resolved), resolved
    return DatasetManifest(base_dir=os.path.dirname(resolved)), resolved


def _sandbox_entry_paths(entry):
    """Enforce the root on the paths a manifest entry resolves to — a
    hand-edited manifest can name anything, so the sandbox must hold on the
    *resolved* plan, not only on the manifest file (the pipeline-tool rule:
    paths inside the document are client input too).

    An entry's optional ``Target`` is a second source of exactly the same kind,
    so it is walked too; skipping it would leave the whole sandbox open through
    one extra key."""
    plan = entry.entries()
    for item in plan:
        _resolve(item["path"], must_exist=True)
    if entry.target:
        for item in entry.target_entries():
            _resolve(item["path"], must_exist=True)
    return plan


def _mcp_source_block(pattern, paths, base, fmt, times, time_from, sort):
    """One sandboxed source family -> a Source object stored relative to `base`.

    Shared by a manifest entry's Source and its optional Target so the two
    cannot diverge on sandboxing or on path portability.
    """
    from .._dataset import portable_relpath

    if pattern is not None:
        _resolve_pattern(pattern)  # sandbox + non-empty, before storing
        head, tail = os.path.split(str(pattern))
        rel_head = portable_relpath(_resolve(head or "."), base)
        # Not os.path.normpath/join: on Windows those re-introduce a native
        # backslash into what must stay a portable, "/"-only manifest path
        # (doc/datasets.md) -- string-join instead, since rel_head is already
        # normalized and `tail` (from os.path.split) never contains its own
        # separator.
        stored = f"{rel_head}/{tail}" if rel_head not in ("", ".") else tail
        source = {"Pattern": stored}
    else:
        resolved = [_resolve(p, must_exist=True) for p in paths]
        source = {"Paths": [portable_relpath(p, base) for p in resolved]}
        if sort:
            source["Sort"] = True
    if fmt:
        source["Format"] = fmt
    if times is not None:
        source["Times"] = [float(t) for t in times]
    if time_from:
        source["TimeFrom"] = time_from
    return source


def tool_dataset_add(
    manifest_path,
    input_pattern=None,
    input_paths=None,
    entry_id=None,
    input_format=None,
    time_from=None,
    times=None,
    sort=False,
    target_pattern=None,
    target_paths=None,
    target_format=None,
    target_time_from=None,
    target_times=None,
    target_sort=False,
    split=None,
    tags=None,
    group=None,
    notes=None,
    metadata=None,
):
    """Add a case to a dataset manifest (created if absent).

    Same input shape as the `sequence` tool: exactly one of input_pattern (a
    sandboxed glob) or input_paths. The source is validated (expanded once)
    and stored relative to the manifest's directory, which is the manifest's
    resolution anchor.
    """
    if (input_pattern is None) == (input_paths is None):
        raise ValueError(
            "meshio++: mcp: give exactly one of input_pattern or input_paths"
        )
    if target_pattern is not None and target_paths is not None:
        raise ValueError(
            "meshio++: mcp: give at most one of target_pattern or target_paths"
        )
    manifest, resolved_manifest = _load_manifest(manifest_path, must_exist=False)
    base = os.path.dirname(resolved_manifest)

    source = _mcp_source_block(
        input_pattern, input_paths, base, input_format, times, time_from, sort
    )
    target = None
    if target_pattern is not None or target_paths is not None:
        target = _mcp_source_block(
            target_pattern,
            target_paths,
            base,
            target_format,
            target_times,
            target_time_from,
            target_sort,
        )
    entry = manifest.add(
        source,
        target=target,
        id=entry_id,
        split=split,
        tags=tags or (),
        group=group,
        notes=notes,
        metadata=metadata,
    )
    manifest.save(resolved_manifest)
    return _json_safe(
        {
            "manifest_path": resolved_manifest,
            "entry_id": entry.id,
            "num_steps": len(entry.entries()),
            "num_target_steps": len(entry.target_entries()) if entry.target else None,
            "num_entries": len(manifest),
        }
    )


def tool_dataset_list(manifest_path, split=None, tags=None, group=None, resolve=False):
    """List a manifest's entries, optionally filtered by split/tags/group.

    resolve=True also expands each entry's plan (paths/steps/times — no mesh
    is read) with every resolved path sandbox-checked.
    """
    manifest, resolved_manifest = _load_manifest(manifest_path)
    entries = manifest.entries(split=split, tags=tags or None, group=group)
    payload = []
    for entry in entries:
        item = entry.to_dict()
        if resolve:
            item["resolved"] = _sandbox_entry_paths(entry)
        payload.append(item)
    return _json_safe(
        {
            "manifest_path": resolved_manifest,
            "name": manifest.name,
            "num_entries": len(manifest),
            "num_matching": len(entries),
            "splits": {str(k): v for k, v in manifest.splits().items()},
            "entries": payload,
        }
    )


def tool_dataset_update(
    manifest_path,
    entry_ids=None,
    all_entries=False,
    split=None,
    assign_splits=None,
    seed=0,
    by_group=False,
    add_tags=None,
    remove_tags=None,
    group=None,
    notes=None,
    metadata=None,
    drop_metadata=None,
):
    """Curate a manifest: set splits (or assign by fractions), tag, annotate.

    entry_ids (or all_entries=true) selects; split/add_tags/remove_tags apply
    to the selection; assign_splits={"train": 0.8, ...} reassigns every entry
    deterministically (seed; by_group keeps groups together); group/notes/
    metadata/drop_metadata need exactly one selected entry.
    """
    manifest, resolved_manifest = _load_manifest(manifest_path)
    ids = "all" if all_entries else list(entry_ids or ())
    if assign_splits is not None:
        manifest.assign_splits(dict(assign_splits), seed=int(seed), by_group=by_group)
    if split is not None:
        if not ids:
            raise ValueError("meshio++: mcp: split needs entry_ids or all_entries")
        manifest.set_split(ids, split)
    if add_tags or remove_tags:
        if not ids:
            raise ValueError("meshio++: mcp: tagging needs entry_ids or all_entries")
        manifest.tag(ids, add=add_tags or (), remove=remove_tags or ())
    if group is not None or notes is not None or metadata or drop_metadata:
        if ids == "all" or len(ids) != 1:
            raise ValueError(
                "meshio++: mcp: group/notes/metadata apply to exactly one entry_id"
            )
        manifest.annotate(
            ids[0],
            notes=notes,
            group=group,
            metadata=metadata,
            drop_metadata=drop_metadata or (),
        )
    manifest.save(resolved_manifest)
    return _json_safe(
        {
            "manifest_path": resolved_manifest,
            "num_entries": len(manifest),
            "splits": {str(k): v for k, v in manifest.splits().items()},
        }
    )


def tool_dataset_find(root_dir=".", max_depth=2):
    """Find dataset manifests under a directory (sandboxed walk).

    Every ``*.json`` at most ``max_depth`` levels below ``root_dir`` that
    ``DatasetManifest.load`` accepts is listed with its content hash -- the
    identity a browser-side card (which never sees an absolute path) binds
    to. Other JSON files are skipped silently; dot-directories are not
    entered.
    """
    import hashlib

    from .._dataset import DatasetManifest, portable_relpath

    base = _resolve(root_dir)
    if not os.path.isdir(base):
        raise ValueError(f"meshio++: mcp: directory not found: '{base}'")
    depth = int(max_depth)
    if depth < 0:
        raise ValueError("meshio++: mcp: max_depth must be >= 0")
    found = []
    for current, dirs, files in os.walk(base):
        rel = os.path.relpath(current, base)
        level = 0 if rel == "." else rel.count(os.sep) + 1
        dirs[:] = (
            sorted(d for d in dirs if not d.startswith(".")) if level < depth else []
        )
        for name in sorted(files):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(current, name)
            try:
                manifest = DatasetManifest.load(path)
            except Exception:  # noqa: BLE001 - not a manifest, or a broken one
                continue
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            found.append(
                {
                    "path": path,
                    "relpath": portable_relpath(path, base),
                    "sha256": digest,
                    "name": manifest.name,
                    "num_entries": len(manifest),
                    "splits": {str(k): v for k, v in manifest.splits().items()},
                    "mtime": int(os.path.getmtime(path) * 1000),
                }
            )
    return _json_safe({"root": base, "manifests": found})


def tool_dataset_health(
    manifest_path, split=None, entry_ids=None, quality=True, all_steps=False
):
    """Scan a manifest's entries and summarize their health (server side).

    Per entry: step count, NaN/Inf counts over the data arrays (quality:*
    arrays excluded -- their NaN means "N/A for this cell type"), inverted /
    degenerate cell counts and the worst scaled Jacobian from
    compute_quality, and the arrays present. Per manifest: split balance,
    totals, fields missing across entries, and the bad entries. Meshes are
    read one at a time; every entry's resolved paths are sandbox-checked
    (a hand-edited manifest is client input too).
    """
    import hashlib

    from ._health import manifest_health

    manifest, resolved_manifest = _load_manifest(manifest_path)
    selected = manifest.entries(split=split)
    if entry_ids:
        wanted = list(entry_ids)
        known = {e.id for e in selected}
        unknown = [i for i in wanted if i not in known]
        if unknown:
            raise ValueError(f"meshio++: mcp: unknown entry id(s): {unknown}")
        selected = [e for e in selected if e.id in set(wanted)]
    report = manifest_health(
        manifest,
        selected,
        quality=bool(quality),
        all_steps=bool(all_steps),
        before_entry=_sandbox_entry_paths,
    )
    with open(resolved_manifest, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    report["manifest_path"] = resolved_manifest
    report["sha256"] = digest
    return _json_safe(report)


# --------------------------------------------------------------------------- #
# Training jobs (doc/dashboard.md, doc/physicsnemo.md)                        #
# --------------------------------------------------------------------------- #
_PHYSICSNEMO_DOC = "doc/physicsnemo.md"


def _require_training_frameworks(op, model_name="meshgraphnet"):
    """The trainer subprocess's frameworks -- checked here, BEFORE spawning, so
    a missing one is a named payload rather than a dead subprocess. Skipped when
    MESHIOPLUSPLUS_TRAIN_COMMAND names a trainer from another interpreter (the
    override's whole point).

    Only what the chosen family imports: a convolutional model never touches
    PyTorch Geometric, so demanding it would refuse a runnable job.
    """
    from .._gpu import _require_framework
    from ._jobs import TRAIN_COMMAND_ENV

    if os.environ.get(TRAIN_COMMAND_ENV):
        return
    if model_name != "srresnet":
        _require_framework(
            op, "torch_geometric", "pip install torch_geometric", doc=_PHYSICSNEMO_DOC
        )
    _require_framework(
        op, "physicsnemo", "pip install nvidia-physicsnemo", doc=_PHYSICSNEMO_DOC
    )


def tool_train_defaults(manifest_path, fields=None, target_fields=None):
    """What a training launch form needs: the data arrays the manifest's
    first entry carries, the splits, and a complete default spec."""
    from ..physicsnemo import has_physicsnemo, has_torch_geometric
    from ..physicsnemo._train import TrainSpec, spec_to_dict

    manifest, resolved_manifest = _load_manifest(manifest_path)
    entries = list(manifest)
    if not entries:
        raise ValueError(
            f"meshio++: mcp: the manifest '{resolved_manifest}' has no entries"
        )
    plan = _sandbox_entry_paths(entries[0])
    meta = read_metadata(plan[0]["path"])
    spec = TrainSpec(
        manifest=resolved_manifest,
        fields=tuple(fields or ()),
        target_fields=tuple(target_fields or ()),
        run_dir=get_runs_dir(),
    )
    return _json_safe(
        {
            "manifest_path": resolved_manifest,
            "num_entries": len(manifest),
            "splits": {
                ("" if k is None else str(k)): v for k, v in manifest.splits().items()
            },
            "available_fields": {
                "point": list(meta.get("point_data_names", [])),
                "cell": list(meta.get("cell_data_names", [])),
            },
            "runs_dir": get_runs_dir(),
            "frameworks": {
                "torch_geometric": has_torch_geometric(),
                "physicsnemo": has_physicsnemo(),
            },
            "spec": spec_to_dict(spec),
        }
    )


def tool_train_start(
    manifest_path,
    fields,
    target_fields,
    train_split="train",
    valid_split="valid",
    epochs=100,
    batch_size=8,
    learning_rate=1e-3,
    seed=0,
    model_name="meshgraphnet",
    processor_size=8,
    hidden_dim=64,
    aggregation="sum",
    scaling_factor=2,
    conv_layer_size=32,
    resid_blocks=8,
    resolution=None,
    cell_size=None,
    bounds=None,
    padding=0.0,
    padding_relative=0.0,
    extrapolate=False,
    fill_value=0.0,
    regions=False,
    kind="node",
    undirected=True,
    edge_features=True,
    float32=True,
    target_offset=0,
    target_delta=False,
    checkpoint_every=10,
    device="auto",
    notes=None,
    tags=None,
):
    """Start a training job against a manifest split.

    Builds the PascalCase spec, lays out `<runs_dir>/<job_id>/` and spawns
    `python -m meshioplusplus.physicsnemo.train --spec` as a subprocess;
    returns the job's initial status. Poll it with train_status /
    train_metrics / train_log.

    model_name picks the family, and the two read different hyperparameters:
    'meshgraphnet' uses processor_size/hidden_dim/aggregation and the graph
    options, 'srresnet' uses scaling_factor/conv_layer_size/resid_blocks and
    the grid options (give exactly one of resolution and cell_size). Needs
    nvidia-physicsnemo, plus torch_geometric for meshgraphnet only (no pip
    extra, deliberately); a missing framework is a named error before anything
    is spawned.
    """
    from ..physicsnemo._train import default_spec, spec_to_dict

    manifest, resolved_manifest = _load_manifest(manifest_path)
    splits = {("" if k is None else str(k)): v for k, v in manifest.splits().items()}
    if train_split not in splits:
        raise ValueError(
            f"meshio++: mcp: the manifest has no split '{train_split}' "
            f"(available: {sorted(splits)})"
        )
    for entry in manifest.entries(split=train_split):
        _sandbox_entry_paths(entry)
    model_name = str(model_name).lower()
    _require_training_frameworks("train_start", model_name)
    common = dict(
        run_dir=get_runs_dir(),
        train_split=str(train_split),
        valid_split=str(valid_split),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        seed=int(seed),
        model_name=model_name,
        float32=bool(float32),
        checkpoint_every=int(checkpoint_every),
        device=str(device),
        notes=notes,
        tags=tuple(tags or ()),
    )
    if model_name == "srresnet":
        family = dict(
            scaling_factor=int(scaling_factor),
            conv_layer_size=int(conv_layer_size),
            resid_blocks=int(resid_blocks),
            resolution=tuple(resolution) if resolution else None,
            cell_size=None if cell_size is None else float(cell_size),
            bounds=tuple(bounds) if bounds else None,
            padding=float(padding),
            padding_relative=float(padding_relative),
            extrapolate=bool(extrapolate),
            fill_value=float(fill_value),
        )
    else:
        family = dict(
            processor_size=int(processor_size),
            hidden_dim=int(hidden_dim),
            aggregation=str(aggregation),
            regions=bool(regions),
            kind=str(kind),
            undirected=bool(undirected),
            edge_features=bool(edge_features),
            target_offset=int(target_offset),
            target_delta=bool(target_delta),
        )
    spec = default_spec(resolved_manifest, fields, target_fields, **common, **family)
    return _json_safe(_jobs_manager().start(spec_to_dict(spec)))


def tool_train_status(job_id):
    """A job's status, progress (epoch/epochs, best, ETA) and last metrics row."""
    return _json_safe(_jobs_manager().status(str(job_id)))


def tool_train_list(status=None, manifest_path=None):
    """Every job under the runs directory (newest first), summarized with its
    hyperparameters and final/best losses; filter by status or manifest."""
    manifest = _resolve(manifest_path, must_exist=True) if manifest_path else None
    manager = _jobs_manager()
    return _json_safe(
        {
            "runs_dir": manager.runs_dir,
            "jobs": manager.list_jobs(status=status, manifest=manifest),
        }
    )


def tool_train_stop(job_id, grace_seconds=10.0):
    """Stop a job: SIGTERM (the trainer finishes its epoch and writes
    final.mdlus), SIGKILL after grace_seconds."""
    return _json_safe(_jobs_manager().stop(str(job_id), grace=float(grace_seconds)))


def tool_train_log(job_id, offset=0, max_bytes=65536):
    """A window of the job's log from a byte offset (tail with next_offset)."""
    return _json_safe(
        _jobs_manager().log(str(job_id), offset=int(offset), max_bytes=int(max_bytes))
    )


def tool_train_metrics(job_id, since_epoch=0):
    """The per-epoch metrics rows from since_epoch on."""
    return _json_safe(
        _jobs_manager().metrics(str(job_id), since_epoch=int(since_epoch))
    )


def tool_train_checkpoints(job_id):
    """The job's .mdlus checkpoints with epoch / validation loss / size."""
    return _json_safe(_jobs_manager().checkpoints(str(job_id)))


def tool_train_mark_best(job_id, checkpoint):
    """Copy one of the job's checkpoints (and its card) to best.mdlus."""
    return _json_safe(_jobs_manager().mark_best(str(job_id), str(checkpoint)))


def tool_train_predict(
    manifest_path,
    job_id=None,
    checkpoint=None,
    entry_ids=None,
    split="test",
    step=0,
    output_dir=None,
):
    """Predict over a manifest split with a job's checkpoint (its marked
    best, or a named one) or with an explicit .mdlus path; writes
    <column>_pred / <column>_error back into output_dir/<entry_id>.vtu
    (default: the job's predictions/ directory). Needs the frameworks.
    """
    from ..physicsnemo._train import PREDICTIONS_DIR

    manifest, resolved_manifest = _load_manifest(manifest_path)
    for entry in manifest.entries(split=split) if split else manifest:
        _sandbox_entry_paths(entry)
    manager = _jobs_manager()
    if job_id:
        resolved_checkpoint = manager.resolve_checkpoint(str(job_id), checkpoint)
        out = output_dir or os.path.join(manager.job_dir(str(job_id)), PREDICTIONS_DIR)
    elif checkpoint:
        resolved_checkpoint = _resolve(checkpoint, must_exist=True)
        if not output_dir:
            raise ValueError(
                "meshio++: mcp: output_dir is required with an explicit checkpoint"
            )
        out = output_dir
    else:
        raise ValueError("meshio++: mcp: give job_id or checkpoint")
    out = _resolve(out, for_write=True)
    from .._gpu import _require_framework

    _require_framework(
        "train_predict",
        "torch_geometric",
        "pip install torch_geometric",
        doc=_PHYSICSNEMO_DOC,
    )
    _require_framework(
        "train_predict",
        "physicsnemo",
        "pip install nvidia-physicsnemo",
        doc=_PHYSICSNEMO_DOC,
    )
    from ..physicsnemo import predict

    rows = predict(
        resolved_checkpoint,
        manifest,
        entry_ids=list(entry_ids) if entry_ids else None,
        split=split,
        step=int(step),
        output_dir=out,
    )
    finite = [r["rmse"] for r in rows if r.get("rmse") is not None]
    return _json_safe(
        {
            "checkpoint": resolved_checkpoint,
            "output_dir": out,
            "predictions": rows,
            "mean_rmse": (sum(finite) / len(finite)) if finite else None,
        }
    )


def tool_guard_fit(
    manifest_path,
    output_path,
    split="train",
    margin=1.5,
    quality=True,
):
    """Fit a geometry guardrail over a manifest split and write it as JSON."""
    manifest, resolved_manifest = _load_manifest(manifest_path)
    for entry in manifest.entries(split=split) if split else manifest:
        _sandbox_entry_paths(entry)
    out = _resolve(output_path, for_write=True)
    guard = GeometryGuard.fit(
        manifest, split=split, margin=float(margin), quality=bool(quality)
    )
    guard.save(out)
    return _json_safe(
        {
            "manifest_path": resolved_manifest,
            "output_path": out,
            "num_samples": guard.schema.get("num_samples"),
            "threshold": guard.threshold,
            "descriptors": list(guard.names),
        }
    )


def tool_guard_check(input_path, guard_path=None, input_format=None, top=3):
    """Describe one mesh's shape, and score it against a guardrail if given.

    Without guard_path this reports the raw descriptors, which is what a
    caller comparing two parts by hand wants.
    """
    mesh = _load(input_path, input_format)
    descriptors = geometry_descriptors(mesh)
    report = {
        "input_path": _resolve(input_path, must_exist=True),
        "descriptors": descriptors,
    }
    if guard_path is not None:
        guard = GeometryGuard.load(_resolve(guard_path, must_exist=True))
        report.update(guard.check(descriptors, top=int(top)))
        report["guard_path"] = _resolve(guard_path, must_exist=True)
    return _json_safe(report)


def tool_predict_file(
    checkpoint,
    input_path,
    output_path,
    time_step=None,
    target_path=None,
    input_format=None,
    output_format=None,
    device="auto",
):
    """Predict with a trained .mdlus checkpoint on ONE mesh file -- no
    manifest, no split, no entry. A file carrying no truth predicts anyway,
    with rmse/max_error reported as null. Needs the frameworks.
    """
    resolved_checkpoint = _resolve(checkpoint, must_exist=True)
    resolved_input = _resolve(input_path, must_exist=True)
    resolved_target = _resolve(target_path, must_exist=True) if target_path else None
    resolved_output = _resolve(output_path, for_write=True)
    from .._gpu import _require_framework

    _require_framework(
        "predict_file",
        "physicsnemo",
        "pip install nvidia-physicsnemo",
        doc=_PHYSICSNEMO_DOC,
    )
    from ..physicsnemo import predict_file

    return _json_safe(
        predict_file(
            resolved_checkpoint,
            resolved_input,
            resolved_output,
            time_step=None if time_step is None else int(time_step),
            target_path=resolved_target,
            input_format=input_format,
            output_format=output_format,
            device=device,
        )
    )


# --------------------------------------------------------------------------- #
# Gated tools (optional extras; the wrapped functions raise the named error)  #
# --------------------------------------------------------------------------- #
def tool_data_export(input_path, output_path, input_format=None, location="point"):
    """Export data arrays to a Parquet table (needs the [arrow] extra)."""
    mesh = _load(input_path, input_format)
    resolved = _resolve(output_path, for_write=True)
    _write_parquet_fn(mesh, resolved, location=location)
    return _json_safe(
        {"output_path": resolved, "location": location, **_mesh_summary(mesh)}
    )


def tool_export_dataset(
    input_pattern=None,
    input_paths=None,
    output_path=None,
    input_format=None,
    location="point",
    dataset_format="parquet",
    mesh_id="stem",
):
    """Export a set of meshes as one dataset keyed by mesh_id.

    Hive-partitioned Parquet (needs the [arrow] extra), or chunked zarr/hdf5
    groups (the [zarr] extra / h5py). Same input shape as the `sequence`
    tool: exactly one of input_pattern (a sandboxed glob) or input_paths.
    """
    from .._ml import write_dataset

    if (input_pattern is None) == (input_paths is None):
        raise ValueError(
            "meshio++: mcp: give exactly one of input_pattern or input_paths"
        )
    if not output_path:
        raise ValueError("meshio++: mcp: output_path is required")
    if input_pattern is not None:
        resolved_in = _resolve_pattern(input_pattern)
    else:
        resolved_in = [_resolve(p, must_exist=True) for p in input_paths]
    resolved_out = _resolve(output_path, for_write=True)

    manifest = write_dataset(
        resolved_in,
        resolved_out,
        location=location,
        format=dataset_format,
        mesh_id=mesh_id,
        file_format=input_format,
    )
    return _json_safe({"output_path": resolved_out, **manifest})


def tool_export_cae(
    output_dir,
    input_pattern=None,
    input_paths=None,
    input_format=None,
    surface_fields=None,
    volume_fields=None,
    global_params=None,
    global_params_reference=None,
    global_params_order=None,
    name_template="case_{index}.npz",
):
    """Export a set of meshes as one .npz per case in the CAE sample layout.

    The per-sample layout PhysicsNeMo's DoMINO and Transolver datapipes read:
    the triangulated skin, its normals and areas, the volume's nodes, the
    named field blocks and the case's global parameters. Same input shape as
    the `sequence` tool: exactly one of input_pattern (a sandboxed glob) or
    input_paths.
    """
    from ..cae import export_cases

    if (input_pattern is None) == (input_paths is None):
        raise ValueError(
            "meshio++: mcp: give exactly one of input_pattern or input_paths"
        )
    if not output_dir:
        raise ValueError("meshio++: mcp: output_dir is required")
    if input_pattern is not None:
        resolved_in = _resolve_pattern(input_pattern)
    else:
        resolved_in = [_resolve(p, must_exist=True) for p in input_paths]
    resolved_out = _resolve(output_dir, for_write=True)

    written = export_cases(
        resolved_in,
        resolved_out,
        name_template=name_template,
        file_format=input_format,
        surface_fields=list(surface_fields) if surface_fields else None,
        volume_fields=list(volume_fields) if volume_fields else None,
        global_params=dict(global_params) if global_params else None,
        global_params_reference=(
            dict(global_params_reference) if global_params_reference else None
        ),
        global_params_order=(
            list(global_params_order) if global_params_order else None
        ),
    )
    keys = []
    if written:
        import numpy as np

        with np.load(written[0]) as data:
            keys = sorted(str(name) for name in data.files)
    return _json_safe(
        {
            "output_dir": resolved_out,
            "files": written,
            "num_cases": len(written),
            "keys": keys,
        }
    )


def tool_screenshot(
    input_path,
    output_path,
    input_format=None,
    color_by=None,
    width=1280,
    height=960,
    transparent=False,
):
    """Render an off-screen screenshot PNG (needs the [viewer] extra)."""
    mesh = _load(input_path, input_format)
    resolved = _resolve(output_path, for_write=True)
    _screenshot_fn(
        mesh,
        resolved,
        color_by=color_by,
        size=(int(width), int(height)),
        transparent=transparent,
    )
    return _json_safe({"output_path": resolved, "size": [int(width), int(height)]})


# --------------------------------------------------------------------------- #
# Registry — the single source of truth                                       #
# --------------------------------------------------------------------------- #
# name -> {"fn", "wraps" (public meshioplusplus API names this tool covers,
# consumed by the parity-guard test), "gated" (extra name, or None)}.
TOOL_REGISTRY = OrderedDict(
    [
        (
            "formats",
            {
                "fn": tool_formats,
                "wraps": ("extension_to_filetypes", "formats"),
                "gated": None,
            },
        ),
        ("sniff", {"fn": tool_sniff, "wraps": ("sniff_format",), "gated": None}),
        ("info", {"fn": tool_info, "wraps": ("read_metadata",), "gated": None}),
        ("stats", {"fn": tool_stats, "wraps": ("compute_stats",), "gated": None}),
        (
            "quality",
            {
                "fn": tool_quality,
                "wraps": ("compute_quality", "attach_quality"),
                "gated": None,
            },
        ),
        ("data_info", {"fn": tool_data_info, "wraps": ("data_info",), "gated": None}),
        (
            "data_integrate",
            {"fn": tool_data_integrate, "wraps": ("data_integrate",), "gated": None},
        ),
        ("regions", {"fn": tool_regions, "wraps": (), "gated": None}),
        (
            "bandwidth",
            {"fn": tool_bandwidth, "wraps": ("compute_bandwidth",), "gated": None},
        ),
        ("data_preview", {"fn": tool_data_preview, "wraps": (), "gated": None}),
        ("diff", {"fn": tool_diff, "wraps": ("diff", "meshes_equal"), "gated": None}),
        ("convert", {"fn": tool_convert, "wraps": ("read", "write"), "gated": None}),
        (
            "pipeline",
            {"fn": tool_pipeline, "wraps": ("run_pipeline",), "gated": None},
        ),
        (
            "sequence",
            {
                "fn": tool_sequence,
                "wraps": (
                    "write_sequence",
                    "sequence_entries",
                    "run_sequence_pipeline",
                ),
                "gated": None,
            },
        ),
        (
            "extract_surface",
            {"fn": tool_extract_surface, "wraps": ("extract_surface",), "gated": None},
        ),
        (
            "extract_skin",
            {"fn": tool_extract_skin, "wraps": ("extract_skin",), "gated": None},
        ),
        ("reorder", {"fn": tool_reorder, "wraps": ("reorder",), "gated": None}),
        ("clean", {"fn": tool_clean, "wraps": ("clean",), "gated": None}),
        ("crop", {"fn": tool_crop, "wraps": ("crop",), "gated": None}),
        ("slice", {"fn": tool_slice, "wraps": ("slice",), "gated": None}),
        (
            "isosurface",
            {"fn": tool_isosurface, "wraps": ("isosurface",), "gated": None},
        ),
        ("gradient", {"fn": tool_gradient, "wraps": ("gradient",), "gated": None}),
        ("hessian", {"fn": tool_hessian, "wraps": ("hessian",), "gated": None}),
        (
            "curvature",
            {"fn": tool_curvature, "wraps": ("compute_curvature",), "gated": None},
        ),
        ("repair", {"fn": tool_repair, "wraps": ("repair",), "gated": None}),
        (
            "shrinkwrap",
            {"fn": tool_shrinkwrap, "wraps": ("shrinkwrap",), "gated": None},
        ),
        (
            "sobolev_deform",
            {"fn": tool_sobolev_deform, "wraps": ("sobolev_deform",), "gated": None},
        ),
        (
            "estimate_error",
            {"fn": tool_estimate_error, "wraps": ("estimate_error",), "gated": None},
        ),
        ("grid", {"fn": tool_grid, "wraps": ("grid",), "gated": None}),
        ("voxelize", {"fn": tool_voxelize, "wraps": ("voxelize",), "gated": None}),
        (
            "grid_sample",
            {"fn": tool_grid_sample, "wraps": ("sample_grid",), "gated": None},
        ),
        (
            "grid_scatter",
            {"fn": tool_grid_scatter, "wraps": ("scatter_grid",), "gated": None},
        ),
        (
            "grid_resample",
            {"fn": tool_grid_resample, "wraps": ("resample_grid",), "gated": None},
        ),
        (
            "grid_power_spectrum",
            {
                "fn": tool_grid_power_spectrum,
                "wraps": ("power_spectrum",),
                "gated": None,
            },
        ),
        (
            "subsample",
            {"fn": tool_subsample, "wraps": ("subsample_points",), "gated": None},
        ),
        (
            "proximity_graph",
            {
                "fn": tool_proximity_graph,
                "wraps": ("proximity_graph",),
                "gated": None,
            },
        ),
        (
            "compute_sdf",
            {"fn": tool_compute_sdf, "wraps": ("compute_sdf",), "gated": None},
        ),
        (
            "distance_to_surface",
            {
                "fn": tool_distance_to_surface,
                "wraps": ("distance_to_surface",),
                "gated": None,
            },
        ),
        (
            "sample_distance",
            {"fn": tool_sample_distance, "wraps": ("sample_distance",), "gated": None},
        ),
        (
            "surface_watertight_check",
            {
                "fn": tool_surface_watertight_check,
                "wraps": ("surface_watertight_check",),
                "gated": None,
            },
        ),
        ("transform", {"fn": tool_transform, "wraps": ("transform",), "gated": None}),
        (
            "convert_cells",
            {"fn": tool_convert_cells, "wraps": ("convert_cells",), "gated": None},
        ),
        (
            "subdivide",
            {"fn": tool_subdivide, "wraps": ("subdivide",), "gated": None},
        ),
        (
            "agglomerate",
            {"fn": tool_agglomerate, "wraps": ("agglomerate",), "gated": None},
        ),
        ("refine", {"fn": tool_refine, "wraps": ("refine",), "gated": None}),
        (
            "undo_green",
            {"fn": tool_undo_green, "wraps": ("undo_green",), "gated": None},
        ),
        ("decimate", {"fn": tool_decimate, "wraps": ("decimate",), "gated": None}),
        (
            "decimate_volume",
            {"fn": tool_decimate_volume, "wraps": ("decimate_volume",), "gated": None},
        ),
        ("remesh", {"fn": tool_remesh, "wraps": ("remesh",), "gated": None}),
        (
            "remesh_volume",
            {"fn": tool_remesh_volume, "wraps": ("remesh_volume",), "gated": None},
        ),
        (
            "optimize_volume",
            {"fn": tool_optimize_volume, "wraps": ("optimize_volume",), "gated": None},
        ),
        ("smooth", {"fn": tool_smooth, "wraps": ("smooth",), "gated": None}),
        ("merge", {"fn": tool_merge, "wraps": ("merge",), "gated": None}),
        ("split", {"fn": tool_split, "wraps": ("split",), "gated": None}),
        ("partition", {"fn": tool_partition, "wraps": ("partition",), "gated": None}),
        (
            "interpolate",
            {"fn": tool_interpolate, "wraps": ("interpolate",), "gated": None},
        ),
        (
            "conservative_interpolate",
            {
                "fn": tool_conservative_interpolate,
                "wraps": ("conservative_interpolate",),
                "gated": None,
            },
        ),
        (
            "data_manage",
            {
                "fn": tool_data_manage,
                "wraps": ("data_manage", "data_drop", "data_keep", "data_rename"),
                "gated": None,
            },
        ),
        (
            "data_convert",
            {
                "fn": tool_data_convert,
                "wraps": ("point_data_to_cell_data", "cell_data_to_point_data"),
                "gated": None,
            },
        ),
        ("data_calc", {"fn": tool_data_calc, "wraps": ("data_calc",), "gated": None}),
        (
            "data_condition",
            {"fn": tool_data_condition, "wraps": ("data_condition",), "gated": None},
        ),
        (
            "dataset_add",
            {"fn": tool_dataset_add, "wraps": ("DatasetManifest",), "gated": None},
        ),
        (
            "dataset_list",
            {"fn": tool_dataset_list, "wraps": (), "gated": None},
        ),
        (
            "dataset_update",
            {"fn": tool_dataset_update, "wraps": (), "gated": None},
        ),
        (
            "dataset_find",
            {"fn": tool_dataset_find, "wraps": (), "gated": None},
        ),
        (
            "dataset_health",
            {"fn": tool_dataset_health, "wraps": (), "gated": None},
        ),
        (
            "train_defaults",
            {"fn": tool_train_defaults, "wraps": (), "gated": None},
        ),
        (
            "train_start",
            {"fn": tool_train_start, "wraps": (), "gated": "physicsnemo"},
        ),
        ("train_status", {"fn": tool_train_status, "wraps": (), "gated": None}),
        ("train_list", {"fn": tool_train_list, "wraps": (), "gated": None}),
        ("train_stop", {"fn": tool_train_stop, "wraps": (), "gated": None}),
        ("train_log", {"fn": tool_train_log, "wraps": (), "gated": None}),
        ("train_metrics", {"fn": tool_train_metrics, "wraps": (), "gated": None}),
        (
            "train_checkpoints",
            {"fn": tool_train_checkpoints, "wraps": (), "gated": None},
        ),
        (
            "train_mark_best",
            {"fn": tool_train_mark_best, "wraps": (), "gated": None},
        ),
        (
            "guard_fit",
            {"fn": tool_guard_fit, "wraps": ("GeometryGuard",), "gated": None},
        ),
        (
            "guard_check",
            {
                "fn": tool_guard_check,
                "wraps": ("geometry_descriptors",),
                "gated": None,
            },
        ),
        (
            "predict_file",
            {"fn": tool_predict_file, "wraps": (), "gated": "physicsnemo"},
        ),
        (
            "train_predict",
            {"fn": tool_train_predict, "wraps": (), "gated": "physicsnemo"},
        ),
        (
            "data_export",
            {"fn": tool_data_export, "wraps": ("write_parquet",), "gated": "arrow"},
        ),
        (
            "export_dataset",
            {"fn": tool_export_dataset, "wraps": ("write_dataset",), "gated": "arrow"},
        ),
        (
            "export_cae",
            {"fn": tool_export_cae, "wraps": ("cae",), "gated": None},
        ),
        (
            "screenshot",
            {"fn": tool_screenshot, "wraps": ("screenshot",), "gated": "viewer"},
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Dispatch                                                                    #
# --------------------------------------------------------------------------- #
def guard(fn, /, **kwargs):
    """Run a tool; a failure is a ``{"error", "error_type"}`` payload, never a
    traceback. The one guard, shared by the FastMCP wrappers and the HTTP
    front-end's dispatch."""
    try:
        return fn(**kwargs)
    except Exception as e:  # noqa: BLE001 - the client needs a payload
        return {"error": str(e), "error_type": type(e).__name__}


def call_tool(name, kwargs=None):
    """Dispatch a tool by registry name with keyword arguments.

    The single entry point the HTTP front-end (``POST /api/tools/<name>``)
    goes through, so it inherits the sandbox and the strict-JSON sanitizer
    every tool already applies. An unknown name is a ``KeyError`` (the
    caller's 404); a tool failure -- including bad keyword arguments -- is a
    guarded payload.
    """
    try:
        spec = TOOL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"meshio++: mcp: unknown tool '{name}'") from None
    return guard(spec["fn"], **(kwargs or {}))
