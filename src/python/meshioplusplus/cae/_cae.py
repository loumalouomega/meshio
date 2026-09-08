"""The CAE ``.npz`` sample layout PhysicsNeMo's DoMINO/Transolver pipes read.

One ``.npz`` per case, carrying the *superset* of the keys those two datapipes
consume (each selects its own subset through ``keys_to_read``; extra keys are
ignored). Transcribed from the Kratos ``PhysicsNeMoApplication``'s own
``CaeDatasetExportProcess`` -- the layout, not its Kratos-typed code -- so a
meshio++ case and a Kratos case are interchangeable in the same dataset
directory.

**This is a tabular sample export, not a mesh format in the usual sense.**
It reads back (see :func:`read`), but a round trip is lossy by construction:
the surface is a triangulation of the input's skin, and the volume half is
node coordinates plus nodal fields, with no volume connectivity at all.

Three rules are the reader's, not ours, and each produces silently wrong
results rather than an error when broken:

* **Every array must have ``ndim >= 1``.** ``NpzFileReader`` does
  ``in_data[key][:]``, which a 0-d array cannot serve -- hence ``TIME`` and
  ``STEP`` as shape-``(1,)`` arrays and every scalar global parameter as one
  too.
* **No key but the two real ones may contain the substring ``volume``.** That
  reader takes the volume row count from ``next(key for key in in_data.keys()
  if "volume" in key)`` -- the FIRST such key in file order, over every key in
  the file rather than the requested ones. A sidecar named
  ``...volume_field_names`` would therefore hand it a length taken from a
  names array. Ours are called ``node_field_*`` for exactly that reason, and a
  global parameter whose name contains ``volume`` is refused.
* **``volume_mesh_centers`` and ``volume_fields`` must share their row
  count**, since the slice computed from one is applied to the other.

Everything here is pure numpy; nothing imports torch or physicsnemo.
"""

from __future__ import annotations

import os

import numpy as np

from .._common import warn
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

#: Prefix for the keys meshio++ adds beyond the upstream layout. The datapipes
#: ignore unknown keys, so these cost the consumer nothing and are what lets
#: :func:`read` split the concatenated field blocks back into named arrays.
SIDECAR_PREFIX = "meshioplusplus:"

#: Where the provenance block rides: a **bytes** array written as the FIRST
#: member of the (uncompressed, stored) zip, so it lands in the file's first
#: bytes and the ordinary head scanner finds it with no npz-specific code. A
#: unicode array would be UCS-4 and interleave NULs between characters, which
#: that scanner cannot read.
PROVENANCE_KEY = SIDECAR_PREFIX + "provenance"

#: The geometry keys, always written.
SURFACE_KEYS = (
    "stl_coordinates",
    "stl_faces",
    "stl_centers",
    "stl_areas",
    "surface_mesh_centers",
    "surface_normals",
    "surface_areas",
)

#: The volume keys, written only for a volume input.
VOLUME_KEYS = ("volume_mesh_centers", "volume_fields")


def _as_f32_points(points):
    """Points as contiguous ``(n, 3)`` float32, 2-D input padded with zeros."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise WriteError(f"meshio++: cae: points must be 2-D, got {points.shape}")
    if points.shape[1] == 3:
        return np.ascontiguousarray(points, dtype=np.float32)
    padded = np.zeros((len(points), 3), dtype=np.float32)
    padded[:, : points.shape[1]] = points[:, :3]
    return padded


def _numeric(array):
    return np.asarray(array).dtype.kind in "fiub"


def _columns(array, rows):
    """One data array flattened to ``(rows, width)`` float32."""
    values = np.asarray(array, dtype=np.float64).reshape(rows, -1)
    return np.ascontiguousarray(values, dtype=np.float32)


def _triangulated_surface(mesh):
    """The mesh's skin as triangles, with the parent-cell map retained."""
    from .._convert_cells import convert_cells
    from .._surface import extract_surface

    surface = extract_surface(mesh, record_parent_ids=True)
    return convert_cells(surface, mode="simplexify", record_parent_ids=True)


def _triangulate(mesh):
    """A surface mesh reduced to triangles (higher-order linearized first)."""
    from .._convert_cells import convert_cells

    out = mesh
    if any(block.type not in ("triangle",) for block in out.cells):
        out = convert_cells(out, mode="linearize")
        out = convert_cells(out, mode="simplexify")
    return out


def _triangle_arrays(surface):
    """``stl_*`` plus the surface aliases, from a triangle-only mesh."""
    blocks = [b for b in surface.cells if b.type == "triangle"]
    if not blocks:
        raise WriteError(
            "meshio++: cae: the surface has no triangles after tessellation"
        )
    faces = np.concatenate([np.asarray(b.data, dtype=np.int64) for b in blocks])
    points = _as_f32_points(surface.points)
    corners = points[faces].astype(np.float64)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = cross / (2.0 * areas)[:, None]
    degenerate = ~np.isfinite(normals).all(axis=1)
    if degenerate.any():
        warn(
            f"cae: {int(degenerate.sum())} degenerate (zero-area) triangles have "
            "no normal; writing a zero vector for those rows"
        )
        normals[degenerate] = 0.0
    centers = np.ascontiguousarray(corners.mean(axis=1), dtype=np.float32)
    arrays = {
        "stl_coordinates": points,
        # FLATTENED, int32: DoMINO feeds this straight to signed_distance_field.
        "stl_faces": np.ascontiguousarray(faces, dtype=np.int32).ravel(),
        "stl_centers": centers,
        "stl_areas": np.ascontiguousarray(areas, dtype=np.float32),
        "surface_normals": np.ascontiguousarray(normals, dtype=np.float32),
    }
    arrays["surface_mesh_centers"] = arrays["stl_centers"]
    arrays["surface_areas"] = arrays["stl_areas"]
    return arrays, faces


def _parent_cell_values(surface, source, num_triangles):
    """The source mesh's ``cell_data``, replicated onto each skin triangle.

    ``extract_surface`` carries no cell fields -- only ``surface:parent_cell``,
    the parent's **global block-major** index -- so the values are gathered
    here through :func:`meshioplusplus._regions.block_bases`, the single owner
    of that numbering. Nothing re-derives a block offset.
    """
    from .._regions import block_bases

    parents = surface.cell_data.get("surface:parent_cell")
    if parents is None or source is None:
        return {}
    parent = np.concatenate([np.asarray(a, dtype=np.int64) for a in parents])
    if len(parent) != num_triangles:
        return {}
    bases = block_bases(source.cells)
    out = {}
    for name, arrays in source.cell_data.items():
        if name.startswith(("surface:", "convert:")) or not all(
            _numeric(a) for a in arrays
        ):
            continue
        try:
            flat = np.concatenate(
                [np.asarray(a).reshape(len(a), -1) for a in arrays], axis=0
            )
        except ValueError:
            continue
        if len(flat) != int(bases[-1]):
            continue
        out[name] = flat[parent]
    return out


def _vertex_average(values, faces):
    """A nodal field at each triangle's vertices, averaged onto the triangle."""
    rows = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    return rows[faces].mean(axis=1)


def _field_block(names, arrays, rows):
    """Concatenate named arrays into ``(rows, sum widths)`` plus a sidecar."""
    if not names:
        return None, [], []
    blocks = []
    widths = []
    for name in names:
        values = _columns(arrays[name], rows)
        blocks.append(values)
        widths.append(int(values.shape[1]))
    return np.ascontiguousarray(np.concatenate(blocks, axis=1)), list(names), widths


def _check_global_name(name):
    if "volume" in str(name):
        raise WriteError(
            f"meshio++: cae: global parameter '{name}' contains 'volume'; the "
            "reader takes its volume row count from the first key whose name "
            "does, so such a name would corrupt every volume read"
        )


def write(
    filename,
    mesh,
    *,
    surface=None,
    surface_fields=None,
    volume_fields=None,
    global_params=None,
    global_params_reference=None,
    global_params_order=None,
    time=None,
    step=None,
):
    """Write one case in the CAE ``.npz`` sample layout.

    A mesh with 3-D cells is the *volume*: its nodes become
    ``volume_mesh_centers``, its numeric ``point_data`` becomes
    ``volume_fields``, and its skin (extracted and triangulated here) becomes
    the ``stl_*``/``surface_*`` half, with the volume's ``cell_data`` gathered
    onto each triangle through its parent cell. A surface mesh is written as
    the surface alone, with no volume keys -- which is what DoMINO's
    surface-only model type reads.

    ``surface_fields``/``volume_fields`` name and order the columns of the two
    concatenated blocks (default: every numeric array, sorted);
    ``global_params`` are case parameters such as ``{"stream_velocity": 30.0}``,
    written both individually and stacked. ``time``/``step`` become the
    shape-``(1,)`` ``TIME``/``STEP`` arrays when given.
    """
    from .. import _provenance

    dims = sorted({block.dim for block in mesh.cells}) if len(mesh.cells) else []
    top = dims[-1] if dims else 0
    if top < 2:
        raise WriteError(
            "meshio++: cae: this layout needs a surface or volume mesh; the "
            f"input's highest cell dimension is {top}"
        )

    volume = mesh if top == 3 else None
    if surface is None:
        surface = (
            _triangulated_surface(mesh) if volume is not None else _triangulate(mesh)
        )
    else:
        surface = _triangulate(surface)

    arrays, faces = _triangle_arrays(surface)
    num_triangles = len(faces)

    # Surface field candidates: the surface's own point data, vertex-averaged,
    # plus its cell data and whatever the parent volume cells carry.
    candidates = {}
    for name, values in surface.point_data.items():
        if _numeric(values):
            candidates[name] = _vertex_average(values, faces)
    for name, blocks in surface.cell_data.items():
        if name.startswith(("surface:", "convert:")):
            continue
        if all(_numeric(b) for b in blocks):
            try:
                candidates[name] = np.concatenate(
                    [np.asarray(b).reshape(len(b), -1) for b in blocks], axis=0
                )
            except ValueError:
                continue
    candidates.update(_parent_cell_values(surface, volume, num_triangles))
    candidates = {
        name: values
        for name, values in candidates.items()
        if len(np.asarray(values)) == num_triangles
    }

    if surface_fields is None:
        chosen_surface = sorted(candidates)
    else:
        chosen_surface = [str(name) for name in surface_fields]
        missing = [name for name in chosen_surface if name not in candidates]
        if missing:
            raise WriteError(
                f"meshio++: cae: no surface field named {missing[0]!r} "
                f"(available: {', '.join(sorted(candidates)) or 'none'})"
            )

    node_candidates = {}
    if volume is not None:
        node_candidates = {
            name: values
            for name, values in volume.point_data.items()
            if _numeric(values) and len(np.asarray(values)) == len(volume.points)
        }
    if volume_fields is None:
        chosen_volume = sorted(node_candidates)
    else:
        if volume is None:
            raise WriteError(
                "meshio++: cae: volume_fields was given but the input has no "
                "3-D cells, so there is no volume half to write"
            )
        chosen_volume = [str(name) for name in volume_fields]
        missing = [name for name in chosen_volume if name not in node_candidates]
        if missing:
            raise WriteError(
                f"meshio++: cae: no volume field named {missing[0]!r} "
                f"(available: {', '.join(sorted(node_candidates)) or 'none'})"
            )

    surface_block, surface_names, surface_widths = _field_block(
        chosen_surface, candidates, num_triangles
    )
    if surface_block is not None:
        arrays["surface_fields"] = surface_block

    node_names, node_widths = [], []
    if volume is not None:
        arrays["volume_mesh_centers"] = _as_f32_points(volume.points)
        volume_block, node_names, node_widths = _field_block(
            chosen_volume, node_candidates, len(volume.points)
        )
        if volume_block is not None:
            arrays["volume_fields"] = volume_block

    params = dict(global_params or {})
    references = dict(global_params_reference or {})
    for name in params:
        _check_global_name(name)
    unknown = sorted(set(references) - set(params))
    if unknown:
        raise WriteError(
            f"meshio++: cae: global_params_reference names {unknown[0]!r}, "
            "which has no matching global_params entry"
        )
    if global_params_order is None:
        order = sorted(params)
    else:
        order = [str(name) for name in global_params_order]
        if sorted(order) != sorted(params):
            raise WriteError(
                "meshio++: cae: global_params_order must name exactly the "
                f"global_params keys ({sorted(params)}), got {sorted(order)}"
            )
    for name in order:
        arrays[name] = np.array([float(params[name])], dtype=np.float32)
    if order:
        arrays["global_params_values"] = np.array(
            [[float(params[name])] for name in order], dtype=np.float32
        )
        arrays["global_params_reference"] = np.array(
            [[float(references.get(name, params[name]))] for name in order],
            dtype=np.float32,
        )

    if time is not None:
        arrays["TIME"] = np.array([float(time)], dtype=np.float32)
    if step is not None:
        arrays["STEP"] = np.array([int(step)], dtype=np.int64)

    if surface_names:
        arrays[SIDECAR_PREFIX + "surface_field_names"] = np.asarray(surface_names)
        arrays[SIDECAR_PREFIX + "surface_field_widths"] = np.asarray(
            surface_widths, dtype=np.int64
        )
    if node_names:
        # `node_`, never `volume_`: see this module's docstring.
        arrays[SIDECAR_PREFIX + "node_field_names"] = np.asarray(node_names)
        arrays[SIDECAR_PREFIX + "node_field_widths"] = np.asarray(
            node_widths, dtype=np.int64
        )

    block = _provenance.lines(_provenance.SlotTier.BLOCK)
    encoded = [line.encode("utf-8") for line in block]
    # One byte wider than the longest line, so EVERY row -- the longest one
    # included -- ends in at least one NUL. Without that pad the last line runs
    # straight into the zip's next local-file header and the head scanner reads
    # `...v10.35.0PK\x03\x04` as the tag.
    width = max((len(line) for line in encoded), default=1) + 1
    ordered = {PROVENANCE_KEY: np.asarray(encoded, dtype=f"S{width}")}
    ordered.update(arrays)
    for key in ordered:
        if key in (PROVENANCE_KEY,) or key in VOLUME_KEYS:
            continue
        if "volume" in key:
            raise WriteError(
                f"meshio++: cae: key '{key}' contains 'volume'; only "
                f"{' and '.join(VOLUME_KEYS)} may"
            )
    np.savez(filename, **ordered)


def _split_block(data, prefix, block_key):
    """A concatenated field block back into named arrays."""
    if block_key not in data:
        return {}
    block = np.asarray(data[block_key])
    names_key = SIDECAR_PREFIX + prefix + "_field_names"
    widths_key = SIDECAR_PREFIX + prefix + "_field_widths"
    if names_key not in data or widths_key not in data:
        # A file from another writer: keep the block whole rather than guess.
        return {block_key: block}
    names = [str(name) for name in np.asarray(data[names_key])]
    widths = [int(width) for width in np.asarray(data[widths_key])]
    out = {}
    start = 0
    for name, width in zip(names, widths):
        values = block[:, start : start + width]
        out[name] = values[:, 0] if width == 1 else np.ascontiguousarray(values)
        start += width
    return out


def _global_field_data(data):
    """Every scalar/stacked parameter plus ``TIME``/``STEP`` as field_data."""
    skip = set(SURFACE_KEYS) | set(VOLUME_KEYS) | {"surface_fields"}
    out = {}
    for key in data.files:
        if key in skip or key.startswith(SIDECAR_PREFIX):
            continue
        out[key] = np.asarray(data[key])
    return out


def read(filename, part="surface"):
    """Read one case back.

    ``part="surface"`` returns the triangulated surface with the field block
    split back into named ``cell_data`` (plus ``surface_normals``,
    ``surface_areas`` and ``stl_centers``); ``part="volume"`` returns the
    volume's nodes as a ``vertex`` point cloud with its own fields as
    ``point_data``. Global parameters, ``TIME`` and ``STEP`` ride along as
    ``field_data`` either way.
    """
    if part not in ("surface", "volume"):
        raise ReadError(
            f"meshio++: cae: part must be 'surface' or 'volume', not {part!r}"
        )
    with np.load(filename) as data:
        field_data = _global_field_data(data)
        if part == "volume":
            if "volume_mesh_centers" not in data.files:
                raise ReadError(
                    "meshio++: cae: this file carries no volume half; read it "
                    "with part='surface'"
                )
            points = np.asarray(data["volume_mesh_centers"])
            point_data = _split_block(data, "node", "volume_fields")
            cells = [
                CellBlock(
                    "vertex", np.arange(len(points), dtype=np.int64).reshape(-1, 1)
                )
            ]
            return Mesh(points, cells, point_data=point_data, field_data=field_data)

        if "stl_coordinates" not in data.files:
            raise ReadError(
                "meshio++: cae: this file carries no surface half "
                "(no stl_coordinates)"
            )
        points = np.asarray(data["stl_coordinates"])
        faces = np.asarray(data["stl_faces"]).reshape(-1, 3).astype(np.int64)
        cell_data = {}
        for name, values in _split_block(data, "surface", "surface_fields").items():
            cell_data[name] = [values]
        for key in ("surface_normals", "surface_areas", "stl_centers"):
            if key in data.files:
                cell_data[key] = [np.asarray(data[key])]
        return Mesh(
            points,
            [CellBlock("triangle", faces)],
            cell_data=cell_data,
            field_data=field_data,
        )


def export_cases(
    source,
    output_dir,
    *,
    name_template="case_{index}.npz",
    file_format=None,
    times=None,
    time_from="auto",
    **write_kwargs,
):
    """Write one ``.npz`` per step of a sequence source.

    The single owner of the batch loop the CLI verb and the MCP tool share.
    Streams: the plan is expanded first and each step read individually, so
    one mesh is alive at a time however large the dataset.

    :returns: the list of files written.
    """
    from .._helpers import read as read_mesh
    from .._sequence import sequence_entries

    entries = sequence_entries(
        source, file_format=file_format, times=times, time_from=time_from
    )
    os.makedirs(str(output_dir), exist_ok=True)
    written = []
    for index, entry in enumerate(entries):
        mesh = read_mesh(entry["path"], file_format, time_step=entry.get("step", 0))
        out = os.path.join(
            str(output_dir), name_template.format(index=index, step=index)
        )
        kwargs = dict(write_kwargs)
        kwargs.setdefault("time", entry.get("time"))
        kwargs.setdefault("step", index)
        write(out, mesh, **kwargs)
        written.append(out)
        del mesh
    return written


register_format("cae", [".npz"], read, {"cae": write})
