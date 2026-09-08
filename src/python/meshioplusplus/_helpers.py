from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from ._common import num_nodes_per_cell
from ._exceptions import ReadError, WriteError
from ._files import is_buffer
from ._mesh import CellBlock, Mesh

extension_to_filetypes = {}
reader_map = {}
_writer_map = {}

# Formats spread across sibling files, which a single in-memory buffer cannot
# serve. "gid"'s ascii flavour writes a `.post.msh`/`.post.res` pair; its
# binary/hdf5 flavours are single-file, but the buffer path cannot express a
# flavour choice, so the format as a whole is excluded here, same as the
# other three.
#
# "pmsh" and "zarr" are directory stores (many files under one path, the
# `openfoam` shape) and "usd" hands its path to a pxr layer, which opens the
# file itself -- none of the three can be given a buffer either.
_MULTIFILE_FORMATS = (
    "tetgen",
    "triangle",
    "ensight",
    "gid",
    "pmsh",
    "zarr",
    "usd",
)


def formats() -> dict:
    """What this build can read and write, and which extension implies which.

    ``{"readable": [...], "writable": [...], "extensions": {".stl": ["stl"], ...}}``
    — all sorted, and the extension keys carry their leading dot.

    Public because a file dialog needs exactly this and the registries backing
    it are private. Reaching into ``_helpers.reader_map`` is what
    ``tools/paraview-meshioplusplus-plugin.py`` does, with nothing to catch a
    rename; the Blender add-on calls this instead.
    """
    return {
        "readable": sorted(reader_map),
        "writable": sorted(_writer_map),
        "extensions": {
            ext: list(fmts) for ext, fmts in sorted(extension_to_filetypes.items())
        },
    }


def register_format(
    format_name: str, extensions: list[str], reader, writer_map
) -> None:
    for ext in extensions:
        if ext not in extension_to_filetypes:
            extension_to_filetypes[ext] = []
        extension_to_filetypes[ext].append(format_name)

    if reader is not None:
        reader_map[format_name] = reader

    _writer_map.update(writer_map)


def deregister_format(format_name: str):
    for value in extension_to_filetypes.values():
        if format_name in value:
            value.remove(format_name)

    if format_name in reader_map:
        reader_map.pop(format_name)

    if format_name in _writer_map:
        _writer_map.pop(format_name)


def _filetypes_from_path(path: Path) -> list[str]:
    ext = ""
    out: list[str] = []
    for suffix in reversed(path.suffixes):
        # `ext` grows from the shortest suffix to the longest as the loop
        # proceeds, so PREPENDING (not appending) puts the longest match
        # first -- what lets ".post.msh" resolve to "gid" rather than to
        # ".msh"'s own first candidate ("ansys") in _pick_best_format.
        # Behaviour-preserving for every extension registered before compound
        # ones existed: the only pre-existing compound-extension formats
        # (".vol.gz", ".post.gz", ".dato.gz") have no registered ".gz" alone,
        # so at most one iteration of this loop ever contributed to `out`
        # for any real filename -- prepend and append were indistinguishable
        # until "gid" made a shorter AND a longer suffix both match at once.
        ext = (suffix + ext).lower()
        try:
            out = extension_to_filetypes[ext] + out
        except KeyError:
            pass

    if not out:
        raise ReadError(f"Could not deduce file format from path '{path}'.")
    return out


def _apply_read_filter(mesh, points_only: bool, arrays):
    """Drop the data arrays a selective read did not ask for.

    Applied unconditionally after every read, which is what makes the option
    mean the same thing for all formats: readers with a native selective path
    (vtu/vtp/xdmf/gmsh) never materialize the unwanted arrays in the first
    place, and for them this pass is a no-op. Everything else is read whole and
    trimmed here -- correct, just not faster.
    """
    if not points_only and arrays is None:
        return mesh

    keep = set() if points_only else set(arrays)
    for attr in ("point_data", "cell_data", "field_data"):
        current = getattr(mesh, attr, None)
        if not current:
            continue
        for name in [k for k in current if k not in keep]:
            del current[name]
    return mesh


def _call_reader(reader, arg, points_only: bool, arrays, time_step: int = 0):
    """Invoke `reader`, passing the selective-read options when it accepts them.

    Readers are plain callables registered by each format, and only some take
    the options. Rather than maintaining a second registry of which do, ask
    directly -- and note that `_apply_read_filter` guarantees correct semantics
    either way, so this is purely an optimization.

    `time_step` is handled separately from the narrowing options, because it is
    not one: no caller-side filter can recover a step that was never read, so a
    non-default request must reach a reader that understands it or fail loudly
    rather than silently returning the first step.
    """
    kwargs = {}
    if points_only or arrays is not None:
        kwargs.update(points_only=points_only, arrays=arrays)
    if time_step:
        kwargs.update(time_step=time_step)
    if not kwargs:
        return reader(arg)
    try:
        return reader(arg, **kwargs)
    except TypeError:
        if time_step:
            # Unlike the narrowing options, this one cannot be emulated after
            # the fact -- say so instead of handing back the wrong step.
            raise ReadError(
                f"meshio++: this format's reader does not support time_step "
                f"(requested {time_step})"
            ) from None
        # Reader has no selective support; the caller-side filter handles it.
        return reader(arg)


def read(
    filename,
    file_format: Union[str, None] = None,
    points_only=False,
    arrays=None,
    time_step=0,
):
    """Reads an unstructured mesh with added data.

    :param filenames: The files/PathLikes to read from.
    :type filenames: str
    :param points_only: Read geometry only, skipping every data array.
    :param arrays: Restrict data to these names. ``None`` (default) reads every
        array; an empty list reads none. Names absent from the file are ignored.
    :param time_step: Which step of a multi-step file to materialize. ``0``
        (default) is the first, preserving the historical behaviour; negative
        counts from the end. Out of range is an error naming the available
        count, and a format whose reader has no time concept raises rather than
        silently returning the first step. Currently honoured by ``exodus``.

    :returns mesh{2,3}d: The mesh data.
    """
    if is_buffer(filename, "r"):
        mesh = _read_buffer(filename, file_format, points_only, arrays, time_step)
    else:
        mesh = _read_file(Path(filename), file_format, points_only, arrays, time_step)
    return _apply_read_filter(mesh, points_only, arrays)


def read_metadata(filename, file_format: Union[str, None] = None) -> dict:
    """Summarize a mesh file without loading its heavy arrays.

    Returns a dict with ``num_points``, ``point_dim``, ``num_cells``,
    ``cell_blocks``, the ``*_data_names`` lists, ``format``, ``time_values`` and
    ``fell_back_to_full_read``. ``bbox_min``/``bbox_max`` are present only when
    a bounding box was computed -- absent rather than ``None``, so "not
    computed" cannot be mistaken for a real box at the origin.

    ``provenance`` is the block the file carries (one line per entry, comment
    punctuation stripped), always present and empty for a file with none;
    ``provenance_recognised`` says whether its first line is meshio++'s own
    tag format. See :doc:`provenance`.

    ``regions`` lists each named region's ``name``/``kind``/``dim``/``tag``/
    ``num_entries`` (not the entries themselves) -- always present, empty for a
    format that maps no regions or on a native metadata path that declined a
    full read (VTU/VTP/XDMF/Gmsh 4.1 today; none of those currently map regions
    at all, so this is never a wrong answer). Cheap whenever the summary was
    produced from an already-read mesh (every fallback path, and Exodus, which
    always falls back).

    ``fell_back_to_full_read`` is the honest part: formats without a native
    metadata path are read in full and summarized. The answer is always
    correct; this says whether it was cheap.
    """
    if not is_buffer(filename, "r"):
        try:
            from . import _core

            return _core.read_metadata(str(filename), file_format or "")
        except Exception:
            pass

    # Fallback for buffers, Python-only formats, and any file the C++ core
    # declines (gmsh $Entities, multi-piece VTU, ...). Deliberately lives here
    # rather than in each format shim: read_metadata has no per-format Python
    # twin, so one fallback covers every format at once.
    mesh = read(filename, file_format)
    resolved_format = file_format
    if resolved_format is None and not is_buffer(filename, "r"):
        # The C++ attempt above never got a name to report either (it was
        # given "" too), so resolve it the same way `_read_file` would --
        # otherwise a file_format=None caller silently gets format="" here.
        path = Path(filename)
        try:
            resolved_format = _filetypes_from_path(path)[0]
        except ReadError:
            from ._sniff import sniff_format

            resolved_format = sniff_format(path) or None
    out = _metadata_from_mesh(mesh, resolved_format)
    # Provenance lives in the file's bytes, so unlike everything else in a
    # summary it cannot come from the Mesh -- fill it here, where the path is
    # still in hand. A buffer has no path to scan and keeps the empty default.
    if not is_buffer(filename, "r"):
        from ._provenance import read_provenance_lines

        lines_found, recognised = read_provenance_lines(filename)
        out["provenance"] = lines_found
        out["provenance_recognised"] = recognised
    return out


def _metadata_from_mesh(mesh, file_format: Union[str, None]) -> dict:
    """The pure-Python twin of the C++ ``metadata_from_mesh``."""
    import numpy as np

    blocks = []
    for block in mesh.cells:
        data = np.asarray(block.data) if not isinstance(block.data, list) else None
        blocks.append(
            {
                "type": block.type,
                "num_cells": len(block.data),
                "nodes_per_cell": (
                    int(data.shape[1]) if data is not None and data.ndim >= 2 else 0
                ),
                "ragged": data is None,
            }
        )

    out = {
        "num_points": len(mesh.points),
        "point_dim": int(np.asarray(mesh.points).shape[1]) if len(mesh.points) else 0,
        "num_cells": sum(b["num_cells"] for b in blocks),
        "cell_blocks": blocks,
        "point_data_names": sorted(mesh.point_data),
        "cell_data_names": sorted(mesh.cell_data),
        "field_data_names": sorted(mesh.field_data),
        "fell_back_to_full_read": True,
        "format": file_format or "",
        # Always present. A pure-Python read hands back a Mesh, which holds one
        # step and no record of how many there were -- empty unless the format's
        # reader attached the full series as a `.time_values` side channel on
        # the mesh itself (the Exodus reader does this; see `_exodus.py`).
        "time_values": [float(v) for v in getattr(mesh, "time_values", []) or []],
        # The mesh is already in memory (this path always fully reads), so
        # regions cost nothing extra to report -- mirrors metadata_from_mesh.
        "regions": [
            {
                "name": r.name,
                "kind": r.kind,
                "dim": r.dim,
                "tag": r.tag,
                "num_entries": len(r),
            }
            for r in getattr(mesh, "regions", [])
        ],
        # Always present, matching the C++ path. Filled by `read_metadata`
        # when it has a real path; a Mesh alone cannot supply it.
        "provenance": [],
        "provenance_recognised": False,
    }
    if len(mesh.points):
        pts = np.asarray(mesh.points, dtype=float)
        out["bbox_min"] = tuple(pts.min(axis=0)[:3])
        out["bbox_max"] = tuple(pts.max(axis=0)[:3])
    return out


def _read_buffer(
    filename, file_format: Union[str, None], points_only=False, arrays=None, time_step=0
):
    if file_format is None:
        raise ReadError("File format must be given if buffer is used")
    if file_format in _MULTIFILE_FORMATS:
        raise ReadError(
            f"{file_format} format is spread across multiple files "
            "and so cannot be read from a buffer"
        )
    if file_format not in reader_map:
        raise ReadError(f"Unknown file format '{file_format}'")

    return _call_reader(
        reader_map[file_format], filename, points_only, arrays, time_step
    )


def _read_file(
    path: Path,
    file_format: Union[str, None],
    points_only=False,
    arrays=None,
    time_step=0,
):
    if not path.exists():
        raise ReadError(f"File {path} not found.")

    if file_format:
        possible_file_formats = [file_format]
    else:
        # deduce possible file formats from extension, falling back to a
        # conservative content sniff when the extension is unknown/ambiguous
        try:
            possible_file_formats = _filetypes_from_path(path)
        except ReadError:
            from ._sniff import sniff_format

            sniffed = sniff_format(path)
            if not sniffed:
                raise
            possible_file_formats = [sniffed]

    for file_format in possible_file_formats:
        if file_format not in reader_map:
            raise ReadError(f"Unknown file format '{file_format}' of '{path}'.")

        try:
            return _call_reader(
                reader_map[file_format], str(path), points_only, arrays, time_step
            )
        except ReadError as e:
            print(e)

    if len(possible_file_formats) == 1:
        msg = f"Couldn't read file {path} as {possible_file_formats[0]}"
    else:
        lst = ", ".join(possible_file_formats)
        msg = f"Couldn't read file {path} as either of {lst}"

    raise ReadError(msg)


def write_points_cells(
    filename,
    points: ArrayLike,
    cells: Union[dict[str, ArrayLike], list[Union[tuple[str, ArrayLike], CellBlock]]],
    point_data: Union[dict[str, ArrayLike], None] = None,
    cell_data: Union[dict[str, list[ArrayLike]], None] = None,
    field_data=None,
    point_sets: Union[dict[str, ArrayLike], None] = None,
    cell_sets: Union[dict[str, list[ArrayLike]], None] = None,
    file_format: Union[str, None] = None,
    **kwargs,
):
    points = np.asarray(points)
    mesh = Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )
    mesh.write(filename, file_format=file_format, **kwargs)


def _pick_best_format(file_formats, mesh):
    # Only reconsider when gmsh is the extension's OWN ambiguity (a plain
    # ".msh", whose registered candidates are ansys/gmsh/freefem) -- not when
    # a longer, unambiguous compound extension already resolved to something
    # else (".post.msh" -> gid, which still lists "gmsh" as a shorter-suffix
    # candidate in `file_formats`). A specific extension match must beat this
    # tag heuristic, so the heuristic only applies when gmsh (or one of its
    # ".msh" siblings) is genuinely the top candidate.
    if file_formats[0] in ("ansys", "freefem", "gmsh") and "gmsh" in file_formats:
        gmsh_keys = {"gmsh:physical", "gmsh:geometrical", "gmsh:dim_tags"}
        med_keys = {"cell_tags", "point_tags"}
        has_gmsh = bool(gmsh_keys & set(mesh.cell_data.keys())) or bool(
            gmsh_keys & set(mesh.point_data.keys())
        )
        has_med = (
            bool(med_keys & set(mesh.cell_data.keys()))
            or bool(med_keys & set(mesh.point_data.keys()))
            or any(k.startswith("med:") for k in mesh.field_data)
        )
        if has_gmsh or has_med:
            return "gmsh"
    return file_formats[0]


def write(filename, mesh: Mesh, file_format: Union[str, None] = None, **kwargs):
    """Writes mesh together with data to a file.

    :params filename: File to write to.
    :type filename: str

    :params point_data: Named additional point data to write to the file.
    :type point_data: dict
    """
    if is_buffer(filename, "r"):
        if file_format is None:
            raise WriteError("File format must be supplied if `filename` is a buffer")
        if file_format in _MULTIFILE_FORMATS:
            raise WriteError(
                f"{file_format} format is spread across multiple files, "
                "and so cannot be written to a buffer"
            )
    else:
        path = Path(filename)
        if not file_format:
            # deduce possible file formats from extension
            file_formats = _filetypes_from_path(path)
            # Pick the best format when several match the extension
            file_format = _pick_best_format(file_formats, mesh)

    try:
        writer = _writer_map[file_format]
    except KeyError:
        formats = sorted(list(_writer_map.keys()))
        raise WriteError(f"Unknown format '{file_format}'. Pick one of {formats}")

    # check cells for sanity
    for cell_block in mesh.cells:
        key = cell_block.type
        value = cell_block.data
        if key in num_nodes_per_cell:
            if value.shape[1] != num_nodes_per_cell[key]:
                raise WriteError(
                    f"Unexpected cells array shape {value.shape} for {key} cells. "
                    + f"Expected shape [:, {num_nodes_per_cell[key]}]."
                )
        else:
            # we allow custom keys <https://github.com/nschloe/meshio/issues/501> and
            # cannot check those
            pass

    # Write
    # Bound scope-less provenance notes to this write, so a note raised by an
    # earlier operation cannot attach itself to this file. No-op inside a
    # caller's own scope. See _provenance.begin_write().
    from ._provenance import begin_write

    begin_write()
    return writer(filename, mesh, **kwargs)
