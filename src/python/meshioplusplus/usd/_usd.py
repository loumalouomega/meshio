"""OpenUSD (``.usd``/``.usda``/``.usdc``): time-sampled stages for a twin.

The interchange format Omniverse, usdview and every USD-aware DCC tool read,
so a solve (or a surrogate's prediction) becomes a scrubbable 3-D asset. The
layout is the plain UsdGeom one every viewer understands, transcribed from the
Kratos ``PhysicsNeMoApplication``'s own exporter: one ``UsdGeomMesh`` (or
``UsdGeomPoints`` for a cloud) per prim, ``points`` time-sampled per step, and
topology time-sampled **only on the steps where it actually changes**, so an
adaptive-remeshing series stays valid while a fixed-topology one stays compact.

Three things are worth not rediscovering:

* **USD holds n-gons natively** (``faceVertexCounts`` alongside
  ``faceVertexIndices``), so unlike the trimesh and physicsnemo bridges this
  writer does **not** simplexify: a quad-dominant hex skin arrives as quads.
* **Fields are primvars, and interpolation is the whole contract.** Point data
  is ``vertex``-interpolated, cell data ``uniform``; a width-3 array becomes a
  ``Float3Array`` and anything else a flat array with ``SetElementSize``.
  Getting the interpolation token wrong renders plausible, wrong colours.
* **``Usd.Stage.CreateNew`` refuses to clobber an existing layer**, so the
  target is unlinked first.

``usd-core`` (Pixar's self-contained PyPI build -- no Omniverse install) is an
optional dependency behind the ``[usd]`` extra, imported lazily inside the
functions that need it. The format is registered unconditionally so a missing
install raises this format's own named error rather than letting ``.usda``
fall through to something else.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .._common import warn
from .._exceptions import ReadError, WriteError
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

#: Where the provenance block rides: the root layer's ``documentation`` field,
#: USD's own free-text slot. A text ``.usda`` therefore also surrenders it to
#: the ordinary byte scanner; a crate layer needs pxr, which is why
#: `read_provenance_lines` prefers the pxr path when it is available.
PROVENANCE_SLOT = "documentation"

#: Custom-data key holding the hash of the topology last authored. Kratos's own
#: spelling, kept verbatim so a stage written by either tool reads the same.
TOPOLOGY_KEY = "physicsNemo:lastTopologyHash"

DEFAULT_PRIM_PATH = "/meshioplusplus/mesh"

#: meshio++ cell types USD can hold directly, by node count. Everything else
#: is either reduced first (3-D cells -> their skin, higher-order -> linear) or
#: dropped with a note.
_SURFACE_TYPES = {"triangle": 3, "quad": 4}


def _require_usd(op):
    """The four pxr modules this format needs, or a named install error."""
    from .._interop import _require

    _require("pxr", "usd", op)
    from pxr import Sdf, Usd, UsdGeom, Vt

    return Usd, UsdGeom, Sdf, Vt


def _numeric(array):
    return np.asarray(array).dtype.kind in "fiub"


def _as_vec3(points):
    """Points as contiguous ``(n, 3)`` float32, 2-D padded with zeros."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise WriteError(f"meshio++: usd: points must be 2-D, got {points.shape}")
    if points.shape[1] == 3:
        return np.ascontiguousarray(points, dtype=np.float32)
    padded = np.zeros((len(points), 3), dtype=np.float32)
    padded[:, : points.shape[1]] = points[:, :3]
    return padded


def _surface_payload(mesh):
    """The renderable surface of ``mesh``, plus the notes taken getting there.

    A volume mesh becomes its skin (points compacted, ``point_data`` gathered),
    with the volume's own ``cell_data`` gathered onto each face through the
    parent-cell map -- ``extract_surface`` carries only that map, never the
    fields. A surface mesh is used as-is, with higher-order blocks linearized;
    a mesh with no 2-D or 3-D cells is a point cloud.
    """
    from .._convert_cells import convert_cells
    from .._regions import block_bases
    from .._surface import extract_surface

    notes = []
    dims = sorted({block.dim for block in mesh.cells}) if len(mesh.cells) else []
    top = dims[-1] if dims else 0

    if top < 2:
        if len(mesh.cells):
            notes.append(
                "USD renders surfaces and point clouds; "
                f"{len(mesh.cells)} cell block(s) of dimension {top} were "
                "dropped and the points written as a cloud"
            )
        return mesh.points, [], dict(mesh.point_data), {}, notes

    source = mesh
    if top == 3:
        surface = extract_surface(mesh, record_parent_ids=True)
        notes.append("volume cells were replaced by their boundary surface")
    else:
        surface = mesh
        source = None

    if any(
        block.type not in _SURFACE_TYPES and block.dim == 2 for block in surface.cells
    ):
        surface = convert_cells(surface, mode="linearize")
        notes.append("higher-order surface cells were linearized")

    blocks = [b for b in surface.cells if b.dim == 2]
    if not blocks:
        return surface.points, [], dict(surface.point_data), {}, notes

    point_data = {
        name: np.asarray(values)
        for name, values in surface.point_data.items()
        if _numeric(values)
    }
    dropped = [n for n in surface.point_data if n not in point_data]
    if dropped:
        notes.append(f"non-numeric point_data {sorted(dropped)} was dropped")

    # Cell data, per kept block. The surface's own arrays first, then the
    # parent volume's gathered through `surface:parent_cell` (global
    # block-major -- resolved via block_bases, never re-derived).
    keep = [i for i, b in enumerate(surface.cells) if b.dim == 2]
    cell_data = {}
    for name, arrays in surface.cell_data.items():
        if name.startswith(("surface:", "convert:")):
            continue
        rows = [np.asarray(arrays[i]) for i in keep]
        if rows and all(_numeric(r) for r in rows):
            cell_data[name] = rows

    parents = surface.cell_data.get("surface:parent_cell")
    if parents is not None and source is not None:
        bases = block_bases(source.cells)
        parent_rows = [np.asarray(parents[i], dtype=np.int64) for i in keep]
        for name, arrays in source.cell_data.items():
            if name.startswith(("surface:", "convert:")) or name in cell_data:
                continue
            if not all(_numeric(a) for a in arrays):
                continue
            try:
                flat = np.concatenate(
                    [np.asarray(a).reshape(len(a), -1) for a in arrays], axis=0
                )
            except ValueError:
                continue
            if len(flat) != int(bases[-1]):
                continue
            cell_data[name] = [flat[rows] for rows in parent_rows]

    return surface.points, blocks, point_data, cell_data, notes


def _topology(blocks):
    """``(faceVertexCounts, faceVertexIndices)`` for the kept surface blocks."""
    counts = []
    indices = []
    for block in blocks:
        if isinstance(block.data, list):
            for row in block.data:
                row = np.asarray(row, dtype=np.int64)
                counts.append(len(row))
                indices.append(row)
        else:
            data = np.asarray(block.data, dtype=np.int64)
            counts.extend([data.shape[1]] * len(data))
            indices.append(data.ravel())
    if not counts:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
    return (
        np.asarray(counts, dtype=np.int32),
        np.concatenate([np.asarray(i, dtype=np.int32) for i in indices]),
    )


class SeriesWriter:
    """One stage, many time samples -- the fan-in the sequence engine drives.

    ``write(time, mesh)`` authors one sample; the topology is re-authored only
    when it changes, so a fixed-topology run leaves ``faceVertexIndices`` with
    no time samples at all and a remeshed one carries a sample per change.
    Streams: nothing but the current mesh is held.
    """

    def __init__(
        self,
        path,
        *,
        prim_path=DEFAULT_PRIM_PATH,
        up_axis="Z",
        meters_per_unit=1.0,
        time_codes_per_second=1.0,
    ):
        Usd, UsdGeom, _, _ = _require_usd("usd")
        if up_axis not in ("Y", "Z"):
            raise WriteError(
                f"meshio++: usd: up_axis must be 'Y' or 'Z', not {up_axis!r}"
            )
        if not str(prim_path).startswith("/"):
            raise WriteError(
                f"meshio++: usd: prim_path must be absolute, got {prim_path!r}"
            )
        target = str(path)
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(target):
            # CreateNew refuses to clobber an existing layer.
            os.unlink(target)
        self._stage = Usd.Stage.CreateNew(target)
        UsdGeom.SetStageUpAxis(
            self._stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y
        )
        UsdGeom.SetStageMetersPerUnit(self._stage, float(meters_per_unit))
        self._stage.SetTimeCodesPerSecond(float(time_codes_per_second))
        self._prim_path = str(prim_path)
        self._samples = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _update_time_range(self, time_code):
        stage = self._stage
        if stage.HasAuthoredTimeCodeRange():
            start = min(stage.GetStartTimeCode(), time_code)
            end = max(stage.GetEndTimeCode(), time_code)
        else:
            start = end = time_code
        stage.SetStartTimeCode(start)
        stage.SetEndTimeCode(end)

    def write(self, time, mesh):
        """Author one sample at ``time``; ``None`` authors a default one.

        A default-value sample is what a plain single-mesh write produces: the
        stage then carries no time samples at all, which is what a viewer with
        no timeline expects to see.
        """
        from .._interop import _emit

        Usd, UsdGeom, Sdf, Vt = _require_usd("usd")
        code = Usd.TimeCode.Default() if time is None else Usd.TimeCode(float(time))

        points, blocks, point_data, cell_data, notes = _surface_payload(mesh)
        _emit("usd", notes)
        coords = _as_vec3(points)

        if blocks:
            geom = UsdGeom.Mesh.Define(self._stage, self._prim_path)
            counts, indices = _topology(blocks)
            digest = hashlib.sha1(counts.tobytes() + indices.tobytes()).hexdigest()
            prim = geom.GetPrim()
            if prim.GetCustomDataByKey(TOPOLOGY_KEY) != digest:
                geom.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts), code)
                geom.GetFaceVertexIndicesAttr().Set(
                    Vt.IntArray.FromNumpy(indices), code
                )
                prim.SetCustomDataByKey(TOPOLOGY_KEY, digest)
            num_cells = len(counts)
        else:
            geom = UsdGeom.Points.Define(self._stage, self._prim_path)
            num_cells = 0

        geom.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords), code)
        if len(coords):
            lo = coords.min(axis=0).astype(np.float32)
            hi = coords.max(axis=0).astype(np.float32)
            geom.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi])), code)

        primvars = UsdGeom.PrimvarsAPI(geom.GetPrim())
        for name, values in sorted(point_data.items()):
            _set_primvar(
                UsdGeom, Sdf, Vt, primvars, name, values, len(coords), "vertex", code
            )
        for name, rows in sorted(cell_data.items()):
            try:
                joined = np.concatenate(
                    [np.asarray(r).reshape(len(r), -1) for r in rows], axis=0
                )
            except ValueError:
                continue
            _set_primvar(
                UsdGeom, Sdf, Vt, primvars, name, joined, num_cells, "uniform", code
            )

        if time is not None:
            self._update_time_range(float(time))
        self._samples += 1

    def close(self):
        """Write the provenance block and save the layer."""
        if self._closed:
            return
        from .. import _provenance

        layer = self._stage.GetRootLayer()
        layer.documentation = "\n".join(_provenance.lines(_provenance.SlotTier.BLOCK))
        layer.Save()
        self._closed = True


def _set_primvar(UsdGeom, Sdf, Vt, primvars, name, values, rows, interpolation, code):
    """One primvar, typed by width and interpolated by data location."""
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) != rows:
        warn(
            f"usd: '{name}' has shape {values.shape}, which does not match the "
            f"{rows} {interpolation} element(s); dropped"
        )
        return
    token = getattr(UsdGeom.Tokens, interpolation)
    width = values.shape[1]
    if values.dtype.kind in "iub" and width != 3:
        primvar = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.IntArray, token)
        if width != 1:
            primvar.SetElementSize(width)
        primvar.Set(
            Vt.IntArray.FromNumpy(np.ascontiguousarray(values, dtype=np.int32).ravel()),
            code,
        )
        return
    values = np.ascontiguousarray(values, dtype=np.float32)
    if width == 3:
        primvar = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.Float3Array, token)
        primvar.Set(Vt.Vec3fArray.FromNumpy(values), code)
        return
    primvar = primvars.CreatePrimvar(name, Sdf.ValueTypeNames.FloatArray, token)
    if width != 1:
        primvar.SetElementSize(width)
    primvar.Set(Vt.FloatArray.FromNumpy(values.ravel()), code)


def write(
    filename,
    mesh,
    *,
    prim_path=DEFAULT_PRIM_PATH,
    up_axis="Z",
    meters_per_unit=1.0,
    time_code=None,
):
    """Write ``mesh`` as a single-sample USD stage.

    A volume mesh is written as its boundary surface (n-gons kept -- USD holds
    them natively), a surface mesh as-is, a cell-less mesh as a point cloud.
    ``point_data`` becomes ``vertex`` primvars and ``cell_data`` ``uniform``
    ones. ``.usda`` is text, ``.usd``/``.usdc`` the binary crate encoding --
    the extension picks it. Needs ``usd-core`` (``pip install
    meshioplusplus[usd]``).
    """
    with SeriesWriter(
        filename,
        prim_path=prim_path,
        up_axis=up_axis,
        meters_per_unit=meters_per_unit,
    ) as writer:
        writer.write(time_code, mesh)


def _geom_prims(stage, UsdGeom):
    """Every renderable prim, in stage traversal order."""
    out = []
    for prim in stage.Traverse():
        if not prim.IsActive():
            continue
        if prim.IsA(UsdGeom.Mesh):
            out.append(("mesh", UsdGeom.Mesh(prim)))
        elif prim.IsA(UsdGeom.Points):
            out.append(("points", UsdGeom.Points(prim)))
    return out


def _sample_times(prims):
    """Every time code any prim's points attribute is sampled at, sorted."""
    times = set()
    for _, geom in prims:
        times.update(float(t) for t in geom.GetPointsAttr().GetTimeSamples())
    return sorted(times)


def _resolve_time_step(times, time_step):
    """Exodus's rule: 0 is the first step, negatives count from the end."""
    count = len(times)
    if count == 0:
        return None
    index = int(time_step)
    if index < 0:
        index += count
    if index < 0 or index >= count:
        raise ReadError(
            f"meshio++: usd: time step {time_step} is out of range; the stage "
            f"carries {count} step(s)"
        )
    return times[index]


def _blocks_from_topology(counts, indices):
    """Faces grouped by vertex count into triangle/quad/polygon blocks.

    Fixed order (triangle, quad, polygon) so the block layout is a function of
    the file's content rather than of its face order. An n-gon that is not a
    triangle or a quad lands in one ragged ``polygon`` block, which is what
    meshio++ stores a jagged row set as.
    """
    rows = []
    start = 0
    for count in counts:
        rows.append(indices[start : start + count])
        start += count
    tris = [r for r in rows if len(r) == 3]
    quads = [r for r in rows if len(r) == 4]
    polys = [r for r in rows if len(r) not in (3, 4)]
    blocks = []
    order = []
    if tris:
        blocks.append(CellBlock("triangle", np.asarray(tris, dtype=np.int64)))
        order.append([i for i, r in enumerate(rows) if len(r) == 3])
    if quads:
        blocks.append(CellBlock("quad", np.asarray(quads, dtype=np.int64)))
        order.append([i for i, r in enumerate(rows) if len(r) == 4])
    if polys:
        blocks.append(
            CellBlock("polygon", [np.asarray(r, dtype=np.int64) for r in polys])
        )
        order.append([i for i, r in enumerate(rows) if len(r) not in (3, 4)])
    return blocks, order


def _primvar_values(primvar, code):
    """One primvar's values as ``(rows, width)`` numpy, or ``None``."""
    values = primvar.Get(code)
    if values is None:
        values = primvar.Get()
    if values is None:
        return None
    array = np.asarray(values)
    if array.dtype == object:
        return None
    element = int(primvar.GetElementSize() or 1)
    if array.ndim == 1 and element > 1:
        array = array.reshape(-1, element)
    return array


def read(filename, time_step=0):
    """Read a USD stage's renderable prims back as one mesh.

    ``vertex`` primvars become ``point_data`` and ``uniform`` ones
    ``cell_data``; ``faceVarying`` and ``constant`` have no meshio++ counterpart
    and are skipped with a warning. Several prims are concatenated, with a
    ``usd:prim`` cell array naming which is which. ``time_step`` selects one
    authored sample (negatives count from the end), and the full list rides on
    the returned mesh as ``time_values`` so ``read_metadata`` reports it.
    """
    Usd, UsdGeom, _, _ = _require_usd("usd")
    target = str(filename)
    if not os.path.isfile(target):
        raise ReadError(f"meshio++: usd: '{target}' does not exist")
    stage = Usd.Stage.Open(target)
    if stage is None:
        raise ReadError(f"meshio++: usd: '{target}' is not a readable USD stage")

    prims = _geom_prims(stage, UsdGeom)
    if not prims:
        raise ReadError(
            f"meshio++: usd: '{target}' carries no UsdGeomMesh or UsdGeomPoints prim"
        )

    times = _sample_times(prims)
    chosen = _resolve_time_step(times, time_step)
    code = Usd.TimeCode.Default() if chosen is None else Usd.TimeCode(chosen)

    all_points = []
    all_blocks = []
    prim_of_block = []
    prim_paths = []
    point_arrays = {}
    cell_arrays = {}
    offset = 0
    skipped = set()

    for prim_index, (kind, geom) in enumerate(prims):
        prim_paths.append(str(geom.GetPrim().GetPath()))
        points = np.asarray(geom.GetPointsAttr().Get(code), dtype=np.float64)
        if points.size == 0:
            points = points.reshape(0, 3)
        all_points.append(points)

        blocks, order = [], []
        if kind == "mesh":
            counts = geom.GetFaceVertexCountsAttr().Get(code)
            indices = geom.GetFaceVertexIndicesAttr().Get(code)
            if counts is None:
                counts = geom.GetFaceVertexCountsAttr().Get()
            if indices is None:
                indices = geom.GetFaceVertexIndicesAttr().Get()
            if counts is not None and indices is not None:
                blocks, order = _blocks_from_topology(
                    np.asarray(counts, dtype=np.int64),
                    np.asarray(indices, dtype=np.int64) + offset,
                )
        all_blocks.extend(blocks)
        prim_of_block.extend([prim_index] * len(blocks))

        primvars = UsdGeom.PrimvarsAPI(geom.GetPrim())
        for primvar in primvars.GetPrimvars():
            name = str(primvar.GetPrimvarName())
            interpolation = str(primvar.GetInterpolation())
            values = _primvar_values(primvar, code)
            if values is None:
                continue
            if interpolation == "vertex" and len(values) == len(points):
                point_arrays.setdefault(name, {})[prim_index] = values
            elif interpolation == "uniform" and order:
                if len(values) != sum(len(rows) for rows in order):
                    continue
                # `order` holds, per emitted block, the face rows it took from
                # this prim's own face list -- so a uniform primvar splits back
                # by exactly the same grouping the topology did.
                cell_arrays.setdefault(name, {})[prim_index] = [
                    values[np.asarray(rows, dtype=np.int64)] for rows in order
                ]
            else:
                skipped.add(interpolation)
        offset += len(points)

    if skipped:
        warn(
            "usd: primvars interpolated as "
            f"{', '.join(sorted(skipped))} have no meshio++ counterpart and "
            "were skipped"
        )

    points = (
        np.concatenate(all_points) if all_points else np.zeros((0, 3), dtype=np.float64)
    )

    point_data = {}
    for name, per_prim in point_arrays.items():
        if len(per_prim) != len(prims):
            # A field only some prims carry cannot be a whole-mesh array.
            warn(f"usd: point primvar '{name}' is not on every prim; dropped")
            continue
        point_data[name] = np.concatenate(
            [np.asarray(per_prim[i]) for i in range(len(prims))]
        )

    cell_data = {}
    for name, per_prim in cell_arrays.items():
        blocks = []
        cursors = {index: 0 for index in per_prim}
        for block_index, block in enumerate(all_blocks):
            owner = prim_of_block[block_index]
            if owner in per_prim:
                blocks.append(per_prim[owner][cursors[owner]])
                cursors[owner] += 1
            else:
                # `cell_data` must carry one array per block; a prim that does
                # not have this primvar contributes zeros rather than a hole.
                sample = next(iter(per_prim.values()))[0]
                width = np.asarray(sample).reshape(len(sample), -1).shape[1]
                shape = (len(block.data),) + ((width,) if width > 1 else ())
                blocks.append(np.zeros(shape, dtype=np.float64))
        cell_data[name] = blocks

    if len(prims) > 1 and all_blocks:
        cell_data["usd:prim"] = [
            np.full(len(block.data), prim_of_block[i], dtype=np.int64)
            for i, block in enumerate(all_blocks)
        ]

    field_data = {}
    if len(prims) > 1:
        field_data["usd:prim_paths"] = np.asarray(prim_paths)

    mesh = Mesh(
        points,
        all_blocks,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
    )
    # The Exodus side channel: `_metadata_from_mesh` reads this, so
    # `read_metadata(...)["time_values"]` and the sequence engine's step count
    # both work with no per-format metadata reader.
    mesh.time_values = times
    return mesh


register_format("usd", [".usd", ".usda", ".usdc"], read, {"usd": write})
