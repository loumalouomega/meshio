"""Zarr mesh I/O in PhysicsNeMo's own ``io_zarr`` layout.

A ``.zarr`` store is a **directory**, and this writes the exact group/array
shape ``physicsnemo.mesh.io.from_zarr`` reads: a Zarr **v3** group carrying
``physicsnemo_mesh_type="Mesh"``, arrays ``points``/``cells``, and groups
``point_data``/``cell_data``/``global_data``. ``MeshReader`` accepts such a
store wherever it accepts a ``.pmsh`` (it sniffs for ``zarr.json`` inside the
sample directory), so a training set can be written in either shape.

Zarr **3.x is required to write** and the error says so by name: zarr-python
2.x cannot produce a v3 store at all, and a v2 store is not what upstream
reads. Reading uses only ``open_group``, so it is not the version that gates
that half -- but a store this writer did not produce is refused explicitly
rather than half-understood, which is why ``write_dataset``'s own zarr output
(a different layout entirely, see :doc:`ml`) fails here by name.

The single-simplex reduction, and every drop it makes, is the physicsnemo
bridge's own ``_to_physicsnemo_payload`` -- imported lazily, so
``import meshioplusplus`` still never pulls in that subpackage.
"""

from __future__ import annotations

import os
import shutil
import warnings

import numpy as np

from .._exceptions import ReadError, WriteError
from .._helpers import register_format

#: The root attribute upstream dispatches on.
TYPE_ATTR = "physicsnemo_mesh_type"

#: Where the provenance block rides. Zarr has no comment syntax, but a group
#: attribute is exactly the free-text slot a header comment is elsewhere -- and
#: being JSON, `read_provenance_lines` recovers it with no zarr import at all.
PROVENANCE_ATTR = "meshioplusplus:provenance"

#: Upstream's own chunk policy, so a store written here reads back with the
#: same partial-read behaviour as one written by `to_zarr`.
DEFAULT_CHUNK_ROWS = 200_000

DATA_GROUPS = ("point_data", "cell_data", "global_data")


def _require_zarr(op):
    """The zarr module, or a named install error; writing needs 3.x."""
    from .._interop import _require

    return _require("zarr", "zarr", op)


def _require_zarr3(op):
    zarr = _require_zarr(op)
    version = getattr(zarr, "__version__", "0")
    try:
        major = int(str(version).split(".")[0])
    except ValueError:
        major = 0
    if major < 3:
        raise ImportError(
            f"meshio++: {op}: writing this layout needs zarr>=3 (found "
            f"{version}); zarr-python 2.x cannot produce a v3 store, which is "
            "what physicsnemo's own reader expects. Upgrade with "
            "`pip install 'zarr>=3'`."
        )
    return zarr


def _tensordict_attrs(batch_shape):
    """The sidecar attribute tensordict's own reader wants.

    ``from_zarr`` never looks at it, but a store written here should be
    openable with ``tensordict.from_zarr`` too, and it costs one dict.
    """
    return {"batch_size": [int(n) for n in batch_shape], "version": 1}


def _create_array(group, name, values, chunk_rows, zstd_level, zarr):
    # `asarray`, never `ascontiguousarray`: the latter promotes a 0-d array to
    # 1-d, which would turn a scalar `global_data` entry into a length-1 one
    # and lose the distinction upstream's own stores keep.
    values = np.asarray(values)
    if values.ndim == 0:
        chunks = ()
    else:
        values = np.ascontiguousarray(values)
        rows = min(int(chunk_rows), max(int(values.shape[0]), 1))
        chunks = (rows,) + tuple(int(n) for n in values.shape[1:])
    # `compressors` must be passed EXPLICITLY either way: omitting it does not
    # mean "no compression", it means zarr's own default, which is zstd at
    # level 0.
    kwargs = {
        "shape": values.shape,
        "dtype": values.dtype,
        "chunks": chunks,
        "compressors": (
            [zarr.codecs.ZstdCodec(level=int(zstd_level))] if zstd_level > 0 else None
        ),
    }
    array = group.create_array(name, **kwargs)
    array[...] = values
    return array


def _looks_like_a_store(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "zarr.json"))


def write(
    filename,
    mesh,
    *,
    manifold_dim="auto",
    float32=True,
    chunk_rows=DEFAULT_CHUNK_ROWS,
    zstd_level=3,
):
    """Write ``mesh`` as a Zarr v3 store in the physicsnemo mesh layout.

    ``manifold_dim``/``float32`` are :func:`meshioplusplus.pmsh.write`'s;
    ``chunk_rows``/``zstd_level`` are upstream's own chunk and compression
    policy, and ``zstd_level=0`` writes uncompressed.
    """
    from .. import _provenance
    from .._interop import _emit
    from ..physicsnemo import _to_physicsnemo_payload

    zarr = _require_zarr3("zarr")
    payload = _to_physicsnemo_payload(mesh, manifold_dim=manifold_dim, float32=float32)
    _emit("zarr", payload["notes"])

    root_path = str(filename)
    if os.path.exists(root_path):
        if not _looks_like_a_store(root_path):
            raise WriteError(
                f"meshio++: zarr: '{root_path}' exists and is not a zarr store "
                "(no zarr.json); refusing to remove it"
            )
        shutil.rmtree(root_path)

    root = zarr.open_group(root_path, mode="w")
    root.attrs[TYPE_ATTR] = "Mesh"
    root.attrs["__tensordict__"] = _tensordict_attrs([])
    root.attrs[PROVENANCE_ATTR] = _provenance.lines(_provenance.SlotTier.BLOCK)

    points = np.ascontiguousarray(payload["points"])
    cells = payload["cells"]
    if cells is None:
        cells = np.zeros((0, 1), dtype=np.int64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    _create_array(root, "points", points, chunk_rows, zstd_level, zarr)
    _create_array(root, "cells", cells, chunk_rows, zstd_level, zarr)

    batches = {
        "point_data": [len(points)],
        "cell_data": [len(cells)],
        "global_data": [],
    }
    for name in DATA_GROUPS:
        group = root.create_group(name)
        group.attrs["__tensordict__"] = _tensordict_attrs(batches[name])
        for key, values in payload[name].items():
            _create_array(group, key, np.asarray(values), chunk_rows, zstd_level, zarr)

    # Upstream's own `to_zarr` consolidates too, and a store written without
    # it reads more slowly over a network filesystem. zarr warns that v3
    # consolidated metadata is not in the spec yet; that is upstream's choice
    # to live with, not a caller's problem, so it is silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(root.store)


def _read_group(root, name, notes):
    if name not in root:
        return {}
    group = root[name]
    out = {}
    for key, array in group.arrays():
        out[str(key)] = np.asarray(array[...])
    for key, _ in group.groups():
        notes.append(
            f"'{name}/{key}' is a nested group; meshio++ data dictionaries are "
            "flat, so it was dropped"
        )
    return out


def read(filename):
    """Read a Zarr store written in the physicsnemo mesh layout."""
    from .._interop import _emit
    from ..physicsnemo import _from_physicsnemo_payload

    zarr = _require_zarr("zarr")
    root_path = str(filename)
    if not os.path.isdir(root_path):
        raise ReadError(
            f"meshio++: zarr: '{root_path}' is not a directory; a zarr store "
            "is a directory of chunked arrays, not a single file"
        )
    try:
        root = zarr.open_group(root_path, mode="r")
    except Exception as e:
        raise ReadError(f"meshio++: zarr: cannot open '{root_path}': {e}") from None

    attrs = dict(root.attrs)
    if "meshioplusplus_dataset" in attrs:
        raise ReadError(
            f"meshio++: zarr: '{root_path}' is a write_dataset() training "
            "dataset (one subgroup per mesh, tabular columns), not a single "
            "mesh; read it with the tooling in doc/ml.md instead"
        )
    declared = attrs.get(TYPE_ATTR)
    if declared == "DomainMesh":
        raise ReadError(
            f"meshio++: zarr: '{root_path}' holds a physicsnemo DomainMesh "
            "(an interior mesh plus named boundaries); meshio++ reads a single "
            "Mesh store"
        )
    if declared not in (None, "Mesh"):
        raise ReadError(
            f"meshio++: zarr: '{root_path}' declares {TYPE_ATTR}={declared!r}; "
            "only 'Mesh' is supported"
        )
    if "points" not in root:
        raise ReadError(f"meshio++: zarr: '{root_path}' has no points array")

    notes = []
    points = np.asarray(root["points"][...])
    cells = None
    if "cells" in root:
        cells = np.asarray(root["cells"][...])
        if cells.size == 0:
            cells = None

    payload = {
        "points": points,
        "cells": cells,
        "point_data": _read_group(root, "point_data", notes),
        "cell_data": _read_group(root, "cell_data", notes),
        "global_data": _read_group(root, "global_data", notes),
    }
    _emit("zarr", notes)
    return _from_physicsnemo_payload(payload)


register_format("zarr", [".zarr"], read, {"zarr": write})
