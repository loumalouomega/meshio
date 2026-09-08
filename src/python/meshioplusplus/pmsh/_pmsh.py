"""PhysicsNeMo ``.pmsh``: the tensordict memory-mapped mesh directory.

A ``.pmsh`` is a **directory**, not a file -- ``physicsnemo.mesh.Mesh.save``
uses the path it is given verbatim, and the ``.pmsh`` name is only what
``MeshReader``'s default ``**/*.pmsh`` glob matches. The layout is a
tensorclass memmap tree: two ``meta.json`` files describing shapes and torch
dtypes, one per data group, and raw headerless little-endian C-order blobs.

**Hand-rolled in pure numpy rather than delegating to ``Mesh.save``**, which
is the whole point: a solver box with no torch, no CUDA and no NVIDIA stack
must still be able to write the training set the framework reads. The gated
parity tests pin this against physicsnemo's own ``save``/``load``, in both
directions, so the transcription cannot drift.

Two facts are load-bearing and neither is guessable from the docs:

* **A zero-element array has no blob.** tensordict's memmap format does not
  persist an empty tensor, so ``Mesh.save`` on a point cloud writes no
  ``cells.memmap`` at all while still declaring ``"shape": [0, 1]``. A reader
  must therefore treat a declared-but-absent blob as empty rather than as a
  corrupt store, and a writer must omit it or produce a tree that differs
  from theirs.
* **The extension does not pick the codec.** ``MeshReader._load_sample``
  sniffs for ``zarr.json`` inside the sample directory and reads a Zarr store
  from a ``.pmsh``-named path if it finds one. This reader does the same.

``physicsnemo.mesh.Mesh`` holds exactly ONE simplex kind, so the reduction
(select one topological dimension, linearize + simplexify, drop what the
target cannot hold, each drop warned) is the bridge's own
``_to_physicsnemo_payload`` -- imported lazily, so ``import meshioplusplus``
still never pulls in the ``physicsnemo`` subpackage.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from .._exceptions import ReadError, WriteError
from .._helpers import register_format

#: The tensorclass whose tree this is. Written into the root ``meta.json``
#: verbatim: ``Mesh.load`` resolves this string to a class and fails by name
#: on anything it does not recognise.
MESH_TYPE = "<class 'physicsnemo.mesh.mesh.Mesh'>"

#: The nested TensorDict's own ``_type`` string, on every group's meta.
TENSORDICT_TYPE = "<class 'tensordict._td.TensorDict'>"

#: The three data groups, in the order ``Mesh.save`` declares them.
DATA_GROUPS = ("point_data", "cell_data", "global_data")

#: numpy dtype <-> the ``str(torch.dtype)`` spelling the metas carry. One
#: table both directions, so the two can never disagree.
_TORCH_DTYPES = {
    "torch.float16": "<f2",
    "torch.float32": "<f4",
    "torch.float64": "<f8",
    "torch.int8": "|i1",
    "torch.int16": "<i2",
    "torch.int32": "<i4",
    "torch.int64": "<i8",
    "torch.uint8": "|u1",
    "torch.bool": "|b1",
}
_NUMPY_DTYPES = {np.dtype(v).str: k for k, v in _TORCH_DTYPES.items()}
# Native-endian aliases, so a big-endian host still resolves its own arrays.
for _torch_name, _code in _TORCH_DTYPES.items():
    _NUMPY_DTYPES.setdefault(np.dtype(_code).name, _torch_name)


def _torch_dtype_name(array):
    """The ``str(torch.dtype)`` spelling for ``array``'s dtype."""
    dtype = np.dtype(array.dtype)
    name = _NUMPY_DTYPES.get(dtype.str) or _NUMPY_DTYPES.get(dtype.name)
    if name is None:
        raise WriteError(
            f"meshio++: pmsh: dtype '{dtype}' has no torch counterpart "
            f"(supported: {', '.join(sorted(_TORCH_DTYPES))})"
        )
    return name


def _leaf_meta(array):
    """One leaf entry of a ``meta.json``, exactly as tensordict writes it."""
    return {
        "device": "cpu",
        "shape": [int(n) for n in array.shape],
        "dtype": _torch_dtype_name(array),
        "is_nested": False,
    }


def _write_blob(path, array):
    """One raw little-endian C-order blob, or nothing when it is empty.

    The emptiness rule is tensordict's, not ours: it does not persist a
    zero-element tensor, so writing one here would produce a tree that no
    physicsnemo-written store has.
    """
    if array.size == 0:
        return
    code = _TORCH_DTYPES[_torch_dtype_name(array)]
    np.ascontiguousarray(array, dtype=np.dtype(code)).tofile(path)


def _write_group(directory, arrays, batch_shape):
    """One data group: its own ``meta.json`` plus one blob per array."""
    os.makedirs(directory, exist_ok=True)
    meta = {name: _leaf_meta(array) for name, array in arrays.items()}
    meta["shape"] = [int(n) for n in batch_shape]
    meta["device"] = "cpu"
    meta["_type"] = TENSORDICT_TYPE
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    for name, array in arrays.items():
        _write_blob(os.path.join(directory, f"{name}.memmap"), array)


def _looks_like_a_store(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "meta.json"))


def write(filename, mesh, *, manifold_dim="auto", float32=True):
    """Write ``mesh`` as a ``.pmsh`` directory.

    ``manifold_dim`` selects the one simplex kind the target holds
    (``"auto"`` = the highest dimension present); non-simplex cells at that
    dimension are tessellated and everything the target cannot hold is
    dropped **with a warning**. ``float32`` follows the upstream convention
    for point coordinates; cells are always int64.
    """
    from .._interop import _emit
    from ..physicsnemo import _to_physicsnemo_payload

    payload = _to_physicsnemo_payload(mesh, manifold_dim=manifold_dim, float32=float32)
    _emit("pmsh", payload["notes"])

    root = str(filename)
    if os.path.exists(root):
        if not _looks_like_a_store(root):
            raise WriteError(
                f"meshio++: pmsh: '{root}' exists and is not a .pmsh store "
                "(no meta.json); refusing to remove it"
            )
        shutil.rmtree(root)
    inner = os.path.join(root, "_tensordict")
    os.makedirs(inner)

    with open(os.path.join(root, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"_type": MESH_TYPE}, fh)

    points = np.ascontiguousarray(payload["points"])
    cells = payload["cells"]
    if cells is None:
        # The empty-cells sentinel `Mesh.__post_init__` itself restores.
        cells = np.zeros((0, 1), dtype=np.int64)
    cells = np.ascontiguousarray(cells, dtype=np.int64)

    meta = {
        "points": _leaf_meta(points),
        "cells": _leaf_meta(cells),
    }
    for group in DATA_GROUPS:
        meta[group] = {"type": "TensorDict"}
    meta["shape"] = []
    meta["device"] = "cpu"
    meta["_type"] = TENSORDICT_TYPE
    with open(os.path.join(inner, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    _write_blob(os.path.join(inner, "points.memmap"), points)
    _write_blob(os.path.join(inner, "cells.memmap"), cells)

    batches = {
        "point_data": [len(points)],
        "cell_data": [len(cells)],
        "global_data": [],
    }
    for group in DATA_GROUPS:
        _write_group(
            os.path.join(inner, group),
            {name: np.asarray(a) for name, a in payload[group].items()},
            batches[group],
        )


def _read_leaf(directory, name, entry):
    """One leaf array, memory-mapped copy-on-write.

    ``mode="c"`` is what makes the returned arrays both writeable (meshio++'s
    standing contract) and lazily paged, which is the entire reason this
    format is faster to load than a VTU: nothing is read until it is touched.
    """
    shape = tuple(int(n) for n in entry["shape"])
    torch_name = entry.get("dtype")
    code = _TORCH_DTYPES.get(torch_name)
    if code is None:
        raise ReadError(
            f"meshio++: pmsh: unknown dtype '{torch_name}' for '{name}' "
            f"(supported: {', '.join(sorted(_TORCH_DTYPES))})"
        )
    dtype = np.dtype(code)
    path = os.path.join(directory, f"{name}.memmap")
    if not os.path.isfile(path):
        # Declared but absent is legal for an EMPTY array only: tensordict
        # does not persist a zero-element tensor. Anything else is a truncated
        # store and must fail rather than come back as zeros.
        count = 1
        for extent in shape:
            count *= extent
        if count:
            raise ReadError(
                f"meshio++: pmsh: '{path}' is missing but its meta declares "
                f"shape {list(shape)}"
            )
        return np.empty(shape, dtype=dtype)
    return np.memmap(path, dtype=dtype, mode="c", shape=shape)


def _read_group(directory):
    """One data group's arrays, or an empty dict when the group is absent."""
    meta_path = os.path.join(directory, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return {
        name: _read_leaf(directory, name, entry)
        for name, entry in meta.items()
        if isinstance(entry, dict) and "shape" in entry and "dtype" in entry
    }


def read(filename):
    """Read a ``.pmsh`` directory (or a Zarr store living under that name)."""
    root = str(filename)
    if not os.path.isdir(root):
        raise ReadError(
            f"meshio++: pmsh: '{root}' is not a directory; a .pmsh store is a "
            "directory of memory-mapped arrays, not a single file"
        )
    if os.path.isfile(os.path.join(root, "zarr.json")):
        # What MeshReader._load_sample does: the extension names the role, the
        # contents name the codec.
        from ..zarr._zarr import read as read_zarr

        return read_zarr(root)

    from ..physicsnemo import _from_physicsnemo_payload

    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        raise ReadError(f"meshio++: pmsh: '{root}' has no meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        declared = json.load(fh).get("_type")
    if declared != MESH_TYPE:
        raise ReadError(
            f"meshio++: pmsh: '{root}' declares _type {declared!r}; only "
            f"{MESH_TYPE!r} is supported (a .pdmsh DomainMesh is not)"
        )

    inner = os.path.join(root, "_tensordict")
    inner_meta = os.path.join(inner, "meta.json")
    if not os.path.isfile(inner_meta):
        raise ReadError(f"meshio++: pmsh: '{root}' has no _tensordict/meta.json")
    with open(inner_meta, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if "points" not in meta:
        raise ReadError(f"meshio++: pmsh: '{root}' declares no points array")

    points = _read_leaf(inner, "points", meta["points"])
    cells = None
    if "cells" in meta:
        cells = _read_leaf(inner, "cells", meta["cells"])
        if cells.size == 0:
            cells = None

    return _from_physicsnemo_payload(
        {
            "points": points,
            "cells": cells,
            "point_data": _read_group(os.path.join(inner, "point_data")),
            "cell_data": _read_group(os.path.join(inner, "cell_data")),
            "global_data": _read_group(os.path.join(inner, "global_data")),
        }
    )


register_format("pmsh", [".pmsh"], read, {"pmsh": write})
