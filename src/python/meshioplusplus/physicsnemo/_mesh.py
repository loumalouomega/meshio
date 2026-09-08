"""The ``physicsnemo.mesh.Mesh`` bridge's gated half.

The ONLY module in the package importing ``physicsnemo.mesh`` -- reached
exclusively through the lazy :func:`to_physicsnemo`/:func:`from_physicsnemo`
entry points, never at package import time: ``import physicsnemo`` pulls in
NVIDIA Warp unconditionally (~1.5 s, ~2700 modules), a cost the pure half
must never pay.

Deliberately the thinnest possible layer: everything decidable without torch
lives in the pure payload builders (``_to_physicsnemo_payload`` /
``_from_physicsnemo_payload``), and this module only wraps numpy in tensors
and touches the tensorclass constructor + public attributes -- the narrowest
surface of ``physicsnemo.mesh``: the tensorclass constructor and its public
attributes, never a private one. The ``.pmsh`` and Zarr on-disk layouts are
written and read by :mod:`meshioplusplus.pmsh` and :mod:`meshioplusplus.zarr`
instead -- in pure numpy, so a box with no torch can produce a training set --
with parity against upstream's own ``save``/``load`` pinned in both
directions. Pin ``nvidia-physicsnemo>=2.1,<2.3``.
"""

import numpy as np
import torch
from physicsnemo.mesh import Mesh


def to_mesh(payload):
    """The numpy payload (pure half's output) as a ``physicsnemo.mesh.Mesh``.

    Upstream conventions honoured: CPU tensors, cells ``int64``; data
    tensors keep their numpy dtypes (``io_pyvista`` does the same).
    """
    cells = payload["cells"]
    return Mesh(
        points=torch.from_numpy(payload["points"]),
        cells=None if cells is None else torch.from_numpy(cells),
        point_data={
            name: torch.from_numpy(array)
            for name, array in payload["point_data"].items()
        },
        cell_data={
            name: torch.from_numpy(array)
            for name, array in payload["cell_data"].items()
        },
        global_data={
            name: torch.from_numpy(np.ascontiguousarray(array))
            for name, array in payload["global_data"].items()
        },
    )


def from_mesh(pm):
    """A ``physicsnemo.mesh.Mesh``'s public attributes as a numpy payload."""

    def leaves(tensordict):
        return {
            str(key): value.detach().cpu().numpy()
            for key, value in tensordict.items(include_nested=True, leaves_only=True)
        }

    cells = pm.cells.detach().cpu().numpy()
    return {
        "points": pm.points.detach().cpu().numpy(),
        "cells": cells if cells.size else None,
        "point_data": leaves(pm.point_data),
        "cell_data": leaves(pm.cell_data),
        "global_data": leaves(pm.global_data),
    }
