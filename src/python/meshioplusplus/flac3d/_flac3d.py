"""
I/O for FLAC3D format.
"""

from __future__ import annotations

import re
import struct

import numpy as np

from .. import _provenance
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._mesh import Mesh

# The slot a group is written into when its name does not name one. Every
# real FLAC3D file uses "Default"; the reference writer used to emit
# `SLOT 1`, which no reader here or elsewhere treats as special.
DEFAULT_SLOT = "Default"

meshio_only = {
    "zone": {
        "tetra": "tetra",
        "tetra10": "tetra",
        "pyramid": "pyramid",
        "pyramid13": "pyramid",
        "wedge": "wedge",
        "wedge12": "wedge",
        "wedge15": "wedge",
        "wedge18": "wedge",
        "hexahedron": "hexahedron",
        "hexahedron20": "hexahedron",
        "hexahedron24": "hexahedron",
        "hexahedron27": "hexahedron",
    },
    "face": {
        "triangle": "triangle",
        "triangle6": "triangle",
        "triangle7": "triangle",
        "quad": "quad",
        "quad8": "quad",
        "quad9": "quad",
    },
}


numnodes_to_meshio_type = {
    "zone": {4: "tetra", 5: "pyramid", 6: "wedge", 8: "hexahedron"},
    "face": {3: "triangle", 4: "quad"},
}


meshio_to_flac3d_type = {
    "triangle": "T3",
    "quad": "Q4",
    "tetra": "T4",
    "pyramid": "P5",
    "wedge": "W6",
    "hexahedron": "B8",
}


flac3d_to_meshio_order = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 4, 2, 3],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 4, 2, 3, 6, 7, 5],
}


meshio_to_flac3d_order = {
    "triangle": [0, 1, 2],
    "quad": [0, 1, 2, 3],
    "tetra": [0, 1, 2, 3],
    "pyramid": [0, 1, 3, 4, 2],
    "wedge": [0, 1, 3, 2, 4, 5],
    "hexahedron": [0, 1, 3, 4, 2, 7, 5, 6],
}


meshio_to_flac3d_order_2 = {
    "tetra": [0, 2, 1, 3],
    "pyramid": [0, 3, 1, 4, 2],
    "wedge": [0, 2, 3, 1, 5, 4],
    "hexahedron": [0, 3, 1, 4, 2, 5, 7, 6],
}


def _merge(a: dict, b: dict) -> dict:
    return {**a, **b}


def read(filename):
    """Read FLAC3D f3grid grid file."""
    # Read a small block of the file to assess its type
    # See <https://code.activestate.com/recipes/173220/>
    with open_file(filename, "rb") as f:
        block = f.read(8)
        binary = b"\x00" in block

    mode = "rb" if binary else "r"
    with open_file(filename, mode) as f:
        out = read_buffer(f, binary)

    return out


def _resolve_group_ids(sets, cell_ids, offset):
    """FLAC3D group member ids -> global block-major cell indices.

    ``cell_ids`` are the file's own ids for one category (zones or faces), in
    the order the cells were read; ``offset`` is where that category starts in
    the concatenated ``f_cells + z_cells`` cell list. An id the file never
    defined resolves to ``-1`` and stays ``-1`` -- adding the offset to it
    would land it on a real cell of the other category.
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    inv = np.full(int(cell_ids.max()) + 1 if len(cell_ids) else 0, -1, dtype=np.int64)
    if len(cell_ids):
        inv[cell_ids] = np.arange(len(cell_ids), dtype=np.int64)

    out = {}
    for key, value in sets.items():
        value = np.asarray(value, dtype=np.int64).reshape(-1)
        idx = np.full(len(value), -1, dtype=np.int64)
        known = (value >= 0) & (value < len(inv))
        idx[known] = inv[value[known]]
        idx[idx >= 0] += offset
        out[key] = idx
    return out


def read_buffer(f, binary):
    """Read binary or ASCII file."""
    points = []
    point_ids = {}
    f_cells = []
    z_cells = []
    f_cell_sets = {}
    z_cell_sets = {}
    f_cell_ids = []
    z_cell_ids = []

    pidx = 0
    if binary:
        # Not sure what the first bytes represent, the format might be wrong
        # It does not seem to be useful anyway
        _ = struct.unpack("<2I", f.read(8))

        (num_nodes,) = struct.unpack("<I", f.read(4))
        for pidx in range(num_nodes):
            pid, point = _read_point_binary(f)
            points.append(point)
            point_ids[pid] = pidx

        for flag in ["zone", "face"]:
            if flag == "zone":
                cell_ids = z_cell_ids
                cells = z_cells
                cell_sets = z_cell_sets
            else:
                cell_ids = f_cell_ids
                cells = f_cells
                cell_sets = f_cell_sets

            (num_cells,) = struct.unpack("<I", f.read(4))
            for _ in range(num_cells):
                cell_id, cell = _read_cell_binary(f, point_ids)
                cell_ids.append(cell_id)
                _update_cells(cells, cell, flag)
                # mapper[flag][cid] = [cidx]
                # cidx += 1

            (num_groups,) = struct.unpack("<I", f.read(4))
            for _ in range(num_groups):
                name, slot, data = _read_cell_group_binary(f)
                cell_sets[f"{flag}:{name}:{slot}"] = np.array(data)
    else:
        while True:
            line = f.readline()

            if not line:
                break

            if line.strip() == "":
                continue

            split = line.rstrip().split()

            if split[0] == "G":
                pid, point = _read_point_ascii(split)
                points.append(point)
                point_ids[pid] = pidx
                pidx += 1

            elif split[0] == "Z":
                cell_id, cell = _read_cell_ascii(split, point_ids)
                z_cell_ids.append(cell_id)
                _update_cells(z_cells, cell, "zone")

            elif split[0] == "F":
                cell_id, cell = _read_cell_ascii(split, point_ids)
                f_cell_ids.append(cell_id)
                _update_cells(f_cells, cell, "face")

            elif split[0] == "ZGROUP":
                # ZGROUP "Region 2" SLOT 1
                name, slot, data = _read_cell_group_ascii(f, line)
                # Watch out! data refers to the global cell_ids, so we need to
                # adapt this later.
                z_cell_sets[f"zone:{name}:{slot}"] = np.asarray(data)

            elif split[0] == "FGROUP":
                name, slot, data = _read_cell_group_ascii(f, line)
                # Watch out! data refers to the global cell_ids, so we need to
                # adapt this later.
                f_cell_sets[f"face:{name}:{slot}"] = np.asarray(data)

    cells = f_cells + z_cells

    # enforce int type, empty numpy arrays have type float64
    f_cell_ids = np.asarray(f_cell_ids, dtype=int)
    z_cell_ids = np.asarray(z_cell_ids, dtype=int)
    z_offset = len(f_cell_ids)

    cell_ids = np.concatenate([f_cell_ids, z_cell_ids + z_offset], dtype=np.int64)

    cell_blocks = [
        (key, np.array(indices)[:, flac3d_to_meshio_order[key]])
        for key, indices in cells
    ]

    # sanity check, but not really necessary
    # _, counts = np.unique(z_ cell_ids, return_counts=True)
    # assert np.all(counts == 1), "Zone cell IDs not unique"
    # _, counts = np.unique(f_ cell_ids, return_counts=True)
    # assert np.all(counts == 1), "Zone cell IDs not unique"

    # FLAC3D contains global cell ids. Create an inverse array that maps the
    # global IDs to the running index (0, 1,..., n) that's used in meshio.
    f_cell_sets = _resolve_group_ids(f_cell_sets, f_cell_ids, 0)
    z_cell_sets = _resolve_group_ids(z_cell_sets, z_cell_ids, z_offset)

    cell_sets = _merge(f_cell_sets, z_cell_sets)

    # `cell_sets` now holds indices into the *global* cell list, but meshio++'s
    # `cell_sets` is a write-through view over `mesh.regions` and takes
    # per-block **local** indices -- `_regions.blocks_to_global` adds the block
    # base itself. So split *and rebase*: leaving the global values in place
    # made the base be added twice and dropped every member past its own
    # block's length, emptying whole groups (issue #76).
    bases = np.cumsum([0] + [len(cb[1]) for cb in cell_blocks])
    for key, data in cell_sets.items():
        if np.any(data < 0):
            warn(
                f'FLAC3D: group "{key}" names cells the file does not '
                "define; dropping them."
            )
            data = data[data >= 0]
        d = np.digitize(data, bases[1:])
        cell_sets[key] = [data[d == k] - bases[k] for k in range(len(cell_blocks))]

    # assert len(cell_ids) == sum(len(block) for _, block in cell_blocks)

    # also store the cell_ids
    cell_data = {}
    if len(cell_blocks) > 0:
        cell_data = {
            "cell_ids": np.split(
                cell_ids, np.cumsum([len(block) for _, block in cell_blocks][:-1])
            )
        }

    return Mesh(
        points=np.array(points),
        cells=cell_blocks,
        cell_data=cell_data,
        cell_sets=cell_sets,
    )


def _read_point_ascii(buf_or_line):
    """Read point coordinates."""
    pid = int(buf_or_line[1])
    point = [float(l) for l in buf_or_line[2:]]
    return pid, point


def _read_point_binary(buf_or_line):
    """Read point coordinates."""
    pid, x, y, z = struct.unpack("<I3d", buf_or_line.read(28))
    return pid, [x, y, z]


def _read_cell_ascii(buf_or_line, point_ids):
    """Read cell connectivity."""
    cid = int(buf_or_line[2])
    cell = buf_or_line[3:]
    is_b7 = buf_or_line[1] == "B7"
    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])
    return cid, cell


def _read_cell_binary(buf_or_line, point_ids):
    """Read cell connectivity."""
    cid, num_verts = struct.unpack("<2I", buf_or_line.read(8))
    cell = struct.unpack(f"<{num_verts}I", buf_or_line.read(4 * num_verts))
    is_b7 = num_verts == 7
    cell = [point_ids[int(l)] for l in cell]
    if is_b7:
        cell.append(cell[-1])
    return cid, cell


def _strip_quotes(text: str) -> str:
    """Remove one matching pair of surrounding single or double quotes."""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def _read_cell_group_binary(buf_or_line):
    # Group name
    (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
    (name,) = struct.unpack(f"<{num_chars}s", buf_or_line.read(num_chars))
    name = name.decode()

    # Slot name
    (num_chars,) = struct.unpack("<H", buf_or_line.read(2))
    (slot,) = struct.unpack(f"<{num_chars}s", buf_or_line.read(num_chars))
    slot = slot.decode()

    # Zones
    (num_zones,) = struct.unpack("<I", buf_or_line.read(4))
    data = struct.unpack(f"<{num_zones}I", buf_or_line.read(4 * num_zones))
    return name, slot, data


def _read_cell_group_ascii(buf_or_line, line: str):
    # a group line read
    # ```
    # ZGROUP 'group five' SLOT 5
    # ```
    m = re.match(r"^([A-Z]+) *[\'\"](.*?)[\'\"] *([A-Z]+) *(.*?) *$", line)
    if m is None:
        raise ReadError(
            'Expected line of the form\n```\nZGROUP "group name" SLOT 5\n```\n '
            + f"but got \n```\n{line}\n```\n"
        )
    assert m.group(1) in {"ZGROUP", "FGROUP"}
    assert m.group(3) == "SLOT"
    name = m.group(2)
    # The slot is the raw remainder of the line and may or may not be quoted
    # (`SLOT "Default"` and `SLOT 5` are both real). Strip a matching pair so
    # the ascii and binary readers -- the latter reads a length-prefixed
    # string, never quoted -- agree on the group key for the same mesh.
    slot = _strip_quotes(m.group(4))

    i = buf_or_line.tell()
    line = buf_or_line.readline()
    data = []
    while True:
        line = line.rstrip().split()
        if line and (line[0] not in {"*", "ZGROUP", "FGROUP"}):
            data += [int(l) for l in line]
        else:
            buf_or_line.seek(i)
            break
        i = buf_or_line.tell()
        line = buf_or_line.readline()

    return name, slot, data


def _update_cells(cells, cell, flag):
    """Update cell list."""
    cell_type = numnodes_to_meshio_type[flag][len(cell)]
    if len(cells) > 0 and cell_type == cells[-1][0]:
        cells[-1][1].append(cell)
    else:
        cells.append((cell_type, [cell]))


def _split_group_key(key):
    """``{zone|face}:{name}:{slot}`` -> ``(flag, name, slot)``.

    The reader builds that composite key because a FLAC3D group is identified
    by all three: ZGROUP and FGROUP are separate namespaces and a slot
    partitions the groups within one. Decomposing it again on write is what
    makes a file read from disk a fixed point -- otherwise the flag and slot
    are re-prefixed on every round trip. A name in any other shape belongs to
    neither category in particular and is placed by its members.
    """
    m = re.match(r"^(zone|face):(.*):([^:]*)$", key)
    if m is None:
        return None, key, "Default"
    return m.group(1), m.group(2), m.group(3)


def split_f_z(mesh):
    # FLAC3D makes a difference between ZONES (3D-cells only) and FACES
    # (2D-cells only). Split cells into zcells and fcells, along with the cell
    # sets etc.
    zblocks = []
    fblocks = []
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in meshio_only["zone"]:
            zblocks.append(i)
        elif cell_block.type in meshio_only["face"]:
            fblocks.append(i)
    zcells = [mesh.cells[i] for i in zblocks]
    fcells = [mesh.cells[i] for i in fblocks]

    def gather(blocks, cset):
        """Per-block local indices -> 1-based ids in this category's own space.

        ZONES and FACES are numbered independently in a FLAC3D file (both
        starting at 1), so each category's running counter walks only its own
        blocks. Zipping one category's block sizes against the whole cell list
        is what used to misalign them.
        """
        out = []
        gid = 0
        for i in blocks:
            idx = np.asarray(cset[i], dtype=np.int64).reshape(-1)
            out.append(idx + (gid + 1))
            gid += len(mesh.cells[i])
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    zsets = {}
    fsets = {}
    for key, cset in mesh.cell_sets.items():
        flag, _, _ = _split_group_key(key)
        # An empty group whose name already names this category is re-emitted
        # empty: the name is information (`detail/region_remap.hpp`'s rule),
        # and it is what keeps a file read from disk a byte-level fixed point.
        if flag != "face":
            values = gather(zblocks, cset)
            if len(values) or flag == "zone":
                zsets[key] = values
        if flag != "zone":
            values = gather(fblocks, cset)
            if len(values) or flag == "face":
                fsets[key] = values

    return zcells, fcells, zsets, fsets


def write(filename, mesh: Mesh, float_fmt: str = ".16e", binary: bool = False):
    """Write FLAC3D f3grid grid file."""
    skip = [
        c.type
        for c in mesh.cells
        if c.type not in meshio_only["zone"] and c.type not in meshio_only["face"]
    ]
    if skip:
        warn(
            "FLAC3D only stores 3D zones and 2D faces. " f'Skipping {", ".join(skip)}.'
        )

    # split into face/zone data
    zcells, fcells, zsets, fsets = split_f_z(mesh)

    mode = "wb" if binary else "w"
    with open_file(filename, mode) as f:
        if binary:
            # Don't know what these values represent
            f.write(struct.pack("<2I", 1375135718, 3))
        else:
            f.write(_provenance.render_lines(_provenance.SlotTier.BLOCK, "* "))

        _write_points(f, mesh.points, binary, float_fmt)
        # Make gid an array such that its value can be persitently altered
        # inside the functions. ZONES and FACES are numbered independently in
        # a FLAC3D file -- both start at 1 -- so each section gets its own
        # counter, matching what the reader (and every real file) expects.
        cells = _translate_zcells(mesh.points, zcells)
        _write_cells(f, cells, "zone", binary, np.array(0))
        _write_groups(f, zsets, "zone", binary)
        #
        cells = _translate_fcells(fcells)
        _write_cells(f, cells, "face", binary, np.array(0))
        _write_groups(f, fsets, "face", binary)


def _write_points(f, points, binary, float_fmt=None):
    """Write points coordinates."""
    if binary:
        f.write(struct.pack("<I", len(points)))
        for i, point in enumerate(points):
            f.write(struct.pack("<I3d", i + 1, *point))
    else:
        f.write("* GRIDPOINTS\n")
        for i, point in enumerate(points):
            fmt = "G\t{:8}\t" + "\t".join(3 * ["{:" + float_fmt + "}"]) + "\n"
            f.write(fmt.format(i + 1, *point))


def _write_cells(f, cells, flag: str, binary: bool, gid):
    """Write cells."""
    if binary:
        f.write(
            struct.pack(
                "<I", sum(len(c[1]) for c in cells if c[0] in meshio_only[flag])
            )
        )
        for _, cdata in cells:
            num_cells, num_verts = cdata.shape
            tmp = np.column_stack(
                (
                    np.arange(1, num_cells + 1) + gid,
                    np.full(num_cells, num_verts),
                    cdata + 1,
                )
            ).astype(int)
            f.write(struct.pack(f"<{(num_verts + 2) * num_cells}I", *tmp.ravel()))
            gid += num_cells
    else:
        entity = "ZONES" if flag == "zone" else "FACES"
        abbrev = entity[0]

        f.write(f"* {entity}\n")
        for ctype, cdata in cells:
            fmt = f"{abbrev} {{}} {{}} " + " ".join(["{}"] * cdata.shape[1]) + "\n"
            for entry in cdata + 1:
                gid += 1
                f.write(fmt.format(meshio_to_flac3d_type[ctype], gid, *entry))


def _decompose_group_name(label, flag):
    """``"zone:Brick1:Default"`` -> ``("Brick1", "Default")`` for flag ``zone``.

    The exact inverse of the reader's ``f"{flag}:{name}:{slot}"``, which is
    what makes a file read from disk a fixed point -- without it every round
    trip re-prefixes the flag and re-appends the slot. The split is on the
    *last* colon because a FLAC3D group name may itself contain one while a
    slot may not. A name in any other shape keeps its whole self and takes the
    default slot, so a region carried in from another format (``solid``) is
    written as ``ZGROUP "solid" SLOT "Default"``.
    """
    prefix = f"{flag}:"
    rest = label[len(prefix) :] if label.startswith(prefix) else label
    name, sep, slot = rest.rpartition(":")
    return (name, slot) if sep else (rest, DEFAULT_SLOT)


def _write_groups(f, materials, flag, binary) -> None:
    """Write groups."""
    materials = materials or {}

    if binary:
        f.write(struct.pack("<I", len(materials)))
        for label, group in materials.items():
            name, slot = _decompose_group_name(label, flag)
            # Encode first: the length prefixes count *bytes*, not characters.
            nb, sb = name.encode(), slot.encode()
            fmt = f"<H{len(nb)}sH{len(sb)}sI{len(group)}I"
            f.write(struct.pack(fmt, len(nb), nb, len(sb), sb, len(group), *group))
    else:
        flg = "ZGROUP" if flag == "zone" else "FGROUP"

        f.write(f"* {flag.upper()} GROUPS\n")
        for label, group in materials.items():
            name, slot = _decompose_group_name(label, flag)
            f.write(f'{flg} "{name}" SLOT "{slot}"\n')
            _write_table(f, group)


def _translate_zcells(points, cells):
    """Reorder meshio cells to FLAC3D zones.

    Four first points must form a right-handed coordinate system (outward
    normal vectors). Reorder corner points according to sign of scalar triple
    products.
    """

    # See <https://stackoverflow.com/a/42386330/353337>
    def slicing_summing(a, b, c):
        c0 = b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        c1 = b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2]
        c2 = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
        return a[:, 0] * c0 + a[:, 1] * c1 + a[:, 2] * c2

    zones = []
    for cell_block in cells:
        assert cell_block.type in meshio_only["zone"]

        # Compute scalar triple products
        key = meshio_only["zone"][cell_block.type]
        tmp = points[cell_block.data[:, meshio_to_flac3d_order[key][:4]].T]
        det = slicing_summing(tmp[1] - tmp[0], tmp[2] - tmp[0], tmp[3] - tmp[0])

        # Reorder corner points
        data = np.where(
            (det > 0)[:, None],
            cell_block.data[:, meshio_to_flac3d_order[key]],
            cell_block.data[:, meshio_to_flac3d_order_2[key]],
        )
        zones.append((key, data))

    return zones


def _translate_fcells(cells):
    """Reorder meshio cells to FLAC3D faces."""
    faces = []
    for cell_block in cells:
        assert cell_block.type in meshio_only["face"]

        key = meshio_only["face"][cell_block.type]
        data = cell_block.data[:, meshio_to_flac3d_order[key]]
        faces.append((key, data))

    return faces


def _write_table(f, data, ncol: int = 20):
    """Write group data table."""
    nrow = len(data) // ncol
    lines = np.split(data, np.full(nrow, ncol).cumsum())
    for line in lines:
        if len(line):
            f.write(" {}\n".format(" ".join([str(l) for l in line])))
