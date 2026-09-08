"""Byte-level baseline pins for the VTU/VTP writers and the whole-file readers.

These tests exist to make regressions *loud* while the I/O layer is refactored
(selective reads, mmap, the codec abstraction). They deliberately avoid hashing
zlib-compressed output: deflate is not byte-stable across zlib versions, so a
committed hash of it would fail on an unrelated toolchain. Instead:

* ASCII and *uncompressed* binary output are hashed directly -- both are fully
  deterministic given the same mesh.
* The zlib path is pinned structurally: its decoded payload must equal the
  uncompressed variant's payload byte-for-byte, which catches any change to the
  framing, byte order, or array ordering without depending on deflate's output.
"""

from __future__ import annotations

import base64
import hashlib
import re
import struct
import zlib

import numpy as np
import pytest

import meshioplusplus

_core = pytest.importorskip("meshioplusplus._core")

# zlib is found via find_package like HDF5/netCDF (see CMakeLists.txt's
# MESHIOPLUSPLUS_WITH_ZLIB comment) and so can be compiled out too (e.g.
# Windows CI, which has no system zlib). These tests call `_core` directly,
# bypassing the write()/read() shims' Python fallback, so they need the guard.
requires_zlib = pytest.mark.skipif(
    not getattr(_core, "__has_zlib__", False), reason="build has no zlib"
)


def _baseline_mesh() -> meshioplusplus.Mesh:
    """A small mesh exercising 2 cell blocks, point_data and cell_data.

    Built locally rather than taken from ``tests/python/helpers`` because several
    pure-Python writers mutate the mesh they are handed, which would corrupt a
    shared module-level fixture for every later test in the session.
    """
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    cells = [
        ("triangle", np.array([[0, 1, 4], [1, 2, 4]])),
        ("quad", np.array([[0, 1, 2, 3]])),
    ]
    return meshioplusplus.Mesh(
        points,
        cells,
        point_data={
            "alpha": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            "beta": np.arange(15, dtype=np.float64).reshape(5, 3),
        },
        cell_data={
            "gamma": [np.array([10.0, 20.0]), np.array([30.0])],
            "tag": [np.array([1, 2], dtype=np.int32), np.array([3], dtype=np.int32)],
        },
    )


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Regenerate with: pytest tests/python/test_io_baseline.py -k report -s
#
# These hashes cover the whole file, and every writer here emits a provenance
# comment carrying the release version ("Written by meshio++ v<version>"), so
# **a version bump changes all four and they must be refreshed in the same
# change**. They were recorded at v10.19.0 and went stale for three releases
# before anyone noticed, which is what this note is for.
BASELINE_HASHES = {
    "vtu_ascii": "8a638070474b148f8807ca500f6346c52d70c22d13b84fca0683a0d9dd1abcbb",
    "vtu_binary_raw": "68b68d9c15eeb027d7125a183c0c90f8b935439bcfe38d7acec0706ed2861026",
    "vtp_ascii": "d2c993e0c46eb5a1838a25fb9daa5cfa935ccdb48668ac2d06bb556f2111ecc4",
    "vtp_binary_raw": "37c36fc7450e453128d90375e9f624b0e2008fde9bee572b22fb53a0b5337d63",
}


def _write(tmp_path, fmt, binary, zlib_on):
    """Write straight through ``_core``, bypassing ``meshioplusplus.write()``.

    That bypass is deliberate -- these tests pin the writer's own output, not
    the shim's -- but it means the provenance record's usual reset never runs:
    only the public ``write()`` calls ``_provenance.begin_write()``, and a
    "no scope" record otherwise carries whatever note the *previous* write
    (anywhere in the same test session, e.g. a PLY int64->int32 cast) left
    behind, per ``detail/provenance.hpp``'s documented caller-manages-state
    contract for direct writer calls. Reset it ourselves so this file's
    byte-pinning is independent of what ran before it.
    """
    from meshioplusplus import _provenance

    _provenance.begin_write()
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"baseline.{fmt}"
    getattr(_core, f"{fmt}_write")(str(path), _baseline_mesh(), binary, zlib_on)
    return path


@pytest.mark.parametrize(
    "fmt,binary,key",
    [
        ("vtu", False, "vtu_ascii"),
        ("vtu", True, "vtu_binary_raw"),
        ("vtp", False, "vtp_ascii"),
        ("vtp", True, "vtp_binary_raw"),
    ],
)
def test_writer_bytes_are_pinned(tmp_path, fmt, binary, key):
    """Deterministic (non-deflated) writer output must not drift."""
    expected = BASELINE_HASHES[key]
    if expected is None:
        pytest.skip(f"baseline hash for {key} not recorded yet")
    assert _sha(_write(tmp_path, fmt, binary, False)) == expected


def _decoded_payloads(text: str) -> list[bytes]:
    """Decode every binary <DataArray> body in a VTU/VTP document.

    Handles both framings: uncompressed (a UInt32 byte-count header followed by
    the raw payload) and zlib's block scheme (num_blocks / max_block /
    last_block / per-block compressed sizes, then the deflated blocks).

    Note the compressed body is *two* independently '='-padded base64 strings
    concatenated -- the header and the block data are encoded separately (see
    ``vtu_encode_binary``), so it must be decoded in two pieces by character
    count, exactly as ``vtu_decode_zlib`` does. Decoding it as one stream
    silently corrupts everything after the header's padding.
    """
    compressed = 'compressor="vtkZLibDataCompressor"' in text
    out = []
    for body in re.findall(
        r'format="binary"[^>]*>\s*([A-Za-z0-9+/=\s]+?)\s*</DataArray>', text
    ):
        b64 = "".join(body.split())
        if not compressed:
            raw = base64.b64decode(b64)
            (total,) = struct.unpack("<I", raw[:4])
            out.append(raw[4 : 4 + total])
            continue

        # hsz == 4: decode just enough to learn num_blocks, then the full header.
        (nblocks,) = struct.unpack("<I", base64.b64decode(b64[:8])[:4])
        header_chars = ((4 * (3 + nblocks) + 2) // 3) * 4
        header = base64.b64decode(b64[:header_chars])
        _, max_block, last_block = struct.unpack("<III", header[:12])
        sizes = struct.unpack(f"<{nblocks}I", header[12 : 12 + 4 * nblocks])

        blob = base64.b64decode(b64[header_chars:])
        payload, off = b"", 0
        for i, size in enumerate(sizes):
            expected = last_block if i + 1 == nblocks else max_block
            payload += zlib.decompress(blob[off : off + size], bufsize=expected)
            off += size
        out.append(payload)
    return out


@requires_zlib
@pytest.mark.parametrize("fmt", ["vtu", "vtp"])
def test_zlib_payload_matches_uncompressed(tmp_path, fmt):
    """Compression must change only the framing, never the payload bytes."""
    raw = _write(tmp_path / "raw", fmt, True, False)
    deflated = _write(tmp_path / "zl", fmt, True, True)

    raw_text = raw.read_text()
    deflated_text = deflated.read_text()
    assert 'compressor="vtkZLibDataCompressor"' not in raw_text
    assert 'compressor="vtkZLibDataCompressor"' in deflated_text

    payloads = _decoded_payloads(raw_text)
    assert payloads, "no binary DataArray bodies found -- the regex needs updating"
    assert payloads == _decoded_payloads(deflated_text)


@pytest.mark.parametrize("fmt", ["vtu", "vtp"])
@pytest.mark.parametrize(
    "binary,zlib_on",
    [(False, False), (True, False), pytest.param(True, True, marks=requires_zlib)],
)
def test_roundtrip_is_lossless(tmp_path, fmt, binary, zlib_on):
    src = _baseline_mesh()
    path = _write(tmp_path, fmt, binary, zlib_on)
    got = getattr(_core, f"{fmt}_read")(str(path))

    assert np.allclose(np.asarray(got.points)[:, :3], src.points)
    assert set(got.point_data) == set(src.point_data)
    for name, arr in src.point_data.items():
        assert np.allclose(np.asarray(got.point_data[name]), arr)


def test_report(tmp_path):
    """Not an assertion -- prints hashes so BASELINE_HASHES can be filled in."""
    for fmt, binary, key in [
        ("vtu", False, "vtu_ascii"),
        ("vtu", True, "vtu_binary_raw"),
        ("vtp", False, "vtp_ascii"),
        ("vtp", True, "vtp_binary_raw"),
    ]:
        print(f'    "{key}": "{_sha(_write(tmp_path / key, fmt, binary, False))}",')
