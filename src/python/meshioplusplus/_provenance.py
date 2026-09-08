"""The provenance record every writer renders into its own header slot.

Two layers. ``TAG`` (v10.15.0) is the unconditional one-line credit: always
emitted, by both engines, character-identically. Nothing about it changed
here.

The ``Record``/``scope`` surface (v10.16.0) is the **opt-in** richer block --
source, target, the operation chain and the conversion assumptions accepted
on the way. ``Mode.OFF`` is the default, so a caller who asks for nothing
gets byte-for-byte what v10.15.0 wrote.

This is the Python twin of ``detail/provenance.hpp`` -- the C++ header's own
doc comment explains *why* a scoped context rather than growing an existing
options struct (``WriteOptions``/``write`` kwargs bypass most write paths
entirely) and *why* thread-local rather than process-global (concurrent
writes must not cross-contaminate). Both apply here unchanged; Python's
``threading.local`` is the direct analogue of C++'s ``thread_local``.

Deliberately **not** carrying which engine rendered the file: the roadmap
asks for that diagnostic, but it contradicts the harder guarantee that the
C++ core and the Python fallback emit character-identical bytes -- an engine
marker is exactly the ``(C++ core)``-vs-``v{version}`` drift v10.15.0
removed. It is reported through :func:`current_record` instead, where it
costs nothing (see :func:`meshioplusplus._core.<fmt>_write`'s callers, which
set it before delegating to the C++ engine).
"""

from __future__ import annotations

import enum
import json
import os
import threading
import time
from dataclasses import dataclass, field

from .__about__ import __version__

#: The canonical one-line provenance tag every writer emits, wrapped in each
#: format's own comment syntax at each writer's existing header position.
TAG = f"Written by meshio++ v{__version__}"


class Mode(enum.Enum):
    """How much provenance a writer should render."""

    #: Only :data:`TAG` -- exactly what v10.15.0 wrote. The default.
    OFF = "off"
    #: The full block where the slot allows it, degrading silently otherwise.
    BEST_EFFORT = "best_effort"
    #: The full block, or a ``WriteError`` naming the format whose slot
    #: cannot hold one.
    REQUIRED = "required"


class SlotTier(enum.Enum):
    """What a format's header slot can physically hold.

    Mirrors ``detail::SlotTier`` in the C++ header exactly -- see its doc
    comment for what each tier means and why ``Required`` only ever throws
    for :attr:`NONE`.
    """

    NONE = "none"
    BOUNDED = "bounded"
    SINGLE_LINE = "single_line"
    BLOCK = "block"


@dataclass
class Note:
    """One conversion assumption accepted on the way to the output file."""

    category: str
    detail: str


@dataclass
class Record:
    """Everything the record can carry (the twin of ``ProvenanceRecord``)."""

    source_path: str = ""
    source_format: str = ""
    target_format: str = ""
    encoding: str = ""
    codec: str = ""
    float_format: str = ""
    operations: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    timestamp: str = ""


_state = threading.local()

#: ``Mode`` values keyed by the integer the C API/pybind bridge uses --
#: ``detail::ProvenanceMode``'s underlying values (Off=0, BestEffort=1,
#: Required=2).
_MODE_BY_INT = {0: Mode.OFF, 1: Mode.BEST_EFFORT, 2: Mode.REQUIRED}
_INT_BY_MODE = {v: k for k, v in _MODE_BY_INT.items()}

#: ``SlotTier`` values keyed the same way (``detail::SlotTier``'s values).
_TIER_BY_INT = {
    0: SlotTier.NONE,
    1: SlotTier.BOUNDED,
    2: SlotTier.SINGLE_LINE,
    3: SlotTier.BLOCK,
}
_INT_BY_TIER = {v: k for k, v in _TIER_BY_INT.items()}


def _stack():
    if not hasattr(_state, "stack"):
        _state.stack = []
    return _state.stack


def _ambient():
    """The record notes land in when no scope is open.

    Provenance is on by default, so a writer-side :func:`note` fires during an
    ordinary ``write()`` with nothing scoped. Per-thread, and cleared by
    :func:`lines` once it has rendered, so one file's assumptions can never
    surface in the next one's header. The C++ twin (``g_ambient_record``) does
    exactly this; when the extension is loaded it is the one in charge and
    this is only the pure-Python fallback's copy.
    """
    if not hasattr(_state, "ambient"):
        _state.ambient = Record()
    return _state.ambient


def _active_record():
    """The scope's record when one is open, else the ambient one."""
    stack = _stack()
    return stack[-1][1] if stack else _ambient()


def default_mode() -> Mode:
    """The mode used when no scope is open -- :data:`Mode.BEST_EFFORT` unless
    turned off.

    Provenance is **on by default**: with nothing scoped a writer still
    renders any conversion assumptions recorded while it ran. The
    source/target/operation-chain fields need a caller or driver to set them
    and so stay absent, and no timestamp is added, which is what keeps default
    output deterministic.

    Overridden by ``MESHIOPLUSPLUS_PROVENANCE`` (``off``/``none``/``0``, or
    ``required``), read once, and by :func:`set_default_mode`.
    """
    core = _core_module()
    if core is not None:
        return _MODE_BY_INT[core.provenance_default_mode()]
    if not hasattr(_state, "default_mode"):
        env = os.environ.get("MESHIOPLUSPLUS_PROVENANCE")
        if env in ("off", "none", "0"):
            _state.default_mode = Mode.OFF
        elif env == "required":
            _state.default_mode = Mode.REQUIRED
        else:
            _state.default_mode = Mode.BEST_EFFORT
    return _state.default_mode


def set_default_mode(mode: Mode) -> None:
    """Sets what :func:`default_mode` returns, overriding the environment."""
    core = _core_module()
    if core is not None:
        core.provenance_set_default_mode(_INT_BY_MODE[mode])
    _state.default_mode = mode


def begin_write() -> None:
    """Marks the start of a write, bounding scope-less notes to it.

    With provenance on by default a writer-side :func:`note` fires during an
    ordinary ``write()`` with nothing scoped, and those notes need a lifetime
    or they attach to whatever file is written *next* -- including an
    unrelated one. This gives them one: :func:`meshioplusplus.write` calls it,
    and it resets the ambient record so only notes raised by *this* write can
    be rendered.

    **A no-op while a scope is open** -- the caller's scope owns the lifetime
    then, and spanning more than one write is the whole point of opening one.

    The deliberate trade: a note raised *before* the write begins (the
    operation-side sites) is discarded rather than misattributed. Capturing
    those needs an explicit :class:`scope` spanning the operations and the
    write.
    """
    core = _core_module()
    if core is not None:
        core.provenance_begin_write()
    if not _stack():
        _state.ambient = Record()


def _core_module():
    """The compiled extension, or ``None`` when it is not built.

    Mirrors the rest of this package's own convention (``_helpers.py``'s
    ``try: from . import _core / except Exception: pass``): a missing
    extension is expected on a pure-Python-fallback install, never an error
    here. Cached after the first attempt, which is safe -- whether the
    extension is importable does not change during a process's lifetime.
    """
    try:
        return _core_module._mod
    except AttributeError:
        try:
            from . import _core as mod
        except Exception:
            mod = None
        _core_module._mod = mod
        return mod


def current_mode() -> Mode:
    """The active mode for this thread (``Mode.OFF`` when no scope is open).

    Reads through to the C++ engine's own thread-local state when the
    compiled extension is available -- see the module docstring's note on
    the bridge, and :class:`scope`, which is what keeps the two in sync.
    """
    core = _core_module()
    if core is not None:
        return _MODE_BY_INT[core.provenance_current_mode()]
    stack = _stack()
    return stack[-1][0] if stack else Mode.OFF


def current_record() -> Record:
    """The active record for this thread; empty when no scope is open."""
    core = _core_module()
    if core is not None:
        d = core.provenance_current_record()
        return Record(
            source_path=d["source_path"],
            source_format=d["source_format"],
            target_format=d["target_format"],
            encoding=d["encoding"],
            codec=d["codec"],
            float_format=d["float_format"],
            operations=list(d["operations"]),
            notes=[Note(c, det) for c, det in d["notes"]],
            timestamp=d["timestamp"],
        )
    stack = _stack()
    return stack[-1][1] if stack else Record()


def note(category: str, detail: str) -> None:
    """Records one conversion assumption against the active scope.

    A **no-op outside a scope**, which is what lets it be called
    unconditionally from a writer without knowing whether anyone is
    listening. Duplicate ``(category, detail)`` pairs are collapsed: a
    per-cell warning must not produce a per-cell record.

    When the compiled extension is loaded this call is mirrored into its own
    thread-local record too (see :class:`scope`), so a note made from Python
    is not lost when a subsequent ``_core.<fmt>_write`` call renders the
    file.
    """
    core = _core_module()
    if core is not None:
        core.provenance_note(category, detail)
    rec = _active_record()
    for n in rec.notes:
        if n.category == category and n.detail == detail:
            return
    rec.notes.append(Note(category, detail))


def set_source(path: str, fmt: str) -> None:
    """Sets the source path/format on the active record. No-op outside a scope."""
    core = _core_module()
    if core is not None:
        core.provenance_set_source(path, fmt)
    rec = _active_record()
    rec.source_path = path
    rec.source_format = fmt


def set_target(
    fmt: str, encoding: str = "", codec: str = "", float_format: str = ""
) -> None:
    """Sets the target fields on the active record. No-op outside a scope."""
    core = _core_module()
    if core is not None:
        core.provenance_set_target(fmt, encoding, codec, float_format)
    rec = _active_record()
    rec.target_format = fmt
    rec.encoding = encoding
    rec.codec = codec
    rec.float_format = float_format


def add_operation(rendered: str) -> None:
    """Appends one step to the active record's operation chain. No-op outside
    a scope."""
    core = _core_module()
    if core is not None:
        core.provenance_add_operation(rendered)
    _active_record().operations.append(rendered)


def timestamp() -> str:
    """The timestamp a new scope is stamped with by default.

    Honours ``SOURCE_DATE_EPOCH`` (the reproducible-builds convention) so a
    byte-comparable build can pin it; returns empty when
    ``MESHIOPLUSPLUS_PROVENANCE_TIMESTAMP=off``. Host and user are never
    recorded -- these files get shared. Character-identical in spirit (not
    necessarily in value, since the two engines are not stamped at the same
    instant) to ``detail::provenance_timestamp()``; both honour the same two
    environment variables the same way, including ``SOURCE_DATE_EPOCH``
    outranking the blanket ``off`` switch.
    """
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if os.environ.get("MESHIOPLUSPLUS_PROVENANCE_TIMESTAMP") == "off" and epoch is None:
        return ""
    if epoch is not None:
        try:
            t = float(epoch)
        except ValueError:
            t = time.time()
    else:
        t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class scope:
    """Installs a :class:`Record` as the active one for this thread.

    A context manager (the Python idiom for ``ProvenanceScope``'s RAII):
    nests correctly (the previous record, if any, is restored on exit) and
    an exception inside the ``with`` block still restores it. Kept as a
    plain class rather than ``@contextmanager`` so :meth:`get` can read the
    live record while the scope is still open.

    >>> with scope(Mode.BEST_EFFORT) as s:
    ...     set_source("in.vtu", "vtu")
    ...     note("regions-dropped", "Side regions have no OBJ equivalent")
    ...     s.get().notes
    [Note(category='regions-dropped', detail='Side regions have no OBJ equivalent')]
    """

    def __init__(self, mode: Mode, record: "Record | None" = None):
        self._mode = mode
        self._record = record if record is not None else Record()
        self._pushed_core = False

    def __enter__(self) -> "scope":
        if not self._record.timestamp:
            self._record.timestamp = timestamp()
        _stack().append((self._mode, self._record))
        # Also open a matching scope on the C++ side, so a later
        # `_core.<fmt>_write` call -- which reads only ITS OWN thread-local
        # state, knowing nothing about this Python stack -- renders under
        # the same mode/record instead of silently falling back to Off.
        core = _core_module()
        if core is not None:
            core.provenance_scope_push(
                _INT_BY_MODE[self._mode],
                {
                    "source_path": self._record.source_path,
                    "source_format": self._record.source_format,
                    "target_format": self._record.target_format,
                    "encoding": self._record.encoding,
                    "codec": self._record.codec,
                    "float_format": self._record.float_format,
                    "timestamp": self._record.timestamp,
                    "operations": list(self._record.operations),
                    "notes": [(n.category, n.detail) for n in self._record.notes],
                },
            )
            self._pushed_core = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pushed_core:
            core = _core_module()
            if core is not None:
                core.provenance_scope_pop()
        _stack().pop()

    def get(self) -> Record:
        """The record as it stands now, including every :func:`note` made
        since the scope opened."""
        return current_record()


def _render_block(rec: Record) -> list:
    lines = []
    if rec.source_path or rec.source_format:
        line = "Converted from " + (rec.source_path or "(in-memory)")
        if rec.source_format:
            line += f" ({rec.source_format})"
        lines.append(line)
    if rec.target_format:
        line = f"Written as {rec.target_format}"
        extras = []
        if rec.encoding:
            extras.append(rec.encoding)
        if rec.codec:
            extras.append(f"codec={rec.codec}")
        if rec.float_format:
            extras.append(f"float_format={rec.float_format}")
        if extras:
            line += " (" + ", ".join(extras) + ")"
        lines.append(line)
    for op in rec.operations:
        lines.append(f"Operation: {op}")
    for n in rec.notes:
        lines.append(f"Note [{n.category}]: {n.detail}")
    if rec.timestamp:
        lines.append(f"Timestamp: {rec.timestamp}")
    return lines


def render_lines(tier: SlotTier, prefix: str) -> str:
    """``lines(tier)``, each line prefixed with ``prefix`` and
    newline-terminated, ready to write. The twin of
    ``detail::provenance_render_lines`` -- see its C++ doc comment for which
    formats use this shape and which need something else."""
    return "".join(prefix + line + "\n" for line in lines(tier))


def render_xml_comment(tier: SlotTier) -> str:
    """``lines(tier)`` wrapped as one XML comment. The twin of
    ``detail::provenance_render_xml_comment``."""
    rendered = lines(tier)
    if len(rendered) == 1:
        return f"<!--{rendered[0]}-->"
    return "<!--\n" + "".join(line + "\n" for line in rendered) + "-->"


def lines(tier: SlotTier) -> list:
    """The lines a writer should emit, without any comment prefix.

    Always at least :data:`TAG`, as line 0. See ``detail::provenance_lines``'s
    doc comment in the C++ header for the full mode/tier matrix -- this is
    its exact twin, including that ``Mode.REQUIRED`` only ever raises for
    :attr:`SlotTier.NONE`.

    Bridges to the compiled extension's own ``detail::provenance_lines`` when
    it is available, which is what makes a *pure-Python* writer running while
    a scope opened from Python is active (or bridged in from a C++-driven
    caller) agree with the C++ writers byte for byte -- one rendering engine,
    not two independently-maintained ones.
    """
    core = _core_module()
    if core is not None:
        return list(core.provenance_lines(_INT_BY_TIER[tier]))

    out = [TAG]
    stack = _stack()
    # A scope's mode wins; otherwise the process default, which is BEST_EFFORT
    # unless a caller or MESHIOPLUSPLUS_PROVENANCE turned it off.
    mode = stack[-1][0] if stack else default_mode()
    record = stack[-1][1] if stack else _ambient()
    # Whatever happens below, the ambient record must not survive this write: a
    # note fired while writing one file would otherwise reappear in the header
    # of the next. A scope owns its own record and is unaffected.
    if not stack:
        _state.ambient = Record()

    if mode is Mode.OFF:
        return out

    if tier is SlotTier.NONE:
        if mode is Mode.REQUIRED:
            from ._exceptions import WriteError

            raise WriteError(
                "meshio++: provenance: this format's header slot cannot hold any "
                "provenance -- Mode.REQUIRED cannot be honoured here"
            )
        return out

    if tier is not SlotTier.BLOCK:
        return out

    out.extend(_render_block(record))
    return out


# ---------------------------------------------------------------------------
# Read-back (v10.17.0). The twin of `detail::read_provenance_lines` /
# `detail::scan_provenance_text` -- see the C++ header for why this is one
# scanner rather than 44 parsers, and why it returns raw lines rather than a
# re-parsed Record.
# ---------------------------------------------------------------------------

#: The line shapes :func:`_render_block` emits, plus the tag itself. Kept in
#: lockstep with ``kProvenancePrefixes`` in ``detail/provenance.cpp``.
_PREFIXES = (
    "Written by meshio++ v",
    "Converted from ",
    "Written as ",
    "Operation: ",
    "Note [",
    "Timestamp: ",
)

#: How much of a file's head to scan. The block always sits at or near the top
#: of every slot that can hold one.
_SCAN_BYTES = 8192


def _strip_marker(line: str) -> str:
    """Strips one format's leading comment marker, if present."""
    s = line.strip()
    for marker in ("<!--", "//", "comment "):
        if s.startswith(marker):
            return s[len(marker) :].strip()
    if s[:1] in ("#", "!", "*", "$", "%"):
        return s[1:].strip()
    return s


def _candidates(raw: str):
    """Yields every substring of ``raw`` that could be a provenance line.

    Three shapes, each earning its place exactly as in the C++ twin: the
    plain comment-stripped line; the body of a keyword slot that *wraps* the
    text (Tecplot's ``TITLE = "..."``, Ansys's ``(1 "...")``), whose closing
    punctuation is removed only because that opener put it there -- never
    unconditionally, since ``Converted from x (fmt)`` and
    ``Operation: Clean(Weld=true)`` both end in a legitimate ``)``; and each
    ``|``-delimited cell of a box-drawn banner (OpenFOAM), whose credit sits
    in an interior cell rather than at the line's start.
    """
    stripped = _strip_marker(raw)
    yield stripped

    for lead in ('TITLE = "', 'TITLE="', '(1 "'):
        if stripped.startswith(lead):
            body = stripped[len(lead) :]
            yield body.rstrip(')"').strip()

    if "|" in raw:
        for cell in raw.split("|"):
            yield cell.strip()


def scan_provenance_text(text: str):
    """Recovers a provenance block from text already in hand.

    :returns: ``(lines, recognised)`` -- the block's lines as found (comment
        punctuation stripped, in order), and whether the first is meshio++'s
        own tag format.
    """
    found = []
    for raw in text.splitlines():
        for candidate in _candidates(raw):
            if not candidate.startswith(_PREFIXES):
                continue
            while candidate.endswith("-->"):
                candidate = candidate[:-3].strip()
            found.append(candidate)
            break
    recognised = bool(found) and found[0].startswith("Written by meshio++ v")
    return found, recognised


def _scan_zarr_store(directory: str):
    """The provenance block a ``zarr`` store keeps in its root attributes.

    Stdlib JSON rather than the zarr API: this enriches a summary and must
    work on a machine with no zarr installed. Consolidated metadata leaves the
    root ``attributes`` object in place, so one file is enough.
    """
    meta = os.path.join(directory, "zarr.json")
    try:
        with open(meta, "r", encoding="utf-8") as fh:
            attrs = json.load(fh).get("attributes") or {}
    except (OSError, ValueError):
        return [], False
    block = attrs.get("meshioplusplus:provenance")
    if not isinstance(block, list):
        return [], False
    return scan_provenance_text("\n".join(str(line) for line in block))


def _scan_usd_layer(path: str):
    """The block a USD layer keeps in its root-layer ``documentation``.

    ``None`` when pxr is not installed -- the caller then falls back to the
    byte scanner, which reaches a text ``.usda`` and not a crate layer.
    """
    try:
        from pxr import Sdf
    except Exception:
        return None
    try:
        layer = Sdf.Layer.FindOrOpen(path)
    except Exception:
        return [], False
    if layer is None:
        return [], False
    return scan_provenance_text(layer.documentation or "")


def read_provenance_lines(path, max_bytes: int = _SCAN_BYTES):
    """Recovers the provenance block a writer left in ``path``.

    Best-effort: an unopenable path is "nothing found", never an exception --
    this enriches a summary and must not be able to fail one. Bridges to the
    compiled extension when present so the two engines cannot disagree about
    what counts as a block.
    """
    text = str(path)
    if os.path.isdir(text):
        # A directory store holds no head bytes to scan. `zarr` keeps the
        # block in its root group's attributes, which are plain JSON, so this
        # reads them without importing zarr; `pmsh` has no slot at all, and
        # inventing a sidecar file would put a file `Mesh.load` never wrote
        # into someone else's layout.
        return _scan_zarr_store(text)
    if text.lower().endswith((".usd", ".usda", ".usdc")):
        found = _scan_usd_layer(text)
        if found is not None:
            return found
        # Without pxr, fall through to the byte scanner: it finds the block in
        # a `.usda` (text) layer and nothing in a crate one.

    core = _core_module()
    if core is not None:
        try:
            lines_found, recognised = core.read_provenance_lines(str(path), max_bytes)
            return list(lines_found), bool(recognised)
        except Exception:
            return [], False
    try:
        with open(path, "rb") as fh:
            head = fh.read(max_bytes)
    except OSError:
        return [], False
    # Binary slots (STL's 80-byte header, EnSight's str80 records) NUL-pad
    # their text; treating NUL as a line break is what lets one line-oriented
    # scan find those too.
    return scan_provenance_text(head.replace(b"\0", b"\n").decode("utf-8", "replace"))
