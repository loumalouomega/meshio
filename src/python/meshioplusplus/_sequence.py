"""Multi-file / transient datasets: a *set* of files (or the steps inside one
multi-step file) treated as one ordered logical sequence.

This is how transient solver output actually arrives -- ``out_0000.vtu …
out_0500.vtu`` -- and how most of the 42 formats have to express time, since
only a minority carry several steps natively.

It is a **driver, not a new operation**. Everything here reads and writes
through the public ``read``/``write`` and runs operation chains through
``_pipeline``'s existing step dispatch, so the settings vocabulary has exactly
one owner. Like ``_pipeline.run_pipeline``, this is a pure-Python twin of the
C++ engine (``operations/sequence.hpp``): it dispatches over the public Python
API, which is what gives it the per-format Python fallbacks the C++ driver
lacks. The two are pinned against each other by ``tests/python/test_sequence.py``.

Three shapes, inferred from how many files the input expands to and whether the
output path carries a ``{step}``/``{index}`` token:

``fan-out``
    one multi-step file -> ``out_{step}.vtu``. The answer for every format that
    cannot express time, which is most of them.
``fan-in``
    N single-step files -> one multi-step file (XDMF today).
``sequence``
    N files -> N files, with the operation chain applied per step.

**The streaming invariant.** ``read_sequence`` is a generator and yields exactly
one live ``Mesh`` per iteration; ``write_sequence`` and
``run_sequence_pipeline`` read step *i*, write step *i*, and release it. Nothing
here holds the sequence in memory -- that is a contract, not an optimization,
because the whole feature exists so a 500-step dataset is traversable on a
laptop. ``tests/python/test_sequence.py`` pins it with weak references rather
than trusting the prose.

See ``doc/sequences.md``.
"""

import os
import pathlib
import re

from ._exceptions import ReadError, WriteError
from ._helpers import read, write

__all__ = [
    "read_sequence",
    "write_sequence",
    "sequence_entries",
    "run_sequence_pipeline",
    "TimeSeries",
]


# The `field_data` key carrying a single mesh's own time value: a length-1
# array, generalizing the `exodus:time` convention the Exodus reader/writer
# already round-trips. Colon-namespaced like `partition:part` -- a bare "time"
# would collide with a solver's own field far too often. Twin of
# `meshioplusplus::kSequenceTimeKey`.
TIME_KEY = "meshio:time"

# The formats whose writer can emit a MULTI-step file. There is no file to
# probe on the write side, so unlike the step *count* (which comes from
# `read_metadata`) this is a small owned set, kept honest by a test that
# cross-checks it against what a real two-step fan-in accepts.
#
# **`gid` joined in v10.19.0**, through `_core.gid_write_series` rather than a
# `_SeriesWriter`: it pulls steps (a callable returning `(time, mesh)` or None)
# instead of being pushed to, which is what lets a generator drive it with one
# mesh alive.
# **`usd` joined in v10.35.0**: one stage carries many time samples, so a
# fan-in is its natural shape (`usd.SeriesWriter`, pushed like XDMF's).
_SERIES_WRITERS = ("xdmf", "gid", "usd")

# The formats whose step COUNT can be discovered, so a bare `convert in.X
# out.Y` on one of them might silently write step 0 of many. Consulted before
# `read_metadata`, which for a format with no native metadata reader costs a
# full read -- so this list is what keeps the multi-step guard free for the
# other 39 formats. Same discipline as `_SERIES_WRITERS`: a small owned set,
# kept honest by a test that cross-checks it against which readers actually
# accept a `time_step` kwarg.
#
# **MED is deliberately absent.** It honours `time_step` in the C++ core, but
# has no entry in `registry_metadata_readers()`, so there is no count to read:
# probing it would cost a full read and still report one step. That is a
# recorded gap in MED's metadata support (doc/sequences.md), and listing it
# here would buy nothing while doubling the work of every MED convert. It
# closes for free the moment `read_med_metadata` fills `time_values`.
#
# **`gid` joined in v10.19.0.** Its reader had always honoured `time_step`, but
# `read_gid_metadata` never opened the `.post.res` sibling where steps live, so
# it reported one step and there was nothing here to gate on -- exactly the
# shape of gap MED still has above.
#
# **`usd` joined in v10.35.0.** Unlike the other three its count comes from a
# FULL read -- `read_metadata` has no native USD path, so it falls back and
# reads the stage, whose reader attaches `mesh.time_values` (the Exodus
# side-channel). Correct but not cheap; stated in doc/sequences.md.
_TIME_CAPABLE_READERS = ("xdmf", "exodus", "gid", "usd")

# Formats whose "file" is a DIRECTORY. A glob must keep those entries, which
# `os.path.isfile` alone would drop -- and it is a suffix test rather than a
# registry lookup because the plan is built before any format is resolved.
# Deliberately narrower than "every directory": an ordinary subdirectory that
# happens to match a pattern is still skipped. The C++ `sequence_expand` has
# no counterpart, which costs nothing there -- these two formats are
# Python-only and unreadable from that surface anyway.
DIRECTORY_FORMAT_SUFFIXES = (".pmsh", ".zarr")


def is_sample_path(path) -> bool:
    """Whether ``path`` is one step's worth of data: a file, or a directory
    store (see :data:`DIRECTORY_FORMAT_SUFFIXES`)."""
    text = str(path)
    if os.path.isfile(text):
        return True
    return os.path.isdir(text) and text.lower().endswith(DIRECTORY_FORMAT_SUFFIXES)


_SEQUENCE_INPUT_KEYS = ("Pattern", "Paths", "Times", "TimeFrom")
_SEQUENCE_DOC_KEYS = ("Mode", "Parallel", "Workers")

_DIGITS = re.compile(r"\d+")


# --------------------------------------------------------------------------- #
# The pure units. No I/O -- pinned against the C++ ones through `_core`, never
# merely transcribed: a comparator or a matcher that disagreed across the
# boundary would order a transient dataset differently depending on which
# engine ran it, silently.
# --------------------------------------------------------------------------- #


def _natural_key_parts(text):
    """Split into alternating non-digit / digit runs, tagged for comparison."""
    parts = []
    pos = 0
    for match in _DIGITS.finditer(text):
        if match.start() > pos:
            parts.append((False, text[pos : match.start()]))
        digits = match.group()
        parts.append((True, digits.lstrip("0") or "0"))
        pos = match.end()
    if pos < len(text):
        parts.append((False, text[pos:]))
    return parts


def natural_less(a, b):
    """Natural-numeric ordering, so ``out_9.vtu`` sorts before ``out_10.vtu``.

    The rule, which is a documented contract (see ``doc/sequences.md``):

    1. Both strings split into maximal digit / non-digit runs, compared run by
       run.
    2. Non-digit runs compare byte by byte (as ``bytes``, so the ordering
       matches C++'s ``unsigned char`` comparison rather than Python's
       code-point one -- they differ above U+007F).
    3. Digit runs compare numerically: leading zeros stripped, shorter is less,
       equal lengths compare lexicographically. Done on the digits themselves,
       never through ``int()``, so a 40-digit name cannot surprise.
    4. A digit run sorts before a non-digit run at the same position.
    5. All runs equal -> the *unstripped* strings compare byte-wise, so
       ``out_1`` < ``out_01`` deterministically. This is what makes the
       comparator a strict weak ordering.
    6. The whole path is compared, not the basename.
    """
    pa, pb = _natural_key_parts(a), _natural_key_parts(b)
    for (da, va), (db, vb) in zip(pa, pb):
        if da != db:
            return da  # rule 4
        if da:
            if len(va) != len(vb):
                return len(va) < len(vb)
            if va != vb:
                return va < vb
        elif va != vb:
            ba, bb = va.encode("utf-8", "surrogateescape"), vb.encode(
                "utf-8", "surrogateescape"
            )
            return ba < bb
    if len(pa) != len(pb):
        return len(pa) < len(pb)
    # Rule 5: the tie-break that makes this a strict weak ordering.
    return a.encode("utf-8", "surrogateescape") < b.encode("utf-8", "surrogateescape")


def _natural_sort_key(text):
    """A ``sorted(key=)`` adaptor for :func:`natural_less`."""
    key = []
    for is_digit, value in _natural_key_parts(text):
        # (kind, length, value): kind 0 sorts digit runs first (rule 4), length
        # before value gives numeric order on the stripped digits (rule 3).
        if is_digit:
            key.append((0, len(value), value.encode("utf-8", "surrogateescape")))
        else:
            key.append((1, 0, value.encode("utf-8", "surrogateescape")))
    return (key, text.encode("utf-8", "surrogateescape"))


def glob_match(pattern, name):
    """Match ``name`` against ``pattern``.

    The meshio++ pattern language is exactly ``*`` (any run, possibly empty)
    and ``?`` (exactly one character) -- **no** ``**``, no ``[set]``, no brace
    expansion. Deliberately narrower than :mod:`glob`/:mod:`fnmatch`, and
    hand-written rather than delegated to either, so this matcher and the C++
    one cannot accept different things: ``[abc]`` is three literal characters
    here, not a set.
    """
    p = n = 0
    star = -1
    mark = 0
    while n < len(name):
        if p < len(pattern) and pattern[p] in ("?", name[n]):
            p += 1
            n += 1
        elif p < len(pattern) and pattern[p] == "*":
            star = p
            p += 1
            mark = n
        elif star >= 0:
            p = star + 1
            mark += 1
            n = mark
        else:
            return False
    while p < len(pattern) and pattern[p] == "*":
        p += 1
    return p == len(pattern)


def pattern_has_token(path):
    """Whether ``path`` carries a ``{step}`` or ``{index}`` token."""
    text = str(path)
    return "{step}" in text or "{index}" in text


def expand_pattern(pattern, index, count):
    """Expand ``{step}`` / ``{index}`` in an output pattern.

    ``{index}`` is the plain decimal index; ``{step}`` is that index zero-padded
    to ``max(4, digits(count - 1))``, so a 12-step run writes ``out_0000 …
    out_0011`` and a 20000-step run widens rather than sorting wrongly in a
    directory listing.

    Substring replacement, **not** ``str.format``: this matches the C++ twin and
    means an unrelated ``{`` in the path is a literal. Note that the older
    ``{key}``/``{part}`` expansions in ``_cli/_split.py`` and
    ``_cli/_partition.py`` *do* use ``str.format`` and so raise on a stray
    brace; that asymmetry between the Python and C++ CLIs is pre-existing, and
    these tokens deliberately inherit the C++ side's semantics rather than
    introducing a third convention.
    """
    text = str(pattern)
    width = max(4, len(str(count - 1)) if count > 1 else 1)
    return text.replace("{step}", str(index).zfill(width)).replace(
        "{index}", str(index)
    )


# --------------------------------------------------------------------------- #
# Expansion and the time-value precedence.
# --------------------------------------------------------------------------- #


def _glob(pattern):
    """Expand a ``*``/``?`` pattern to an ordered list of existing files.

    The C++ twin, ``seq_split_pattern``, finds the rightmost of either ``/``
    or ``\\`` in ONE pass (``find_last_of("/\\\\")``). This used to instead
    prefer ``os.sep`` unconditionally whenever it appeared anywhere in the
    text -- correct for a purely native-separator pattern, but wrong for a
    portable ``/``-style pattern (a hand-edited dataset manifest, per
    ``doc/datasets.md``) joined onto a Windows-native ``base_dir``: the mixed
    string's LAST separator is still the ``/`` right before the glob suffix,
    and re-splitting on the last ``\\`` instead silently walked one directory
    too high, so the glob suffix (e.g. ``cases/case_*.vtu``) never matched a
    bare filename. Mirror the C++ split exactly: rightmost of either.
    """
    text = str(pattern)
    seps = ("/",) if os.sep == "/" else ("/", os.sep)
    idx = max((text.rfind(s) for s in seps), default=-1)
    if idx == -1:
        head, sep, base = "", "", text
    else:
        head, sep, base = text[:idx], text[idx], text[idx + 1 :]
    directory = head if sep else "."
    if not directory:
        directory = os.sep
    if "*" in directory or "?" in directory:
        raise ValueError(
            "meshio++: sequence: the directory part of a pattern is taken "
            f"literally, so '{directory}' cannot contain '*' or '?'; glob one "
            "directory at a time"
        )
    try:
        names = os.listdir(directory)
    except OSError as e:
        raise ReadError(
            f"meshio++: sequence: cannot list directory '{directory}' for "
            f"pattern '{text}': {e}"
        ) from None
    matched = [
        os.path.join(directory, name)
        for name in names
        if glob_match(base, name) and is_sample_path(os.path.join(directory, name))
    ]
    matched.sort(key=_natural_sort_key)
    if not matched:
        raise ReadError(f"meshio++: sequence: pattern '{text}' matched no files")
    return matched


def _num_steps_and_times(path, file_format):
    """``read_metadata``-derived step times, or ``[]`` for a single-step file.

    Registry-derived rather than a per-format table -- but **gated** on
    :data:`_TIME_CAPABLE_READERS` first, because ``read_metadata`` on a format
    with no native metadata reader costs a full read, and a plain single-file
    run must not pay for a probe whose answer is known to be "one step".
    """
    from ._helpers import _filetypes_from_path, read_metadata

    fmt = file_format
    if not fmt:
        try:
            candidates = _filetypes_from_path(pathlib.Path(str(path)))
        except Exception:
            return []
        fmt = candidates[0] if candidates else None
    if fmt not in _TIME_CAPABLE_READERS:
        return []
    try:
        meta = read_metadata(path, file_format=fmt)
    except Exception:
        return []
    times = meta.get("time_values") or []
    return [float(t) for t in times]


def num_steps(path, file_format=None):
    """How many time steps ``path`` carries; at least 1.

    Registry-derived (``read_metadata``'s ``time_values``) rather than a
    per-format table, and skipped outright for a format that cannot carry time
    -- which is what keeps this free for the 38 such formats, since
    ``read_metadata`` on a format with no native metadata reader costs a full
    read.

    A format with a time concept but no metadata reader reports 1: MED is
    today's case, a documented gap that closes the moment ``read_med_metadata``
    fills ``time_values``.
    """
    from ._helpers import _filetypes_from_path

    fmt = file_format
    if not fmt:
        try:
            candidates = _filetypes_from_path(pathlib.Path(str(path)))
        except Exception:
            # An unresolvable extension is a capability question, not a read:
            # answer 1 and let the real read produce the diagnostics. (Without
            # this, merely *asking* how many steps a path has would raise
            # before the pipeline validated its operations.)
            return 1
        fmt = candidates[0] if candidates else None
    return max(1, len(_num_steps_and_times(path, fmt)))


def _time_from_filename(path):
    """The last maximal digit run of the stem, or ``None``.

    "Last", so ``run17/out_0042.vtu`` gives 42 -- and the stem, so the
    directory's own digits never leak in.
    """
    stem = pathlib.Path(str(path)).stem
    runs = _DIGITS.findall(stem)
    if not runs:
        return None
    try:
        return float(int(runs[-1]))
    except (ValueError, OverflowError):
        return None


def _time_from_mesh(mesh):
    """A mesh's own single time value, or ``None``.

    ``meshio:time``, or a length-1 ``exodus:time`` -- the convention this
    generalizes. A longer ``exodus:time`` is the writer's whole-history vector
    and is deliberately not read as a per-file scalar.
    """
    import numpy as np

    for key in (TIME_KEY, "exodus:time"):
        value = mesh.field_data.get(key)
        if value is None:
            continue
        flat = np.asarray(value).ravel()
        if flat.size == 1:
            return float(flat[0])
    return None


def sequence_entries(
    source, *, file_format=None, times=None, time_from="auto", sort=False
):
    """The ordered plan for a sequence, without reading any heavy data.

    :param source: a glob pattern (a ``str`` containing ``*``/``?``), a single
        path, or an iterable of paths.
    :param file_format: forced input format; ``None`` resolves per file.
    :param times: explicit per-entry times; its length must equal the entry
        count.
    :param time_from: ``"auto"`` (the documented precedence), ``"file"``,
        ``"filename"`` or ``"index"``.
    :param sort: also natural-numeric sort an explicit path *list*. A pattern
        is always sorted; a caller-supplied list is a stated order and is not
        re-ordered unless asked.
    :returns: a list of ``{"path", "step", "time", "time_source"}`` dicts, in
        order. ``time_source`` is one of ``explicit``/``file``/``filename``/
        ``index`` -- reported rather than left to be guessed, because "this
        time is 0.25 because the file said so" and "this time is 3 because
        nothing said anything" are different facts.
    """
    if time_from not in ("auto", "file", "filename", "index"):
        raise ValueError(
            "meshio++: sequence: time_from must be 'auto', 'file', 'filename' "
            f"or 'index', not {time_from!r}"
        )

    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source) if isinstance(source, os.PathLike) else source
        files = _glob(text) if ("*" in text or "?" in text) else [text]
    else:
        files = [os.fspath(p) if isinstance(p, os.PathLike) else str(p) for p in source]
        if not files:
            raise ValueError("meshio++: sequence: the input path list is empty")
        if sort:
            files = sorted(files, key=_natural_sort_key)

    entries = []
    for path in files:
        if not os.path.exists(path):
            raise ReadError(f"meshio++: sequence: input file not found: '{path}'")
        step_times = []
        if time_from in ("auto", "file"):
            step_times = _num_steps_and_times(path, file_format)
        for k in range(max(1, len(step_times))):
            entry = {"path": path, "step": k, "time": 0.0, "time_source": "index"}
            if k < len(step_times):
                entry["time"] = step_times[k]
                entry["time_source"] = "file"
            entries.append(entry)

    if times is not None:
        times = [float(t) for t in times]
        if len(times) != len(entries):
            raise ValueError(
                f"meshio++: sequence: {len(times)} explicit time value(s) for "
                f"{len(entries)} sequence entr(ies); the counts must match"
            )
        for entry, value in zip(entries, times):
            entry["time"] = value
            entry["time_source"] = "explicit"
        return entries

    for i, entry in enumerate(entries):
        if time_from == "index":
            entry["time"] = float(i)
            entry["time_source"] = "index"
            continue
        if entry["time_source"] == "file":
            continue  # a multi-step file already told us
        if time_from != "file":
            value = _time_from_filename(entry["path"])
            if value is not None:
                entry["time"] = value
                entry["time_source"] = "filename"
                continue
        # Provisional: a single-step file's `meshio:time` needs a real read, so
        # the drivers upgrade this from the mesh they were going to read
        # anyway. No file is ever read twice just to find its time.
        entry["time"] = float(i)
        entry["time_source"] = "index"
    return entries


def _resolve_time(entry, mesh, time_from):
    """Upgrade a provisional index time from the mesh now in hand."""
    if entry["time_source"] != "index" or time_from == "index":
        return entry["time"]
    value = _time_from_mesh(mesh)
    return entry["time"] if value is None else value


# --------------------------------------------------------------------------- #
# The lazy read surface and the writers.
# --------------------------------------------------------------------------- #


def read_sequence(
    source, *, file_format=None, times=None, time_from="auto", sort=False, **read_kwargs
):
    """Yield ``(time, mesh)`` for each step of a sequence, lazily.

    A **generator**, not a list: exactly one ``Mesh`` is alive per iteration, so
    a 500-step dataset is traversable without materialising it. Calling
    ``list(read_sequence(...))`` is the caller's choice and the caller's memory.

    Nothing is read until the first ``next()`` -- the entry plan is built on
    first iteration, so a bad pattern raises there rather than at call time.

    :param source: a glob pattern, a single path, or an iterable of paths.
    :param read_kwargs: forwarded to :func:`meshioplusplus.read`
        (``points_only``, ``arrays``, ...). A per-entry ``time_step`` is
        supplied by the driver and must not be passed here.
    """
    if "time_step" in read_kwargs:
        raise TypeError(
            "meshio++: sequence: read_sequence supplies time_step per entry; "
            "select a single step with meshioplusplus.read instead"
        )
    entries = sequence_entries(
        source, file_format=file_format, times=times, time_from=time_from, sort=sort
    )
    for entry in entries:
        mesh = read(
            entry["path"],
            file_format=file_format or None,
            time_step=entry["step"],
            **read_kwargs,
        )
        yield _resolve_time(entry, mesh, time_from), mesh


class TimeSeries:
    """An ordered ``(time, Mesh)`` sequence, held as **one value** with random
    access — the Python data-model counterpart of the C/Fortran/Julia/R
    sequence handles, which already give lazy, indexable per-step traversal
    (``mio_sequence_read(seq, i)``, ``read_step(seq, i)``, ...).

    ``read_sequence`` is a single-pass generator: once consumed, it is gone,
    and it cannot answer ``len(...)`` or be indexed. A ``TimeSeries`` is built
    once (an ordinary :func:`sequence_entries` expansion) and can then be
    indexed, sliced, iterated over more than once, or interrogated for its
    length or its full time range — the "hold a series as one value" case
    ``doc/roadmap.md`` had left open.

    **It still honours the streaming invariant**: only the entry *plan*
    (paths, per-file step indices, time values) is held — never a mesh.
    ``series[i]`` performs exactly one read and returns immediately; nothing
    is cached across accesses, so repeated indexing costs a repeated read, by
    design, precisely so that holding a 500-entry ``TimeSeries`` costs no more
    memory than holding its plan.

    :param source: a glob pattern, a single path, or an iterable of paths (see
        :func:`sequence_entries`).
    :param read_kwargs: forwarded to :func:`meshioplusplus.read` on every
        access (``points_only``, ``arrays``, ...).

    .. code-block:: python

        series = TimeSeries("out_*.vtu")
        len(series)                  # 12, without reading anything heavy
        t0, mesh0 = series[0]        # exactly one read
        t_last, mesh_last = series[-1]
        for t, mesh in series:       # a fresh, independent pass each time
            ...
        series.times                 # [0.0, 1.0, ..., 11.0], from the plan alone
    """

    def __init__(
        self,
        source,
        *,
        file_format=None,
        times=None,
        time_from="auto",
        sort=False,
        **read_kwargs,
    ):
        if "time_step" in read_kwargs:
            raise TypeError(
                "meshio++: sequence: TimeSeries supplies time_step per entry; "
                "select a single step with meshioplusplus.read instead"
            )
        self._entries = sequence_entries(
            source,
            file_format=file_format,
            times=times,
            time_from=time_from,
            sort=sort,
        )
        self._file_format = file_format
        self._time_from = time_from
        self._read_kwargs = read_kwargs

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        entry = self._entries[index]  # a plain list index: negative works too
        mesh = read(
            entry["path"],
            file_format=self._file_format or None,
            time_step=entry["step"],
            **self._read_kwargs,
        )
        return _resolve_time(entry, mesh, self._time_from), mesh

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"TimeSeries({len(self)} step(s))"

    @property
    def times(self):
        """Every step's time value, from the plan alone -- no reads."""
        return [e["time"] for e in self._entries]

    @property
    def paths(self):
        """Every step's source file path, from the plan alone -- no reads."""
        return [e["path"] for e in self._entries]

    def entries(self):
        """The full plan (see :func:`sequence_entries`): a list of
        ``{"path", "step", "time", "time_source"}`` dicts, one per step."""
        return list(self._entries)


def _series_target_format(path, file_format):
    """The format a multi-step write to ``path`` would use."""
    from ._helpers import _filetypes_from_path

    if file_format:
        return file_format
    candidates = _filetypes_from_path(pathlib.Path(str(path)))
    return candidates[0] if candidates else None


def _check_series_target(path, file_format):
    """Refuse a multi-step write to a format that cannot hold one, by name."""
    fmt = _series_target_format(path, file_format)
    if fmt not in _SERIES_WRITERS:
        raise WriteError(
            f"meshio++: sequence: format '{fmt}' cannot hold a multi-step "
            f"series (only {' / '.join(repr(f) for f in _SERIES_WRITERS)} can); "
            "write one file per step with an output path containing '{step}' "
            "instead"
        )


def write_sequence(path, steps, *, file_format=None, **write_kwargs):
    """Write an iterable of ``(time, mesh)`` pairs.

    A ``{step}``/``{index}`` token in ``path`` writes one file per step
    (fan-out); a plain path writes one multi-step file (fan-in), which today
    means XDMF and raises by name for anything else -- never a silent
    truncation to the first step.

    Streams: at most one ``Mesh`` is held, so ``steps`` may be a generator over
    a dataset far larger than memory.

    :returns: the list of paths written.
    """
    if pattern_has_token(path):
        # The zero-padding width comes from the total count, which only a sized
        # iterable can report. A generator gets the default width of 4 for
        # every file -- self-consistent, and never a run whose first files were
        # padded differently from its last, which is what deriving the width
        # from the running index would produce past 10000 steps.
        count = len(steps) if hasattr(steps, "__len__") else 0
        written = []
        for i, (time, mesh) in enumerate(steps):
            out = expand_pattern(path, i, count)
            mesh.field_data[TIME_KEY] = _time_array(time)
            write(out, mesh, file_format=file_format or None, **write_kwargs)
            written.append(out)
        return written

    _check_series_target(path, file_format)

    fmt = _series_target_format(path, file_format)

    if fmt == "usd":
        from .usd._usd import SeriesWriter

        # Pushed, like XDMF's: one stage, one time sample per step, topology
        # re-authored only when it changes -- so one mesh is alive at a time.
        with SeriesWriter(path, **write_kwargs) as writer:
            for time, mesh in steps:
                writer.write(time, mesh)
        return [str(path)]

    if fmt == "gid":
        from . import _core

        # Pull, not push: gid's series writer asks for the next step, so the
        # iterator is consumed one item at a time and only one mesh is ever
        # alive -- the same streaming property the XDMF branch below has.
        it = iter(steps)

        def _next():
            return next(it, None)

        _core.gid_write_series(str(path), _next)
        return [str(path)]

    with _SeriesWriter(path) as writer:
        for time, mesh in steps:
            writer.write(time, mesh)
    return [str(path)]


def _time_array(value):
    import numpy as np

    return np.array([float(value)])


class _SeriesWriter:
    """One interface over the two XDMF time-series writers.

    Prefers the C++ one (``_core.XdmfTimeSeriesWriter``) because it writes its
    heavy-data companion as a **sibling** of the ``.xdmf``, whereas the
    pure-Python ``xdmf.TimeSeriesWriter`` writes ``<stem>.h5`` relative to the
    process's working directory (a documented divergence -- see
    ``doc/xdmf_time_series.md``). For a sequence the output path is routinely
    not in the CWD, so the Python writer would leave a ``.xdmf`` pointing at a
    companion that is not where it says it is.

    Falls back to the Python writer when the C++ one is unavailable (no HDF5 in
    the build, or an older ``_core``), which is the repo-wide shim pattern.
    """

    def __init__(self, path):
        from . import _core

        self._path = str(path)
        self._wrote_grid = False
        # Follow the build, exactly like the registry's own xdmf entry does
        # ("HDF when HDF5 is available, XML otherwise"): an HDF-format series
        # is unreadable by a core with no HDF5 support, including the very
        # core that just wrote it, so relying on either writer's own default
        # of "HDF" would produce a file only the pure-Python (single-grid-only)
        # reader could ever read back.
        data_format = "HDF" if getattr(_core, "__has_hdf5__", False) else "XML"
        writer_cls = getattr(_core, "XdmfTimeSeriesWriter", None)
        if writer_cls is not None:
            try:
                self._impl = writer_cls(self._path, data_format)
                self._cpp = True
                return
            except Exception:
                pass
        from .xdmf import TimeSeriesWriter

        self._impl = TimeSeriesWriter(self._path, data_format)
        self._cpp = False

    def __enter__(self):
        self._impl.__enter__()
        self._wrote_grid = False
        return self

    def __exit__(self, *exc):
        return self._impl.__exit__(*exc)

    def write(self, time, mesh):
        if not self._wrote_grid:
            if self._cpp:
                self._impl.write_points_cells(mesh)
            else:
                self._impl.write_points_cells(mesh.points, mesh.cells)
            self._wrote_grid = True
        if self._cpp:
            self._impl.write_data(time, mesh)
        else:
            self._impl.write_data(
                time, point_data=mesh.point_data, cell_data=mesh.cell_data
            )


# --------------------------------------------------------------------------- #
# The settings-document driver.
# --------------------------------------------------------------------------- #


def _is_sequence_document(doc, input_path=None, output_path=None):
    """Whether a settings document describes a sequence rather than one file.

    True when it uses any sequence-only key, or when its output path carries a
    ``{step}``/``{index}`` token. The token counts because a document naming
    ``out_{step}.vtu`` obviously means a series -- writing a file with a
    literal brace in its name is nobody's intent.
    """
    if not isinstance(doc, dict):
        return False
    if any(key in doc for key in _SEQUENCE_DOC_KEYS):
        return True
    inp = doc.get("Input")
    if isinstance(inp, dict) and any(k in inp for k in _SEQUENCE_INPUT_KEYS):
        return True
    out = doc.get("Output")
    path = output_path
    if path is None and isinstance(out, dict):
        path = out.get("Path")
    if bool(path) and pattern_has_token(path):
        return True
    # A multi-step input aimed at a single-step output must not quietly become
    # step 0 -- route it here so the driver refuses by name. An explicit
    # Input.Options.TimeStep IS a deliberate single-step selection, so it opts
    # out. The probe is skipped for the formats that cannot carry time at all
    # (see _TIME_CAPABLE_READERS), which is what keeps it free.
    if not isinstance(inp, dict):
        return False
    if (inp.get("Options") or {}).get("TimeStep"):
        return False
    source = input_path if input_path is not None else inp.get("Path")
    if not source:
        return False
    return num_steps(source, inp.get("Format")) > 1


def _infer_mode(entries, out_path, stated):
    """Infer the mode, then check a stated one agrees. See ``doc/sequences.md``."""
    token = pattern_has_token(out_path)
    files = len({entry["path"] for entry in entries})
    if token:
        inferred = "fan-out" if files == 1 and len(entries) > 1 else "sequence"
    elif len(entries) > 1:
        if files == 1:
            raise ValueError(
                f"meshio++: sequence: the input has {len(entries)} time steps "
                f"but the output path '{out_path}' names a single file; add "
                "'{step}' to write one file per step, or select a single step "
                "(--time-step on the CLI, Input.Options.TimeStep in a settings "
                "document)"
            )
        inferred = "fan-in"
    else:
        inferred = "sequence"
    if stated not in (None, "auto") and stated != inferred:
        raise ValueError(
            f"meshio++: sequence: Mode says '{stated}' but the input ({files} "
            f"file(s), {len(entries)} step(s)) and output '{out_path}' describe "
            f"'{inferred}'"
        )
    return inferred


def _source_from_input(inp, input_path):
    """``Input`` -> the source argument :func:`sequence_entries` takes."""
    given = [k for k in ("Path", "Pattern", "Paths") if k in inp]
    if input_path is not None:
        return input_path
    if not given:
        raise ValueError(
            "meshio++: sequence: Input.Path is required (or Input.Pattern / "
            "Input.Paths)"
        )
    if len(given) > 1:
        raise ValueError(
            "meshio++: sequence: Input names more than one source; set exactly "
            "one of Path, Pattern or Paths"
        )
    if "Path" in inp:
        return inp["Path"]
    if "Pattern" in inp:
        if not inp["Pattern"]:
            raise ValueError("meshio++: sequence: Input.Pattern must not be empty")
        return inp["Pattern"]
    paths = inp["Paths"]
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            "meshio++: sequence: Input.Paths must be a non-empty array of strings"
        )
    return list(paths)


def _run_one(args):
    """One (read -> chain -> write) unit, at module scope so it is picklable.

    ``ProcessPoolExecutor`` cannot ship a closure, and the whole point of the
    parallel path is that each file is independent -- so the unit takes plain
    data and returns plain data.
    """
    (
        entry,
        index,
        count,
        file_format,
        read_kwargs,
        steps_spec,
        out_path,
        out_format,
        write_kwargs,
        time_from,
    ) = args
    from ._pipeline import _apply_step

    mesh = read(
        entry["path"],
        file_format=file_format or None,
        time_step=entry["step"],
        **read_kwargs,
    )
    time = _resolve_time(entry, mesh, time_from)
    steps, warnings = [], []
    for step in steps_spec:
        mesh = _apply_step(mesh, step, steps, warnings)
    token = pattern_has_token(out_path)
    if token:
        # `meshio:time` labels a file that is one step OF A SERIES, so it is
        # attached only for a pattern. A document with a single plain output
        # must produce byte-identical output to the single-file pipeline it
        # degenerates to.
        mesh.field_data[TIME_KEY] = _time_array(time)
    path = expand_pattern(out_path, index, count) if token else str(out_path)
    write(path, mesh, file_format=out_format or None, **write_kwargs)
    return steps, warnings


def run_sequence_pipeline(settings, input_path=None, output_path=None):
    """Run a whole sequence settings document: expand, then per step read ->
    apply the chain -> write.

    The sequence-aware twin of :func:`meshioplusplus.run_pipeline`, which
    delegates here automatically for a document that uses any sequence key or
    names a ``{step}``/``{index}`` output. See ``doc/sequences.md`` for the
    schema, the ordering rule and the mode-inference table.

    :returns: the run report ``{"steps": [...], "warnings": [...]}`` -- one
        entry per (step, op), PascalCase counter keys, exactly as the
        single-file runner's.
    """
    from ._pipeline import (
        _check_keys,
        _read_kwargs_from,
        _resolve_settings,
        _validate_step,
        _write_kwargs_from,
    )

    doc = _resolve_settings(settings)
    if not isinstance(doc, dict):
        raise ValueError("meshio++: pipeline: the settings document must be an object")
    _check_keys(
        doc,
        "the settings document",
        ("Version", "Input", "Operations", "Output", "Mode", "Parallel", "Workers"),
    )
    version = doc.get("Version", 1)
    if not isinstance(version, int) or isinstance(version, bool) or version != 1:
        raise ValueError(
            f"meshio++: pipeline: unsupported Version {version!r} (this build knows 1)"
        )

    inp = doc.get("Input")
    out = doc.get("Output")
    if not isinstance(inp, dict):
        raise ValueError("meshio++: pipeline: Input is required and must be an object")
    if not isinstance(out, dict):
        raise ValueError("meshio++: pipeline: Output is required and must be an object")
    _check_keys(
        inp,
        "Input",
        ("Path", "Format", "Options", "Pattern", "Paths", "Times", "TimeFrom"),
    )
    _check_keys(out, "Output", ("Path", "Format", "Encoding", "Codec", "FloatFormat"))

    mode = doc.get("Mode")
    if mode is not None and mode not in ("auto", "sequence", "fan-in", "fan-out"):
        raise ValueError(
            "meshio++: sequence: Mode must be 'sequence', 'fan-in' or "
            f"'fan-out', not {mode!r}"
        )
    workers = doc.get("Workers", 0)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError(
            "meshio++: sequence: Workers must be a non-negative integer "
            "(0 means one per core)"
        )

    out_path = output_path if output_path is not None else out.get("Path")
    if not out_path:
        raise ValueError("meshio++: pipeline: Output.Path is required")
    source = _source_from_input(inp, input_path)

    warnings = []
    read_kwargs = _read_kwargs_from(inp, warnings)
    steps_spec = doc.get("Operations", [])
    if not isinstance(steps_spec, list):
        raise ValueError("meshio++: pipeline: Operations must be an array of steps")
    # Validate the whole chain before anything is read: a typo in step 7 must
    # not cost reading 500 files first.
    for step in steps_spec:
        _validate_step(step)
    write_kwargs = _write_kwargs_from(out, out_path)

    entries = sequence_entries(
        source,
        file_format=inp.get("Format"),
        times=doc["Input"].get("Times") if "Times" in inp else None,
        time_from=inp.get("TimeFrom", "auto"),
    )
    resolved_mode = _infer_mode(entries, out_path, mode)

    fallbacks = sum(1 for e in entries if e["time_source"] == "index")
    if fallbacks and inp.get("TimeFrom") != "index":
        warnings.append(
            f"sequence: no time value found for {fallbacks} of {len(entries)} "
            "entr(ies); using the integer index for those"
        )

    parallel = bool(doc.get("Parallel", False))
    if parallel and resolved_mode == "fan-in":
        # Not a silent serialization: the series writer is one stateful handle
        # whose steps must be appended in order, and the obvious workaround
        # (parallel reads feeding an ordered queue) would hold `workers + 1`
        # meshes and break the streaming invariant. See doc/sequences.md.
        raise ValueError(
            "meshio++: sequence: Parallel is not available for a fan-in; the "
            "series writer appends steps in order, and buffering them would "
            "break the streaming guarantee"
        )

    steps_report = []
    if resolved_mode == "fan-in":
        _check_series_target(out_path, out.get("Format"))
        with _SeriesWriter(out_path) as writer:
            for entry in entries:
                # Streaming: one mesh enters scope per iteration and leaves it.
                mesh = read(
                    entry["path"],
                    file_format=inp.get("Format") or None,
                    time_step=entry["step"],
                    **read_kwargs,
                )
                time = _resolve_time(entry, mesh, inp.get("TimeFrom", "auto"))
                for step in steps_spec:
                    mesh = _apply_step_shim(mesh, step, steps_report, warnings)
                writer.write(time, mesh)
        return {"steps": steps_report, "warnings": warnings}

    jobs = [
        (
            entry,
            i,
            len(entries),
            inp.get("Format"),
            read_kwargs,
            steps_spec,
            out_path,
            out.get("Format"),
            write_kwargs,
            inp.get("TimeFrom", "auto"),
        )
        for i, entry in enumerate(entries)
    ]

    if parallel and len(jobs) > 1:
        # Files are embarrassingly parallel at the driver level. A process pool
        # rather than threads because each worker reads, runs the chain and
        # writes independently, and the operations release no GIL. Results are
        # collected in submission order, so the report is identical to the
        # serial run's.
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers or None) as pool:
            for job_steps, job_warnings in pool.map(_run_one, jobs):
                steps_report.extend(job_steps)
                warnings.extend(job_warnings)
    else:
        for job in jobs:
            job_steps, job_warnings = _run_one(job)
            steps_report.extend(job_steps)
            warnings.extend(job_warnings)
    return {"steps": steps_report, "warnings": warnings}


def _apply_step_shim(mesh, step, steps, warnings):
    """Indirection so ``_pipeline`` is imported lazily (it imports this module)."""
    from ._pipeline import _apply_step

    return _apply_step(mesh, step, steps, warnings)
