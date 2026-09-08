"""Multi-file / transient datasets (``meshioplusplus._sequence``).

Four things are being pinned here, and they are pinned deliberately rather than
incidentally:

* **The pure units** -- the natural-numeric comparator, the glob matcher and
  the ``{step}``/``{index}`` expander -- against the **C++ ones** through
  ``_core``, not merely against expectations transcribed here. A comparator or
  a matcher that disagreed across the boundary would order a transient dataset
  differently depending on which engine ran it, silently.
* **The time-value precedence**, including which source each entry actually
  used, since "the file said 0.25" and "nothing said anything, so this is
  position 3" are different facts.
* **The streaming invariant** -- one live ``Mesh`` at a time -- with weak
  references, so a regression names the retainer instead of merely being slow.
* **Backward compatibility**: an existing single-file settings.json must take a
  physically unchanged path, byte-identical output included.
"""

import gc
import inspect
import json
import os
import weakref

import numpy as np
import pytest

import meshioplusplus
from meshioplusplus import _sequence
from meshioplusplus._cli._main import main

try:
    from meshioplusplus import _core
except ImportError:  # pragma: no cover - a pure-Python build
    _core = None

needs_core = pytest.mark.skipif(
    _core is None or not hasattr(_core, "sequence_natural_less"),
    reason="_core predates the sequence layer",
)
needs_core_json = pytest.mark.skipif(
    _core is None or not getattr(_core, "__has_json__", False),
    reason="_core has no JSON parser",
)


def _mesh():
    return meshioplusplus.Mesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        [("triangle", np.array([[0, 1, 2]]))],
        point_data={"t": np.array([0.0, 1.0, 2.0])},
    )


@pytest.fixture()
def steps(tmp_path):
    """Twelve single-step .vtu files named out_0 … out_11 (unpadded, on
    purpose: out_10 must not sort before out_9).

    Written uncompressed on purpose: the default ``compression="zlib"``
    would, on a ``-DMESHIOPLUSPLUS_WITH_ZLIB=OFF`` build, silently fall back
    to the pure-Python writer's own stdlib-zlib codec (the repo-wide
    shim pattern), producing a file the C++ *reader* cannot decompress when
    a test reaches it directly through ``_core`` rather than through the
    shim -- an accident of the local zlib build flag, unrelated to what
    these tests are actually pinning.
    """
    for i in range(12):
        meshioplusplus.write(str(tmp_path / f"out_{i}.vtu"), _mesh(), compression=None)
    return tmp_path


# --------------------------------------------------------------------------- #
# The pure units, pinned against C++                                          #
# --------------------------------------------------------------------------- #

# (a, b) pairs where a must sort strictly before b.
ORDER_CASES = [
    ("out_9.vtu", "out_10.vtu"),  # the headline case
    ("out_0009.vtu", "out_0010.vtu"),  # zero-padded
    ("out_0009.vtu", "out_10.vtu"),  # mixed padding, same comparison
    ("run2_step10.vtu", "run10_step2.vtu"),  # two numeric runs: the first wins
    ("run2_step2.vtu", "run2_step10.vtu"),
    ("a/out_9.vtu", "a/out_10.vtu"),  # the whole path, not the basename
    ("a/out_10.vtu", "b/out_1.vtu"),
    ("1abc", "abc1"),  # a digit run sorts first
    ("out_01", "out_1"),  # rule 5's deterministic tie-break
    ("", "a"),
    ("2", "10"),
    ("x" + "9" * 39 + "8", "x" + "9" * 40),  # 40 digits: no int() overflow
]

GLOB_CASES = [
    ("out_*.vtu", "out_0001.vtu", True),
    ("out_*.vtu", "out_0001.vtk", False),
    ("*", "anything", True),
    ("*", "", True),
    ("out_?.vtu", "out_1.vtu", True),
    ("out_?.vtu", "out_.vtu", False),
    ("out_?.vtu", "out_12.vtu", False),
    ("a*b*c", "axxbyyc", True),
    ("a*b*c", "abd", False),
    ("**", "ab", True),
    # Deliberately narrower than fnmatch: a character class is three literals.
    ("f[ab].vtu", "fa.vtu", False),
    ("f[ab].vtu", "f[ab].vtu", True),
    ("a*a*a*a*a*a*b", "a" * 40, False),  # must terminate, must be false
]


@pytest.mark.parametrize("a,b", ORDER_CASES)
def test_natural_order(a, b):
    assert _sequence.natural_less(a, b)
    assert not _sequence.natural_less(b, a)


def test_natural_order_is_irreflexive():
    for a, _ in ORDER_CASES:
        assert not _sequence.natural_less(a, a)


@needs_core
@pytest.mark.parametrize("a,b", ORDER_CASES)
def test_natural_order_matches_cpp(a, b):
    # The parity that matters: the two engines must order a dataset the same.
    assert _core.sequence_natural_less(a, b) is _sequence.natural_less(a, b)
    assert _core.sequence_natural_less(b, a) is _sequence.natural_less(b, a)


def test_natural_sort_agrees_with_the_comparator(steps):
    names = [f"out_{i}.vtu" for i in range(12)]
    # Lexicographically out_10 lands right after out_1, i.e. third overall --
    # which is exactly the wrong answer the natural rule exists to avoid.
    assert sorted(names)[2] == "out_10.vtu"
    assert sorted(names, key=_sequence._natural_sort_key) == names


@pytest.mark.parametrize("pattern,name,expected", GLOB_CASES)
def test_glob_match(pattern, name, expected):
    assert _sequence.glob_match(pattern, name) is expected


@needs_core
@pytest.mark.parametrize("pattern,name,expected", GLOB_CASES)
def test_glob_match_matches_cpp(pattern, name, expected):
    assert _core.sequence_glob_match(pattern, name) is expected


def test_glob_directory_split_prefers_the_rightmost_separator_on_windows(monkeypatch):
    """A portable, "/"-style manifest pattern joined onto a Windows-native
    base_dir must still find the true last path component.

    ``_glob`` used to unconditionally re-split on ``os.sep`` whenever it
    appeared *anywhere* in the text, even after a correct "/" split had
    already been found -- so a pattern like ``<native base>\\cases/case_*.vtu``
    (a manifest's own ``"cases/case_*.vtu"`` joined onto a Windows tmp_path)
    re-split one directory too high, leaving ``cases/case_*.vtu`` itself as
    the "basename" glob_match compares against a bare filename -- which never
    matches, since glob_match has no "/" of its own (it is single-component
    only). Mirrors the C++ twin's ``find_last_of("/\\\\")``, which never had
    this defect.

    A real backslash is not a path separator on POSIX (it is a legal filename
    character), so this cannot be reproduced with real files on a non-Windows
    machine -- it mocks ``os.listdir``/``os.path.isfile`` instead, asserting
    the DIRECTORY the fixed split computes and passes to ``os.listdir`` is
    the true last component, not one level too high.
    """
    monkeypatch.setattr(os, "sep", "\\", raising=False)
    windows_style_prefix = r"C:\Users\runner\AppData\Local\Temp\pytest\case0"
    pattern = f"{windows_style_prefix}\\cases/case_*.vtu"
    expected_dir = f"{windows_style_prefix}\\cases"

    seen_dirs = []

    def fake_listdir(path):
        seen_dirs.append(path)
        if path == expected_dir:
            return ["case_0.vtu", "case_1.vtu", "readme.txt"]
        return []  # the pre-fix (wrong, one-level-too-high) directory: empty

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(os.path, "isfile", lambda p: True)

    matched = _sequence._glob(pattern)
    assert seen_dirs == [expected_dir]
    assert sorted(matched) == [
        os.path.join(expected_dir, "case_0.vtu"),
        os.path.join(expected_dir, "case_1.vtu"),
    ]


@pytest.mark.parametrize(
    "pattern,index,count,expected",
    [
        ("out_{step}.vtu", 0, 12, "out_0000.vtu"),
        ("out_{step}.vtu", 11, 12, "out_0011.vtu"),
        ("out_{index}.vtu", 11, 12, "out_11.vtu"),
        ("out_{step}.vtu", 7, 20000, "out_00007.vtu"),  # widens past 10000
        ("{index}/out_{step}.vtu", 3, 5, "3/out_0003.vtu"),
        ("{step}_{step}.vtu", 3, 5, "0003_0003.vtu"),
    ],
)
def test_expand_pattern(pattern, index, count, expected):
    assert _sequence.expand_pattern(pattern, index, count) == expected


@needs_core
def test_expand_pattern_matches_cpp():
    for pattern, index, count, _ in [
        ("out_{step}.vtu", 0, 12, ""),
        ("out_{index}.vtu", 11, 12, ""),
        ("out_{step}.vtu", 7, 20000, ""),
    ]:
        assert _core.sequence_expand_pattern(
            pattern, index, count
        ) == _sequence.expand_pattern(pattern, index, count)


def test_expand_pattern_uses_substring_replacement_like_cpp():
    # The documented divergence from the older {key}/{part} expansions, which
    # use str.format and raise on a stray brace. These tokens follow the C++
    # side instead, so both CLIs agree.
    assert _sequence.expand_pattern("o{ther}_{step}.vtu", 1, 5) == "o{ther}_0001.vtu"


# --------------------------------------------------------------------------- #
# Expansion and the time-value precedence                                     #
# --------------------------------------------------------------------------- #


def test_expansion_orders_naturally(steps):
    entries = meshioplusplus.sequence_entries(str(steps / "out_*.vtu"))
    assert [os.path.basename(e["path"]) for e in entries] == [
        f"out_{i}.vtu" for i in range(12)
    ]


def test_empty_pattern_is_an_error_not_an_empty_sequence(tmp_path):
    with pytest.raises(meshioplusplus.ReadError, match="matched no files"):
        meshioplusplus.sequence_entries(str(tmp_path / "nothing_*.vtu"))


def test_pattern_directory_component_is_literal():
    with pytest.raises(ValueError, match="taken literally"):
        meshioplusplus.sequence_entries("some*dir/out_*.vtu")


def test_explicit_list_keeps_its_order_unless_asked(steps):
    paths = [str(steps / f"out_{i}.vtu") for i in (2, 0, 1)]
    entries = meshioplusplus.sequence_entries(paths)
    assert os.path.basename(entries[0]["path"]) == "out_2.vtu"
    entries = meshioplusplus.sequence_entries(paths, sort=True)
    assert os.path.basename(entries[0]["path"]) == "out_0.vtu"


def test_time_precedence_explicit_beats_everything(steps):
    entries = meshioplusplus.sequence_entries(
        str(steps / "out_*.vtu"), times=[float(i) * 10 for i in range(12)]
    )
    assert entries[3]["time"] == 30.0
    assert entries[3]["time_source"] == "explicit"


def test_time_explicit_length_mismatch_names_both_counts(steps):
    with pytest.raises(ValueError, match="2 explicit time value.*12 sequence"):
        meshioplusplus.sequence_entries(str(steps / "out_*.vtu"), times=[0.0, 1.0])


def test_time_from_filename_is_the_last_digit_run(tmp_path):
    run = tmp_path / "run17"
    run.mkdir()
    for name in ("out_0001.vtu", "out_0042.vtu"):
        meshioplusplus.write(str(run / name), _mesh())
    entries = meshioplusplus.sequence_entries(str(run / "out_*.vtu"))
    assert [e["time"] for e in entries] == [1.0, 42.0]
    assert entries[0]["time_source"] == "filename"


def test_time_from_index_is_the_recorded_fallback(tmp_path):
    for stem in ("alpha", "beta", "gamma"):
        meshioplusplus.write(str(tmp_path / f"{stem}.vtu"), _mesh())
    entries = meshioplusplus.sequence_entries(str(tmp_path / "*.vtu"))
    assert [e["time"] for e in entries] == [0.0, 1.0, 2.0]
    assert {e["time_source"] for e in entries} == {"index"}


def test_time_from_index_overrides_a_parseable_filename(steps):
    entries = meshioplusplus.sequence_entries(
        str(steps / "out_*.vtu"), time_from="index"
    )
    assert entries[5]["time_source"] == "index"
    assert entries[5]["time"] == 5.0


def test_time_from_file_reads_a_series_step(steps, tmp_path):
    series = str(tmp_path / "series.xdmf")
    meshioplusplus.write_sequence(
        series, meshioplusplus.read_sequence(str(steps / "out_*.vtu"))
    )
    entries = meshioplusplus.sequence_entries(series)
    assert len(entries) == 12
    assert entries[4]["time_source"] == "file"
    assert entries[4]["step"] == 4


def test_time_from_rejects_an_unknown_name(steps):
    with pytest.raises(ValueError, match="time_from must be"):
        meshioplusplus.sequence_entries(str(steps / "out_*.vtu"), time_from="vibes")


# --------------------------------------------------------------------------- #
# Fan-out, fan-in and the round trip                                          #
# --------------------------------------------------------------------------- #


def test_fan_in_then_fan_out_reproduces_every_step(steps, tmp_path):
    originals = [meshioplusplus.read(str(steps / f"out_{i}.vtu")) for i in range(12)]
    series = str(tmp_path / "series.xdmf")
    meshioplusplus.write_sequence(
        series, meshioplusplus.read_sequence(str(steps / "out_*.vtu"))
    )
    written = meshioplusplus.write_sequence(
        str(tmp_path / "back_{step}.vtu"), meshioplusplus.read_sequence(series)
    )
    assert len(written) == 12
    for original, path in zip(originals, written):
        assert meshioplusplus.meshes_equal(original, meshioplusplus.read(path))


def test_round_trip_is_stable(steps, tmp_path):
    series = str(tmp_path / "s.xdmf")
    meshioplusplus.write_sequence(
        series, meshioplusplus.read_sequence(str(steps / "out_*.vtu"))
    )
    first = meshioplusplus.write_sequence(
        str(tmp_path / "a_{step}.vtu"), meshioplusplus.read_sequence(series)
    )
    series2 = str(tmp_path / "s2.xdmf")
    meshioplusplus.write_sequence(
        series2, meshioplusplus.read_sequence(str(tmp_path / "a_*.vtu"))
    )
    second = meshioplusplus.write_sequence(
        str(tmp_path / "b_{step}.vtu"), meshioplusplus.read_sequence(series2)
    )
    for a, b in zip(first, second):
        assert open(a, "rb").read() == open(b, "rb").read()


def test_fan_in_to_a_non_series_format_raises_by_name(steps, tmp_path):
    with pytest.raises(meshioplusplus.WriteError) as excinfo:
        meshioplusplus.write_sequence(
            str(tmp_path / "series.vtu"),
            meshioplusplus.read_sequence(str(steps / "out_*.vtu")),
        )
    message = str(excinfo.value)
    assert "'vtu'" in message  # names the format
    assert "{step}" in message  # names the remedy


def test_series_writers_list_agrees_with_reality(steps, tmp_path):
    # The anti-drift gate for the small owned set: a format that grows a
    # multi-step writer without being listed turns this red naming itself.
    #
    # 'gid' is skipped on a build with no gidpost (needs zlib at compile time,
    # see gid/__init__.py) -- its absence here is a build configuration, not a
    # _SERIES_WRITERS/reality mismatch, which is what this gate actually checks.
    # 'usd' is skipped without usd-core for the same reason.
    #
    # 'usd' also needs a real extension: pxr chooses its encoding from the
    # suffix and refuses to author a layer named '.out' at all, so testing it
    # with the shared name would report a failure that says nothing about this
    # list. Every other writer is content with '.out'.
    from meshioplusplus import _core
    from meshioplusplus._interop import _importable

    has_gidpost = getattr(_core, "__has_gidpost__", False)
    suffixes = {"usd": ".usda"}
    for fmt in sorted(meshioplusplus._helpers._writer_map):
        if fmt == "gid" and not has_gidpost:
            continue
        if fmt == "usd" and not _importable("pxr"):
            continue
        works = True
        try:
            meshioplusplus.write_sequence(
                str(tmp_path / f"series_{fmt}{suffixes.get(fmt, '.out')}"),
                meshioplusplus.read_sequence(str(steps / "out_*.vtu")),
                file_format=fmt,
            )
        except Exception:
            works = False
        claimed = fmt in _sequence._SERIES_WRITERS
        assert claimed == works, (
            f"format {fmt!r}: _SERIES_WRITERS says {claimed} but a real fan-in "
            f"{'succeeded' if works else 'failed'}"
        )


def test_time_capable_readers_list_agrees_with_reality(steps):
    # The twin gate for the read side: a reader that accepts `time_step` must
    # be listed, or the multi-step guard would silently skip it.
    path = str(steps / "out_0.vtu")
    for fmt in _sequence._TIME_CAPABLE_READERS:
        reader = meshioplusplus._helpers.reader_map[fmt]
        assert "time_step" in inspect.signature(reader).parameters, (
            f"{fmt!r} is listed in _TIME_CAPABLE_READERS but its reader takes "
            "no time_step; the multi-step guard would probe it for nothing"
        )
    # And a format that is NOT listed must genuinely not take one.
    assert (
        "time_step"
        not in inspect.signature(meshioplusplus._helpers.reader_map["vtu"]).parameters
    )
    assert _sequence.num_steps(path) == 1


# --------------------------------------------------------------------------- #
# The streaming invariant                                                     #
# --------------------------------------------------------------------------- #


def test_read_sequence_is_lazy(steps):
    assert inspect.isgeneratorfunction(meshioplusplus.read_sequence)
    gen = meshioplusplus.read_sequence(str(steps / "out_*.vtu"))
    # Nothing has happened yet -- not even the expansion.
    assert next(gen)[1].points.shape == (3, 3)
    gen.close()


def test_read_sequence_holds_exactly_one_mesh(steps):
    # Deterministic under CPython refcounting: after moving to the next step,
    # the previous mesh must be unreachable. If anything retained it this
    # fails immediately and points at the retainer.
    refs = []
    for _, mesh in meshioplusplus.read_sequence(str(steps / "out_*.vtu")):
        gc.collect()
        assert all(r() is None for r in refs), "a previous step is still alive"
        refs.append(weakref.ref(mesh))
        del mesh
    gc.collect()
    assert all(r() is None for r in refs)


def test_fan_in_holds_no_more_than_one_mesh_across_many_steps(tmp_path):
    """A wider-scale version of ``test_read_sequence_holds_exactly_one_mesh``,
    but driven through the full ``write_sequence`` path over enough steps
    (200) that an accidental accumulation would be unmistakable.

    A raw ``tracemalloc`` byte threshold was tried here and dropped: pymalloc
    arena behaviour, GC scheduling and numpy's allocator all vary enough
    across Python versions and OSes (observed anywhere from ~1 KB/step to
    ~34 KB/step for the exact same code) that no fixed byte bound is
    portable. Weak references give a deterministic answer instead -- under
    CPython refcounting a released, non-cyclic object is freed immediately,
    so a genuine accumulation shows up as *growing* numbers of live
    references, not a fuzzy byte count.

    One benign overlap is unavoidable and allowed for: ``write_sequence``'s
    own ``for time, mesh in steps:`` loop variable only rebinds once
    ``next()`` returns, so while step *N*'s mesh is being produced, step
    *N-1*'s mesh may still be referenced by that loop variable for a moment.
    Anything *older* than the immediately preceding step must already be
    dead.
    """
    for i in range(200):
        meshioplusplus.write(str(tmp_path / f"out_{i}.vtu"), _mesh())

    refs = []

    def watched():
        for time, mesh in meshioplusplus.read_sequence(str(tmp_path / "out_*.vtu")):
            gc.collect()
            assert all(r() is None for r in refs[:-1]), "an older step is still alive"
            refs.append(weakref.ref(mesh))
            yield time, mesh

    meshioplusplus.write_sequence(str(tmp_path / "series.xdmf"), watched())
    gc.collect()
    assert all(r() is None for r in refs)
    assert len(refs) == 200


# --------------------------------------------------------------------------- #
# TimeSeries: the "hold a series as one value" case, with random access       #
# --------------------------------------------------------------------------- #


def test_timeseries_gives_random_access_without_materializing(steps):
    ts = meshioplusplus.TimeSeries(str(steps / "out_*.vtu"))
    assert len(ts) == 12
    # times/paths/entries() answer from the plan alone -- no reads.
    assert ts.times == [float(i) for i in range(12)]
    assert os.path.basename(ts.paths[3]) == "out_3.vtu"
    assert ts.entries()[3]["time_source"] == "filename"

    t0, mesh0 = ts[0]
    assert t0 == 0.0
    assert mesh0.points.shape == (3, 3)
    t_last, _ = ts[-1]
    assert t_last == 11.0

    sliced = ts[2:5]
    assert [t for t, _ in sliced] == [2.0, 3.0, 4.0]
    assert repr(ts) == "TimeSeries(12 step(s))"


def test_timeseries_is_reusable_unlike_read_sequence(steps):
    # A generator is exhausted after one pass; a TimeSeries is not -- that is
    # the entire point of "holding a series as one value".
    ts = meshioplusplus.TimeSeries(str(steps / "out_*.vtu"))
    first_pass = [t for t, _ in ts]
    second_pass = [t for t, _ in ts]
    assert first_pass == second_pass == ts.times


def test_timeseries_each_access_is_one_independent_read(steps):
    # Repeated indexing must not share mesh state between accesses.
    ts = meshioplusplus.TimeSeries(str(steps / "out_*.vtu"))
    _, mesh_a = ts[3]
    mesh_a.point_data["marker"] = mesh_a.points[:, 0].copy()
    _, mesh_b = ts[3]
    assert "marker" not in mesh_b.point_data


def test_timeseries_rejects_time_step_kwarg(steps):
    with pytest.raises(TypeError, match="time_step"):
        meshioplusplus.TimeSeries(str(steps / "out_*.vtu"), time_step=1)


def test_timeseries_forwards_read_kwargs(steps):
    ts = meshioplusplus.TimeSeries(str(steps / "out_*.vtu"), points_only=True)
    _, mesh = ts[0]
    assert not mesh.point_data


# --------------------------------------------------------------------------- #
# The settings document                                                       #
# --------------------------------------------------------------------------- #


def test_per_step_pipeline_matches_applying_the_steps_individually(steps, tmp_path):
    operations = [{"Op": "Quality"}, {"Op": "Clean"}]
    report = meshioplusplus.run_pipeline(
        {
            "Version": 1,
            "Input": {"Pattern": str(steps / "out_*.vtu")},
            "Operations": operations,
            "Output": {"Path": str(tmp_path / "seq_{step}.vtu")},
        }
    )
    assert len(report["steps"]) == 12 * len(operations)

    for i in range(12):
        one = str(tmp_path / f"one_{i}.vtu")
        meshioplusplus.run_pipeline(
            {
                "Version": 1,
                "Input": {"Path": str(steps / f"out_{i}.vtu")},
                "Operations": operations,
                "Output": {"Path": one},
            }
        )
        via_sequence = meshioplusplus.read(str(tmp_path / f"seq_{i:04d}.vtu"))
        alone = meshioplusplus.read(one)
        assert meshioplusplus.meshes_equal(via_sequence, alone)


def test_mode_asserts_rather_than_selects(steps, tmp_path):
    with pytest.raises(ValueError, match="Mode says 'fan-in'.*describe 'sequence'"):
        meshioplusplus.run_pipeline(
            {
                "Version": 1,
                "Mode": "fan-in",
                "Input": {"Paths": [str(steps / "out_0.vtu")]},
                "Output": {"Path": str(tmp_path / "o.xdmf")},
            }
        )


def test_a_multi_step_input_to_a_single_file_output_refuses(steps, tmp_path):
    series = str(tmp_path / "series.xdmf")
    meshioplusplus.write_sequence(
        series, meshioplusplus.read_sequence(str(steps / "out_*.vtu"))
    )
    with pytest.raises(ValueError, match=r"12 time steps.*\{step\}"):
        meshioplusplus.run_pipeline(
            {
                "Version": 1,
                "Input": {"Path": series},
                "Output": {"Path": str(tmp_path / "one.vtu")},
            }
        )


def test_parallel_with_fan_in_errors_by_name(steps, tmp_path):
    with pytest.raises(ValueError, match="Parallel is not available for a fan-in"):
        meshioplusplus.run_pipeline(
            {
                "Version": 1,
                "Parallel": True,
                "Input": {"Pattern": str(steps / "out_*.vtu")},
                "Output": {"Path": str(tmp_path / "s.xdmf")},
            }
        )


def test_parallel_output_is_identical_to_serial(steps, tmp_path):
    def run(parallel, prefix):
        doc = {
            "Version": 1,
            "Input": {"Pattern": str(steps / "out_*.vtu")},
            "Operations": [{"Op": "Quality"}],
            "Output": {"Path": str(tmp_path / (prefix + "_{step}.vtu"))},
        }
        if parallel:
            doc["Parallel"] = True
            doc["Workers"] = 2
        return meshioplusplus.run_pipeline(doc)

    serial = run(False, "ser")
    parallel = run(True, "par")
    assert serial["steps"] == parallel["steps"]
    for i in range(12):
        a = open(str(tmp_path / f"ser_{i:04d}.vtu"), "rb").read()
        b = open(str(tmp_path / f"par_{i:04d}.vtu"), "rb").read()
        assert a == b


@pytest.mark.parametrize(
    "doc,message",
    [
        ({"Input": {"Path": "a", "Pattern": "*"}}, "more than one source"),
        ({"Input": {"Bogus": 1}}, "Bogus"),
        ({"Input": {"Paths": []}}, "non-empty"),
        ({"Mode": "sideways", "Input": {"Path": "a"}}, "Mode must be"),
        ({"Workers": -1, "Input": {"Pattern": "*"}}, "Workers must be"),
        ({"Input": {"Pattern": "*", "TimeFrom": "vibes"}}, "time_from must be"),
    ],
)
def test_schema_errors_name_the_offender(doc, message, tmp_path):
    doc = dict(doc)
    doc.setdefault("Version", 1)
    doc.setdefault("Output", {"Path": str(tmp_path / "o_{step}.vtu")})
    with pytest.raises(ValueError, match=message):
        meshioplusplus.run_sequence_pipeline(doc)


def test_a_plain_settings_document_is_unchanged(steps, tmp_path):
    """The backward-compatibility guard: a document with no sequence key and a
    plain output must take the single-file path, output included."""
    doc = {
        "Version": 1,
        "Input": {"Path": str(steps / "out_3.vtu")},
        "Operations": [{"Op": "Quality"}],
        "Output": {"Path": str(tmp_path / "plain.vtu")},
    }
    report = meshioplusplus.run_pipeline(doc)
    assert len(report["steps"]) == 1
    mesh = meshioplusplus.read(str(tmp_path / "plain.vtu"))
    # No meshio:time: that key labels one step OF A SERIES, and this is not one.
    assert _sequence.TIME_KEY not in mesh.field_data


@needs_core_json
def test_cpp_matches_python(steps, tmp_path):
    operations = [{"Op": "Quality"}, {"Op": "Clean"}]
    doc = {
        "Version": 1,
        "Input": {"Pattern": str(steps / "out_*.vtu")},
        "Operations": operations,
        "Output": {"Path": str(tmp_path / "py_{step}.vtu")},
    }
    py_report = meshioplusplus.run_pipeline(doc)

    doc["Output"] = {"Path": str(tmp_path / "cpp_{step}.vtu")}
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(doc))
    cpp_report = _core.run_sequence_file(str(settings))

    assert py_report["steps"] == cpp_report["steps"]
    assert py_report["warnings"] == cpp_report["warnings"]
    for i in range(12):
        a = meshioplusplus.read(str(tmp_path / f"py_{i:04d}.vtu"))
        b = meshioplusplus.read(str(tmp_path / f"cpp_{i:04d}.vtu"))
        assert meshioplusplus.meshes_equal(a, b)


# --------------------------------------------------------------------------- #
# The CLI                                                                     #
# --------------------------------------------------------------------------- #


def test_cli_fan_in_accepts_a_pattern_and_a_pre_expanded_argv(steps, tmp_path):
    quoted = str(tmp_path / "quoted.xdmf")
    argv_form = str(tmp_path / "argv.xdmf")
    files = [str(steps / f"out_{i}.vtu") for i in range(12)]

    assert main(["convert", str(steps / "out_*.vtu"), quoted]) == 0
    args = ["convert", files[0]]
    for path in files[1:]:
        args += ["--input", path]
    assert main(args + [argv_form]) == 0

    # Both forms must describe the same dataset.
    a = meshioplusplus.sequence_entries(quoted)
    b = meshioplusplus.sequence_entries(argv_form)
    assert len(a) == len(b) == 12
    for i in range(12):
        assert meshioplusplus.meshes_equal(
            meshioplusplus.read(quoted, time_step=i),
            meshioplusplus.read(argv_form, time_step=i),
        )


def test_cli_fan_out(steps, tmp_path):
    series = str(tmp_path / "series.xdmf")
    assert main(["convert", str(steps / "out_*.vtu"), series]) == 0
    assert main(["convert", series, str(tmp_path / "back_{step}.vtu")]) == 0
    assert os.path.isfile(str(tmp_path / "back_0011.vtu"))


def test_cli_refuses_to_truncate_a_multi_step_input(steps, tmp_path):
    series = str(tmp_path / "series.xdmf")
    assert main(["convert", str(steps / "out_*.vtu"), series]) == 0
    with pytest.raises((ValueError, meshioplusplus.WriteError)):
        main(["convert", series, str(tmp_path / "one.vtu")])
    # ... but an explicit --time-step is a deliberate single-step selection.
    main(["convert", "--time-step=-1", series, str(tmp_path / "last.vtu")])
    assert os.path.isfile(str(tmp_path / "last.vtu"))


def test_cli_no_sequence_is_the_escape_hatch(steps, tmp_path):
    out = str(tmp_path / "plain.vtu")
    main(["convert", "--no-sequence", str(steps / "out_0.vtu"), out])
    assert os.path.isfile(out)


# --------------------------------------------------------------------------- #
# MCP                                                                         #
# --------------------------------------------------------------------------- #


def test_mcp_sequence_tool(steps, tmp_path):
    from meshioplusplus.mcp import _tools

    report = _tools.tool_sequence(
        input_pattern=str(steps / "out_*.vtu"),
        output_path=str(tmp_path / "s.xdmf"),
    )
    json.dumps(report, allow_nan=False)  # strict JSON
    assert len(report["steps_plan"]) == 12
    assert os.path.isfile(report["output_path"])


def test_mcp_sequence_pattern_cannot_escape_the_sandbox(steps, tmp_path):
    from meshioplusplus.mcp import _tools

    _tools.set_root(str(tmp_path))
    try:
        with pytest.raises(ValueError, match="outside the configured root"):
            _tools.tool_sequence(
                input_pattern="../*.vtu", output_path=str(tmp_path / "s.xdmf")
            )
        with pytest.raises(ValueError, match="outside the configured root"):
            _tools.tool_sequence(
                input_pattern="/etc/*", output_path=str(tmp_path / "s.xdmf")
            )
    finally:
        _tools.set_root(None)
