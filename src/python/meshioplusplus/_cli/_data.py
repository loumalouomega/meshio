"""The nested ``meshioplusplus data <verb>`` command group.

Nine of the fifteen verbs are thin front-ends over the five data operations:
``rename``/``drop``/``keep`` call ``data_manage``, ``to-cell``/``to-point``
call the two averaging entry points, ``clamp``/``normalize`` call
``data_condition`` with a preset mode, and ``calc``/``info`` map one-to-one.
``export``/``export-dataset`` write the arrays to Parquet/zarr/hdf5 through
:mod:`meshioplusplus._interop`/:mod:`meshioplusplus._ml` — **tabular data
exports for analytics, not mesh formats**: neither round-trips geometry, and
pyarrow/zarr are optional extras, so their imports stay inside the handlers.
Neither has a counterpart in the native C++ CLI.

``gradient``, ``hessian``, ``integrate`` and ``estimate-error`` are a slight
departure: all four are **mesh** operations (``operations/gradient.hpp``,
``operations/hessian.hpp``, ``operations/data_integrate.hpp``,
``operations/error.hpp``), not part of the ``data_*`` family, because they
read geometry (cell measures, face tables) rather than only data arrays.
They live here anyway because they consume and produce data arrays, which is
where a user looks for them. ``hessian`` is ``gradient``'s companion one
order further -- a composition of two ``gradient`` calls, not a new kernel;
``estimate-error`` composes ``gradient`` with the point↔cell averaging round
trip to produce the ZZ recovery-based error indicator ``refine``'s own
``--where`` selector consumes, closing the adaptive loop; ``integrate`` is
``gradient``'s companion the other direction, reducing a ``cell_data`` field
to a physical total/mean instead of differentiating it. See
``doc/gradient.md`` / ``doc/hessian.md`` / ``doc/field_integration.md`` /
``doc/error.md``.

This is the repository's first two-level subcommand. It needs no change to
``_main.py``'s dispatch because ``set_defaults(func=...)`` on the *inner*
parser overrides the outer one, so the existing ``args.func(args)`` call
already reaches the right handler.
"""

import json

from .._data_average import cell_data_to_point_data, point_data_to_cell_data
from .._data_calc import data_calc
from .._data_condition import data_condition
from .._data_info import data_info
from .._data_integrate import data_integrate
from .._data_manage import data_drop, data_keep, data_rename
from .._error import estimate_error
from .._gradient import gradient
from .._helpers import _writer_map, read, reader_map, write
from .._hessian import hessian

_LOCATION_FLAGS = ("point", "cell", "field")


def _add_io_args(parser, with_output=True):
    """The shared ``INFILE [OUTFILE]`` + format-override arguments."""
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    if with_output:
        parser.add_argument("outfile", type=str, help="mesh file to be written to")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    if with_output:
        parser.add_argument(
            "--output-format",
            "-o",
            type=str,
            choices=sorted(list(_writer_map.keys())),
            help="output file format",
            default=None,
        )


def _split_names(value):
    """``"a,b"`` -> ``["a", "b"]``, ignoring empty entries."""
    if not value:
        return []
    return [part for part in (p.strip() for p in value.split(",")) if part]


def _split_rename(value):
    """``"OLD:NEW"`` -> ``("OLD", "NEW")``, splitting on the LAST colon.

    Data names routinely contain colons (``gmsh:physical``), so splitting on the
    last one lets ``--point gmsh:physical:tag`` rename ``gmsh:physical`` to
    ``tag``. The same rule is implemented in the native CLI and documented in
    ``doc/cli.md``.
    """
    if ":" not in value:
        raise ValueError(
            f"data rename: expected OLD:NEW, got '{value}' (no ':' separator)"
        )
    old, _, new = value.rpartition(":")
    if not old or not new:
        raise ValueError(f"data rename: expected OLD:NEW, got '{value}'")
    return old, new


def _split_assignment(value):
    """``"name = expr"`` -> ``("name", "expr")``, splitting on the FIRST ``=``."""
    if "=" not in value:
        raise ValueError(
            f"data calc: expected 'NAME = EXPRESSION', got '{value}' (no '=')"
        )
    name, _, expr = value.partition("=")
    name = name.strip()
    expr = expr.strip()
    if not name or not expr:
        raise ValueError(f"data calc: expected 'NAME = EXPRESSION', got '{value}'")
    return name, expr


# --- data info -------------------------------------------------------------


def add_info_args(parser):
    _add_io_args(parser, with_output=False)
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")


def info_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    arrays = data_info(mesh)
    if args.json:
        print(json.dumps(arrays, indent=2))
        return 0
    print("<meshio++ data summary>")
    if not arrays:
        print("  (the mesh carries no data arrays)")
        return 0
    header = (
        f"  {'location':<11} {'name':<20} {'dtype':<6} {'comp':>4} "
        f"{'entries':>8} {'min':>12} {'max':>12} {'mean':>12} {'nan':>5} {'inf':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in arrays:
        print(
            f"  {a['location']:<11} {a['name']:<20} {a['dtype']:<6} "
            f"{a['num_components']:>4} {a['num_entries']:>8} "
            f"{a['min']:>12.6g} {a['max']:>12.6g} {a['mean']:>12.6g} "
            f"{a['num_nan']:>5} {a['num_inf']:>5}"
        )
        if a["inconsistent_blocks"]:
            print(f"  {'':<11} {'':<20} (warning: cell blocks disagree in components)")
    return 0


# --- data rename / drop / keep --------------------------------------------


def add_rename_args(parser):
    _add_io_args(parser)
    for loc in _LOCATION_FLAGS:
        parser.add_argument(
            f"--{loc}",
            action="append",
            default=[],
            metavar="OLD:NEW",
            help=f"rename a {loc}_data array (repeatable; split on the last ':')",
        )


def rename_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    total = 0
    for loc in _LOCATION_FLAGS:
        for spec in getattr(args, loc):
            old, new = _split_rename(spec)
            mesh = data_rename(mesh, loc, old, new)
            print(f"renamed {loc}_data '{old}' -> '{new}'")
            total += 1
    if total == 0:
        print("data rename: nothing to do (pass --point/--cell/--field OLD:NEW)")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


def add_drop_args(parser):
    _add_io_args(parser)
    for loc in _LOCATION_FLAGS:
        parser.add_argument(
            f"--{loc}",
            type=str,
            default=None,
            metavar="A,B",
            help=f"comma-separated {loc}_data array names to drop",
        )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="skip names that do not exist instead of failing",
    )


def drop_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    for loc in _LOCATION_FLAGS:
        names = _split_names(getattr(args, loc))
        if names:
            mesh = data_drop(mesh, loc, names, ignore_missing=args.ignore_missing)
            print(f"dropped {loc}_data: {', '.join(names)}")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


def add_keep_args(parser):
    _add_io_args(parser)
    for loc in _LOCATION_FLAGS:
        parser.add_argument(
            f"--{loc}",
            type=str,
            default=None,
            metavar="A,B",
            help=f"comma-separated {loc}_data array names to keep "
            "(the location is left untouched if this is omitted)",
        )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="skip names that do not exist instead of failing",
    )


def keep_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    for loc in _LOCATION_FLAGS:
        value = getattr(args, loc)
        # A location the user did not name at all is left alone; naming it with
        # an empty list means "keep nothing there".
        if value is None:
            continue
        names = _split_names(value)
        mesh = data_keep(mesh, loc, names, ignore_missing=args.ignore_missing)
        print(f"kept {loc}_data: {', '.join(names) if names else '(nothing)'}")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


# --- data to-cell / to-point ----------------------------------------------


def add_to_cell_args(parser):
    _add_io_args(parser)
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        metavar="A,B",
        help="comma-separated point_data names (default: all)",
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default="",
        help="append this to each output name (default: same name)",
    )


def to_cell_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    keys = _split_names(args.keys)
    out = point_data_to_cell_data(mesh, keys=keys, suffix=args.target_suffix)
    moved = keys if keys else sorted(mesh.point_data.keys())
    print(f"averaged {len(moved)} point_data array(s) onto the cells")
    write(args.outfile, out, file_format=args.output_format)
    return 0


def add_to_point_args(parser):
    _add_io_args(parser)
    parser.add_argument(
        "--keys",
        type=str,
        default=None,
        metavar="A,B",
        help="comma-separated cell_data names (default: all)",
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="weight by cell measure (area/volume) instead of counting cells equally",
    )
    parser.add_argument(
        "--target-suffix",
        type=str,
        default="",
        help="append this to each output name (default: same name)",
    )


def to_point_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    keys = _split_names(args.keys)
    out = cell_data_to_point_data(
        mesh, keys=keys, weighted=args.weighted, suffix=args.target_suffix
    )
    moved = keys if keys else sorted(mesh.cell_data.keys())
    how = "measure-weighted" if args.weighted else "unweighted"
    print(f"averaged {len(moved)} cell_data array(s) onto the points ({how})")
    write(args.outfile, out, file_format=args.output_format)
    return 0


# --- data calc -------------------------------------------------------------


def add_calc_args(parser):
    _add_io_args(parser)
    for loc in _LOCATION_FLAGS:
        parser.add_argument(
            f"--{loc}",
            action="append",
            default=[],
            metavar="NAME = EXPR",
            help=f"derive a {loc}_data array (repeatable)",
        )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an array that already exists",
    )


def calc_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    total = 0
    for loc in _LOCATION_FLAGS:
        for spec in getattr(args, loc):
            name, expr = _split_assignment(spec)
            mesh = data_calc(
                mesh, expr, location=loc, output=name, overwrite=args.overwrite
            )
            print(f"computed {loc}_data '{name}' = {expr}")
            total += 1
    if total == 0:
        print("data calc: nothing to do (pass --point/--cell/--field 'NAME = EXPR')")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


# --- data clamp / normalize ------------------------------------------------


def _add_condition_common(parser):
    _add_io_args(parser)
    for loc in _LOCATION_FLAGS:
        parser.add_argument(
            f"--{loc}",
            type=str,
            default=None,
            metavar="A,B",
            help=f"comma-separated {loc}_data names (default: all at that location)",
        )
    parser.add_argument(
        "--magnitude",
        action="store_true",
        help="condition by row magnitude instead of per component",
    )
    parser.add_argument(
        "--nan",
        type=str,
        choices=["ignore", "replace", "fail"],
        default="ignore",
        help="what reaches the output for non-finite values (default: ignore)",
    )
    parser.add_argument(
        "--nan-value",
        type=float,
        default=0.0,
        help="replacement used with --nan replace",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="store the result as NAME+SUFFIX instead of replacing in place",
    )


def _selected_locations(args):
    """The (location, names) pairs the user actually asked for."""
    out = []
    for loc in _LOCATION_FLAGS:
        value = getattr(args, loc)
        if value is not None:
            out.append((loc, _split_names(value)))
    return out


def add_clamp_args(parser):
    _add_condition_common(parser)
    parser.add_argument(
        "--min", dest="lo", type=float, required=True, help="lower bound"
    )
    parser.add_argument(
        "--max", dest="hi", type=float, required=True, help="upper bound"
    )


def clamp_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    selected = _selected_locations(args)
    if not selected:
        print("data clamp: nothing to do (pass --point/--cell/--field NAME)")
    for loc, names in selected:
        mesh = data_condition(
            mesh,
            location=loc,
            keys=names,
            mode="clamp",
            scope="magnitude" if args.magnitude else "component",
            lo=args.lo,
            hi=args.hi,
            nan_policy=args.nan,
            nan_replacement=args.nan_value,
            suffix=args.suffix,
        )
        label = ", ".join(names) if names else "(all)"
        print(f"clamped {loc}_data {label} to [{args.lo:g}, {args.hi:g}]")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


def add_normalize_args(parser):
    _add_condition_common(parser)
    parser.add_argument(
        "--to",
        type=str,
        default="0,1",
        metavar="LO,HI",
        help="target range (default: 0,1)",
    )
    parser.add_argument(
        "--zero-mean",
        action="store_true",
        help="standardize to zero mean and unit standard deviation instead",
    )


def normalize_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    parts = _split_names(args.to)
    if not args.zero_mean and len(parts) != 2:
        raise ValueError(f"data normalize: --to expects LO,HI, got '{args.to}'")
    lo, hi = (0.0, 1.0) if args.zero_mean else (float(parts[0]), float(parts[1]))
    mode = "standardize" if args.zero_mean else "normalize"

    selected = _selected_locations(args)
    if not selected:
        print("data normalize: nothing to do (pass --point/--cell/--field NAME)")
    for loc, names in selected:
        mesh = data_condition(
            mesh,
            location=loc,
            keys=names,
            mode=mode,
            scope="magnitude" if args.magnitude else "component",
            lo=lo,
            hi=hi,
            nan_policy=args.nan,
            nan_replacement=args.nan_value,
            suffix=args.suffix,
        )
        label = ", ".join(names) if names else "(all)"
        target = "zero mean / unit std" if args.zero_mean else f"[{lo:g}, {hi:g}]"
        print(f"normalized {loc}_data {label} to {target}")
    write(args.outfile, mesh, file_format=args.output_format)
    return 0


def add_export_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument("outfile", type=str, help="Parquet file to be written to")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--location",
        type=str,
        choices=("point", "cell"),
        default="point",
        help="which data location to export (default: point)",
    )


def export_cmd(args):
    # Imported here, not at module scope: pyarrow is an optional extra and the
    # other nine verbs must keep working without it.
    from .._interop import write_parquet

    mesh = read(args.infile, file_format=args.input_format)
    write_parquet(mesh, args.outfile, location=args.location)
    print(f"exported {args.location}_data to {args.outfile}")
    return 0


# --- data export-dataset ----------------------------------------------------


def add_export_dataset_args(parser):
    parser.add_argument(
        "infiles",
        type=str,
        nargs="+",
        help="mesh files: several paths, or ONE quoted glob pattern "
        "(natural-numeric ordering), or one multi-step file",
    )
    parser.add_argument(
        "outpath",
        type=str,
        help="dataset to write (a directory for parquet, a store path for "
        "zarr, a file for hdf5)",
    )
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--location",
        type=str,
        choices=("point", "cell"),
        default="point",
        help="which data location to export (default: point)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("parquet", "zarr", "hdf5"),
        default="parquet",
        help="on-disk dataset layout (default: parquet, hive-partitioned)",
    )
    parser.add_argument(
        "--mesh-id",
        type=str,
        choices=("stem", "index"),
        default="stem",
        help="how each mesh's id is derived (default: the file stem)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def export_dataset_cmd(args):
    # Imported here, not at module scope: the dataset backends are optional
    # extras and the other verbs must keep working without them.
    from .._ml import write_dataset

    source = args.infiles if len(args.infiles) > 1 else args.infiles[0]
    manifest = write_dataset(
        source,
        args.outpath,
        location=args.location,
        format=args.format,
        mesh_id=args.mesh_id,
        file_format=args.input_format,
    )
    if not args.quiet:
        entries = manifest["entries"]
        rows = sum(e["num_rows"] for e in entries)
        print(
            f"wrote {len(entries)} mesh(es), {rows} row(s), "
            f"{len(manifest['columns'])} column(s) to {args.outpath} "
            f"({args.format})"
        )
    return 0


# --- data export-cae --------------------------------------------------------


def _parse_named_value(text, flag):
    """``name=value`` for the repeatable global-parameter flags."""
    if "=" not in text:
        raise ValueError(
            f"meshio++: data export-cae: {flag} takes name=value, got {text!r}"
        )
    name, _, value = text.partition("=")
    try:
        return name.strip(), float(value)
    except ValueError:
        raise ValueError(
            f"meshio++: data export-cae: {flag} value for '{name.strip()}' must "
            f"be a number, got {value!r}"
        ) from None


def add_export_cae_args(parser):
    parser.add_argument(
        "infiles",
        type=str,
        nargs="+",
        help="mesh files: several paths, or ONE quoted glob pattern "
        "(natural-numeric ordering), or one multi-step file",
    )
    parser.add_argument(
        "outdir", type=str, help="directory to write one .npz per case into"
    )
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--surface-fields",
        type=str,
        default=None,
        metavar="A,B",
        help="comma-separated surface field names, in column order "
        "(default: every numeric array, sorted)",
    )
    parser.add_argument(
        "--volume-fields",
        type=str,
        default=None,
        metavar="A,B",
        help="comma-separated volume (nodal) field names, in column order",
    )
    parser.add_argument(
        "--global",
        dest="globals_",
        type=str,
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="a case parameter, repeatable (e.g. --global stream_velocity=30)",
    )
    parser.add_argument(
        "--global-reference",
        type=str,
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="reference value for a case parameter, repeatable "
        "(default: the parameter's own value)",
    )
    parser.add_argument(
        "--global-order",
        type=str,
        default=None,
        metavar="A,B",
        help="stacking order of global_params_values (DoMINO is order-"
        "sensitive); default alphabetical",
    )
    parser.add_argument(
        "--name-template",
        type=str,
        default="case_{index}.npz",
        help="output file name pattern (default: case_{index}.npz)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def export_cae_cmd(args):
    from ..cae import export_cases

    source = args.infiles if len(args.infiles) > 1 else args.infiles[0]
    params = dict(
        _parse_named_value(text, "--global") for text in (args.globals_ or [])
    )
    references = dict(
        _parse_named_value(text, "--global-reference")
        for text in (args.global_reference or [])
    )
    written = export_cases(
        source,
        args.outdir,
        name_template=args.name_template,
        file_format=args.input_format,
        # `or None` throughout: an omitted flag means "every numeric array",
        # which is not the same request as an explicit empty list.
        surface_fields=_split_names(args.surface_fields) or None,
        volume_fields=_split_names(args.volume_fields) or None,
        global_params=params or None,
        global_params_reference=references or None,
        global_params_order=_split_names(args.global_order) or None,
    )
    if not args.quiet:
        print(f"wrote {len(written)} case(s) to {args.outdir}")
    return 0


# --- data gradient ---------------------------------------------------------


def add_gradient_args(parser):
    _add_io_args(parser)
    parser.add_argument(
        "--array",
        type=str,
        required=True,
        metavar="NAME",
        help="point_data array to differentiate (cell_data has no derivative)",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="gradient",
        choices=("gradient", "divergence", "curl"),
        help="which differential operator to apply (default: gradient)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="green-gauss",
        choices=("green-gauss", "least-squares"),
        help="how to reconstruct the per-cell derivative (default: green-gauss)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="cell",
        choices=("cell", "point"),
        help="where the result is stored (default: cell)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="NAME",
        help="output array name (default: <array>:<op>)",
    )
    parser.add_argument(
        "--component",
        type=int,
        default=None,
        metavar="I",
        help="differentiate only this component (gradient only; default: all)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing array of the output name instead of failing",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def gradient_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    out, report = gradient(
        mesh,
        args.array,
        operator=args.op,
        method=args.method,
        location=args.location,
        output=args.output,
        component=args.component,
        overwrite=args.overwrite,
        return_report=True,
    )
    if not args.quiet:
        name = args.output or f"{args.array}:{args.op}"
        print(f"wrote {args.location}_data '{name}' ({args.method})")
        print(f"  cells skipped (NaN):      {report['num_skipped']}")
        print(f"  cells fell back to GG:    {report['num_fallback']}")
    write(args.outfile, out, file_format=args.output_format)
    return 0


# --- data hessian --------------------------------------------------------------


def add_hessian_args(parser):
    _add_io_args(parser)
    parser.add_argument(
        "--array",
        type=str,
        required=True,
        metavar="NAME",
        help="scalar point_data array to differentiate twice (cell_data has no derivative)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="green-gauss",
        choices=("green-gauss", "least-squares"),
        help="how each internal gradient pass reconstructs its derivative (default: green-gauss)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="cell",
        choices=("cell", "point"),
        help="where the result is stored (default: cell)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="NAME",
        help="output array name (default: <array>:hessian)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing array of the output name instead of failing",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def hessian_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    out, report = hessian(
        mesh,
        args.array,
        method=args.method,
        location=args.location,
        output=args.output,
        overwrite=args.overwrite,
        return_report=True,
    )
    if not args.quiet:
        name = args.output or f"{args.array}:hessian"
        print(f"wrote {args.location}_data '{name}' ({args.method})")
        print(f"  cells skipped (NaN):      {report['num_skipped']}")
        print(f"  cells fell back to GG:    {report['num_fallback']}")
    write(args.outfile, out, file_format=args.output_format)
    return 0


# --- data integrate ----------------------------------------------------------


def add_integrate_args(parser):
    _add_io_args(parser, with_output=False)
    parser.add_argument(
        "--array",
        action="append",
        metavar="NAME",
        dest="arrays",
        help="cell_data array to integrate (repeatable; default: every "
        "cell_data array)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")


def integrate_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    report = data_integrate(mesh, arrays=args.arrays)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print("<meshio++ field integration>")
    if not report:
        print("  (the mesh carries no cell_data arrays)")
        return 0
    for arr in report:
        print(f"  {arr['name']} ({arr['num_components']} component(s))")
        for region in [arr["domain"]] + arr["regions"]:
            label = region["name"] or "(whole mesh)"
            totals = ", ".join(f"{v:.6g}" for v in region["total_per_component"])
            means = ", ".join(f"{v:.6g}" for v in region["mean_per_component"])
            print(
                f"    {label:<20} cells={region['num_cells']:<6} "
                f"skipped={region['num_skipped']:<4} total=[{totals}] mean=[{means}]"
            )
    return 0


# --- data estimate-error ----------------------------------------------------


def add_estimate_error_args(parser):
    _add_io_args(parser)
    parser.add_argument(
        "--array",
        type=str,
        required=True,
        metavar="NAME",
        help="point_data array to estimate the recovered-gradient error of "
        "(cell_data has no derivative to recover)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="zz",
        choices=("zz",),
        help="estimator family (default: zz, the only one that exists today)",
    )
    parser.add_argument(
        "--marking",
        type=str,
        default="none",
        choices=("none", "absolute", "fraction", "dorfler"),
        help="how to turn the indicator into a boolean error:marked array "
        "(default: none)",
    )
    parser.add_argument(
        "--marking-value",
        type=float,
        default=0.0,
        metavar="V",
        help="meaning depends on --marking: an absolute threshold, a "
        "fraction in (0, 1] of cells, or the Doerfler bulk fraction theta "
        "in (0, 1] (ignored for 'none')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="NAME",
        help="indicator array name (default: error:zz)",
    )
    parser.add_argument(
        "--marked",
        type=str,
        default=None,
        metavar="NAME",
        help="marking array name (default: error:marked; ignored when "
        "--marking is 'none')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing array of an output name instead of failing",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def estimate_error_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    out, report = estimate_error(
        mesh,
        args.array,
        method=args.method,
        marking=args.marking,
        marking_value=args.marking_value,
        output=args.output,
        marked_name=args.marked,
        overwrite=args.overwrite,
        return_report=True,
    )
    if not args.quiet:
        name = args.output or "error:zz"
        print(f"wrote cell_data '{name}' ({args.method})")
        print(f"  global error:             {report['global_error']:.6g}")
        print(f"  cells skipped (NaN):      {report['num_skipped']}")
        if args.marking != "none":
            marked_name = args.marked or "error:marked"
            print(f"  wrote cell_data '{marked_name}' ({args.marking})")
            print(f"  cells marked:             {report['num_marked']}")
    write(args.outfile, out, file_format=args.output_format)
    return 0


# --- group wiring ----------------------------------------------------------

_VERBS = (
    (
        "info",
        "Summarize every data array (dtype, shape, min/max/mean, NaN/inf)",
        add_info_args,
        info_cmd,
    ),
    ("rename", "Rename data arrays", add_rename_args, rename_cmd),
    ("drop", "Drop data arrays by name", add_drop_args, drop_cmd),
    ("keep", "Keep only the named data arrays", add_keep_args, keep_cmd),
    ("to-cell", "Average point_data onto the cells", add_to_cell_args, to_cell_cmd),
    ("to-point", "Average cell_data onto the points", add_to_point_args, to_point_cmd),
    (
        "calc",
        "Derive a new array from an elementwise expression",
        add_calc_args,
        calc_cmd,
    ),
    ("clamp", "Clamp values into [min, max]", add_clamp_args, clamp_cmd),
    (
        "gradient",
        "Differentiate a point_data field (gradient / divergence / curl)",
        add_gradient_args,
        gradient_cmd,
    ),
    (
        "hessian",
        "Second derivative of a scalar point_data field, gradient's "
        "companion one order further",
        add_hessian_args,
        hessian_cmd,
    ),
    (
        "integrate",
        "Cell-measure-weighted total/mean of a cell_data field, whole mesh "
        "and per named Cell region",
        add_integrate_args,
        integrate_cmd,
    ),
    (
        "estimate-error",
        "Estimate the ZZ recovery-based error of a point_data field, and "
        "optionally mark cells for refinement",
        add_estimate_error_args,
        estimate_error_cmd,
    ),
    (
        "normalize",
        "Rescale values to a target range (or zero mean / unit std)",
        add_normalize_args,
        normalize_cmd,
    ),
    (
        "export",
        "Export data arrays to Parquet for analytics (NOT a mesh format: "
        "geometry is not round-tripped; needs `pip install meshioplusplus[arrow]`)",
        add_export_args,
        export_cmd,
    ),
    (
        "export-dataset",
        "Export a SET of meshes as one dataset keyed by mesh_id "
        "(hive-partitioned Parquet, or chunked zarr/hdf5 groups; "
        "needs `[arrow]`, `[zarr]` or h5py respectively)",
        add_export_dataset_args,
        export_dataset_cmd,
    ),
    (
        "export-cae",
        "Export a SET of meshes as one .npz per case in the CAE sample "
        "layout PhysicsNeMo's DoMINO/Transolver datapipes read",
        add_export_cae_args,
        export_cae_cmd,
    ),
)


def add_args(parser):
    sub = parser.add_subparsers(title="data subcommands", dest="data_command")
    sub.required = True
    for name, help_text, add, func in _VERBS:
        p = sub.add_parser(name, help=help_text)
        add(p)
        p.set_defaults(func=func)


def data_cmd(args):
    """Reached only if argparse somehow admits a bare ``data`` invocation."""
    print("error: 'data' requires a subcommand: " + ", ".join(v[0] for v in _VERBS))
    return 2
