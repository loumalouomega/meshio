from .._helpers import _writer_map, read, reader_map, write
from .._sobolev_deform import sobolev_deform


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh to be read from")
    parser.add_argument("outfile", type=str, help="deformed mesh to be written to")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--output-format",
        "-o",
        type=str,
        choices=sorted(list(_writer_map.keys())),
        help="output file format",
        default=None,
    )
    parser.add_argument(
        "--array",
        type=str,
        required=True,
        help="point_data array holding the raw displacement",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        required=True,
        help="smoothing length in mesh units; 0 applies the raw field",
    )
    parser.add_argument(
        "--fixed-points-array",
        type=str,
        default="",
        help="point_data array whose nonzero entries pin their point",
    )
    parser.add_argument(
        "--fix-boundary",
        action="store_true",
        help="also pin every point on a boundary facet",
    )
    parser.add_argument(
        "--record-filtered",
        action="store_true",
        help="attach sobolev:displacement",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=128,
        help="conjugate-gradient iteration cap (default: 128)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="relative residual tolerance (default: 1e-10)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the report"
    )


def sobolev_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    out, report = sobolev_deform(
        mesh,
        args.array,
        args.length_scale,
        fixed_points=args.fixed_points_array or None,
        fix_boundary=args.fix_boundary,
        record_filtered=args.record_filtered,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        return_report=True,
    )

    if not args.quiet:
        state = "converged" if report["converged"] else "NOT converged"
        print("sobolev_deform")
        print(f"  iterations:          {report['num_iterations']} ({state})")
        print(f"  relative residual:   {report['residual']:g}")
        print(f"  points fixed:        {report['num_fixed']}")
        print(f"  points isolated:     {report['num_isolated']}")
        print(f"  max displacement:    {report['max_displacement']:g}")

    write(args.outfile, out, file_format=args.output_format)
    return 0
