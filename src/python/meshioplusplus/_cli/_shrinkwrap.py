from .._helpers import _writer_map, read, reader_map, write
from .._shrinkwrap import shrinkwrap


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh whose points move")
    parser.add_argument("target", type=str, help="surface mesh to project onto")
    parser.add_argument("outfile", type=str, help="moved mesh to be written to")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--target-format",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="target file format",
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
        "--offset",
        type=float,
        default=0.0,
        help="signed offset along the hit feature's pseudonormal "
        "(a negative value needs the --offset=-X form)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.0,
        help="leave points farther than this alone; <= 0 means unlimited",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="point_data array selecting (integer) or blending (float) each point",
    )
    parser.add_argument(
        "--target-region",
        type=str,
        default="",
        help="restrict the target to this named cell region",
    )
    parser.add_argument(
        "--normal-weight",
        type=str,
        choices=["angle", "area"],
        default="angle",
        help="vertex-pseudonormal weighting of the target (default: angle)",
    )
    parser.add_argument(
        "--record-distance",
        action="store_true",
        help="attach shrinkwrap:distance",
    )
    parser.add_argument(
        "--record-closest-cell",
        action="store_true",
        help="attach shrinkwrap:closest_cell",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the report"
    )


def shrinkwrap_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    target = read(args.target, file_format=args.target_format)
    out, report = shrinkwrap(
        mesh,
        target,
        offset=args.offset,
        max_distance=args.max_distance,
        weights=args.weights or None,
        target_region=args.target_region,
        normal_weight=args.normal_weight,
        record_distance=args.record_distance,
        record_closest_cell=args.record_closest_cell,
        return_report=True,
    )

    if not args.quiet:
        q = report["quality"]
        print("shrinkwrap")
        print(f"  points projected:    {report['num_projected']}")
        print(f"  points missed:       {report['num_missed']}")
        print(f"  points skipped:      {report['num_skipped']}")
        print(f"  max displacement:    {report['max_displacement']:g}")
        print(
            "  target: "
            + (
                "watertight"
                if q["watertight"]
                else (
                    f"{q['boundary_edges']} boundary / "
                    f"{q['non_manifold_edges']} non-manifold / "
                    f"{q['inconsistent_pairs']} inconsistent edge(s)"
                )
            )
        )

    write(args.outfile, out, file_format=args.output_format)
    return 0
