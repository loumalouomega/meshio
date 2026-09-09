from .._helpers import _writer_map, read, reader_map, write
from .._repair import repair


def add_args(parser):
    parser.add_argument("infile", type=str, help="surface mesh to be read from")
    parser.add_argument("outfile", type=str, help="repaired mesh to be written to")
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
        "--no-fix-orientation",
        action="store_true",
        help="do not rewind triangles so neighbours agree",
    )
    parser.add_argument(
        "--no-orient-outward",
        action="store_true",
        help="do not flip closed components with a negative signed volume",
    )
    parser.add_argument(
        "--no-fill-holes", action="store_true", help="do not fan-fill boundary loops"
    )
    parser.add_argument(
        "--no-split-non-manifold",
        action="store_true",
        help="do not duplicate bowtie (pinched) vertices",
    )
    parser.add_argument(
        "--max-hole-edges",
        type=int,
        default=10,
        help="longest boundary loop still filled; <= 0 means no limit (default: 10)",
    )
    parser.add_argument(
        "--weld-tolerance",
        type=float,
        default=0.0,
        help="weld coincident points within this distance first (default: 0, off)",
    )
    parser.add_argument(
        "--record-provenance",
        action="store_true",
        help="attach repair:parent_point and repair:hole",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the report"
    )


def _quality_line(q):
    if q["watertight"]:
        return "watertight"
    return (
        f"{q['boundary_edges']} boundary / {q['non_manifold_edges']} non-manifold / "
        f"{q['inconsistent_pairs']} inconsistent edge(s), "
        f"{q['degenerate_triangles']} degenerate triangle(s)"
    )


def repair_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    out, report = repair(
        mesh,
        fix_orientation=not args.no_fix_orientation,
        orient_outward=not args.no_orient_outward,
        fill_holes=not args.no_fill_holes,
        split_non_manifold=not args.no_split_non_manifold,
        max_hole_edges=args.max_hole_edges,
        weld_tolerance=args.weld_tolerance,
        record_provenance=args.record_provenance,
        return_report=True,
    )

    if not args.quiet:
        print("repair")
        print(f"  before:              {_quality_line(report['quality_before'])}")
        print(f"  after:               {_quality_line(report['quality_after'])}")
        print(f"  triangles rewound:   {report['num_flipped']}")
        print(
            f"  components:          {report['num_components']} "
            f"(largest {report['largest_component']}, "
            f"{report['num_oriented_outward']} oriented outward, "
            f"{report['num_unorientable']} unorientable)"
        )
        print(f"  vertices split:      {report['num_vertices_split']}")
        print(
            f"  holes:               {report['num_holes_filled']} filled, "
            f"{report['num_holes_skipped']} skipped "
            f"(of {report['num_holes_detected']} detected)"
        )
        print(
            f"  added:               {report['num_faces_added']} face(s), "
            f"{report['num_points_added']} point(s)"
        )
        if report["points_welded"]:
            print(f"  points welded:       {report['points_welded']}")

    write(args.outfile, out, file_format=args.output_format)
    return 0
