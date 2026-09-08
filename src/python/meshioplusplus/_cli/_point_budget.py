"""``subsample``: reduce a mesh to a point cloud under a token budget."""

from .._helpers import _writer_map, read, reader_map, write
from .._point_budget import _METHODS, subsample_points


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument("outfile", type=str, help="point cloud to be written to")
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
        "--count", "-n", type=int, required=True, help="how many points to keep"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=list(_METHODS),
        default="farthest",
        help="farthest (exact farthest-point sampling, O(N*count)), grid (lattice "
        "representatives then farthest-point sampling, O(N + count^2) -- the "
        "scalable choice above a few hundred thousand points) or random",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="drives random, and --start random"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="0",
        help="index of the first selected point for farthest/grid, or 'random' "
        "to draw it from --seed",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="'xlo,ylo,zlo,xhi,yhi,zhi': only points inside this box are "
        "candidates (negatives need --bounds=)",
    )
    parser.add_argument(
        "--record-ids",
        action="store_true",
        help="attach budget:original_point_id to the output",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def subsample_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    bounds = None
    if args.bounds is not None:
        bounds = [float(x) for x in args.bounds.split(",")]
        if len(bounds) != 6:
            raise ValueError("subsample: --bounds expects 'xlo,ylo,zlo,xhi,yhi,zhi'")
    start = None if args.start.lower() == "random" else int(args.start)
    out = subsample_points(
        mesh,
        args.count,
        method=args.method,
        seed=args.seed,
        start=start,
        bounds=bounds,
        record_ids=args.record_ids,
    )
    if not args.quiet:
        print(f"subsampled {len(mesh.points)} points to {len(out.points)}")
        print(f"  method:         {args.method}")
        if bounds is not None:
            print(f"  bounds:         {bounds}")
        print(f"  point_data:     {', '.join(sorted(out.point_data)) or '(none)'}")
    write(args.outfile, out, file_format=args.output_format)
    return 0
