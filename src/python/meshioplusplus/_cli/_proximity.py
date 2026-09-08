"""``proximity-graph``: a graph built from geometry rather than connectivity."""

import numpy as np

from .._helpers import _writer_map, read, reader_map, write
from .._mesh import Mesh
from .._proximity import _graph_positions, proximity_graph


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh or point cloud to read")
    parser.add_argument("outfile", type=str, help="graph to write, as line cells")
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--radius",
        type=float,
        default=None,
        help="link every pair closer than this (the interaction cutoff)",
    )
    group.add_argument(
        "--knn",
        type=int,
        default=None,
        metavar="K",
        help="link each point to its K nearest, then symmetrize",
    )
    parser.add_argument(
        "--box",
        type=str,
        default=None,
        help="'L' or 'Lx,Ly,Lz': a periodic box; pairs are linked by their "
        "minimum image",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=("node", "cell"),
        default="node",
        help="vertices are mesh points (node) or cell centroids (cell)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def proximity_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    box = None
    if args.box is not None:
        box = [float(x) for x in args.box.split(",")]
    edges = proximity_graph(
        mesh,
        method="radius" if args.radius is not None else "knn",
        radius=args.radius,
        max_neighbors=args.knn,
        box_size=box,
        kind=args.kind,
    )
    points = _graph_positions(mesh, args.kind)
    # Each undirected edge once: a `line` cell carries no direction, so
    # writing both would double every segment.
    half = edges[:, edges[0] < edges[1]]
    degree = np.bincount(edges[0], minlength=len(points)).astype(np.int64)
    out = Mesh(
        points,
        [("line", np.ascontiguousarray(half.T))],
        point_data={"degree": degree},
    )
    if not args.quiet:
        isolated = int((degree == 0).sum())
        print(f"{len(points)} vertices, {half.shape[1]} edges")
        rule = f"radius {args.radius}" if args.radius is not None else f"{args.knn}-nn"
        print(f"  rule:           {rule}" + (f", box {box}" if box else ""))
        print(
            f"  degree:         min {int(degree.min()) if len(degree) else 0}, "
            f"mean {degree.mean() if len(degree) else 0.0:.1f}, "
            f"max {int(degree.max()) if len(degree) else 0}"
        )
        print(f"  isolated:       {isolated}")
    write(args.outfile, out, file_format=args.output_format)
    return 0
