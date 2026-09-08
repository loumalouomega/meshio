import math

from .._curvature import compute_curvature
from .._helpers import _writer_map, read, reader_map, write


def add_args(parser):
    parser.add_argument("infile", type=str, help="surface mesh to be read from")
    parser.add_argument(
        "outfile", type=str, help="mesh with the curvature arrays to be written to"
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
        "--output-format",
        "-o",
        type=str,
        choices=sorted(list(_writer_map.keys())),
        help="output file format",
        default=None,
    )
    parser.add_argument(
        "--no-mean", action="store_true", help="do not attach curvature:mean"
    )
    parser.add_argument(
        "--no-gaussian", action="store_true", help="do not attach curvature:gaussian"
    )
    parser.add_argument(
        "--dual-area",
        type=str,
        choices=["mixed-voronoi", "barycentric"],
        default="mixed-voronoi",
        help="dual area each curvature is divided by (default: mixed-voronoi)",
    )
    parser.add_argument(
        "--include-boundary",
        action="store_true",
        help="compute a biased value at boundary vertices instead of leaving them NaN",
    )
    parser.add_argument(
        "--record-area",
        action="store_true",
        help="also attach curvature:area, the dual area each curvature was divided by",
    )
    parser.add_argument(
        "--record-principal",
        action="store_true",
        help="also attach curvature:principal, the (n, 2) pair k1 >= k2",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="",
        help="restrict to this named cell region (default: every surface cell)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def curvature_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)

    out, report = compute_curvature(
        mesh,
        mean=not args.no_mean,
        gaussian=not args.no_gaussian,
        dual_area=args.dual_area,
        include_boundary=args.include_boundary,
        record_area=args.record_area,
        record_principal=args.record_principal,
        region=args.region,
        return_report=True,
    )

    if not args.quiet:
        q = report["quality"]
        print(f"curvature ({args.dual_area})")
        # The Gauss-Bonnet oracle, printed beside its expected value: on a
        # closed surface the defects sum to 2*pi*chi whatever the tessellation,
        # so a reader can check the result without knowing the estimator.
        print(
            f"  total angle defect:       {report['total_angle_defect']:.6f}"
            f"   (4*pi = {4.0 * math.pi:.6f} for a closed genus-0 surface)"
        )
        print(f"  boundary vertices (NaN):  {report['num_boundary']}")
        print(f"  isolated vertices (NaN):  {report['num_isolated']}")
        print(f"  degenerate triangles:     {report['num_degenerate']}")
        print(
            "  surface: "
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
