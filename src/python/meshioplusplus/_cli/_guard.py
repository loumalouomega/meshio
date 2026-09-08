"""``guard-fit`` / ``guard-check``: geometry guardrails over a dataset."""

import json

from .._dataset import DatasetManifest
from .._guard import GeometryGuard
from .._helpers import read, reader_map


def add_fit_args(parser):
    parser.add_argument("manifest", type=str, help="the dataset manifest to fit on")
    parser.add_argument("outfile", type=str, help="where to write the guard JSON")
    parser.add_argument(
        "--split", type=str, default="train", help="which split to fit on"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="threshold = margin x the worst training score (default 1.5)",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="skip the quality descriptors (faster on large meshes)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def guard_fit_cmd(args):
    manifest = DatasetManifest.load(args.manifest)
    guard = GeometryGuard.fit(
        manifest,
        split=args.split,
        margin=args.margin,
        quality=not args.no_quality,
    )
    guard.save(args.outfile)
    if not args.quiet:
        print(f"fitted on {guard.schema.get('num_samples')} mesh(es) -> {args.outfile}")
        print(f"  descriptors:    {len(guard.names)}")
        print(f"  threshold:      {guard.threshold:.6g}")
    return 0


def add_check_args(parser):
    parser.add_argument("guardfile", type=str, help="a fitted guard, or a model card")
    parser.add_argument("infile", type=str, help="the mesh to score")
    parser.add_argument(
        "--input-format",
        "-i",
        type=str,
        choices=sorted(list(reader_map.keys())),
        help="input file format",
        default=None,
    )
    parser.add_argument(
        "--top", type=int, default=3, help="how many descriptors to name"
    )
    parser.add_argument("--json", action="store_true", help="emit the raw report")


def guard_check_cmd(args):
    guard = GeometryGuard.load(args.guardfile)
    mesh = read(args.infile, file_format=args.input_format)
    report = guard.check(mesh, top=args.top)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"{args.infile}: {report['verdict']}")
    print(f"  score:          {report['score']:.6g}")
    print(f"  threshold:      {report['threshold']:.6g}")
    for item in report["worst"]:
        print(
            f"  {item['name']:<24} {item['value']:.6g} "
            f"(train mean {item['mean']:.6g}, z {item['z']:+.3g})"
        )
    if report["missing"]:
        print(f"  not comparable: {', '.join(report['missing'])}")
    return 0
