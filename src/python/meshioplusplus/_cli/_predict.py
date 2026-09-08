"""``predict``: run a trained checkpoint on one mesh file."""

import sys

from .._helpers import _writer_map, reader_map


def add_args(parser):
    parser.add_argument("checkpoint", type=str, help="a trained .mdlus checkpoint")
    parser.add_argument("infile", type=str, help="mesh file to predict on")
    parser.add_argument("outfile", type=str, help="where to write the prediction")
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
        "--time-step",
        type=int,
        default=None,
        help="which step of a multi-step input to predict on (default the first)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="the paired mesh a t->t+n or coarse/fine checkpoint compares against",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto (default), cpu, or cuda[:N]",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def predict_cmd(args):
    from ..physicsnemo import predict_file

    try:
        row = predict_file(
            args.checkpoint,
            args.infile,
            args.outfile,
            time_step=args.time_step,
            target_path=args.target,
            input_format=args.input_format,
            output_format=args.output_format,
            device=args.device,
        )
    except ImportError as e:
        # The frameworks are a deliberate non-dependency; say so by name
        # rather than surfacing a traceback.
        print(str(e), file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"predicted {args.infile} -> {row['output_path']}")
        print(f"  rows:           {row['num_rows']}")
        if row.get("rmse") is None:
            print("  error:          not measured (the input carries no truth)")
        else:
            print(f"  rmse:           {row['rmse']:.6g}")
            print(f"  max error:      {row['max_error']:.6g}")
        if row.get("coverage") is not None:
            print(f"  coverage:       {row['coverage']:.4f}")
    return 0
