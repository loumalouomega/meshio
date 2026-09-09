"""``tessellate``: isoparametric subdivision of a mesh's curved cells onto a
refinement lattice, writing the ``tessellate:*`` provenance arrays."""

from .._helpers import _writer_map, read, reader_map, write
from .._tessellation import tessellate


def add_args(parser):
    parser.add_argument("infile", type=str, help="mesh file to be read from")
    parser.add_argument("outfile", type=str, help="tessellated mesh to be written to")
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
        "--levels",
        type=int,
        default=2,
        help="divisions per axis of the reference lattice (default: 2)",
    )
    parser.add_argument(
        "--no-curved",
        action="store_true",
        help="disable curved-cell handling entirely (a full no-op)",
    )
    parser.add_argument(
        "--no-fields",
        action="store_true",
        help="do not interpolate point_data/cell_data onto the output",
    )
    parser.add_argument(
        "--record-stencil",
        action="store_true",
        help="also attach tessellate:stencil/weights as real point_data "
        "(expensive; needed to reconstruct a Tessellation after a file "
        "round trip)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="suppress the summary"
    )


def tessellate_cmd(args):
    mesh = read(args.infile, file_format=args.input_format)
    tess = tessellate(
        mesh,
        levels=args.levels,
        curved=not args.no_curved,
        fields=not args.no_fields,
        record_stencil=args.record_stencil,
    )
    if not args.quiet:
        print(f"tessellated {len(mesh.points)} points to {len(tess.mesh.points)}")
        print(f"  curved source cells:  {tess.schema['num_curved_source_cells']}")
        print(f"  pass-through cells:   {tess.schema['num_pass_through_source_cells']}")
        print(f"  output cells:         {tess.schema['num_cells']}")
        w = tess.schema["watertight"]
        print(
            "  watertight: "
            + (
                "yes"
                if w["num_facets_with_bad_count"] == 0
                else f"NO ({w['num_facets_with_bad_count']} bad facet(s))"
            )
        )
    write(args.outfile, tess.mesh, file_format=args.output_format)
    return 0
