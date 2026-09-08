import argparse
from sys import version_info

from ..__about__ import __version__
from . import (
    _agglomerate,
    _ascii,
    _binary,
    _clean,
    _compress,
    _conservative_interpolate,
    _convert,
    _convert_cells,
    _crop,
    _data,
    _dataset,
    _decimate,
    _decimate_volume,
    _decompress,
    _diff,
    _extract_surface,
    _grid_transfer,
    _info,
    _interpolate,
    _isosurface,
    _merge,
    _optimize_volume,
    _partition,
    _pipeline,
    _point_budget,
    _predict,
    _proximity,
    _quality,
    _refine,
    _regions,
    _remesh,
    _remesh_volume,
    _reorder,
    _sdf,
    _slice,
    _smooth,
    _split,
    _stats,
    _subdivide,
    _transform,
    _undo_green,
    _view,
    _voxelize,
)


def main(argv=None):
    parent_parser = argparse.ArgumentParser(
        description="Mesh input/output tools.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parent_parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=_get_version_text(),
        help="display version information",
    )

    subparsers = parent_parser.add_subparsers(title="subcommands", dest="command")
    subparsers.required = True

    parser = subparsers.add_parser("convert", help="Convert mesh files", aliases=["c"])
    _convert.add_args(parser)
    parser.set_defaults(func=_convert.convert)

    parser = subparsers.add_parser("info", help="Print mesh info", aliases=["i"])
    _info.add_args(parser)
    parser.set_defaults(func=_info.info)

    parser = subparsers.add_parser("compress", help="Compress mesh file")
    _compress.add_args(parser)
    parser.set_defaults(func=_compress.compress)

    parser = subparsers.add_parser("decompress", help="Decompress mesh file")
    _decompress.add_args(parser)
    parser.set_defaults(func=_decompress.decompress)

    parser = subparsers.add_parser("ascii", help="Convert to ASCII", aliases=["a"])
    _ascii.add_args(parser)
    parser.set_defaults(func=_ascii.ascii)

    parser = subparsers.add_parser("binary", help="Convert to binary", aliases=["b"])
    _binary.add_args(parser)
    parser.set_defaults(func=_binary.binary)

    parser = subparsers.add_parser(
        "quality", help="Print mesh quality metrics", aliases=["q"]
    )
    _quality.add_args(parser)
    parser.set_defaults(func=_quality.quality)

    parser = subparsers.add_parser(
        "extract-surface",
        help="Extract the boundary surface/edges",
        aliases=["surface"],
    )
    _extract_surface.add_args(parser)
    parser.set_defaults(func=_extract_surface.extract_surface_cmd)

    parser = subparsers.add_parser(
        "reorder",
        help="Renumber nodes/elements (RCM / Morton / Hilbert)",
    )
    _reorder.add_args(parser)
    parser.set_defaults(func=_reorder.reorder_cmd)

    parser = subparsers.add_parser(
        "diff",
        help="Compare two meshes (nonzero exit code if different)",
    )
    _diff.add_args(parser)
    parser.set_defaults(func=_diff.diff_cmd)

    parser = subparsers.add_parser(
        "merge",
        help="Merge two or more meshes into one (optional welding)",
    )
    _merge.add_args(parser)
    parser.set_defaults(func=_merge.merge_cmd)

    parser = subparsers.add_parser(
        "transform",
        help="Affine transform (translate / scale / rotate / matrix / units)",
    )
    _transform.add_args(parser)
    parser.set_defaults(func=_transform.transform_cmd)

    parser = subparsers.add_parser(
        "clean",
        help="Weld / prune / de-dup a mesh in one pass",
    )
    _clean.add_args(parser)
    parser.set_defaults(func=_clean.clean_cmd)

    parser = subparsers.add_parser(
        "crop",
        help="Subset a mesh by a bounding box, half-space or cell_data predicate",
    )
    _crop.add_args(parser)
    parser.set_defaults(func=_crop.crop_cmd)

    parser = subparsers.add_parser(
        "slice",
        help="Planar cross-section of a mesh (volume -> surface, surface -> lines)",
    )
    _slice.add_args(parser)
    parser.set_defaults(func=_slice.slice_cmd)

    parser = subparsers.add_parser(
        "isosurface",
        help="Level set of a scalar point_data field (volume -> surface, surface -> lines)",
    )
    _isosurface.add_args(parser)
    parser.set_defaults(func=_isosurface.isosurface_cmd)

    parser = subparsers.add_parser(
        "voxelize",
        help="Regular hexahedron grid around a mesh (whole box / surface / interior)",
    )
    _voxelize.add_args(parser)
    parser.set_defaults(func=_voxelize.voxelize_cmd)

    parser = subparsers.add_parser(
        "grid-sample",
        help="Sample a mesh's point data onto a regular grid (for a CNN)",
    )
    _grid_transfer.add_sample_args(parser)
    parser.set_defaults(func=_grid_transfer.grid_sample_cmd)

    parser = subparsers.add_parser(
        "grid-scatter",
        help="Write a grid's fields back onto a mesh (grid-sample's inverse)",
    )
    _grid_transfer.add_scatter_args(parser)
    parser.set_defaults(func=_grid_transfer.grid_scatter_cmd)

    parser = subparsers.add_parser(
        "grid-resample",
        help="Resample a grid onto a finer or coarser lattice over the same box",
    )
    _grid_transfer.add_resample_args(parser)
    parser.set_defaults(func=_grid_transfer.grid_resample_cmd)

    parser = subparsers.add_parser(
        "grid-spectrum",
        help="Azimuthally averaged power spectrum of a field on a regular grid",
    )
    _grid_transfer.add_spectrum_args(parser)
    parser.set_defaults(func=_grid_transfer.grid_spectrum_cmd)

    parser = subparsers.add_parser(
        "subsample",
        help="Reduce a mesh to a point cloud under a token budget (farthest-point)",
    )
    _point_budget.add_args(parser)
    parser.set_defaults(func=_point_budget.subsample_cmd)

    parser = subparsers.add_parser(
        "proximity-graph",
        help="Build a radius or k-nearest-neighbour graph over a mesh's points",
    )
    _proximity.add_args(parser)
    parser.set_defaults(func=_proximity.proximity_cmd)

    parser = subparsers.add_parser(
        "predict",
        help="Run a trained PhysicsNeMo checkpoint on one mesh file",
    )
    _predict.add_args(parser)
    parser.set_defaults(func=_predict.predict_cmd)

    parser = subparsers.add_parser(
        "sdf",
        help="Signed distance field: generate a grid over a surface and fill it",
    )
    _sdf.add_args(parser)
    parser.set_defaults(func=_sdf.sdf_cmd)

    parser = subparsers.add_parser(
        "split",
        help="Partition a mesh into multiple files (type / region / regions / component)",
    )
    _split.add_args(parser)
    parser.set_defaults(func=_split.split_cmd)

    parser = subparsers.add_parser(
        "regions",
        help="List a mesh's named regions (name/kind/dim/tag/entry count)",
    )
    _regions.add_args(parser)
    parser.set_defaults(func=_regions.regions_cmd)

    parser = subparsers.add_parser(
        "convert-cells",
        help="Convert the element representation (linearize / simplexify / elevate)",
    )
    _convert_cells.add_args(parser)
    parser.set_defaults(func=_convert_cells.convert_cells_cmd)

    parser = subparsers.add_parser(
        "subdivide",
        help=(
            "Polyhedrally refine: split every eligible 3D cell into one "
            "polyhedral child per face, connected to a new interior point"
        ),
    )
    _subdivide.add_args(parser)
    parser.set_defaults(func=_subdivide.subdivide_cmd)

    parser = subparsers.add_parser(
        "agglomerate",
        help=(
            "Polyhedrally coarsen: merge groups of cells into single larger "
            "polyhedral cells via greedy seed-and-grow over the shared-face dual"
        ),
    )
    _agglomerate.add_args(parser)
    parser.set_defaults(func=_agglomerate.agglomerate_cmd)

    parser = subparsers.add_parser(
        "refine",
        help=(
            "Refine (subdivide cells into same-type children), every cell or a "
            "selected subset with a conforming closure"
        ),
    )
    _refine.add_args(parser)
    parser.set_defaults(func=_refine.refine_cmd)

    parser = subparsers.add_parser(
        "undo-green",
        help=(
            "Restore a transitional (green) cell to its coarse parent, read "
            "verbatim from the coarse mesh"
        ),
    )
    _undo_green.add_args(parser)
    parser.set_defaults(func=_undo_green.undo_green_cmd)

    parser = subparsers.add_parser(
        "decimate",
        help="Reduce a surface mesh's face count by quadric edge collapse",
    )
    _decimate.add_args(parser)
    parser.set_defaults(func=_decimate.decimate_cmd)

    parser = subparsers.add_parser(
        "decimate-volume",
        help="Reduce a tet mesh's cell count by quadric tet-edge collapse",
    )
    _decimate_volume.add_args(parser)
    parser.set_defaults(func=_decimate_volume.decimate_volume_cmd)

    parser = subparsers.add_parser(
        "remesh",
        help="Replace a surface's triangulation with a new, well-shaped one (ACVD)",
    )
    _remesh.add_args(parser)
    parser.set_defaults(func=_remesh.remesh_cmd)

    parser = subparsers.add_parser(
        "remesh-volume",
        help="Retetrahedralize a volume (or closed surface) by isosurface stuffing",
    )
    _remesh_volume.add_args(parser)
    parser.set_defaults(func=_remesh_volume.remesh_volume_cmd)

    parser = subparsers.add_parser(
        "optimize-volume",
        help="ODT-remesh a tetrahedral mesh (relocate vertices + flip connectivity)",
    )
    _optimize_volume.add_args(parser)
    parser.set_defaults(func=_optimize_volume.optimize_volume_cmd)

    parser = subparsers.add_parser(
        "partition",
        help="Decompose into N balanced parts (SFC / KaHIP) for domain decomposition",
    )
    _partition.add_args(parser)
    parser.set_defaults(func=_partition.partition_cmd)

    parser = subparsers.add_parser(
        "pipeline",
        help="Run a settings.json operation chain (read -> ops -> write)",
    )
    _pipeline.add_args(parser)
    parser.set_defaults(func=_pipeline.pipeline_cmd)

    parser = subparsers.add_parser(
        "smooth",
        help="Relax node positions to improve element shape (Laplacian / Taubin)",
    )
    _smooth.add_args(parser)
    parser.set_defaults(func=_smooth.smooth_cmd)

    parser = subparsers.add_parser(
        "interpolate",
        help="Sample data arrays from a source mesh onto a target mesh "
        "(nearest / barycentric)",
    )
    _interpolate.add_args(parser)
    parser.set_defaults(func=_interpolate.interpolate_cmd)

    parser = subparsers.add_parser(
        "conservative-interpolate",
        help="Mass-preserving cross-mesh field transfer (overlap-measure "
        "weighted, not pointwise)",
    )
    _conservative_interpolate.add_args(parser)
    parser.set_defaults(func=_conservative_interpolate.conservative_interpolate_cmd)

    parser = subparsers.add_parser(
        "stats",
        help="Print geometric statistics (bbox, area, volume, ...)",
    )
    _stats.add_args(parser)
    parser.set_defaults(func=_stats.stats_cmd)

    parser = subparsers.add_parser(
        "view",
        help="Open a mesh in an interactive viewer (desktop or browser)",
    )
    _view.add_args(parser)
    parser.set_defaults(func=_view.view_cmd)

    parser = subparsers.add_parser(
        "screenshot",
        help="Render a mesh to a PNG without opening a window",
    )
    _view.add_screenshot_args(parser)
    parser.set_defaults(func=_view.screenshot_cmd)

    # Nested group: `meshioplusplus data <verb>`. The inner parsers each call
    # set_defaults(func=...), which overrides the outer default, so the
    # `args.func(args)` dispatch below reaches the right handler unchanged.
    parser = subparsers.add_parser(
        "data",
        help="Inspect / rename / average / compute on mesh data arrays",
    )
    _data.add_args(parser)
    parser.set_defaults(func=_data.data_cmd)

    # Second nested group: `meshioplusplus dataset <verb>` — curate the
    # hand-editable DatasetManifest JSON (doc/datasets.md). Same wiring.
    parser = subparsers.add_parser(
        "dataset",
        help="Catalogue / split / tag / annotate a dataset manifest",
    )
    _dataset.add_args(parser)
    parser.set_defaults(func=_dataset.dataset_cmd)

    args = parent_parser.parse_args(argv)

    return args.func(args)


def _get_version_text():
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return "\n".join(
        [
            f"meshio++ {__version__} [Python {python_version}]",
            "Copyright (c) 2015-2021 Nico Schlömer et al. (as meshio)",
            "Copyright (c) 2025 Vicente Mataix Ferrándiz",
            "Copyright (c) 2026 the meshio++ contributors",
        ]
    )
