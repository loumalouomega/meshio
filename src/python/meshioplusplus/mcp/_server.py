"""FastMCP registration layer: typed wrappers over :mod:`._tools`.

This is the only module in the package that imports the ``mcp`` SDK, so it is
only importable with the ``[mcp]`` extra installed (which needs Python >= 3.10
— the SDK's own floor, not meshio++'s). All behaviour lives in the pure layer;
each wrapper here contributes exactly two things: the typed signature FastMCP
derives the tool's JSON schema from, and the docstring the client shows the
model. Wrappers never leak tracebacks — every call is guarded and failures
come back as ``{"error", "error_type"}`` payloads the model can act on.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP, Image

from ..__about__ import __version__
from . import _tools

_INSTRUCTIONS = """\
meshio++ mesh I/O and processing. All tools are stateless and file-path based:
they read mesh files (40+ formats: VTK/VTU/VTP, Gmsh, Exodus, MED, Abaqus,
STL, OBJ, PLY, XDMF, CGNS, ...), operate, and write output files, returning a
JSON report. Start with `formats` to see supported formats, `info` for a fast
file summary, and `convert` to translate between formats. Inspection tools
(`stats`, `quality`, `data_info`, `regions`, `diff`) return reports without
writing anything. Mesh operations (clean, refine, decimate, smooth, slice,
partition, ...) take input_path/output_path. Paths may be confined to a root
directory by the server's --root option.
"""


# The one guard, shared with the HTTP front-end's dispatch: a tool failure is
# a {"error", "error_type"} payload, never a traceback (or a 500).
_guard = _tools.guard


def create_server(root: Optional[str] = None, host: Optional[str] = None) -> FastMCP:
    """Build the FastMCP server with every tool and resource registered.

    ``host`` matters only for the HTTP front-end: FastMCP enables its
    DNS-rebinding protection (Host/Origin allow-lists) when the host is a
    loopback name, so a ``--host 0.0.0.0`` server must be told, or the
    ``/mcp`` endpoint would reject every non-loopback ``Host`` header.
    """
    if root is not None:
        _tools.set_root(root)
    kwargs = {"host": host} if host is not None else {}
    server = FastMCP("meshioplusplus", instructions=_INSTRUCTIONS, **kwargs)
    _register_inspection(server)
    _register_conversion(server)
    _register_operations(server)
    _register_data(server)
    _register_dataset(server)
    _register_training(server)
    _register_gated(server)
    _register_resources(server)
    return server


# --------------------------------------------------------------------------- #
# Inspection                                                                  #
# --------------------------------------------------------------------------- #
def _register_inspection(server: FastMCP) -> None:
    @server.tool()
    def formats() -> dict:
        """List every readable and writable mesh format and the extension map."""
        return _guard(_tools.tool_formats)

    @server.tool()
    def sniff(input_path: str) -> dict:
        """Identify a mesh file's format from its leading bytes and extension."""
        return _guard(_tools.tool_sniff, input_path=input_path)

    @server.tool()
    def info(input_path: str, file_format: Optional[str] = None) -> dict:
        """Fast file summary without loading heavy arrays: point/cell counts,
        cell blocks, data-array names, named regions, time steps, format."""
        return _guard(_tools.tool_info, input_path=input_path, file_format=file_format)

    @server.tool()
    def stats(
        input_path: str, file_format: Optional[str] = None, time_step: int = 0
    ) -> dict:
        """Geometric statistics: bounding box, centroid, total area, signed and
        unsigned volume, per-type cell counts, inverted-cell count."""
        return _guard(
            _tools.tool_stats,
            input_path=input_path,
            file_format=file_format,
            time_step=time_step,
        )

    @server.tool()
    def quality(
        input_path: str,
        file_format: Optional[str] = None,
        output_path: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> dict:
        """Mesh-quality metrics (scaled jacobian, aspect ratio, skewness,
        angles, ...) as per-metric min/max/mean + histograms. Pass output_path
        to also write a mesh copy carrying quality:<metric> cell data."""
        return _guard(
            _tools.tool_quality,
            input_path=input_path,
            file_format=file_format,
            output_path=output_path,
            output_format=output_format,
        )

    @server.tool()
    def data_info(input_path: str, file_format: Optional[str] = None) -> dict:
        """Describe every point/cell/field data array: dtype, shape, components,
        min/max/mean (whole-array and per component), NaN/Inf counts."""
        return _guard(
            _tools.tool_data_info, input_path=input_path, file_format=file_format
        )

    @server.tool()
    def data_integrate(
        input_path: str,
        file_format: Optional[str] = None,
        arrays: Optional[List[str]] = None,
    ) -> dict:
        """Cell-measure-weighted total/mean of cell_data arrays (whole mesh,
        and independently for every named Cell region) -- gradient's
        integration companion. arrays defaults to every cell_data array."""
        return _guard(
            _tools.tool_data_integrate,
            input_path=input_path,
            file_format=file_format,
            arrays=arrays,
        )

    @server.tool()
    def regions(input_path: str, file_format: Optional[str] = None) -> dict:
        """List named regions (point/cell/side groups: gmsh physical groups,
        Exodus sets, Abaqus NSET/ELSET, ...) with kind, dim, tag and size."""
        return _guard(
            _tools.tool_regions, input_path=input_path, file_format=file_format
        )

    @server.tool()
    def bandwidth(input_path: str, file_format: Optional[str] = None) -> dict:
        """Node-numbering bandwidth (max node-index spread over any cell)."""
        return _guard(
            _tools.tool_bandwidth, input_path=input_path, file_format=file_format
        )

    @server.tool()
    def data_preview(
        input_path: str,
        array: str,
        location: str = "point",
        offset: int = 0,
        limit: int = 100,
        file_format: Optional[str] = None,
    ) -> dict:
        """Return a bounded window of one data array's values (location:
        point | cell | field; cell arrays are concatenated block-major)."""
        return _guard(
            _tools.tool_data_preview,
            input_path=input_path,
            array=array,
            location=location,
            offset=offset,
            limit=limit,
            file_format=file_format,
        )

    @server.tool()
    def diff(
        path_a: str,
        path_b: str,
        format_a: Optional[str] = None,
        format_b: Optional[str] = None,
        atol: float = 1e-12,
        rtol: float = 1e-9,
        unordered: bool = False,
        max_reported: int = 10,
    ) -> dict:
        """Compare two mesh files. Returns a verdict (identical / equal within
        tolerance / different), an `equal` boolean, and per-section detail."""
        return _guard(
            _tools.tool_diff,
            path_a=path_a,
            path_b=path_b,
            format_a=format_a,
            format_b=format_b,
            atol=atol,
            rtol=rtol,
            unordered=unordered,
            max_reported=max_reported,
        )


# --------------------------------------------------------------------------- #
# Conversion                                                                  #
# --------------------------------------------------------------------------- #
def _register_conversion(server: FastMCP) -> None:
    @server.tool()
    def convert(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        points_only: bool = False,
        arrays: Optional[List[str]] = None,
        time_step: int = 0,
        mode: str = "auto",
        compression: Optional[str] = None,
    ) -> dict:
        """Convert a mesh between formats (formats inferred from extensions
        unless given). points_only/arrays/time_step narrow the read. mode
        selects ascii|binary output where the format supports it; compression
        selects zlib|lz4|zstd|lzma (VTU/VTP block codecs), gzip (CGNS/H5M/XDMF)
        or 'none' to decompress."""
        return _guard(
            _tools.tool_convert,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            points_only=points_only,
            arrays=arrays,
            time_step=time_step,
            mode=mode,
            compression=compression,
        )

    @server.tool()
    def pipeline(
        settings_path: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> dict:
        """Run a settings.json operation pipeline: read Input.Path, apply the
        Operations chain (Transform/Gradient/Refine/Clean/... -- PascalCase
        ops and keys, see doc/pipeline.md), write Output.Path.
        input_path/output_path override the paths in the settings file; both
        the settings file and the paths inside it stay inside the sandbox
        root when one is configured."""
        return _guard(
            _tools.tool_pipeline,
            settings_path=settings_path,
            input_path=input_path,
            output_path=output_path,
        )

    @server.tool()
    def sequence(
        output_path: str,
        input_pattern: Optional[str] = None,
        input_paths: Optional[List[str]] = None,
        mode: Optional[str] = None,
        times: Optional[List[float]] = None,
        time_from: str = "auto",
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> dict:
        """Treat a set of files (or the steps inside one multi-step file) as one
        transient dataset. Give exactly one of input_pattern (a glob -- '*' and
        '?' only) or input_paths. A '{step}'/'{index}' token in output_path
        writes one file per step (fan-out); a plain path writes one multi-step
        file (fan-in, XDMF only -- any other format fails by name rather than
        silently keeping step 0). Ordering is natural-numeric, so out_9 precedes
        out_10. mode optionally asserts 'sequence'/'fan-in'/'fan-out'.
        See doc/sequences.md."""
        return _guard(
            _tools.tool_sequence,
            input_pattern=input_pattern,
            input_paths=input_paths,
            output_path=output_path,
            mode=mode,
            times=times,
            time_from=time_from,
            input_format=input_format,
            output_format=output_format,
        )


# --------------------------------------------------------------------------- #
# Mesh operations                                                             #
# --------------------------------------------------------------------------- #
def _register_operations(server: FastMCP) -> None:
    @server.tool()
    def extract_surface(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        record_parent_ids: bool = False,
    ) -> dict:
        """Extract the boundary surface (3D volume -> boundary faces, 2D
        surface -> boundary edges)."""
        return _guard(
            _tools.tool_extract_surface,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            record_parent_ids=record_parent_ids,
        )

    @server.tool()
    def extract_skin(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        linearize: bool = False,
    ) -> dict:
        """Extract the outer skin of a volume mesh (optionally linearized)."""
        return _guard(
            _tools.tool_extract_skin,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            linearize=linearize,
        )

    @server.tool()
    def reorder(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "rcm",
    ) -> dict:
        """Renumber nodes and cells (method: rcm | morton | hilbert). The
        report includes the bandwidth before and after."""
        return _guard(
            _tools.tool_reorder,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            method=method,
        )

    @server.tool()
    def clean(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        weld: bool = False,
        atol: float = 1e-8,
        remove_orphans: bool = True,
        drop_degenerate: bool = True,
        drop_duplicate_cells: bool = True,
    ) -> dict:
        """Clean a mesh: optionally weld coincident points, then drop
        degenerate cells, duplicate cells and orphan points. Reports counts."""
        return _guard(
            _tools.tool_clean,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            weld=weld,
            atol=atol,
            remove_orphans=remove_orphans,
            drop_degenerate=drop_degenerate,
            drop_duplicate_cells=drop_duplicate_cells,
        )

    @server.tool()
    def crop(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        plane_origin: Optional[List[float]] = None,
        plane_normal: Optional[List[float]] = None,
        where_array: Optional[str] = None,
        where_compare: str = "<",
        where_value: float = 0.0,
        mode: str = "all",
        record_ids: bool = False,
    ) -> dict:
        """Crop to a bounding box (bbox: [xmin,ymin,zmin,xmax,ymax,zmax]), a
        half-space (plane_origin + plane_normal, keeps the +normal side), or a
        cell_data predicate (where_array + where_compare in <, <=, >, >=, ==, !=
        + where_value). Give exactly one. A non-finite cell value never matches
        the predicate. mode 'all' keeps cells with every node inside, 'any' with
        any node inside -- it applies to bbox/plane only, since a cell_data value
        is already one per cell. Inside/outside a surface composes: run
        distance_to_surface with location='center', then crop on 'sdf:distance'
        < 0."""
        return _guard(
            _tools.tool_crop,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            bbox=bbox,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            where_array=where_array,
            where_compare=where_compare,
            where_value=where_value,
            mode=mode,
            record_ids=record_ids,
        )

    @server.tool(name="slice")
    def slice_mesh(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        origin: Optional[List[float]] = None,
        normal: Optional[List[float]] = None,
        record_parent_ids: bool = False,
    ) -> dict:
        """Planar cross-section: the true intersection with a plane, one
        topological dimension below the input (default plane: z = 0)."""
        return _guard(
            _tools.tool_slice,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            origin=origin if origin is not None else [0.0, 0.0, 0.0],
            normal=normal if normal is not None else [0.0, 0.0, 1.0],
            record_parent_ids=record_parent_ids,
        )

    @server.tool()
    def isosurface(
        input_path: str,
        output_path: str,
        array: str,
        isovalues: Union[List[float], float],
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        component: Optional[int] = None,
        record_parent_ids: bool = False,
    ) -> dict:
        """Contour a point_data scalar at one or more isovalues; the output
        carries iso:value / iso:index cell data per contour."""
        return _guard(
            _tools.tool_isosurface,
            input_path=input_path,
            output_path=output_path,
            array=array,
            isovalues=isovalues,
            input_format=input_format,
            output_format=output_format,
            component=component,
            record_parent_ids=record_parent_ids,
        )

    @server.tool()
    def grid(
        output_path: str,
        dims: List[int],
        output_format: Optional[str] = None,
        origin: Optional[List[float]] = None,
        spacing: Optional[List[float]] = None,
        max_cells: int = 20000000,
    ) -> dict:
        """Generate a regular hexahedron lattice of dims=(nx, ny, nz) cells. The
        only tool that reads no input mesh: it creates one. Useful as a background
        grid, a sampling domain, or a fixture that needs no file."""
        return _guard(
            _tools.tool_grid,
            output_path=output_path,
            dims=dims,
            output_format=output_format,
            origin=origin,
            spacing=spacing,
            max_cells=max_cells,
        )

    @server.tool()
    def voxelize(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        cell_size: Optional[float] = None,
        bounds: Optional[List[float]] = None,
        padding: float = 0.0,
        padding_relative: float = 0.0,
        fill: str = "all",
        attach_occupancy: bool = False,
        max_cells: int = 20000000,
        sign: str = "pseudonormal",
    ) -> dict:
        """Build a regular hexahedron grid around a mesh. Give exactly one of
        resolution (nx, ny, nz) or cell_size. fill='all' keeps the whole bounding
        box, 'surface' keeps only cells a triangle passes through, and 'inside'
        keeps only cells whose centre is inside the surface. Reports the lattice
        origin/spacing/dims and how many cells were kept."""
        return _guard(
            _tools.tool_voxelize,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            resolution=resolution,
            cell_size=cell_size,
            bounds=bounds,
            padding=padding,
            padding_relative=padding_relative,
            fill=fill,
            attach_occupancy=attach_occupancy,
            max_cells=max_cells,
            sign=sign,
        )

    @server.tool()
    def grid_sample(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        cell_size: Optional[float] = None,
        bounds: Optional[List[float]] = None,
        padding: float = 0.0,
        padding_relative: float = 0.0,
        fields: Optional[List[str]] = None,
        extrapolate: bool = False,
        fill_value: float = 0.0,
        max_cells: int = 20000000,
    ) -> dict:
        """Sample a mesh's point data onto a regular grid - the step a CNN or a
        superresolution model needs before it can see a mesh at all. Give exactly
        one of resolution (nx, ny, nz cell counts) or cell_size. Writes the grid
        as a lattice mesh; use .vti so the lattice survives the round trip.
        Reports the channel names (a k-component array expands to name_0..name_k)
        and the coverage, the fraction of grid points inside the mesh - a low
        coverage means the grid is mostly fill and a model would learn it."""
        return _guard(
            _tools.tool_grid_sample,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            resolution=resolution,
            cell_size=cell_size,
            bounds=bounds,
            padding=padding,
            padding_relative=padding_relative,
            fields=fields,
            extrapolate=extrapolate,
            fill_value=fill_value,
            max_cells=max_cells,
        )

    @server.tool()
    def grid_scatter(
        grid_path: str,
        target_path: str,
        output_path: str,
        grid_format: Optional[str] = None,
        target_format: Optional[str] = None,
        output_format: Optional[str] = None,
        fields: Optional[List[str]] = None,
        on_conflict: str = "error",
    ) -> dict:
        """Write a grid's fields back onto a mesh's points by trilinear
        interpolation - grid_sample's inverse, and the step that turns a model's
        output back into something every format can hold. Consecutive
        name_0/name_1/name_2 channels rebuild one multi-component array.
        on_conflict is 'error', 'overwrite' or 'suffix'."""
        return _guard(
            _tools.tool_grid_scatter,
            grid_path=grid_path,
            target_path=target_path,
            output_path=output_path,
            grid_format=grid_format,
            target_format=target_format,
            output_format=output_format,
            fields=fields,
            on_conflict=on_conflict,
        )

    @server.tool()
    def grid_resample(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        factor: Optional[int] = None,
        resolution: Optional[List[int]] = None,
        fields: Optional[List[str]] = None,
    ) -> dict:
        """Resample a grid onto a finer or coarser lattice covering the same box,
        by trilinear interpolation. This is the baseline a superresolution model
        has to beat. Give exactly one of factor (an integer upscale) or
        resolution (nx, ny, nz cell counts)."""
        return _guard(
            _tools.tool_grid_resample,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            factor=factor,
            resolution=resolution,
            fields=fields,
        )

    @server.tool()
    def grid_power_spectrum(
        input_path: str,
        field: str,
        input_format: Optional[str] = None,
        max_bins: int = 256,
    ) -> dict:
        """The azimuthally averaged power spectrum of one field on a regular
        grid - whether a super-resolved or generated field carries the right
        small-scale content, which a pointwise error cannot see. Needs an
        isotropic grid (equal spacing on every axis) and reports power per
        wavenumber bin plus the mode count per bin. The power sums to
        mean(field**2) exactly."""
        return _guard(
            _tools.tool_grid_power_spectrum,
            input_path=input_path,
            field=field,
            input_format=input_format,
            max_bins=max_bins,
        )

    @server.tool()
    def subsample(
        input_path: str,
        output_path: str,
        count: int,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "farthest",
        seed: int = 0,
        start: Optional[int] = 0,
        bounds: Optional[List[float]] = None,
        record_ids: bool = False,
    ) -> dict:
        """Reduce a mesh to a point cloud of exactly count points, so a large
        surface fits a transformer's token budget. method is 'farthest' (exact
        farthest-point sampling, O(N*count), the most uniform coverage), 'grid'
        (lattice representatives then farthest-point sampling, O(N + count^2),
        the scalable choice above a few hundred thousand points) or 'random'.
        start is the index of the first selected point (None draws it from
        seed); bounds (xlo, ylo, zlo, xhi, yhi, zhi) restricts the candidates to
        a box. The output keeps the selected points, their point_data and Point
        regions, and gets a vertex block so every format can hold it; cells and
        cell_data are dropped. record_ids attaches budget:original_point_id."""
        return _guard(
            _tools.tool_subsample,
            input_path=input_path,
            output_path=output_path,
            count=count,
            input_format=input_format,
            output_format=output_format,
            method=method,
            seed=seed,
            start=start,
            bounds=bounds,
            record_ids=record_ids,
        )

    @server.tool()
    def proximity_graph(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "radius",
        radius: Optional[float] = None,
        max_neighbors: Optional[int] = None,
        box_size: Optional[List[float]] = None,
        kind: str = "node",
    ) -> dict:
        """Build a graph from geometry rather than connectivity, for a particle
        state that has positions and no cells, or for contact edges between
        surfaces that are near but not connected. method is 'radius' (every pair
        closer than radius -- the physical interaction cutoff, degree varies with
        density) or 'knn' (each point's max_neighbors nearest, then symmetrized,
        so degree is near-constant). box_size (one value or one per axis) turns
        on the periodic minimum-image convention, so a pair either side of a
        boundary is linked by its short image; a radius past half the smallest
        box side is refused, the minimum image being ambiguous there. kind is
        'node' (mesh points) or 'cell' (cell centroids). The output is the graph
        as line cells over those positions, with a 'degree' point array."""
        return _guard(
            _tools.tool_proximity_graph,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            method=method,
            radius=radius,
            max_neighbors=max_neighbors,
            box_size=box_size,
            kind=kind,
        )

    @server.tool()
    def compute_sdf(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        structure: str = "voxel",
        resolution: Optional[List[int]] = None,
        cell_size: Optional[float] = None,
        bounds: Optional[List[float]] = None,
        padding: float = 0.0,
        padding_relative: float = 0.1,
        root_resolution: int = 8,
        max_depth: int = 4,
        band_cells: float = 1.0,
        max_cells: int = 20000000,
        sign: str = "pseudonormal",
        location: str = "corner",
        band: float = 0.0,
    ) -> dict:
        """Generate a grid over a surface and fill it with signed distances --
        the one call that turns a surface into a field. structure='voxel' is a
        dense lattice sized by resolution (nx, ny, nz) or cell_size, exactly one
        of which must be given; structure='octree' refines near the surface
        instead and sizes itself from root_resolution and max_depth, so passing
        resolution or cell_size with it is an error. An octree's output is
        1-irregular (it has hanging nodes). Reports the root dims, the finest
        spacing, the octree depth and the surface's watertightness. Write the
        result as .vti to keep the grid header; no other format carries it."""
        return _guard(
            _tools.tool_compute_sdf,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            structure=structure,
            resolution=resolution,
            cell_size=cell_size,
            bounds=bounds,
            padding=padding,
            padding_relative=padding_relative,
            root_resolution=root_resolution,
            max_depth=max_depth,
            band_cells=band_cells,
            max_cells=max_cells,
            sign=sign,
            location=location,
            band=band,
        )

    @server.tool()
    def distance_to_surface(
        input_path: str,
        surface_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        surface_format: Optional[str] = None,
        output_format: Optional[str] = None,
        sign: str = "pseudonormal",
        location: str = "corner",
        band: float = 0.0,
        record_inside: bool = False,
        record_closest_cell: bool = False,
    ) -> dict:
        """Attach the signed distance from a mesh's points (location='corner') or
        cell centres (location='center') to a surface, as sdf:distance. Negative
        is inside. sign='winding-number' is robust to holes and inconsistent
        winding but costs O(triangles) per query."""
        return _guard(
            _tools.tool_distance_to_surface,
            input_path=input_path,
            surface_path=surface_path,
            output_path=output_path,
            input_format=input_format,
            surface_format=surface_format,
            output_format=output_format,
            sign=sign,
            location=location,
            band=band,
            record_inside=record_inside,
            record_closest_cell=record_closest_cell,
        )

    @server.tool()
    def sample_distance(
        input_path: str,
        points: List[List[float]],
        input_format: Optional[str] = None,
        sign: str = "pseudonormal",
        band: float = 0.0,
    ) -> dict:
        """Signed distances from a list of [x, y, z] points to a surface, without
        writing a mesh. Negative is inside."""
        return _guard(
            _tools.tool_sample_distance,
            input_path=input_path,
            points=points,
            input_format=input_format,
            sign=sign,
            band=band,
        )

    @server.tool()
    def surface_watertight_check(
        input_path: str, input_format: Optional[str] = None
    ) -> dict:
        """Report what is wrong with a surface, in numbers: boundary edges,
        non-manifold edges, inconsistently wound pairs and degenerate triangles.
        A signed distance is only meaningful where these are zero."""
        return _guard(
            _tools.tool_surface_watertight_check,
            input_path=input_path,
            input_format=input_format,
        )

    @server.tool()
    def gradient(
        input_path: str,
        output_path: str,
        array: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        operator: str = "gradient",
        method: str = "green-gauss",
        location: str = "cell",
        output: Optional[str] = None,
        component: Optional[int] = None,
        overwrite: bool = False,
    ) -> dict:
        """Gradient, divergence or curl of a point_data field.

        operator: gradient|divergence|curl. method: green-gauss|least-squares.
        location: cell|point. The result is named <array>:<operator> unless
        `output` overrides it; a gradient of an nc-component field has 3*nc
        components laid out [component][derivative]. Divergence and curl need a
        2- or 3-component field. Cells that cannot be differentiated (below the
        mesh dimension, ragged, or a 3-D Lagrange type) yield NaN and are
        reported in num_skipped; least-squares cells with a degenerate
        neighbourhood fall back to Green-Gauss and are reported in
        num_fallback."""
        return _guard(
            _tools.tool_gradient,
            input_path=input_path,
            output_path=output_path,
            array=array,
            input_format=input_format,
            output_format=output_format,
            operator=operator,
            method=method,
            location=location,
            output=output,
            component=component,
            overwrite=overwrite,
        )

    @server.tool()
    def hessian(
        input_path: str,
        output_path: str,
        array: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "green-gauss",
        location: str = "cell",
        output: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict:
        """Hessian (second derivative) of a scalar point_data field --
        gradient's companion one order further, for curvature-based adaptive
        refinement.

        A composition of TWO gradient calls, not a new numerical kernel: the
        field is differentiated once (point location), then that (n,3)
        gradient is differentiated again with the default gradient operator,
        producing (n,9) -- the flattened row-major 3x3 Hessian. method is
        forwarded to BOTH internal passes. The result is named
        <array>:hessian unless `output` overrides it. Input must have exactly
        one component -- a vector field's Hessian is a separate quantity per
        component, call this once per component. A field that is at most
        linear has an exactly zero Hessian everywhere; for a genuinely
        quadratic field the composition is exact on a structured/symmetric
        mesh away from its own boundary and a good, standard, but genuinely
        approximate curvature estimate on an irregular mesh. Cells that
        cannot be evaluated yield NaN and are reported in num_skipped;
        least-squares cells with a degenerate neighbourhood in either
        internal pass fall back to Green-Gauss and are reported in
        num_fallback. A curvature-driven refinement indicator needs no new
        tool: `norm(...)` in data_calc on the 9-component output is exactly
        its Frobenius norm, ready for refine's where selector."""
        return _guard(
            _tools.tool_hessian,
            input_path=input_path,
            output_path=output_path,
            array=array,
            input_format=input_format,
            output_format=output_format,
            method=method,
            location=location,
            output=output,
            overwrite=overwrite,
        )

    @server.tool()
    def curvature(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        mean: bool = True,
        gaussian: bool = True,
        dual_area: str = "mixed-voronoi",
        include_boundary: bool = False,
        record_area: bool = False,
        record_principal: bool = False,
        region: str = "",
    ) -> dict:
        """Per-vertex mean (H) and Gaussian (K) curvature of a surface mesh --
        the signed distance's natural companion as a node feature: sdf says
        how far a point is from the surface, this says how the surface bends
        there.

        K is the angle defect and H the cotangent Laplace-Beltrami operator,
        the estimators the discrete-differential-geometry convergence results
        are about. Writes curvature:mean and curvature:gaussian, optionally
        curvature:area (the dual area each was divided by) and
        curvature:principal ((n,2), k1 >= k2). Triangles come from the same
        fan convert_cells(simplexify) uses, so a quad mesh's curvature is the
        curvature of its canonical triangulation; a volume or polyhedron block
        is refused by name pointing at extract_surface and a higher-order one
        pointing at linearize. dual_area is "mixed-voronoi" (default, better
        on an irregular tessellation) or "barycentric" (cruder but
        branch-free). Boundary vertices are NaN unless include_boundary, and
        isolated ones always; both are counted. total_angle_defect is the
        oracle: on a CLOSED surface it is 2*pi*chi exactly -- 4*pi for
        anything sphere-like -- whatever the tessellation and whichever dual
        area, so a value that is not that means the input is not closed or the
        result is not sane. H's sign is orientation-dependent and K's is not,
        so check quality.inconsistent_pairs before trusting a sign: a nonzero
        count means facets disagree about which side is out and H is
        sign-flipped in patches. This never repairs its input."""
        return _guard(
            _tools.tool_curvature,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            mean=mean,
            gaussian=gaussian,
            dual_area=dual_area,
            include_boundary=include_boundary,
            record_area=record_area,
            record_principal=record_principal,
            region=region,
        )

    @server.tool()
    def repair(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        fix_orientation: bool = True,
        orient_outward: bool = True,
        fill_holes: bool = True,
        split_non_manifold: bool = True,
        max_hole_edges: int = 10,
        weld_tolerance: float = 0.0,
        record_provenance: bool = False,
    ) -> dict:
        """Repair a surface mesh's orientation, holes and pinched vertices --
        the three defects `clean` does not touch (it welds, drops degenerate
        and duplicate cells, and prunes orphans; it never rewinds a triangle
        or closes a hole).

        The passes run in order: weld (opt-in, `weld_tolerance`) ->
        triangulate (quads and polygons fan exactly as convert_cells
        simplexify does, blocks staying 1:1) -> split bowties (a vertex whose
        triangle star is edge-disconnected is duplicated, geometry unchanged)
        -> orient (a BFS per connected component by the TOPOLOGICAL half-edge
        rule -- two triangles sharing an edge agree iff they traverse it in
        opposite directions, which is exact where a normal-angle test
        mis-orients across a sharp crease; fewest flips wins ties) -> fill
        holes (every traceable boundary loop of at most max_hole_edges edges,
        `<= 0` for no limit, gets one centroid point and one triangle per loop
        edge, wound to AGREE with the surrounding surface) -> orient outward
        (every closed component whose signed volume is negative is flipped
        whole). Lower-dimensional blocks (boundary lines) ride along; fill
        triangles land in one trailing triangle block, added only when there
        is one, so a closed input keeps its block count.

        Reports the input's and the output's defect counts, so what was fixed
        and what remains are both visible. Non-manifold EDGES (used by three or
        more triangles) are neither split nor crossed -- they are counted;
        nested cavities are not detected, so every closed component is
        oriented outward on its own. Point and Cell regions survive (a split
        copy joins its source's); Side regions are dropped, since a flip
        permutes a triangle's edge numbering."""
        return _guard(
            _tools.tool_repair,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            fix_orientation=fix_orientation,
            orient_outward=orient_outward,
            fill_holes=fill_holes,
            split_non_manifold=split_non_manifold,
            max_hole_edges=max_hole_edges,
            weld_tolerance=weld_tolerance,
            record_provenance=record_provenance,
        )

    @server.tool()
    def shrinkwrap(
        input_path: str,
        target_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        target_format: Optional[str] = None,
        output_format: Optional[str] = None,
        offset: float = 0.0,
        max_distance: float = 0.0,
        weights: str = "",
        target_region: str = "",
        normal_weight: str = "angle",
        record_distance: bool = False,
        record_closest_cell: bool = False,
    ) -> dict:
        """Project a mesh's points onto a target triangle surface:
        x' = x + w (p + offset*n - x), with p the closest point on the target
        and n the unit pseudonormal there -- the fitting step a scanned skin, a
        CAD shell or a coarse solve needs before it can be used as a template.

        It is ONE projection, not an iteration: there is no self-intersection
        guard and no inversion guard, because a wrap is a fit and not a
        smoothing. Every point of the input moves whatever cells it carries --
        a volume mesh's interior points are projected too; only the TARGET must
        be a surface (quads and polygons are fanned, a volume or higher-order
        block is refused by name). Use `weights` to select or blend: an
        integer/bool point_data array selects (nonzero moves), a float one
        blends, applied unclamped so a caller can overshoot on purpose. A
        point farther than max_distance (`<= 0` means unlimited) is left where
        it is and counted, as is one whose hit feature has no direction to
        offset along.

        The offset goes along the pseudonormal of the hit FEATURE (face, edge
        or vertex), not of the selected triangle: at a crease the offset
        surface's normal is the bisector, so a face normal would land the
        point off that surface and make the result depend on which of two
        equidistant faces won the tie-break. Connectivity, data, regions and
        property sets pass through verbatim, and the points keep their
        dtype."""
        return _guard(
            _tools.tool_shrinkwrap,
            input_path=input_path,
            target_path=target_path,
            output_path=output_path,
            input_format=input_format,
            target_format=target_format,
            output_format=output_format,
            offset=offset,
            max_distance=max_distance,
            weights=weights,
            target_region=target_region,
            normal_weight=normal_weight,
            record_distance=record_distance,
            record_closest_cell=record_closest_cell,
        )

    @server.tool()
    def sobolev_deform(
        input_path: str,
        output_path: str,
        array: str,
        length_scale: float,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        fixed_points_array: str = "",
        fix_boundary: bool = False,
        record_filtered: bool = False,
        max_iterations: int = 128,
        tolerance: float = 1e-10,
    ) -> dict:
        """Sobolev (Helmholtz-filtered) deformation: smooth a raw per-point
        displacement field through the mesh's own P1 finite-element operators,
        then move the points by the smoothed field.

        Solves (M + l^2 K) u = M d per ambient component and sets x' = x + u,
        with K the P1 stiffness matrix assembled from the simplex edge Gram
        matrix, M a uniform vertex mass and l = length_scale -- a
        screened-Poisson low-pass filter whose cutoff wavelength is l, which
        is what turns a jagged per-node displacement (a shape gradient, a
        scattered measurement, a model's raw output) into one a mesh can
        follow without tangling. length_scale = 0 applies the raw field at the
        free points.

        `array` names the (n, dim) point_data displacement. Every cell block
        at the mesh's top topological dimension must be a linear simplex --
        line, triangle or tetra -- since that is what the assembly is defined
        on; a quadratic block is refused naming linearize, anything else
        naming convert_cells(simplexify). Lower-dimensional blocks ride along,
        and a point in no top-dimensional cell receives its raw displacement.

        Nothing is pinned by default: an unfixed boundary carries the natural
        Neumann condition, so a constant displacement is preserved exactly.
        fixed_points_array (an integer/bool point_data mask) and fix_boundary
        impose zero-Dirichlet rows instead. Non-convergence within
        max_iterations is reported through `converged` with the last iterate
        returned -- a partially smoothed field is still usable, and `residual`
        says how far it got. This is a pure coordinate move: connectivity,
        every data array, regions and property sets pass through."""
        return _guard(
            _tools.tool_sobolev_deform,
            input_path=input_path,
            output_path=output_path,
            array=array,
            length_scale=length_scale,
            input_format=input_format,
            output_format=output_format,
            fixed_points_array=fixed_points_array,
            fix_boundary=fix_boundary,
            record_filtered=record_filtered,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    @server.tool()
    def estimate_error(
        input_path: str,
        output_path: str,
        array: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "zz",
        marking: str = "none",
        marking_value: float = 0.0,
        output: Optional[str] = None,
        marked: Optional[str] = None,
        overwrite: bool = False,
    ) -> dict:
        """ZZ recovery-based error indicator of a point_data field, plus
        optional marking of cells for refinement.

        A composition of `gradient` (Green-Gauss, cell location) with the
        measure-weighted point<->cell averaging round trip: the indicator is
        sqrt(|measure| * sum((recovered - raw)^2)) per cell, attached as
        `output` (default "error:zz"). marking: none|absolute|fraction|dorfler;
        when not "none" a second Int64 0/1 array `marked` (default
        "error:marked") is attached too, so refine's own `where` selector
        needs no change at all -- the intended use is
        `refine(..., where="error:marked > 0.5")`. marking_value's meaning
        depends on marking: an absolute indicator threshold, a fraction in
        (0, 1] of cells, or the Doerfler bulk fraction theta in (0, 1].
        Cells that cannot be evaluated read NaN in the indicator array and 0
        (never NaN) in the marking array, and are reported in num_skipped
        (excluded from global_error and from num_marked)."""
        return _guard(
            _tools.tool_estimate_error,
            input_path=input_path,
            output_path=output_path,
            array=array,
            input_format=input_format,
            output_format=output_format,
            method=method,
            marking=marking,
            marking_value=marking_value,
            output=output,
            marked=marked,
            overwrite=overwrite,
        )

    @server.tool()
    def transform(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        translate: Optional[List[float]] = None,
        scale: Optional[Union[List[float], float]] = None,
        rotate_axis: Optional[List[float]] = None,
        rotate_degrees: Optional[float] = None,
        matrix: Optional[List[List[float]]] = None,
        scale_units: Optional[float] = None,
        rotate_vector_data: bool = False,
    ) -> dict:
        """Affine-transform point coordinates: translate [dx,dy,dz], scale
        (scalar or [sx,sy,sz]), rotate (axis + degrees), an explicit 4x4
        matrix, or a uniform unit-scale factor. rotate_vector_data also
        rotates 3-vector/9-tensor point data."""
        return _guard(
            _tools.tool_transform,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            translate=translate,
            scale=scale,
            rotate_axis=rotate_axis,
            rotate_degrees=rotate_degrees,
            matrix=matrix,
            scale_units=scale_units,
            rotate_vector_data=rotate_vector_data,
        )

    @server.tool()
    def convert_cells(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        mode: str = "linearize",
        record_parent_ids: bool = False,
    ) -> dict:
        """Convert the element representation: linearize (drop higher-order
        nodes), simplexify (split into triangles/tets) or elevate (linear ->
        quadratic)."""
        return _guard(
            _tools.tool_convert_cells,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            mode=mode,
            record_parent_ids=record_parent_ids,
        )

    @server.tool()
    def subdivide(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        record_parent_ids: bool = False,
    ) -> dict:
        """Polyhedrally refine: split every eligible 3D cell into one
        polyhedral child per face, connected to a new interior point. No
        per-type template table is needed -- it handles tabulated types and
        existing polyhedron blocks uniformly. Automatically conforming."""
        return _guard(
            _tools.tool_subdivide,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            record_parent_ids=record_parent_ids,
        )

    @server.tool()
    def agglomerate(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        target_group_size: int = 8,
    ) -> dict:
        """Polyhedrally coarsen: merge groups of cells into single larger
        polyhedral cells via greedy seed-and-grow over the shared-face dual.
        Non-volume blocks pass through unchanged; points are never pruned or
        renumbered (clean with remove_orphans=True is the follow-up for
        that)."""
        return _guard(
            _tools.tool_agglomerate,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            target_group_size=target_group_size,
        )

    @server.tool()
    def refine(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        levels: int = 1,
        record_parent_ids: bool = False,
        cells: Optional[List[int]] = None,
        region: Optional[str] = None,
        where: Optional[str] = None,
        closure: str = "redgreen",
        record_levels: bool = False,
        record_hierarchy: bool = False,
    ) -> dict:
        """Refine: subdivide cells into congruent same-type children, `levels`
        times. With no selector every cell is refined; give at most one of
        `cells` (global block-major indices), `region` (a cell region selects its
        cells, a point region every cell touching it) or `where` (a threshold on
        a scalar cell_data array, e.g. "quality:scaled_jacobian < 0.3") and only
        those are, with the resulting hanging nodes resolved by `closure` —
        "redgreen" keeps that local, "propagate" reaches the whole connected
        component, and "balanced" keeps the hanging nodes and only enforces 2:1
        balance (the output is then NOT conforming; the constrained nodes are
        reported in refine:hanging). `record_levels` attaches refine:level.
        `record_hierarchy` attaches refine:cell_id/refine:parent_id -- the
        persistent parent/child hierarchy a multigrid caller resolves across
        the sequence of meshes it keeps; also forces refine:entity to be
        attached even when the closure leaves no hanging node."""
        return _guard(
            _tools.tool_refine,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            levels=levels,
            record_parent_ids=record_parent_ids,
            cells=cells,
            region=region,
            where=where,
            closure=closure,
            record_levels=record_levels,
            record_hierarchy=record_hierarchy,
        )

    @server.tool()
    def undo_green(
        coarse_path: str,
        fine_path: str,
        output_path: str,
        coarse_format: Optional[str] = None,
        fine_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> dict:
        """Restore fine's transitional (green) cells to their original
        parent, read verbatim from coarse (the mesh a prior refine() call
        with record_hierarchy=True, record_levels=True was run on) --
        undoes refine's known quality-degradation issue with repeated
        selective passes over the same region. Reports
        num_groups_undone/num_cells_removed."""
        return _guard(
            _tools.tool_undo_green,
            coarse_path=coarse_path,
            fine_path=fine_path,
            output_path=output_path,
            coarse_format=coarse_format,
            fine_format=fine_format,
            output_format=output_format,
        )

    @server.tool()
    def decimate(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        ratio: Optional[float] = None,
        target_faces: Optional[int] = None,
        max_error: Optional[float] = None,
        placement: str = "optimal",
        preserve_boundary: bool = True,
        preserve_features: bool = True,
        feature_angle: float = 30.0,
    ) -> dict:
        """Decimate a surface mesh by quadric edge collapse. Give exactly one
        stopping criterion: ratio (fraction of faces to KEEP), target_faces,
        or max_error. Reports faces/points removed and rejections."""
        return _guard(
            _tools.tool_decimate,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            ratio=ratio,
            target_faces=target_faces,
            max_error=max_error,
            placement=placement,
            preserve_boundary=preserve_boundary,
            preserve_features=preserve_features,
            feature_angle=feature_angle,
        )

    @server.tool()
    def decimate_volume(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        ratio: Optional[float] = None,
        target_cells: Optional[int] = None,
        max_error: Optional[float] = None,
        placement: str = "optimal",
        preserve_boundary: bool = False,
        preserve_features: bool = True,
        feature_angle: float = 30.0,
    ) -> dict:
        """Decimate a tetrahedral mesh by quadric-error tet-edge collapse. Give
        exactly one stopping criterion: ratio (fraction of tets to KEEP),
        target_cells, or max_error. Boundary vertices participate by default
        (preserve_boundary=False, unlike surface decimate). Reports
        tets/points removed and rejections."""
        return _guard(
            _tools.tool_decimate_volume,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            ratio=ratio,
            target_cells=target_cells,
            max_error=max_error,
            placement=placement,
            preserve_boundary=preserve_boundary,
            preserve_features=preserve_features,
            feature_angle=feature_angle,
        )

    @server.tool()
    def remesh(
        input_path: str,
        output_path: str,
        num_clusters: int,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        subdivide: Optional[int] = None,
        subsample_ratio: float = 10.0,
        max_subdivide: int = 4,
        max_iterations: int = 100,
        max_repair_passes: int = 10,
        metric: str = "isotropic",
        gradation: float = 0.0,
        preserve_boundary: bool = True,
        max_anisotropy: float = 4.0,
    ) -> dict:
        """Replace a surface mesh's triangulation with a new,
        near-uniformly-sized, well-shaped one at num_clusters vertices
        (approximated centroidal Voronoi diagram clustering). Unlike every
        other resolution-changing tool, the output has NO correspondence to
        the input -- point_data/cell_data/named regions are dropped,
        field_data is carried; use interpolate/conservative_interpolate to
        transfer a field onto the result. metric is "isotropic" (default,
        fast, rounds sharp features), "quadric" (Garland-Heckbert error,
        preserves sharp edges/corners), or "anisotropic" (clusters shaped by
        a local curvature tensor, elongated along low-curvature directions --
        see max_anisotropy). gradation is the curvature-gradation exponent
        gamma in the item weight area * kappa**gamma (0.0 default disables
        gradation). preserve_boundary (default True) seeds and pins the
        input's open boundary and emits a line dual cell along it, a no-op
        on a closed mesh. max_anisotropy (default 4.0) is, under
        metric="anisotropic", the maximum ratio between the two in-plane
        target edge lengths a curvature tensor may request; 1.0 recovers
        the isotropic shape; an error to set away from the default under
        any other metric."""
        return _guard(
            _tools.tool_remesh,
            input_path=input_path,
            output_path=output_path,
            num_clusters=num_clusters,
            input_format=input_format,
            output_format=output_format,
            subdivide=subdivide,
            subsample_ratio=subsample_ratio,
            max_subdivide=max_subdivide,
            max_iterations=max_iterations,
            max_repair_passes=max_repair_passes,
            metric=metric,
            max_anisotropy=max_anisotropy,
            gradation=gradation,
            preserve_boundary=preserve_boundary,
        )

    @server.tool()
    def remesh_volume(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        resolution: Optional[List[int]] = None,
        cell_size: Optional[float] = None,
        bounds: Optional[List[float]] = None,
        padding: float = 0.0,
        padding_relative: float = 0.1,
        max_cells: int = 20000000,
        max_tets: int = 20000000,
        warp_fraction: float = 0.35,
        watertight_check: str = "warn",
    ) -> dict:
        """Retetrahedralize a volume mesh (or closed surface) at a chosen
        resolution by isosurface stuffing -- the volumetric sibling of remesh.
        Give exactly one of resolution (nx, ny, nz) or cell_size. Like remesh,
        the output has NO correspondence to the input -- point_data/cell_data/
        named regions are dropped, field_data is carried. warp_fraction (0-1,
        default 0.35) trades boundary tet quality for a small, measured chance
        of non-manifold boundary edges (reported as num_non_manifold_edges);
        0 gives an exactly watertight but lower-quality boundary."""
        return _guard(
            _tools.tool_remesh_volume,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            resolution=resolution,
            cell_size=cell_size,
            bounds=bounds,
            padding=padding,
            padding_relative=padding_relative,
            max_cells=max_cells,
            max_tets=max_tets,
            warp_fraction=warp_fraction,
            watertight_check=watertight_check,
        )

    @server.tool()
    def optimize_volume(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        max_iterations: int = 10,
        relocate: bool = True,
        flip: bool = True,
        preserve_boundary: bool = True,
        min_improvement: float = 1e-6,
    ) -> dict:
        """ODT-remesh a tetrahedral mesh: raise its worst element quality by
        relocating vertices AND flipping connectivity (2-3/3-2, predicate-free).
        The genuine "ODT remeshing" sibling of remesh_volume (which generates a
        fresh lattice mesh) and of smooth method="odt" (which only moves points
        on fixed connectivity). Tet-only and C++-core-only. The point set is
        invariant, so point_data and named Point regions carry; cell_data and
        Cell/Side regions are dropped. With preserve_boundary the boundary
        surface is exactly preserved."""
        return _guard(
            _tools.tool_optimize_volume,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            max_iterations=max_iterations,
            relocate=relocate,
            flip=flip,
            preserve_boundary=preserve_boundary,
            min_improvement=min_improvement,
        )

    @server.tool()
    def smooth(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "taubin",
        iterations: int = 10,
        lambda_factor: float = -1.0,
        mu: float = -0.34,
        fix_boundary: bool = True,
        preserve_features: bool = True,
        feature_angle: float = 30.0,
        guard_inversion: bool = True,
    ) -> dict:
        """Smooth point coordinates (taubin, the feature-preserving default;
        laplacian; or odt, optimal-Delaunay-triangulation smoothing, tet-only
        and C++-core-only with no pure-Python fallback). Pure coordinate move;
        boundary and feature nodes pinned by default. lambda_factor < 0 means
        the method's own default."""
        return _guard(
            _tools.tool_smooth,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            method=method,
            iterations=iterations,
            lambda_factor=lambda_factor,
            mu=mu,
            fix_boundary=fix_boundary,
            preserve_features=preserve_features,
            feature_angle=feature_angle,
            guard_inversion=guard_inversion,
        )

    @server.tool()
    def merge(
        input_paths: List[str],
        output_path: str,
        output_format: Optional[str] = None,
        weld: bool = False,
        atol: float = 1e-8,
        source_tag: bool = True,
        data_policy: str = "intersection",
        drop_duplicate_cells: bool = False,
    ) -> dict:
        """Merge two or more mesh files into one; weld=true fuses coincident
        points within atol. data_policy: intersection | fill."""
        return _guard(
            _tools.tool_merge,
            input_paths=input_paths,
            output_path=output_path,
            output_format=output_format,
            weld=weld,
            atol=atol,
            source_tag=source_tag,
            data_policy=data_policy,
            drop_duplicate_cells=drop_duplicate_cells,
        )

    @server.tool()
    def split(
        input_path: str,
        input_format: Optional[str] = None,
        by: str = "type",
        tag: Optional[str] = None,
        output_dir: Optional[str] = None,
        name_template: str = "{stem}_{key}.vtu",
    ) -> dict:
        """Split into submeshes (by: type | component | tag | region | regions;
        tag names an integer cell-data array). Writes one file per piece into
        output_dir (default: the input's directory) and returns their paths."""
        return _guard(
            _tools.tool_split,
            input_path=input_path,
            input_format=input_format,
            by=by,
            tag=tag,
            output_dir=output_dir,
            name_template=name_template,
        )

    @server.tool()
    def partition(
        input_path: str,
        nparts: int,
        input_format: Optional[str] = None,
        method: str = "auto",
        imbalance: float = 0.03,
        mode: str = "eco",
        seed: int = 0,
        record_ids: bool = False,
        ghost_layers: int = 0,
        weights: Optional[str] = None,
        output_dir: Optional[str] = None,
        name_template: str = "{stem}_part{part}.vtu",
    ) -> dict:
        """Partition into exactly nparts balanced pieces (method: auto | sfc |
        kahip). Writes one file per part and returns their paths."""
        return _guard(
            _tools.tool_partition,
            input_path=input_path,
            nparts=nparts,
            input_format=input_format,
            method=method,
            imbalance=imbalance,
            mode=mode,
            seed=seed,
            record_ids=record_ids,
            ghost_layers=ghost_layers,
            weights=weights,
            output_dir=output_dir,
            name_template=name_template,
        )

    @server.tool()
    def interpolate(
        source_path: str,
        target_path: str,
        output_path: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        output_format: Optional[str] = None,
        method: str = "nearest",
        arrays: Optional[List[str]] = None,
        extrapolate: bool = False,
        default_value: float = 0.0,
        on_conflict: str = "error",
    ) -> dict:
        """Sample the source mesh's data arrays onto the target mesh's
        geometry (method: nearest | barycentric); writes the target copy with
        the sampled arrays attached."""
        return _guard(
            _tools.tool_interpolate,
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            source_format=source_format,
            target_format=target_format,
            output_format=output_format,
            method=method,
            arrays=arrays,
            extrapolate=extrapolate,
            default_value=default_value,
            on_conflict=on_conflict,
        )

    @server.tool()
    def conservative_interpolate(
        source_path: str,
        target_path: str,
        output_path: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        output_format: Optional[str] = None,
        arrays: Optional[List[str]] = None,
        default_value: float = 0.0,
        on_conflict: str = "error",
    ) -> dict:
        """Mass-preservingly (overlap-measure weighted) sample the source
        mesh's data arrays onto the target mesh's geometry; unlike
        interpolate, conserves sum(value * measure) over the shared region."""
        return _guard(
            _tools.tool_conservative_interpolate,
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            source_format=source_format,
            target_format=target_format,
            output_format=output_format,
            arrays=arrays,
            default_value=default_value,
            on_conflict=on_conflict,
        )


# --------------------------------------------------------------------------- #
# Data operations                                                             #
# --------------------------------------------------------------------------- #
def _register_data(server: FastMCP) -> None:
    @server.tool()
    def data_manage(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        keep: Optional[List[List[str]]] = None,
        drop: Optional[List[List[str]]] = None,
        rename: Optional[List[List[str]]] = None,
        ignore_missing: bool = False,
    ) -> dict:
        """Keep/drop/rename data arrays. keep/drop: [[location, name], ...];
        rename: [[location, old, new], ...] (location: point | cell | field).
        Applied keep -> drop -> rename."""
        return _guard(
            _tools.tool_data_manage,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            keep=keep,
            drop=drop,
            rename=rename,
            ignore_missing=ignore_missing,
        )

    @server.tool()
    def data_convert(
        input_path: str,
        output_path: str,
        direction: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        arrays: Optional[List[str]] = None,
        weighted: bool = False,
        nan_policy: str = "ignore",
        nan_replacement: float = 0.0,
    ) -> dict:
        """Average data arrays between locations (direction: point_to_cell |
        cell_to_point; weighted uses cell measures for cell_to_point)."""
        return _guard(
            _tools.tool_data_convert,
            input_path=input_path,
            output_path=output_path,
            direction=direction,
            input_format=input_format,
            output_format=output_format,
            arrays=arrays,
            weighted=weighted,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement,
        )

    @server.tool()
    def data_calc(
        input_path: str,
        output_path: str,
        expression: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        location: str = "point",
        output_name: str = "",
        overwrite: bool = False,
    ) -> dict:
        """Evaluate an arithmetic expression over data arrays into a new array
        (operators + - * /, functions abs/sqrt/min/max/norm; back-tick-quote
        names with spaces, e.g. "speed = norm(velocity)")."""
        return _guard(
            _tools.tool_data_calc,
            input_path=input_path,
            output_path=output_path,
            expression=expression,
            input_format=input_format,
            output_format=output_format,
            location=location,
            output_name=output_name,
            overwrite=overwrite,
        )

    @server.tool()
    def data_condition(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        operation: str = "clamp",
        location: str = "point",
        arrays: Optional[List[str]] = None,
        scope: str = "component",
        lo: float = 0.0,
        hi: float = 1.0,
        nan_policy: str = "ignore",
        nan_replacement: float = 0.0,
        suffix: str = "",
        preserve_dtype: bool = True,
    ) -> dict:
        """Condition data values: clamp to [lo, hi], normalize into [lo, hi],
        or standardize (zero mean, unit std). scope: component | magnitude."""
        return _guard(
            _tools.tool_data_condition,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            output_format=output_format,
            operation=operation,
            location=location,
            arrays=arrays,
            scope=scope,
            lo=lo,
            hi=hi,
            nan_policy=nan_policy,
            nan_replacement=nan_replacement,
            suffix=suffix,
            preserve_dtype=preserve_dtype,
        )


# --------------------------------------------------------------------------- #
# Dataset manifests (doc/datasets.md)                                         #
# --------------------------------------------------------------------------- #
def _register_dataset(server: FastMCP) -> None:
    @server.tool()
    def dataset_add(
        manifest_path: str,
        input_pattern: Optional[str] = None,
        input_paths: Optional[List[str]] = None,
        entry_id: Optional[str] = None,
        input_format: Optional[str] = None,
        time_from: Optional[str] = None,
        times: Optional[List[float]] = None,
        sort: bool = False,
        target_pattern: Optional[str] = None,
        target_paths: Optional[List[str]] = None,
        target_format: Optional[str] = None,
        target_time_from: Optional[str] = None,
        target_times: Optional[List[float]] = None,
        target_sort: bool = False,
        split: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Add a case to a dataset manifest JSON (created if absent). Give
        exactly one of input_pattern (a glob) or input_paths; the source is
        validated now and stored relative to the manifest's directory.
        The optional target_* family records a paired coarse/fine series for a
        superresolution dataset: the two must have the same number of steps at
        the same instants, checked here. Leave it out for the ordinary case --
        an entry without a target is self-supervised, one mesh supplying both
        sides. Optional curation: split, tags, group, notes, metadata."""
        return _guard(
            _tools.tool_dataset_add,
            manifest_path=manifest_path,
            input_pattern=input_pattern,
            input_paths=input_paths,
            entry_id=entry_id,
            input_format=input_format,
            time_from=time_from,
            times=times,
            sort=sort,
            target_pattern=target_pattern,
            target_paths=target_paths,
            target_format=target_format,
            target_time_from=target_time_from,
            target_times=target_times,
            target_sort=target_sort,
            split=split,
            tags=tags,
            group=group,
            notes=notes,
            metadata=metadata,
        )

    @server.tool()
    def dataset_list(
        manifest_path: str,
        split: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        resolve: bool = False,
    ) -> dict:
        """List a dataset manifest's entries, optionally filtered by split /
        tags (must carry all) / group (path or descendant). resolve=true also
        expands each entry's file/step/time plan (no mesh is read)."""
        return _guard(
            _tools.tool_dataset_list,
            manifest_path=manifest_path,
            split=split,
            tags=tags,
            group=group,
            resolve=resolve,
        )

    @server.tool()
    def dataset_update(
        manifest_path: str,
        entry_ids: Optional[List[str]] = None,
        all_entries: bool = False,
        split: Optional[str] = None,
        assign_splits: Optional[dict] = None,
        seed: int = 0,
        by_group: bool = False,
        add_tags: Optional[List[str]] = None,
        remove_tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Optional[dict] = None,
        drop_metadata: Optional[List[str]] = None,
    ) -> dict:
        """Curate a dataset manifest: set a split on selected entries, assign
        splits by fractions (assign_splits={"train": 0.8, ...}, deterministic
        via seed; by_group keeps groups together), add/remove tags, or set one
        entry's group/notes/metadata."""
        return _guard(
            _tools.tool_dataset_update,
            manifest_path=manifest_path,
            entry_ids=entry_ids,
            all_entries=all_entries,
            split=split,
            assign_splits=assign_splits,
            seed=seed,
            by_group=by_group,
            add_tags=add_tags,
            remove_tags=remove_tags,
            group=group,
            notes=notes,
            metadata=metadata,
            drop_metadata=drop_metadata,
        )

    @server.tool()
    def dataset_find(root_dir: str = ".", max_depth: int = 2) -> dict:
        """Find dataset manifests: every *.json at most max_depth levels
        below root_dir that parses as a DatasetManifest, with its name,
        entry count, splits, modification time and SHA-256 content hash."""
        return _guard(_tools.tool_dataset_find, root_dir=root_dir, max_depth=max_depth)

    @server.tool()
    def dataset_health(
        manifest_path: str,
        split: Optional[str] = None,
        entry_ids: Optional[List[str]] = None,
        quality: bool = True,
        all_steps: bool = False,
    ) -> dict:
        """Scan a dataset manifest's entries (optionally one split / given
        ids) and report their health: per entry the step count, NaN/Inf
        counts over data arrays, inverted/degenerate cells, worst scaled
        Jacobian and arrays present; per manifest the split balance, totals,
        fields missing across entries and the bad entries. Reads one mesh at
        a time (step 0, or every step with all_steps)."""
        return _guard(
            _tools.tool_dataset_health,
            manifest_path=manifest_path,
            split=split,
            entry_ids=entry_ids,
            quality=quality,
            all_steps=all_steps,
        )


# --------------------------------------------------------------------------- #
# Training jobs (doc/dashboard.md)                                            #
# --------------------------------------------------------------------------- #
def _register_training(server: FastMCP) -> None:
    @server.tool()
    def train_defaults(
        manifest_path: str,
        fields: Optional[List[str]] = None,
        target_fields: Optional[List[str]] = None,
    ) -> dict:
        """What a training launch needs: the point/cell data arrays the
        manifest's first entry carries, its splits, whether the frameworks
        are installed, and a complete default training spec."""
        return _guard(
            _tools.tool_train_defaults,
            manifest_path=manifest_path,
            fields=fields,
            target_fields=target_fields,
        )

    @server.tool()
    def train_start(
        manifest_path: str,
        fields: List[str],
        target_fields: List[str],
        train_split: str = "train",
        valid_split: str = "valid",
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        seed: int = 0,
        model_name: str = "meshgraphnet",
        processor_size: int = 8,
        hidden_dim: int = 64,
        aggregation: str = "sum",
        scaling_factor: int = 2,
        conv_layer_size: int = 32,
        resid_blocks: int = 8,
        resolution: Optional[List[int]] = None,
        cell_size: Optional[float] = None,
        bounds: Optional[List[float]] = None,
        padding: float = 0.0,
        padding_relative: float = 0.0,
        extrapolate: bool = False,
        fill_value: float = 0.0,
        regions: bool = False,
        kind: str = "node",
        undirected: bool = True,
        edge_features: bool = True,
        float32: bool = True,
        target_offset: int = 0,
        target_delta: bool = False,
        checkpoint_every: int = 10,
        device: str = "auto",
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> dict:
        """Start training PhysicsNeMo's MeshGraphNet on a manifest split
        (fields -> target_fields) as a background job under the runs
        directory; returns the job id and initial status. Needs
        torch_geometric + nvidia-physicsnemo (no pip extra); a missing
        framework is a named error."""
        return _guard(
            _tools.tool_train_start,
            manifest_path=manifest_path,
            fields=fields,
            target_fields=target_fields,
            train_split=train_split,
            valid_split=valid_split,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            model_name=model_name,
            processor_size=processor_size,
            hidden_dim=hidden_dim,
            aggregation=aggregation,
            scaling_factor=scaling_factor,
            conv_layer_size=conv_layer_size,
            resid_blocks=resid_blocks,
            resolution=resolution,
            cell_size=cell_size,
            bounds=bounds,
            padding=padding,
            padding_relative=padding_relative,
            extrapolate=extrapolate,
            fill_value=fill_value,
            regions=regions,
            kind=kind,
            undirected=undirected,
            edge_features=edge_features,
            float32=float32,
            target_offset=target_offset,
            target_delta=target_delta,
            checkpoint_every=checkpoint_every,
            device=device,
            notes=notes,
            tags=tags,
        )

    @server.tool()
    def train_status(job_id: str) -> dict:
        """A training job's status (running/finished/failed/stopped), epoch
        progress, best validation loss, ETA and last metrics row."""
        return _guard(_tools.tool_train_status, job_id=job_id)

    @server.tool()
    def train_list(
        status: Optional[str] = None, manifest_path: Optional[str] = None
    ) -> dict:
        """Every training job (newest first) with its hyperparameters and
        final/best losses; optionally filtered by status or manifest."""
        return _guard(
            _tools.tool_train_list, status=status, manifest_path=manifest_path
        )

    @server.tool()
    def train_stop(job_id: str, grace_seconds: float = 10.0) -> dict:
        """Stop a job: SIGTERM lets the trainer finish its epoch and write
        final.mdlus; SIGKILL after grace_seconds."""
        return _guard(
            _tools.tool_train_stop, job_id=job_id, grace_seconds=grace_seconds
        )

    @server.tool()
    def train_log(job_id: str, offset: int = 0, max_bytes: int = 65536) -> dict:
        """A window of a job's stdout/stderr from a byte offset; poll with
        next_offset to tail it."""
        return _guard(
            _tools.tool_train_log, job_id=job_id, offset=offset, max_bytes=max_bytes
        )

    @server.tool()
    def train_metrics(job_id: str, since_epoch: int = 0) -> dict:
        """A job's per-epoch metrics rows (train/valid loss, lr, timing)."""
        return _guard(_tools.tool_train_metrics, job_id=job_id, since_epoch=since_epoch)

    @server.tool()
    def train_checkpoints(job_id: str) -> dict:
        """A job's .mdlus checkpoints (periodic/best/final) with epoch,
        validation loss, size and which is marked best."""
        return _guard(_tools.tool_train_checkpoints, job_id=job_id)

    @server.tool()
    def train_mark_best(job_id: str, checkpoint: str) -> dict:
        """Mark one of a job's checkpoints as best (copied to best.mdlus)."""
        return _guard(_tools.tool_train_mark_best, job_id=job_id, checkpoint=checkpoint)

    @server.tool()
    def train_predict(
        manifest_path: str,
        job_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
        entry_ids: Optional[List[str]] = None,
        split: Optional[str] = "test",
        step: int = 0,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Predict over a manifest split with a job's (best or named)
        checkpoint or an explicit .mdlus, writing <column>_pred /
        <column>_error data arrays into output_dir/<entry_id>.vtu; returns
        per-entry RMSE. Needs the frameworks."""
        return _guard(
            _tools.tool_train_predict,
            manifest_path=manifest_path,
            job_id=job_id,
            checkpoint=checkpoint,
            entry_ids=entry_ids,
            split=split,
            step=step,
            output_dir=output_dir,
        )

    @server.tool()
    def guard_fit(
        manifest_path: str,
        output_path: str,
        split: Optional[str] = "train",
        margin: float = 1.5,
        quality: bool = True,
    ) -> dict:
        """Fit a geometry guardrail over a manifest split and write it as JSON.

        A trained surrogate answers any mesh it is given, and the answer for a
        part unlike anything it saw is finite, plausible and wrong. This
        describes the shapes in a split — extents, centroid, area, volume,
        counts, surface area and quality summaries — so a new mesh can be
        scored against them. The descriptors are deliberately NOT invariant:
        a scaled part is a different part, and a model trained on brackets 10
        cm across has learnt physics at that scale. The threshold is margin
        times the worst training score."""
        return _guard(
            _tools.tool_guard_fit,
            manifest_path=manifest_path,
            output_path=output_path,
            split=split,
            margin=margin,
            quality=quality,
        )

    @server.tool()
    def guard_check(
        input_path: str,
        guard_path: Optional[str] = None,
        input_format: Optional[str] = None,
        top: int = 3,
    ) -> dict:
        """Describe one mesh's shape, and score it against a guardrail if given.

        Returns the raw descriptors always; with guard_path (a fitted guard, or
        a model card carrying one) it adds the score, the threshold, a verdict
        of 'in' or 'out', and the top descriptors that put it there — so a flag
        is actionable rather than a bare number. Advisory: nothing is refused."""
        return _guard(
            _tools.tool_guard_check,
            input_path=input_path,
            guard_path=guard_path,
            input_format=input_format,
            top=top,
        )

    @server.tool()
    def predict_file(
        checkpoint: str,
        input_path: str,
        output_path: str,
        time_step: Optional[int] = None,
        target_path: Optional[str] = None,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        device: str = "auto",
    ) -> dict:
        """Predict with a trained .mdlus checkpoint on ONE mesh file — no
        manifest, no split, no entry, for a mesh that was never catalogued.
        Everything the prediction needs comes from the checkpoint's own model
        card: which model family wrote it, the sample options, the read
        options, the column contract and the normalization. time_step picks a
        step of a multi-step input; target_path supplies the paired mesh a
        t->t+n or coarse/fine checkpoint compares against. A file carrying no
        truth predicts anyway, with rmse/max_error reported as null rather
        than measured against itself. Needs the frameworks."""
        return _guard(
            _tools.tool_predict_file,
            checkpoint=checkpoint,
            input_path=input_path,
            output_path=output_path,
            time_step=time_step,
            target_path=target_path,
            input_format=input_format,
            output_format=output_format,
            device=device,
        )


# --------------------------------------------------------------------------- #
# Gated tools (optional extras)                                               #
# --------------------------------------------------------------------------- #
def _register_gated(server: FastMCP) -> None:
    @server.tool()
    def data_export(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        location: str = "point",
    ) -> dict:
        """Export data arrays to a Parquet table (location: point | cell).
        Needs the [arrow] extra; a missing install returns a named error."""
        return _guard(
            _tools.tool_data_export,
            input_path=input_path,
            output_path=output_path,
            input_format=input_format,
            location=location,
        )

    @server.tool()
    def export_dataset(
        input_pattern: Optional[str] = None,
        input_paths: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        input_format: Optional[str] = None,
        location: str = "point",
        dataset_format: str = "parquet",
        mesh_id: str = "stem",
    ) -> dict:
        """Export a SET of meshes as one dataset keyed by mesh_id — a
        hive-partitioned Parquet directory (dataset_format: parquet, needs the
        [arrow] extra) or chunked zarr/hdf5 groups ([zarr] / h5py). Give
        exactly one of input_pattern (a glob) or input_paths; mesh_id: stem |
        index. Returns the dataset manifest."""
        return _guard(
            _tools.tool_export_dataset,
            input_pattern=input_pattern,
            input_paths=input_paths,
            output_path=output_path,
            input_format=input_format,
            location=location,
            dataset_format=dataset_format,
            mesh_id=mesh_id,
        )

    @server.tool()
    def export_cae(
        output_dir: str,
        input_pattern: Optional[str] = None,
        input_paths: Optional[List[str]] = None,
        input_format: Optional[str] = None,
        surface_fields: Optional[List[str]] = None,
        volume_fields: Optional[List[str]] = None,
        global_params: Optional[Dict[str, float]] = None,
        global_params_reference: Optional[Dict[str, float]] = None,
        global_params_order: Optional[List[str]] = None,
        name_template: str = "case_{index}.npz",
    ) -> dict:
        """Export a SET of meshes as one .npz per case in the CAE sample
        layout PhysicsNeMo's DoMINO/Transolver datapipes read: the
        triangulated skin with normals and areas, the volume's nodes, the
        named field blocks and the case's global parameters. Give exactly one
        of input_pattern (a glob) or input_paths. Returns the files written
        and the keys of the first one."""
        return _guard(
            _tools.tool_export_cae,
            output_dir=output_dir,
            input_pattern=input_pattern,
            input_paths=input_paths,
            input_format=input_format,
            surface_fields=surface_fields,
            volume_fields=volume_fields,
            global_params=global_params,
            global_params_reference=global_params_reference,
            global_params_order=global_params_order,
            name_template=name_template,
        )

    @server.tool()
    def screenshot(
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        color_by: Optional[str] = None,
        width: int = 1280,
        height: int = 960,
        transparent: bool = False,
    ):
        """Render an off-screen PNG screenshot of the mesh (optionally colored
        by a data array). Needs the [viewer] extra (polyscope); a missing
        install returns a named error."""
        try:
            report = _tools.tool_screenshot(
                input_path=input_path,
                output_path=output_path,
                input_format=input_format,
                color_by=color_by,
                width=width,
                height=height,
                transparent=transparent,
            )
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "error_type": type(e).__name__}
        return [Image(path=report["output_path"]), report]


# --------------------------------------------------------------------------- #
# Resources                                                                   #
# --------------------------------------------------------------------------- #
def _register_resources(server: FastMCP) -> None:
    @server.resource("meshioplusplus://formats")
    def formats_resource() -> str:
        """The supported-formats registry as JSON."""
        return json.dumps(_tools.formats_payload(), indent=2)

    @server.resource("meshioplusplus://version")
    def version_resource() -> str:
        """The installed meshio++ version."""
        return __version__


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="meshioplusplus-mcp",
        description="meshio++ MCP server (stdio transport).",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("MESHIOPLUSPLUS_MCP_ROOT") or None,
        help=(
            "confine every tool's input/output paths to this directory "
            "(default: the MESHIOPLUSPLUS_MCP_ROOT env var, else unrestricted)"
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"meshioplusplus {__version__}"
    )
    http = parser.add_argument_group(
        "HTTP front-end",
        "serve the browser dataset manager's JSON API and MCP over HTTP from "
        "this process instead of stdio (needs the [dashboard] extra; "
        "doc/dashboard.md)",
    )
    http.add_argument("--http", action="store_true", help="serve over HTTP")
    http.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    http.add_argument("--port", type=int, default=None, help="port (default 8765)")
    http.add_argument(
        "--token",
        default=None,
        help="the bearer token to require (default: a fresh random one, printed)",
    )
    http.add_argument(
        "--no-token",
        action="store_true",
        help="require no token (only on a machine you alone use)",
    )
    http.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="an extra browser origin to admit (loopback origins and the hosted "
        "docs site are always admitted); repeatable",
    )
    http.add_argument(
        "--runs-dir",
        default=None,
        help="where training runs land (default: <root or cwd>/runs)",
    )
    http.add_argument(
        "--webhook",
        default=None,
        metavar="URL",
        help="POST a JSON payload to this URL when a training job finishes, "
        "fails or is stopped (server-side by design: a client-supplied URL "
        "would be server-side request forgery)",
    )
    args = parser.parse_args(argv)
    if args.runs_dir:
        _tools.set_runs_dir(args.runs_dir)
    if args.webhook:
        _tools.set_webhook(args.webhook)
    if not args.http:
        create_server(root=args.root).run()
        return 0
    from . import _require_http

    try:
        _http = _require_http()
    except ImportError as e:
        import sys

        print(str(e), file=sys.stderr)
        return 1
    host = args.host or _http.DEFAULT_HOST
    port = args.port if args.port is not None else _http.DEFAULT_PORT
    token = None if args.no_token else (args.token or _http.new_token())
    app = _http.build_app(
        create_server(root=args.root, host=host),
        token=token,
        allowed_origins=[*_http.DEFAULT_ALLOWED_ORIGINS, *args.allow_origin],
        root=_tools.get_root(),
        runs_dir=args.runs_dir,
        watch_jobs=bool(args.webhook),
    )
    _http.serve(app, host=host, port=port)
    return 0
