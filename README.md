<p align="center">
  <a href="https://github.com/loumalouomega/meshioplusplus"><img alt="meshio++" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/doc/logo/logo-with-text.svg" width="60%"></a>
  <p align="center">I/O for mesh files.</p>
</p>


[![PyPi Version](https://img.shields.io/pypi/v/meshioplusplus.svg?style=flat-square)](https://pypi.org/project/meshioplusplus/) [![npm Version](https://img.shields.io/npm/v/%40meshioplusplus%2Fwasm.svg?style=flat-square)](https://www.npmjs.com/package/@meshioplusplus/wasm) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/meshioplusplus.svg?style=flat-square)](https://pypi.org/project/meshioplusplus/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21629061.svg?style=flat-square)](https://doi.org/10.5281/zenodo.21629061)

[![C++][c++-image]][c++standard] [![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/meshioplusplus/) [![C](https://img.shields.io/badge/C-99-a8b9cc.svg?style=flat-square&logo=c&logoColor=white)](doc/c_api.md) [![Fortran](https://img.shields.io/badge/Fortran-2008-734f96.svg?style=flat-square&logo=fortran&logoColor=white)](doc/fortran.md) [![Julia](https://img.shields.io/badge/Julia-1.9%2B-9558b2.svg?style=flat-square&logo=julia&logoColor=white)](doc/julia.md) [![R](https://img.shields.io/badge/R-4.0%2B-276dc3.svg?style=flat-square&logo=r&logoColor=white)](doc/r.md) [![WebAssembly](https://img.shields.io/badge/WebAssembly-npm-654ff0.svg?style=flat-square&logo=webassembly&logoColor=white)](https://www.npmjs.com/package/@meshioplusplus/wasm) [![TypeScript](https://img.shields.io/badge/TypeScript-viewer-3178c6.svg?style=flat-square&logo=typescript&logoColor=white)](src/viewer/) [![Spack](https://img.shields.io/badge/spack-meshioplusplus-1f79c2.svg?style=flat-square)](https://packages.spack.io/package.html?name=meshioplusplus) [![Blender](https://img.shields.io/badge/Blender-4.2%2B-ea7600.svg?style=flat-square&logo=blender&logoColor=white)](doc/blender.md)

[![GitHub stars](https://img.shields.io/github/stars/loumalouomega/meshioplusplus.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/loumalouomega/meshioplusplus) [![PyPi downloads](https://img.shields.io/pypi/dm/meshioplusplus.svg?style=flat-square)](https://pypistats.org/packages/meshioplusplus)
[![GitHub release date](https://img.shields.io/github/release-date/loumalouomega/meshioplusplus?style=flat-square&label=release)](https://github.com/loumalouomega/meshioplusplus/releases/latest) [![Commits since latest release](https://img.shields.io/github/commits-since/loumalouomega/meshioplusplus/latest?style=flat-square&label=commits%20since)](https://github.com/loumalouomega/meshioplusplus/compare/v8.7.0...master) [![GitHub last commit](https://img.shields.io/github/last-commit/loumalouomega/meshioplusplus?style=flat-square&label=latest%20commit)](https://github.com/loumalouomega/meshioplusplus/commit/master)

[![gh-actions](https://img.shields.io/github/actions/workflow/status/loumalouomega/meshioplusplus/ci.yml?branch=master&style=flat-square)](https://github.com/loumalouomega/meshioplusplus/actions?query=workflow%3Aci) [![codecov](https://img.shields.io/codecov/c/github/loumalouomega/meshioplusplus.svg?style=flat-square)](https://app.codecov.io/gh/loumalouomega/meshioplusplus) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

[c++-image]: https://img.shields.io/badge/C++-20-blue.svg?style=flat&logo=c%2B%2B
[c++standard]: https://isocpp.org/std/the-standard

There are various mesh formats available for representing unstructured meshes. meshio++ can read and write all of the following and smoothly converts between them:

> [Abaqus](https://help.3ds.com/2024/english/dssimulia_established/SIMACAEMODRefMap/simamod-c-inputsyntax.htm) (`.inp`),
> ANSYS msh (`.msh`),
> [Ansys/APDL coded database](https://www.ansys.com) (`.cdb`, `.inp`),
> [AVS-UCD](https://lanl.github.io/LaGriT/pages/docs/read_avs.html) (`.avs`),
> [CGNS](https://cgns.github.io/) (`.cgns`),
> [DOLFIN XML](https://manpages.ubuntu.com/manpages/jammy/en/man1/dolfin-convert.1.html) (`.xml`),
> [COMSOL](https://www.comsol.com) (`.mphtxt`),
> [Exodus](https://nschloe.github.io/meshio/exodus.pdf) (`.e`, `.exo`),
> [EnSight Gold](https://vis.lbl.gov/archive/NERSC/Software/ensight/doc/OnlineHelp/UM-C11.pdf) (geometry, `.case`/`.geo`),
> [FLAC3D](https://www.itascacg.com/software/flac3d) (`.f3grid`),
> [FLUX](https://www.altair.com/flux/) (mesh `.pf3`, field `.dex`),
> [FreeFem++](https://freefem.org/) (`.msh`),
> [GiD postprocess](https://www.gidsimulation.com/) (`.post.msh`/`.post.res`, `.post.bin`, `.post.h5`; writing via a vendored gidpost, reading is meshio++'s own code),
> [H5M](https://www.mcs.anl.gov/~fathom/moab-docs/h5mmain.html) (`.h5m`),
> [HMF](https://loumalouomega.github.io/meshioplusplus/formats/hmf) (`.hmf`, experimental, meshio++-specific),
> [I-deas Universal / UNV](https://www.ceas3.uc.edu/sdrluff/) (`.unv`),
> [ANSYS Fluent interpolation](https://github.com/victorsndvg/FEconv) (`.ip`),
> [Kratos/MDPA](https://github.com/KratosMultiphysics/Kratos/wiki/Input-data) (`.mdpa`),
> [Medit](https://people.sc.fsu.edu/~jburkardt/data/medit/medit.html) (`.mesh`, `.meshb`),
> [MED/Salome](https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html) (`.med`),
> [Modulef](https://github.com/victorsndvg/FEconv) (mesh `.mfm`, field `.mff`),
> [Nastran](https://help.autodesk.com/view/NSTRN/2019/ENU/?guid=GUID-42B54ACB-FBE3-47CA-B8FE-475E7AD91A00) (bulk data, `.bdf`, `.fem`, `.nas`),
> [Netgen](https://github.com/ngsolve/netgen) (`.vol`, `.vol.gz`),
> [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/datasource/precomputed#mesh-representation-of-segmented-object-surfaces),
> [Gmsh](https://gmsh.info/doc/texinfo/gmsh.html#File-formats) (format versions 2.2, 4.0, and 4.1, `.msh`),
> [OBJ](https://en.wikipedia.org/wiki/Wavefront_.obj_file) (`.obj`),
> [OFF](https://segeval.cs.princeton.edu/public/off_format.html) (`.off`),
> [OpenFOAM polyMesh](https://www.openfoam.com/) (`.foam`),
> [PERMAS](https://www.intes.de) (`.post`, `.post.gz`, `.dato`, `.dato.gz`),
> [PLY](<https://en.wikipedia.org/wiki/PLY_(file_format)>) (`.ply`),
> [STL](<https://en.wikipedia.org/wiki/STL_(file_format)>) (`.stl`),
> [Tecplot .dat](http://paulbourke.net/dataformats/tp/),
> [TetGen .node/.ele](https://wias-berlin.de/software/tetgen/fformats.html),
> [Triangle .node/.ele/.poly](https://www.cs.cmu.edu/~quake/triangle.html),
> [SVG](https://www.w3.org/TR/SVG/) (output only; 2D direct, 3D via skin projection) (`.svg`),
> [TikZ](https://tikz.dev/) (LaTeX output only; 2D direct, 3D via skin projection) (`.tikz`),
> [SU2](https://su2code.github.io/docs_v7/Mesh-File/) (`.su2`),
> [UGRID](https://www.simcenter.msstate.edu/software/documentation/ug_io/3d_grid_file_type_ugrid.html) (`.ugrid`),
> [VTI](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html) (VTK XML ImageData; a regular lattice) (`.vti`),
> [VTK](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf) (`.vtk`),
> [VTP](https://docs.vtk.org/en/latest/vtk_file_formats/vtkxml_file_format.html) (`.vtp`),
> [VTU](https://vtk.org/Wiki/VTK_XML_Formats) (`.vtu`),
> [WKT](https://en.wikipedia.org/wiki/Well-known_text_representation_of_geometry) ([TIN](https://en.wikipedia.org/wiki/Triangulated_irregular_network)) (`.wkt`),
> [XDMF](https://xdmf.org/index.php/XDMF_Model_and_Format) (`.xdmf`, `.xmf`).

<p align="center">
  <img alt="" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/doc/logo/logo-icon-square.png" width="64">
</p>

meshio++ ships a **C++20 core** (built with pybind11 + scikit-build-core) that reads and writes most formats with zero-copy numpy at the I/O boundary, plus optional HDF5/netCDF acceleration and a **selectable parallel backend** (`AUTO` by default — prefers OpenMP, then STL+TBB, then sequential; override with `-DMESHIOPLUSPLUS_PARALLEL_BACKEND=...`, including a bring-your-own [Kokkos](https://kokkos.org) host backend). Every format has a pure-Python fallback, so behaviour and file compatibility are identical whether or not the native libraries are present. For a standalone C++ build use `build/configure.sh` (Linux/macOS) or `build/configure.bat` (Windows). Full docs (install, data model, per-format options, CLI) live at [the documentation site](https://loumalouomega.github.io/meshioplusplus/) (sources under [`doc/`](https://github.com/loumalouomega/meshioplusplus/tree/main/doc)).

Install with

```
pip install meshioplusplus[all]
```

(`[all]` pulls in all optional dependencies. By default, meshio++ only uses numpy.) You can then use the command-line tool

<!--pytest-codeblocks:skip-->

```sh
meshioplusplus convert    input.msh output.vtk   # convert between two formats

meshioplusplus info       input.xdmf             # show some info about the mesh

meshioplusplus compress   input.vtu              # compress the mesh file
meshioplusplus decompress input.vtu              # decompress the mesh file

meshioplusplus binary     input.msh              # convert to binary format
meshioplusplus ascii      input.msh              # convert to ASCII format

meshioplusplus merge      a.vtu b.vtu out.vtu    # merge meshes (optional --weld)

meshioplusplus transform  in.vtu out.vtu --translate 1,2,3   # affine transform
meshioplusplus clean      in.vtu out.vtu --weld              # weld / prune / de-dup
meshioplusplus crop       in.vtu out.vtu --bbox 0,0,0,1,1,1  # subset by region
meshioplusplus crop       in.vtu out.vtu --where "d<0"       # ... or by a data predicate
meshioplusplus split      in.vtu 'out_{key}.vtu' --by type   # split by criterion
meshioplusplus stats      mesh.vtu                           # geometric statistics
meshioplusplus convert-cells in.msh out.vtu --mode simplexify  # hexes -> tetra
meshioplusplus subdivide  in.vtu out.vtu                      # polyhedral refinement, any 3D cell
meshioplusplus refine     in.vtu out.vtu --levels 2          # uniform subdivision
meshioplusplus refine     in.vtu out.vtu --where "q<0.3"     # adaptive, closed conformingly
meshioplusplus partition  in.vtu 'out_{part}.vtu' --nparts 4 # N balanced parts
meshioplusplus remesh-volume in.vtu out.vtu --cell-size 0.5  # retetrahedralize (isosurface stuffing)
meshioplusplus optimize-volume in.vtu out.vtu               # ODT-remesh a tet mesh (relocate + flips)
meshioplusplus smooth     in.vtu out.vtu --iterations 20     # relax node positions
meshioplusplus smooth     in.vtu out.vtu --method odt        # ODT smoothing, tet-only
meshioplusplus interpolate src.vtu tgt.vtu out.vtu           # transfer fields across meshes
meshioplusplus slice      in.vtu out.vtu --normal 0,0,1      # planar cross-section
meshioplusplus isosurface in.vtu out.vtu --array T --values 350  # level set of a field
meshioplusplus sdf        skin.stl field.vti --resolution 128,128,128  # signed distance field
meshioplusplus data gradient in.vtu out.vtu --array T           # grad / div / curl of a field
meshioplusplus data hessian  in.vtu out.vtu --array T           # second derivative, gradient's companion
meshioplusplus data estimate-error in.vtu out.vtu --array T --marking dorfler --marking-value 0.6  # ZZ error indicator + marking
meshioplusplus data integrate in.vtu --array density            # total / mean, per region

meshioplusplus data info  mesh.vtu                           # summarize data arrays
meshioplusplus data calc  in.vtu out.vtu --point "s = norm(v)"   # derive a field
meshioplusplus data to-cell  in.vtu out.vtu --keys T         # point -> cell average
meshioplusplus data normalize in.vtu out.vtu --cell damage --to 0,1

meshioplusplus pipeline   settings.json                      # run a whole declarative
                                                             # read -> ops -> write chain
```

with any of the supported formats.

The same verbs are available as a **standalone C++ binary** that needs no Python: grab a ready-to-run, statically-linked build for Linux/macOS/Windows from the [GitHub Releases](https://github.com/loumalouomega/meshioplusplus/releases) page, or build it yourself with `build/configure.sh --cli --build` (or `-DMESHIOPLUSPLUS_BUILD_CLI=ON`). It links only the C++ core. Named [regions](https://loumalouomega.github.io/meshioplusplus/regions) — and so point/cell *sets* — are carried there since v8.1.0, so `info` lists them and `diff` compares them; `convert -s/-d` is still Python-only.

A whole *chain* of operations can be described declaratively in one `settings.json` (`Input` → `Operations` → `Output`) and run with `meshioplusplus pipeline settings.json` — or `meshioplusplus.run_pipeline(...)` from Python, and the same engine from C/Fortran/Julia/R/WASM. See [the settings pipeline](https://loumalouomega.github.io/meshioplusplus/pipeline).

In Python, simply do

<!--pytest-codeblocks:skip-->

```python
import meshioplusplus

mesh = meshioplusplus.read(
    filename,  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshioplusplus convert --help for all possible formats
)
# mesh.points, mesh.cells, mesh.cells_dict, ...

# mesh.vtk.read() is also possible
```

to read a mesh. To write, do

```python
import meshioplusplus

# two triangles and one quad
points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
    [2.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]]),
    ("quad", [[1, 4, 5, 3]]),
]

mesh = meshioplusplus.Mesh(
    points,
    cells,
    # Optionally provide extra data on points, cells, etc.
    point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
    # Each item in cell data must match the cells array
    cell_data={"a": [[0.1, 0.2], [0.4]]},
)
mesh.write(
    "foo.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# Alternative with the same options
meshioplusplus.write_points_cells("foo.vtk", points, cells)
```

For both input and output, you can optionally specify the exact `file_format` (in case you would like to enforce ASCII over binary VTK, for example).

#### Skin extraction

`meshioplusplus.extract_skin` derives the boundary surface of a 3D volume mesh (the [Kratos `SkinDetectionProcess`](https://github.com/KratosMultiphysics/Kratos) face-hashing algorithm — faces occurring exactly once are boundary; points are compacted, `point_data` follows):

<!--pytest-codeblocks:skip-->

```python
vol = meshioplusplus.read("part.msh")     # tetra/hexa/wedge/pyramid mesh
skin = meshioplusplus.extract_skin(vol)   # triangle/quad/... surface mesh
```

The **STL and PLY writers do this automatically** for volume meshes (pass `skin=False` for the legacy drop-volume-cells behavior), and the **SVG/TikZ writers render 3D meshes** by projecting the skin through an orthographic camera (`azimuth`/`elevation`/`roll` in degrees, default the classic CAD isometric view) with painter's-algorithm depth ordering — that is exactly how the Stanford-bunny logo above is drawn.

#### Publication-quality vector figures

The SVG and TikZ writers can colour each face by a data array, turning them into figures you can drop straight into a paper — resolution-independent, and with **no extra dependency**: the colormaps are built into the core.

<!--pytest-codeblocks:skip-->

```python
annotated = meshioplusplus.attach_quality(mesh)
meshioplusplus.write(
    "quality.svg", annotated,
    color_by="quality:scaled_jacobian",   # or any point_data / cell_data name
    cmap="viridis",                       # viridis / coolwarm / turbo
    colorbar=True,
)
```

<img alt="a bracket coloured by scaled Jacobian" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/doc/public/images/color_by_quality.svg" width="85%">

*The bundled bracket coloured by element quality — the same figure `tools/gen_doc_images.py` regenerates.*

**Point data** colours a face by the mean of its corner values, **cell data** by its owning cell's value — for a volume mesh, tracked through the extracted skin's parent-cell provenance, so a per-cell material or metric lands on the right facet. Multi-component arrays reduce to a `component` or to their magnitude; `vmin`/`vmax` set the range (default: the drawn faces' finite range), and non-finite values take `nan_color`. From the command line:

<!--pytest-codeblocks:skip-->

```sh
meshioplusplus convert mesh.vtu figure.svg --color-by temperature --colorbar
```

Colouring is available from Python, from C++ directly, and from both CLIs; the flat C/Fortran/WebAssembly bindings reach these writers through the shared registry and always emit the default styling.

#### Surface extraction

`meshioplusplus.extract_surface` is the general form of skin extraction: it picks the dimension automatically (a volume mesh → boundary faces, a 2D surface mesh → boundary edges) and can record each facet's parent cell id (`record_parent_ids=True`). See the [surface extraction docs](https://meshioplusplus.readthedocs.io) (`doc/extract_surface.md`).

<!--pytest-codeblocks:skip-->

```python
surf = meshioplusplus.extract_surface(vol)                  # faces (or edges for a 2D mesh)
edges = meshioplusplus.extract_surface(sheet, record_parent_ids=True)
```

#### Mesh quality

`meshioplusplus.compute_quality` scores every cell on a set of geometric quality metrics (area/volume, scaled Jacobian, aspect ratio, skewness, interior/dihedral angles, warpage) and flags inverted/degenerate cells; `attach_quality` writes them back as `cell_data`. See `doc/mesh_quality.md`.

<!--pytest-codeblocks:skip-->

```python
report = meshioplusplus.compute_quality(mesh)
print(report["num_inverted"], "inverted cells")
annotated = meshioplusplus.attach_quality(mesh)   # metrics as cell_data
```

#### Reordering / renumbering

`meshioplusplus.reorder` renumbers nodes and elements to reduce sparse-matrix bandwidth (Reverse Cuthill–McKee) or improve cache locality (Morton / Hilbert space-filling curves). It is a pure permutation — geometry and all data preserved — and returns the applied node/cell permutations so external arrays can be remapped. `compute_bandwidth` measures the before/after connectivity bandwidth. See `doc/reorder.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.reorder(mesh, method="rcm")            # "morton" / "hilbert" too
out, node_perm, cell_perms = meshioplusplus.reorder(mesh, return_permutation=True)
print(meshioplusplus.compute_bandwidth(mesh), "->", meshioplusplus.compute_bandwidth(out))
```

#### Comparison (diff)

`meshioplusplus.diff` compares two meshes and reports whether they are equivalent within a tolerance (`abs_err <= atol + rtol*|expected|`), with a structured breakdown (points, cells, data, named sets) and an overall verdict (`identical` / `equal within tolerance` / `different`); `meshes_equal` is the boolean wrapper for test suites. An optional `unordered=True` mode matches nodes by spatial proximity, so a shuffled node order still compares equal. See `doc/diff.md`.

<!--pytest-codeblocks:skip-->

```python
assert meshioplusplus.meshes_equal(a, b, atol=1e-8)         # ideal in a regression test
report = meshioplusplus.diff(a, b, unordered=True)          # tolerant to shuffled node order
print(report["verdict"])
```

The `meshioplusplus diff a.vtu b.vtu` CLI verb sets a nonzero exit code when meshes differ, for direct use in CI / Makefiles.

#### Merge / combine

`meshioplusplus.merge` combines two or more meshes into one: it concatenates points (offsetting connectivity so indices stay valid), merges cell blocks by type, concatenates data (per a configurable `data_policy`), and tags each cell's origin. With `weld=True` it fuses coincident nodes across inputs within `atol` using a spatial hash (never O(N²)) — the standard way to stitch adjacent blocks into a watertight mesh. Overlapping set / field-data names are namespaced by source id. See `doc/merge.md`.

<!--pytest-codeblocks:skip-->

```python
combined = meshioplusplus.merge([a, b, c])                 # concatenate
welded = meshioplusplus.merge([left, right], weld=True, atol=1e-8)  # fuse the shared interface
```

#### Editing (transform / clean / crop / split) and statistics

A bundle of dependency-free mesh-editing utilities:

- **`meshioplusplus.transform`** — apply an affine transform (translate / scale / rotate / 4×4 matrix / unit-scale) to the points; connectivity and data are carried through. See `doc/transform.md`.
- **`meshioplusplus.clean`** — weld coincident points (spatial hash), drop degenerate and duplicate cells, and remove orphaned points, in one toggleable pass. See `doc/clean.md`.
- **`meshioplusplus.crop`** — extract the part of a mesh inside a bounding box or half-space, pruning unused points (`mode="all"`/`"any"`). See `doc/crop.md`.
- **`meshioplusplus.split`** — partition a mesh into several by cell type, connected component (flood-fill), region (`cell_sets` / integer tag), or one piece per named Cell region (`by="regions"`, cross-binding, not a partition — overlapping regions overlap). See `doc/split.md`. `meshioplusplus regions` (CLI) / `read_metadata(...)["regions"]` lists a mesh's regions cheaply, without a full read where possible.
- **`meshioplusplus.compute_stats`** — geometric statistics (bounding box, centroid, per-type counts, area, signed/unsigned volume, inverted cells) — the geometric complement to `info`. See `doc/stats.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.transform(mesh, rotate=("z", 90))
out = meshioplusplus.clean(mesh, weld=True, atol=1e-8)
sub = meshioplusplus.crop(mesh, bbox=[0, 0, 0, 1, 1, 1])
pieces = meshioplusplus.split(mesh, by="type")             # {"triangle": ..., ...}
s = meshioplusplus.compute_stats(mesh)                     # dict of measures
```

#### Cell conversion (linearize / simplexify / elevate)

**`meshioplusplus.convert_cells`** converts a mesh's *element representation* — which cell types it is built from — while leaving the object it describes intact. See `doc/convert_cells.md`.

- **`mode="linearize"`** — every higher-order cell becomes its linear base (`tetra10` → `tetra`, `hexahedron27` → `hexahedron`), keeping the corner connectivity verbatim and pruning the nodes that become unreferenced.
- **`mode="simplexify"`** — every cell is decomposed into simplices of the same topological dimension (`quad` → 2 `triangle`, `hexahedron` → 6 `tetra`, `wedge` → 3, `pyramid` → 2, an n-gon into an (n−2)-triangle fan). No points are added, each parent's `cell_data` is replicated to its children, and every emitted simplex is positively oriented with volume conserved.
- **`mode="elevate"`** — every linear cell is promoted to its serendipity quadratic counterpart (`triangle` → `triangle6`, `hexahedron` → `hexahedron20`), adding one node per unique edge at the edge midpoint with `point_data` set to the endpoint mean.

<!--pytest-codeblocks:skip-->

```python
linear = meshioplusplus.convert_cells(mesh, mode="linearize")
tets = meshioplusplus.convert_cells(mesh, mode="simplexify")   # hexes -> tetra
quadratic = meshioplusplus.convert_cells(mesh, mode="elevate")
```

Each mode is idempotent on cells it does not apply to, so it is safe on a mixed-order mesh, and output is byte-identical across mesh backends and thread counts.

#### Polyhedral refinement (subdivide)

**`meshioplusplus.subdivide`** splits every eligible 3D cell into one polyhedral child per face, connected to a new interior point. `refine` and `decimate` both raise by name on a polyhedron, pointing at `convert_cells(mode="simplexify")` — both are built on fixed same-type subdivision templates, and an arbitrary polyhedron has none. `subdivide` needs no per-type table at all: tabulated types (reduced to corners for a quadratic variant) and existing polyhedron blocks are handled uniformly, so the same code covers every 3D cell type the mesh already supports. Automatically conforming (no closure, no hanging nodes), unlike `refine`. See `doc/subdivide.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.subdivide(mesh, record_parent_ids=True)
```

#### Polyhedral coarsening (agglomerate)

**`meshioplusplus.agglomerate`** merges groups of cells into single larger polyhedral cells — the many-to-one counterpart to `subdivide`. `decimate` raises by name on a polyhedron, pointing at `convert_cells(mode="simplexify")` — its fixed-template QEM edge collapse has no analogue for merging arbitrary polyhedral cells. `agglomerate` is a genuinely different algorithm: greedy seed-and-grow over the mesh's shared-face dual, absorbing face-adjacent neighbours by accumulated shared-face area until a target group size; each group emits one polyhedron whose faces are exactly its external boundary, conserving volume exactly. Non-volume blocks pass through unchanged; points are never pruned or renumbered (`clean(mesh, remove_orphans=True)` is the follow-up for a minimal point set). See `doc/agglomerate.md`.

<!--pytest-codeblocks:skip-->

```python
coarse = meshioplusplus.agglomerate(mesh, target_group_size=8)
```

#### Refinement

**`meshioplusplus.refine`** subdivides cells into congruent children of the *same* cell type, increasing a mesh's resolution: `line` → 2, `triangle` → 4, `quad` → 4, `tetra` → 8, `wedge` → 8, `hexahedron` → 8, with `levels=n` applying the templates `n` times. Given a **selection** — a cell list, a region name, or a `cell_data` threshold — it refines only those cells and resolves the resulting hanging nodes, so the output is still conforming. See `doc/refine.md`.

New nodes sit at the midpoints of the parent's edges, quad faces and (hexahedron only) body, and carry the mean of that entity's corner values for every `point_data` array — so a linear field is interpolated exactly. Mid-edge and quad-face-centre nodes are **shared** between every cell touching the entity, so the refined mesh has no hanging nodes; each parent's `cell_data` row is replicated to its children.

`record_hierarchy=True` attaches two persistent `cell_data` arrays, `refine:cell_id`/`refine:parent_id`: a stable identity that survives across separate `refine` calls — **a link between two meshes, not a tree inside one**, resolved by a multigrid caller against the sequence of meshes it keeps.

<!--pytest-codeblocks:skip-->

```python
fine = meshioplusplus.refine(mesh)                      # one level
finer = meshioplusplus.refine(mesh, levels=2)           # 64x the cells in 3D
tagged = meshioplusplus.refine(mesh, record_parent_ids=True)

# adaptive: refine the worst cells and close up conformingly around them
graded = meshioplusplus.refine(
    meshioplusplus.attach_quality(mesh),
    where="quality:scaled_jacobian < 0.3",
    record_levels=True,                                 # colour by refine:level
)

# multigrid: keep the coarse mesh, resolve the fine mesh's parent ids against it
fine = meshioplusplus.refine(coarse, cells=[4, 8], record_hierarchy=True)
```

Children inherit the parent's orientation (zero newly-inverted cells for a well-oriented input), and volume is conserved — exactly for `tetra` always, and for `wedge`/`hexahedron` when the parent is affine. Higher-order cells, `pyramid`, and ragged blocks have no same-type subdivision and raise by name.

#### Green-element undo

**`meshioplusplus.undo_green(coarse, fine)`** restores `refine`'s transitional ("green") cells back to their original parent — the standard rule for selective refinement: before a new pass touches a region a prior pass already closed up, restore the transitional cell to its parent and re-split from scratch, rather than refining the transitional children directly (which degrades element quality without bound over repeated passes). It is a **two-mesh** operation, like `interpolate`: `coarse` is the mesh a prior `refine(coarse, ..., record_hierarchy=True, record_levels=True)` call was run on, `fine` is that call's output. Since `refine`'s point map is always the identity, a green parent's exact connectivity and cell_data are already sitting, byte-for-byte, in `coarse` — so this is a lookup and substitution, not a reconstruction. See `doc/undo_green.md`.

<!--pytest-codeblocks:skip-->

```python
fine = meshioplusplus.refine(coarse, cells=[4, 8], record_hierarchy=True, record_levels=True)
# ... later, decide to refine a different region ...
undone = meshioplusplus.undo_green(coarse, fine)
redone = meshioplusplus.refine(undone, cells=[12, 19], record_hierarchy=True, record_levels=True)
```

#### Decimation

**`meshioplusplus.decimate`** is `refine`'s inverse: it *reduces* a surface mesh's face count by greedy quadric-error-metric (Garland–Heckbert) edge collapse, preserving shape, boundaries and features. Exactly one stopping criterion is given — `ratio` (fraction of faces to keep), `target_faces`, or `max_error` — and the output is all-triangle (`quad`/`polygon` blocks are triangulated first, block structure kept 1:1). See `doc/decimate.md`.

<p align="center">
<img alt="a refined sphere before and after decimation to 25% of its faces" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/doc/public/images/decimate_before_after.png" width="85%">
</p>

<!--pytest-codeblocks:skip-->

```python
coarse = meshioplusplus.decimate(mesh, ratio=0.25)            # keep 25% of the faces
coarse = meshioplusplus.decimate(mesh, target_faces=5000)     # absolute face budget
coarse, report = meshioplusplus.decimate(mesh, max_error=1e-6, return_report=True)
```

Boundary vertices (once-used-edge test) and feature vertices (face normals differing by more than `feature_angle`, default 30°) are pinned by default, so an open patch keeps its outline exactly and a cube keeps its corners; the link condition and a normal-flip guard reject any collapse that would change topology, create a non-manifold edge, or fold the surface. Float `point_data` blends along the collapsed edge; integer arrays keep the survivor's value. Volume meshes raise by name — run `extract_surface` first, then decimate the skin.

#### Volume decimation

**`meshioplusplus.decimate_volume`** is `decimate`'s volume-mesh sibling — a separate operation, not a mode on it — reducing a **tetrahedral** mesh's cell count by greedy quadric-error-metric **tet**-edge collapse. Unlike `decimate`, boundary vertices *participate* by default (`preserve_boundary=False`): every vertex accumulates a quadric from its incident boundary-triangle planes only, so a purely interior vertex's quadric is exactly zero and interior-only edges are scored by squared length instead, always ranking behind boundary-touching collapses. See `doc/decimate_volume.md`.

<!--pytest-codeblocks:skip-->

```python
coarse = meshioplusplus.decimate_volume(mesh, ratio=0.25)             # keep 25% of the tets
coarse = meshioplusplus.decimate_volume(mesh, target_cells=5000)      # absolute tet budget
coarse, report = meshioplusplus.decimate_volume(mesh, max_error=1e-6, return_report=True)
```

Validity is guarded by an exact vertex-link set-equality condition, a duplicate-tet check, and a tet-inversion guard, plus — for boundary-touching collapses — `decimate`'s own ring/shared-face link condition and normal-flip check reused over the mesh's own outer skin. Tet-only: any non-tetra 3D block raises by name pointing at `convert_cells(mode="simplexify")`.

#### Partitioning

**`meshioplusplus.partition`** decomposes a mesh into exactly N balanced pieces for domain decomposition — the count-driven complement to the criterion-driven `split`. See `doc/partition.md`.

- **SFC** (the default fallback, always available, dependency-free): cells are cut into contiguous ranges along a Hilbert space-filling curve of their centroids — equal-weight part sizes differ by at most one cell, `weights=<cell_data>` balances a per-cell cost instead, and the assignment is deterministic and byte-identical across mesh backends and thread counts.
- **KaHIP** (the optional quality path): the shared-face dual graph goes through [KaHIP](https://github.com/KaHIP/KaHIP)'s serial `kaffpa()`, which actively minimizes the edge cut. Configure `imbalance` (default 3%), `mode` (`fast`/`eco`/`strong`, default `eco` — eco/strong carry the quality) and `seed`. **KaHIP is MIT-licensed like meshio++ itself, so enabling it changes nothing about licensing**; it is bring-your-own (`-DMESHIOPLUSPLUS_WITH_KAHIP=ON` + `KAHIP_ROOT`, Conan `with_kahip`, vcpkg feature `kahip` — next to the HDF5/zstd-style optional deps), links only the serial interface (no MPI), and `pip install meshioplusplus[kahip]` gives pure-Python installs the same quality path via the MIT `kahip` wheel. Requesting it where absent fails by name — never a silent downgrade.

<!--pytest-codeblocks:skip-->

```python
pieces = meshioplusplus.partition(mesh, 4)                    # list of 4 meshes
labels = meshioplusplus.partition_labels(mesh, 4)             # per-block Int64 part ids
quality = meshioplusplus.partition(mesh, 16, method="kahip", mode="strong")
```

Pieces keep the input's block structure 1:1, so they recombine into the input: every cell lands in exactly one piece. `ghost_layers=N` instead grows each piece by N shared-node layers of its neighbours' cells (an MPI-style halo), tagged `partition:ghost`.

#### Surface remeshing (ACVD clustering)

**`meshioplusplus.remesh`** replaces a **surface** mesh's own triangulation with a new, near-uniformly-sized, well-shaped one at a caller-chosen vertex count, by approximated centroidal Voronoi diagram (ACVD) clustering — the one resolution-changing operation that does not inherit the input's element shapes, so it can *raise* quality at every target count rather than only remove elements. See `doc/remesh.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.remesh(mesh, num_clusters=5000)
out = meshioplusplus.remesh(mesh, num_clusters=5000, metric="quadric")       # preserves sharp edges/corners
out = meshioplusplus.remesh(mesh, num_clusters=5000, metric="anisotropic", max_anisotropy=4.0)
```

`metric="isotropic"` (default) is fast and rounds sharp features; `"quadric"` (Garland-Heckbert error) pins clusters onto edges/corners instead; `"anisotropic"` shapes clusters with a local curvature tensor, elongating elements along low-curvature directions. The output has entirely **new** points and connectivity — no point/cell map — so `point_data`/`cell_data`/named regions are dropped and `field_data` carries through; transfer a field onto the result with `interpolate`/`conservative_interpolate`.

#### Volumetric remeshing (isosurface stuffing)

**`meshioplusplus.remesh_volume`** is `remesh`'s volumetric sibling — the tet-mesh counterpart `decimate_volume` needed but could not itself provide, since QEM edge collapse can only remove elements. It retetrahedralizes a **volume** mesh (or a closed surface, unlike `remesh`) at a caller-chosen resolution by isosurface stuffing over a body-centered cubic (BCC) lattice, generating an entirely new mesh rather than working on the input's own cells. See `doc/remesh_volume.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.remesh_volume(mesh, cell_size=0.5)
out = meshioplusplus.remesh_volume(mesh, resolution=(64, 64, 64))
out, report = meshioplusplus.remesh_volume(mesh, cell_size=0.5, return_report=True)
print(report["num_tets"], report["num_non_manifold_edges"])
```

Every uncut lattice tet has a dihedral angle from a small, mesh-size-independent fixed set. `warp_fraction` (default `0.35`) moves lattice vertices near the surface onto it, trading a small, *measured* chance of non-manifold boundary edges (reported in `num_non_manifold_edges`) for substantially better boundary tet quality; `0` disables warping and gives an exactly watertight but lower-quality boundary. Implemented from the published description of Labelle & Shewchuk's isosurface stuffing (SIGGRAPH 2007) only — no predicate library needed, unlike literal Delaunay/CVD tetrahedralization.

#### ODT remeshing (relocate + flip connectivity)

**`meshioplusplus.optimize_volume`** raises a tetrahedral mesh's worst element quality by *ODT remeshing* — relocating vertices AND changing connectivity. It is the genuine "ODT remeshing" and the third member of a trio whose other two each do half the job: `smooth(method="odt")` moves points on *fixed* connectivity, `remesh_volume` discards the input's tets for a fresh lattice mesh. It alternates the ODT vertex relocation with quality-improving topological **flips** (2-3 and 3-2), **predicate-free** — a flip is applied only when a pure signed-volume test finds the local configuration convex and the minimum scaled Jacobian strictly improves (no in-sphere/Delaunay predicate). See `doc/optimize_volume.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.optimize_volume(mesh)                       # a tetrahedral mesh
out, report = meshioplusplus.optimize_volume(mesh, return_report=True)
print(report["num_flips"], report["min_quality_before"], report["min_quality_after"])
```

The flips touch only interior faces/edges, so the boundary surface is preserved exactly (watertight in ⇒ watertight out); the point set is invariant, so `point_data`/`field_data` and named Point regions carry through while `cell_data`/Cell/Side regions are dropped. Tet-only, and (like `remesh_volume`) C++-core only.

#### Smoothing

**`meshioplusplus.smooth`** relaxes point coordinates to improve element shape, leaving topology and every data value alone: **only the points move**. See `doc/smooth.md`.

Three methods. **Laplacian** (`x <- x + lambda*L(x)`, the edge-neighbour centroid displacement) smooths strongly per pass but shrinks — over 40 iterations on a jittered 8×8 quad grid it contracts the bounding box by 57%. **Taubin** (the default) follows each `+lambda` pass with a larger-magnitude `-mu` pass that deliberately un-shrinks, leaving the same grid 3.6% smaller. **ODT** (optimal-Delaunay-triangulation smoothing, **tet-only**) instead moves each free interior vertex to the closed-form volume-weighted average of its incident tets' circumcenters — the "ODT" half of the volumetric-remeshing roadmap item, closed as *smoothing on existing connectivity* rather than remeshing. Neighbours for Laplacian/Taubin are the nodes joined by an actual cell *edge*, not the element clique, so a structured hex block is a fixed point rather than being bevelled toward a sphere. Boundary nodes, feature nodes (incident boundary facet normals differing by more than `feature_angle`), an optional `frozen` mask, and the nodes of blocks whose edge topology is unknown are all pinned by default, and the inversion guard rejects any move that would turn a valid cell inverted.

<!--pytest-codeblocks:skip-->

```python
relaxed = meshioplusplus.smooth(mesh)                          # 10 Taubin iterations
harder = meshioplusplus.smooth(mesh, iterations=40)            # shrink-free even so
lap = meshioplusplus.smooth(mesh, method="laplacian", lambda_=0.4)  # note the underscore
odt = meshioplusplus.smooth(tet_mesh, method="odt", iterations=10)  # tet-only, C++-core only
out, report = meshioplusplus.smooth(mesh, return_report=True)  # nodes moved, max displacement
```

`lambda_` carries a trailing underscore because `lambda` is a Python keyword, and a negative value means "this method's own default" (0.5 Laplacian, 0.33 Taubin). Point and cell counts, connectivity, `cell_data`, `field_data`, `point_data` values and the points array's dtype all come through unchanged, and output is byte-identical across mesh backends and thread counts.

#### Field transfer (interpolation)

**`meshioplusplus.interpolate`** samples a **source** mesh's data arrays onto a **target** mesh — the first cross-mesh operation that transfers data (diff compares, merge concatenates). The result is a copy of the target, its own geometry/data/sets preserved exactly, with the requested source arrays sampled on: source `point_data` at the target's points, source `cell_data` by nearest source-cell centroid. `method="nearest"` (default) copies the nearest source point's value bit-for-bit; `method="barycentric"` simplexifies the source first and interpolates linearly — exact on a linear field — with `default_value`/`extrapolate` deciding what happens outside the source domain. Both search grids are bucket-grid spatial hashes (never O(N²)), and output is byte-identical across backends, thread counts and the C++/numpy boundary. See `doc/interpolate.md`.

<!--pytest-codeblocks:skip-->

```python
coarse = meshioplusplus.read("solution.vtu")   # carries point_data "T"
fine = meshioplusplus.read("remeshed.vtu")
mapped = meshioplusplus.interpolate(coarse, fine, method="barycentric")
meshioplusplus.write("mapped.vtu", mapped)
```

**`meshioplusplus.conservative_interpolate`** is a separate, mass-preserving sibling: over the region the two meshes share, `sum(target value * target measure)` equals `sum(source value * source measure)`, a property `interpolate`'s pointwise sampling does not have. Both meshes are simplexified first (accepting ragged/polyhedron blocks for free), overlapping simplex pairs are measured with an exact geometric clip (Sutherland-Hodgman in 2D, a bounded tetrahedron-tetrahedron clip in 3D), and a target cell's value is the overlap-measure-weighted mean of every source cell it intersects. Unlike `interpolate`, an unset `arrays` transfers every source `point_data` **and** `cell_data` array. `point_data` is transferred by composition (`point_data_to_cell_data` → the same clip algorithm → `cell_data_to_point_data`), a layered approximation rather than exact nodal conservation. C++-core only, with no pure-Python fallback (the 3D clip's discrete branches could disagree near a degenerate overlap). See `doc/conservative_interpolate.md`.

#### Slicing / cross-sections

**`meshioplusplus.slice`** computes the planar cross-section of a mesh — the actual intersection of the mesh with a plane, one topological dimension below the cut cells: a 3D volume mesh yields a `triangle`/`quad` surface, a 2D surface mesh a `line` mesh. Unlike `crop` (plane mode), which keeps whole cells on one side, `slice` computes the intersection and lowers the dimension. It uses robust marching tetrahedra (the input is simplexified first, so each cell's cross-section is a well-defined convex primitive), deduping crossing points on shared edges into single nodes so the section is watertight, and winding every face consistently toward the `+normal` side. Each section cell inherits its parent's `cell_data`; `record_parent_ids=True` attaches `slice:parent_cell`, and `point_data` is interpolated at the cut. Output is byte-identical across backends, thread counts and the C++/numpy boundary. See `doc/slice.md`.

<!--pytest-codeblocks:skip-->

```python
vol = meshioplusplus.read("part.vtu")                          # a tetra/hex/wedge mesh
section = meshioplusplus.slice(vol, origin=(0, 0, 0.5), normal=(0, 0, 1))
meshioplusplus.write("section.vtu", section)                  # a triangle/quad surface at z=0.5
```

(`slice` shadows the Python built-in only as a module attribute — `meshioplusplus.slice` is intended.)

#### Isosurfaces / contours

**`meshioplusplus.isosurface`** computes the level set of a scalar field — the locus where a `point_data` array equals a given isovalue, as a mesh one topological dimension below the cut cells: a 3D volume mesh yields a `triangle`/`quad` surface, a 2D surface mesh a `line` contour. It is the data-driven sibling of `slice` (which cuts where the distance to a plane is zero) and shares its marching-tetrahedra cutter, so contours are watertight in the same way. The field must be `point_data` — `cell_data` is piecewise constant and has no level set, so naming one raises and points at `cell_data_to_point_data`. Several isovalues land in one mesh, cut in ascending order and tagged per cell with a Float64 `iso:value` and an Int64 `iso:index` (the ordinal — the integer tag `split(by="region", tag=…)` needs). The contoured field reads back as **exactly** the isovalue on the cut points, faces are wound toward increasing field, and an out-of-range isovalue is an empty contour rather than an error. Output is byte-identical across backends, thread counts and the C++/numpy boundary. See `doc/isosurface.md`.

<!--pytest-codeblocks:skip-->

```python
vol = meshioplusplus.read("part.vtu")                          # carrying point_data["T"]
shells = meshioplusplus.isosurface(vol, "T", [300.0, 350.0, 400.0])
meshioplusplus.write("shells.vtu", shells)                     # three tagged contour surfaces
```

#### Field derivatives (gradient / divergence / curl)

**`meshioplusplus.gradient`** differentiates a `point_data` field: its gradient, divergence or curl. meshio++ could already transform, transfer, summarize and contour a field — this is what lets it *differentiate* one, which is the missing input for contouring a derived quantity (`|∇T|`, vorticity) and for the gradient-based error indicators that drive the selective `refine`. Two methods: **Green-Gauss** (the default) applies the divergence theorem over the cell, fanning each face into triangles about its corner average, which is **exact for a linear field on any cell** — planar faces or not, because the fan surface is closed; **least-squares** fits over the cells sharing a node and falls back to Green-Gauss (counted, never silently wrong) on a degenerate neighbourhood. An `nc`-component input yields `3·nc` gradient components laid out `[component][derivative]`, so a scalar gives `(n, 3)` and a 3-vector `(n, 9)`; divergence gives 1 and curl 3. Output is `Float64`, named `<input>:gradient` / `:divergence` / `:curl`, at either the cell (default) or point location. A `cell_data` input raises by name — a piecewise-constant field has no derivative — and cells that cannot be differentiated yield NaN and are counted rather than approximated. Geometry, regions and existing data pass through bit-identically; output is byte-identical across backends, thread counts and the C++/numpy boundary. See `doc/gradient.md`.

<!--pytest-codeblocks:skip-->

```python
import numpy as np

vol = meshioplusplus.read("solution.vtu")                      # carrying point_data["T"]
g = meshioplusplus.gradient(vol, "T", location="point")        # point_data["T:gradient"], (n, 3)

grad = np.asarray(g.point_data["T:gradient"])
g.point_data["gradT"] = np.sqrt((grad**2).sum(axis=1))
shells = meshioplusplus.isosurface(g, "gradT", [2.0])          # contour where T changes fastest
```

These operations are exposed across every binding surface (Python, C API, Fortran, WASM) and as the CLI verbs `meshioplusplus quality`, `meshioplusplus extract-surface`, `meshioplusplus reorder`, `meshioplusplus diff`, `meshioplusplus merge`, `meshioplusplus transform`, `meshioplusplus clean`, `meshioplusplus crop`, `meshioplusplus slice`, `meshioplusplus split`, `meshioplusplus stats`, `meshioplusplus convert-cells`, `meshioplusplus subdivide`, `meshioplusplus agglomerate`, `meshioplusplus refine`, `meshioplusplus undo-green`, `meshioplusplus partition`, `meshioplusplus remesh`, `meshioplusplus remesh-volume`, `meshioplusplus optimize-volume`, `meshioplusplus smooth`, `meshioplusplus interpolate`, `meshioplusplus conservative-interpolate`, and `meshioplusplus isosurface` (plus `meshioplusplus data gradient`, `meshioplusplus data hessian`, `meshioplusplus data estimate-error` and `meshioplusplus data integrate`, mesh operations grouped under `data` because that is where a user looks for them).

#### Second derivatives (Hessian)

**`meshioplusplus.hessian`** computes the Hessian (second derivative) of a **scalar** `point_data` field — `gradient`'s companion one order further, for curvature-based adaptive refinement. A **composition** of two `gradient` calls, not a new numerical kernel: the field is differentiated once (point location), then that `(n, 3)` gradient is differentiated again with the default gradient operator, producing `(n, 9)` — the flattened row-major 3x3 Hessian. `method` forwards to both internal passes. A field that is at most linear has an exactly zero Hessian everywhere — the one mesh-shape-independent guarantee; a genuinely quadratic field's composition is exact on a structured/symmetric mesh away from its own boundary and a good, standard, but genuinely approximate curvature estimate on an irregular mesh. Input must have exactly one component. A curvature-driven refinement indicator needs no new code: `data_calc`'s `norm(...)` on the 9-component output is exactly its Frobenius norm, ready for `refine`'s `where` selector. See `doc/hessian.md`.

```python
h = meshioplusplus.hessian(vol, "T")                            # cell_data["T:hessian"], (n, 9)
curv = meshioplusplus.data_calc(h, "norm(`T:hessian`)", location="cell", output="curv")
adapted = meshioplusplus.refine(curv, where="curv > 3.0")       # refine where curvature is largest
```

#### Error estimation (Zienkiewicz-Zhu recovery + marking)

**`meshioplusplus.estimate_error`** estimates the per-cell recovered-gradient error of a `point_data` field, and can mark cells for refinement — the piece that closes the adaptive loop: `gradient` differentiates and selective `refine`'s `--where` consumes any scalar `cell_data` predicate; this produces one, so estimate → mark → refine is three verbs and no new numerical code. It is a **composition**, not a new kernel: `gradient` (Green-Gauss, cell location) → the measure-weighted point↔cell averaging round trip recovers a smoothed gradient, and `eta_K = sqrt(|measure| · sum((recovered − raw)²))` per cell is the standard ZZ indicator. `marking` turns it into a boolean `error:marked` array: `"absolute"` (threshold), `"fraction"` (top N by indicator), or `"dorfler"` (the smallest indicator-descending prefix covering a bulk fraction of the total — the usual AMR criterion). Cells that cannot be evaluated read NaN in the indicator and 0 (never NaN) in the marking array, counted and excluded from the global error. See `doc/error.md`.

```python
out, report = meshioplusplus.estimate_error(
    vol, "T", marking="dorfler", marking_value=0.6, return_report=True,
)
print(report["global_error"], report["num_marked"])
adapted = meshioplusplus.refine(out, where="error:marked > 0.5")
```

#### Field integration (total / mean, per-region)

**`meshioplusplus.data_integrate`** computes a cell-measure-weighted total and mean of one or more `cell_data` arrays — `gradient`'s integration counterpart (`gradient` differentiates a field, this integrates one), for a density field's total mass, a heat-flux field's total power, or an occupied volume. Every sum is weighted by `|measure(cell)|` (the cell's own length/area/volume); a cell whose measure is not computable, or a component whose value is non-finite, is excluded from that component's numerator **and** denominator — never given a fallback weight of 1, unlike `cell_data_to_point_data`'s own measure weighting, since a silent unit-weight substitution would corrupt a physical total. Reported for the whole mesh and independently for every named `Cell` region — regions are **not a partition**, so a cell in two regions contributes fully to both. A `point_data`-only name raises by name, pointing at `point_data_to_cell_data`. The mesh is never modified; this is a read-only report, like `data_info` and `compute_stats`. See `doc/field_integration.md`.

```python
report = meshioplusplus.data_integrate(mesh, arrays=["density"])
report[0]["domain"]["total_per_component"]   # sum(value * |measure|) -- total mass
report[0]["regions"]                         # the same, independently, per named Cell region
```

#### Data operations (rename / average / calc / condition / summarize)

A second bundle operates on the **data arrays** a mesh carries (`point_data` / `cell_data` / `field_data`) rather than on its geometry, which none of them ever modifies:

- **`meshioplusplus.data_rename` / `data_drop` / `data_keep`** — rewrite which arrays a mesh carries and under what names; values, dtypes and shapes are copied verbatim. See `doc/data_manage.md`.
- **`meshioplusplus.point_data_to_cell_data` / `cell_data_to_point_data`** — move data between locations by averaging, optionally weighted by cell area/volume. See `doc/data_average.md`.
- **`meshioplusplus.data_calc`** — derive a new array from an elementwise expression (`+ - * /`, parentheses, `abs`/`sqrt`/`min`/`max`/`norm`) evaluated by a hand-written parser — no external parser library, no arbitrary-code path. See `doc/data_calc.md`.
- **`meshioplusplus.data_condition`** — clamp, normalize to a target range, or standardize to zero mean / unit standard deviation, per component or by row magnitude. See `doc/data_condition.md`.
- **`meshioplusplus.data_info`** — a read-only per-array summary (dtype, shape, components, min/max/mean, NaN/inf counts) — the data-side complement to `info` and `compute_stats`. See `doc/data_info.md`.

<!--pytest-codeblocks:skip-->

```python
out = meshioplusplus.data_calc(mesh, "norm(velocity)", location="point", output="speed")
out = meshioplusplus.point_data_to_cell_data(out, keys=["speed"], suffix="_c")
out = meshioplusplus.data_condition(out, "cell", ["speed_c"], mode="normalize")
out = meshioplusplus.data_rename(out, "point", "T", "temperature")
arrays = meshioplusplus.data_info(out)                     # list of per-array dicts
```

These are likewise exposed across every binding surface, and as the nine CLI verbs under the `meshioplusplus data` group (`info`, `rename`, `drop`, `keep`, `to-cell`, `to-point`, `calc`, `clamp`, `normalize`). See `doc/data_operations.md`. A second nested group, `meshioplusplus dataset` (`add`, `list`, `split`, `tag`, `annotate`), curates the hand-editable [dataset manifests](https://loumalouomega.github.io/meshioplusplus/datasets.html) used for ML training collections (Python CLI only, like `data export`).

#### Time series

The [XDMF format](https://xdmf.org/index.php/XDMF_Model_and_Format) supports time series with a shared mesh. You can write times series data using meshio++ with

<!--pytest-codeblocks:skip-->

```python
with meshioplusplus.xdmf.TimeSeriesWriter(filename) as writer:
    writer.write_points_cells(points, cells)
    for t in [0.0, 0.1, 0.21]:
        writer.write_data(t, point_data={"phi": data})
```

and read it with

<!--pytest-codeblocks:skip-->

```python
with meshioplusplus.xdmf.TimeSeriesReader(filename) as reader:
    points, cells = reader.read_points_cells()
    for k in range(reader.num_steps):
        t, point_data, cell_data = reader.read_data(k)
```

#### Transient / multi-file datasets

Most formats cannot express time at all, so transient output usually arrives as
a *set* of files (`out_0000.vtu … out_0500.vtu`). meshio++ treats such a set as
one ordered dataset:

```bash
meshioplusplus convert 'out_*.vtu' series.xdmf         # fan-in (quote the glob!)
meshioplusplus convert series.xdmf 'step_{step}.vtu'   # fan-out
meshioplusplus pipeline transient.json                 # chain applied per step
```

<!--pytest-codeblocks:skip-->

```python
for time, mesh in meshioplusplus.read_sequence("out_*.vtu"):  # lazy
    ...
meshioplusplus.write_sequence("series.xdmf",
                              meshioplusplus.read_sequence("out_*.vtu"))
```

Ordering is **natural-numeric**, so `out_10.vtu` follows `out_9.vtu`; each step's
time comes from an explicit list, the file, its filename or its index, and which
one applied is reported. Fan-in and fan-out **stream** — one mesh is alive at a
time, whatever the step count — and a multi-step input aimed at a single-step
output is an error naming `{step}`, never a silent write of step 0. Available
from Python, both CLIs, C, Fortran, Julia and R. See
[`doc/sequences.md`](doc/sequences.md).

### Interactive viewer

[![Try it in your browser](https://img.shields.io/badge/try%20it-in%20your%20browser-4c9ffe?logo=webassembly&logoColor=white)](https://loumalouomega.github.io/meshioplusplus/viewer/)

<img alt="the meshio++ browser viewer" src="https://loumalouomega.github.io/meshioplusplus/viewer/browser-viewer.png" width="85%">

*The browser viewer — reading, rendering and converting entirely client-side.*

One call, two backends:

```python
import meshioplusplus

mesh = meshioplusplus.read("part.msh")
meshioplusplus.view(mesh)                      # pick a backend automatically
meshioplusplus.view(mesh, backend="polyscope") # a native desktop window
meshioplusplus.view(mesh, backend="browser")   # vtk.js, in a browser or notebook
```

The **desktop** backend is [Polyscope](https://polyscope.run), an optional Python-only extra (`pip install meshioplusplus[viewer]`). It draws solids you can slice into, colours by any point or cell array, and renders headless screenshots for CI and docs:

```python
meshioplusplus.screenshot(mesh, "part.png", color_by="temperature")
```

<img alt="the example bracket in Polyscope, coloured by scaled Jacobian" src="https://loumalouomega.github.io/meshioplusplus/viewer/desktop-viewer.png" width="70%">

*The bundled `example.msh` bracket coloured by element quality — this image is generated by `screenshot()` itself.*

The **[browser](https://loumalouomega.github.io/meshioplusplus/viewer/)** backend needs nothing extra. The same app is hosted as a live demo: drag in any supported format, colour by point or cell data, and convert and download to another format — all client-side, with no server and no upload. Since it runs the WebAssembly build, every format meshio++ reads works there too.

A second page, the **[dataset manager](https://loumalouomega.github.io/meshioplusplus/viewer/dataset.html)**, curates the [dataset manifests](https://loumalouomega.github.io/meshioplusplus/datasets.html) used for ML training collections visually — an overview of every manifest in a directory (cards with split balance, health badges and thumbnails, plus a manifest diff view, and — with an optional local companion process — server-side scans and PhysicsNeMo training launched, followed, compared and previewed from the page; see [the dashboard page](https://loumalouomega.github.io/meshioplusplus/dashboard.html)), directory picking with in-place manifest save (Chromium), per-entry previews with a time-series scrubber, and NaN/Inf/quality scanning — against the same hand-editable JSON the CLI and Python API use:

<img alt="the meshio++ dataset manager, previewing a transient case with a step scrubber and a per-array NaN/Inf summary table" src="https://loumalouomega.github.io/meshioplusplus/viewer/dataset-manager.png" width="85%">

*Splits, tags and groups on the left; a live preview and per-array summary on the right — all against the manifest JSON the CLI and Python API read and write.*

From the command line:

```sh
meshioplusplus view part.msh
meshioplusplus screenshot part.msh part.png --size 1600 1200
```

See [the viewer docs](https://loumalouomega.github.io/meshioplusplus/viewer.html) for how volume meshes are handled and what each backend can and cannot do.

### Interoperability

Hand a mesh straight to the tools you reach for next — no file round-trip, and the numpy buffers are **shared**, not copied, wherever the target accepts them as they are:

<!--pytest-codeblocks:skip-->

```python
import meshioplusplus

mesh = meshioplusplus.read("bracket.msh")

grid = meshioplusplus.to_pyvista(mesh)      # a pyvista.UnstructuredGrid
tm = meshioplusplus.to_trimesh(mesh)        # a trimesh.Trimesh (triangles only)
```

Both directions exist (`from_pyvista`, `from_trimesh`). Mixed-type meshes are the normal case, and named regions ride along as `region:<name>` mask arrays plus a metadata sidecar, so even a gmsh physical group's integer tag survives a PyVista round-trip. `zero_copy_only=True` turns any step that would copy into a named error instead of a silent one.

Data arrays also export to Apache Arrow and Parquet for the analytics stack, or straight to a pandas / polars frame with no Parquet detour:

<!--pytest-codeblocks:skip-->

```python
df = meshioplusplus.to_pandas(mesh, location="cell")    # a pandas.DataFrame
pf = meshioplusplus.to_polars(mesh, location="point")   # a polars.DataFrame

meshioplusplus.write_parquet(mesh, "cells.parquet", location="cell")

import pandas
pandas.read_parquet("cells.parquet").head()
```

<!--pytest-codeblocks:skip-->

```sh
meshioplusplus data export bracket.msh cells.parquet --location cell
```

Multi-component arrays keep their shape wherever the target can hold it — Arrow `fixed_size_list` columns, polars `pl.Array` columns — while `to_pandas` (whose columns are one-dimensional) flattens them into suffixed columns `v_0`/`v_1`/`v_2` with the grouping recorded in `df.attrs`. The mesh's counts, cell types and region names travel in the Arrow schema metadata / `df.attrs`. This is a **data** export, not a mesh format — it does not round-trip geometry and is deliberately not in the format registry.

PyVista, trimesh, pyarrow, pandas and polars are Python-only optional extras (`pip install meshioplusplus[interop]`, or one at a time with `[pyvista]` / `[trimesh]` / `[arrow]` / `[pandas]` / `[polars]`). They are kept out of `[all]`, which means "the optional dependencies the *formats* need". None of them reaches the C++/WebAssembly/C/Fortran core, which stays dependency-free.

Meshes also hand off to the GPU through the standard exchange protocols — DLPack as the primary export (which covers host arrays too), `__cuda_array_interface__` consumed on the way back:

<!--pytest-codeblocks:skip-->

```python
gpu = meshioplusplus.to_cupy(mesh)            # one host→device transfer per array
gpu.point_data["T"] *= 2.0                    # any CuPy / RAPIDS kernel
meshioplusplus.write("out.vtu", meshioplusplus.from_cupy(gpu))
```

The host→device move is always a bus transfer — what this removes is the file round-trip and every *extra* copy around it. `to_dlpack(mesh)` exports host arrays any DLPack consumer (PyTorch, JAX, Numba, …) adopts in place. There is deliberately no `[gpu]` extra: CuPy wheels are CUDA-version-specific, so install the one matching your toolkit (e.g. `pip install cupy-cuda13x` for CUDA 13.x, `cupy-cuda12x` for 12.x).

And for machine-learning pipelines, meshes become graphs, feature matrices, datasets and framework tensors directly:

<!--pytest-codeblocks:skip-->

```python
ei = meshioplusplus.edge_index(mesh)                  # (2, E) int64 — PyG/DGL layout
fm = meshioplusplus.feature_matrix(mesh, "point")     # (N, F) float64 + recorded columns
meshioplusplus.write_dataset("out_*.vtu", "dataset/") # mesh_id-keyed partitioned Parquet
t = meshioplusplus.to_torch(mesh)                     # torch tensors, adopted zero-copy
```

`feature_matrix`'s column order is a stated, versioned contract recorded in the returned schema, so training and inference cannot silently disagree; `write_dataset` streams a glob / directory / transient series into hive-partitioned Parquet (or chunked zarr/hdf5 groups, `[zarr]`/h5py) with a strict shared schema and a JSON manifest; `to_torch`/`to_jax` adopt the DLPack payload per framework (no `[torch]`/`[jax]` extra, deliberately — the CuPy precedent). See [the ML docs](https://loumalouomega.github.io/meshioplusplus/ml.html).

A *collection* of solution outputs is catalogued by a hand-editable [dataset manifest](https://loumalouomega.github.io/meshioplusplus/datasets.html) (`DatasetManifest` — sources, train/valid/test splits, tags, groups, notes; curated in Python, by the `meshioplusplus dataset` CLI group, over MCP, or with a text editor, all against the same JSON), and the [PhysicsNeMo adapter](https://loumalouomega.github.io/meshioplusplus/physicsnemo.html) (`meshioplusplus.physicsnemo`) trains straight off it: `graph_sample` builds the MeshGraphNet tensor set per mesh, `field_stats`/`edge_stats` stream normalization stats in PhysicsNeMo's own convention, `make_dataset` yields a PyTorch Geometric dataset and `make_reader` a Gen-2 `Reader`, and `run_training`/`predict` (also `python -m meshioplusplus.physicsnemo.train --spec`, and the dashboard's *Start training*) run the loop and write predictions back onto the mesh — with a worked, GPU-executed end-to-end example in [`example/physicsnemo/`](example/physicsnemo/).

See [the interoperability docs](https://loumalouomega.github.io/meshioplusplus/interop.html) for the full mapping tables, the zero-copy contract, and the Open3D/DOLFINx design sketch, and [the GPU docs](https://loumalouomega.github.io/meshioplusplus/gpu.html) for the device handoff.

For a **grid-shaped** model — a 3-D CNN, or a superresolution network taking a coarse grid to a fine one — the shape a mesh has to become is a dense array, and `sample_grid` is that step:

```python
spec  = mio.GridSpec.from_mesh(mesh, resolution=(64, 64, 64))
array = mio.sample_grid(mesh, spec, fields=["T", "vel"])   # (C, D, H, W), channels first then z,y,x
array.coverage                                             # how much of the grid is actually inside the mesh
result = mio.scatter_grid(prediction, mesh)                # and back onto the mesh, trilinearly
```

Grids are ordinary hexahedron meshes, so `view`, `crop` and every writer work on one; cache them as `.vti`, which stores the lattice exactly. `power_spectrum` reports whether a super-resolved field carries the right small-scale content, which a pointwise error cannot see. See [mesh and regular grids](https://loumalouomega.github.io/meshioplusplus/grids.html).

For a **point-cloud** model — Transolver, FLARE, DoMINO, anything whose cost is quadratic in its token count — the shape is a fixed number of points, and `select_points` is that step:

```python
budget = mio.select_points(mesh, 4096)                     # farthest-point sampling; method="grid" scales to millions
tokens = budget.take(mio.feature_matrix(mesh).matrix)      # (4096, F), the column contract intact
cloud  = mio.subsample_points(mesh, budget)                # a vertex-block point cloud every writer accepts
```

The selection is kept in selection order, so one farthest-point budget at 8192 serves 4096 and 2048 by slicing. See [point-cloud budgets](https://loumalouomega.github.io/meshioplusplus/point_budgets.html).

For a **particle** method there is no connectivity to build a graph from at all — a smoothed-particle or discrete-element state is a cloud of positions, and what makes two particles interact is the interaction radius:

```python
edges = mio.proximity_graph(cloud, radius=0.015)                    # or method="knn", max_neighbors=16
edges = mio.proximity_graph(cloud, radius=0.015, box_size=[2, 2, 2])  # periodic: minimum image
attrs = mio.edge_vectors(cloud.points, edges)                       # displacement, then its norm
```

For a **transient** surrogate the shape is a window — the last K states of every node, oldest first — and `iter_windows` builds them in three schemes over a manifest, while `rollout` feeds a model its own predictions back and reports how the error grows, which a one-step validation loss cannot show. `Augmentation` randomizes each case's pose per epoch, drawn deterministically from `(seed, epoch, index)` and replayed coherently across a paired input and target.

A trained model answers any mesh it is given, and the answer for a part unlike anything it saw is finite, plausible and wrong. `GeometryGuard` is the check it cannot make for itself — fit a description of the shapes a dataset contains, score a new one against it, and get back the descriptors that put it there. The descriptors are deliberately not scale- or position-invariant, because a scaled part is a different part.

Once a model is trained, applying it to one mesh needs no manifest at all — `meshioplusplus predict model.mdlus part.vtu part_pred.vtu`, or `predict_file` from Python. Everything comes from the checkpoint's own model card, and a mesh carrying no truth predicts anyway rather than reporting an error against itself.

The same call adds *world* edges beside a mesh's own — contact between surfaces that are near but not connected — and `bistride_hierarchy` coarsens a graph so a signal crosses the mesh in logarithmically many message-passing steps. See [proximity graphs](https://loumalouomega.github.io/meshioplusplus/proximity_graphs.html).

If PhysicsNeMo itself is unfamiliar, [**PhysicsNeMo basics**](https://loumalouomega.github.io/meshioplusplus/physicsnemo/overview.html) is a fourteen-page map of the framework — what a `Module` and a `.mdlus` checkpoint are, which of the 25 architecture families fits the shape of your data, how simulation output becomes batched tensors, and where meshio++ ends and the framework begins. Each page closes by naming what meshio++ supplies, or by saying plainly that nothing does.

### MCP server

Every operation in this README is also exposed to AI agents as a tool over the [Model Context Protocol](https://modelcontextprotocol.io/) — reading/writing all the formats, conversion, and the full mesh- and data-operation suite:

```sh
pip install "meshioplusplus[mcp]"     # the mcp SDK needs Python >= 3.10
claude mcp add meshioplusplus -- meshioplusplus-mcp
```

Then ask the agent to convert, inspect, slice, partition, … and it drives the 74 tools itself. Tools are stateless and file-path based (optionally sandboxed with `--root DIR`), and every report is strict JSON. `meshioplusplus-mcp --http` (`pip install "meshioplusplus[dashboard]"`) serves the same tools over HTTP — MCP over streamable HTTP for agents, plus the JSON API the browser [dataset dashboard](https://loumalouomega.github.io/meshioplusplus/dashboard.html) uses as its local companion process. See [the MCP docs](https://loumalouomega.github.io/meshioplusplus/mcp.html) for the tool table and client setup.

### Blender add-on

Blender ships Python and reads almost no FEA formats. The add-on puts all 43 of meshio++'s behind `File > Import`.

It is a Blender 4.2+ **extension**, so the meshio++ wheel travels inside the zip — no pip step, no network at install time, nothing to configure. Download `meshioplusplus-<version>-<platform>.zip` from the [latest release](https://github.com/loumalouomega/meshioplusplus/releases) and drag it into a Blender window, or use _Edit / Preferences / Add-ons / Install from Disk_.

Volume meshes arrive as their boundary surface — Blender has no tetrahedron — with each `cell_data` array carried through to the faces of the cell that owned it, so a solid can still be coloured by its material. **Quads and n-gons are kept, not triangulated.** `point_data`, `cell_data` and named regions land as Blender attributes on the right domain.

The same bridge is two public functions, usable from Blender's scripting console: `to_blender(mesh)` and `from_blender(obj)`. See [the Blender docs](https://loumalouomega.github.io/meshioplusplus/blender.html).

### ParaView plugin

<img alt="gmsh paraview" src="https://nschloe.github.io/meshio/gmsh-paraview.png" width="60%">
*A Gmsh file opened with ParaView.*

If you have downloaded a binary version of ParaView, you may proceed as follows.

- Install meshio++ for the Python major version that ParaView uses (check `pvpython --version`)
- Open ParaView
- Find the file `paraview-meshioplusplus-plugin.py` of your meshio++ installation (on Linux: `~/.local/share/paraview-5.9/plugins/`) and load it under _Tools / Manage Plugins / Load New_
- _Optional:_ Activate _Auto Load_

You can now open all meshio++-supported files in ParaView.

### Benchmarks

How much does the C++ core help? The [`benchmark/`](https://github.com/loumalouomega/meshioplusplus/tree/main/benchmark) folder times read/write conversions against the original pure-Python [meshio](https://github.com/nschloe/meshio) on the formats both support (same in-memory mesh, same machine). The headline input is the bundled [`example.msh`](https://github.com/loumalouomega/meshioplusplus/blob/main/example/example.msh) — a real Gmsh bracket (~52k nodes, ~293k cells).

<img alt="meshio vs meshio++ speedup on example.msh" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/benchmark/plots/benchmark_speedup.svg" width="85%">

meshio++'s biggest wins are the parallel and text paths: **VTU binary+zlib ~16× write** (the zlib blocks run across cores via an OpenMP backend with dynamic scheduling — hybrid P+E-core CPUs load-balance too), **VTU ASCII ~7× write / ~5× read**, and mixed-topology **XDMF read ~10×**. The binary and HDF5 formats that used to be *slower* — VTK/Gmsh binary, UGRID, and MED — are now at or above parity after an optimisation pass (bulk-buffered binary I/O, single-instruction `bswap` endianness conversion, a real parallel backend, an Eigen-backed MED transpose, **zero-copy cell reconstruction** that moves the connectivity buffer straight into the mesh, and uninitialised reader buffers + thread-parallel block copies so nothing is written twice); binary **reads** now match or beat numpy's `fromfile` — Gmsh ~1.7×, single-type VTK ~1.45×, and even mixed-topology VTK ~1.1×. Output stays byte-identical throughout.

The speedup is per-element: text/parallel formats climb out of the small-mesh regime and plateau (large meshes realise the full speedup):

<img alt="speedup vs mesh size" src="https://raw.githubusercontent.com/loumalouomega/meshioplusplus/main/benchmark/plots/benchmark_scaling.svg" width="85%">

Full methodology and a reproducible notebook are on the [Benchmarks](https://loumalouomega.github.io/meshioplusplus/benchmarks) doc page (source: [`benchmark/01_benchmark.ipynb`](https://github.com/loumalouomega/meshioplusplus/blob/main/benchmark/01_benchmark.ipynb)).

### Reading only what you need

```python
import meshioplusplus

mesh = meshioplusplus.read("big.vtu", points_only=True)   # geometry, no data arrays
mesh = meshioplusplus.read("big.vtu", arrays=["u", "p"])  # only these arrays
meta = meshioplusplus.read_metadata("big.vtu")            # counts/names, no heavy arrays

mesh = meshioplusplus.read("run.exo", time_step=-1)       # the last step of a time series
meta["time_values"]                                       # how many steps there are
```

VTU, VTP, XDMF and Gmsh skip the unwanted array bodies outright; other formats are read in full and filtered, and `meta["fell_back_to_full_read"]` says which happened. `time_step` picks one step of a multi-step file (`0` = the first, negative counts from the end); out of range is an error naming the available count rather than a silent fallback to step 0. Currently honoured by Exodus. A `lenient` option downgrades "this reader cannot represent construct X" errors to a warning plus a skip (currently MDPA's `Table`/`Geometries`/`Mesh`/`Constraints` blocks) — not "ignore all errors": a malformed file still fails. Large files can also be memory-mapped (automatic above 16 MiB), which roughly halves peak memory during a read. See [selective reads](doc/selective_read.md) and [memory-mapped reading](doc/mmap.md).

VTK XML output can additionally use **lz4** (ParaView-readable) or **zstd** (a meshio++ extension) instead of zlib, when built with `-DMESHIOPLUSPLUS_WITH_LZ4=ON` / `-DMESHIOPLUSPLUS_WITH_ZSTD=ON`. zlib remains the default. See [compression codecs](doc/codecs.md).

### Installation

meshio++ is [available from the Python Package Index](https://pypi.org/project/meshioplusplus/), so simply run

```
pip install meshioplusplus
```

to install.

Additional dependencies (`netcdf4`, `h5py`) are required for some of the output formats and can be pulled in by

```
pip install meshioplusplus[all]
```

For JavaScript / browser use, the C++ core also ships as a WebAssembly npm package covering every format above — including the HDF5- and netCDF-backed ones (CGNS, H5M, HMF, MED, Exodus), as of v8.0.0:

```
npm install @meshioplusplus/wasm
```

See the [WebAssembly / JavaScript](https://loumalouomega.github.io/meshioplusplus/wasm) doc page for usage and the format-support table.

meshio++ is also on [Spack](https://spack.io), upstream in [`spack/spack-packages`](https://github.com/spack/spack-packages/pull/5624) — no checkout of this repo required:

```
spack install py-meshioplusplus +hdf5 +netcdf +zlib
```

See [Installation → Spack](https://loumalouomega.github.io/meshioplusplus/installation#spack) for the standalone C API package and the full variant list.

### C++ API

The full C++ core installs as a normal CMake package — the real `Mesh`, the format registry, every mesh/data operation, and the header-only Kratos bridge, none of which fit through a C ABI:

```
cmake -S . -B build -DMESHIOPLUSPLUS_BUILD_PYTHON=OFF -DMESHIOPLUSPLUS_INSTALL_CPP=ON
cmake --build build && cmake --install build --prefix /opt/meshioplusplus
```

```cmake
find_package(meshioplusplus 10.34.0 EXACT CONFIG REQUIRED COMPONENTS CXX)
target_link_libraries(my_solver PRIVATE meshioplusplus::core)
```

`EXACT` is the conservative pin: the C++ API makes **no ABI promise** (`Mesh`,
`ModelPart` and `GeometricalEntity` are header-defined types whose layout moves
with the headers), so library and consumer must be built from compatible
headers. The finer pin is `MESHIOPLUSPLUS_ABI_VERSION`, which moves only when a
change really would break an already-compiled consumer — so a release that
cannot affect you costs no rebuild. Either way a mismatch now fails at **link**
time rather than corrupting memory, and the C API is the stable one — pin
`find_package(meshioplusplus 10 … COMPONENTS C)` there. See
[ABI compatibility](https://loumalouomega.github.io/meshioplusplus/abi).

```cpp
#include "meshioplusplus/registry.hpp"
#include "meshioplusplus/operations/partition.hpp"

auto mesh  = meshioplusplus::registry_read("bracket.msh", "", {});
auto parts = meshioplusplus::partition(mesh, {.mNParts = 8});
```

All three [mesh backends](#c-mesh-backends) install side by side (`meshioplusplus::core_meshio`, `::core_native`, `::core_kratos`), so one prefix serves consumers that disagree about the backend; each carries its own backend macro, making a mismatch a compile or link error rather than silent UB. Kratos consumers get the whole `ModelPart` surface: application entity names such as `SmallDisplacementElement3D4N` are preserved end to end (file → `Mesh` → `ModelPart` → file), material data crosses as `Properties` key/value pairs, and nested SubModelParts round-trip as `parent/child` region names. meshio++ itself is **serial** — there is no MPI anywhere in the API; `partition(mesh, {nparts, ghost_layers})` produces the shared-node halo an MPI assembly needs and each rank takes its own piece. See the [C++ API](https://loumalouomega.github.io/meshioplusplus/cpp_api) doc page.

### C / Fortran API

For HPC codes written in C or Fortran, the C++ core also builds as an installable shared library (`libmeshioplusplus`, pure-C99 header, pkg-config + `find_package` support) with a modern OO Fortran 2008 module on top:

```
./build/configure.sh --fortran --tests --build     # --c-api for the C API alone
cmake --install build/cpp-release --prefix /opt/meshioplusplus
```

```c
mio_mesh* m = mio_read("in.msh", NULL);
printf("%lld points\n", (long long)mio_mesh_num_points(m));
mio_write("out.vtu", m, NULL);
mio_mesh_free(m);
```

```fortran
use meshioplusplus
type(mio_mesh) :: m
call m%read("in.msh")
call m%write("out.vtu")
call m%free()
```

The C API is also packaged for **Conan** (root [`conanfile.py`](conanfile.py)) and **vcpkg** (overlay port under [`packages/vcpkg/meshioplusplus/`](packages/vcpkg/meshioplusplus)), both driving the same install/`find_package` path, plus **Spack** (upstream in [`spack/spack-packages`](https://github.com/spack/spack-packages/pull/5624), no checkout needed):

```
conan create . -o meshioplusplus/*:with_hdf5=True
vcpkg install meshioplusplus --overlay-ports=ports
spack install meshioplusplus +fortran +hdf5
```

Full mesh access (build meshes from raw arrays, zero-copy readback) is covered on the [C API](https://loumalouomega.github.io/meshioplusplus/c_api) and [Fortran](https://loumalouomega.github.io/meshioplusplus/fortran) doc pages.

### Julia / R bindings

The same installed C library also carries bindings for **Julia** and **R**, the two remaining languages of the scientific-computing audience. Both are layered on `libmeshioplusplus` exactly as the Fortran module is — no new C++, and the core stays untouched:

```julia
import MeshioPlusPlus as mio
m = mio.read("bracket.msh")
mio.write(mio.extract_surface(m), "surface.vtu")
```

```r
library(meshioplusplus)
m <- mio_read("bracket.msh")
mio_write(mio_extract_surface(m), "surface.vtu")
```

Julia and R are both **column-major**, so — as in Fortran — points shaped `(dim, n)` and connectivity `(nodes_per_cell, n)` are the *same memory* as the C API's row-major shapes, and nothing is ever transposed. Node indices are **1-based**, with the shift applied inside the copying accessors only; Julia additionally exposes genuine zero-copy borrows (`points_ptr`, `connectivity_ptr`) whose validity window is enforced rather than merely documented. R is **copy-only** — R vectors are R-managed, so a borrow cannot survive into R — and says so plainly instead of implying parity.

> [!IMPORTANT]
> **The Julia binding is not MIT.** [`bindings/julia/`](bindings/julia) is released under the **GNU General Public License, version 3 (GPL-3.0)** — a copyleft license, not a permission-required one: anyone may use, modify or sell it commercially with no permission needed, but distributing it or a modified version of it must be under GPL-3.0 too, with source available; purely private use carries no obligation. Everything else in this repository, including the C API it calls and the R binding, remains MIT.

See the [Julia](https://loumalouomega.github.io/meshioplusplus/julia) and [R](https://loumalouomega.github.io/meshioplusplus/r) doc pages.

### Single-header C++

The whole C++ core is also amalgamated into one self-contained, [STB](https://github.com/nothings/stb)-style header — [`src/single_include/meshioplusplus/meshioplusplus.hpp`](src/single_include/meshioplusplus/meshioplusplus.hpp) — with pugixml bundled and no external dependencies by default. Drop it in, no CMake or linking required:

```cpp
// in exactly ONE .cpp:
#define MESHIOPLUSPLUS_IMPLEMENTATION
#include "meshioplusplus/meshioplusplus.hpp"
// elsewhere: just #include it (declarations only)
```

```
g++ -std=c++20 -I src/single_include main.cpp
```

It is generated by `./tools/amalgamate.sh` and kept in sync by CI. See the [single-header](https://loumalouomega.github.io/meshioplusplus/single_header) doc page (optional HDF5/netCDF/zlib formats via `MESHIOPLUSPLUS_HAS_*` macros).

[`example/cpp/`](example/cpp/) is the C++ counterpart of [`example/python/`](example/python/): the same tour of meshio++, called directly against this single header instead of the Python bindings, on the [xeus-cpp](https://github.com/compiler-research/xeus-cpp) Jupyter kernel — no PyVista either, renders go through meshio++'s own SVG writer. [`example/julia/`](example/julia/) and [`example/r/`](example/r/) are the same tour again, called through the Julia and R bindings on their own Jupyter kernels ([IJulia](https://github.com/JuliaLang/IJulia.jl) / [IRkernel](https://irkernel.github.io/)) — since those flat-ABI bindings can't drive the SVG writer's data-driven colouring either, quality/field renders are small charts instead of colour.

### C++ mesh backends

Standalone C++ builds (no Python) can swap the in-memory mesh structure at compile time via `MESHIOPLUSPLUS_MESH_BACKEND` — every format works identically under each backend:

- **MESHIO** (default; the Python extension and PyPI wheels always use it) — mirrors the Python `meshio.Mesh`;
- **NATIVE** — the fastest pure-C++ structure (canonical Float64/Int64 storage, cell-type enum, CSR ragged blocks); the WebAssembly build uses it;
- **KRATOS** — a [Kratos Multiphysics](https://github.com/KratosMultiphysics/Kratos)-style `ModelPart` (Nodes/Elements/Conditions/SubModelParts) plus a header-only templated bridge that populates a real `Kratos::ModelPart` with no Kratos build dependency.

```
./build/configure.sh --mesh-backend NATIVE --tests --build
```

All three also install side by side from a single prefix (`MESHIOPLUSPLUS_INSTALL_CPP=ON`), as `meshioplusplus::core_meshio` / `::core_native` / `::core_kratos` — see the [C++ API](https://loumalouomega.github.io/meshioplusplus/cpp_api) page. See the [C++ mesh backends](https://loumalouomega.github.io/meshioplusplus/cpp_backends) doc page.

### Testing

To run the meshio++ unit tests, check out this repository, install it with the test extras, and type

```
pytest tests/python/
```

### License

meshio++ is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License), with **one exception**: the Julia binding in [`bindings/julia/`](bindings/julia) is released under the [GNU General Public License, version 3](bindings/julia/LICENSE) (GPL-3.0). GPL-3.0 is copyleft, not permission-required: anyone may use, modify or sell it commercially without asking, but distributing it (or a modified version) must be under GPL-3.0 too, with source available; purely private/internal use carries no obligation at all. Nothing else is affected: the C++ core, the C API that binding calls, and the R binding are all MIT.

### Acknowledgements

**meshio++ is a fork of [meshio](https://github.com/nschloe/meshio)** by [Nico Schlömer](https://github.com/nschloe) and its contributors (MIT). meshio is where the `Mesh` data model, the cell-type naming, and the great majority of the format readers and writers come from; this fork adds a C++20 core, an operations layer, and the C/Fortran/WebAssembly bindings on top of that foundation. The original copyright is retained in [`LICENSE`](LICENSE).

Some code and test fixtures also come from [Simvia's `meshlane` fork](https://github.com/simvia-tech/meshlane) of meshio (MIT), credited per-change in [`CITATION.cff`](CITATION.cff) and [`CHANGELOG.md`](CHANGELOG.md).

The C++ core is dependency-free by design. Everything below is either optional, bundled, or confined to one binding or tool.

#### Runtime dependencies (Python)

| Project | Used for | License |
| --- | --- | --- |
| [NumPy](https://numpy.org/) | the array type the whole data model is built on | BSD-3-Clause |
| [Rich](https://github.com/Textualize/rich) | CLI output formatting | MIT |

#### Optional dependencies

| Project | Extra | Used for | License |
| --- | --- | --- | --- |
| [Polyscope](https://polyscope.run/) | `[viewer]` | the desktop viewer and `screenshot()` | MIT |
| [PyVista](https://pyvista.org/) | `[pyvista]` / `[interop]` | `to_pyvista()` / `from_pyvista()` | BSD-3-Clause |
| [trimesh](https://trimesh.org/) | `[trimesh]` / `[interop]` | `to_trimesh()` / `from_trimesh()` | MIT |
| [pyarrow](https://arrow.apache.org/) | `[arrow]` / `[interop]` | the Arrow/Parquet data export | Apache-2.0 |
| [pandas](https://pandas.pydata.org/) | `[pandas]` / `[interop]` | `to_pandas()` | BSD-3-Clause |
| [polars](https://pola.rs/) | `[polars]` / `[interop]` | `to_polars()` | MIT |
| [zarr](https://zarr.dev/) | `[zarr]` | the `write_dataset(format="zarr")` chunked-dataset backend | MIT |
| [CuPy](https://cupy.dev/) | — (wheels are CUDA-version-specific, e.g. `cupy-cuda13x` — see the GPU docs) | `to_cupy()` / `from_cupy()` | MIT |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | `[mcp]` | the `meshioplusplus-mcp` server exposing every operation to AI agents | MIT |
| [h5py](https://www.h5py.org/) | `[all]` | the Python fallback for CGNS, H5M, MED, XDMF | BSD-3-Clause |
| [netCDF4](https://unidata.github.io/netcdf4-python/) | `[all]` | the Python fallback for Exodus | MIT |
| [KaHIP](https://kahip.github.io/) | `[kahip]` / CMake | the quality graph-partitioning backend | MIT |
| [zstandard](https://github.com/indygreg/python-zstandard), [lz4](https://github.com/python-lz4/python-lz4) | `[codecs]` | the Python fallback for the optional VTK block codecs | BSD-3-Clause / BSD-2-Clause |

#### Bundled and build-time

| Project | Used for | License |
| --- | --- | --- |
| [pugixml](https://pugixml.org/) | XML parsing in the C++ core (vendored in `src/cpp/third_party/`) | MIT |
| [Eigen](https://eigen.tuxfamily.org/) | the MED Fortran↔C transpose (git submodule, optional) | MPL-2.0 |
| [pybind11](https://github.com/pybind/pybind11) | the Python bindings | BSD-3-Clause |
| [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | the CMake-driven build backend | Apache-2.0 |
| [Emscripten](https://emscripten.org/) | the WebAssembly build | MIT / NCSA |
| [GoogleTest](https://github.com/google/googletest) | the C++ test suite | BSD-3-Clause |
| [zlib](https://zlib.net/), [Zstandard](https://facebook.github.io/zstd/), [LZ4](https://lz4.org/) | optional compression codecs | zlib / BSD-3-Clause / BSD-2-Clause |
| [HDF5](https://www.hdfgroup.org/solutions/hdf5/), [netCDF](https://www.unidata.ucar.edu/software/netcdf/) | optional native paths for HDF5/netCDF-backed formats | BSD-3-Clause / MIT-like |

#### Browser viewer (isolated under [`src/viewer/`](src/viewer/))

| Project | Used for | License |
| --- | --- | --- |
| [vtk.js](https://kitware.github.io/vtk-js/) | all rendering, colour maps and the scalar bar | BSD-3-Clause |
| [Vite](https://vite.dev/) | the app build | MIT |
| [vite-plugin-singlefile](https://github.com/richardtallent/vite-plugin-singlefile) | the self-contained wheel-bundled build | MIT |
| [Playwright](https://playwright.dev/) | the end-to-end tests and the documentation screenshots | Apache-2.0 |

#### Also

[Kratos Multiphysics](https://github.com/KratosMultiphysics/Kratos) (BSD-3-Clause) is the source of the skin-detection algorithm, the EnSight writer logic, the KaHIP partitioning approach and the `FindKaHIP.cmake` module, all these 3 implementation are from the original author of this project as well. Also thanks to its `ModelPart` design informs the KRATOS mesh backend. [VTK](https://vtk.org/) and [Verdict](https://github.com/sandialabs/verdict) (both BSD-3-Clause) define the mesh-quality formulas. The documentation is built with [VitePress](https://vitepress.dev/) (MIT) and [Doxygen](https://www.doxygen.nl/) (GPL-2.0, used as a tool only). The logo renders the [Stanford Bunny](https://www.thingiverse.com/thing:88208) ("Stanford Bunny — Digitized!" by MakerBot, CC-BY).

Thank you to all of them.
