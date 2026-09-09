---
title: Mesh and geometry
description: physicsnemo.mesh - the representation, the calculus on it, generating geometry from implicit functions, and where it meets a general mesh library.
---

# Mesh and geometry

`physicsnemo.mesh` is the largest subpackage, and the one where the impedance mismatch with a general mesh library is biggest.

## The representation, and the one thing to know about it

`physicsnemo.mesh.Mesh` is **points plus simplices plus fields**. Simplices: line segments, triangles, tetrahedra. That is the whole vocabulary.

Real meshes are not simplicial. Hexahedra, prisms, pyramids, quadrilaterals and every quadratic geometry have to be **tessellated** into simplices before PhysicsNeMo will look at them, and that is where the subtlety is: tessellating each element independently splits shared faces along contradictory diagonals, leaving gaps between neighbours that no amount of downstream care recovers.

`DomainMesh` adds named boundaries to a `Mesh` — the natural target for a mesh's named regions.

Meshes save and load in a memory-mapped format (`.pmsh`), which is what `MeshDataset` reads.

![The Mesh tensorclass with points, cells and three field dictionaries](/images/physicsnemo/mesh_data_model.svg)

*Figure 1: The data model. Field rank lives in the tensor shape; the type is parametrized by two dimensions; cells are simplices and nothing else.*

## What is in there

| Submodule | What it does |
|---|---|
| `tessellation` | `triangulate`, `fill_interior` — simplices from polygons and surfaces |
| `calculus` | gradient, divergence, curl, Laplacian, integrals on a mesh — LSQ and discrete-exterior-calculus backends, autograd-differentiable |
| `generate` | implicit geometry: `sdf_box`/`sdf_sphere`-style primitives, `sdf_union`/`sdf_difference`/`sdf_intersection` combinators, `marching_cubes`, `mesh_implicit_domain`, `refit_mesh_to_implicit` |
| `spatial` | `signed_distance_field` (a 3-tuple since 2.2: distances, hit points, hit faces), `BVH` for containing-cell and nearest-facet queries, `ClusterTree` for Barnes-Hut style far-field aggregation |
| `remeshing` | `remesh` (Warp-backed) and `partition_cells` surface clustering |
| `deformation` | mesh-quality energies: strain, measure, bending, and **simplex inversion** — the term that stops an optimizer tearing the mesh |
| `geometry` | areas, normals, circumcenters, cotangent weights, dual volumes |
| `neighbors` | point-to-cell, cell-to-cell and point-to-point adjacency |
| `boundaries` | facet extraction and boundary categorization |
| `sampling` | containing-cell search, barycentric coordinates, point sampling |
| `repair` | hole filling, orientation fixing, duplicate and degenerate removal |
| `subdivision` | linear, loop and butterfly refinement |
| `transformations` | rotate, scale, translate, deform |
| `curvature` | `mean_curvature_vertices`, `gaussian_curvature_vertices` (cotangent Laplace-Beltrami) — a natural node feature next to the SDF |
| `smoothing` | `smooth_laplacian` |
| `projections` | `extrude` (an N-D mesh swept into N+1), `embed`, `project` |
| `primitives` | canonical meshes (cubes, spheres, planar shapes, procedural surfaces) for tests and demos |
| `validation` | `validate`, `quality_metrics`, `statistics` |
| `visualization` | `draw` through matplotlib or pyvista |
| `io` | `from_pyvista`/`to_pyvista` (auto-triangulates polyhedral cells — no provenance), `to_zarr`/`from_zarr` (2.2) |

## Two upstream behaviours worth knowing

The **mesh-calculus gradient layout flipped** between releases: 2.1's least-squares backend was channel-major, 2.2 is derivative-first `(N, D, C)` from every backend. A layout change that keeps shapes identical passes every test whose fixture is symmetric, which is exactly how it goes unnoticed — a canary gradient must be asymmetric and non-square to catch it.

**Boundary surfaces come out inconsistently wound.** Anything using an extracted surface for a signed distance has to re-orient it first, or the sign of the distance field flips from patch to patch.

## In meshio++

This is the page where the two libraries overlap most, so it is worth being precise about what each is for. PhysicsNeMo's mesh package exists to make a mesh *differentiable* — every operation on it carries a gradient back to a model. meshio++'s exists to make a mesh *portable and correct* — forty-odd formats, every cell type, exact conservation, no framework. They are complements, and the honest division is: read, convert, repair and measure with meshio++; put the result on a device and differentiate it with physicsnemo.

Where meshio++ has a direct counterpart:

| PhysicsNeMo | meshio++ |
|---|---|
| `Mesh`, `DomainMesh` | [`to_physicsnemo`, `from_physicsnemo`](../physicsnemo.md) — the bridge, one topological dimension at a time |
| `calculus` | [`gradient`](../gradient.md), [`hessian`](../hessian.md), [`data_integrate`](../field_integration.md) — exact for a linear field, not differentiable |
| `generate` | [`compute_sdf`](../sdf.md) and [`isosurface`](../isosurface.md) — an SDF lattice and its contour |
| `spatial.signed_distance_field` | [`sample_distance`](../sdf.md) — with the angle-weighted pseudonormal sign a nearest-triangle normal gets wrong at a spike |
| `remeshing` | [`remesh`](../remesh.md), [`remesh_volume`](../remesh_volume.md), [`optimize_volume`](../optimize_volume.md) |
| `repair` | [`repair`](../repair.md) — orientation, hole filling, bowtie splitting (v10.38.0); [`clean`](../clean.md) for the weld, degenerate/duplicate drop and orphan prune |
| `transformations.deform.shrinkwrap` | [`shrinkwrap`](../shrinkwrap.md) (v10.38.0) — one projection, offset along the hit feature's pseudonormal rather than the selected triangle's |
| `transformations.deform.sobolev_deform` | [`sobolev_deform`](../sobolev_deform.md) (v10.38.0) — the same screened-Poisson filter over the same P1 operators, not differentiable |
| `subdivision` | [`refine`](../refine.md), [`subdivide`](../subdivide.md), and [`agglomerate`](../agglomerate.md) going the other way |
| `boundaries` | [`extract_surface`, `extract_skin`](../extract_surface.md), and [named regions](../regions.md) for the naming |
| `validation` | [`compute_quality`](../mesh_quality.md), [`compute_stats`](../stats.md) |
| `primitives` | [`grid`](../voxelize.md); the rest are [on the roadmap](../roadmap.md) |
| `.pmsh` | [`pmsh`](../formats/pmsh.md) (v10.35.0) — the memmap layout written in pure numpy, so a box with no torch can produce a training set; [`zarr`](../formats/zarr.md) writes the chunked alternative `MeshReader` also accepts |

**Tessellation, and the thing that goes wrong.** [`convert_cells(mode="simplexify")`](../convert_cells.md) is meshio++'s answer to the simplices-only constraint: quadrilaterals fan into triangles, hexahedra into six tetrahedra by a canonical Freudenthal fan around the main diagonal 0–6, wedges into three, pyramids into two, and higher-order cells are linearized first. The diagonal is **fixed rather than chosen per cell**, which is precisely the conformity point above: two neighbouring hexahedra agree on how their shared face splits because neither of them chose.

meshio++ now has the other half too — [`tessellate`](../tessellation.md) (v10.39.0) isoparametrically subdivides a curved cell (`triangle6`/`quad8`/`quad9`/`tetra10`/`hexahedron27`) through synthetic points on a refinement lattice, interpolated on the way in, and records a full **provenance map** (`Tessellation.source_point`/`source_cell`) back to the cell each synthetic point/simplex was carved out of. `record_parent_ids` still gives a per-conversion parent index; `Tessellation.gather`/`.scatter`/`.aggregate` are the general "carry a prediction back through a tessellation" facility — a prediction made on a tetrahedron is written onto the hexahedron it was carved out of by `scatter`ing it through that map, and `Graph.Tessellate` wires the same thing into `graph_sample`/training/prediction directly.

**Generation** goes the other way too. `remesh_volume` takes a closed surface and produces a genuinely new tetrahedral mesh (isosurface stuffing over a body-centred cubic lattice), so an SDF-defined shape can be contoured with `isosurface`, meshed, and handed to a solver — without a Delaunay predicate kernel anywhere in the chain.

Next: [Symbolic and physics](./symbolic_and_physics.md).
