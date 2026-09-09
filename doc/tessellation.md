# Curved-cell tessellation

`tessellate(mesh, levels=2, curved=True, fields=True, record_stencil=False)` isoparametrically subdivides a mesh's curved cells onto a regular refinement lattice, returning a `Tessellation` — the linearized output mesh plus a recorded provenance map from every synthetic point/cell back to the original geometry it was carved out of.

It closes roadmap section 1's last open bullet: [`convert_cells(mode="simplexify")`](/convert_cells) already linearizes and splits conformingly, but it keeps only a curved cell's corner nodes (the mid-side/face/body nodes' curvature is dropped entirely) and records no per-simplex provenance at all.

```python
import meshioplusplus as mp

mesh = mp.read("curved_bracket.msh")
tess = mp.tessellate(mesh, levels=2)
mp.write("curved_bracket_tessellated.vtu", tess.mesh)
```

Pure Python over existing machinery — `_convert_cells`'s tables, `_data_average`'s cell measures, `_regions.block_bases` — matching every other module in this section of the roadmap (`_point_budget.py`, `_proximity.py`, `_guard.py`). No C++, ABI, WASM or binding change.

## Scope: five curved types

`CURVED_TYPES` is exactly `("triangle6", "quad8", "quad9", "tetra10", "hexahedron27")` — the serendipity/full-Lagrange quadratics this repo's own tables (`_convert_cells._ELEVATE`, `_refine_templates.QUAD_FACES`) already describe a basis for.

`hexahedron20` is **deliberately excluded**: its faces are `quad8` while `hexahedron27`'s are `quad9` — they disagree on the face *interior* (`quad9` has a face-centre node, `quad8` does not), and curving both would let two neighbouring cells' watertightness keys silently merge two different face-interior positions into one point.

Every other cell — linear types, `hexahedron20`, `wedge15`/`18`, `pyramid13`/`14`, the VTK-Lagrange family, ragged and polyhedron blocks — passes through into the output mesh **completely unchanged**: same points (by index), same connectivity, same `cell_data` row. This is what makes `tessellate(linear_mesh)` an exact identity, pinned by the "no-op on a linear mesh" test.

An earlier design considered simplexifying every cell (curved or not) into one uniform simplex output mesh. It was rejected here because it would make the no-op oracle false by construction — an all-linear mesh would still change cell types — for no benefit the roadmap bullet asked for.

## The lattice, and why it needs no external library

`levels` is **divisions per axis**, not applications of a doubling operator — unlike [`refine`](/refine), whose own "levels" *is* `2**levels` divisions. This is deliberate: a curved cell's own tessellation resolution is naturally small (2-4), and an exponential blow-up per level would make the parameter surprising.

Two lattice families, chosen per base shape:

- **Tensor bases** (`quad`, `hexahedron`): a plain `(n+1)`-point-per-axis grid at `t = linspace(-1, 1, n+1)`, whose endpoints are exactly `-1.0`/`1.0` in floating point — what makes the boundary classification below exact rather than tolerance-based.
- **Simplex bases** (`triangle`, `tetra`): built by recursively applying the same fixed same-type subdivision template `refine` already documents for a single reference cell — triangle to 4 children (three corner triangles plus one central one, unambiguous, no diagonal choice exists for a triangle) and tetra to 8 children (four corner tetrahedra plus the central octahedron split along the fixed *interior* diagonal between the midpoints of edges (0,1) and (2,3), in the exact ring order CLAUDE.md records for `refine`'s own template).

Every point this produces is an exact dyadic rational with denominator dividing `2**levels`, so a simple `round(coord * 2**levels)` integer key deduplicates points created by sibling recursive calls with no floating-point tolerance anywhere.

Every corner/edge-midpoint/face-centre/body-centre node's parametric position is **derived**, never hardcoded, from three tables this repo already owns and tests: hand-written reference elements (chosen so the barycentric solve is an exact identity matrix), `_convert_cells._ELEVATE` (which edge is mid-node `k`, the exact contract `convert_cells`'s own `"elevate"` mode already commits to) and `_refine_templates.QUAD_FACES` (which face is centre-node `k`, for `quad9`/`hexahedron27`). No permutation constant is written down anywhere in the module; a wrong convention in either upstream table would show up here as a wrong node position, not as a silent local relabelling.

## Isoparametric shape functions

Evaluated at each lattice point in reference space, then applied to the curved cell's own real (possibly curved) node positions — the standard isoparametric map `x(xi) = sum_i N_i(xi) * X_i`. Three families:

- `quad9`/`hexahedron27` — full tensor-product Lagrange (a genuine bubble/body node exists, so this is a plain product of per-axis 1-D quadratic factors, no serendipity correction needed).
- `quad8` — serendipity (8 nodes, no bubble): the standard closed-form corner/edge-mid functions.
- `triangle6`/`tetra10` — barycentric quadratics (`N_i = lambda_i (2 lambda_i - 1)` at a corner, `N_ij = 4 lambda_i lambda_j` at the midpoint of edge `(i, j)`), with the barycentric weights solved generically from the reference corners rather than hardcoded to a specific numeric layout.

## Watertightness: point identity across cells

A lattice point's identity is a function of which original entity it lies on, classified from its own **corner weights** (the same weights the isoparametric map already computes) rather than from coordinate comparisons: count how many of a point's corner weights are exactly nonzero. One nonzero corner means the point *is* that corner (a kept, original mesh point); two means an edge point; three (tetra10) or four out of eight (hexahedron27) means a face point; all nonzero means interior.

This needs no lookup table for "which edge/face is this" at all — the nonzero corners *are* the edge/face, addressed by their own global point ids. The resulting key deduplicates a point across every cell that touches it:

- an original point uses its own global id directly.
- an edge point uses `(min(ga, gb), max(ga, gb), t')`, `t'` the quantized fractional distance from the smaller-id endpoint — a fraction is intrinsic to the pair, not to which of the two cells (or which direction) computed it.
- a face point uses a **canonical traversal** of the face's corner ids: start at the globally smallest id, then continue toward whichever of its two cyclic neighbours has the smaller id. That rule is a property of the abstract cycle of ids alone — "which neighbour is smaller" has one answer regardless of which cell, or which direction, listed the face — so two cells describing the same physical face always compute the same canonical order. The face's local coordinates are then the corner weights re-keyed by global id (rather than by local position), which is what makes them invariant under the cell's own arbitrary local numbering: the two opposite corners of a bilinear patch are preserved by every rotation and reflection of a face's own cyclic order, so summing the pair keyed this way gives the identical value from either side.
- an interior point is always unique to one cell (no neighbour ever needs to agree with it), so no canonicalization is needed at all.

**Output cell shape**: a curved cell's linearized base determines its output. `triangle6`/`tetra10` refine directly into more triangles/tets (the recursive template never produces anything else). `quad8`/`quad9` refine into a quad lattice, then split into two triangles per sub-quad by "smallest id, diagonal to the corner two steps away" — provably direction-independent for a 4-cycle (2 steps is its own inverse), so no consistency issue exists for a 2-D mesh anyway. `hexahedron27` refines into a hex lattice; each sub-hexahedron becomes 12 tetrahedra via one new interior "apex" point (this sub-cell's own centroid — an interior key, never shared) plus, for each of its 6 quad faces, the identical "smallest id, diagonal two steps away" split into 2 triangles, each combined with the apex into one tetrahedron. This is the one place a genuine cross-cell consistency requirement exists (two hexahedra can share a whole quad face), and it is why the diagonal rule is keyed by **global** ids rather than a cell's own local numbering — a merely fixed local diagonal (the `convert_cells(mode="simplexify")` precedent) does not generally guarantee agreement between two independently-numbered neighbours.

`Tessellation.schema["watertight"]` reports `num_boundary_facets`, `num_interior_facets` and `num_facets_with_bad_count` (an internal triangular facet used by anything but exactly two tetrahedra, or a boundary one by anything but exactly one) — a live diagnostic, not just something the test suite checks.

## Fields

`fields=True` (default) interpolates every input `point_data` array onto the output mesh's points (component-wise, using each output point's own weights against its parent cell's real node values — exactly `Tessellation.gather`'s own formula) and replicates every input `cell_data` row onto its cell's children (a plain gather by `source_cell`, since `cell_data` is piecewise-constant and has no interpolant to speak of). `field_data` copies through unchanged.

`fields=False` skips all of this, leaving only the `tessellate:*` bookkeeping arrays — useful when a caller wants to attach a different array later (a model's own prediction) via `Tessellation.scatter`/`Tessellation.aggregate` rather than the input's own fields.

## Regions

**Point** regions remap fully, via the point-identity dedup key every kept point (a pass-through cell's node, or a curved cell's own corner) already goes through. **Cell** regions remap only for cells that stayed pass-through (a genuine 1:1 correspondence); an entry naming a curved (tessellated) cell has no single output cell to become, so it is dropped — silently would be wrong, so it is warned about, matching every other operation's region-dropping convention. **Side** regions are always dropped (a tessellated cell's facets have no correspondence with the original's at all).

## The `Tessellation` object

`mesh` is the tessellated (linear) output. `source_point` (`(P,)` int64 `point_data`, `tessellate:source_point`) is the original point index a tessellated point was kept from, or `-1` for a synthetic one. `source_cell` (`(C,)` int64 `cell_data`, `tessellate:source_cell`) is the GLOBAL (block-major) index of the original cell a tessellated cell descends from. `sub_index` (`(C,)` int64 `cell_data`, `tessellate:sub_index`) is a cell's index among its own source cell's children (0 for a pass-through cell).

- `gather(values)` — gather a per-source-point array onto the tessellated mesh's points, using each point's own recorded stencil/weights.
- `scatter(values)` — the reverse: write per-tessellated-point values back onto the source mesh's own points via `source_point`. A synthetic point is dropped; where several tessellated points share one source point (an original corner touched by several cells), their values are averaged. A source point that is a curved cell's own mid-side/face/body node (not a corner) has no single tessellated point recording its identity and comes back `NaN` — an honest answer, not a guess.
- `aggregate(values, reduction="mean")` — per-source-cell aggregation of a per-tessellated-cell array: `"mean"` (plain average over a source cell's children), `"weighted_mean"` (weighted by each child's own `measures()`) or `"first"` (the first child in `sub_index` order). Returns `(unique_source_cells, aggregated_values)`.
- `measures()` — `|measure|` (length/area/volume by topological dimension) of every cell of the tessellated mesh, one row per entry of `source_cell`.
- `Tessellation.from_mesh(tessellated, source)` — reconstruct a `Tessellation` from a mesh already carrying the `tessellate:*` arrays (e.g. after a file round trip), against the source mesh it was built from. Mirrors `refine`'s `refine:entity` stale-key guard: if `tessellate:stencil`/`tessellate:weights` are present, every stencil is checked to still reproduce its own point's coordinates against `source`'s *current* points — a mesh moved or renumbered since the stencil was written fails this, and the stencil is warned about and dropped (degrading to an identity stencil for kept points, all-zero for synthetic ones) rather than trusted.

`record_stencil=True` additionally attaches `tessellate:stencil`/`tessellate:weights` as real `point_data` (`(P, 27)` int64/float64, `-1`/`0.0` padding past however many entries a point actually uses) — off by default, since a `(P, 27)` array is expensive to persist, and needed only to reconstruct a full `Tessellation` (including `gather`) from a written-and-reread file via `from_mesh`.

## PhysicsNeMo integration

`Graph.Tessellate` (`true` for every default, or an object narrowing `Levels`/`Curved`/`Fields`) tessellates each sample's mesh before it reaches `graph_sample`, so a curved-cell dataset feeds a GNN a linear mesh without losing its curvature. Scoped to `Kind: "node"` graphs with `Graph.Regions: false` — tessellation's synthetic points carry no region membership of their own, so both are refused by name rather than silently producing an incomplete result. The resulting `Tessellation` lives only for that one sample's lifetime, matching the streaming invariant.

At prediction time, `predict_mesh`/`predict_file` rebuild the same tessellation from the checkpoint's own model card and scatter/aggregate the prediction back onto the *original* mesh's own points/cells — a prediction made on a tetrahedron is written onto the hexahedron it was carved out of. See [the PhysicsNeMo adapter](/physicsnemo) and [`doc/physicsnemo/mesh_and_geometry.md`](/physicsnemo/mesh_and_geometry).

## CLI and MCP

```bash
meshioplusplus tessellate curved_bracket.msh curved_bracket_tessellated.vtu --levels 2
```

`--no-curved`, `--no-fields` and `--record-stencil` mirror the Python API's `curved=False`/`fields=False`/`record_stencil=True`. The `tessellate` MCP tool takes the same options and reports `num_curved_source_cells`, `num_pass_through_source_cells` and `watertight` alongside the usual mesh summary.
