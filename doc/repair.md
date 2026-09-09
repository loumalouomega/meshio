# Surface repair

`repair(mesh, fix_orientation=…, fill_holes=…, split_non_manifold=…)` fixes the three surface defects [`clean`](/clean) does not touch: triangles that disagree about which side is out, holes, and pinched (bowtie) vertices. It is a mesh **operation**, not a file format, uses only standard C++, and runs under every mesh backend.

```python
import meshioplusplus as mp

mesh = mp.read("scan.stl")

out = mp.repair(mesh)                                   # every pass, the defaults
out = mp.repair(mesh, fill_holes=False)                 # orientation and bowties only
out = mp.repair(mesh, max_hole_edges=0)                 # fill every hole, however large
out = mp.repair(mesh, weld_tolerance=1e-9)              # weld coincident points first

out, report = mp.repair(mesh, return_report=True)
report["quality_before"], report["quality_after"]       # what was wrong, what remains
report["num_flipped"], report["num_holes_filled"], report["num_vertices_split"]
```

`clean` is still the place for welding, dropping degenerate and duplicate cells and pruning orphans; `repair` composes it through `weld_tolerance` rather than duplicating it. The two are complementary and it is normal to run both.

## What it does, in order

Weld (opt-in) → triangulate → split bowties → orient → fill holes → orient outward.

The order is not arbitrary. **Splitting comes before orienting** because a pinched boundary vertex has boundary degree four, and a loop through it cannot be traced; splitting first turns a hole that would be skipped into one that is filled. **Orienting comes before filling** because a fill inherits its winding from the one triangle on each boundary edge, which only means something once that neighbourhood agrees with itself. **Orienting outward comes last** because a sphere with a hole in it has no enclosed volume to take the sign of until the hole is closed.

## Orientation: the topological rule, not a normal test

Two triangles sharing an edge agree about which side is out **iff they traverse that edge in opposite directions**. That is the rule here, propagated by a breadth-first walk per connected component, and it is exact.

The obvious alternative — comparing the two triangles' normals and flipping when the dot product is negative — is what NVIDIA PhysicsNeMo's `fix_orientation` does, and it is a local-flatness approximation: on any crease sharper than 90 degrees it flips a triangle that was already correct. A strip folded at 150 degrees is consistently wound and `repair` leaves it entirely alone; a normal test rewinds half of it. That case is pinned by a test.

Boundary edges and non-manifold edges are walls the walk does not cross, so each connected component is oriented independently. Within a component, whichever of the two consistent assignments needs **fewer flips** wins, so a mostly-correct mesh is barely touched. A component with no consistent assignment at all (a Möbius strip) is counted in `num_unorientable` and given the walk's best effort rather than refused.

`orient_outward` then flips whole any **closed** component whose signed volume is negative, so an inside-out closed surface comes back outward. Nested cavities are not detected: every closed component is oriented outward on its own.

## Holes: filled to agree with their neighbours

Each traceable boundary loop of at most `max_hole_edges` edges (`<= 0` means no limit) gets one new point at the loop's centroid and one triangle per loop edge. The fan is wound **against** the one triangle on each boundary edge, so the shared edge ends up used twice in opposite directions and the filled surface is consistent. Upstream winds the fan from the loop's own traversal instead, which can leave the fill disagreeing with the surface it closes.

A loop through a vertex whose boundary degree is not two cannot be traced; it is counted in `num_holes_detected` and `num_holes_skipped` and left open, never guessed at. This is the case `split_non_manifold` usually turns into two traceable loops.

The fill is flat and unsmoothed: it closes the surface, it does not reconstruct what was missing.

## Bowties: split, not merged

A vertex whose incident triangles form more than one edge-connected group is a pinch — two sheets of surface touching at a point. Each group after the first gets a duplicate of the vertex at exactly the same position, so the geometry is unchanged and the sheets become separate components. `num_vertices_split` counts the vertices, `num_points_added` the copies.

Non-manifold **edges** (used by three or more triangles) are a different problem and are deliberately not addressed: splitting one is a topology decision with no single right answer. They are counted in `quality_after` and the orientation walk does not cross them.

## Both verdicts are reported

`quality_before` and `quality_after` each carry the four counts [`sample_distance`](/sdf) reports — boundary edges, non-manifold edges, inconsistently wound pairs, degenerate triangles — plus `watertight`. Reporting both is the point: it says what was fixed *and* what remains, and the "after" figure is computed from the output itself by the same predicate, so it cannot flatter the result.

## Arrays

Nothing is attached unless `record_provenance=True`, which adds two Int64 arrays:

| Array | Location | Meaning |
| --- | --- | --- |
| `repair:parent_point` | point | the input point each output point came from; `-1` for a hole centroid |
| `repair:hole` | cell | `-1` for an input triangle, the hole's ordinal for a fill triangle |

## Output shape

All-triangle at the surface, with blocks 1:1 with the input (a quad block becomes a triangle block of twice the rows). Lower-dimensional blocks — the boundary `line` blocks a gmsh surface routinely carries — ride along verbatim. Fill triangles land in **one trailing `triangle` block**, added only when there is one, so an already-closed input keeps its block count.

Points are the originals, then the split copies, then the hole centroids. `point_data` copies inherit their source's row and centroids get the mean of their loop's rows, dtype preserved; the fill block's `cell_data` rows are NaN for float arrays and 0 for integer ones. Point and Cell regions survive — a split copy joins its source's regions — and Side regions are dropped by name, since rewinding a triangle permutes its edge numbering.

## No numpy fallback

`repair` is C++-core only. The outward-orientation pass decides on the sign of a rounded enclosed volume, and a second, independently written implementation could land on the other side of that branch for a near-degenerate component and then diverge macroscopically rather than in the last bit. It is on by default, so a twin would have to exclude the default path to be safe. Without the compiled core the module raises `NotImplementedError` naming the reason — the same policy [`subdivide`](/subdivide) and [`agglomerate`](/agglomerate) already state.

## Pipeline

`Repair` is a settings-pipeline step, with the keys `FixOrientation`, `OrientOutward`, `FillHoles`, `SplitNonManifold`, `MaxHoleEdges`, `WeldTolerance` and `RecordProvenance`. The report carries the counters, and two warnings say what could not be fixed: an unorientable component, and non-manifold edges that remain.

## CLI

```sh
meshioplusplus repair IN OUT \
    [--no-fix-orientation] [--no-orient-outward] \
    [--no-fill-holes] [--no-split-non-manifold] \
    [--max-hole-edges N] [--weld-tolerance T] \
    [--record-provenance] [--quiet]
```

Both CLIs produce byte-identical files. See the [CLI reference](/cli).

## Other languages

```c
mio_repair_opts opts;
mio_repair_opts_init(&opts);
opts.max_hole_edges = 0;                /* no limit */
mio_repair_report report;
mio_mesh* out = mio_repair(mesh, &opts, &report);
report.num_flipped;
```

```fortran
type(mio_mesh) :: out
integer(int64) :: nflip
logical :: wt
out = m%repair(num_flipped=nflip, watertight_after=wt)
```

```julia
r = repair(mesh; max_hole_edges=0)
r.mesh, r.num_flipped, r.quality_after.watertight
```

```r
r <- mio_repair(mesh, max_hole_edges = 0)
r$mesh; r$num_flipped; r$quality_after$watertight
```

```js
const r = mod.repair(mesh);
r.mesh; r.numFlipped; r.qualityAfter.watertight;
```

The index maps the C++ result carries (`mPointMap`, `mCellMaps`) are not exposed on the flat ABI — the same gap [`smooth`](/smooth)'s frozen mask has.
