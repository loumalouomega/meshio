# Shrinkwrap

`shrinkwrap(mesh, target, offset=…, max_distance=…, weights=…)` projects a mesh's points onto a target triangle surface — the fitting step a scanned skin, a CAD shell or a coarse solve needs before it can serve as a template. It is a mesh **operation**, not a file format, uses only standard C++/numpy, and runs under every mesh backend.

```python
import meshioplusplus as mp

template = mp.read("template.vtu")
target = mp.read("scan.stl")

out = mp.shrinkwrap(template, target)                       # straight onto the surface
out = mp.shrinkwrap(template, target, offset=0.5)           # onto the offset surface
out = mp.shrinkwrap(template, target, max_distance=2.0)     # leave far points alone
out = mp.shrinkwrap(template, target, weights="mask")       # move only the selected points

out, report = mp.shrinkwrap(template, target, return_report=True)
report["num_projected"], report["num_missed"], report["max_displacement"]
```

## One projection, not an iteration

Each point moves once: `x' = x + w (p + offset·n̂ − x)`, with `p` the closest point on the target and `n̂` the unit normal there. There is no self-intersection guard and no cell-inversion guard, exactly as in NVIDIA PhysicsNeMo's own `shrinkwrap` — a wrap is a *fit*, not a smoothing. Wrapping a template far from its target, or with a large offset, can fold it; check the result with [`compute_quality`](/mesh_quality) if that is a risk, and relax it with [`smooth`](/smooth) or [`sobolev_deform`](/sobolev_deform) if it is.

The closest point comes from the same bucket-grid nearest-triangle search [`sample_distance`](/sdf) uses, so the point a query is projected *to* is the point `sample_distance` measures *to*. That search is totally ordered on `(distance², triangle id)`, so the accelerator's bucket size cannot change the answer — a property a test proves rather than assumes.

## The offset goes along the feature normal

At a face hit the normal is the triangle's own. At an **edge** or a **vertex** hit it is that feature's pseudonormal — the sum of the incident faces' unit normals, the same tables the signed distance's sign already reads.

This is a deliberate divergence from upstream, which offsets along the selected triangle's normal. The offset surface of a creased mesh is the rounded one — its Minkowski sum with a ball — whose normal at the crease is the *bisector* of the two incident faces. Offsetting along one face's normal from the crease lands the point off that surface by a factor `1/cos(θ/2)` and, worse, makes the result depend on which of two equidistant faces won the tie-break. The pseudonormal is a property of the feature, not of the tie-break. A 90-degree "book" target pins the difference.

`normal_weight` chooses how a **vertex** pseudonormal is weighted: `"angle"` (the default, geometrically the right choice) or `"area"` (free of `acos`, and therefore the mode the numpy twin reproduces bit for bit under a nonzero offset).

## What moves, and what does not

**Every point of the source mesh**, whatever cells it carries — a volume mesh's interior points are projected too. Only the *target* must be a surface; a volume or higher-order target is refused by name pointing at [`extract_surface`](/extract_surface) or `linearize`.

`weights` names a scalar `point_data` array on the source. A **float** array is a blend factor, applied unclamped so a caller can deliberately overshoot or pull back; an **integer or boolean** array is a selection, and a zero weight means the point is never even queried (counted in `num_skipped`). The Python API also accepts an array directly, attaching it for the call and removing it afterwards.

A point farther than `max_distance` (`<= 0` means unlimited) is left where it is and counted in `num_missed`, as is one whose hit feature has no direction to offset along — every triangle touching it degenerate.

`target_region` restricts the target to one named `Cell` region, so a template can be wrapped onto a named patch rather than the whole scan.

## The target's verdict is reported

`quality` carries the target's four defect counts and `watertight`. A target whose facets disagree about which side is out will offset different points to different sides, so a non-watertight target is warned about rather than silently accepted. Run [`repair`](/repair) on it first if the offset matters.

## Arrays

Nothing is attached unless asked for:

| Array | Location | Meaning |
| --- | --- | --- |
| `shrinkwrap:distance` | point | the distance to the target *before* the move; NaN where the point was not queried |
| `shrinkwrap:closest_cell` | point | the target cell the point projected onto; `-1` where not queried |

Everything else — connectivity, `point_data`, `cell_data`, `field_data`, regions and property sets — passes through verbatim, since this is a pure coordinate move. The points array keeps its input dtype.

## numpy fallback: a near-complete twin

The numpy reference reuses the [signed distance](/sdf)'s own brute-force search and normal tables, and the projection itself is `+ − × ÷` and one correctly rounded `sqrt`, so its output is bit-identical to the compiled core's — with one exclusion. The **angle-weighted vertex pseudonormal** goes through `acos`, which is not correctly rounded; a *sign* does not care about its last bits, but an *offset* puts them straight into a coordinate. So with a nonzero offset under `normal_weight="angle"` the reference raises `NotImplementedError` naming `normal_weight="area"`, which is branch-free and bit-exactly twinned. Offset 0 under either weighting is twinned too.

## Not a pipeline step

`shrinkwrap` needs a second (target) mesh, so like [`interpolate`](/interpolate) and [`diff`](/diff) it is not a settings-pipeline step; naming it in a document is refused with a message pointing at the CLI verb.

## CLI

```sh
meshioplusplus shrinkwrap IN TARGET OUT \
    [--offset=X] [--max-distance D] \
    [--weights NAME] [--target-region NAME] \
    [--normal-weight angle|area] \
    [--record-distance] [--record-closest-cell] [--quiet]
```

A negative offset needs the `--offset=-0.5` form. Both CLIs produce byte-identical files. See the [CLI reference](/cli).

## Other languages

```c
mio_shrinkwrap_opts opts;
mio_shrinkwrap_opts_init(&opts);
opts.offset = 0.5;
opts.weights = "mask";
mio_shrinkwrap_report report;
mio_mesh* out = mio_shrinkwrap(mesh, target, &opts, &report);
report.num_projected;
```

```fortran
type(mio_mesh) :: out
integer(int64) :: nproj
out = mio_shrinkwrap(mesh, target, offset=0.5_real64, num_projected=nproj)
```

```julia
r = shrinkwrap(mesh, target; offset=0.5, weights="mask")
r.mesh, r.num_projected, r.max_displacement
```

```r
r <- mio_shrinkwrap(mesh, target, offset = 0.5, weights = "mask")
r$mesh; r$num_projected; r$max_displacement
```

```js
const r = mod.shrinkwrap(mesh, target, 0.5, 0, 'mask');
r.mesh; r.numProjected; r.maxDisplacement;
```
