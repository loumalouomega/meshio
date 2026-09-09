# Per-vertex curvature

`compute_curvature(mesh, mean=…, gaussian=…, dual_area=…)` computes the per-vertex mean (`H`) and Gaussian (`K`) curvature of a surface — the [signed distance](/sdf)'s natural companion as a node feature: `sdf` says how far a point is from a surface, this says how the surface bends there. It is a mesh **operation**, not a file format, uses only standard C++/numpy, and runs under every mesh backend.

```python
import meshioplusplus as mp

mesh = mp.read("part.stl")

out = mp.compute_curvature(mesh)                                    # curvature:mean, curvature:gaussian
out = mp.compute_curvature(mesh, dual_area="barycentric")           # cruder, branch-free, twinnable in numpy
out = mp.compute_curvature(mesh, record_area=True, record_principal=True)

out, report = mp.compute_curvature(mesh, return_report=True)
report["total_angle_defect"]        # 2*pi*chi for a CLOSED surface — 4*pi for a sphere
report["quality"]["inconsistent_pairs"]  # nonzero means H's sign is not trustworthy
```

## The two estimators

`K` (Gaussian curvature) is the **angle defect**: `2π` minus the sum of the incident triangle angles at a vertex, divided by a dual area. `H` (mean curvature) is the **cotangent Laplace-Beltrami operator**: `‖Lp‖ / 2A`, signed by whether the discrete mean-curvature normal agrees with the vertex's own area-weighted normal. Both are the standard discrete estimators — the ones the discrete-differential-geometry convergence literature is about, and the ones NVIDIA PhysicsNeMo's own `gaussian_curvature_vertices`/`mean_curvature_vertices` use, so the numbers here are directly comparable with a model's.

**Deliberately not [`remesh`](/remesh)'s estimator.** `remesh.cpp` fits an osculating paraboloid over each 1-ring to drive its own curvature gradation and anisotropic metric — a legitimate estimator, but one that returns only the larger-magnitude principal curvature as a *magnitude*, runs on remesh's own subdivided working copy rather than the caller's mesh, and has no structural invariant to test against. It is left exactly as it is; `compute_curvature` is a second, independent estimator, over the caller's own mesh, with a testable oracle.

## Gauss-Bonnet: the oracle

For a **closed** surface, the sum of every vertex's angle defect is `2π·χ` — `4π` for anything sphere-like (genus 0), whatever the tessellation and whichever dual area is chosen. `compute_curvature` reports that sum as `total_angle_defect` precisely so the invariant is checkable from every binding, not only from a gtest — it is the cheapest way to know a result is sane before trusting the per-vertex arrays. On an **open** surface it is `2π·χ` minus the boundary's total turning, a different (still meaningful) quantity.

Both dual areas partition the surface exactly, so `total_angle_defect` is bit-identical under either — the defect never touches the area it is later divided by.

## Two dual areas, one measured tradeoff

`dual_area` picks what a vertex's curvature is divided by:

* **`"mixed-voronoi"`** (default) — Meyer et al.'s mixed Voronoi area: the true Voronoi cell where a triangle is non-obtuse, a bisected area where it is not. Reuses the cotangents the mean-curvature pass already computes, so it is nearly free, and it converges markedly better on an irregular tessellation.
* **`"barycentric"`** — a third of each incident triangle's area. Cruder, but **branch-free**, which is what makes it bit-exactly reproducible in the numpy fallback (see below).

Measured, not guessed, on a unit-radius icosphere (subdivision 3): `mixed-voronoi`'s worst relative error is `H = 6.4e-6`, `K = 0.0055`; `barycentric`'s is `H = 0.1437`, `K = 0.1474` — more than an order of magnitude cruder, which is the whole reason `mixed-voronoi` is the default rather than the twinnable one.

## `H` is orientation-dependent, `K` is not

The mean-curvature normal's sign comes from the surface's own winding, so a mesh whose facets disagree about which side is out yields sign-flipped patches with **no error raised** — `compute_curvature` never inspects orientation globally, only locally through the cotangent weights. That is why the result always carries the input's surface `quality` (the same four counts [`sample_distance`](/sdf) reports: boundary edges, non-manifold edges, inconsistent pairs, degenerate triangles): check `quality["inconsistent_pairs"]` before trusting a sign. A nonzero count triggers a warning naming the fix: run [`repair`](/repair) first. **This operation never silently repairs its input.**

## Boundary and isolated vertices

A boundary vertex — one on an edge used by exactly one triangle — has no closed 1-ring, so both estimators are biased there; the default is to leave it `NaN` (`include_boundary=False`). Opting in (`include_boundary=True`) computes a value anyway: `K` uses the geodesic form `π - Σθ`, `H` the raw one-sided operator, and both are honestly biased. An **isolated** vertex — referenced by no surviving triangle — is `NaN` either way, since there is nothing to average.

Degenerate (zero-area) triangles are skipped by the same predicate `soup_quality` uses, so the two agree about what "degenerate" means; skipped triangles are counted in the report but never propagate a `NaN` into a vertex that has other, valid contributions.

## Arrays

| Array | Location | Shape | Written when |
|---|---|---|---|
| `curvature:mean` | `point_data` | `(n,)` | `mean=True` (default) |
| `curvature:gaussian` | `point_data` | `(n,)` | `gaussian=True` (default) |
| `curvature:area` | `point_data` | `(n,)` | `record_area=True` |
| `curvature:principal` | `point_data` | `(n, 2)`, `k1 ≥ k2` | `record_principal=True` |

`curvature:principal` is free — no new machinery, just `H ± √(H² - K)`; the radicand is negative only through discretization error, so it is clamped rather than turned into `NaN`.

## Region restriction

`region="name"` restricts the estimator to a single named `Cell` region, leaving every vertex not touched by it `NaN`/untouched (unless it is also referenced by cells outside the region, whose contribution it still receives) — the same `Cell`-region convention [`sample_distance`](/sdf) uses.

## Triangulation

Triangles come from the same fan [`convert_cells(simplexify)`](/convert_cells) uses, so a quad or rectangular-polygon mesh's curvature is the curvature of its **canonical triangulation**, not of the quad surface itself. A volume or polyhedron block is refused by name pointing at [`extract_surface`](/surface); a higher-order block is refused pointing at `linearize`.

## numpy fallback: a partial twin

`dual_area="barycentric"` has a **real, bit-exact numpy twin** — every expression on that path is `+ - * / sqrt` and one `atan2`, and it is pinned against the compiled core with exact array equality, angle-defect included.

`dual_area="mixed-voronoi"` (the **default**) has **no** numpy fallback and raises `NotImplementedError` naming the reason when the compiled core is unavailable: its obtuse/non-obtuse test is a discrete branch on a sign, so two implementations could land on opposite sides of a near-right-angle corner and disagree macroscopically rather than in the last ulp — the same class of refusal `_smooth.py`'s inversion guard and `_sdf.py`'s `sign="winding-number"` mode already make. The raise only fires when `meshioplusplus._core` is genuinely absent, which it never is on a normal `pip install`.

## Where it lives, and why

Unlike `gradient`/`hessian`/`estimate_error`/`data_integrate`, `compute_curvature` takes **no input data array** — it reads only geometry and topology and manufactures a field from nothing. That is the shape [`compute_quality`](/) and [`compute_stats`](/stats) have, both top-level CLI verbs, not the `data <verb>` group's shape (whose four mesh operations all take `--array NAME`, operating on a field the caller already has). `curvature` is a top-level verb for the same reason.

## CLI

```sh
meshioplusplus curvature IN OUT \
    [--no-mean] [--no-gaussian] \
    [--dual-area mixed-voronoi|barycentric] \
    [--include-boundary] [--record-area] [--record-principal] \
    [--region NAME] [--quiet]
```

Both CLIs produce byte-identical files. See the [CLI reference](/cli).

## Other languages

```c
mio_curvature_opts opts;
mio_curvature_opts_init(&opts);
opts.record_principal = 1;
mio_curvature_report report;
mio_mesh* out = mio_compute_curvature(mesh, &opts, &report);
report.total_angle_defect;
```

```fortran
type(mio_mesh) :: out
real(real64) :: defect
out = m%curvature(record_principal=.true., total_angle_defect=defect)
```

```julia
r = compute_curvature(mesh; record_principal=true)
r.mesh, r.total_angle_defect, r.quality.inconsistent_pairs
```

```r
r <- mio_compute_curvature(mesh, record_principal = TRUE)
r$mesh; r$total_angle_defect
```

```js
const r = await m.computeCurvature(mesh, true, true, 'mixed-voronoi', false, true, true, '');
r.mesh.point_data['curvature:mean'];
r.totalAngleDefect;
```

Also reachable as a `convertSurfaceOps`/settings-pipeline step (`{op: 'curvature', mean, gaussian, dualArea, includeBoundary, recordArea, recordPrincipal, region}` in WASM; `{"Op": "Curvature", "Mean": ..., "Gaussian": ..., "DualArea": ..., "IncludeBoundary": ..., "RecordArea": ..., "RecordPrincipal": ..., "Region": ...}` in a `settings.json`), exactly like `gradient`/`hessian` — a pure data step that changes no geometry, so the mesh passes straight through with the requested arrays attached.

The browser viewer has no dedicated Curvature chip; the arrays attach like any other pipeline step's output and are available in the colour-by menu once the pipeline runs.
