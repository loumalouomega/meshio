# Sobolev deformation

`sobolev_deform(mesh, array, length_scale, …)` smooths a raw per-point displacement field through the mesh's own P1 finite-element operators, then moves the points by the smoothed field. It is a mesh **operation**, not a file format, uses only standard C++, and runs under every mesh backend.

```python
import meshioplusplus as mp

mesh = mp.read("part.vtu")                  # carries a raw displacement in point_data["d"]

out = mp.sobolev_deform(mesh, "d", 0.5)                          # filter, then displace
out = mp.sobolev_deform(mesh, "d", 0.5, fix_boundary=True)       # pin the boundary
out = mp.sobolev_deform(mesh, "d", 0.5, fixed_points="clamped")  # pin a point set
out = mp.sobolev_deform(mesh, "d", 0.0)                          # apply d unfiltered

out, report = mp.sobolev_deform(mesh, "d", 0.5, return_report=True)
report["num_iterations"], report["residual"], report["converged"]
```

## What it solves, and why

Per ambient component,

```
(M + ℓ² K) u = M d,      x' = x + u
```

with `K` the P1 stiffness matrix, `M` a uniform vertex mass and `ℓ` the `length_scale`. That is a screened-Poisson low-pass filter whose cutoff wavelength is `ℓ`: it is what turns a jagged per-node displacement — a shape gradient, a scattered measurement, a model's raw output — into one a mesh can actually follow without tangling. Short-wavelength content is suppressed in proportion to `(ℓ/h)²`, long-wavelength content passes through nearly untouched.

`length_scale = 0` short-circuits to applying `d` directly at the free points, with no solve.

Compare with [`smooth`](/smooth), which moves points toward their neighbours' centroid and knows nothing about a field: `smooth` improves a mesh's *shape*, this filters a *displacement* the caller already has and then applies it.

## Three choices matched to the reference

Three of NVIDIA PhysicsNeMo's own choices are reproduced rather than improved on, because each is defensible and because matching them makes a parity test against the reference implementation a valid oracle — one that this repo runs, on a triangle surface and a tetrahedral volume, with and without pinned points.

**`K` is assembled from the simplex edge Gram matrix**, `K_loc = |c| B G⁻¹ Bᵀ` with `B = [−1; I]`, so one formula serves a polyline in 2-D, a triangle surface in 3-D and a tetrahedral volume alike. No cotangent special case, no ambient projection.

**`M` is uniform** — the mean of the positive lumped P1 vertex masses, applied to every vertex. That makes the filter's response depend on `ℓ` alone rather than on the local element size, and is what keeps the operator self-adjoint in plain vertex coordinates.

**One global solve**, Jacobi-preconditioned conjugate gradients, matrix-free, with one global stopping test `‖r‖ ≤ tolerance · ‖b‖`. The components decouple through `K`'s block structure but share the convergence history.

## Scope: linear simplices at the top dimension

Every cell block at the mesh's top topological dimension must be a `line`, a `triangle` or a `tetra` — that is what the Gram-matrix assembly is defined on. A quadratic block is refused naming `linearize`, anything else naming [`convert_cells(simplexify)`](/convert_cells). Lower-dimensional blocks ride along untouched: their points are still points.

A point in no top-dimensional cell is *isolated*. It has no stiffness row, so it receives its raw displacement verbatim and is counted in `num_isolated`.

## Pinning, and the Neumann default

Nothing is pinned by default. An unfixed boundary carries the natural homogeneous Neumann condition, which has a consequence worth stating: **a constant displacement is preserved exactly**, in zero iterations, because the initial guess is already the answer. That is the cheapest check that the operator is assembled correctly, and a test asserts it bit-for-bit.

`fixed_points` pins points by id, by the name of one of `mesh.point_sets`, or by the name of an integer/boolean `point_data` array. `fix_boundary` additionally pins every point on a boundary facet of the top-dimensional cells. Pinned points get zero-Dirichlet rows and do not move at all.

## Non-convergence is reported, not raised

If the iteration cap is reached first, `converged` is false, `residual` says how far it got, and the **last iterate is returned** — a partially smoothed field is still a usable one, and a warning names the situation. Upstream raises instead, which suits an autograd graph; here a pipeline user gets the warning in the log and an API user gets the flag. Raise `max_iterations` or lower `length_scale` if it happens.

## Determinism

The operator is applied in *gather* form: each vertex's row is evaluated by one thread from a fixed sequence of incident cells in ascending cell order, with no scatter and no atomics. Every inner product is a fixed-chunk parallel partial sum folded serially, with a chunk size that is a constant rather than a function of the thread count. The iterate is therefore byte-identical across mesh backends and thread counts.

## No numpy fallback

`sobolev_deform` is C++-core only. The conjugate-gradient stopping test is a branch on a rounded reduction and the iterate depends on how many iterations ran, so a numpy transcription could stop one iteration early or late and disagree macroscopically rather than in the last bit. Without the compiled core the module raises `NotImplementedError` naming the reason — the policy [`subdivide`](/subdivide) and [`agglomerate`](/agglomerate) already state, applied to a solver.

## Arrays

`record_filtered=True` attaches the filtered displacement as `sobolev:displacement`, Float64 `(n, dim)` — useful for inspecting what the filter did without differencing two meshes.

Everything else — connectivity, every data array, regions and property sets — passes through, since this is a pure coordinate move, and the points keep their input dtype.

## Pipeline

`SobolevDeform` is a settings-pipeline step, with the keys `Array`, `LengthScale`, `FixedPointsArray`, `FixBoundary`, `RecordFiltered`, `MaxIterations` and `Tolerance`. Note the pipeline's pin is by array name only; a point-set name or an id list is a Python-API convenience.

## CLI

```sh
meshioplusplus sobolev-deform IN OUT \
    --array NAME --length-scale L \
    [--fixed-points-array NAME] [--fix-boundary] \
    [--record-filtered] [--max-iterations N] [--tolerance T] [--quiet]
```

Both CLIs produce byte-identical files. See the [CLI reference](/cli).

## Other languages

```c
mio_sobolev_opts opts;
mio_sobolev_opts_init(&opts);
opts.array = "d";
opts.length_scale = 0.5;
mio_sobolev_report report;
mio_mesh* out = mio_sobolev_deform(mesh, &opts, &report);
report.converged;
```

```fortran
type(mio_mesh) :: out
logical :: conv
out = m%sobolev_deform('d', 0.5_real64, converged=conv)
```

```julia
r = sobolev_deform(mesh, "d", 0.5; fix_boundary=true)
r.mesh, r.num_iterations, r.converged
```

```r
r <- mio_sobolev_deform(mesh, "d", 0.5, fix_boundary = TRUE)
r$mesh; r$num_iterations; r$converged
```

```js
const r = mod.sobolevDeform(mesh, 'd', 0.5, '', true);
r.mesh; r.numIterations; r.converged;
```

The C++ `mFixedPoints` mask is not exposed on the flat ABI — pin by array name there, the same gap [`smooth`](/smooth)'s frozen mask has.
