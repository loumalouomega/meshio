---
title: Geometry guardrails
description: Out-of-distribution detection on a mesh's shape — non-invariant descriptors, a diagonal-Gaussian score, and an advisory verdict wired into training, prediction and dataset health.
---

# Geometry guardrails

A trained surrogate will answer any mesh you give it. It has no way to say "this part is nothing like what I was trained on", and the answer it returns in that case is finite, plausible and wrong — which is worse than an error, because nothing downstream flags it.

A guardrail is the missing check: fit a description of the *shapes* a model was trained on, and score a new one against it before trusting the prediction.

```python
import meshioplusplus as mio

guard = mio.GeometryGuard.fit("dataset_manifest.json", split="train")
guard.save("guard.json")

report = guard.check(mio.read("new_part.vtu"))
report["verdict"]     # 'in' or 'out'
report["worst"]       # the descriptors that put it there, with z-values
```

The library is unusually well placed for this. [`compute_stats`](stats.md), [`compute_quality`](quality.md) and [`extract_surface`](extract_surface.md) already produce every descriptor, [`dataset_health`](dashboard.md#health-summaries) already walks a manifest one mesh at a time, and fitting a density over those summaries is a small amount of code on top. Pure Python; the C++ core, the WASM build and every binding are untouched.

## The descriptors are not invariant, deliberately

The usual instinct with shape descriptors is to normalize away position, scale and orientation. Here that is exactly backwards: **a scaled part is a different part.** A model trained on brackets 10 cm across has learnt physics at that scale, and handed one 3 m across it will interpolate confidently into a regime it never saw.

So extents, centroid, area and volume enter **raw**. A guard fitted on centimetre parts flags a metre one, and a part displaced 50 units from where every training case sat is flagged on its centroid. That is the whole point, and a test pins it: scaling a mesh by two must multiply its area by four and its volume by eight in the descriptor set, which a bounding-box-normalized implementation would report unchanged.

| descriptor | source |
| --- | --- |
| `extent_x/y/z`, `centroid_x/y/z` | `compute_stats` — where the part is and how big |
| `log10_num_points`, `log10_num_cells` | counts, as logarithms: twice the cells is a small change, a thousand times is not |
| `total_area`, `unsigned_volume`, `num_inverted` | `compute_stats` |
| `surface_area`, `log10_num_surface_cells` | `compute_stats(extract_surface(mesh))` — how convoluted the shape is, which a bounding box cannot say |
| `scaled_jacobian_min/_mean`, `aspect_ratio_mean` | `compute_quality` — the mesh's own condition |

A descriptor that does not apply to a mesh is **`NaN`**, never zero and never silently omitted: a point cloud has no surface, and scoring a fabricated zero against a training mean would flag it for the wrong reason. Those are excluded from the score and listed under `missing`.

## The score

Each descriptor is standardized against the training mean and standard deviation, and the score is the root mean square of those z-values — a diagonal Gaussian. It is deliberately the simplest thing that works: a full covariance over sixteen descriptors needs far more cases than a training set usually has, and a mixture model needs a choice of components nobody can justify from the data.

The threshold is `margin` (default 1.5) times the **worst training score**, so "out" means *further from the training set than anything in it was, by a stated factor* rather than a probability nobody calibrated. Every training score is kept in the document, so a caller can re-threshold without re-fitting.

A standard deviation is floored relative to its own mean. `num_inverted` is exactly zero across a well-formed training set, and without the floor any mesh that differs would score infinite — large and finite is what keeps scores comparable and lets `worst` rank them at all.

What comes back names the worst descriptors, so a flag is actionable: "this part is forty times the volume of anything I trained on" is a different problem from "this mesh has inverted cells".

## In a training run

`Guard` is a top-level spec block. The guard is fitted on the training split when the run starts and stored **in the model card**, so it travels with the checkpoint that `mark_best` copies:

```jsonc
"Guard": true                          // or {"Margin": 2.0, "Quality": false}
```

`predict`, `predict_mesh` and `predict_file` then score every input they are given and add a `guard` block to each report row. A mesh that scores out raises a warning naming the worst descriptor — and **the prediction is still made and written**. That is deliberate: a model cannot refuse to answer, so the honest thing is to answer and say loudly that the input is unlike anything it was trained on. Fitting is advisory too — a fit that fails is reported and the run goes on, because refusing to train for want of a warning system would be the wrong trade.

## In a dataset's health

`dataset_health` takes a guard and adds a `guard` block per entry plus a manifest-level `out_of_distribution` list. It is kept **separate from `bad_entries`**: an unusual shape is a fact about the dataset, not a defect in the mesh, and conflating the two would have a curator deleting the most interesting cases.

## CLI and MCP

```bash
meshioplusplus guard-fit dataset_manifest.json guard.json --split train --margin 1.5
meshioplusplus guard-check guard.json new_part.vtu
```

`guard-check` also accepts a **model card** in place of a guard file, so a checkpoint's own guardrail can be used directly. The `guard_fit` and `guard_check` MCP tools take the same arguments; `guard_check` without a guard reports the raw descriptors, which is what a caller comparing two parts by hand wants.
