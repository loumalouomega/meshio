---
title: Point-cloud budgets
description: Reducing a large surface cloud to a fixed number of points — farthest-point sampling, lattice representatives, a bounding-box filter — so it fits a transformer's token budget.
---

# Point-cloud budgets

A transformer-shaped model — Transolver, FLARE, DoMINO, any architecture whose cost grows quadratically with the number of tokens — does not take a mesh either. It takes a *fixed number of points*, and a real surface mesh has a few hundred thousand of them where the model wants a few thousand. Reducing a cloud to a budget is the whole of the data path such a model needs before [`feature_matrix`](ml.md#the-feature-matrix-and-its-contract-feature_matrix) can build its token table, and this page is that path: a selection as a value, three ways to make one, and a mesh carrying only the selected points.

Everything here is pure Python over machinery that already exists. The quantization behind the `"grid"` method is [`GridSpec`](grids.md)'s, the region model is the mesh's own, and the token table is composition rather than a duplicate of `feature_matrix`. Nothing in the C++ core, the WASM build or any binding changed.

```python
import meshioplusplus as mio

mesh = mio.read("wing_surface.stl")                  # 380k points

budget = mio.select_points(mesh, 4096)               # farthest-point sampling
budget.indices                                       # (4096,) int64, in selection order
tokens = budget.take(mio.feature_matrix(mesh).matrix)   # (4096, F), the column contract intact

cloud = mio.subsample_points(mesh, budget, record_ids=True)
mio.write("wing_4096.vtp", cloud)                    # a point cloud every writer accepts
```

## The selection as a value: `PointBudget`

`select_points` returns a `PointBudget`: `indices` into the source mesh's points, the `method`, the `count`, and a `schema` recording how the selection was made (method, seed, start point, bounds, the lattice the `"grid"` method used) so a budget written into a model card can be reproduced.

**Selection order is information, so the indices are not sorted.** For `"farthest"` every prefix of the selection is itself the farthest-point sample of that smaller budget, so one selection at 8192 serves 4096 and 2048 as well by slicing — `budget.prefix(2048)` is that budget, and a test pins the property. `budget.sorted_indices` gives index order when a consumer wants it, and `budget.take(values)` gathers the selected rows of any per-point array in selection order. `to_dict()`/`from_dict()` round-trip the whole thing through JSON, indices included.

## Three methods, and what they cost

| method | what it does | cost | coverage |
|---|---|---|---|
| `"farthest"` (default) | exact greedy farthest-point sampling: the first point is `start`, every later one is the candidate farthest from everything selected so far | `O(N · count)` — 50k points / 1024 in 0.10 s, 200k / 2048 in 1.1 s, 1M / 4096 in 22.6 s (one core) | the reference |
| `"grid"` | quantize the cloud onto a coarse lattice sized so the occupied cells outnumber the budget a few times over, take the point nearest each occupied cell's centre, then farthest-point sample over those representatives | `O(N + count²)` — the same three sizes in 0.04 s, 0.15 s, 0.8 s | near-uniform: on the 56k-point Stanford bunny at 2048 points the largest uncovered gap is 3.2 length units against `"farthest"`'s 2.7 |
| `"random"` | a uniform draw | `O(count)` | the baseline: the same draw from the bunny leaves a gap of 13.8, five times either of the others |

The numbers are measured on the shipped implementation, and the table is on the page so the wait is not a surprise: `"farthest"` is fine to a few hundred thousand points, and above that `"grid"` is the answer rather than an approximate `"farthest"` — a million-point cloud then costs about what a fifty-thousand-point one does. The fill distance — the largest distance from any source point to its nearest selected point, i.e. the size of the largest region the model never sees — is the coverage oracle in the tests, computed by brute force so it depends on nothing but numpy.

Seeding follows the library's one convention: a keyword-only `seed` fed to a locally constructed `random.Random`, applied to a deterministically ordered input. `"farthest"` and `"grid"` are deterministic given `start` (the index of the first selected point, 0 by default); `seed` reaches them only through `start=None`, which draws the first point. Ties in the farthest-point step break on the lowest index, so a selection is a function of the input alone.

`bounds=(xlo, ylo, zlo, xhi, yhi, zhi)` restricts the candidates to a box — the third piece of the roadmap bullet — while the returned indices stay indices into the *whole* mesh. A `count` larger than the number of candidates, a `start` outside them, a cloud with fewer distinct positions than the budget (a duplicated point is never selected twice) and an unknown method each fail by name.

`select_points` also accepts a plain `(N, d)` array in place of a mesh, `d` in 1..3, so a table that never was a mesh can be budgeted too.

## The point cloud: `subsample_points`

`subsample_points(mesh, budget_or_count, *, record_ids=False, **select_kwargs)` returns a mesh carrying only the budgeted points, in selection order. What survives: the points, every per-point `point_data` array gathered to them, `field_data`, and every **Point** region remapped. What does not: the cells, `cell_data` and Cell/Side regions — a point cloud has no cells, and that is the operation rather than a limitation, so each drop is one warning naming what went. The output gets a single `vertex` block so every writer accepts it (`.vtu`, `.vtp`, `.ply` all round-trip a cloud), and `record_ids=True` attaches `budget:original_point_id`, the index each point had in the source.

Because a subsampled mesh keeps its point data and Point regions, `feature_matrix` on it has the **same columns** as on the source, and the table it builds equals `budget.take(feature_matrix(mesh).matrix)` row for row — pinned by a test, since that equality is what lets a preprocessing script and a training script agree without sharing code.

## Composition, not duplication

The recommended way to get a token table is the one at the top of the page: build the full feature matrix and gather the budget's rows. That preserves `feature_matrix`'s versioned column contract for free, and it means a budget made once can slice a matrix of coordinates, a matrix of fields and a matrix of region one-hots alike. There is deliberately no `feature_matrix(mesh, budget=...)` parameter — one gather is one line, and a second entry point into that function's contract would be a second thing to keep in step.

## CLI and MCP

```bash
meshioplusplus subsample wing.stl wing_4096.vtp --count 4096                 # farthest-point
meshioplusplus subsample wing.stl wing_4096.vtp --count 4096 --method grid   # the scalable one
meshioplusplus subsample wing.stl tip.vtp --count 512 --bounds=0.8,-1,-1,1,1,1 --record-ids
```

`--start N` names the first selected point, `--start random` draws it from `--seed`; `--record-ids` attaches `budget:original_point_id`. The MCP tool is `subsample`, with the same parameters (`count`, `method`, `seed`, `start`, `bounds`, `record_ids`); it reports the method, the count and the source point count alongside the usual mesh summary. `PointBudget` and `select_points` themselves are in-memory values with no path form, so they are consciously exempted from the tool parity guard.

## Scope

This page is the *budgeting*. It is not a third model family in `TrainSpec` — training a Transolver is a separate piece of work — and the proximity graphs a particle model wants are a separate roadmap item. See [ML data handling](ml.md) for the feature matrix, [mesh and regular grids](grids.md) for the grid path, and [the PhysicsNeMo models page](physicsnemo/models.md) for which architecture wants which shape.
