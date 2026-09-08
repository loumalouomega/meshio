<!--pytest-codeblocks:skipfile-->
# Changelog

This document records every released version of meshio++ — new formats, new operations,
notable enhancements, and breaking changes. Breaking changes are called out explicitly as
**Breaking:**; everything else is additive unless stated otherwise.

**Keep this file current: add an entry in the same change as every version bump.** See the
"Version bumps" section of `CLAUDE.md`.

## v10.35.0 (2026-09-08)

**Formats a physics-ML pipeline expects.** meshio++ could describe a training set but not write one in any shape NVIDIA PhysicsNeMo's own loaders read: `MeshDataset` globs for `**/*.pmsh` or sniffs a Zarr store, the DoMINO and Transolver datapipes want one `.npz` per case in a fixed key layout, and there was no USD export for a twin at all. This release adds all four as ordinary meshio++ formats, so `read`/`write`/`convert`/`read_sequence` reach them with no new API. Python-only throughout — no C++, ABI, WASM or binding change. Closes roadmap section 1's formats bullet, which is removed. See [`doc/formats/pmsh.md`](doc/formats/pmsh.md), [`doc/formats/zarr.md`](doc/formats/zarr.md), [`doc/formats/cae.md`](doc/formats/cae.md) and [`doc/formats/usd.md`](doc/formats/usd.md).

- **New `pmsh` format** — PhysicsNeMo's memory-mapped mesh directory, **hand-rolled in pure numpy rather than delegated to `Mesh.save`**, and that is the point: a solver box with no torch, no CUDA and no NVIDIA stack can write the training set the framework reads. Parity is pinned in both directions by tests that gate on an installed physicsnemo. Arrays come back memory-mapped copy-on-write, so they are writeable and paged in only as touched. Two facts neither guessable nor optional: **a zero-element array has no blob** (tensordict does not persist an empty tensor, so a point cloud's `cells.memmap` is absent while its shape is still declared), and **a `.pmsh` directory containing `zarr.json` is a Zarr store** — the extension names the role, the contents name the codec, exactly as `MeshReader._load_sample` does.
- **New `zarr` format** — the same mesh in PhysicsNeMo's `io_zarr` layout: a v3 group with the type attribute, `points`/`cells` arrays, the three data groups, upstream's own 200k-row chunking and zstd level 3, and consolidated metadata. Writing needs zarr 3.x and says so by name, since zarr-python 2.x cannot produce a v3 store at all. **Not `write_dataset(format="zarr")`**, which writes a training *table*; such a store is refused here by name rather than half-read, and vice versa.
- **New `cae` format** — the per-case `.npz` sample layout, transcribed from the Kratos `PhysicsNeMoApplication`'s own exporter, so a meshio++ case and a Kratos case are interchangeable in one dataset directory. A volume mesh's nodes and nodal fields become the volume half while its skin — extracted and triangulated here — becomes the surface half, with the volume's `cell_data` gathered onto each triangle through its parent cell. Three rules belong to the reader on the other side and each fails silently rather than loudly when broken: every array must be at least 1-D, the two volume keys must share their row count, and **no other key may contain the substring `volume`** (that reader takes its volume length from the first such key in file order, over every key in the file) — so meshio++'s own sidecars are named `node_field_*` and a global parameter containing `volume` is refused.
- **New `usd` format** — time-sampled OpenUSD stages, read and written, behind a new `[usd]` extra (`usd-core`). **n-gons are kept**, unlike every other bridge here: USD stores a face-vertex count per face, so a quad-dominant hex skin arrives as quads. `point_data` becomes `vertex` primvars and `cell_data` `uniform` ones. It joins the sequence engine's fan-in writers: one stage, one sample per step, and topology re-authored **only when it changes**, so a remeshed series stays valid while a fixed-topology one stays compact.
- **`data export-cae` CLI verb and an `export_cae` MCP tool** (the registry goes 78 → 79) writing a whole run at once from a glob, a path list or a multi-step file, one mesh alive at a time.
- **The sequence engine accepts directory stores.** A glob plan entry is now a file *or* a directory whose suffix is `.pmsh`/`.zarr`, so `write_sequence("out_{step}.pmsh", …)` then `read_sequence("out_*.pmsh")` works; deliberately narrower than "any directory". The MCP path sandbox likewise admits an existing directory as an input, and its file-download endpoint keeps rejecting one.
- **Provenance in three of the four.** Zarr keeps the block in a root attribute (plain JSON, so it is recovered without importing zarr), USD in the root layer's `documentation`, and `cae` in a NUL-padded bytes array written as the archive's first member — which the ordinary head scanner then finds with no `.npz`-specific code. `pmsh` has no free-text slot and honestly reports none rather than adding a sidecar file upstream's own loader never wrote.
- The `nvidia-physicsnemo` pin widens to `>=2.1,<2.3`, which is what the parity tests are verified against.

## v10.34.0 (2026-09-08)

**Geometry guardrails.** A trained surrogate answers any mesh it is given, and the answer for a part unlike anything it saw is finite, plausible and wrong — worse than an error, because nothing downstream flags it. This release is the missing check: fit a description of the *shapes* a model was trained on, and score a new one against it. Pure Python over `compute_stats`, `compute_quality` and `extract_surface`, which already produce every descriptor — no C++, ABI, WASM or binding change. Closes roadmap section 1's geometry-guardrails bullet, which is removed. See [`doc/geometry_guardrails.md`](doc/geometry_guardrails.md).

- **New `geometry_descriptors(mesh)` and `GeometryGuard`** (`fit`/`score`/`check`, JSON round-trip). Sixteen descriptors from the three existing operations: extents and centroid, counts as logarithms, area, volume, inverted count, the outer surface's area and facet count, and the scaled-Jacobian and aspect-ratio summaries.
- **The descriptors are deliberately NOT invariant, and that is the design.** The usual instinct is to normalize away position and scale; here that is backwards, because a scaled part is a different part and a model trained on brackets 10 cm across has learnt physics at that scale. Extents, centroid, area and volume enter raw, so a guard fitted on centimetre parts flags a metre one and a displaced part is flagged on its centroid. Pinned by a test asserting that scaling a mesh by two multiplies its descriptor area by four and its volume by eight — which a bounding-box-normalized implementation would report unchanged.
- **A diagonal Gaussian, and a threshold that means something.** Each descriptor is standardized against the training mean and standard deviation and the score is the root mean square of those z-values — deliberately the simplest thing that works, since a full covariance over sixteen descriptors needs far more cases than a training set has. The threshold is `margin` (default 1.5) times the **worst training score**, so "out" means further from the training set than anything in it was, by a stated factor, rather than a probability nobody calibrated; every training score is kept so a caller can re-threshold without re-fitting. A standard deviation is floored relative to its own mean: `num_inverted` is *exactly* zero across a well-formed training set, and without the floor any mesh that differs would score infinite instead of large-and-finite.
- **A descriptor that does not apply is `NaN`, never zero and never omitted** — a point cloud has no surface, and scoring a fabricated zero against a training mean would flag it for the wrong reason. Those are excluded from the score and listed as `missing`. What comes back names the worst descriptors, so a flag is actionable.
- **Wired in three places, all advisory.** A top-level `Guard` spec block fits on the training split and stores the guard **in the model card**, so it travels with the checkpoint `mark_best` copies; `predict`/`predict_mesh`/`predict_file` score every input, add a `guard` block to the report row and warn naming the worst descriptor — **the prediction is still made and written**, because a model cannot refuse to answer and the honest thing is to answer and say loudly that the input is unlike its training set. `dataset_health` gains a per-entry `guard` and a manifest-level `out_of_distribution` list, kept **separate from `bad_entries`**: an unusual shape is a fact about the dataset, not a defect in the mesh.
- **`guard-fit` and `guard-check` CLI verbs and `guard_fit`/`guard_check` MCP tools** (the registry goes 76 → 78). `guard-check` accepts a fitted guard *or* a model card carrying one, and without a guard the tool reports the raw descriptors.

## v10.33.0 (2026-09-08)

**Temporal windows, rollout, and dataset augmentation.** A steady-state surrogate maps one mesh to one answer; a transient one maps a short history to the next state, and nothing here built that history. This release adds the windows, the rollout evaluation that measures whether a transient model survives more than one step, and the per-epoch augmentation wrapper. It also fixes the one thing the augmentation depends on: `transform(rotate_vector_data=True)` rotated `point_data` only, so a rotated mesh carried an unrotated per-cell velocity. Closes roadmap section 1's temporal-windowing and augmentation bullets, both removed. See [`doc/physicsnemo.md`](doc/physicsnemo.md#temporal-windows-and-rollout).

- **Breaking:** **`transform(rotate_vector_data=True)` now rotates `cell_data` too**, in both engines — float arrays whose trailing dimension is 3 (`R·v`) or 9 (`R·A·Rᵀ`), at either location; an integer array is never rotated, so a material tag is safe. A vector living on a cell rotates exactly as one living on a point does, and the old behaviour was silently wrong: nothing about an array says which frame it is in. Output changes only for a mesh that carries such an array *and* passes the flag, which defaults off. The C API, Fortran, Julia, R and WASM inherit it with no signature change.
- **New `make_window`, `window_samples`, `iter_windows`, `window_stats`** — windows of K states, **oldest first**, in three schemes: `single_step` (each window paired with the state that followed, `T − K` samples), `one_shot` (the first window paired with the last state, one sample) and `time_conditional` (the first window plus a normalized time channel, so error does not compound). `make_window` is the single owner of the layout, and the recorded `x_columns` name each column's age (`T@-2`, `T@-1`, `T@-0`) so a stored contract catches a reversed ordering rather than leaving it to a convention. The streaming invariant holds: at most K *feature matrices* and one mesh alive, each step read exactly once, pinned by a read-counting test.
- **New `rollout(predict_fn, series, history_size=)` → `Rollout`** — seeded with K true states, then fed its own predictions, reporting the per-step RMSE. `predict_fn` is **injected**, which is what keeps the module torch-free: the caller owns the framework, normalization and device. Pinned by two models whose behaviour is known exactly — a perfect one (zero error throughout) and persistence, whose error grows 1, 2, 3, 4 under a real rollout and stays flat at 1 under teacher forcing, which is the distinction the test exists to make.
- **New `Augmentation(rotation=, scale=, translation=, seed=)`** and an `augmentation=`/`epoch=` pair on `iter_samples` and `make_dataset`. Every draw is a pure function of `(seed, epoch, index)` — no global generator, nothing carried between calls — so a run is reproducible and two processes of a distributed job agree. Scaling is isotropic on purpose: an anisotropic scale changes what a vector field means. **A paired sample gets one draw, replayed on the target**, since augmenting the two independently would teach the model that a part rotates between one step and the next.
- **A top-level `Augmentation` spec block**, refused by name for the `srresnet` family (a grid sampled on a fixed lattice would have its own coverage changed by rotating the mesh under it), written back only when set, applied to the **train** split only and recorded in the model card. The dataset's `set_epoch` is called between epochs through a capability check, so the shared epoch loop needs no family branch.
- Scope, stated: there is deliberately **no** new `TrainSpec` model family and no torch `Dataset` for windows — no shipped family consumes one, and unused surface with no consumer is what the spec's strict unknown-key refusal exists to prevent.

## v10.32.0 (2026-09-08)

**Single-mesh inference.** `predict` needed a `DatasetManifest`, a split and an entry; there was no way to point a trained checkpoint at one arbitrary mesh file that was never catalogued anywhere, and no CLI verb for prediction at all. This release is that path — `predict_mesh`, `predict_file`, a `predict` verb and a `predict_file` MCP tool — built by extracting the per-mesh body already inside `predict`'s loop rather than by adding a second inference path. Pure Python; no C++, ABI, WASM or binding change. Closes roadmap section 1's single-mesh-inference bullet, which is removed. See [`doc/physicsnemo.md`](doc/physicsnemo.md#single-mesh-inference).

- **New `predict_file(checkpoint, input_path, output_path, time_step=, target_path=, device=)` → a report row**, and **`predict_mesh(checkpoint, mesh, target_mesh=, device=)` → `(mesh, row)`** for the in-memory form. Everything the prediction needs comes from the checkpoint's own **model card** — which family wrote it (so a grid checkpoint needs no different call), the sample options, the read options, the column contract and both normalizations — so nothing consults a manifest. `time_step` picks a step of a multi-step input; `target_path` supplies the paired mesh a t→t+n or coarse/fine checkpoint compares against.
- **`predict` and `predict_grid` are now loops over that same body**, with the checkpoint loaded once per call rather than per entry. Behaviour-preserving by construction and pinned as such: the gated end-to-end test asserts `predict_file` returns the *same* `rmse`, `max_error` and `num_rows` for a case as `predict` does through the manifest, and the same prediction arrays.
- **A mesh carrying no truth now predicts anyway.** With the card's target fields absent from the input — the ordinary situation for a mesh that was never part of a dataset — the `<column>_pred` arrays are written, no `<column>_error` arrays are, and `rmse`/`max_error` come back `None`. Without this rule `graph_sample` would quietly take `y` from the input's own step and report an error of a prediction against itself. **`predict` inherits it**: an entry whose meshes lack the target field now predicts with a `None` rmse instead of raising.
- **A `predict` CLI verb** (`meshioplusplus predict model.mdlus in.vtu out.vtu`), which fails by name and returns 1 when the frameworks are absent rather than surfacing a traceback, and a **`predict_file` MCP tool** (gated `physicsnemo`; the registry goes 75 → 76), whose checkpoint, input, target and output all pass through the path sandbox.

## v10.31.0 (2026-09-08)

**Proximity graphs.** `edge_index` builds a graph out of a mesh's connectivity, which is exactly the wrong graph for a particle state: a smoothed-particle or discrete-element cloud has positions and no cells, and what links two particles is the interaction radius. This release is that path — radius and k-nearest-neighbour neighbourhoods with a periodic minimum-image mode, world edges alongside the mesh edges, and bistride multiscale coarsening. Pure numpy over the bucket grid `interpolate` and `merge` already use, vectorized because a proximity graph queries every point rather than a few; no SciPy, no KD-tree, and no C++, ABI, WASM or binding change. Narrows roadmap section 1 to its remaining bullets by closing the proximity-graph one, which is removed. See [`doc/proximity_graphs.md`](doc/proximity_graphs.md).

- **New `proximity_graph(mesh_or_points, method=, radius=, max_neighbors=, box_size=, kind=, undirected=)` → `(2, E)` int64.** `"radius"` links every pair closer than the cutoff — the physical rule, whose degree varies with local density; `"knn"` links each point to its nearest `max_neighbors` and symmetrizes, so degree is near-constant and a sparse region still has edges. Both are exact and neither is seeded. The output goes through the **same canonicalization `edge_index` uses**, factored out into one owner, so a mesh graph and a proximity graph over the same pair set are byte-identical — pinned by a recorded hash of `edge_index` before and after the refactor.
- **A periodic `box_size` turns on the minimum-image convention** in the search *and* in `edge_vectors`, so a pair either side of a boundary is linked and the vector between them is short: two points at `x = 0.01` and `x = 0.99` in a unit box are 0.02 apart, not 0.98. Unwrapped positions are taken modulo the box. A radius past half the smallest box side is **refused by name** — a pair then has two images at the same distance and "the" minimum image does not exist. The periodic lattice divides each axis into a whole number of cells at least a radius wide rather than cells of exactly one radius: sizing them at the radius leaves a short partial cell at the seam, and two points either side of it can be within the cutoff while sitting two buckets apart, which a single-offset neighbourhood never looks at. That was a real bug, found by the brute-force parity test on a box whose sides are not multiples of the radius.
- **The lattice is an accelerator and never part of the answer.** Every candidate is compared under one total order, so the bucket size cannot change the output — which is what let the k-nearest sizing be retuned on measurement alone (33.6 s → 20.5 s at 200k points and k=16, identical edges), and what a test pins across a sixteen-fold range of it. Below a couple of thousand points the exact `O(N²)` route runs instead, and it is the oracle the grid path is checked against on every run. Measured: 50k points in 1.3 s (radius, 3.3 M edges) and 4.5 s (k=16); 200k in 6.4 s and 23.0 s.
- **New `edge_vectors(points, edge_index, box_size=)`** — displacement then norm, the stable PhysicsNeMo convention `graph_sample` already emitted, now with a periodic mode and one owner instead of two transcriptions.
- **New `bistride_hierarchy(edge_index, positions, num_levels=)` → `BistrideHierarchy`**, the multiscale pooling a deep graph network needs so a signal crosses the mesh in logarithmically many message-passing steps. Each level two-colours every connected component by breadth-first depth from the node nearest its centroid, keeps the smaller class, and joins survivors within two hops. Two details are load-bearing and both are pinned: the seed is the centroid-nearest node, so the classes are the geometric checkerboard rather than an artefact of node numbering; and the join is over the adjacency **plus self-loops**, without which a survivor loses the reach of the neighbours that were dropped. The published 16-node path reference reproduces exactly.
- **`graph_sample` gained `proximity=` and `world_edges=`.** `proximity` replaces the mesh edges (the particle case); `world_edges` adds a second set as **separate** `world_edge_index`/`world_edge_attr` arrays, because PyTorch Geometric increments any attribute whose name contains `index` when it batches, while a concatenated set could not be split apart afterwards. **Breaking-adjacent:** `graph_sample_version` moves 2 → 3, so a schema stored under an earlier release now compares unequal — the drift guard working, since a checkpoint trained on a proximity graph must not be replayed on a mesh one. `edge_stats` takes the same two arguments and goes through the **same shared edge builder**, since statistics gathered over a different graph than training uses normalize the wrong thing with nothing downstream to report it.
- **`TrainSpec` gained `Graph.Proximity`**, validated when the document is read rather than an epoch later, and written back only when set so an unchanged spec's document is byte-for-byte what it was. There is deliberately no `Graph.WorldEdges` and no bistride block: no shipped model family reads a second edge set, and a key a run would silently ignore is what the strict unknown-key refusal exists to prevent.
- **A `proximity-graph` CLI verb and a `proximity_graph` MCP tool** (the registry goes 74 → 75), both writing the graph as `line` cells with a `degree` array and reporting the degree spread and isolated count — the numbers that say whether the radius was well chosen. `edge_vectors`/`bistride_hierarchy`/`BistrideHierarchy` are exempted from the tool parity guard as in-memory values.

## v10.30.0 (2026-09-08)

**Point-cloud budgets.** Nothing in the library reduced a large surface cloud to the fixed token count a Transolver-, FLARE- or DoMINO-shaped model takes; `feature_matrix` produced the node table and stopped there. This release is the budgeting: a selection as a value, three ways to make one, and a mesh carrying only the selected points. Pure Python over existing machinery — no C++, ABI, WASM or binding change. Closes roadmap section 1's point-cloud bullet, which is removed. See [`doc/point_budgets.md`](doc/point_budgets.md).

- **New `select_points(mesh, count, method=, seed=, start=, bounds=)` → `PointBudget`**, whose `indices` are kept **in selection order**: for farthest-point sampling every prefix of the selection is the selection at that smaller budget, so one budget at 8192 serves 4096 and 2048 by slicing (`budget.prefix(n)`, pinned by a test). `take(values)` gathers the selected rows of any per-point array and `to_dict`/`from_dict` round-trip the budget through JSON, so it can live in a model card. The recommended token table is composition, `budget.take(feature_matrix(mesh).matrix)`, which keeps that function's column contract intact — there is deliberately no `feature_matrix(budget=...)` parameter.
- **Three methods with measured costs.** `"farthest"` is exact greedy farthest-point sampling, `O(N·count)` — 50k points / 1024 in 0.10 s, 200k / 2048 in 1.1 s, 1M / 4096 in 22.6 s on one core (the inner loop runs over three contiguous coordinate arrays with in-place ufuncs, six times faster than a row-wise `(N, 3)` reduction and identical in result). `"grid"` quantizes the cloud onto a coarse `GridSpec` lattice, keeps the point nearest each occupied cell's centre and farthest-point samples over those representatives, `O(N + count²)` — the same three sizes in 0.04 s, 0.15 s and 0.8 s, the scalable choice above a few hundred thousand points. `"random"` is the baseline. Coverage is measured by fill distance, the largest gap a budget leaves unseen: on the 56k-point Stanford bunny at 2048 points, 2.7 (`farthest`), 3.2 (`grid`) and 13.8 (`random`) length units.
- **Seeding follows the library's one convention** — a keyword-only `seed` fed to a locally constructed `random.Random` — and reaches the two deterministic methods only through `start=None`, which draws the first point; ties break on the lowest index so a selection is a function of the input alone. `bounds` restricts the candidates to a box while the indices stay indices into the whole mesh. A duplicated point is never selected twice: a cloud with fewer distinct positions than the budget fails by name rather than padding.
- **New `subsample_points(mesh, budget_or_count, record_ids=)`** — the selected points, their `point_data`, `field_data` and every Point region remapped, with a single `vertex` block so `.vtu`/`.vtp`/`.ply` all accept the cloud; cells, `cell_data` and Cell/Side regions are dropped with one warning each naming what went. `record_ids` attaches `budget:original_point_id`. `feature_matrix` on the subsampled mesh has the same columns as on the source and equals the gathered table row for row, pinned by a test.
- **A `subsample` CLI verb and a `subsample` MCP tool** (the registry goes 73 → 74); `PointBudget`/`select_points` are consciously exempted from the tool parity guard as in-memory values. The [PhysicsNeMo models page](doc/physicsnemo/models.md), which still described the grid data path as "on the roadmap" two releases after it shipped, now names what exists for each data shape.

## v10.29.0 (2026-09-04)

**Superresolution end to end.** `TrainSpec` grows a second model family, `predict` dispatches on the model card, the dashboard gains a model selector, and a GPU-executed worked example beats the trilinear baseline 19x on RMSE and 93x on the power spectrum. Closes roadmap section 1's grid bullet, which is removed. Python and browser only; no C++, WASM or binding change. See [`doc/physicsnemo.md`](doc/physicsnemo.md#superresolution-the-srresnet-family).

- **New `srresnet` family in `TrainSpec`** — `Model.Name: "srresnet"` with `ScalingFactor`/`ConvLayerSize`/`ResidBlocks`/kernels/`ActivationFn`, and a `Grid` block taking the same lattice vocabulary as `GridSpec.from_mesh`. **Cross-family strictness**: a hyperparameter meant for the other family, the wrong block, a `ScalingFactor` outside {2, 4, 8}, or `Grid.Squeeze` on an SRResNet (which is `Conv3d` throughout) each raise by name. A silently dropped key is how a run ends up training a model nobody asked for. `spec_to_dict` emits only the family's own blocks, so the MeshGraphNet document is unchanged.
- **A grid run needs `torch` and `nvidia-physicsnemo` but not `torch_geometric`.** PyG exists to batch ragged graphs and a dense `(C, D, H, W)` tensor is not one, so demanding it would refuse a runnable job over a dependency whose prebuilt wheels routinely lag torch. The grid dataset is a plain `torch.utils.data.Dataset` in the new `physicsnemo/_torch.py`; `_frameworks` and `_require_training_frameworks` take the model name.
- **The grid model card records the layout as a literal string** (`"channels_first_zyx"`) alongside both grid specs, because a checkpoint that does not say which axis is which produces finite, plausible, transposed numbers and nothing downstream can tell. `predict` reads the card and dispatches on it, writing `<field>_pred`/`_true`/`_error` into an ordinary `.vtu` — so the dashboard's prediction preview works on a superresolution run with no browser change. Rows gain `coverage` and **`spectrum_rel_l2`**, the relative L2 between the predicted and true azimuthally averaged spectra.
- **Fixed: the input normalizer was not as wide as `x`.** It was built field-by-field while the model was sized from the recorded columns, so with `Graph.Regions` on the two disagreed. With one field numpy **broadcast** the single value across every column — no error, and the region one-hots silently normalized by the field's own statistics; with several fields it was a shape crash. It is now built column by column from the recorded sources, with a one-hot contributing mean 0 / std 1, and has a regression test.
- **`normalizers_from_card` treats a missing `edge_normalization` as empty** — a convolutional model has no edges — and the trainer's epoch loop is now shared by both families, with only the forward pass differing. The MeshGraphNet path is unchanged, which its own end-to-end test confirms.
- **MCP and the dashboard**: `train_start` gains `model_name` and the grid parameters (with the parity-guarded server mirror); `train_list` lifts each family's own hyperparameters and reports `None` for the other's; the launch form gains a model selector that swaps which block it posts, and refuses a malformed resolution before starting anything.
- **New `example/physicsnemo/superresolution.py`**, executed for real: 60 cases of a multi-scale 3-D field, 150 epochs in 149 s on one GPU. Test-split RMSE 0.0081 against the baseline's 0.1556, and `spectrum_rel_l2` 0.0027 against 0.2493. That the two ratios differ so much is the point — the baseline gets the large scales roughly right and loses the small ones, which a pointwise error under-reports by design.

## v10.28.0 (2026-09-04)

The **dataset side of superresolution**: a manifest entry can now describe a paired coarse/fine series, and the PhysicsNeMo adapter can turn one into the grid pair a convolutional model trains on. Pure Python — no C++, WASM or binding change. Narrows roadmap section 1's first bullet to its last item. See [`doc/datasets.md`](doc/datasets.md#paired-cases) and [`doc/physicsnemo.md`](doc/physicsnemo.md#grid-samples).

- **`DatasetEntry` gains an optional `Target`** — a second source of exactly the same shape as `Source`, validated by the same code (parameterized on the block name, so a `Target` failure names `Target`). An entry without one is **self-supervised**, and that is the ordinary case rather than a fallback: a superresolution dataset is usually one high-resolution solve per case with the coarse input made by sampling it less finely. Absent, every existing manifest serializes byte-identically — pinned by the browser twin's byte-parity fixture, regenerated from a real Python `save()`.
- **The pairing is checked, and the second check is the one that matters.** Equal step counts catches the obvious mismatch; **agreeing times** catches the silent one, where a coarse run written at a different output interval yields two plans of the same length whose steps are at different instants, so index pairing would train the model to map one moment onto another. Times are compared only where both sides carry a real one, and an explicit `Times` list always overrides `TimeFrom` — declaring times is taken to mean them.
- **New `GridSpec.upscale_samples` / `sample_scaling_factor`**, and this is the release's load-bearing detail: a convolutional upsampler multiplies **sample** counts, not cells. `SRResNet(scaling_factor=2)` maps `(B,C,5,5,5)` to `(B,C,10,10,10)`, while `upscale(2)` on a 4×4×4-cell grid gives 9×9×9 points — right for resampling, where all 125 coarse points nest, and wrong for a model pair, where it surfaces as a shape error deep inside a loss. Both preserve the box exactly. Pinned by a gated test that runs a real SRResNet at s=2 and s=4 and asserts the output shape equals the pair's target.
- **New `grid_sample_pair` / `iter_grid_samples` / `grid_stats`** in `meshioplusplus.physicsnemo`, mirroring `graph_sample`/`iter_samples`/`field_stats` and honouring the same streaming invariant. `grid_stats` is deliberately *not* a mode of `field_stats`: a grid's statistics include the fill wherever the lattice reaches outside the mesh, so normalizing a grid with node stats would be silently wrong. `squeeze_grid` gained `reduce="mean"` and `expand_grid` a `size=`, for the 2-D-operator thin-axis idiom.
- **Surfaces**: `dataset add --target` (with its own `--target-format`/`--target-times`/`--target-time-from`/`--target-sort`); `dataset_add`'s `target_*` MCP parameters; `dataset_health` reporting `target_steps`/`pairing_error`, with a broken pair counting as a bad entry; and the browser's strict manifest twin plus its two health twins. **The MCP path sandbox now walks the target plan** — a `Target` is client input inside the document, so skipping it would have left the whole sandbox open through one extra key.
- `DatasetManifest._with` now derives its field list from the dataclass instead of enumerating it by hand. It runs on every `save()`, so a field missing from that list was silently dropped from the document — a bug this release would otherwise have shipped.

## v10.27.0 (2026-09-04)

The **mesh-to-grid data path** a convolutional or superresolution model needs: sample a mesh's fields onto a regular lattice, resample the lattice, scatter a prediction back onto a mesh, and measure whether the result carries the right small-scale content. Pure Python over existing machinery — no C++, WASM or binding change. Narrows roadmap section 1's first bullet. See [`doc/grids.md`](doc/grids.md).

- **New `sample_grid` / `scatter_grid` / `interpolate_grid` / `resample_grid`**, over the new `GridSpec` (a lattice as a serializable value) and `GridArray` (values plus the channel contract). Sampling is `interpolate(..., "barycentric")` onto the lattice's own points, so it inherits that operation's exactness for a linear field — measured at 1.8e-15 through the whole coarse-grid-to-upsample-to-fine-mesh chain — and it passes the lattice as a **cell-less** target, so a 128³ grid costs points but no connectivity. The channel contract is `feature_matrix`'s, so a multi-component array expands to `v_0`/`v_1`/`v_2` by the one rule the pandas export already uses, and `scatter_grid` inverts it so an array's *shape* survives the round trip as well as its values.
- **The layout is `(C, D, H, W)` — channels first, then z, y, x**, which is `Conv3d`'s and VTK ImageData's, and what `grid`'s x-fastest numbering already produces. `spec.dims` is world-order cell counts and `spec.shape` is tensor-order point counts; they are deliberately different words, and every schema records `"layout": "channels_first_zyx"` verbatim, because a transposed grid trains and predicts plausible numbers in the wrong places.
- **Trilinear is its own implementation rather than a third `interpolate` method**, and the difference is measured, not asserted: on a unit cell carrying `u = x·y`, `interpolate_grid` reproduces the field **exactly** (it is trilinear) while barycentric interpolation over the cell's simplex decomposition gives twice the value at the centre. Both are exact for a linear field, so a test using only one would have seen nothing.
- **New `power_spectrum`**, azimuthally averaged, whose power sums to `mean(field**2)` exactly — Parseval holds only because every mode is binned including those past the isotropic Nyquist, so `counts` is reported rather than the tail truncated. An anisotropic lattice raises rather than averaging something not rotationally meaningful. The rotation-invariance test is the one that discriminates a shell average from per-axis binning; a plausible per-axis implementation disagrees on it by 7e-4 against an assertion of 1e-15.
- **`coverage`** — the fraction of lattice points inside the source — is reported on every sample, because a grid around a concave domain is mostly fill and a model trained on it learns the fill. It is free when `extrapolate` is off (a constant probe channel rides the same interpolation call, exact because barycentric weights sum to one) and costs one extra pass when it is on.
- **`squeeze_grid`/`expand_grid`** for the 2-D-operator thin-axis idiom, taking a **world** axis. A lattice always has at least two points per axis, so `index` is required whenever the axis is longer than one rather than silently keeping the first plane.
- **Four CLI verbs** (`grid-sample`, `grid-resample`, `grid-scatter`, `grid-spectrum`, sharing `voxelize`'s six-field lattice vocabulary) and **four MCP tools** (`grid_sample`, `grid_scatter`, `grid_resample`, `grid_power_spectrum`; the registry goes 69 → 73). Cache grids as `.vti`, which stores the lattice as origin/spacing/extent so the spec is recovered exactly; a grid written to `.vtu` is a lossy round trip for this purpose and reading one back as a grid fails by name. `cell_data` is refused in both directions, pointing at `data to-point`.

## v10.26.0 (2026-09-04)

Documentation only. Adds a fourteen-page **PhysicsNeMo concepts** section explaining the framework a physics-ML pipeline sits on top of, and opens a new roadmap section for the data paths meshio++ does not yet have. No code, no ABI change, no binding change.

- **New `doc/physicsnemo/` section, and a new "Physics ML" sidebar group** holding it alongside the existing ML, dataset-manifest, adapter and dashboard pages, so the concepts, the adapter and the workflow read as one story. The pages cover what a `Module` and a `.mdlus` checkpoint are and why a model card is the thing that stops you silently misusing one; the 25 architecture families as a chart keyed on the shape of one sample; the layers and the GPU functionals you can call with no model at all; datapipes and the four contracts that bite; the simplices-only mesh type and what tessellating into it costs; the three different things "residual" means; diffusion, metrics and ONNX; uncertainty and whether an error bar is honest; the active-learning loop; where the time in a pipeline actually goes; the two kinds of parallelism; the companion packages; and the 2.2 surface with its upgrade traps.
- **Every page closes with what meshio++ actually supplies**, named against real operations rather than claimed in general — and, where it supplies nothing, says so. Two of those are worth repeating: the mesh page states that `convert_cells(simplexify)` splits conformingly because its hexahedron diagonal is fixed rather than chosen per cell, and that the *provenance map* which would let a prediction on a tetrahedron be written back onto its parent hexahedron does not exist; and the uncertainty page states that a geometry guardrail is the gap this library is unusually well placed to fill, since `compute_stats`, `compute_quality` and `extract_surface` already produce every descriptor it needs.
- **Twelve figures**, ten adapted and two newly drawn (the architecture decision chart and the active-learning loop, both hand-authored SVG rather than a new Mermaid or math dependency for four diagrams). Every figure was rendered and measured: a script checks each `<text>` element against its own box and the canvas, and no label overflows either by more than a few pixels. Figures now get a card treatment in both themes.
- **New roadmap section 1, "Physics-ML data paths beyond graphs"** (the previous sections 1–6 renumber to 2–7). It names the Kratos `PhysicsNeMoApplication` as the reference implementation to port each item from, and lists nine gaps with the grid/superresolution path first — mesh-to-lattice sampling and trilinear scatter-back, coarse/fine dataset pairing, an `srresnet` family in `TrainSpec`, and an azimuthally averaged power spectrum. Reconnaissance for that first item is recorded in the section rather than left to be redone: `interpolate(..., "barycentric")` onto a simplexified `grid` already reproduces a linear field at every lattice point to within 2e-15, and it accepts a cell-less target, so the sampler never materializes hexahedron connectivity.

## v10.25.0 (2026-09-04)

Closes roadmap section 1 — "Dataset dashboard and training integration" — in full, and removes it from the roadmap: run history and comparison, an in-viewer prediction preview, run-completion webhooks, and a design pass over the whole dashboard. Browser and Python only; no C++, WASM or binding change. See `doc/dashboard.md`.

- **Run history and comparison.** *Runs…* opens a filterable, sortable table of every run the companion process knows (an unknown value sorts last in both directions), and ticking runs overlays their validation curves on one chart with their hyperparameters side by side — only the rows they disagree on highlighted. **At most three runs compare at once**, which is a measured limit rather than a round number: overlaid curves cross, so their colours must separate in every pair, and the categorical palette's first three slots pass that check on this page's surface while the fourth does not.
- **Prediction preview.** The run panel's *Predict & show* runs one case through the run's checkpoint on the server and renders the result in the page's own mesh viewer, coloured by the prediction's error field. It uses the viewer's file slot rather than the dataset stage, so the entry being curated stays staged.
- **Run-completion webhooks.** `meshioplusplus-mcp --http --webhook URL` POSTs a JSON payload once per job when it finishes, fails or is stopped, so an outcome reaches a chat channel or a CI system without a browser tab staying open. It fires from the one place a terminal transition is observed (so exactly once, however many clients poll), and with a webhook configured the server runs a small watcher so a run that ends while nobody is looking still notifies. Like the trainer command it is **server-side only** — a client-supplied URL the server then fetches would be server-side request forgery.
- **A design pass over the dashboard**: consistent focus rings, hover affordance on cards, rows and checkpoints, short panel entrances and a pulsing "running" badge, every one of them dropped under `prefers-reduced-motion`. Two layout defects (a wrapping run title with clipped inputs, and a runs header that overflowed into a horizontal scroll) were found by rendering the pages and looking at them, and the documentation screenshots for both new views are now captured by `src/viewer/tools/screenshots.mjs`.

## v10.24.0 (2026-09-04)

Trains a model from the dataset dashboard: an in-package **trainer**, a **job manager** exposed as MCP tools, and the launch form and run panel that drive them — closing roadmap section 1's training-launch, monitoring, log-tailing and checkpoint-browser bullets. See `doc/physicsnemo.md` and `doc/dashboard.md`.

- **New `meshioplusplus.physicsnemo.run_training` / `predict`**, and `python -m meshioplusplus.physicsnemo.train --spec spec.json`. A run is described by a **training spec** — a hand-editable PascalCase settings document (`Manifest`, `Fields`, `TargetFields`, splits, hyperparameters, `Model`, `Graph`), strictly parsed like a dataset manifest — and everything it writes lands in one run directory: `metrics.jsonl` (one row per epoch), `progress.json`, the normalization stats in PhysicsNeMo's own key convention, and `checkpoints/` (periodic `save_checkpoint` state, plus `best.mdlus` and `final.mdlus`). **Every `.mdlus` carries a model card** recording the field names, the column contract and the normalization it was trained under; `predict` reads that card and refuses by name on feature drift. SIGTERM finishes the epoch, writes `final.*` and exits 143, so a stopped run leaves a usable checkpoint. The pure half (`physicsnemo/_train.py`: spec, layout, metrics, card) is stdlib+numpy on Python 3.8 and runs in the default CI matrix; `physicsnemo/train.py` is the only module importing torch, PyTorch Geometric and PhysicsNeMo.
- **The model is sized from the recorded feature schema**, not from `len(Fields)` — with region one-hots the two differ, and the example's old script would have built the wrong first layer.
- **New job manager** (`meshioplusplus/mcp/_jobs.py`) and **ten `train_*` MCP tools**: `train_defaults`, `train_start`, `train_status`, `train_metrics`, `train_log`, `train_list`, `train_stop`, `train_checkpoints`, `train_mark_best`, `train_predict`. The manager never imports torch — it spawns the trainer as a subprocess in its own process group, derives status from the process and the trainer's own files (so a run survives a server restart), and tails the log by byte offset. `train_start`/`train_predict` are gated on `torch_geometric` + `nvidia-physicsnemo` and check **before** spawning.
- **Dataset manager:** a manifest the server can see gains a *Training* section — fields, targets, splits and hyperparameters, then **Start training** — and a run panel that follows the job: a log-scaled loss chart (hand-rolled SVG, no new dependency; hover crosshair, legend and direct labels), progress and ETA, a stop control, the checkpoint list with downloads and *mark best*, and the live log with a follow toggle. A finished run raises a browser notification; **Runs…** reopens any past run, so closing the tab does not lose one.
- `example/physicsnemo/train.py` and `infer.py` are now thin wrappers over the shipped trainer and cannot drift from it; the committed results are from a fresh GPU run (100 epochs over 160 graphs in 50.3 s, validation MSE 7.9×10⁻¹ → 1.6×10⁻⁴, mean test RMSE 0.0027).

## v10.23.0 (2026-09-04)

The MCP server gains an **HTTP front-end** — the local companion process the browser dataset manager talks to — and two dataset tools, closing roadmap section 1's server-side health-summary bullet and the "companion process" half of its training-launch bullet. See `doc/dashboard.md` and `doc/mcp.md`.

- **`meshioplusplus-mcp --http`** (new `[dashboard]` extra: the `mcp` SDK plus Starlette and uvicorn, not in `[all]`) serves, from one process, MCP over streamable HTTP at `/mcp` for agents (SSE on an SDK without it) and a JSON API for the dataset manager: `GET /api/health`, `POST /api/tools/<name>` (every registry tool through the same sandbox and strict-JSON sanitizer; a tool failure stays a `200` `{"error", "error_type"}` payload) and a sandboxed `GET /api/files?path=` download. Loopback bind and a per-start bearer token by default; `--host`, `--port`, `--token`/`--no-token`, `--allow-origin` (loopback origins and the hosted docs site are always admitted; Chrome's Private Network Access preflight is answered) and `--runs-dir`. Without the extra the flag fails by name. `src/python/meshioplusplus/mcp/_http.py` is the only module importing Starlette/uvicorn.
- **New tools `dataset_find` and `dataset_health`.** `dataset_find` walks a sandboxed directory for manifests and reports each file's SHA-256 (the identity a browser-side card binds to, since the File System Access API never reveals a path); `dataset_health` is the server-side producer of the dashboard's health summary (`data_info` NaN/Inf over the data arrays with `quality:*` excluded, `compute_quality` inverted/degenerate counts and worst scaled Jacobian, arrays present, split balance, fields missing across entries, bad entries), one mesh alive at a time, every entry's resolved paths containment-checked. Both live in the pure layer (`_health.py` is stdlib + meshioplusplus, Python 3.8) and are tested in the default matrix.
- **`meshioplusplus.mcp._tools.call_tool(name, kwargs)`** is the one dispatch path (the guard moved out of `_server.py`); `create_server` takes a `host` so FastMCP's DNS-rebinding protection matches a non-loopback bind.
- **Dataset manager:** a *Companion process* panel (URL + token, remembered in IndexedDB and retried on the next visit); cards are bound to the server's manifests by content hash (`on server` chip, **Scan (server)** on the card and in the curation view — the server's summary fills the same badges and entry scans), and manifests only the server knows appear as server-only cards. Everything server-dependent stays hidden until the health probe succeeds, so a page with no server is unchanged.
- Test surface: `tests/python/test_mcp_http.py` (Starlette `TestClient`: auth, dispatch and error shapes, CORS allow-list, PNA, file downloads — the `mcp` CI job installs the extra plus `httpx`), `tool_dataset_health`/`tool_dataset_find`/`call_tool` in `test_mcp.py`, an `api.test.mjs` unit suite and a Playwright spec against a same-origin routed fake (service workers blocked: the COOP/COEP worker would answer the fetch first).
- Docs: `doc/mcp.md` and the README stated 54 and 45 tools against 57 registered; both now say the real count (59).

## v10.22.0 (2026-09-04)

The dataset manager gains an **overview depth**: a card per manifest in the picked directory, health summaries, thumbnails and a manifest diff view — closing the multi-dataset overview, drill-down, browser-side health-summary and manifest-diffing bullets of roadmap section 1. Browser-only (TypeScript over the existing `dataset.html` and WASM build); no C++, binding, wheel-asset or Python change. See `doc/dashboard.md`.

- **Overview cards.** Picking a directory now lands on a grid of every root-level `*.json` manifest — name/description, entry count, last-modified time, a split-balance bar with a colour-and-label legend, tag/group chips, health badges and a preview thumbnail — sortable by name, entries, modification time or health (worst first). A manifest the strict parser rejects is a flagged card, not a missing one. **New manifest** and **Scan all** (with optional thumbnails, one staged entry at a time) live beside the grid.
- **Drill-down and back.** A card's *Open* enters the unchanged curation view; *← All datasets* returns. Unsaved edits survive the round trip (the card follows the in-memory document and is flagged dirty; re-opening it does not re-read the file), and *Scan all* refuses to run over unsaved edits.
- **Health summaries.** Per-entry scans now also record degenerate cells and the list of data arrays present, and are cached in IndexedDB keyed by the manifest, the entry and its sources' newest modification time. The manifest-level summary (split balance, NaN/Inf and inverted/degenerate totals, worst scaled Jacobian, *fields missing across entries*, unreadable entries) is shown as badges on the card and in a new **Health** section of the curation view. The quality-NaN-means-N/A rule moved into `src/viewer/src/dataset/health.ts`, its single home.
- **Manifest diff.** *Diff…* compares two manifests — the open one on disk, its current edits, any root manifest, or pasted JSON — structurally (document fields; entries added/removed; per-entry field changes, metadata compared by value) and as a line diff over the stable serialization.
- Test surface: three new unit suites (`overview`, `health`, `diff`) and three Playwright specs on the typed `window.__datasetState`; `EntryScan`/`DatasetState` are additive.


Adds a **Blender add-on**, closing the first bullet of roadmap section 3 ("Ecosystem reach"). Blender ships Python and reads almost no FEA formats — STL, OBJ, PLY and essentially nothing else; all 43 of meshio++'s are now available under `File > Import`. See `doc/blender.md`.

- **New public API `to_blender(mesh)` / `from_blender(obj_or_mesh)` / `has_blender()`** (`src/python/meshioplusplus/_blender.py`). Python-only over the numpy the readers already return; the C++/WASM/C/Fortran core is untouched. Split like `_interop.py` and `mcp/`: the whole behaviour lives in the pure `_to_blender_payload`/`_mesh_from_blender_arrays` pair, which imports no `bpy` and is tested in the default CI matrix.
- **Quads and n-gons are kept, not triangulated.** Blender's vertices + loops + polygons model holds them natively, unlike `trimesh`, so the reduction is `_viewer_browser._renderable_surface` (extract a volume's boundary, gather each `cell_data` array through the owning cell, linearize) rather than `_interop`'s triangles-only `_triangulate`. A mesh therefore looks the same in Blender, in `view()` and in the hosted browser demo.
- **`point_data` and `cell_data` become Blender attributes** on the `POINT` and `FACE` domains, named regions become `region:<name>` boolean attributes with the same JSON sidecar `to_pyvista` uses (so a region survives a trip out through one bridge and back through the other), and `field_data` becomes custom properties.
- **New Blender 4.2+ extension** in `src/blender/`, with the meshio++ wheel bundled inside the zip — no pip step and no network access at install time. Per-platform zips (`linux-x64`, `windows-x64`, `macos-x64`, `macos-arm64`) are built by `src/blender/build_extension.py` and attached to every `v*` GitHub Release. `blender_manifest.toml` is generated rather than committed, because the bundled wheel file names carry the version; a new `blender-packaging` CI job asserts the generated manifest agrees with `pyproject.toml` and with the zip's contents.
- **New public `meshioplusplus.formats()`** returning `{"readable", "writable", "extensions"}`. The add-on's file dialog needs exactly this, and the registries behind it are private — reaching into `_helpers.reader_map` is what the ParaView plugin does, with nothing to catch a rename. `meshioplusplus.mcp._tools.formats_payload()` now delegates to it.
- `_gpu._require_framework` gained a backward-compatible `qualifier=` keyword so the Blender error can name a CPython pin instead of an accelerator. Every existing torch/JAX/PhysicsNeMo message is byte-identical, pinned by a test.
- There is deliberately **no `[blender]` pip extra**: the `bpy` wheel on PyPI is several hundred megabytes and pinned to one CPython minor, and inside Blender `bpy` is a builtin pip must never touch. The `torch`/`jax`/CuPy reasoning, and the same error shape.

## v10.20.2 (2026-08-28)

Re-cuts the release: `v10.20.1`'s tag push landed during a GitHub Actions outage — the `wasm`, `wheels` and `ci` workflows never even started a run for it (no run was created at all), `packages`'s jobs all came back `cancelled`, and a concurrent `docs` run on `master` failed to start with a spurious "workflow file issue". `v10.20.1` itself already carried both fixes below; only the release artifacts were never produced. No code changes beyond the version bump.

## v10.20.1 (2026-08-26)

Fixes two release-CI regressions introduced by v10.20.0's `gid`/static-zlib work, both caught by the `v10.20.0` tag's own CI run.

- Fixed the Windows CLI release build: `MESHIOPLUSPLUS_ZLIB_STATIC`'s vendored `zlibstatic` target never picked up `MESHIOPLUSPLUS_STATIC_RUNTIME`'s `/MT` static-CRT setting, so it stayed on the MSVC default `/MD` while the rest of the statically-linked CLI switched to `/MT` — the exact CRT-mismatch failure mode (unresolved `__imp_`-prefixed externals from zlib's `gzlib.c`/`gzread.c`/`gzwrite.c`) the surrounding code already documents for every other target. `zlibstatic` (and the unused sibling `zlib` shared target) now get the same `MSVC_RUNTIME_LIBRARY` property whenever both options are on.
- Fixed a WASM provenance leak: `writeMesh`/`convert`/`convertSurface`/`convertSurfaceOps` now reset the scope-less provenance record immediately before writing (mirroring Python's public `write()`), so an operation note left by an earlier, unrelated scope-less pipeline call can no longer leak into a later default write's header.

## v10.20.0 (2026-08-26)

Closes roadmap section 1's last bullet — `gid` write support now ships in every release artifact, not just from-source builds. See `doc/formats/gid.md`.

- **The release CLI binaries and every published PyPI wheel now carry `gid` write support.** `cli.yml` and `wheels.yml` used to build `-DMESHIOPLUSPLUS_WITH_ZLIB=OFF` on every leg (not just the Windows wheels, as this project's own docs previously — incorrectly — scoped it: `wheels.yml`'s shared `CIBW_ENVIRONMENT` turns zlib off on all four OS legs, Linux included), which turned the GiD writer off along with the rest of the optional native paths. A plain `find_package(ZLIB)` was never the fix, since it links the system libz **dynamically**, defeating the "one dependency-free artifact" premise those two workflows exist for.
- **New `MESHIOPLUSPLUS_ZLIB_STATIC` CMake option** (OFF by default, orthogonal to `MESHIOPLUSPLUS_STATIC_RUNTIME`): when ON, `FetchContent` vendors a pinned, SHA256-checked zlib 1.3.1 release and links only its static `zlibstatic` target, aliased to `ZLIB::ZLIB` so every existing zlib/gidpost consumer needed no further change. Both `cli.yml` and `wheels.yml` now pass it; their dependency-verification steps (Linux `ldd`, and a newly-added macOS `otool -L` check) assert no dynamic `libz` dependency appears.
- **Breaking:** the default `.vtu`/`.vtp` write from the release CLI binaries and every published wheel is now **zlib-compressed** instead of uncompressed. `registry.cpp`'s VTU/VTP writer entries already follow a "default follows the build" convention identical to XDMF's HDF-vs-XML choice (`#ifdef MESHIOPLUSPLUS_HAS_ZLIB` picks compressed over uncompressed for the default write), so gaining zlib for `gid` gains it for these two formats' defaults too. The output is still standard, universally-readable VTK XML — only the raw bytes differ from prior releases; pass `--codec none`/`zlib=False` to keep writing uncompressed.

## v10.19.0 (2026-08-25)

Closes roadmap section 1's reader bullets: `gid` is now **read and written in all three flavours**, so the section is narrowed to a short list of refinements rather than a missing capability. See `doc/formats/gid.md`.

- **The `gid` format gained a reader** — ascii (`.post.msh`/`.post.res`), compressed binary (`.post.bin`) and HDF5 (`.post.h5`). gidpost's public API has zero read functions, so this is meshio++'s own code against the on-disk grammar, in a separate translation unit outside `gid.cpp`'s gidpost guard.
- **`gid` is now readable in strictly more build configurations than it is writable.** Writing needs gidpost, which is hard-gated on zlib; reading needs *nothing* for the ascii flavour, zlib for binary, HDF5 for hdf5. The statically-linked release CLI binaries and the Windows wheels build with zlib off — they still cannot write GiD, but they now read the ascii flavour. `gid_available` reports the write side (its meaning is unchanged from v10.18.0), and the new `gid_readable` the read side.
- **Two real-world variants are handled that meshio++'s own writer cannot produce**, both found in a genuine third-party (Kratos-produced) file: the full node table repeated in every `MESH` block (de-duplicated by node id), and element ids restarting at 1 per block (tracked per block, so results resolve through the Gauss-point set's mesh name). Gapped and non-contiguous node ids are supported.
- `ReadOptions::mTimeStep` selects one step of a multi-step results file and is honoured natively; `read_metadata` reports block shapes from the `MESH` headers and declines (falling back to a full read) for anything it cannot summarize cheaply. A trailing material column round-trips as `cell_data["gmsh:physical"]`; an all-zero column is deliberately not surfaced, since the binary and HDF5 writers always emit one.
- `hexahedron27`/`wedge15`/`pyramid13` are supported in both directions (see below).
- `sniff_format` gained two GiD signatures (`GiD Post Results File`, and `MESH "` — with the quote, since a bare `MESH ` is exactly the generic token the sniffer's own contract refuses to claim). The deflated `.post.bin` and the generic-HDF5-magic `.post.h5` are deliberately not sniffable.
- **Reader conformance to CIMNE's published grammar**, in three places none of which meshio++'s own writer can reach (gidpost emits one fixed casing, always writes a mesh name, and always spells the 1-D type `Linear`), so each is pinned by a hand-authored fixture rather than a round trip: structural **keywords are matched case-insensitively**, as GiD specifies — the manual's own worked example opens a block with `Coordinates` and closes it with `end coordinates`, so a case-sensitive reader rejects the specification's own example file; the **mesh name is optional** (`MESH dimension 3 ElemType Linear Nnode 2` is legal), and is no longer taken from the `dimension` keyword, which had let a `GaussPoints` set declared `OnMesh "dimension"` bind to a nameless mesh and attach a result not belonging to it; and **both spellings of the 1-D element type** are read, gidpost emitting `Linear` while CIMNE's current grammar names it `Line`.
- **Resolved a documentary conflict over `hexahedron20`'s mid-edge node order.** CIMNE's own GiD 6-era figure numbers the mid-edge nodes bottom-ring/verticals/top-ring — exactly Kratos's *internal* order, the one Kratos's own GiD writer permutes *away* from before emitting a file — which taken at face value said meshio++'s identity mapping was wrong. Confirmed: GiD's actual expected order is the one Kratos **writes**, i.e. the post-swap order, which is meshio++'s own `hexahedron20` table — identity is correct, and the figure is outdated (CIMNE's current grammar dropped it entirely, specifying no mid-edge order at all). The `GidOrdering` test suite's comment was corrected regardless of this resolution: it pins that the writer applies *no permutation*, and cannot pin that meshio++'s order is GiD's, since its expected positions come from meshio++'s own edge table.
- **Added `hexahedron27`/`wedge15` (and confirmed `pyramid13`) support, closing this section's "unverified node orderings" item.** Both were previously refused in both directions since their GiD orderings had never been independently verified. Derived directly from Kratos Multiphysics's own geometry classes (`kratos/geometries/hexahedra_3d_27.h`, `prism_3d_15.h`, `pyramid_3d_13.h`), cross-checked against a second, Element-agnostic Kratos source (`kratos/input_output/vtk_output.cpp`'s Kratos-to-VTK conversion, mirrored in `ensight_output.cpp`) rather than the narrower `gid_mesh_container.h` reorder `hexahedron20`'s own resolution originally cited — that reorder, found while deriving these two, turns out to live only in that file's *Conditions*-writing path, not Elements; `hexahedron20`'s conclusion is unaffected, since `vtk_output.cpp` independently reproduces the identical swap, and is now the more precise citation for it too. `hexahedron27`/`wedge15` need genuine self-inverse permutations (`dst[c] = src[p[c]]`, the `med_node_perm()` convention already used elsewhere in this repo, shared between writer and reader in a new `gid_common.hpp` table so the two cannot drift): Kratos's internal node order splits mid-edge nodes bottom-ring/VERTICALS/top-ring where meshio++'s own table splits them bottom-ring/top-ring/VERTICALS, and `hexahedron27` additionally permutes its six face-centre nodes. `pyramid13` needs no permutation at all — the one type of the three with **no** Kratos-GiD precedent whatsoever, since Kratos never registers a GiD mesh container for any pyramid; its ordering rests on Kratos's internal geometry convention alone, confirmed to already match meshio++'s via `vtk_output.cpp` explicitly skipping any conversion for it.
- **Added `Matrix`/`PlainDeformationMatrix`/`MainMatrix`/`LocalAxes` and all three `Complex*` result types** — all nine `GiD_ResultType`s round-trip. A caller **declares** a non-default type via `field_data["gid:result_type:<name>"]` (`GidResultType`/`kGidResultTypePrefix` in `gid.hpp`; `meshioplusplus.gid.ResultType`/`RESULT_TYPE_PREFIX` in Python) — chosen over a side-channel struct because the registry's `(path, mesh)` writers cannot carry one, which would have hidden the declaration from the CLI/WASM/C API/Fortran; `field_data` reaches every one of them for free, as `gmsh:physical`/`med:num` already do. **Corrects a false claim this repo previously made**: GiD's `Matrix:6` is `Sxx Syy Szz Sxy Syz Sxz`, which is *already* meshio/VTK's symmetric-tensor order (no permutation needed), not a differing one as earlier documented — that false claim was the stated reason for splitting a 6-component array into six scalars rather than mapping it to `Matrix`. `ComplexVector` interleaves real/imaginary parts per component while `ComplexMatrix` blocks them (every real, then every imaginary) — the same family, opposite conventions, both taken verbatim from CIMNE's Customization Manual and pinned literally by position-encoding fixtures. An undeclared array keeps the historical inference exactly (so output for a mesh with no declarations is byte-identical to before this existed), an illegal component count for a declared type is a `WriteError` naming the array and the legal counts, and a declaration is recorded on read only when it carries information beyond what the inference would already have produced.
- **Arbitrary Gauss points per element**, closing an item `doc/roadmap.md` had listed not as a gap but as a permanent data-model **boundary** ("a boundary, not a defect to fix"). A G-point, k-component `cell_data` array is stored **flat** as `(ncells, G*k)`, Gauss-point-major — the order GiD's own `Values` rows arrive in, so neither direction re-packs — declared by `field_data["gid:gauss_points:<name>"]`. This reuses the `gid:result_type:` mechanism including its "declare only what carries information" rule: **`G == 1` declares nothing**, so the historical one-value-per-element path is untouched and its bytes are unchanged (pinned against a baseline captured before the change). The declaration is required rather than inferred because `(ncells, 3)` is genuinely ambiguous — a 3-component vector at one Gauss point, or a scalar at three — and a result type's legal widths are checked against **k**, never `G*k`.
- **Gauss-point natural coordinates.** GiD places points itself (`Natural Coordinates: Internal`) only for specific counts per family (triangle 1/3/6, quad 1/4/9, tetra 1/4/10, hexahedron 1/8/27, wedge 1/6, pyramid 1/5; **line accepts any count and forbids `Given` outright**), so any other count supplies them via `field_data["gid:gauss_coords:<celltype>:<G>"]` — `G*dim` doubles, point-major, keyed by `(cell type, G)` because that is what a GiD Gauss set actually depends on, so two arrays sharing a block and a count share one set. Omitting them for a count GiD cannot place is a `WriteError` naming the type and its legal counts, never a file GiD would silently reject. Reading captures a `Given` set's coordinates into the same key so it can be written back — **ASCII-only**, since binary stores them as raw double records the string-record scanner cannot reach and the HDF5 attribute layout is unverified; the *count* is read in all three flavours, so G>1 values round-trip everywhere.
- **Fixed a pre-existing binary-reader bug this work surfaced.** gidpost's `CPostBinary_WriteValues` suppresses a repeated element id exactly as the ASCII writer does — its own source comments that this is "only useful when writing values for gauss points in elements" — but meshio++'s binary reader assumed one id per record, so the moment a result carried more than one Gauss point it misread a float value as an id and failed with a truncation error. It now consumes exactly `G*k` reals after each id, which is sound because gidpost resets `m_LastID` per `Values` block and element ids are unique within one.
- **Documented a separate, also pre-existing binary-flavour limitation** rather than leaving it to be mistaken for a Gauss-point defect: the binary stream carries neither a row width nor a component count, so its reader can only use the declared type's canonical width (Scalar 1, Vector 3, Matrix 6). A 2-component `Vector` is legal GiD and round-trips through ASCII, but is unrecoverable from binary at **any** G including the default 1. Pinned by a test so it stays visible.
- **`ResultGroup` blocks are read** (ascii flavour). A ResultGroup packs several results sharing one analysis, step and location into a single wide `Values` row; each `ResultDescription` member is unpacked into an **ordinary result**, so point/cell routing, the flat Gauss-point layout, result-type recording and `time_step` selection all apply to it unchanged, with no second code path to drift. Member widths come from `:N` when present — a `ResultDescription` is the one place GiD states a width, which a plain `Result` header does not — and otherwise from the manual's rule that inside a group a `Vector` is always 3 components and a `Matrix` always 6. A type admitting several widths with no `:N` (the `Complex*` family), or a row width disagreeing with the members' total, is refused by name rather than guessed, since either would silently mis-associate every column after it. Binary and HDF5 **refuse by name**; the binary path previously *skipped* the record silently, losing the whole group with no diagnostic at all. meshio++ still never writes a ResultGroup.
- **Fixed a pre-existing defect that made the reader's refusals unreachable.** The optional-`.post.res` handler caught **every** `ReadError` from the results parse and cleared all results — so a malformed results file silently yielded geometry only, discarding even results that had already parsed cleanly, and none of the reader's carefully-named refusals ever reached a caller of `read`. An **absent** sibling is still silent (the file is genuinely optional), but a **malformed** one now emits a warning naming the problem. Note the warning goes to the C++ stderr *file descriptor*, so Python tests capturing it need `capfd`, not `capsys`.
- **New `ascii_zipped` write mode** (`GidMode::AsciiZipped`), the write counterpart to gidpost's `GiD_PostAsciiZipped` — the same two sibling files as `ascii`, gzipped, sharing that branch entirely since only compression differs. Reading has always handled it via the reader's gzip sniffing. Spelled with an underscore rather than a hyphen so no binding needs the symbol translation `gradient`'s hyphenated method names required in Julia. **`auto` deliberately never resolves to it**: no extension can express "zipped" (a gzipped file still ends `.post.msh`), and inferring it would change what every existing `.post.msh` write produces. Appending the enumerator is additive — `GidMode` already declares an explicit `: int` underlying type — so the ABI is unaffected.
- **Corrected a roadmap item that rested on a false premise.** `doc/roadmap.md` listed "Regions ↔ GiD mesh groups", proposing that `Group`/`End Group` be mapped to named regions. CIMNE's manual describes `Group` as wrapping the `MESH` blocks belonging to **one time step**, so GiD can swap meshes as the step changes — it exists for re-meshing and adaptive analyses, not as an entity set — and the postprocess format has **no node/element set concept at all**, the material column (already round-tripping as `gmsh:physical`) being its only grouping. There is no region mapping to build; the bullet now says so, and points at the transient axis as `Group`'s real analogue.
- **Multi-step series round-trip in both directions.** Steps live in the `.post.res` headers; `read_metadata` now reports them as `time_values` through a header-only scan that skips every `Values` body without tokenising it, and `gid` joins the sequence layer's step-capable formats. Fan-out (`out_{step}.vtu`), fan-in (N meshes → one multi-step file), `TimeSeries` and `info --fast` step counts all work. **This closed an unlisted gap, not a roadmap item**: the reader had always honoured `time_step`, but metadata never opened the results sibling, so it reported *one* step whatever the file held — `gid` could be read at step N but never asked how many steps it had, which made it invisible to the whole sequence layer.
- **`Group`/`OnGroup` support, for what it actually is.** A `Group` wraps the `MESH` blocks of one mesh serving a consecutive run of steps, so GiD can swap meshes as the step changes (re-meshing / adaptive analyses). Reading now resolves the requested step to its owning group and uses only that group's meshes, results and **nodes** — each group carries its own coordinates. Previously the wrapper was skipped and every group's blocks were merged into one garbled mesh while `OnGroup` results were dropped outright; no test had a multi-group file, so the bug was latent. Writing emits a group only when a step's mesh *differs* from the previous one's, so a fixed-mesh series emits none and keeps the single-step output shape.
- The multi-step writer is deliberately **not** a public class: `write_gid_series` is a free function that *pulls* steps and is driven by `sequence_to_timeseries`. The transient surface meshio++ exposes is the sequence layer, which already carries every binding, so this needed no `GidTimeSeriesWriter` with its own C/Fortran/Julia/R/WASM surfaces — only one pybind function, since Python's sequence engine is a separate pure-Python implementation rather than a shim over the C++ one.
- **Fixed two upstream gidpost bugs** that made its HDF5 flavour single-use per process. `G_num_HDF5_files` was incremented on open but never decremented on close, so the second HDF5 file opened in a process — even strictly after the first was closed — was refused; and that refusal path returned without unlocking gidpost's process-global mutex, deadlocking every subsequent gidpost call, including ones for the unrelated ascii and binary flavours. Both are fixed in the vendored copy and recorded in `src/cpp/third_party/gidpost/README.meshioplusplus.md` for re-application on a version bump. A single-file program never reaches either bug, which is why v10.18.0 shipped without noticing.

## v10.18.0 (2026-08-25)

Closes the vendoring/build/write-path portion of roadmap section 1 (`doc/roadmap.md`); the reader remains open and the section is narrowed rather than removed. See `doc/formats/gid.md`.

- **New `gid` format: the GiD postprocess format, write-only, on a vendored hardcopy of CIMNE's gidpost 2.14** (`src/cpp/third_party/gidpost/`, BSD-2-Clause-Views). Three on-disk flavours (`GidMode::Ascii`/`Binary`/`Hdf5`, inferred from the path or passed explicitly): ascii writes a `<stem>.post.msh`/`<stem>.post.res` sibling pair, binary one deflated `<stem>.post.bin`, hdf5 one `<stem>.post.h5` (needing `MESHIOPLUSPLUS_WITH_HDF5=ON` in addition to gidpost's own HDF5 flavour, which needs the HDF5 high-level library specifically). Hard-gated on zlib (gidpost deflates unconditionally): `MESHIOPLUSPLUS_WITH_GIDPOST` (default ON) auto-disables when zlib is unavailable, and the writer always exists, raising a named error citing the missing build flags rather than ever falling through to another format for a `.post.msh` path.
- Cell-type mapping covers `vertex`/`line`/`line3`/`triangle`/`triangle6`/`quad`/`quad8`/`quad9`/`tetra`/`tetra10`/`hexahedron`/`hexahedron20`/`wedge`/`pyramid`, every one identity (no node permutation) -- independently cross-checked against Kratos Multiphysics's own production GiD writer. `hexahedron27`/`wedge15`/`pyramid13` (orderings not independently verified), `polygon`/`polyhedron`, and higher-degree Lagrange types raise by name rather than guess.
- `point_data` writes `GiD_OnNodes`; `cell_data` writes `GiD_OnGaussPoints` against a synthetic one-point Gauss set per block (GiD has no "on cells" concept). An integral `gmsh:physical` cell_data array becomes the geometry file's material-id column. A result array's component count maps to `GiD_Scalar`/`GiD_Vector` for 1/2/3 components; anything else splits into that many named scalars (a 6-component array is deliberately not mapped to `GiD_Matrix` -- meshio++ has no tensor declaration and GiD's own symmetric-tensor component order differs from meshio/VTK's).
- Named regions, `field_data`, and multi-step results are not carried; provenance renders as one `# Name: value` "user attribute" line per entry (an HDF5 group attribute in the `hdf5` flavour).
- **No pure-Python reference exists for this format** -- gidpost cannot be cheaply reimplemented, and a second implementation of its node-ordering permutations risks disagreeing with the first near exactly the cases the ordering tests exist to pin. `meshioplusplus.gid` is a C++-core-only surface, registered unconditionally (never deregistered when unavailable, unlike the `openfoam` writer's own precedent) so `.post.msh` always resolves to this format's own actionable error rather than falling through to `gmsh`/`ansys`/`freefem`.
- **Fixed extension-dispatch resolution to try the longest matching suffix first**, in both the C++ registry (`resolve_format`) and the Python `_helpers._filetypes_from_path` -- `.post.msh` previously resolved to `.msh` (in C++) or to `.msh`'s own shortest-match-first candidate list headed by `ansys` (in Python), silently bypassing any writer registered for the full compound extension. Verified behaviour-preserving for every extension registered before this change.
- Reachable from every registry-driven surface (WASM, C API, Fortran, native CLI) with no per-binding code beyond the shared registry entries. Absent from the statically-linked release CLI binaries and the Windows wheels, both of which build with zlib off.

## v10.17.0 (2026-08-24)

Closes roadmap section 1 in full; the section has been removed from
`doc/roadmap.md`. See `doc/provenance.md`.

**ABI: `MESHIOPLUSPLUS_ABI_VERSION` 10 -> 11.** This is the one item a C++
consumer must act on: `MeshMetadata` gained two fields (below), a Tier A
layout change, so its `sizeof` moved 256 -> 288 and the C++ variant
libraries' `SOVERSION` becomes 11. Recompile against the new headers; the C
ABI is unaffected (`mio_read_metadata` is an opaque handle, so its new
accessors are purely additive).

- **Provenance is now on by default.** An ordinary `write()` records the
  conversion assumptions raised while it ran, with no scope needed (it
  shipped opt-in in v10.16.0). The fields that need a caller to supply them
  -- source, target, operation chain -- stay absent without a scope, and no
  timestamp is added, so a write that loses nothing is byte-for-byte what
  v10.15.0 produced and every byte-pinned test is untouched. Turn it off with
  `MESHIOPLUSPLUS_PROVENANCE=off` or `set_default_provenance_mode(Off)`.
- **Scope-less notes are bounded to the write that raised them.** Without
  this a note had no lifetime: `extract_surface(A)` dropping a region put
  `Note [regions-dropped]: extract_surface: ...` into an unrelated mesh B's
  header. The public write entry points (`meshioplusplus.write`,
  `registry_write_ex`, `mio_write`) now reset the scope-less record first. A
  note raised *before* a write is dropped rather than misattributed -- use an
  explicit scope spanning the operations and the write to capture those --
  and the low-level C++ writers, called directly, still leave that to the
  caller.
- **Reading a provenance block back** -- `read_metadata()` now reports the
  block a file carries as `provenance` (one line per entry, comment
  punctuation stripped) plus `provenance_recognised`, and both CLIs' `info`
  print a `Provenance:` section, labelled `(not written by meshio++)` when
  the block does not start the way this library writes one. New
  `MeshMetadata::mProvenance`/`mProvenanceRecognised`, C accessors
  `mio_read_metadata_num_provenance_lines`/`_provenance_line`/
  `_provenance_recognised`, and a WASM `readProvenance(path)`.
- **One scanner, not forty-four parsers.** The rendered block's content lines
  are format-independent by construction, so recovery is a single pass over
  the file's head bytes rather than a parser per format. Three wrappings need
  more than a leading-marker strip: a keyword slot that wraps the text
  (Tecplot's `TITLE = "..."`, Ansys's `(1 "...")`), whose closing punctuation
  is removed only because that opener put it there -- never unconditionally,
  since `Converted from x (fmt)` legitimately ends in `)`; a box-drawn banner
  (OpenFOAM); and a NUL-padded fixed-width binary slot (STL, EnSight).
  Exodus is the one exception, its block riding a netCDF `title` attribute,
  read natively and fed to the same scanner.
- **Raw lines, not a re-parsed record.** A block can be hand-edited,
  truncated by a fixed-width slot, or written by a later release carrying
  unknown fields; returning what is there, plus a flag for whether it starts
  like ours, cannot silently drop or mis-attribute any of those.
- **Never re-emitted** -- writers render from the live record only, so
  converting a file N times leaves one block, not N.
- **Fortran, Julia, R and WASM bindings** for the scope API, each giving the
  C ABI's begin/end pair a lifetime in its own idiom: Julia a `do`-block, R
  an `on.exit`-paired `mio_with_provenance()`, WASM a
  `withProvenance(mode, fn)` callback -- all three closing the scope even
  when the body throws. Fortran keeps the explicit pair, matching its own
  `m%free()` convention.
- **Wider conversion-assumption coverage** (OFF, PLY, STL, UNV, CGNS in both
  engines; MDPA, MED, OpenFOAM in the C++ writers), guarded by a new
  cross-engine test asserting that wherever both engines write a mesh they
  record the same notes. Deliberately not an exhaustive sweep: most `warn()`
  calls are reader-side or user-error diagnostics rather than conversion
  assumptions, and some are stale -- `avsucd`'s "can only write one cell data
  array" fires while both engines demonstrably write every array. Each site
  needs checking, not translating; `doc/provenance.md` records this.

## v10.16.0 (2026-08-24)

Closes roadmap section 1 in full except surfacing provenance on read
(bullet 8, left open) -- the opt-in richer provenance record `doc/roadmap.md`
called for after v10.15.0's audit-and-normalize pass. See
`doc/provenance.md` for the full design note.

- **Opt-in provenance record** (`detail/provenance.hpp` + `_provenance.py`)
  -- a caller can now wrap a write in a scope
  (`meshioplusplus::detail::ProvenanceScope` in C++,
  `meshioplusplus._provenance.scope` in Python) to have a writer with a
  free-text header slot render, alongside the unconditional one-line credit,
  the source path/format, the target format/encoding/codec/float-format
  actually used, the operation chain, the conversion assumptions accepted
  along the way, and an ISO-8601 timestamp (`SOURCE_DATE_EPOCH`-aware, with
  an explicit off switch). With no scope open, output is byte-for-byte what
  v10.15.0 wrote -- the record is opt-in everywhere, so no existing
  byte-pinned test needed an exemption.
- **Thread-local scope, not a `WriteOptions` field** -- reconnaissance found
  that only 4 of the write paths reach `registry_write_ex`
  (`WriteOptions`'s home); Python's own path and WASM's both bypass it
  entirely, and growing `WriteOptions` is a Tier A ABI change. The record
  lives in a thread-local RAII-scoped context instead (the
  `set_buffer_allocator` shape), read by every writer exactly where it
  already read the v10.15.0 tag, so no writer signature changed.
- **The Python<->C++ bridge** -- opening a scope from Python also opens a
  matching one on the C++ side (`bindings/python/_core.cpp`'s
  `provenance_scope_push`/`_pop`), and every note/set call mirrors into
  both, so a scope opened from Python is honoured by the ~40 of 44 formats
  whose write goes through the compiled writer, not just the pure-Python
  fallbacks.
- **`SlotTier`** (`None`/`Bounded`/`SingleLine`/`Block`) classifies what
  each writer's own header slot can hold; `Mode::Required` raises only for
  `SlotTier::None` (no slot at all) -- degrading to a smaller slot's honest
  maximum is not treated as a failure.
- **Deliberately no engine marker in the file** -- the roadmap asked for one,
  but it directly contradicts the harder guarantee that the C++ core and its
  Python fallback emit character-identical bytes. Which engine wrote a file
  is reported through `current_provenance()`/`scope.get()` instead.
- **Operation-chain sourcing** -- the settings pipeline (both the C++ engine
  and Python's separate pure-Python engine) records each step it runs,
  pinned to render identically across the two for the common parameter
  shapes.
- **Conversion-assumption capture wired at two reference sites** --
  `detail::warn_regions_dropped` (the 9-operation choke point) and the OFF
  writer's cell-type-skip warning, in both engines with matching wording.
  Extending coverage to the remaining ~60-80 sites is recorded as
  mechanical follow-up work, not attempted exhaustively here.
- **C ABI**: `mio_provenance_scope_begin`/`_end`, `mio_provenance_note`,
  `mio_provenance_set_source`, `mio_provenance_set_target`. Fortran/Julia/R/
  WASM bindings are a recorded follow-up.
- Tier C additive ABI change (new enums/structs/class/functions in
  `detail/provenance.hpp`; `kProvenanceTag` itself is byte-identical to
  v10.15.0) -- `MESHIOPLUSPLUS_ABI_VERSION` stays 10.

## v10.15.0 (2026-08-24)

**Breaking:** normalized the one-line provenance credit every writer with a
free-text header slot emits, per `doc/roadmap.md` section 1's "audit and
normalize" bullet. ~25 writers previously hand-wrote their own version of the
line and had drifted three ways: a stale `meshio` (not `meshio++`) name in
several Python writers, a `(C++ core)`-vs-`v{version}` split that made the
C++/Python fallback boundary visible in output bytes, and four writers
(`obj`, `ply`, `exodus`, `flac3d`) embedding a wall-clock timestamp that made
writing the same mesh twice produce different bytes. Every affected writer
now emits one canonical line, `Written by meshio++ v<release>`, from a single
source on each side (`detail::kProvenanceTag` in C++,
`meshioplusplus._provenance.TAG` in Python), so the two engines are
character-identical and output is deterministic. `nastran` is the one
documented exception: its C++ reader is gated on a sentinel comment line the
Python writer never emits, so the C++ file carries the sentinel followed by
the tag while the Python file carries only the tag — see
[`doc/formats/nastran.md`](doc/formats/nastran.md). `doc/formats.md` gained a
Provenance section auditing every format's comment syntax, header position,
and whether it carries the tag today, which is the reference for the rest of
the roadmap section. Output bytes change for every affected format — anyone
diffing or hashing written files should expect this.

## v10.14.0 (2026-08-24)

**Roadmap §1's "ODT" bullet closed in full** — v10.13.0 shipped ODT
*smoothing* (`SmoothMethod::Odt`, positions only, fixed connectivity); this
release adds the connectivity-changing half, genuine ODT *remeshing*.

- **`optimize_volume`** (new operation, `operations/optimize_volume.{hpp,cpp}`,
  [`doc/optimize_volume.md`](doc/optimize_volume.md)) — raises a tetrahedral
  mesh's worst element quality by *ODT remeshing*: it alternates the ODT
  vertex relocation (reused from `smooth`'s `method="odt"`) with
  quality-improving topological **flips** (2-3 and 3-2), so both the vertex
  positions and the connectivity change. The missing third member of a trio
  whose other two each do half the job: `SmoothMethod::Odt` moves points on
  fixed connectivity, `remesh_volume` discards the input's tets for a fresh
  lattice mesh. **Predicate-free** (in-posture): a flip is applied only when a
  pure signed-volume test finds the local configuration convex AND the minimum
  scaled Jacobian over the new tets strictly beats the minimum over the tets it
  replaces (Freitag & Ollivier-Gooch's improvement rule, monotone in worst
  quality and hence terminating) — no in-sphere/Delaunay predicate. The flips
  touch only interior faces/edges, so with `preserve_boundary` the boundary
  surface is byte-identical to the input's (watertight in ⇒ watertight out);
  the point set is invariant, so `point_data`/`field_data` and Point regions
  carry while `cell_data`/Cell/Side regions are dropped. Tet-only; C++-core
  only (no numpy fallback, the flip acceptance being a discrete sign/near-tie
  branch); byte-identical across mesh backends and thread counts. Shipped
  across every binding surface — Python `optimize_volume`, C
  `mio_optimize_volume`, Fortran `m%optimize_volume`, Julia `optimize_volume`,
  R `mio_optimize_volume`, WASM `optimizeVolume`, the `optimize-volume` CLI
  verb in both CLIs, an `OptimizeVolume` settings-pipeline step, and an
  `optimize_volume` MCP tool. Tier C additive (a wholly new
  header/`.cpp`/C entry point) — no ABI bump (`MESHIOPLUSPLUS_ABI_VERSION`
  stays 10); `OptimizeVolumeOptions` is pinned in `test_abi_layout.cpp` from
  this release.

## v10.13.0 (2026-08-23)

**Roadmap §1 closed in full** — the last remaining bullet, "Volumetric
CVD/ODT", closes by a different algorithm than the one named (honestly
reported as such, not as a literal Delaunay/CVD implementation), plus a
separate closure of the "ODT" half by name.

- **`remesh_volume`** (new operation, `operations/remesh_volume.{hpp,cpp}`,
  [`doc/remesh_volume.md`](doc/remesh_volume.md)) — retetrahedralizes a
  volume mesh (or a closed surface) at a caller-chosen resolution by
  isosurface stuffing (Labelle & Shewchuk, SIGGRAPH 2007, implemented from
  the published description only) over a body-centered cubic (BCC) lattice.
  `remesh`'s volumetric sibling: nothing else in this repo can *raise* a
  tet mesh's quality at a chosen resolution. Every uncut lattice tet has a
  dihedral angle from a fixed, mesh-size-independent set; `warp_fraction`
  (default `0.35`) moves lattice vertices near the surface onto it, trading
  a small, *measured* chance of non-manifold boundary edges (reported as
  `num_non_manifold_edges`) for substantially better boundary tet quality —
  `0` disables warping for an exactly watertight but lower-quality boundary.
  Unlike `remesh`, accepts a volume mesh directly (its boundary is
  extracted internally). Shipped across every binding surface (Python, C
  API, Fortran, Julia, R, WASM, both CLIs, the settings pipeline, MCP) in
  one release.
- **`SmoothMethod::Odt`** (`smooth(mesh, method="odt")`,
  [`doc/smooth.md#odt-smoothing`](doc/smooth.md#odt-smoothing)) —
  optimal-Delaunay-triangulation smoothing on `smooth`'s existing, fixed
  connectivity: each free interior tet vertex moves to the closed-form
  volume-weighted average of its incident tets' circumcenters. Tet-only;
  closes the "ODT" half of the roadmap bullet's name honestly as
  *smoothing*, not *remeshing*. Needed no new `SmoothOptions` field (reuses
  the existing negative-`lambda`-means-"this method's own default"
  sentinel, default `0.9`) and no C API/Fortran/Julia/R/WASM code changes
  beyond the method string, since every surface already passes it through
  unchanged. C++-core only, no numpy fallback (unlike `laplacian`/`taubin`).
- **`MESHIOPLUSPLUS_ABI_VERSION` 9→10.** `remesh_volume` alone is Tier C (a
  wholly new header). The bump is caused by `SmoothMethod::Odt`: giving
  `SmoothMethod` an explicit `: std::uint8_t` underlying type for the first
  time (previously the scoped-enum default `int`) is a Tier A layout
  change under `doc/abi.md`'s own rule, independent of the appended `Odt`
  enumerator itself or of `SmoothOptions` gaining no new field.
  `SmoothOptions` is pinned in `tests/cpp/test_abi_layout.cpp` for the
  first time as a result.

## v10.12.0 (2026-08-20)

**Roadmap §1 closed** — the anisotropic metric ships, closing the last
bullet with a clean scope boundary (Volumetric CVD/ODT remains, explicitly
"not a follow-on task; a project in its own right").

- **`remesh` anisotropic metric** (`metric="anisotropic"`,
  `RemeshOptions::mMaxAnisotropy` / `max_anisotropy`, default `4.0`, a
  measured value) — clusters shaped by a local curvature tensor rather than
  isotropic distance, elongating along low-curvature directions and staying
  compact across sharp ones. Built on the same per-vertex curvature fit
  `mGradation` already computes (widened to keep the principal directions
  instead of collapsing them to a scalar magnitude); packs into the exact
  10-double accumulator `metric="quadric"` already uses, so it costs the
  same per-move solve and needed no new per-cluster storage. The two
  existing `RemeshMetric` branch sites were also converted from implicit
  `if/quadric-else` pairs to explicit exhaustive `switch`es with no
  `default:`, so a future metric is a compiler error instead of a silent
  misclassification.
- Shipped across every `remesh` binding surface: Python, both CLIs
  (`--max-anisotropy`), the settings pipeline (`MaxAnisotropy` step
  param), the MCP tool, and WASM directly against the C++ API; C, Fortran,
  Julia and R via a new `mio_remesh_ex`/`mio_remesh_opts` growth path (the
  `mio_refine_ex` precedent), since the flat `mio_remesh` function had no
  room left to grow a second time. Plain `mio_remesh` is unchanged and now
  delegates internally.
- **Breaking (ABI):** `RemeshOptions` gained a member (`mMaxAnisotropy`) and
  `RemeshMetric` gained an enumerator (`Anisotropic`) — the member is what
  moves it, a Tier A layout change (an *existing* struct, not a wholly new
  header) per `doc/abi.md`'s own criterion — so `MESHIOPLUSPLUS_ABI_VERSION`
  moves 8 → 9 and the installed C++ variants' `SOVERSION` moves with it; the
  C, Fortran, Julia and R surfaces stay at `SOVERSION 0` (the flat ABI's own
  append-only-`reserved` contract, via the new `mio_remesh_ex` growth path,
  is unaffected). See [`doc/abi.md`](doc/abi.md).

See [`doc/remesh.md`](doc/remesh.md).

## v10.11.0 (2026-08-19)

**Roadmap §1 advanced further** — curvature gradation and boundaries/output
manifoldness are shipped, closing two of the four bullets left open after
v10.10.0; the anisotropic metric (now unblocked) and the volumetric
counterpart remain open.

- **`remesh` curvature gradation** (`RemeshOptions::mGradation` /
  `gradation`) — a per-item density weight `area * kappa^gamma` that
  concentrates clusters where the surface bends more sharply, via a new
  local osculating-paraboloid curvature estimator over each vertex's
  1-ring (reusing the clustering's own node-adjacency graph, no new
  neighbourhood machinery). `gradation = 0.0` (the default) disables
  gradation entirely — curvature is never computed — and reproduces plain
  area weighting byte-for-byte, so every pre-existing test and example is
  unaffected. Applies identically under both `metric="isotropic"` and
  `metric="quadric"`.
- **`remesh` boundaries and output manifoldness**
  (`RemeshOptions::mPreserveBoundary` / `preserve_boundary`, default
  `True`) — an open surface's boundary vertices are now detected, seeded
  before the interior, and pinned, with a second, optional `line` dual
  cell block emitted along every boundary edge whose endpoints land in
  different clusters. Non-manifold "bowtie" output vertices are now
  detected and reported separately from disconnected clusters
  (`RemeshResult::mNumNonManifoldVertices`, distinct from the existing
  `mNumIsolatedClusters`), both repaired by the same regrow-and-reminimise
  loop. A clean-room design, not a reproduction of ACVD's own
  boundary-fixing algorithm.
- Both extensions shipped across every `remesh` binding surface in one
  release: Python, C API, Fortran, Julia, R, WASM, both CLIs (`--gradation`,
  `--no-preserve-boundary`), the settings pipeline (`Gradation`,
  `PreserveBoundary` step params), and the MCP tool. See
  [`doc/remesh.md`](doc/remesh.md).

## v10.10.0 (2026-08-19)

**Roadmap §1 advanced** — isotropic CVD remeshing and the ACVDQ
feature-preserving metric are shipped; curvature gradation, the anisotropic
metric, boundary/manifoldness protection and the volumetric counterpart
remain open.

- **`remesh`** (`operations/remesh.hpp`, [`doc/remesh.md`](doc/remesh.md))
  — isotropic and feature-preserving surface remeshing by approximated
  centroidal Voronoi diagram (ACVD) clustering: replaces a surface mesh's
  own triangulation with a new, near-uniformly-sized, well-shaped one at a
  caller-chosen vertex count. The one resolution-changing operation that
  does not work on the input's own triangulation — `refine`/`decimate`/
  `subdivide`/`agglomerate`/`smooth` all inherit the input's element
  shapes, so none of them can *raise* a badly-shaped surface's quality at
  every target count; `remesh` partitions the surface into clusters and
  builds the dual, so output quality is a property of the clustering
  rather than the input. Two metrics: `"isotropic"` (default, area-weighted
  centroidal distance) and `"quadric"` (Garland-Heckbert quadric error,
  preserves sharp edges/corners, reusing this repo's own pre-existing
  quadric machinery rather than a second QEM implementation). The output
  has no correspondence to the input — new points, new connectivity — so
  `point_data`/`cell_data`/named regions are dropped and `field_data`
  carries through; compose with `interpolate`/`conservative_interpolate`
  to transfer a field onto the result. C++-core only, with no numpy
  fallback: the energy-minimisation loop's move-acceptance test is
  inherently sequential, so a second implementation could silently diverge
  into a different clustering rather than a last-ulp difference. Ships on
  every binding surface: Python, the C API, Fortran, Julia, R, WASM
  (`remesh`, reachable as a `convertSurfaceOps`/pipeline step too), both
  CLIs, and an MCP tool. **Attribution**: the isotropic clustering engine
  is derived from [pyacvd](https://github.com/pyvista/pyacvd) (MIT,
  (c) 2017-2024 The PyVista Developers) — an independent implementation of
  Valette & Chassery's published research, not of ACVD's own CeCILL-B
  source, which this project never reads or vendors; see `CITATION.cff`.

## v10.9.0 (2026-08-19)

**Roadmap §1 closed** — second derivatives / Hessian was the section's last
open item; "Field capability beyond derivatives" is now fully shipped.

- **`hessian`** (`operations/hessian.hpp`, [`doc/hessian.md`](doc/hessian.md))
  — the Hessian (second derivative) of a scalar `point_data` field, reachable
  as `meshioplusplus data hessian` in both CLIs: `gradient`'s companion one
  order further. A composition of two `gradient` calls, not a new numerical
  kernel — the field is differentiated once (point location), and that
  `(n, 3)` gradient is differentiated again with the default gradient
  operator, producing `(n, 9)`, the flattened row-major 3x3 Hessian. A field
  that is at most linear has an exactly zero Hessian everywhere — the one
  mesh-shape-independent guarantee, verified rather than assumed. For a
  genuinely quadratic field the composition is exact on a
  structured/symmetric mesh away from its own boundary (also measured) and
  a good, standard, but genuinely approximate curvature estimate on an
  irregular mesh — stated honestly rather than oversold. Input must have
  exactly one component; a vector field's Hessian is a separate quantity
  per component, computed by calling `hessian` once per component. No new
  marking subsystem is needed for curvature-driven adaptive refinement:
  `data_calc`'s `norm(...)` on the 9-component output is exactly its
  Frobenius norm, ready for `refine`'s `where` selector. Ships on every
  binding surface: Python (with a pure-Python composition fallback), the C
  API, Fortran, Julia, R, WASM (`hessian`, reachable as a `convertSurfaceOps`
  pipeline step too), both CLIs, and an MCP tool.

## v10.8.0 (2026-08-19)

**Roadmap §1 advanced** — field integration closes the section's smaller
remaining item; second-derivative/Hessian support remains open.

- **`data_integrate`** (`operations/data_integrate.hpp`,
  [`doc/field_integration.md`](doc/field_integration.md)) — a
  cell-measure-weighted total and mean of one or more `cell_data` arrays,
  reachable as `meshioplusplus data integrate` in both CLIs: `gradient`'s
  integration counterpart (`gradient` differentiates a field, this
  integrates one), for a density field's total mass, a heat-flux field's
  total power, or an occupied volume. Every sum is weighted by the cell's
  own length/area/volume; a cell whose measure is not computable, or a
  component whose value is non-finite, is excluded from that component's
  numerator **and** denominator — never given a fallback weight of 1.
  Reported for the whole mesh and independently for every named `Cell`
  region (regions are not a partition: a cell in two regions contributes
  fully to both). A `point_data`-only name raises by name, pointing at
  `point_data_to_cell_data`. Read-only, like `data_info`/`compute_stats` —
  the mesh is never modified. Ships on every binding surface: Python (with
  a full numpy fallback), the C API (opaque `mio_data_integrate` handle),
  Fortran (`data_integrate`/`data_integrate_region`), Julia, R, WASM
  (`dataIntegrate`), both CLIs, and an MCP tool.

## v10.7.0 (2026-08-18)

**Roadmap §1 advanced** — conservative interpolation was the section's
largest item; field integration and second-derivative/Hessian support
remain open.

- **`conservative_interpolate`** (`operations/conservative_interpolate.hpp`,
  [`doc/conservative_interpolate.md`](doc/conservative_interpolate.md)) —
  mass-preserving cross-mesh field transfer: over the region two meshes
  share, `sum(target value * target measure)` equals `sum(source value *
  source measure)`, a property `interpolate`'s `Barycentric` mode does not
  have. A separate sibling operation, not a third `InterpolateMethod`
  (`InterpolateOptions`/`mio_interpolate` are untouched), mirroring the
  `decimate`/`decimate_volume` split. Both meshes are simplexified first —
  the same call `interpolate`'s own barycentric mode already makes — which
  is what lets ragged and polyhedron blocks through for free, needing no
  general polygon/polyhedron clipper. Overlapping simplex pairs are found via
  a bucket-grid spatial hash (`detail/spatial_hash.hpp`, gaining a new
  `ForEachInBox` query) and measured exactly: a Sutherland-Hodgman convex
  polygon clip in 2D, and — since both operands are always tetrahedra — a
  bounded convex-polytope clip in 3D (source tet faces clipped against the
  target tet's four half-spaces, each cut capped by a fan-triangulated,
  angle-sorted polygon). `cell_data` is remapped directly; `point_data` by
  composition (`point_data_to_cell_data` → the same clip engine →
  `cell_data_to_point_data`), a documented layered approximation rather than
  exact nodal/FEM conservation. Unlike `interpolate`, an empty `arrays`
  covers every source `point_data` **and** `cell_data` array — one algorithm
  regardless of location — and there is deliberately no `extrapolate` flag,
  since a silent uncovered-cell fallback would break the conservation
  guarantee for exactly the cells most likely to need it. C++-core only, no
  numpy fallback, the same reasoning `subdivide`/`agglomerate`/
  `decimate_volume` already document (the 3D clip's discrete branches could
  disagree with a second implementation near a degenerate overlap). Shipped
  across every binding surface: pybind, the C API (`mio_conservative_interpolate`),
  Fortran, Julia, R, WASM (`conservativeInterpolate`), both CLIs
  (`conservative-interpolate`), and the MCP server; excluded from the
  settings pipeline like `Interpolate`/`UndoGreen` (a two-mesh op). No ABI
  change (`MESHIOPLUSPLUS_ABI_VERSION` stays at 7).

## v10.6.0 (2026-08-18)

**Roadmap §1 closed in full** — volume decimation was the section's last
open item.

- **`decimate_volume`** (`operations/decimate_volume.hpp`,
  [`doc/decimate_volume.md`](doc/decimate_volume.md)) — the volume-mesh
  sibling of surface `decimate`: reduces a tetrahedral mesh's cell count by
  greedy quadric-error-metric **tet**-edge collapse. A separate operation,
  not a mode on `decimate` (`DecimateOptions`/`mio_decimate` are untouched).
  Boundary vertices **participate** in decimation with a real quadric-error
  objective by default (`preserve_boundary=False`), unlike `decimate`'s own
  pinned-boundary default — every vertex's quadric is built from its
  incident boundary-triangle planes only, so a purely interior vertex's
  quadric is exactly zero and interior-only edges are scored by squared
  length instead, always ranking behind boundary-touching collapses in the
  greedy queue. Tet-only: any non-tetra 3D block raises pointing at
  `convert_cells(mode="simplexify")`. Validity is guarded by an exact
  vertex-link set-equality condition, a duplicate-tet check, and a
  tet-inversion guard, plus — for boundary-touching collapses — `decimate`'s
  own ring/shared-face link condition and normal-flip check, reused over the
  mesh's own outer skin via machinery hoisted into a new shared
  `detail/decimate_common.hpp` (`decimate.cpp` itself is otherwise
  byte-identical; its own test suite is the regression guard for the hoist).
  C++-core only, no numpy fallback, the same reasoning `subdivide`/
  `agglomerate` already document. Shipped across pybind, the C API
  (`mio_decimate_volume`, its own opaque result type), both CLIs
  (`decimate-volume`), the settings pipeline (`DecimateVolume` step, both
  engines), and the MCP server. No ABI change (`MESHIOPLUSPLUS_ABI_VERSION`
  stays at 7). Fortran/Julia/R/WASM bindings are a recorded follow-up.

## v10.5.0 (2026-08-17)

**Roadmap §1 closed further** — green-element undo is closed; volume
decimation remains as the section's one open item.

- **`undo_green`** (`operations/undo_green.hpp`, [`doc/undo_green.md`](doc/undo_green.md))
  — restores `refine`'s transitional (green) cells back to their original
  parent, the missing half of the standard selective-refinement rule
  ("restore a transitional cell to its parent and re-split from scratch
  before a new refinement pass"); `refine` refines the transitional
  children directly instead, so repeated selective passes over the same
  region degrade element quality without bound. A **two-mesh** operation,
  the repo's second after `interpolate`: `coarse` is the mesh a prior
  `refine(coarse, ..., record_hierarchy=True, record_levels=True)` call was
  run on, `fine` is that call's output. **Design: lookup and substitution,
  not reconstruction** — the originally-planned mechanism (inverting
  `refine`'s per-type subdivision tables against a green group's children,
  a genuinely hard graph-isomorphism-style match) turns out to be
  unnecessary: since `refine()`'s point map is always the identity, a green
  parent's exact connectivity and cell_data are already sitting,
  byte-for-byte, in `coarse` at the row `fine`'s `refine:parent_id` names.
  Classification is per sibling group (cells sharing one `parent_id`, since
  a parent's red/green status is uniform across every child): a singleton
  group is untouched; a group one level deeper than its coarse parent is
  red (kept unchanged); a group at the same level is green (substituted
  with one row read verbatim from `coarse`); a group more than one level
  deeper is refused by name (only a single-pass, `levels=1`, hierarchy is
  supported). The six reserved `refine:*` arrays are unconditionally
  dropped from the output. Points are never pruned or renumbered. Cell
  regions carry through the first genuinely non-injective
  `CellMapKind::Direct` use in the C++ core (several fine cells collapsing
  onto one output row), deduplicated by `Region`'s existing sort+unique;
  named Side regions do not survive. **Has a full numpy twin**, unlike
  `subdivide`/`agglomerate` — there is no winding repair or other discrete
  sign branch anywhere in the algorithm, just array bookkeeping and row
  copies. Reachable as `undo-green` on both CLIs and as an MCP tool;
  deliberately **not** reachable as a settings-pipeline/`convertSurfaceOps`
  step, the same exclusion `Merge`/`Interpolate`/`Split`/`Diff` already
  have, since a two-mesh op does not fit the single-mesh chain. Shipped
  across every binding surface: C++ core, pybind, the C API
  (`mio_undo_green`, a plain `mio_mesh*` with nullable counters like
  `mio_smooth`, not an opaque result handle), Fortran (module-level, like
  `mio_interpolate`), Julia, R, WASM. **No ABI change** — corrects the
  original plan, which assumed a new `RefineOptions` flag (Tier A, a bump
  to 8); the two-mesh substitution design needs no `RefineOptions` change
  at all, so `MESHIOPLUSPLUS_ABI_VERSION` stays at 7.
- Hoisted `RefineHierarchyState`/`refine_read_hierarchy` out of
  `refine.cpp`'s private scope into `detail/refine_hierarchy.{hpp,cpp}` so
  `undo_green` can share the identical Absent/Valid/Invalid hierarchy read
  against a coarse mesh rather than a second transcription; `refine.cpp`'s
  own behaviour is unchanged (verified by its existing test suite).

## v10.4.0 (2026-08-17)

**Roadmap §1 closed for polyhedra** — the polyhedral-*coarsening* half of the
"polyhedral refinement and coarsening" gap is closed, completing that gap in
full alongside v10.3.0's `subdivide`; volume decimation and green-element
undo remain open, and are the two entries left in §1.

- **`agglomerate`** (`operations/agglomerate.hpp`, [`doc/agglomerate.md`](doc/agglomerate.md))
  — polyhedral coarsening, the many-to-one counterpart to `subdivide`:
  merges groups of cells into single larger polyhedral cells via greedy
  seed-and-grow over the mesh's shared-face dual. Built on
  `detail::build_global_faces`'s owner/neighbour pairing (genuine
  face-adjacency in a compact volume-cell index space, deliberately **not**
  `detail::cell_adjacency.hpp`'s node-adjacency, which is both the wrong
  relation — node-sharing alone could fuse cells touching only at a
  pinch-point vertex — and indexed in a different space entirely).
  Seeds are chosen in ascending compact-cell order; each group repeatedly
  absorbs the unclaimed face-neighbour with the largest *accumulated*
  shared-face area (summed across every face the group's current members
  already share with that candidate) until `target_group_size` (default 8)
  is reached or the frontier empties — short groups at mesh boundaries and
  pockets are expected, not errors. The emit step walks each member's own
  face list and drops a face only when its other side is in the *same*
  group (an internal face, hit from both sides and dropped from both), so
  every merged cell's boundary is exactly the union of its members'
  external faces, wound and signed by transcribing the CGNS `NFACE_n`
  writer's own sign handling — **conserving volume exactly**, an identity of
  surviving faces rather than a divergence-theorem coincidence. **One
  polyhedron output block total, with genuinely mixed cell shapes inside**
  — the same no-node-count-grouping simplification `subdivide` already
  established, since `AddPolyhedronBlock` stores ragged CSR with no
  same-shape constraint. Non-volume blocks (2D, the 3D Lagrange family) pass
  through unchanged, and the merged block is emitted at the position the
  first volume block originally occupied, so block order is otherwise
  preserved. **No point compaction** — points are never pruned or
  renumbered, following `subdivide`'s own precedent rather than a
  `surface.cpp`-style used/remap pass; `clean(..., remove_orphans=True)` is
  the documented follow-up for a minimal point set. Regions carry through
  **`CellMapKind::Global`**, a single flat input-global-cell →
  output-global-cell map built once and passed to one batch
  `detail::remap_regions()` call — simpler than `merge`'s own usage of the
  same map kind, which drives a per-input-mesh loop purely for
  merge's region-name collision namespacing across N inputs, a concern
  agglomerate's single input mesh never has. A region collapsed entirely
  into one merged cell survives as a **named but empty** group (`Region::
  Canonicalize`'s dedup + the "the name is information" convention
  `region_remap.cpp` already documents elsewhere), not as a removed one. A
  non-manifold input (a face shared by three or more cells) is refused by
  name, mirroring `subdivide`'s throw-on-`Unorientable` precedent — the
  owner/neighbour face filter is only well-defined on a 2-manifold face.
  **C++-core only, with no numpy fallback at all** — the emit step depends
  transitively on `orient_rings`-repaired faces, the same discrete-winding-
  branch rationale `subdivide` and `_smooth.py`'s inversion guard already
  document. Reachable as an `Agglomerate` settings-pipeline/
  `convertSurfaceOps` step, and across every binding surface: C++ core,
  pybind, the C API (`mio_agglomerate` → `mio_agglomerate_result`,
  deliberately with **no `block` argument and no `_num_cell_maps`
  accessor** on its cell-map accessor, since an agglomerated cell's output
  index is a function of which group it joined rather than which input
  block it came from — the one opaque-result type in this codebase whose
  cell map is a single flat array rather than a per-block collection),
  Fortran, Julia, R, WASM, both CLIs (`agglomerate`), and the MCP
  `agglomerate` tool.

## v10.3.0 (2026-08-08)

**Roadmap §1 narrowed further** — the polyhedral-*refinement* half of the
"polyhedral refinement and coarsening" gap is closed; polyhedral coarsening
(agglomeration), volume decimation and green-element undo remain open.

- **`subdivide`** (`operations/subdivide.hpp`, [`doc/subdivide.md`](doc/subdivide.md))
  — polyhedral refinement: splits every eligible 3D cell into one polyhedral
  child per face, connected to a new interior point. `refine` and `decimate`
  both raise by name on a polyhedron, pointing at `convert_cells(mode=
  "simplexify")` — both are built on fixed same-type subdivision templates,
  and an arbitrary polyhedron has none. `subdivide` needs **no per-type
  table at all**: it goes through the same uniform face-ring machinery
  `gradient`/`compute_quality` already use (`detail::cell_rings`/
  `orient_rings`), which treats a tabulated type and an existing polyhedron
  block identically. Each child's interior apex is the plain corner average
  of the parent's own nodes (deliberately **not** the volume centroid),
  which is what makes the children's total volume conserve to a tight
  tolerance rather than merely by the divergence theorem's abstract
  equality. **Automatically conforming** — a shared face between two input
  cells is never touched, so unlike `refine` there is no closure, no 2:1
  balance, no hanging-node bookkeeping. **One polyhedron output block per
  input block, with genuinely mixed cell shapes inside** — `AddPolyhedronBlock`
  stores cells as ragged CSR with no same-shape constraint, so there is no
  need to group children by node count the way some *readers* (CGNS's
  `NFACE_n`, OpenFOAM, EnSight) do for their own format-compatibility
  reasons. Point and Cell regions survive (`CellMapKind::FirstChild`, the
  same shape `convert_cells` already uses for its own one-to-many splits,
  and with **no point map at all** — subdivide never prunes or renumbers a
  point); named Side regions do not, the same limitation
  `convert_cells(mode="simplexify")` already has. **C++-core only, with no
  numpy fallback** — the winding repair is a discrete branch a second
  implementation could disagree with near-degenerate cells, the same
  reasoning `_convert_cells.py`'s polyhedron branch and `_smooth.py`'s
  inversion guard already document. Reachable as a `Subdivide`
  settings-pipeline/`convertSurfaceOps` step, and across every binding
  surface: C++ core, pybind, the C API (`mio_subdivide`), Fortran, Julia, R,
  WASM, both CLIs (`subdivide`), and the MCP `subdivide` tool.
- **Fixed a real, independently-discovered bug**: `has_skinnable_cells`
  (`surface.cpp`, gating `convertSurfaceOps`/`convert_surface` and the
  `skin=true` default on the STL/PLY/SVG/TikZ writers) excluded *every*
  ragged block including polyhedra, even though `extract_surface`/
  `extract_skin` have supported polyhedron blocks since v9.16.0 — so a mesh
  with only a polyhedron block (`subdivide`'s own output being the first
  thing in the repo to construct one) silently took the unskinned fallback
  path everywhere `has_skinnable_cells` gates. Fixed by recognizing a
  polyhedron block the same way `extract_surface`'s own pre-scan already
  does; pinned by `Skin.HasSkinnableCellsRecognizesAPolyhedronOnlyMesh`.

## v10.2.0 (2026-08-07)

**Roadmap §1 narrowed further** — the error-estimator-helpers gap is closed;
volume decimation, polyhedral refinement/coarsening and green-element undo
remain open.

- **`estimate_error`** (`operations/error.hpp`, [`doc/error.md`](doc/error.md))
  — the Zienkiewicz-Zhu (ZZ) recovery-based error indicator of a `point_data`
  field, plus optional marking, the piece that closes the adaptive loop:
  `gradient` already differentiates a field and selective `refine`'s
  `--where` already consumes any scalar `cell_data` predicate; this produces
  one. Deliberately a **composition, not a new numerical kernel**:
  `gradient(location="cell")` (raw per-cell gradient) → the existing
  measure-weighted point↔cell averaging round trip (recovery) → the
  recovered-minus-raw difference is the local indicator
  `eta_K = sqrt(|measure_K| · sum((recovered − raw)²))`, reported globally as
  `sqrt(sum eta_K²)`. `marking="absolute"|"fraction"|"dorfler"` turns the
  indicator into a boolean `error:marked` array (`refine`'s own `--where`
  selector needs no change at all to consume it — the Dörfler bulk-fraction
  criterion is the usual AMR choice). Cells that cannot be evaluated read NaN
  in `error:zz` and `0` (never NaN) in `error:marked`, counted and excluded
  from the global error and every marking policy's count. Reachable as `data
  estimate-error` in both CLIs alongside `gradient`, for the same reason.
  Reaches every binding surface: C++ core, pybind, the C API
  (`mio_estimate_error`), Fortran, Julia, R, WASM (`estimateError()` plus an
  `EstimateError` `convertSurfaceOps`/settings-pipeline step), both CLIs, and
  the MCP `estimate_error` tool.
- **Fixed a real, independently-discovered bug**: `_data_average.py`'s
  `_FACES` table (the numpy twin's tabulated-cell fan-volume formula, used by
  `cell_data_to_point_data(weighted=True)`, and now also by
  `estimate_error`'s recovery step) had inconsistent face winding for all
  four linear 3D cell types (tetra/hexahedron/wedge/pyramid) — exactly one
  inward-wound face per type — which silently made the reported "volume" of
  a cell depend on how far it sat from the coordinate origin rather than on
  its shape, corrupting `Measure`-weighted recovery for any mesh not
  centred near the origin. Fixed by transcribing
  `detail/cell_faces.cpp`'s Newell-normal-gtested windings verbatim; pinned
  by `test_data_location.py::test_cell_measures_are_translation_invariant`.

## v10.1.0 (2026-08-07)

**Roadmap §1 narrowed** — `refine` gains a persistent parent/child hierarchy
across passes, closing the first half of that bullet (green-element undo
remains open).

- **`refine`'s persistent hierarchy** (`record_hierarchy=True` /
  `--record-hierarchy` / `RefineOptions::mRecordHierarchy`) — two new
  reserved `cell_data` arrays, `refine:cell_id` and `refine:parent_id`, on
  the same reserved-and-maintained pattern as `refine:level`: a cell no pass
  ever split keeps its own id and is its own parent; a split cell's children
  each get a fresh id and carry the parent's id. **The hierarchy is a link
  between two meshes, not a tree inside one** — a multigrid or green-undo
  caller keeps every pass's output mesh, and the fine mesh's `parent_id`
  resolves against the coarse mesh's `cell_id` (implicit, as its global
  block-major index, when the coarse mesh carries none of its own). An input
  already carrying `refine:cell_id` is updated whatever the flag says.
  Setting the flag also forces `refine:entity` to be attached even when the
  closure leaves no hanging node (the normal `redgreen`/`propagate` case),
  since it already records the coarse corners each new fine node is the mean
  of — the multigrid prolongation stencil, which was otherwise unreachable
  outside `balanced`. Reaches every binding surface: C++ core, pybind,
  the C API (`mio_refine_opts::record_hierarchy`, appended into the struct's
  reserved tail so `sizeof(mio_refine_opts)` is unchanged), Fortran, Julia,
  R, WASM (`recordHierarchy` on `refine()` and the `convertSurfaceOps`
  pipeline step), both CLIs' `refine --record-hierarchy`, the MCP `refine`
  tool, and the settings-pipeline `Refine` step's `RecordHierarchy` key.
  See [`doc/refine.md`](doc/refine.md#refinecell_id-and-refineparent_id).
- Closed the two ABI-guard gaps `doc/roadmap.md`'s note flagged:
  `RefineOptions` is now pinned in `tests/cpp/test_abi_layout.cpp`, and the
  Fortran `mio_refine_opts_t` mirror gained a runtime `sizeof` check
  (`check_refine_opts_layout`, the Julia `_check_abi_layout()` precedent) —
  previously the only guard on that struct was C's `static_assert` and
  Julia's own load-time check.
- **Breaking (ABI):** `RefineOptions` gained a member, so
  `MESHIOPLUSPLUS_ABI_VERSION` moves 6 → 7 and the installed C++ variants'
  `SOVERSION` moves with it; the C and Fortran libraries stay at
  `SOVERSION 0` (the flat ABI's own append-only-`reserved` contract is
  unaffected). See [`doc/abi.md`](doc/abi.md).

## v10.0.0 (2026-08-06)

Version-number bump only (9.30.0 → 10.0.0) — no API, behavior or ABI change;
`MESHIOPLUSPLUS_ABI_VERSION` stays 6. Pins mentioning the major version
(`find_package(meshioplusplus 9 …)` and similar) move to `10` in the docs.

## v9.30.0 (2026-08-06)

**Roadmap §1 closed** — the PhysicsNeMo section's last four follow-ups ship
and the section is removed from the roadmap. Pure Python + viewer work; no
C++/binding change, no wasm rebuild, no ABI move.

- **Autoregressive t→t+1 target pairing** (`meshioplusplus.physicsnemo`) —
  `target_offset=n` on `iter_samples`/`make_dataset`/`make_reader` pairs
  step k's inputs with step k+n's targets (each entry contributes
  `len(series) − n` samples; a too-short entry contributes none, with one
  warning naming it); `target_delta=True` makes `y` the increment
  `f_{k+n} − f_k` (the MeshGraphNet convention), normalized by the new
  `field_stats(delta=…)` emitting `{field}_diff_mean`/`{field}_diff_std`.
  A paired sample honestly costs two reads ("at most two meshes alive" —
  `TimeSeries` caches nothing), and a remeshed series fails the row-count
  check by name rather than mis-pairing. The `GraphSample` schema records
  `target_offset`/`target_delta`, so `GRAPH_SAMPLE_VERSION` bumps 1→2 —
  additive, but a stored v1 schema now compares unequal, which is the
  feature-drift guard doing its job.
- **The `physicsnemo.mesh.Mesh` bridge** — `to_physicsnemo(mesh,
  manifold_dim="auto", float32=True)` / `from_physicsnemo(pm)`, built
  despite v9.28.0's recorded deferral, with the risk stated rather than
  avoided: the target is PhysicsNeMo's newest surface, so the bridge is
  gated (named `pip install nvidia-physicsnemo` error), touches only the
  tensorclass constructor and public attributes (never the
  self-declared-unstable `.pmsh` format), and the docs pin
  `nvidia-physicsnemo>=2.1,<2.2`. The target holds exactly one simplex
  kind, so non-simplex cells tessellate through the existing
  `convert_cells` operations and everything it cannot hold (other-dim
  blocks, regions, non-numeric data) is dropped with a warning.
- **Dataset-manager UI: persisted directory handles** — the picked
  directory survives a reload (best-effort IndexedDB) and a **Reopen**
  button re-grants access with one click (the browser requires that user
  gesture); denial or a stale handle falls back to a fresh pick.
- **Dataset-manager UI: per-entry quality summaries** — preview summaries
  and the whole-manifest Scan now include the `quality:*` metrics (worst
  scaled Jacobian, inverted-cell badge), with quality rows excluded from
  the NaN/Inf bad-case counts — a quality metric's NaN means "not
  applicable to this cell type" by design, not a bad value.

## v9.29.0 (2026-08-06)

**The dataset-manager UI** — the last item of the roadmap's PhysicsNeMo
section: a second page of the browser viewer (`dataset.html`, deployed
beside it on Pages) that builds and curates the v9.28.0
[dataset manifests](doc/datasets.md) visually. Pure viewer-stack work — no
C++/binding change, no wasm rebuild (everything it needed was already
bound), and the wheel's embedded viewer bytes are untouched.

- **Workspace**: point it at a local case directory — File System Access
  picker in Chromium-family browsers (the manifest then saves **back in
  place**, so hand edits, CLI edits and UI edits interleave on the same
  file), `webkitdirectory` + download fallback elsewhere, stated in the UI.
- **Curation**: add cases (one file, an explicit list, or a suggested
  `Pattern` verified to match the selection exactly and nothing else),
  assign splits/tags/groups, edit notes/metadata; the page's TypeScript
  manifest model is a strict twin of `_dataset.py` — unknown keys refuse to
  load, and serialization is byte-parity with `DatasetManifest.save`
  (pinned against a Python-written fixture). Fraction-based `assign_splits`
  deliberately stays in Python/the CLI (a JS RNG could not reproduce the
  seeded shuffle — a "same seed, different assignment" trap).
- **Previews**: any entry renders through the viewer's own worker/MEMFS
  pipeline; a multi-step case is fanned out once and gets a step scrubber;
  a per-array summary table (min/max/mean, **NaN/Inf counts**) and a
  whole-manifest **Scan** badge bad cases before they corrupt a training
  split.

## v9.28.0 (2026-08-06)

**The PhysicsNeMo integration** — roadmap §1 shipped through the worked
example; only the browser dataset-manager UI remains. All pure Python; the
C++/WASM/C/Fortran core is untouched and the ABI version does not move.

- **Dataset manifests** (`DatasetManifest`/`DatasetEntry`, `doc/datasets.md`)
  — a hand-editable settings-family JSON cataloguing many solution outputs
  (each possibly a time series) for ML training: per-entry source plan
  (pattern / path / path list, times, ordering), train/valid/test splits
  (deterministic seeded `assign_splits`, `by_group` leakage guard), tags,
  group paths, notes and open metadata. Relative sources resolve against the
  manifest's own directory, so a campaign moves as one portable unit; the
  manifest is the single source of truth — hand edits and tool edits
  interleave against the same file. New nested CLI group
  `meshioplusplus dataset add/list/split/tag/annotate` (Python CLI only) and
  three ungated MCP tools `dataset_add`/`dataset_list`/`dataset_update`
  (42 → 45 tools; the sandbox covers every path a manifest names or
  resolves).
- **PhysicsNeMo adapter** (`meshioplusplus.physicsnemo`,
  `doc/physicsnemo.md`) — meshes to the tensors NVIDIA PhysicsNeMo's
  datapipes expect: `graph_sample` (the MeshGraphNet
  `pos`/`x`/`y`/`edge_index`/`edge_attr` set, edge features in the
  displacement+norm convention, columns carried from the `feature_matrix`
  contract), streaming `field_stats`/`edge_stats` in the
  `node_stats.json`/`edge_stats.json` convention, `make_dataset` (PyTorch
  Geometric `Dataset` — the training path) and `make_reader` (the Gen-2
  `Reader` extension point). The subpackage is deliberately not imported by
  `import meshioplusplus` (PhysicsNeMo's Python floor is 3.11), and there is
  deliberately **no `[physicsnemo]` extra** — the framework hard-depends on
  torch, the exact wheel the no-`[torch]`-extra precedent refuses to pin; a
  missing install raises naming `pip install nvidia-physicsnemo`. The doc
  page opens with the dated reconnaissance note (DGL removed / PyG only, the
  `Reader` ABC, the simplicial `physicsnemo.mesh.Mesh` and why the mesh
  bridge is deferred, the torch_scatter wheel-lag pin).
- **Worked end-to-end example** (`example/physicsnemo/`) — 200 manufactured
  Poisson cases, preprocessing as a settings-pipeline document, manifest
  curation through the real CLI, MeshGraphNet training and inference
  executed on a real GPU (100 epochs / 73.8 s, mean test RMSE 0.0040), with
  predictions written back as ordinary `point_data` and the renders
  committed. Public CI still installs neither torch nor PhysicsNeMo — the
  GPU-work precedent, stated in the docs.

## v9.27.0 (2026-08-05)

**The ML gap, closed.** The roadmap's machine-learning section ships in full —
graphs, feature matrices, dataset export and framework tensors, all pure
Python over existing machinery (the table payload, the smoothing layer's edge
topology, the sequence engine). The section is removed from the roadmap; the
C++/WASM/C/Fortran core is untouched and the ABI version does not move.

- **`edge_index(mesh, kind="node"|"cell", undirected=True)`** — the mesh as a
  graph in the `(2, E)` int64 layout PyTorch Geometric and DGL expect. The
  node graph reuses the smooth/`node_adjacency` edge topology (polygon blocks
  contribute closed rings, routed by type name — a uniform n-gon stores
  rectangularly); `kind="cell"` is the facet-sharing dual in global
  block-major numbering. Both directions by default (the PyG convention),
  lexsorted and deterministic.
- **`feature_matrix(mesh, location, fields=, coords=, regions=)`** — a
  `(N, F)` float64 matrix with a **stable, versioned column-order contract**
  (`FEATURE_SCHEMA_VERSION` 1): coordinates, then data arrays (multi-component
  expanded via the pandas suffix rule — one rule repo-wide), then
  `region:<name>` one-hots in `Region.key` order. The returned
  `FeatureMatrix.columns`/`schema` record the order so training and inference
  cannot silently disagree.
- **`write_dataset(source, path, format="parquet"|"zarr"|"hdf5")`** — a *set*
  of meshes (glob / list / multi-step file, via the sequence machinery,
  streaming one mesh at a time) as one `mesh_id`-keyed dataset:
  hive-partitioned Parquet with a JSON manifest, or chunked Zarr/HDF5 groups
  for out-of-memory datasets. The schema is strict — the first mesh defines
  it, a mismatch is a named error — and a failed run leaves no manifest. New
  `[zarr]` extra (zarr-python 2.x and 3.x); CLI verb `data export-dataset`;
  MCP tool `export_dataset`.
- **`to_torch(mesh, device=)` / `to_jax(mesh)`** — the DLPack payload adopted
  per framework: torch host adoption is genuinely zero-copy (measured),
  `device=` is one recorded bus transfer per array; JAX placement follows
  JAX's default device with a documented x64 fallback. Deliberately **no
  `[torch]`/`[jax]` extra** (the CuPy precedent); `has_torch()`/`has_jax()`/
  `has_zarr()` answer availability without raising.

## v9.26.0 (2026-08-05)

**DataFrames directly.** The Arrow table payload now feeds pandas and polars
without a pyarrow or Parquet detour, closing the first item of the roadmap's
machine-learning section. Pure Python over the existing `_to_table_payload`
seam; the C++/WASM/C/Fortran core is untouched and the ABI version does not
move.

- **`to_pandas(mesh, location=...)`** — one data location as a
  `pandas.DataFrame`. pandas columns are one-dimensional, so multi-component
  arrays become suffixed flat columns (`v_0`/`v_1`/`v_2`) — uniquely here; Arrow
  and polars keep the shape — with the grouping recorded in
  `df.attrs["meshioplusplus:components"]` alongside the full `meshioplusplus:*`
  metadata, so nothing is lost. Takes `zero_copy_only` like every other `to_*`;
  scalar buffer sharing is measured, never assumed, and the expansion (always a
  copy) raises under the flag, pointing at `to_arrow` for shared vector data.
- **`to_polars(mesh, location=...)`** — the polars counterpart. Multi-component
  arrays keep their true `(n, k)` shape as `pl.Array` columns. Polars always
  copies into its own Arrow-backed buffers, so there is deliberately no
  `zero_copy_only` — the frame is independent of the mesh by construction.
- **`has_pandas()` / `has_polars()`**, extras `[pandas]` / `[polars]`, and a
  five-package `[interop]`. There is no `from_pandas`/`from_polars`:
  `from_arrow` already returns plain arrays, and a table never carried the
  geometry.

## v9.25.0 (2026-08-05)

**Signed distance fields, completed.** v9.24.0 shipped the primitive; this
release adds the umbrella that generates its own grid, the adaptive octree, the
format that keeps a generated grid's geometry, and the crop mode that turns a
field back into a subset. The roadmap's signed-distance section is closed and
removed.

- **`compute_sdf(surface, ...)`** — the grid *and* the field in one call. It was
  declared with its option and result layouts final in v9.24.0 precisely so that
  adding the body would be a pure `.cpp` change, and it was.
- **`structure="octree"`** — adaptive: a coarse root lattice, refined only within
  `band_cells` of each cell's **own** diagonal, `max_depth` times, through
  [`refine`](doc/refine.md)'s `Balanced` closure. Against a uniform grid of the
  same *finest* resolution the zero level set is **the same contour** — identical
  facet count and point set — from far fewer cells. The output is 1-irregular
  (it has hanging nodes), exactly as that closure's is.
- **`.vti` (VTK XML ImageData)**, format 42 — a regular lattice whose geometry is
  the `Origin`/`Spacing`/`WholeExtent` attributes rather than a point array.
  **No format persists arbitrary `field_data`**, so a generated grid's `sdf:*`
  header does not survive a write anywhere else; `.vti` does not need it to,
  because those three attributes are the same information. Reading expands the
  extent into explicit `hexahedron` cells, so writing requires a lattice — a
  *partial* grid (`voxelize`'s `surface`/`inside` fills, or an octree, whose
  holes ImageData cannot express) is refused by name rather than silently filled
  in. It reuses VTU's `<DataArray>` codec and block framing verbatim, so
  `--codec zlib|lz4|zstd` works there too.
- **`crop(mesh, where=("array", "<", value))`** — keep the cells whose scalar
  `cell_data` value satisfies a comparison. Deliberately **general rather than
  crop-by-surface**: inside/outside then composes
  (`distance_to_surface(..., location="center")` then `where=("sdf:distance",
  "<", 0)`), and the same one mode also crops by `quality:*`, by a material id,
  or by anything `data calc` produces. It shares `refine`'s comparison
  vocabulary *and its evaluator*, so the two cannot drift on the boundary cases —
  in particular a **non-finite cell value never matches**, including under `!=`,
  where IEEE says `NaN != 1.0` is true.
  `mode="all"/"any"` is **absent, not ignored**, and passing it is an error on
  every surface: bbox and half-space test *points* and then need a rule for
  reducing a cell's several nodes to one verdict, whereas a `cell_data` predicate
  is already one value per cell.
- **Offsetting needed no code.** An offset surface is `compute_sdf` plus
  `isosurface` at a non-zero level; passing `[-r, 0, r]` gives the inner offset,
  the original and the outer offset in one mesh, already tagged with
  `iso:index`. The roadmap item closed by composition rather than by an
  operation.
- Every surface: pybind, the Python shim with a full numpy twin (parity pinned
  byte-for-byte), the C API (`mio_compute_sdf`, `mio_crop_predicate`,
  `mio_compute_sdf_opts`), Fortran, Julia, R, WASM (`computeSdf`,
  `cropPredicate`; the exhaustive name guard grows to 59), both CLIs (a new `sdf`
  verb, `crop --where`), MCP (42 tools), the settings pipeline (`ComputeSdf`, and
  `Crop`'s `Where`/`Compare`/`Value`) and the browser viewer.

Two notes on the tests, both of which took a second attempt. The octree's two
failure modes — a field interpolated from a coarser pass, and a cell selection
carried across a pass — each produce a *valid, plausible* mesh, so each has an
oracle verified to fire by sabotage. Neither fires against a **cube**: a cube's
SDF is piecewise linear near its faces, so interpolating it from a coarser grid
reproduces it exactly (18636 facets either way), and the cube's symmetry leaves
stale low cell indices near the surface anyway. The fixtures are a sphere, and
the band bound was *measured* (0.59 of the coarsest diagonal) before being set at
0.75 — a `2.0` bound had passed under the sabotage.

## v9.24.0 (2026-08-05)

**Regular grids and signed distance to a surface** — the one spatial-query
primitive meshio++ did not have. `extract_surface` gives you a skin, `slice` cuts
it and `isosurface` contours a field on it, but nothing answered *"how far is
this point from the surface, and which side is it on?"* — the thing collision
detection, offsetting, inside/outside queries and voxel-style ML preprocessing
all reduce to.

- **`grid(dims, origin, spacing)`** — a regular hexahedron lattice from nothing.
  The library's first mesh *generator*: every other operation transforms a mesh
  you already have.
- **`voxelize(mesh, ...)`** — a grid around a mesh, keeping the whole bounding
  box (`fill="all"`), only the cells a surface triangle passes through
  (`"surface"`, by exact separating-axis overlap, not a bounding-box test), or
  only the cells inside it (`"inside"`).
- **`sample_distance` / `distance_to_surface` / `surface_watertight_check`** —
  signed distance at arbitrary points or attached to a mesh as `sdf:distance`,
  plus what is wrong with a skin reported in numbers rather than a bare flag.
- **The output is an ordinary `Mesh`** — one `hexahedron` block over a shared
  corner lattice — and that is the whole design. Every writer, `view`/`screenshot`,
  `crop`, `split`, `--color-by` and `isosurface` already work on it, with no new
  code and no new file format. `custom` cells were rejected because
  `cell_type_num_nodes(Custom) == -1` makes a block invisible to `stats`,
  `quality`, `gradient` and `refine`, forfeiting exactly that property.
- Exposed on **every surface**: Python, C, Fortran, Julia, R, WASM, both CLIs
  (`voxelize`), five MCP tools, a `Voxelize` pipeline op and a browser-viewer
  chip. Documented in [`doc/voxelize.md`](doc/voxelize.md) and
  [`doc/sdf.md`](doc/sdf.md).

Three findings from building it, each worth not rediscovering:

- **A reentrant corner does NOT expose the classic sign bug.** The standard
  mistake is taking the sign from the nearest *triangle's* normal instead of the
  nearest *feature's* pseudonormal, and the obvious fixture for it is a concave
  edge — where, it turns out, both incident faces give the correct sign and the
  naive method passes. The failure needs two nearly *opposite* normals, i.e. a
  sharp spike. `tests/cpp/test_surface_distance.cpp` uses a sliver prism whose
  tip subtends ~1.7° and demonstrates the naive method getting it wrong, rather
  than asserting the right answer and hoping.
- **The accelerator is provably unobservable, and that paid for itself.** Every
  candidate comparison is totally ordered on `(squared distance, triangle id)`,
  so the bucket grid cannot change the answer. Sizing buckets by the mean
  triangle alone made a 64³ inside-fill of the 112k-triangle Stanford bunny take
  19.1 s; adding the domain extent to the rule cut it to 2.9 s with
  **byte-identical** output. It also means the numpy reference needs no
  accelerator at all — a brute-force scan is the same computation, which is why
  the two are bit-identical.
- **The numpy twin caught a real C++ bug.** Point compaction after a selective
  fill numbered survivors on first encounter rather than in ascending order,
  making the point ids depend on the hexahedron's node order — a traversal detail
  no caller should be able to observe. Fixed in C++, not papered over in numpy.

No ABI change (`MESHIOPLUSPLUS_ABI_VERSION` stays 6): the six new installed
headers are entirely additive. `operations/sdf.hpp` nevertheless ships its
octree fields populated-but-reserved, because `SdfOptions` and `VoxelOptions`
embed `SurfaceDistanceOptions` by value and growing it later would be a silent
Tier A break — the v9.12.0 pipeline lesson, applied in advance.

`compute_sdf` (generate a grid *and* fill it, including an adaptive octree) is
declared but throws by name; compose `voxelize` with `distance_to_surface` in the
meantime.

## v9.23.0 (2026-08-04)

**`refine(closure="balanced")` no longer tears the mesh across passes, and
`refine:hanging` now reports every constrained node.** Two defects, both reachable
from the public API today via `refine(mesh, cells=[...], levels=2,
closure="balanced")` — or any two chained balanced calls — and both confined to
that closure. The conforming closures (`redgreen`, `propagate`) and uniform
refinement are byte-identical to v9.22.0.

- **Fixed: a second balanced pass produced a torn mesh.** When the 2:1 balance
  rule draws in a coarse cell whose edge already carries a hanging node from an
  earlier pass, that cell's own refinement keyed the edge by its two endpoints,
  allocated a fresh midpoint, and left the two sides of the interface referencing
  distinct — but exactly coincident — nodes. On a 4x4x4 hexahedron block,
  `levels=2` produced 12 such positions and `levels=3` produced 96. The mesh was
  1-irregular *and* torn; only the former was ever documented.
- **Added: `refine:entity`**, the Int64 `(num_points, 4)` `point_data` array
  recording the entity each point was created on — `(-1, -1, a, b)` for the
  midpoint of edge `(a, b)`, the sorted `(p, q, r, s)` for a quad-face centre, and
  `(-1, -1, -1, -1)` for a point `refine` did not create. It is what lets a later
  pass reuse a node an earlier one already placed instead of allocating a
  coincident duplicate. Like `refine:level` the name is **reserved** and the array
  is *maintained* rather than replicated: attached whenever a pass leaves hanging
  nodes and kept thereafter, so the conforming closures never pay for it. A stale
  array — every entry must still reproduce its own point's coordinates, which
  `reorder`/`clean`/`crop`/`merge`/`transform`/`smooth` all invalidate — is warned
  about and ignored rather than trusted.
- **Fixed: `refine:hanging` under-reported.** [`doc/refine.md`](doc/refine.md)
  promises it marks *exactly* the constrained nodes, "neither a superset nor a
  subset"; a two-level balanced refinement of a 4x4x4 block reported 42 of 84, and
  a three-level one 90 of 402. The rule was evaluated over the **input** cells'
  entities, but a cell the balance rule draws in is split into children whose own
  sub-edges the input cell never had, so a node the neighbouring refinement places
  on one of them was constrained without any input entity naming it. It is now
  evaluated over the **emitted** cells. The array was also flowing through the
  generic `point_data` path, which interpolated an Int64 flag to `0.5` and
  truncated it; both reserved names are now excluded from that path and rebuilt.
- Both defects are pinned by counting oracles computed from the output's geometry
  and connectivity alone, and each was verified to fire by sabotage — disabling
  node reuse fails the tear test, and reverting the hanging rule to the input
  cells fails the report test and only that one.
- No ABI change (`MESHIOPLUSPLUS_ABI_VERSION` stays 6): `operations/refine.hpp`
  gains one `inline constexpr const char*` and nothing else.

## v9.22.0 (2026-08-04)

**cgnslib cross-compiled for WebAssembly**, so the browser build reaches parity
with a native cgnslib-enabled one. This is strictly an *addition*: meshio++
reads and writes CGNS itself over raw HDF5, polyhedral `NGON_n`/`NFACE_n`
sections included since v9.21.0. What the MLL buys in WASM is **ADF-backed
containers** — which are not HDF5 at all, and so are unreachable from the
hand-rolled path by construction — and the **CGNS 3.x** section layout.

- `build/build-wasm-deps.sh` gains a third pinned + SHA256-checked source build
  (cgnslib 4.5.2) beside HDF5 1.14.6 and netcdf-c 4.9.3, with
  `--without-cgnslib` to skip it. The prefix name grows accordingly, which
  invalidates CI's `actions/cache` key automatically (it hashes the script).
  `configure-wasm.sh` gains `--with-cgnslib` / `--without-cgnslib`.
- **`hasCgnslib()`** — a new JS binding (bound, forwarded and declared, plus a
  smoke step and an entry in the exhaustive forward guard). Without a probe, a
  build that silently dropped the dependency reads every file meshio++ produces
  itself identically, and the regression would surface only on a user's ADF
  file.
- **Size cost is small**: about **290 KB**, measured — the `.wasm` goes to
  ~6.3 MB sequential / ~6.7 MB threaded. cgnslib is a thin layer over the HDF5
  that was already linked in.

Two things about building it are worth not rediscovering, both recorded as
comments in the script:

- **Configure the TOP-LEVEL directory, not `src/`.** The root `CMakeLists.txt`
  runs the `CHECK_TYPE_SIZE` calls that `src/CMakeLists.txt` reads to pick
  `cgsize_t`'s underlying type; configuring the subdirectory leaves them empty
  and fails with `Can't find suitable int64_t`, which reads like a
  cross-compilation problem and is not.
- **`CGNS_ENABLE_64BIT` cannot be turned on here.** cgnslib forces it off
  whenever `CMAKE_SIZEOF_VOID_P <= 4`, which wasm32 is, so `cgsize_t` is
  32-bit regardless. That is fine — `cgns_mll.cpp` is written in terms of
  `cgsize_t` throughout rather than a fixed width — but the flag is
  deliberately *not* passed, since an option that is accepted and then silently
  overridden reads like a guarantee.

## v9.21.0 (2026-08-04)

**CGNS `NGON_n`/`NFACE_n` — polyhedral cells, hand-rolled in both directions.**
This closes the last bullet of the roadmap's polyhedral-meshes section: every
polyhedral writer now exists, and polyhedral CGNS works in the default build,
the PyPI wheels and WASM rather than only on a build with the optional
[cgnslib backend](doc/formats/cgns.md).

Hand-rolling the **write** side is what made hand-rolling the read side
obligatory too. Writing has no CGNS 3.x-vs-4.0 `ElementStartOffset` split to
absorb — the writer picks its layout — so the reason cgnslib was needed on read
does not apply; but a build able to write a file it cannot read back would be
worse than either extreme. cgnslib remains the answer for ADF containers and for
files written in the 3.x layout, which the hand-rolled reader refuses **by
name** rather than misreading.

- `NGON_n` is emitted ahead of the `NFACE_n` sections that point into it, and
  its faces are deduplicated across the **polyhedral blocks only**, via a new
  block-filtered overload of `detail::build_global_faces`. A mesh mixing
  hexahedra with polyhedra keeps ordinary `HEXA_8` sections for the former;
  putting their faces in the pool would leave `NGON_n` elements no `NFACE_n`
  cell ever references. A 2D jagged block needs no dedup at all — its cells
  *are* faces — so it becomes an `NGON_n` on its own.
- On read, an `NGON_n` becomes `polygon<N>` blocks **only when no `NFACE_n`
  references it**; otherwise it is the shared face pool, and emitting it as
  cells would duplicate every polyhedron's geometry while leaving the cell
  count looking plausible.

**Two bugs found by cgnscheck, neither of which any internal check could see.**
Both made cgnslib corrupt its own heap and abort rather than report anything, so
the tests assert the *return code* as much as the absence of `ERROR` lines:

- `NCell` omitted the polyhedral blocks, because `cell_type_from_name` does not
  know `polygon<N>`/`polyhedron<N>` and they reported dimension 0. cgnscheck
  sizes its cell arrays from `NCell` and then reads `NFACE_n`.
- The file declared `CGNSLibraryVersion` **3.1**. Below 4.0, cgnslib reads
  `NGON_n` with the 3.x inline-length layout — so it spliced our
  `ElementStartOffset` array into the connectivity. A face-based file now
  declares 4.0; a file without such a section reads identically under either
  number and keeps 3.1, so its bytes are unchanged.

- **The h5py twin deliberately does not implement this** and says so by name,
  pointing at the compiled core. The writer deduplicates faces and repairs each
  cell's winding, and the repair is a discrete branch on the sign of an enclosed
  volume — two implementations of such a branch can land on opposite sides for a
  near-degenerate cell and then diverge macroscopically, the reasoning that
  already keeps `smooth`'s inversion guard out of its numpy fallback. It is also
  unreachable in practice, since `_core` ships in every wheel. This keeps
  `test_structural_parity_with_cpp`, the byte-for-byte C++/Python oracle, exact.

- `cgns_write`'s pybind binding now passes `allow_ragged=true`; without it a
  polyhedral mesh never reached the C++ writer at all.

## v9.20.0 (2026-08-04)

The **OpenFOAM polyMesh writer** — the last of the roadmap's polyhedral writers,
and the last read-only format in the tree. Round-tripping an OpenFOAM case is the
headline outcome; a mesh from any other format converts into one too.

**`MESHIOPLUSPLUS_ABI_VERSION` 5 → 6**, because `OpenFoamInfo` gained a data
member (Tier A).

**Breaking (C++ ABI only):** `OpenFoamInfo` gained `mPatchTypes`, so its layout
changed. A C++ consumer that compiled against v9.19.0 headers and passes an
`OpenFoamInfo&` must be recompiled — which the bumped `SOVERSION`
(`libmeshioplusplus_core_*.so.6`) and `detail/abi_version_check.hpp` both enforce
at link time rather than leaving to chance. The C ABI, Python, WASM, Fortran,
Julia and R surfaces are all unaffected.

- **`detail/face_mesh.hpp`** — `build_global_faces` re-expresses a mesh's volume
  cells as a globally deduplicated face list with owner/neighbour pairing. Two
  formats are *defined* in those terms rather than in terms of cells: OpenFOAM's
  polyMesh literally is this structure on disk, and CGNS's `NGON_n`/`NFACE_n`
  pair is the same thing, so cell rows are stored in CGNS's own **signed 1-based**
  encoding and the CGNS writer (next release) copies a row with an element-id
  offset and no per-entry branch.

  Three decisions worth recording. Cells are numbered in a **compact volume-cell
  space**, not `block_bases`' block-major one: every mesh `read_openfoam`
  produces carries its boundary faces as 2D blocks, so in the global numbering
  "cell 8" is routinely a quad. Winding is **repaired unconditionally** —
  `cell_faces.hpp`'s rows are outward on the *reference* element, so an inverted
  hexahedron yields six inward normals and `checkMesh` would reject every one of
  its faces. And one `detail::FacetKey` serves both cell kinds, which is what
  makes a hexahedron and a polyhedron meeting on a face meet on *one* face.

- **`write_openfoam`** — the only meshio++ writer that creates a **directory**.
  All four ordering rules OpenFOAM requires (internal faces first, `owner <
  neighbour`, faces sorted by owner then neighbour, normals owner→neighbour) are
  enforced and then re-validated before anything is written, with one check per
  clause naming the clause it broke; the validator runs in release builds too,
  since release is where large cases get written and its failure means a corrupt
  mesh was about to reach a solver. `.foam` is now in the registry's extension
  table, so `write("case.foam", mesh)` infers the format — previously
  `resolve_format` threw on it, and the flat bindings could not read a case by
  extension either.

  A mesh with no patch tags — anything converted from another format — gets a
  single `defaultFaces` patch of type `patch`, which is what `blockMesh` itself
  produces. Patches are never synthesized from geometry.

- **Two reader fixes** came with it. `parse_boundary` now reads each patch's
  `type` (the **Python reader always has**; the C++ one was behind), and it
  matches braces by **depth** rather than taking the first `}` — so a patch
  carrying a nested sub-dictionary, as every `cyclicAMI` and `mappedWall` does,
  is no longer truncated into garbage patches. The second fix is not optional
  given the first: without it, `type` would be read from a truncated block.

- Patch types needing companion dictionary entries `OpenFoamInfo` cannot carry
  (`cyclic`, `processor`, `mapped*`, …) are **downgraded to `patch` with a
  warning**. Emitting them bare produces a case OpenFOAM refuses to *load*;
  a downgraded case loads and solves with boundary conditions the user can see
  and fix. An unknown type writes `patch`, never `wall` — `wall` selects wall
  functions, so guessing it would silently change a solve's physics.

- `tests/cpp/test_abi_layout.cpp` now pins `OpenFoamInfo`, `GmshInfo` and
  `MdpaInfo`. It pinned **no** format side-channel struct before, which is
  exactly what made growing one look free — `MedInfo` gained four members in
  v9.9.0 with the Tier A guard silent. (`MedInfo` and `ExodusInfo` stay unpinned:
  they exist only in an HDF5 / netCDF build, and a layout snapshot that says
  different things in different configurations is worse than none.)

- **No Python fallback writer**, deliberately. A twin would have to re-implement
  the per-cell winding repair, a discrete branch on the sign of an enclosed
  volume, and two implementations of such a branch can land on opposite sides
  for a near-degenerate cell — the reasoning that already keeps `smooth`'s
  inversion guard out of its numpy fallback. It would also be dead code:
  `openfoam_write` needs no optional dependency, so it is present in every wheel
  and every source build with Python bindings. The writer is registered only when
  the core provides it, so `meshioplusplus.write` never advertises a format that
  always raises.

## v9.19.0 (2026-08-04)

The polyhedral **writers** — the fifth item of the roadmap's polyhedral-meshes
section. Three formats that could read a polyhedron but never write one now
round-trip it.

- **EnSight `nfaced` / `nsided`**, ASCII and binary. The cheapest of the three:
  the wire format is a direct CSR dump (a per-cell count run, then a per-face
  size run, then the node ids) with no orientation contract and no global face
  table, so the writer is the reader's exact inverse and a round trip against
  it is a complete oracle. The blocker was structural, not conceptual —
  `ensight_writable_blocks` returns one `EnsightTypeEntry*` per block and the
  table had no entry for a type with no fixed node count.
- **MED `POE`** (`MED_POLYHEDRON`), which needs **three** 1-based arrays where a
  polygon needs two: `NOD`, `INN` (face → `NOD`) and `IND` (cell → face). MED
  holds one section per type inside a `MAI` group, so every `polyhedron<N>`
  block is canonicalised to a single `POE` on write and regrouped by unique node
  count on read; grouping on the exact meshio type string would try to create
  `POE` once per node count and fail at group creation. This also corrects
  `doc/formats/med.md`, which claimed polyhedra were "Python-only for MED" —
  **neither** path had a `POE` entry.
- **VTU `VTK_POLYHEDRON` (type 42)**, both directions. The reader previously
  threw on merely *seeing* the `faces` array name, before checking whether any
  cell was type 42.
  - **Mixing polyhedra with other cell types works**, where the Python
    reference historically refused it. That was an implementation limit, not a
    format one: `faceoffsets` carries **-1** for a non-polyhedral cell, which
    *is* the mixing mechanism — and an OpenFOAM mesh always mixes hexahedra,
    polyhedra and boundary faces, so inheriting the restriction would have
    defeated the point.
  - `_vtu.py` was relaxed **in lockstep**, on both the read and write sides.
    That is not optional: Windows CI builds every native path off and runs the
    reference implementation, so a C++-only capability would be green on Linux
    and red on Windows. The shim also stops diverting polyhedral meshes to
    Python.
  - The reader reaches type 42 through a **new overload** of
    `detail::reconstruct_cells`; `detail/vtk_cells.hpp` is an installed header,
    so changing the existing signature would have renamed a mangled symbol.

All three bucket polyhedra into `polyhedron<N>` by unique node count on read —
the convention the OpenFOAM, EnSight, MED and CGNS readers already shared.

Also: a `cgnslib` **vcpkg feature** (depending on upstream's `cgns` port), which
is the usual route on Windows, alongside the Conan `with_cgnslib` option added
in v9.18.0. `doc/formats/cgns.md` now lists how to get cgnslib per platform.

## v9.18.0 (2026-08-04)

An optional backend built on the **official CGNS library** (cgnslib, the CGNS
Mid-Level Library), closing the roadmap's `NGON_n`/`NFACE_n` item on the read
side. `-DMESHIOPLUSPLUS_WITH_CGNSLIB=ON`, **OFF by default and bring-your-own**
via `CGNS_ROOT` — never vendored, never downloaded, exactly the KaHIP policy.

It is **additive**. The hand-rolled ADF-over-HDF5 reader and writer are
unchanged and remain the default, so a build without the flag behaves exactly as
before: no format disappears, the WASM artifact is untouched, and the C++/Python
byte-parity oracle still holds.

What it buys, both being things the raw-HDF5 path cannot have at all:

- **ADF-container files.** `.cgns` has two on-disk containers. `read_cgns`
  speaks HDF5 directly, so an ADF file is not merely unimplemented there — it is
  unreachable by construction, and much of the real-world corpus (including the
  CGNS project's own example meshes, which the roadmap wanted as polyhedral
  fixtures) is ADF.
- **`NGON_n` / `NFACE_n` polyhedral sections.** `NGON_n` lists faces; `NFACE_n`
  lists each cell as **signed** face ids, the sign meaning "traverse this face
  reversed" — CGNS's way of orienting a shared face outward from both cells
  using it. `cg_poly_elements_read` also absorbs the CGNS 3.x-vs-4.0
  `ElementStartOffset` split, the most error-prone part of the encoding and the
  single strongest reason to use the MLL rather than hand-roll it. `NGON_n`
  **without** an `NFACE_n` is a face mesh and maps to `polygon` blocks.

Routing: **read** goes through the MLL whenever it is built (the input is not
ours, and the MLL is strictly more capable), with one narrow fallback — the
pre-v9.8.0 legacy layout, which has no ADF node attributes and which the MLL
rejects. That is a specific fallback, not a blanket `catch (...)`: a genuine MLL
error still surfaces. **Write** is untouched and stays on the hand-rolled path.
The consequence is a free cross-engine check: on a cgnslib build the existing
CGNS suite writes with one engine and reads with the other, and all of it passes.

The MLL reader is a **superset**, not a divergence: it reads `FlowSolution_t`
(with the same `_0.._k-1` component rejoin convention) and refuses `MIXED` and
the unverified cubic/quartic Lagrange family by name, matching the hand-rolled
reader rather than silently skipping them.

`meshioplusplus.cgns.read` no longer falls back to the h5py reference when the
file is not HDF5 — the fallback cannot answer that question and would report a
confusing signature error instead of the real one (the `xdmf` `time_step`
precedent). `_core.__has_cgnslib__` and `cgns_has_cgnslib()` report the build.

## v9.17.0 (2026-08-04)

`convert_cells(simplexify)` decomposes polyhedra into tetrahedra — the escape
hatch that makes the remaining resolution-changing operations work on them, and
the fourth item of the roadmap's polyhedral-meshes section.

- **One tetrahedron per (face, edge-of-that-face)**, from the face's corner
  average to the cell's — the *same* fan `detail/polyhedron.hpp` measures. So
  the children's total volume equals the parent's **exactly**, which is a hard
  oracle rather than a tolerance, and every child comes out positively oriented.
- That adds `1 + numFaces` points per cell rather than a single centroid.
  A one-point fan about each face's first node is cheaper and wrong twice over:
  it is diagonal-dependent for non-planar faces (so two cells sharing one would
  disagree), and it inverts on any cell whose faces are not star-shaped about
  whichever node happens to be listed first.
- **`slice`, `isosurface` and `interpolate --barycentric` become
  polyhedron-capable with no further code**, since all three go through
  `marching_prepare` → simplexify. Verified by test rather than assumed.
- Point-adding follows the `elevate` precedent: new coordinates and every
  `point_data` array's new rows are the mean of the same recorded source-node
  list, so they cannot drift apart, and block structure stays 1:1 so the
  `cell_data` and `mCellMaps` contracts are untouched.
- A polyhedron whose faces are **not a closed orientable surface** throws,
  naming the cell — it bounds no volume to decompose. `refine` and `decimate`
  still raise on ragged blocks, but now point at `convert_cells(simplexify)` by
  name.

The pure-numpy `convert_cells` fallback remains rectangular-only and raises
`NotImplementedError` for ragged blocks, as before; this is a C++-core
capability.

## v9.16.0 (2026-08-04)

A geometric kernel for cells bounded by arbitrary polygonal faces
(`detail/polyhedron.hpp`), and the operations that were silently skipping
polyhedra wired onto it — the second item of the roadmap's polyhedral-meshes
section. See [`doc/polyhedra.md`](doc/polyhedra.md).

- **Breaking:** the signed volume of a cell with **non-planar faces** changes.
  `stats`, `cell_measure` (and so `data_average`'s measure weighting) and
  `clean`'s degeneracy test each carried their own copy of a divergence-theorem
  volume that fanned every face about its *first node*; all three now share the
  kernel, which fans about the face's **corner average**. Planar faces are
  unaffected. The apex is now a function of the face alone rather than of the
  cell's storage, so two cells meeting on a warped quad triangulate it
  identically instead of possibly choosing different diagonals — pinned by
  `PolyhedronKernel.CellsSharingAWarpedFaceAgreeOnIt`, which fails against the
  old decomposition. `_stats.py`'s numpy twin moved in lockstep, since a
  Python-fallback build (Windows CI) would otherwise report different volumes
  from a native one; `test_cpp_matches_python` gained a warped fixture, because
  a cube cannot tell the two apart (a parallelogram's corner average *is* its
  area centroid).
- **Winding is repaired, not required.** Real meshes arrive inconsistently
  wound — including this repository's own long-standing `polyhedron5` fixture,
  whose cells traverse a shared edge in the same direction from two faces — so
  `orient_rings` fixes it per cell (BFS over the faces' shared-edge dual, then a
  global flip if the volume came out negative) and reports `Unorientable` only
  when the face set is genuinely not a closed orientable surface.
- **`gradient` is polyhedral**, which is what the roadmap meant by Green-Gauss
  being *naturally* polyhedral: it now integrates over the cell's own face
  rings, so a `polyhedron` block goes through exactly the same code as a
  `hexahedron` and recovers a linear field's gradient just as exactly.
- **`extract_surface` / `extract_skin` handle polyhedra**, including faces of
  any arity (a pentagon lands in a ragged `polygon` block rather than being
  dropped). The facet key is now `detail::FacetKey` — variable arity with a
  4-entry inline fast path — for **one** shared key type, which is what lets a
  hexahedron and a polyhedron that meet on a face cancel each other out.
- **`compute_quality` reports a reduced set** for polyhedra — volume, inverted,
  degenerate — and leaves every metric defined against a reference element the
  cell does not have as `NaN`, rather than inventing a plausible number.
  `QualityReport`'s layout is unchanged.
- **`smooth`'s inversion guard covers polyhedra.** It previously skipped them
  when building its measurable-cell table while nothing pinned their nodes
  either, so their nodes were moved with no guard at all — silently.
- `stats` also now counts a ragged `polygon` block's area, which it previously
  ignored entirely.

- **`smooth`'s boundary marking** covers them too, so `fix_boundary` now
  actually pins a polyhedral mesh's outer nodes; previously *nothing* pinned
  them and a purely polyhedral mesh quietly shrank under smoothing. An n-gon
  boundary face is emitted as several feature records sharing one normal, so
  every corner takes part in feature detection rather than only the first four.
- **`partition`'s KaHIP dual graph** connects polyhedra sharing a face.
  Previously they were isolated vertices, so KaHIP saw a graph with no edges and
  returned a balanced but cut-blind partition — silently. (The default SFC
  method was never affected; it works off cell centroids.)
- **`clean` drops degenerate and duplicate polyhedra**, which it previously kept
  unconditionally while dropping the equivalent hexahedron. The duplicate key is
  the sorted set of sorted faces, so the same solid still collides with itself
  when its faces are relisted in another order or with reversed windings, and
  two different solids on the same node set do not collide.

Both `smooth`'s boundary marking and `partition`'s dual graph moved to
`detail::FacetKey` for this, joining `extract_surface`: **one** facet key type
across cell shapes is what lets a hexahedron and a polyhedron that meet on a
face see each other at all.

## v9.15.0 (2026-08-04)

Polyhedral connectivity across the flat C ABI — the first item of the roadmap's
polyhedral-meshes section, and the one everything else in it depended on.
Ragged blocks (jagged polygons, and polyhedra given as a list of faces per
cell) could previously be *reported* by the C API and then neither built nor
read: `mio_mesh_cell_block_conn` returned `MIO_ERR_UNSUPPORTED` and there was
no setter at all. Both halves are closed, in C, Fortran, Julia and R. New
concept page: [`doc/polyhedra.md`](doc/polyhedra.md).

- **Reading** goes through a new opaque `mio_poly_conn` **snapshot**
  (`_create` / `_get_shape` / `_nodes` / `_face_offsets` / `_cell_offsets` /
  `_free`) carrying the flat CSR triple the JS boundary already used. It is an
  owning copy rather than the ABI's usual zero-copy borrow — and so, unusually,
  stays valid across mutating calls — because the `MESHIO` mesh backend stores
  ragged blocks as nested vectors with no offsets array to point at and the
  `KRATOS` backend rebuilds its blocks lazily.
- **Building** is `mio_mesh_add_polygon_block` / `mio_mesh_add_polyhedron_block`,
  int64-only (a CSR pair is two arrays that must agree; one `dtype` covering
  both invites the wrong pairing). Malformed offsets — not starting at 0, not
  monotonic, not ending at the total — are rejected by name rather than read
  past.
- **`mio_mesh_cell_block_info_ex`** reports `is_polyhedron`, `num_faces` and
  `num_nodes` over a reserved-tail struct, the growable successor to the
  five-argument `mio_mesh_cell_block_info`, which is unchanged.
- Each language gets its **natural** shape over the one flat ABI, the same
  policy that already gives all of them column-major arrays and 1-based
  indices: Fortran the CSR triple (`m%get_polygon_block` /
  `m%get_polyhedron_block`, offsets shifted to 1-based along with the node ids
  because they index a Fortran array), Julia nested vectors (`polygon_block` /
  `polyhedron_block`), R nested lists (`mio_polygon_block()` /
  `mio_polyhedron_block()`), each with the matching setter.
- **WASM** needed no new binding — ragged blocks already rode the mesh object —
  but its `val_to_mesh` polyhedron branch now validates `faceOffsets` the way
  the polygon branch always validated `rowOffsets`; a malformed one used to
  build an out-of-range iterator pair. Two stale doc comments claiming ragged
  blocks throw were also corrected.
- The C API gtests that used to assert ragged blocks were *inaccessible* now
  assert they round-trip, and no longer need HDF5 to build a ragged mesh — a
  real coverage gain on builds without it.

## v9.14.0 (2026-08-03)

Original MDPA ids preserved on write — the write half of the roadmap's MDPA
section, closing it in full (v9.13.0 shipped the read half). Both readers now
attach `point_data`/`cell_data["mdpa:id"]` when a deck's node/element/condition
ids were not already the trivial `1..n` a fresh write would produce, and both
writers honour it when present, writing the original ids back instead of
unconditionally renumbering. Notable points:

- **Only when it matters**: a sequential (or id-less) deck picks up no
  `mdpa:id` at all, so a read → write round trip of it stays on the exact old
  code path — byte-identical output, not merely equivalent.
- **Every reference resolves through the same written id** — connectivity,
  `NodalData`/`ElementalData`/`ConditionalData` row keys, and `SubModelPart`
  node/element/condition lists — never a bare `row + 1`/original id, so the
  file stays internally consistent whichever numbering is actually used.
- A **duplicate value** in either array is a `WriteError` (checked separately
  for elements vs conditions, which have independent Kratos id namespaces),
  since writing one would silently produce an ambiguous file. A missing,
  wrong-length, or wrong-block-count array is treated as unrelated/stale
  metadata and falls back to the old renumbering rather than partially
  trusting it.
- **Fixed a real, pre-existing correctness bug found along the way**: the
  Python reference writer had always re-emitted `SubModelPartElements`/
  `Conditions` membership using the raw original ids captured at read time,
  which was already wrong whenever a plain read→write renumbered entities (the
  universal case before this release) or reclassified one across the
  Elements/Conditions boundary — silently producing a `SubModelPart` reference
  to an id absent from the very file being written. Fixed by resolving each
  raw id through the read-time id maps and the write-time id assignment,
  rather than re-emitting it verbatim. `Begin Mesh`'s `MeshElements`/
  `Conditions` (not `SubModelPart`) deliberately keep the old verbatim
  behaviour, pinned by an existing test; `MeshNodes` gets the preserved-id
  treatment like every other node reference.
- New `meshioplusplus::kMdpaIdName` constant (`"mdpa:id"`), an additive header
  change reviewed in `doc/abi_reviews.md`; `MESHIOPLUSPLUS_ABI_VERSION` stays
  **5**.

See [`doc/formats/mdpa.md`](doc/formats/mdpa.md#original-ids-preserved-on-write-v9-14-0).

## v9.13.0 (2026-08-03)

Arbitrary MDPA node ids. The C++ reader required node ids to be exactly `1..n`
in file order and threw `"MDPA: non-sequential node ids are not supported by
the C++ reader"` otherwise — one of the few constructs that threw even under
`ReadOptions::mLenient`. Real Kratos decks routinely have gaps (SubModelPart
extraction, entity removal and deck merging all leave them, which is why
`ModelPart` keys entities in a hash map), so a genuine production `.mdpa` was
unreadable from WASM, the C API, Fortran, Julia, R and the native CLI — none of
which has a Python fallback. The pure-Python reference was no better: it
discarded the id column and reconstructed row = id − 1 from position, silently
misassigning coordinates and data on a gapped deck through
`meshioplusplus.read()`, which for mdpa is *always* the Python path.

Both readers now resolve connectivity, `NodalData`, `SubModelPartNodes` and
`MeshNodes` through a file-id → row map, built lazily on the first id that is
not `row + 1` (the `abaqus.cpp` `mPointIds` / `unv.cpp` `label_to_index`
pattern). Notable points:

- **Not gated on `mLenient`**: accepting arbitrary ids is strictly more
  *correct*, not more lenient, so no read that succeeded before changes its
  result. A `1..n` (or id-less) deck never leaves the arithmetic path.
- Points come back in **file order**, never sorted by id. Ids themselves are
  not carried onto the mesh.
- A **bare `x y z` row takes its position as its id**, so the id-less form and
  mixed blocks are both well defined. (The reference reader's `np.loadtxt` is
  rectangular and still rejects a *mixed* block; the C++ reader accepts one.)
- A **duplicate node id** now throws by name, always — two coordinate rows
  claiming one id is unrepresentable, not merely incomplete.
- Connectivity naming an undefined node still throws, but the message was
  reworded to name the **file id** and report the node count as context:
  `"connectivity refers to node id N, which the file's Nodes block does not
  define (M nodes read)"`. The old wording ("but the file has M nodes") is
  false for a gapped file, where id 500 can be perfectly valid in a 4-node deck.
- `SubModelPartNodes` naming an unknown id is now dropped with a `log::warn`
  rather than silently, matching what the entity lists already did.
- The Python reference warns when an `Elements`/`Conditions`/`Geometries` block
  precedes the `Nodes` block, since it resolves eagerly and would fall back to
  "row = id − 1" there; the C++ reader defers resolution and is order-independent.

Ids are still **renumbered to `1..n` on write** by both writers — the remaining,
narrowed roadmap §0. `MESHIOPLUSPLUS_ABI_VERSION` stays **5**: `formats/mdpa.hpp`
changed only in its doc comment, with every declaration byte-identical.

## v9.12.0 (2026-08-03)

Multi-file and transient datasets. Every entry point was single-mesh,
single-file: XDMF had a time-series writer and v9.11.0's pipeline ran one input
through an operation chain to one output, but there was no way to treat a *set*
of files as one logical dataset -- which is how transient solver output actually
arrives (`out_0000.vtu … out_0500.vtu`), and how most of the 41 formats have to
express time, since only a minority carry several steps natively. This release
adds fan-out (one multi-step file -> N files), fan-in (N files -> one multi-step
file), globbed/list input and per-step pipeline execution, across the CLI, the
settings document, Python, C, Fortran, Julia, R and WASM. `MESHIOPLUSPLUS_ABI_VERSION`
stays 5 -- the one new installed header (`operations/sequence.hpp`) is purely
additive (Tier C, reviewed in `doc/abi_reviews.md`).

### Sequences

- **`operations/sequence.{hpp,cpp}`** (docs [`doc/sequences.md`](doc/sequences.md)):
  a **driver, not a new operation**. It reads and writes through the existing
  registry and runs operation chains through the existing typed pipeline layer,
  so `run_pipeline_steps` remains the single owner of the step dispatch and a
  settings document still cannot drift from the browser viewer's
  `convertSurfaceOps`.
- **Ordering is natural-numeric and documented**: `out_9.vtu` sorts before
  `out_10.vtu`. Digit runs compare numerically on the digits themselves (never
  through `stoull`, so a 40-digit name cannot overflow), non-digit runs compare
  as `unsigned char`, and a final tie-break on the unstripped strings is what
  makes the comparator a **strict weak ordering** -- without it `out_1` and
  `out_01` are mutually "not less" yet not equivalent, and `std::sort` on a
  directory mixing padded and unpadded names would be undefined behaviour. A
  brute-force gtest checks all four axioms.
- **Time values** follow a documented precedence -- explicit list, the file
  itself (a series step, or `field_data["meshio:time"]`, generalizing the
  length-1 `exodus:time` the Exodus reader/writer already round-trips), the
  last digit run of the filename, then the integer index -- and every entry
  **reports which source it used**, because "the file said 0.25" and "nothing
  said anything, so this is position 3" are different facts.
- **The streaming invariant**: at most one `Mesh` is alive at any point in a
  fan-in, a fan-out or a per-step run, and Python's `read_sequence` is a
  generator. This is a contract, not an optimization -- the feature exists so a
  500-step dataset is traversable on a laptop. It is pinned rather than
  asserted in prose: a gtest measures the peak through the `BufferAllocator`
  hook and requires it to be **O(1) in the step count** (20 files vs 40), and
  the Python suite uses weak references so a regression names the retainer.
- **A multi-step input aimed at a single-step output fails by name**, naming
  the format and pointing at `{step}` and `--time-step`, rather than silently
  writing step 0 -- which is what `convert series.xdmf out.vtu` did before.
- **Both CLIs**: `convert 'out_*.vtu' out.xdmf` (fan-in, quote the glob),
  `convert in.xdmf 'out_{step}.vtu'` (fan-out), and repeated `--input` for an
  argv your shell already expanded. Glob matching lives in the **core**, not
  only in Python, so both CLIs accept exactly the same pattern language --
  deliberately just `*` and `?`, narrower than `glob(3)` and `fnmatch`, so the
  two matchers cannot accept different things.
- **The settings document** gains `Mode`, `Input.Pattern`, `Input.Paths`,
  `Input.Times`, `Input.TimeFrom`, `Parallel` and `Workers`; the operation chain
  runs per step. `Mode` **asserts** the inferred shape rather than selecting it,
  and a mismatch names both. `Parallel` is a Python-driver process pool, legal
  for element-wise runs only: with a fan-in it is an error, not a silent
  serialization, because buffering steps for an ordered writer would break the
  streaming guarantee. A document using none of the new keys takes a physically
  unchanged path.
- **Python**: `read_sequence` (a lazy `(time, Mesh)` generator), `write_sequence`,
  `sequence_entries` and `run_sequence_pipeline`; `run_pipeline` routes a
  sequence document here automatically. **C**: `mio_sequence_*`, whose
  `_read` hands back an **owned** mesh rather than a borrow — the C ABI's way of
  expressing the no-caching rule. **Fortran/Julia/R** wrap that handle in each
  language's idiom. **WASM** gets `sequenceEntries`/`sequenceToTimeseries`/
  `timeseriesToSequence` over MEMFS paths — the same virtual filesystem
  `convert` and `runPipeline` already work on — with `runPipeline` routing a
  transient document itself; `Parallel` is accepted and ignored with a warning
  there, since a wasm module has no processes to pool.
- **MCP**: a `sequence` tool, plus `_resolve_pattern`, which containment-checks
  a pattern's *directory* component before expanding it -- `_resolve`'s
  `os.path.isfile` rejects a glob outright, and the directory is the obvious
  sandbox escape.
- **`TimeSeries` (Python)**: an ordered `(time, Mesh)` sequence held as one
  value with random access -- the counterpart to the C/Fortran/Julia/R sequence
  handles, which already give indexable per-step traversal (`mio_sequence_read
  (seq, i)`, `read_step(seq, i)`, ...) that Python's `read_sequence` generator
  cannot (exhausted after one pass, no `len()`, no indexing). Still holds only
  the entry *plan*, never a mesh: `series[i]` performs exactly one independent
  read, so a 500-entry `TimeSeries` costs no more memory than its plan. This
  closes the one item `doc/roadmap.md` §1 had left open; nothing remains open
  in that theme.
- **`example/julia/03_mesh_operations.ipynb` and `example/r/03_mesh_operations.ipynb`**
  gain the same "Transient sequences" section as the Python/C++ notebooks:
  natural-numeric ordering (a lexicographic sort would put `seq_10` third, not
  last), fan-in/fan-out, and a per-step `run_sequence_file`/
  `mio_sequence_pipeline_run_file` chain, with the mean-height-per-step result
  rendered as a bar chart (these bindings' documented no-PyVista convention).
  Writing them surfaced a real flat-ABI gap: `mio_sequence_to_timeseries` had
  no way to select XDMF's `"XML"` data format, which these HDF5-off notebook
  environments specifically need. Closed with **`mio_sequence_to_timeseries_ex`**
  (a `mio_write_opts*`, mirroring `mio_write_ex`; only `encoding` has anywhere
  to go here -- `Codec`/`FloatFormat` fail by name, since the transient writer
  drives `XdmfTimeSeriesWriter` directly and bypasses the registry) plus an
  `ascii`/`ascii = TRUE` keyword on the Julia and R wrappers.

### Fixed

- **`meshioplusplus.xdmf.read` now accepts `time_step`.** `read_xdmf` has
  honoured `ReadOptions::mTimeStep` since v9.0.0, but the pybind binding never
  passed it through, so a temporal collection was unreachable from Python
  except through `xdmf.TimeSeriesReader` -- on *the* multi-step format. A
  non-default step deliberately does **not** fall back to the pure-Python
  reader, which always returns step 0: that would be a wrong answer, not a
  slower one.
- `tests/cpp/test_abi_layout.cpp` now pins `Pipeline`, `PipelineInput`,
  `PipelineOutput` and `PipelineStep`, which shipped unpinned in v9.11.0.
- **A settings document naming a multi-step input no longer writes step 0.**
  A document that used no sequence key and named a plain output took the
  single-file path, so `{"Input": {"Path": "series.xdmf"}, "Output": {"Path":
  "out.vtu"}}` silently converted the first step — the same silent truncation
  the CLI guard prevents. Every front-end now shares one predicate
  (`sequence_input_needs_driver`), gated on the formats that can carry time so
  the probe costs nothing for the 39 that cannot. Found by the WASM smoke test,
  whose settings surface makes the same routing decision; the Python-only
  refusal test had not covered the C++ engine.
- **The transient XDMF writer now follows the build for its data format,
  exactly like the registry's own `xdmf` entry already does.** Both
  `sequence_to_timeseries` (C++) and Python's `_SeriesWriter` defaulted to
  `"HDF"` unconditionally; on a `-DMESHIOPLUSPLUS_WITH_HDF5=OFF` build (CI's
  Windows leg, and every wheel/binary that ships without HDF5) that either
  threw outright (the C++ engine) or silently produced an HDF-format file via
  the pure-Python writer's own `h5py` (available independently of the C++
  build), which the very same install's C++ reader then could not open --
  swallowed by the read shim's fallback, surfacing as the pure-Python XDMF
  reader's unrelated "Only supports one grid right now" (it has no temporal
  collection support at all). Both now resolve to `"HDF"` only when
  `MESHIOPLUSPLUS_HAS_HDF5` is actually defined, `"XML"` otherwise, with an
  explicit `Binary`/`Ascii` encoding request still winning (and still failing
  by name when the build cannot honour `"HDF"`).
- **The registry's default `vtu`/`vtp` writer entries now follow the build for
  zlib**, the same way the `xdmf` entry already did: they previously hardcoded
  `zlib=true` regardless of `-DMESHIOPLUSPLUS_WITH_ZLIB=OFF`, which every
  Python caller was shielded from by the per-format shim's try-C++-then-Python
  fallback, but which threw for good on any *direct* `_core`/WASM/C-API/Fortran
  caller of the default registry writer on such a build -- exactly the path
  `run_sequence_pipeline`'s per-step `vtu` output takes.
- **`meshioplusplus.mcp._resolve_pattern` now finds a pattern's directory
  component on Windows too.** It split on `os.sep` (`'\\'` on Windows) rather
  than `os.path.split`, so a POSIX-style pattern like `"../*.vtu"` -- what
  every existing caller and test already passes -- had no `os.sep` to split
  on and was treated as a bare filename glob in the sandboxed root instead of
  having its `..` component containment-checked and rejected.

## v9.11.0 (2026-08-02)

New **settings pipeline**: one `settings.json` document describes a whole read →
operation-chain → write run (`{"Input", "Operations": [{"Op": "Transform", ...}, ...],
"Output"}`), executed by the new **`pipeline` verb in both CLIs**, by
`meshioplusplus.run_pipeline` in Python, and by the same C++ engine from every binding —
C (`mio_pipeline_run_file`/`_json`), Fortran, Julia, R, WASM (`runPipeline`) and a new
MCP `pipeline` tool. `MESHIOPLUSPLUS_ABI_VERSION` stays 5 — the one new installed header
(`operations/pipeline.hpp`) is purely additive (Tier C, reviewed in `doc/abi_reviews.md`).

### The pipeline

- ~23 single-mesh operations as steps (`Transform`, `Gradient`, `ConvertCells`, `Refine`,
  `Decimate`, `Smooth`, `Clean`, `Crop`, `Slice`/`Section`, `Isosurface`, `Quality`,
  `Partition` (attaches `partition:part` labels), `ExtractSurface`/`ExtractSkin`,
  `Reorder`, and the data ops `DataDrop`/`DataKeep`/`DataRename`/`DataCalc`/
  `DataCondition`/`ToCell`/`ToPoint`). PascalCase ops and keys, lowercase enum values,
  **strict** parsing (unknown op/key errors by name), steps validated before the input is
  read. Multi-mesh ops (`Merge`/`Interpolate`/`Split`/`Diff`) are rejected pointing at
  their CLI verbs — the recorded v2 follow-up. `Input.Options` narrows the read
  (`PointsOnly`/`DataArrays`/`TimeStep`/`Lenient`/`Mmap`); `Output` selects
  `Encoding`/`Codec`/`FloatFormat` through `registry_write_ex`, so an option the format
  cannot honour is an error. See `doc/pipeline.md`.
- The engine lives in the core (`operations/pipeline.{hpp,cpp}`) in two layers: a typed,
  always-compiled step dispatcher — now the **single owner** the WASM
  `convertSurfaceOps` pipeline also dispatches through (its camelCase op specs and report
  are unchanged; the dedicated `apply_one_op` table is gone) — and a JSON front-end.
- **New git submodule**: [nlohmann/json](https://github.com/nlohmann/json) v3.12.0 at
  `src/cpp/third_party/json`, wired exactly like Eigen (`MESHIOPLUSPLUS_WITH_JSON`,
  default ON with an `EXISTS` probe; PRIVATE include; never in an installed header).
  Without it the typed layer still compiles and only the JSON entry points throw naming
  the flag; wheels and the release CLI binaries carry it, the conan/vcpkg packages keep
  it off (no submodule in a source export — the Eigen rule).
- Python's `run_pipeline` is a pure-Python twin over the public API (per-format fallbacks
  included, sdists fully functional); `_core.run_pipeline_file`/`_json` +
  `_core.pipeline_op_table` expose the C++ engine, and the parity test pins the two
  engines' meshes and reports against each other.

## v9.10.0 (2026-07-31)

New **`gradient`** operation — the gradient, divergence and curl of a `point_data` field.
meshio++ could already transform, transfer (`interpolate`), summarize (`data_info`) and
contour (`isosurface`) a field, but not **differentiate** one, which left two things with no
input: contouring a *derived* quantity (`|grad T|`, vorticity), and the gradient-based error
indicators that drive the selective `refine` shipped in v9.5.0. Both now work end to end.
Dependency-free, on every binding surface, and byte-identical across the three mesh
backends, thread counts and the C++/numpy boundary. **`MESHIOPLUSPLUS_ABI_VERSION` stays 5**
— the two new headers are purely additive (Tier C, reviewed in `doc/abi_reviews.md`).

### Operations

- **`gradient(mesh, array, ...)`** with two methods, three operators and either output
  location. **Green-Gauss** (the default) applies the divergence theorem over the cell,
  fanning each face into triangles about its corner average; that is **exact for a linear
  field on any cell — planar faces or not**, because two faces sharing an edge contribute
  oppositely-wound fan triangles there, so the fan surface is closed and
  `∮ f n dA = V grad f` applies verbatim. **Least-squares** fits over the cells sharing a
  node, with the cell's corner averages as the fit centre so it is likewise exact for a
  linear field; a degenerate neighbourhood falls back to Green-Gauss and is **counted**
  rather than silently wrong. On a 2-D mesh the theorem runs over the corner ring with the
  in-plane normal, exact on a planar cell and invariant under reversal *and* cyclic rotation
  of the ring.
- **Shapes.** An `nc`-component input yields `3 * nc` gradient components, flat and
  row-major as `[component i][derivative j]` at `i * 3 + j`: a scalar gives `(n, 3)` and a
  3-vector `(n, 9)`. Divergence gives 1 component and curl always 3, both needing a 2- or
  3-component field (a 2-component one reads as `(u, v, 0)`, the same padding convention 2-D
  points already use). Output is always `Float64`, named `<input>:gradient` / `:divergence`
  / `:curl` unless overridden — deliberately `name:suffix` rather than the repo's usual
  `prefix:name`, so everything derived from one field sorts next to it.
- **Nothing is faked.** A `cell_data` input raises by name pointing at
  `cell_data_to_point_data`, since a piecewise-constant field has no derivative. Cells that
  cannot be differentiated — below the mesh's own topological dimension, ragged, or a 3-D
  Lagrange type with no face table, or with a degenerate measure — yield **NaN and are
  counted**, never approximated (`compute_quality`'s convention). Geometry, connectivity,
  regions, property sets and every existing array pass through bit-identically.
- **Coordinates and values are recentred on the cell's corner average before any
  arithmetic.** `V = (1/3) sum x_j . A_j` only telescopes because `sum A_j == 0`; on a mesh
  at `x ~ 1e8` the raw form loses eight digits to cancellation and then divides by the
  result. Pinned by a translate-the-mesh invariance test, which was *verified to be inert*
  at a smaller offset before being tightened.
- Exposed on every binding surface (pybind `_core.gradient`, C API `mio_gradient`, Fortran
  `m%gradient`, Julia `gradient`, R `mio_gradient`, WASM `gradient` plus a
  `convertSurfaceOps` pipeline step), as **`data gradient`** in both CLIs — a documented
  departure, since it is a mesh operation living in the `data` group because that is where a
  user looks for it — as an MCP tool, and as a Derivative chip in the browser viewer. Docs:
  [`doc/gradient.md`](doc/gradient.md), including the two worked compositions
  (`gradient` → `isosurface`, `gradient` → `refine --where`) this exists for.

### Fixed

- **`partition(..., ghost_layers > 0)` produced halos that were silently too small on any
  mesh with Int32 connectivity.** The ghost-layer cell→node incidence read connectivity as
  `Conn().As<std::int64_t>()`, which performs **no dtype check** — and a MESHIO-backed mesh
  routinely carries Int32 straight from numpy, so every node id was two fused Int32 entries.
  Most such ids failed the range filter and simply vanished, leaving a plausible-looking but
  undersized halo (plus a read past the end of the last row). The incidence now lives in the
  new `detail/cell_adjacency.hpp`, shared with `gradient`'s least-squares stencil so the two
  cannot disagree about what "shares a node" means, and reads through the dtype-agnostic
  `detail::cell_node_ids`. Output is unchanged for `ghost_layers == 0` and for Int64
  connectivity; the existing test suite was green only because its fixtures were Int64, so
  the regression test uses Int32 and was confirmed to fail before the fix.

## v9.9.0 (2026-07-30)

**The WASM mesh object no longer loses a data array's component shape**, CGNS carries
point/cell data, the `hexahedron27` face-centre defect v9.8.0 documented is fixed, and the
three "handled by Python fallback" gaps that made MED, Exodus and DOLFIN lossy outside
Python are closed. All came out of a downstream WASM consumer re-probing its workarounds
against 9.8.0. Additive on the JS side (a bare `Float64Array` still means a scalar array, so
every existing JS caller is unaffected); two user-visible changes are the *numbering* of
refined hexahedra's face-centre points (same geometry — see the hexahedron27 entry) and
**`MESHIOPLUSPLUS_ABI_VERSION` 4 → 5**, because `MedInfo` gained data members (Tier A).

**Breaking (C++ ABI only):** `MedInfo` gained four fields, so its layout changed. A C++
consumer that compiled against v9.8.0 headers and passes a `MedInfo&` must be recompiled —
which the bumped `SOVERSION` (`libmeshioplusplus_core_*.so.5`) and
`detail/abi_version_check.hpp` both enforce at link time rather than leaving to chance. The
C ABI, Python, WASM, Fortran, Julia and R surfaces are all unaffected.

### WASM / bindings

- **Data-array component counts cross the boundary.** `point_data`/`cell_data`/`field_data`
  crossed as flat, shapeless `Float64Array`s in *both* directions, so an `(n,3)` vector field
  re-entered C++ as `(3n,1)`. Each map now has a sibling `point_data_components` /
  `cell_data_components` / `field_data_components` object, `{name: k}`, following the
  convention `xdmfSeriesWriteDataArrays`' own `components` argument already established
  ("a flat typed array carries no shape"). An absent name means one component, and
  `readMesh` writes an entry only for genuinely multi-component arrays, so scalar-only
  output is unchanged. A length that is not a multiple of its declared count is a catchable
  `Error` naming the array.

  Two consequences that were the actual reported symptoms: **writing a vector field to MED
  now works** (it previously produced a file the MED reader itself rejected with `"field
  data size does not match its declared shape"` — `NCO=1`/`NBR=3n` against `n` points), and
  **object-based operations no longer corrupt vector fields**. The C++ operations were never
  at fault: `subset_gather_rows`, `refine`'s per-component interpolation and
  `reorder_scatter_rows` all derive the row stride from the array's real trailing
  dimensions. With a `(3n,)` array they simply failed their `rows == num_points` test and
  took the pass-through branch, returning stale values of the wrong length. The path-based
  `convert`/`convertSurface`/`convertSurfaceOps` calls never materialize a JS mesh and were
  never affected — which is why the pre-existing "convertSurfaceOps keeps multi-component
  data" smoke step passed throughout.

### Formats

- **CGNS reads and writes point/cell data** (`FlowSolution_t`), where a CGNS export
  previously dropped every field silently. One `FlowSolution_t` per location
  (`GridLocation` = `Vertex` / `CellCenter`; absent reads as `Vertex`, the SIDS default),
  one `DataArray_t` per scalar. **CGNS has no component concept** — no
  `NumberOfComponents` anywhere in the SIDS — so a k-component array is split into k
  siblings named `<name>_0..<name>_{k-1}` and re-joined on read from a *contiguous* run; a
  documented meshio++ convention, like `zstd` for VTU. `cell_data` is written only when
  every cell block is at the zone's `CellDim` (a `CellCenter` array is per-zone, and there
  is no way to distribute one back across blocks of differing dimension without inventing
  values); a mixed-dimension mesh is warn-and-skipped. `FlowSolution_t` is read only for a
  single-zone file. See [`doc/formats/cgns.md`](doc/formats/cgns.md#data-mapping).
- **CGNS gained the external-validation layer** v9.8.0 recorded as a follow-up. `cgnslib`
  is not in apt on a sudo-less machine but is on conda-forge, so: `cgnscheck` reports **zero
  errors** on everything meshio++ writes, for every supported cell type (a new test, gated
  on `shutil.which("cgnscheck")` — it skips with an actionable reason rather than silently
  passing), and a reference `.cgns` **written end to end by cgnslib 4.5.2 itself** is
  committed under `tests/python/meshes/cgns/` (Git LFS; `*.cgns` added to
  `.gitattributes`) and read unconditionally by both readers.
- **MED: a field's `NOM` now carries 16 characters per component**, not a fixed 16. Both
  writers previously wrote one blank 16-char slot whatever the component count, which
  deviates from MED's convention for any k>1 field; when no explicit `med:nom` names are
  supplied, MED's own default spelling `V1..Vk` is generated. Fixed in the C++ **and**
  Python writers together, so they do not diverge. A scalar field's bytes are unchanged.
  Consequence: a k>1 field now reads back with `med:nom` populated where it previously came
  back empty.
- **MED: a mis-shaped field is rejected at write time.** A field's row count is its entity
  count, and there was no write-side check at all — a flattened `(nk,)` vector wrote
  `NBR = nk` against `n` points and produced a file this very reader rejects, so the failure
  surfaced far from its cause. `write_med` now raises a `WriteError` naming the array and
  both counts.
- **Exodus:** the "element attribute must be scalar" guard now tests the product of *all*
  trailing dimensions, matching its Python twin. A 3-D `(n,1,3)` array previously slipped
  past (its `cols()` is 1) and was silently truncated to its first component. The ordinary
  `(n,k)` vector case was already a correct, deliberate error and is unchanged.

#### The three "handled by Python fallback" gaps

Each of these threw `ReadError`/dropped data in the C++ core and was invisible from Python,
where the shim's blanket `except Exception` silently swapped in the pure-Python reference.
WASM, the C API, Fortran, Julia, R and the native CLI have no such fallback, so for them the
construct was a hard failure or a silent loss.

- **MED: `ReadOptions::mLenient` opens the whole Python-only surface**, following the
  mechanism `mdpa` established in v9.1.0. Strict reads are **unchanged** (so the Python shim
  still falls back and the Python surface is byte-identical), but a lenient read now gets
  through a real Salome/Code_Aster file instead of failing on sight of it. Constructs that
  can be *described* are read into `MedInfo` rather than merely skipped — a field's
  `UNI`/`UNT` into `mFieldUnits`, a non-default `NDT`/`NOR`/`PDT` into `mStepMeta`, and every
  step's `PDT` into `mFieldTimeValues`; the structurally unrepresentable ones (a named
  `PFL` profile, an `ELNO`/`ELGA` support, a field mixing nodal and cell supports) drop that
  one field with a warning recorded in `mSkippedConstructs` and keep the rest of the file.
  ELNO/ELGA is *structurally* impossible, not merely unimplemented: the uniform mesh API's
  `cell_data` is always `(n,)` or `(n,k)`, never a per-node-within-cell 3-D shape.
- **MED honours `ReadOptions::mTimeStep`**, and MED is registered in
  `registry_readers_ex()` — which is what makes the options reach WASM, the C API, Fortran,
  Julia, R and both CLIs with no per-binding code. A multi-timestep field used to fail the
  read outright; an explicit step now selects one (0-based, negative counts from the end,
  the `ResolveTimeStep` contract exodus already uses). A non-default step is honoured
  **without** `mLenient`, deliberately: it is a request the Python shim never makes, so no
  Python behaviour depends on it.
- **MED: a block with no `FAM` array reads as family 0** instead of failing the whole file.
  MED spells "belongs to no family" as id 0, so this is the file's own meaning rather than a
  guess. The Python twin was **also** wrong here in a different way — it appended nothing,
  leaving `cell_tags` *shorter* than `mesh.cells`, which the uniform mesh API cannot hold —
  and now zero-fills identically.
- **Exodus writes ordinary `cell_data` as element variables.** `name_elem_var`,
  the `elem_var_tab` truth table and one `vals_elem_var{j}eb{k}` per (variable, block): the
  writer previously emitted none of it, so every `cell_data` array except the
  `exodus:attr:`-prefixed ones was dropped while `point_data` round-tripped. Trailing
  dimensions become extra netCDF dimensions exactly as the nodal path already does, so a
  vector cell field survives. Both writers, both readers.
- **Exodus writes `eb_names` from `Cell` regions**, the inverse of the element-block half of
  the read path — a named block used to come back as the reader's synthetic `Block N`. Only
  written when at least one block is actually named, so a region-less mesh's bytes are
  unchanged.
- **Exodus `time_whole` comes from `field_data["exodus:time"]`** instead of a hard-coded
  `0.0`, and both readers now populate that key with the time of the step they returned — so
  one frame of a transient solve keeps its label through a round trip. (A genuinely
  multi-step Exodus *writer* is a stateful object like `XdmfTimeSeriesWriter` and remains a
  follow-up; this is one mesh, one step, correctly labelled.)
- **Fixed: a heap buffer overflow in the Exodus reader.** Assembling a cell_data array
  allocated a scalar `{total}` buffer and then `memcpy`'d each block's full `Nbytes()` into
  it, so a multi-component element variable wrote `n*k` bytes into `n` bytes' worth of space.
  Pre-existing, but unreachable until this release's writer started emitting element
  variables, and not reachable from any real SEACAS file (none carries a multi-component
  element variable). `Exodus.MultiComponentCellDataRoundTrips` is the regression test.
- **DOLFIN XML writes and reads `point_data`**, as `dim="0"` mesh functions. A
  `mesh_function`'s `dim` attribute is the topological dimension of the entities it lives on,
  so vertices are 0 — this is the format's own notion, not a meshio++ convention, and it is
  the whole discriminator on read. `cell_data` already round-tripped through the same
  `<stem>_<name>.xml` sibling-file mechanism and is untouched. A name used by both locations
  wants the same file, so cell data keeps it and the point array is warn-skipped rather than
  clobbering it; a non-scalar point array is warn-skipped too (a mesh function is one value
  per entity).

  **Not a defect, for the record:** DOLFIN XML's triangle/tetrahedron-only restriction is
  correct by format — it is a simplicial format — and both writers already raise explicitly
  rather than failing silently.

### Operations

- **Fixed: `hexahedron27`'s face-centre table**, whose defect v9.8.0 documented but
  deliberately left in place. `detail/cell_faces.cpp` assigned mid-face nodes 20/22/23 as a
  permuted 3-cycle of the real `vtkTriQuadraticHexahedron::Faces` order, putting
  `extract_surface`/`extract_skin`'s quad9 mid-face node at the wrong position (never a
  wrong topology — facet keying uses corners only). Corrected in lockstep across
  `cell_faces.cpp`, `_skin.py`'s `_CELL_FACES`, `cell_subdivision.cpp`'s quad-face rows and
  `refine_templates.cpp` + `_refine_templates.py`'s absolute 20–23 references, since
  `refine` derives node `20+k` from the k-th quad-face row. The repo's format layer
  (`cgns.cpp`, `gmsh.cpp`) was already on the corrected convention, so this makes the
  codebase self-consistent rather than changing one.

  **User-visible:** `refine` numbers new nodes in slot order, so the six face-centre points
  of every refined hexahedron now get different **ids and coordinate order**. The geometry
  is identical and no test pinned them (only node 26, the body centre, was pinned), but code
  that cached point ids across versions will see the change.

## v9.8.0 (2026-07-29)

**`convert(gmsh → med)` now works directly from every flat binding for real Gmsh 4.1
meshes, and CGNS is a genuine CGNS/SIDS-compliant format instead of a private
tetrahedra-only encoding.** Three gaps reported by a downstream WASM consumer
(CAD-Preview's Gmsh→MED/CGNS bridge), all in the C++ core and invisible from Python
because each format's shim silently falls back to the pure-Python reference on any
exception — a fallback WASM, the C API, Fortran, Julia, R and the native CLI don't have.
Additive: no Python-surface behavior changes (the Python reference's output is the
compatibility baseline for both MED fixes), and `read_cgns`/`write_cgns`'s signatures are
unchanged.

### Formats

- **MED: `gmsh:physical` is now bridged to families natively in C++**, not deferred to
  Python. `write_med` used to throw `"MED: gmsh physical groups handled by Python
  fallback"` unconditionally on any mesh carrying `cell_data["gmsh:physical"]` — and the
  shared registry path (what every flat binding goes through) supplies no fallback, so a
  `.msh` → `.med` conversion was simply impossible from WASM/C API/Fortran. `write_med`
  now folds `gmsh:physical` (via `field_data`-derived names, else `"group_<id>"`, skipping
  an id already covered by a named `Cell` region) into the same per-cell combo pass as
  `Point`/`Cell` regions — a direct C++ port of `_ensure_med_families`'s cell-side
  bridging in `_med.py`, matched step for step.
- **MED: same-type cell blocks are consolidated instead of rejected.** MSH 4.1's
  canonical structure is one cell block per *entity*, so a real 4.1 file routinely carries
  several blocks of the same type — which used to throw `"MED files cannot have two
  sections of the same cell type."` up front, unconditionally, before this bump's
  `gmsh:physical` fix could even be exercised together with a real multi-entity file.
  `write_med` now groups blocks by type (first-seen order) and writes one `MAI/<type>`
  section per type, concatenating connectivity and `FAM`/`NUM` (row-concatenated, written
  for a section only when every contributing block is covered by the source data, else
  dropped for that section with a warning) — mirroring the Python reference's own
  write-time merge, which never had this restriction.
- **CGNS rewritten to a genuine CGNS/SIDS-compliant subset**, readable by
  cgnslib/ParaView/VTK. The previous writer emitted **only the first `tetra` block it
  found** and created empty `ElementRange`/`ElementConnectivity` groups otherwise, so any
  non-tetra mesh — every surface/2-D mesh — wrote a file this library's own reader
  rejected (`HDF5: missing dataset ' data'`; the leading-space name was never the actual
  problem — it is cgnslib's real ADF-over-HDF5 convention, not an ad hoc one, contrary to
  the previous docs here). Every node now carries CGNS's real `name`/`label`/`type`/
  `flags` attributes under HDF5 link+attribute creation-order tracking (load-bearing:
  cgnslib's own node-lookup code has no name-order fallback), with a proper
  `CGNSBase_t`/`Zone_t`/`ZoneType_t`/`Elements_t` tree — one section per cell block, not
  consolidated by type like MED. Covers every fixed-node-count type through
  `hexahedron27`/`pyramid13`, with node-ordering permutations derived from the SIDS
  edge/face conventions and cross-checked against VTK's real translator source; the
  cubic/quartic Lagrange families and ragged (`polygon`/`polyhedron*`) blocks are
  deliberately unsupported (named `WriteError`/`ReadError`, never a guessed ordering).
  2-D-authored meshes now round-trip their point shape instead of always coming back
  `(n,3)`. Backward compatible: a pre-v9.8.0 file (or one from upstream `meshio`) still
  reads via a structural legacy-layout fallback. See
  [`doc/formats/cgns.md`](doc/formats/cgns.md) for the full layout, type table, and what
  CI can and cannot verify (no `cgnslib`/`cgnscheck` reference-fixture layer yet — a
  documented follow-up).

## v9.7.0 (2026-07-29)

**Gmsh MSH 4.1 `$Entities` is supported by the C++ core**, in both directions. Until now
`read_gmsh` threw `Gmsh $Entities not supported by the C++ reader` on sight of the section —
which is the *first* section of every file Gmsh 4.1 writes, so in practice **no real 4.1 file
was readable** from WASM, the C API, Fortran, Julia, R or the native CLI. Python never saw it:
its shim catches any exception and silently re-reads with the pure-Python reference reader, so
the gap was invisible from the one surface that had a fallback. Additive: a mesh with no
`gmsh:dim_tags` writes byte-identical 4.1 output, and 2.2 is untouched.

### Formats

- **Gmsh 4.1 `$Entities` (read).** The section is parsed in ascii and binary, at both `size_t`
  widths real files use (`example/example.msh` is a `4.1 0 4` file). This is not merely "stop
  throwing": `$Entities` is the **only** place 4.1 records physical-group membership — an
  `$Elements` block names an `(entityDim, entityTag)` pair and the physical tag lives on the
  entity — so it is also what makes `cell_data["gmsh:physical"]`, and therefore every named
  [region](doc/regions.md), exist for 4.1 at all. Regions now carry the group's real dimension
  and tag rather than the `-1` placeholders a set-derived region has.

  A file that tags *some* entities gets one `gmsh:physical` array per cell block, with `0`
  (gmsh's "no physical group") for the untagged ones. This deliberately differs from the Python
  reference, which omits the untagged blocks and thereby leaves `gmsh:physical` shorter than
  `mesh.cells` — a shape the uniform mesh API cannot represent. A file that tags *nothing*
  (like `example/example.msh`) still gets no `gmsh:physical` key at all.

- **Gmsh 4.1 `$Entities` (write).** `write_gmsh41` emits `$Entities` and splits `$Nodes` into
  one block per entity whenever the mesh carries `point_data["gmsh:dim_tags"]`, so **4.1 now
  round-trips physical-group membership** — previously only `gmsh22` could, and only from
  Python. ASCII output is byte-identical to the pure-Python reference writer. Without
  `gmsh:dim_tags` there is no entity structure to describe, and the previous single-block
  output is emitted unchanged. The entity set is the union of the node entities and the cell
  entities, not just the former: a straight curve whose only nodes are its endpoints owns none
  of its own yet still carries elements (`example/example.msh` has six), and taking only node
  entities would drop their tags.

- **`GmshInfo` side channel** (`formats/gmsh.hpp`), the `MedInfo`/`ExodusInfo` pattern: it
  carries the `$Entities` bounding-entity tags, which are **signed** (the sign is the
  boundary's orientation) and so cannot be a `Region`. `read_gmsh(path, GmshInfo&, opts)` and
  `write_gmsh41(path, mesh, binary, const GmshInfo&)` are new overloads; the existing
  signatures are unchanged. In Python they surface as `cell_sets["gmsh:bounding_entities"]`
  exactly as the reference reader's do. The shared registry passes none, so the flat bindings
  do not see them — a documented gap, not a silent loss. Note the overloads make `&read_gmsh` /
  `&write_gmsh41` ambiguous as bare function pointers: a compile error, never silent.

- **Gmsh 4.1 metadata without a full read.** `read_gmsh_metadata` no longer declines on
  `$Entities`; it parses that section and `$PhysicalNames` (both small, both ahead of
  `$Elements`) so a summary reports `gmsh:physical` and the named regions — with entry counts —
  from block headers alone. 2.2 still falls back to a full read, as before.

## v9.6.0 (2026-07-29)

MED gains **named regions**, promoting it into the Phase-1 round-trip formats alongside gmsh and
Abaqus. Also closes a real silent-data-loss bug on the write path, adds optional `NUM` global
numbering, and rejects files from a newer MED data model with a clear error. Additive: a mesh
with no regions/`med:num` and a file with `INFOS_GENERALES` `MAJ` ≤ 4 write and read exactly as
before.

### Build / introspection

- **Fixed: the `mcp` extra was unbounded and broke against the SDK's 2.0.0.** `mcp>=1.2` let a
  fresh install resolve 2.0.0, which removed `mcp.server.fastmcp` (replaced by a different
  `mcp.server.mcpserver` API), so importing the server raised `ModuleNotFoundError`. Now
  `mcp>=1.2,<2`. Only a fresh, non-editable `pip install ".[mcp]"` shows this — an existing
  environment keeps whatever 1.x it already had — which is why it surfaced in CI rather than
  locally. The pure tool layer imports no SDK and was never affected; porting to the 2.x server
  API is tracked separately.

- **Compile-time version macros**, so a consumer can feature-detect with the preprocessor the way
  MMG's `MMG_VERSION_GE` allows. `<meshioplusplus/version.hpp>` defines
  `MESHIOPLUSPLUS_VERSION_MAJOR`/`_MINOR`/`_PATCH`, the ordered integer
  `MESHIOPLUSPLUS_VERSION` (`major*10000 + minor*100 + patch`),
  `MESHIOPLUSPLUS_VERSION_STRING`, and `MESHIOPLUSPLUS_VERSION_AT_LEAST(major, minor, patch)` /
  `_BEFORE(...)`; the C header carries the same set as `MIO_VERSION_*`. Until now the release
  version was reachable only at *run* time (`mio_version()`), at configure time (CMake /
  pkg-config), or as the deliberately-different ABI counter — none of which can guard a `#if`.
  The two are complementary, not redundant: the macros describe the header you compiled against
  and the call describes the library you linked, which for a shared build can differ.
  Hand-written rather than `configure_file`d for the same reason `abi_version.hpp` is — the
  single-header amalgamation, pkg-config and hand-written makefiles never run CMake — and
  gated so the duplication cannot drift: `CMakeLists.txt` parses both headers and fails at
  configure time if either disagrees with `project(... VERSION ...)`, and a `static_assert` in
  `c_api.cpp` pins the C macros to the C++ ones.

### Formats

- **MED ↔ named regions.** `FAS`/`GRO` family group names attach as one `Region` per group name
  (`Point` from `NOEUD`, `Cell` — global block-major indices — from `ELEME`) on read, alongside
  the existing `point_tags`/`cell_tags` representation, which is unaffected. On write, when the
  mesh carries no native `point_tags`/`cell_tags` of its own, families are synthesized from
  `Point`/`Cell` regions instead — one family per unique combination of region names an entity
  belongs to, node ids positive from `+1`, element ids negative from `-1` (mirroring the Python
  fallback's `_ensure_med_families`). This closes a real bug: a regions-only mesh (e.g. read from
  Abaqus, with no native MED tags) written through the C++ path used to silently produce a file
  with **no** groups at all, because nothing about the write ever raised to trigger the Python
  bridging. `Side` regions have no MED equivalent and are dropped with a warning. No binding
  change was needed — regions already cross every binding generically. See
  [`doc/formats/med.md`](doc/formats/med.md#named-regions).
- **MED global numbering.** The optional `NUM` datasets Salome/Code_Aster/Kratos write are now
  read/written as `point_data`/`cell_data["med:num"]`. Cell `NUM` is only carried when *every*
  cell block has it — a partial array is not a mesh-wide numbering, and is dropped with a
  warning rather than fabricated.
- **MED version check.** Both the C++ reader and the Python fallback now check
  `INFOS_GENERALES`'s `MAJ` attribute and reject (with the same named `ReadError`) a file written
  by a MED major version newer than 4, instead of failing later with an unrelated structural
  error. Older majors are unaffected.
- MED gains `line4` ↔ `SE4` in its cell-type table (no orientation permutation).

## v9.5.0 (2026-07-28)

`refine` grows from uniform-only to **selective (adaptive) refinement with a conforming
closure** — refine a chosen subset of cells and get back a valid mesh with no hanging nodes,
which is the workflow FEM adaptivity actually needs. Additive: with no selector set the output
is byte-identical to v9.4.1's, which the whole pre-existing refine suite guards.

### Mesh operations

- **`refine` takes a cell selection.** At most one of an explicit list of global (block-major)
  cell indices, a **region** name (a `Cell` region selects its cells, a `Point` region every
  cell with any node in it; a `Side` region is an error — a facet is not a cell), or a trivial
  **`cell_data` predicate** (`quality:scaled_jacobian < 0.3`, which composes directly with
  `attach_quality`). Setting two is an error rather than a precedence rule. A non-finite cell
  value never matches a predicate, deliberately: `compute_quality` reports NaN where a metric
  does not apply, so rejecting such an array would break the headline use case.
- **A conforming closure, driven by one derived rule.** Everything follows from which *edges*
  carry a new node: a quad face gets a centre iff all four of its edges are split, a hexahedron
  a body node iff all twelve are, and a cell's subdivision is the template for its own
  split-edge mask. Two cells sharing an entity read the same edges, so conformity is
  **structural** rather than something the tests merely sample — and with every edge split the
  rules collapse into the old uniform templates, which is what makes uniform output
  byte-identical.
- **`RefineClosure::RedGreen` (default) promotes an affected cell's mask to the smallest
  *admissible* superset.** The admissible sets are closed under intersection, so that promotion
  is a monotone idempotent closure operator and its mesh-wide fixed point is unique and
  independent of the order cells are visited in — determinism follows from the algebra rather
  than from a traversal convention. Per type: every mask is admissible for `line` and
  `triangle`; a `quad` takes either *opposite* edge pair or the full split (a quadrangulation
  of an n-gon satisfies `4Q = B + 2I`, so one or three split edges have no all-quad subdivision
  at any number of interior nodes), so refinement travels along one row rather than the whole
  grid; a `hexahedron` takes unions of its three parallel edge classes, so it travels through
  one dual sheet; a `wedge` splits its triangles, its verticals, or both; a `tetra` takes any
  mask up to two edges plus the four face-triples.
- **`RefineClosure::Propagate` promotes any non-empty mask straight to a full split.** Always
  conforming and defined for every cell type, but **not local**: it converges to uniform
  refinement of the whole edge-connected component. It ships as the always-works baseline and
  as the test oracle, documented as such.
- **`RefineClosure::Balanced` does not close at all**, keeping the hanging nodes and merely
  enforcing 2:1 balance — the classic adaptive-mesh-refinement meaning of "propagate", and the
  only mode whose cost is bounded by the selection rather than by the mesh. A cell is split
  fully or not at all, and is drawn in only when some cell sharing a **node** with it would end
  up more than one level above it. Node adjacency rather than edge adjacency is load-bearing:
  across a hanging interface the coarse cell spans a whole edge while the fine cell has half of
  it, so the two are different entities and an edge-keyed rule is blind to exactly the
  coarse/fine adjacency it polices. On a mesh of uniform level nothing propagates — one cell of
  a 4×4×4 hexahedral block costs 64 → **71** cells, against 125 under `RedGreen` and 512 under
  `Propagate` — and balancing only bites from the second adaptive pass onwards. The output is
  therefore **1-irregular and not conforming**; every constrained node (edge midpoints and
  quad-face centres alike) is reported in the new Int64 `refine:hanging` `point_data` array,
  which is attached only when there are any, so the conforming closures are unaffected.
- **`refine:level`** (opt-in `record_levels`): the Int64 per-cell refinement depth, `0` for a
  cell no full split touched and `+1` per full split, with a transitional child inheriting its
  parent's level because a green split is a closure, not a refinement. The name is reserved —
  an input already carrying it is **updated** rather than replicated, so successive passes
  accumulate, and that is the one observable change to an existing behaviour.
- With `levels > 1` and a selector, level *k* refines the children of level *k−1*'s fully split
  cells. Green cells are **not undone** before a later refinement, so repeated selective passes
  over the same region degrade element quality — documented in `doc/refine.md`, with
  `refine:level` plus `mCellMaps` noted as the hierarchy a future green-undo needs.

### Surfaces

- Exposed everywhere `refine` already was: pybind kwargs, **`mio_refine_ex` + `mio_refine_opts`**
  on the C ABI (`mio_read_opts`' append-only reserved-tail discipline; `mio_refine` is unchanged
  and delegates), optional arguments on Fortran/Julia/R, a fourth `options` argument on the WASM
  `refine` plus the `convertSurfaceOps` pipeline op, and
  `refine … [--cells i,j,k | --region NAME | --where "q < 0.3"] [--closure redgreen|propagate]
  [--record-levels]` in **both** CLIs. The browser viewer's Refine chip gains the predicate and
  closure controls.
- The pure-numpy reference implements all of it and stays a byte-for-byte twin, pinned by
  `tests/python/test_refine.py::test_cpp_matches_python_selective` across both closures and two
  levels; its subdivision tables are checked against the C++ ones through a new
  `_core.refine_mask_table` export rather than transcribed.

### C++ ABI

- **`MESHIOPLUSPLUS_ABI_VERSION` 3 → 4.** `RefineOptions` gained data members, which is a layout
  change to a type consumers can name. `COMPONENTS C` is unaffected and stays `SOVERSION 0`.

## v9.4.1 (2026-07-28)

- **Fixed: `tools/check-abi-version.sh`'s review gate passed vacuously.** It matched any row of
  `doc/abi_reviews.md` whose first column was the current ABI version, and two ABI-3 rows have
  existed since v9.4.0 — so from v9.5.0 onwards *every* header change that held the ABI would
  have matched one of them and exited 0 reporting "records the additive review" about a review
  of a different release. That left Tier B (an edit to the body of an existing `inline`
  function) with no gate at all, since `tests/cpp/test_abi_layout.cpp` provably cannot see one.
  The lookup is now keyed on the ABI version **and** the release version in `CMakeLists.txt`,
  **and** requires every changed header to be named in the matching row — keying on the release
  alone is not enough either, because a header change that skips the version bump still finds
  the previous release's row. `doc/abi_reviews.md`'s `headers changed` column is therefore
  load-bearing, and its contract text (which described the old, weaker behaviour) is corrected.
  `tests/python/test_check_abi_version.py` drives the real script over a throwaway git
  repository and asserts it *fires* — the guard this script never had, and the same lesson the
  backend guard learned in v9.1.0. `MESHIOPLUSPLUS_ABI_VERSION` is unaffected and stays 3.

## v9.4.0 (2026-07-27)

The C++ ABI contract becomes explicit, machine-checkable, and no coarser than the code
requires. Additive except for the `SOVERSION` change noted below.

### C++ ABI

- **`MESHIOPLUSPLUS_ABI_VERSION`, a binary-compatibility counter separate from the release
  version.** `find_package(... X.Y.Z EXACT ...)` keys on the release version, which moves
  whenever anything in the project does — so a release that provably cannot affect a compiled
  consumer still forced every C++ consumer to re-pin and rebuild. v9.3.0 is the case in point:
  its entire installed-header delta was one new `inline constexpr` string in
  `formats/exodus.hpp`. The new counter moves only when the headers stop being compatible, and
  is exported as `MESHIOPLUSPLUS_ABI_VERSION` from `meshioplusplusConfig.cmake` beside the
  existing `MESHIOPLUSPLUS_MESH_BACKEND` / `MESHIOPLUSPLUS_WITH_*` introspection. It lives in
  exactly one place, `abi_version.hpp`; CMake parses it back out of that header rather than
  keeping a second copy. Retroactively: ABI 1 = v9.0.0, 2 = v9.1.0, 3 = v9.2.0 onwards.
  `X.Y.Z EXACT` remains fully supported as the conservative pin.
- **A mismatch now fails at link time instead of corrupting memory.**
  `detail/abi_version_check.hpp` plants a reference to
  `meshioplusplus::detail::abi_version_is_<N>()` in every TU that includes `mesh.hpp`, and the
  library defines exactly one such symbol — the `detail/mesh_backend_check.hpp` technique
  applied to the second axis, deliberately as a *separate* symbol so a consumer with the right
  backend and stale headers is told about the headers. Opt out with
  `MESHIOPLUSPLUS_NO_ABI_VERSION_CHECK`. As for the backend guard, **MSVC + a shared build is a
  documented gap** (`/FAILIFMISMATCH` records are not reliably carried through an import
  library), and `gnu::used` is load-bearing — CI asserts with `nm -uC` that the reference is
  really emitted, which is the check whose absence let the backend guard ship inert through
  v9.1.0.
- **Breaking: the C++ variants' `SOVERSION` now tracks the ABI version**, so they install as
  `libmeshioplusplus_core_<backend>.so.3` rather than `.so.0` and the dynamic linker itself
  refuses to load an incompatible library into an already-linked binary. Existing C++ consumers
  must relink once — a no-op in practice, since that contract already required rebuilding with
  the library. **The C API's `libmeshioplusplus` and the Fortran library keep `SOVERSION 0`**;
  their contract (append-only option structs, pin the major) is unchanged, as are the Python
  wheel, WASM and the Julia/R bindings.
- **Two gates keep the number honest**, because a hand-bumped integer otherwise has exactly the
  "someone forgets" failure mode of the prose rule it replaces: `tests/cpp/test_abi_layout.cpp`
  pins `sizeof`/`alignof` for every boundary type per backend (catches layout changes
  mechanically), and `tools/check-abi-version.sh` fails a build whose installed headers changed
  while the ABI version did not and `doc/abi_reviews.md` records no additive review (catches
  inline-body/ODR changes, which no tool can infer). Neither is sufficient alone.
- **New [`doc/abi.md`](doc/abi.md)** states the criterion the above rests on: scope is *every*
  installed header — not a curated subset, which would have missed `formats/mdpa.hpp`,
  `formats/xdmf_time_series.hpp`, `kratos_bridge.hpp` and every `operations/*.hpp` options
  struct — classified by what the change does. Layout changes and edits to an existing inline
  function's body both bump; purely additive changes do not. `doc/cpp_api.md`'s "Versioning:
  what to pin" is rewritten around it, and no longer describes the `SOVERSION` as meaningless.

## v9.3.0 (2026-07-27)

Exodus support for particle/peridynamics meshes, from
[VSCode-MDPA-Preview#63](https://github.com/loumalouomega/VSCode-MDPA-Preview/issues/63).
Both items are additive.

### Exodus

- **Fixed: a NUL-terminated `elem_type` made a file unreadable.** netCDF text
  attributes carry an explicit length, and NetCDF.jl — what PeriLab and other
  Julia solvers write Exodus with — counts the C string's terminating NUL as part
  of it. A `SPHERE` block therefore arrived as the 7 characters `"SPHERE\0"`,
  matched no key in the C++ reader's type table, and failed the read with
  `Exodus: unknown element type SPHERE` — the NUL invisible in the message,
  because `std::runtime_error::what()` is a `const char*` that stops at it.
  `netCDF4` strips the NUL on the way in, so the Python reference never saw this
  and the shim's silent fallback hid it everywhere **except WASM**, which has no
  fallback; that is why it surfaced in the VS Code extension rather than in
  Python. Both readers now trim trailing NULs and spaces before the type lookup,
  which also covers fixed-width writers that pad with spaces. `SPHERE` itself was
  already mapped to `vertex` on both paths — nothing about the type table
  changed.
- **Per-element attributes now round-trip as `cell_data`.** Exodus stores a fixed
  number of floating-point attributes per element of a block (`attrib{k}`, named
  by `attrib_name{k}`) — the standard home for a `SPHERE`'s **radius**, a beam's
  cross-section, a shell's thickness. meshio++ read none of them. They now read
  and write under the `exodus:attr:` prefix (`cell_data["exodus:attr:RADIUS"]`),
  on both the C++ and the Python path, so the value reaches every binding that
  carries cell_data. The prefix is load-bearing in both directions: on read it
  keeps an attribute apart from a same-named element *variable* (`name_elem_var`
  — constant in time vs. per-time-step, genuinely different concepts), and on
  write it is the only signal saying which `cell_data` arrays belong in
  `attrib{k}`. Values are always float64; a block the file gives no such
  attribute reads as NaN, and on write an all-non-finite block is left out again,
  so a file where only some blocks carry an attribute round-trips exactly rather
  than gaining NaN attributes. A multi-component array under the prefix is a
  `WriteError` naming it, since an Exodus attribute is one value per element.
  Ordinary (non-attribute) `cell_data` is still dropped on write — neither writer
  emits `vals_elem_var`, a pre-existing gap this does not change.
- **The file that failed is now a committed fixture**, at
  `tests/python/meshes/exodus/DCBmodel_PD_solid.e` (Git-LFS): a real PeriLab
  double-cantilever-beam run — 504 `SPHERE` particles in four blocks, 2-D
  coordinates, nine nodal fields and ten time steps whose damage field goes from
  0 to 0.48, so a reader pinned to the first step fails rather than merely
  differs. A hand-authored fixture can reproduce the shape but not the encoding
  (`netCDF4` strips the NUL whatever spelling you pass), so both exist, and a
  byte-level test asserts the fixture still carries the NUL — otherwise a
  re-fetch from an upstream that changed writers would leave the regression
  suite passing while testing nothing. It is redistributed unmodified under
  **BSD-3-Clause** (Copyright (c) 2023 Christian Willberg, Jan-Timo Hesse, DLR)
  with its notice and licence text alongside; permissive, so no obligation
  attaches to the rest of this MIT repository. Credited in `CITATION.cff`.

## v9.2.0 (2026-07-27)

Fixes the gaps a real Kratos Multiphysics consumer hit while building against
v9.1.0. Everything is additive; the two deliberate behaviour changes are called
out in their ledes.

### Kratos consumers

- **`Begin Properties` bodies now ride on the `Mesh`.** v9.1.0 parsed them into
  the `MdpaInfo` side channel, but `ReadFn`/`ReadExFn`/`WriteFn` have no info
  slot, so nothing reachable from `registry_read` could ask for one — every
  registry-based consumer (that is, all of them) got the property *ids* and no
  material data, and the v9.1.0 `Properties` plumbing was unreachable in
  practice. Property sets are now part of the uniform mesh API
  (`AddPropertySet`/`NumPropertySets`/`GetPropertySet`/`HasPropertySet`/
  `FindPropertySet`, ascending by `mId`) on all three backends, `read_mdpa`
  stores them unconditionally, and `KratosMesh::Materialize` fills
  `ModelPart::Properties` with **real values** instead of the id-only sets that
  made `to_model_part`'s "apply property" overload a silent no-op. They are keyed
  by id rather than entity index, so no operation has to remap them;
  shape-preserving operations carry them, restructuring and multi-input ones do
  not. `MdpaInfo` is retained and still wins when supplied — it preserves the
  blocks' *file order*, which the mesh channel deliberately does not.
  **Behaviour change:** a registry-driven `.mdpa` → `.mdpa` write now emits the
  full bodies where it previously emitted empty blocks. A mesh with no property
  sets writes byte-identical output.
- **The automatic tag pass can be narrowed per key.** A Kratos properties id
  arrives as a `gmsh:physical` cell tag, so a deck came back with a spurious
  `gmsh_physical_<id>` SubModelPart beside its real ones, which a consumer had to
  filter by name. For a genuine gmsh file that inference is wanted, so the key
  stays in `KnownTagKeys()` and the caller now says which meaning applies:
  `SetTagSubModelPartKeys(...)`, `ExcludeTagSubModelPartKey(...)` and
  `TagSubModelPartKeys()` beside the existing all-or-nothing
  `SetBuildSubModelPartsFromTags`. The default is unchanged.

### XDMF time series

- **`XdmfSeriesMode::Append` works with the `NamedArray` `WriteData` overload.**
  Both landed in v9.1.0, for the same restartable-solver consumer, and did not
  work together: `Impl::mNumPoints`/`mNumCells` were written only by
  `WritePointsCells`, which an appending writer cannot call (its guard is
  `mHasMesh`, which appending sets), so the overload validated every array
  against zero and rejected all of them with `expected 0 (0 x 1)`. A resumed
  solver had to re-stage a whole `Mesh` per step — exactly the cost the array
  overload exists to avoid. The counts are now recovered from the document, out
  of `<Topology NumberOfElements>` (read first, so a **Mixed** series works —
  `read_xdmf_metadata` declines Mixed outright) and the geometry `<DataItem>`'s
  `Dimensions`. Where a foreign document declares neither, the writer warns once
  and skips the length check rather than rejecting everything.
- **The append path shares the reader's structural resolution.**
  `xdmf_resolve`/`parse_dims` moved to a private `formats/xdmf_doc.hpp`; the
  writer had carried a weaker transcription. Two consequences, both fixes:
  appending to a series whose static grid is not literally named `mesh` no longer
  adds a second static grid, and appending to a non-version-3 document now fails
  instead of silently continuing. Reader output is byte-identical.
- **Moved-from writers are diagnosable.** `SetAutoFlush` lacked the null guard
  its neighbours had. Observers and idempotent operations are now safe no-ops;
  `WritePointsCells` and both `WriteData` overloads throw.

### Build and packaging

- **The mesh-backend link guard actually fires.** `detail/mesh_backend_check.hpp`
  promised a link error naming the backend when a consumer compiles with no
  backend macro against a NATIVE/KRATOS build. It could never fire: the guard is
  an `inline` variable that nothing reads, and such a variable is emitted lazily,
  so no relocation reached the object file — with or without the macro, at `-O0`.
  Dropping `inline`/`const` does not help either (a plain `static const` is
  discarded at `-O2`). Fixed with `gnu::used` on GNU/Clang and
  `#pragma detect_mismatch` on MSVC, which names both backends. **MSVC plus a
  shared build remains a gap** — the mismatch records are not reliably carried
  through a DLL import library — and is now stated rather than implied. A CI step
  compiles a TU without the definitions and requires the link to fail naming the
  backend, with a positive control beside it.
- **The documented `find_package` line works.** `doc/cpp_api.md` and `README.md`
  both printed `find_package(meshioplusplus 9.1 EXACT ...)`, which cannot succeed
  against a `9.1.0` install: under `SameMajorVersion`, `EXACT` is a full string
  comparison. Both now spell all three components, the neighbouring "an exact
  `= 9.1.x` dependency" prose is corrected, and CI asserts that the full version
  satisfies `EXACT`, that `major.minor` does not, and that both documents quote
  the working line verbatim.

### Documented limitations

- The `XdmfTimeSeriesWriter` destructor finalizes, so **deleting the output while
  the writer is alive recreates it** — and with `Append` the next run then
  continues a series you believed deleted, surfacing one run later as a wrong
  step count. Documented on the destructor, on `Finalize()` and in
  `doc/xdmf_time_series.md`; destroy the writer before removing its output.
- `read_xdmf_metadata` still declines a `Mixed` topology. The new
  `xdmf_grid_counts` could summarize it, but that path is load-bearing for
  `registry_read_metadata`'s decline-and-fall-back contract and
  `MeshMetadata::mFellBackToFullRead`.
- Property sets do not cross a boundary where the mesh is materialized in the
  host language (Python's numpy `Mesh`, WASM's JS objects, the C API's
  `mio_mesh` accessors): a `PropertySet` has no numpy or embind analogue.
  File-to-file paths keep them, because the `Mesh` never leaves the core.

## v9.1.0 (2026-07-26)

Closes the gaps a real C++ Kratos Multiphysics consumer hit against v9.0.0 while
wrapping both the I/O and the operations layer. Everything is additive: the
Python wheel, C API, Fortran/Julia/R bindings, WASM and both CLIs behave exactly
as before by default, and the two deliberate behaviour changes are called out in
their ledes below.

### Kratos consumers

- **Kratos entity names are no longer discarded.** `GeometricalEntity` gained an
  optional name (`Name()`/`HasName()`), so `SmallDisplacementElement3D4N` no
  longer degrades to `Element3D4N` on every Kratos -> format -> Kratos trip. It
  is stored as an interned `const std::string*` from a root-owned
  `detail::NamePool`, not a `std::string`: there is one distinct name per *block*
  but one entity per *cell*, so an owned string would add ~320 MB to a 10 M-element
  model part. New five-argument `CreateNewElement`/`CreateNewCondition`
  `CellType` overloads carry it on the bulk-ingest path; the existing four-argument
  ones are untouched, and an absent name still means "derive it from the cell type".
  `bridge_traits` gained a detection-guarded `NameOf` customization point (every
  existing duck-typed source keeps compiling and simply reports no name), and both
  `to_model_part` and `from_model_part` now preserve the name instead of
  re-deriving it.
- **Entity names also survive the `KratosMesh` staging round trip**, via
  `SetBlockEntityName`/`BlockEntityName` (KRATOS-only extras, the shape of
  NATIVE's `PointsData()`/`ConnSpan()`). `CollectEntityBlocks` groups runs by
  *effective* name, so an explicitly-named `Element3D4N` still merges with
  entities materialized without names -- only genuinely different spellings split.
- **`ModelPart` gained a real Properties store** (`CreateNewProperties`,
  `HasProperties`, `GetProperties`, `NumberOfProperties`, `Properties()`), so
  material data crosses the bridge instead of being reduced to a bare id. Values
  are `PropertyValue` key/value pairs in the new backend-neutral
  `meshioplusplus/properties.hpp`, shared with MDPA so there is exactly one
  representation. meshio++ cannot fill a real `Kratos::Properties` itself -- that
  needs `KratosComponents<Variable<T>>::Get`, Kratos's own registry, deliberately
  not linked here -- so a fourth `to_model_part` overload hands each value to a
  caller-supplied applier callback; `doc/cpp_api.md` carries the five-line body.
- **`KratosMesh::Materialize()` now honours `gmsh:physical` as the properties
  id.** Every entity used to land on properties 0, silently dropping the material
  assignment the file carried. No file output changes: `CollectEntityBlocks` /
  `RestoreCellData` rebuild `gmsh:physical` from the staged cell data, not from
  entity properties ids.
- **Nested SubModelParts survive the staging round trip**, using `'/'` as the
  sanctioned path separator -- already what the MDPA reader emits. `RestoreRegions`
  walks recursively and emits `parent/child`; `BuildSubModelPartsFromRegions`
  splits on `'/'` and walks/creates the chain, adding members to the leaf (Kratos's
  own upward propagation fills the ancestors). A name segment that is empty or
  contains `'.'` (which `ModelPart::FullName` reserves) is now a warning and a
  skip rather than a `std::invalid_argument` escaping a lazy `GetModelPart()`;
  the region itself stays on the mesh. A region's `mDim`/`mTag` still come back
  as `-1` when no staged region of that name supplied them -- a SubModelPart has
  nowhere to store them, and inventing a value would be worse.
- **Application-specific Kratos names now resolve in `ModelPart` too.**
  `cell_type_from_kratos_name_or_suffix` (exact lookup, then longest resolving
  suffix) moved into `kratos_names.hpp` as the single owner of that rule;
  `ModelPart::CreateNewElement` and the MDPA reader both go through it, so they
  cannot disagree. Before this, `CreateNewElement("SmallDisplacementElement3D4N",
  ...)` threw -- which is what essentially every real Kratos application calls.

### MDPA

- **`Begin Properties` bodies no longer make a deck unreadable.** They are parsed
  unconditionally (a pure de-throwing, so no read that used to succeed changes)
  and carried in a new `MdpaInfo` side channel -- the `MedInfo`/`ExodusInfo`
  pattern, needed because `NDArray` has ten numeric dtypes and no string one, so
  `CONSTITUTIVE_LAW LinearElastic3DLaw` has no `field_data` representation at all.
  A plain number becomes a Float64 scalar, an inline `Begin Table` an `(n, k)`
  array, and anything else -- a constitutive-law name, a bracketed vector -- is
  kept verbatim as text, which is both lossless and what the pure-Python reference
  does. New overloads `read_mdpa(path, MdpaInfo&, opts)` and
  `write_mdpa(path, mesh, info)`; the existing one- and two-argument forms are
  unchanged, and the registry (so every flat binding) passes none -- a documented
  gap, not a silent loss.
- **Per-block Kratos entity names round-trip** through `MdpaInfo::mEntityNames`.
  The reader's block-splitting key gained the entity name, so two adjacent
  `SmallDisplacementElement3D4N` and `TotalLagrangianElement3D4N` blocks -- both
  `tetra` -- stay separate instead of collapsing onto one name.
- **`write_mdpa` now declares every properties id its rows reference.** The rows
  have always written their `gmsh:physical` value as the property id while the
  header was a hard-coded single `Properties 0`, so a tagged mesh produced a file
  referencing undeclared properties, which Kratos's own `ModelPartIO` rejects. A
  mesh whose ids are all 0 -- every mesh with no `gmsh:physical` -- still emits
  exactly the same two lines and is byte-identical to before.
- **New `ReadOptions::mLenient`** downgrades "this reader cannot represent
  construct X" to a warning plus a skip, recording what was skipped in
  `MdpaInfo::mSkippedConstructs`. It is deliberately *not* "ignore all errors": a
  malformed row, a bad node reference or non-sequential node ids still throw,
  because continuing past those would return a mesh that is quietly wrong rather
  than merely incomplete. It reaches every language through the existing
  `ReadOptions` plumbing (`mio_read_opts.lenient` takes a second former `reserved`
  slot, so the struct size and every preceding offset are unchanged) plus
  `--lenient` on the native CLI. The Python `read` deliberately does *not* gain it:
  mdpa's Python path is the pure-Python reference, which already accepts every
  construct the flag covers, so the parameter would be dead.

### Transient XDMF

- **New `XdmfTimeSeriesWriter::Flush()`.** The light data used to be written only
  at `Finalize()`, so a run that was killed, hit a node failure, or was simply
  still going left heavy data on disk and **no readable `.xdmf`**. `Flush()`
  writes the document as it stands, to a sibling temp file and then `rename`, so a
  crash *during* a flush cannot truncate the previous one; heavy data is flushed
  first, so the `.xdmf` never names a dataset that is not on disk. Opt-in
  `SetAutoFlush(true)` does it after every step -- off by default because a flush
  re-serializes the whole document, making per-step flushing quadratic in the step
  count (and, for the `"XML"` data format, in the data volume too).
- **New append mode** (`XdmfSeriesMode::Append`, a trailing defaulted constructor
  parameter) continues an existing temporal collection instead of overwriting it,
  so a restarted analysis does not destroy the previous run's output. The
  collection is resolved structurally, the same way `read_xdmf` does; the
  heavy-data counter resumes past what is already on disk by scanning the
  container (`h5::open_file_rw` + `H5Lget_name_by_idx`, or probing `<base>N.bin`)
  rather than trusting the document, because a mis-resumed counter would silently
  overwrite `data0`. Appending to a path with no file yet is simply a fresh
  series, so a restartable solver can pass it unconditionally.
- **New `WriteData(time, point_arrays, cell_arrays)` overload** taking
  `NamedArray{mName, mNumComponents, mValues}` vectors -- the granularity a solver
  actually has once `WritePointsCells` has fixed the geometry. Under the KRATOS
  backend this also avoids re-staging the whole ModelPart every output step when
  only the values changed. It shares the step-grid scaffolding and the
  `<Attribute>`/`<DataItem>` emission with the `Mesh` overload, which is kept and
  unchanged -- deliberately *not* re-expressed in terms of the new one, since that
  would force its Int64 cell data to Float64 and change existing output.
- Exposed on every surface: C API (`mio_xdmf_series_create_ex` +
  `mio_xdmf_series_opts`, `_flush`, `_finalized`, `_write_data_arrays` with
  `mio_named_array`), Fortran, Julia (`flush!`, `finalized`, `mode=`/`auto_flush=`,
  and a dict-taking `write_data!`), R, WASM (three files plus the smoke test's
  exhaustive guard) and Python (`flush()`, `auto_flush`, `mode=`,
  `write_data_arrays()`).

### Build, CI and documented contracts

- **`{shared, KRATOS}` is now a CI-covered combination** (`cpp-install` matrix),
  the configuration a Kratos application actually ships and the only one where a
  missing `MESHIOPLUSPLUS_API` on the KRATOS backend is a link error.
- **`MESHIOPLUSPLUS_NO_STD_SPAN` needed no code change.** `std::span` appears in
  exactly two lines repo-wide -- the `#include` and the inline
  `NativeMesh::ConnSpan()` -- so the macro is **ABI-neutral**, and the guard is
  `#ifndef`, so a consumer whose Boost uBLAS collides with `<span>` under MSVC can
  define it themselves against *any* prefix, including a distro or Conan build
  that left it off. `tests/consumer/` gained a
  `MESHIOPLUSPLUS_CONSUMER_NO_STD_SPAN` option and CI a leg that exercises it, so
  that is now a gate rather than a claim.
- **Documented what to pin.** The C API is `SOVERSION 0` + `SameMajorVersion` with
  append-only option structs, so `find_package(meshioplusplus 9 COMPONENTS C)` is
  right. The **C++ API makes no ABI promise** -- `Mesh`, `ModelPart` and
  `GeometricalEntity` are header-defined types whose layout changes with the
  headers, and this release adds a member to `GeometricalEntity` -- so pin
  `9.1 EXACT` for `COMPONENTS CXX` and rebuild the consumer whenever meshio++ is
  rebuilt. See `doc/cpp_api.md`.
- **meshio++ is serial, and that is intended.** There is no MPI in the library, no
  distributed reader or writer and no communicator anywhere in the API; `FindMPI`
  in the generated package config is HDF5's transitive requirement, not
  meshio++'s. The intended distributed workflow is `partition(mesh, {nparts,
  ghost_layers})` -- `mGhostLayers > 0` produces exactly the shared-node halo an
  MPI assembly needs. Now stated in `doc/cpp_api.md` and `doc/partition.md`.

## v9.0.0 (2026-07-25)

Closes the limitations and follow-ups recorded against v8.9.0's installable C++
API — the gaps a simulation-code consumer (the motivating one being a Kratos
Multiphysics application) actually hits.

The major-version bump marks that C++-consumer milestone; it is **not** a breaking
change — everything below is additive, and every existing surface (Python, C API,
Fortran, Julia, R, WASM, both CLIs) behaves as before. One practical consequence of
the number itself: the CMake package's compatibility mode is `SameMajorVersion`, so a
`find_package(meshioplusplus 8.x ...)` minimum-version request rejects a 9.x install —
consumers pinning a minimum should ask for `9.0`.

### Kratos consumers

- **`KratosMesh::GetModelPart()` gained a `const` overload.** It was non-`const`
  only because materialization is lazy, so a wrapper whose API takes
  `const Mesh&` could not reach the ModelPart at all. The ModelPart is a cache
  of the staged mesh, exactly like the staging every other const accessor
  already builds on demand, so the members are now `mutable`.
- **`InvalidateBlocks()`'s rebuild is no longer lossy.** It used to drop ragged
  pass-through blocks and the entire SubModelPart structure — a documented
  "sharp edge" that silently lost data for anyone who mutated the ModelPart and
  wrote the mesh back. Ragged blocks (which never become entities, so a
  ModelPart edit cannot have touched them) are now carried through **at their
  original block positions**, cell_data slices included, and SubModelParts are
  read back as named `Cell`/`Point` regions. Still unrecoverable, because a
  SubModelPart has nowhere to store them: SubModelPart *nesting* (regions are
  flat) and a region's `mDim`/`mTag` (taken from a staged region of the same
  name when one exists, else `-1`).
- **New `.mdpa` C++ reader/writer** (`formats/mdpa.{hpp,cpp}`, registered in
  `registry.cpp`), so Kratos's native format is reachable from the C++ core, the
  C API, Fortran, Julia, R, WASM and the native CLI instead of Python only.
  Entity names resolve through the existing `backends/kratos_names.hpp` tables
  plus a longest-suffix fallback (so real decks' `SmallDisplacementElement3D4N`
  resolves via `Element3D4N`); SubModelParts map to regions; the
  hexahedron20/27 Kratos↔VTK permutation is applied on read and undone on write.
  Constructs with no C++ representation (`Table`, `Geometries`, `Mesh <id>`,
  `Constraints`, non-empty `Properties` bodies, …) throw **by name**.
  **`meshioplusplus.mdpa.read` deliberately stays on the pure-Python reference**:
  that reader returns a non-standard mesh (cell_data nested by cell type, plus
  `mesh.misc_data`/`mesh.geometries_block`) which the C++ `Mesh` cannot carry, so
  a C++-first shim would silently change the module's documented Python output.
  `mdpa.write` is a normal try-C++/fallback shim.

### Partitioning

- **`PartitionOptions::mGhostLayers` is implemented** (it was declared and threw).
  A positive value grows each piece by that many shared-node BFS layers of other
  parts' cells — the halo an MPI domain decomposition needs — tagged with an
  Int64 `partition:ghost` cell_data (0 = owned, L = reached at layer L). The
  pieces then overlap, so partition-of-unity holds only at 0 (the default).
  **Breaking (narrow):** `partition_labels` now *rejects* a nonzero
  `ghost_layers` instead of throwing the old "not implemented" error — a flat
  per-cell label array is the ownership map and a cell can be a ghost of several
  parts at once. The numpy fallback raises `NotImplementedError` rather than
  silently returning unghosted pieces.

### Transient output

- **New `XdmfTimeSeriesWriter`** (`formats/xdmf_time_series.{hpp,cpp}`), the C++
  counterpart of the Python-only `TimeSeriesWriter`; `HDF`/`XML`/`Binary`, pimpl'd
  so pugixml and `hdf5.h` stay out of the installed header, and exposed on the C
  API as `mio_xdmf_series_*`. Verified byte-identical `write_xdmf` output before
  and after the refactor that hoisted the shared payload machinery into
  `xdmfcommon`.
- **XDMF temporal *reading* now works at all.** `read_xdmf` resolved
  `Domain/Grid[0]` — the collection itself — and threw `unknown section Grid` on
  any time series; it now resolves a temporal collection structurally and
  honours `ReadOptions::mTimeStep`, and `read_xdmf_metadata` fills `mTimeValues`.
  XDMF is the second reader after exodus to honour `mTimeStep`.

### C API

- **New `mio_write_ex` / `mio_write_opts`**, the symmetric counterpart of
  `mio_read_ex`/`mio_read_opts` (same append-only `reserved[5]` ABI discipline):
  ASCII/binary encoding, the vtu/vtp block codec, and an ASCII float format.
  `mio_write` is exactly `mio_write_ex(..., <defaults>)` and is unchanged. An
  option the target format cannot honour **fails the call** rather than being
  ignored.
- Backing it, a new `registry_write_ex()` in the core is now the single owner of
  "write this format with these parameters"; the CLI's private variant table was
  deleted in favour of it, so the two cannot drift.
- **`meshioplusplus ascii` now handles text-only formats** in the native CLI.
  "Write this as ASCII" is a sensible request for a text format — it is just the
  normal write — but the native CLI rejected it while the Python CLI accepted
  it, so the two disagreed on which files the verb handled (`.mdpa` was the
  visible case). `binary` still fails, now by name.

### Performance

- **`PointDataNames()`/`CellDataNames()`/`FieldDataNames()` no longer sort on
  every call.** Note the discipline this introduces on the MESHIO backend, whose
  data maps are public struct members that `bindings/python/np_conversions.hpp`
  writes directly (the sanctioned uniform-API exception): anything bypassing the
  `Add*` accessors must call the new `Mesh::InvalidateNameCaches()`.
  `detail::NamedItems` (NATIVE/KRATOS) is sound by construction.

### Packaging

- **The installed prefix is now relocatable.** HDF5 was linked through
  `${HDF5_C_LIBRARIES}` — absolute paths baked into
  `meshioplusplusTargets.cmake`, so the package only resolved on a machine whose
  HDF5 sat exactly where the build's did. It now links the `HDF5::HDF5` imported
  target with a matching `find_dependency`. **The cost, stated plainly:** an
  HDF5-enabled install requires `C` among the consumer's `project()` languages
  (FindHDF5/FindMPI need it), and says so by name if it is missing.

## v8.9.0 (2026-07-25)

### Installable C++ API

- **New `MESHIOPLUSPLUS_INSTALL_CPP` option (default OFF)** installs the full C++ core —
  the whole `src/cpp/include/meshioplusplus/` header tree, the format registry, every mesh
  and data operation, and the header-only `kratos_bridge.hpp` — as exported CMake targets.
  Until now the only installed artifact was the flat C API, which cannot hand out a
  `Mesh` or a `meshioplusplus::ModelPart`. Default OFF, so `pip install .`, the wheels and
  every existing CI leg are unaffected. See [`doc/cpp_api.md`](doc/cpp_api.md).
- **All three mesh backends install side by side** from one prefix, as
  `meshioplusplus::core_meshio` / `::core_native` / `::core_kratos`, with
  `meshioplusplus::core` aliasing the build's default backend. One installed meshio++
  therefore serves consumers that disagree about the backend, instead of each needing a
  private install. `MESHIOPLUSPLUS_INSTALL_CPP_BACKENDS` trims the set (each backend is a
  full, independent compile of the core).
- **`find_package(meshioplusplus CONFIG COMPONENTS CXX|C|Fortran)`** now genuinely enforces
  its components: asking for one the install does not carry fails at `find_package` time
  rather than at the link.
- **Mesh-backend mismatches are now diagnosed, not silently undefined.** Each variant
  carries its `MESHIOPLUSPLUS_MESH_BACKEND_*` macro in `INTERFACE_COMPILE_DEFINITIONS`, so
  a CMake consumer cannot disagree by accident; defining two is a compile error
  (`mesh.hpp`), and defining none while linking a NATIVE/KRATOS build is now a link error
  naming the expected backend (`detail/mesh_backend_check.hpp`, opt out with
  `MESHIOPLUSPLUS_NO_BACKEND_LINK_CHECK`). Previously this was an ODR violation that
  usually surfaced as unrelated memory corruption.
- **New `MESHIOPLUSPLUS_API` export macro** (`export.hpp`) annotating the public C++
  surface, with the C++ libraries built `-fvisibility=hidden` /
  `VISIBILITY_INLINES_HIDDEN`. Shared builds export the documented API and nothing else,
  on every platform including Windows (`__declspec(dllexport/dllimport)`).
- **New `MESHIOPLUSPLUS_NO_STD_SPAN` option**, propagated to consumers. It already existed
  as a macro in `native_mesh.hpp` (for the MSVC `<span>` / Boost uBLAS collision that
  Kratos hits) but there was no way to set it from CMake.
- The generated config now emits real `find_dependency()` calls (ZLIB, zstd, lz4, netCDF,
  KaHIP, OpenMP/TBB/Kokkos) for the C++ targets, and **installs `cmake/FindKaHIP.cmake`**
  beside itself so a KaHIP build resolves for a consumer with no copy of its own. The
  C-API-only install still emits none, exactly as before. HDF5 is deliberately not
  re-found (it is linked by absolute path), which also keeps a CXX-only consumer from
  needing `enable_language(C)`.
- **New `lib/pkgconfig/meshioplusplus-cxx.pc`** for non-CMake build systems, carrying the
  C++ include dir, `-std=c++20` and the backend macro. Kept separate from the C API's
  `meshioplusplus.pc`, whose `Cflags` must stay valid for a plain-C compile.
- **New `tests/consumer/`**, a standalone CMake project built against an install prefix
  with the source tree unreachable, plus a `cpp-install` CI job running it for every
  backend in both a static and a shared configuration.
- Conan gains `with_cxx_api` / `cxx_api_backends` options (and Conan components when the
  C++ API is on); vcpkg gains `cxx-api`, `cxx-api-native` and `cxx-api-kratos` features.
- `build/configure.sh` / `configure.bat` gain `--install-cpp` and `--cpp-backends <LIST>`,
  and print the matching `cmake --install` line, so the documented convenience path can
  produce the C++ install without hand-writing the CMake invocation.

### Fixes

- **The vendored pugixml and Eigen include directories are no longer PUBLIC** on the core
  object library. Neither is referenced by any installed header, and both would otherwise
  land on every consumer's include path — pugixml colliding with a consumer's own vendored
  copy (Kratos vendors one), and Eigen's, being a git-submodule path inside the source
  tree, breaking `find_package` outright on any machine without that tree.
- `meshioplusplus.pc`'s `Libs.private` was missing `-lzstd` / `-llz4` despite the build
  options existing, so a static link against a zstd- or lz4-enabled build under-linked.
- The pybind11 `_core` module's wheel install rule is now gated on scikit-build-core's
  `SKBUILD` variable. Its `DESTINATION meshioplusplus` is relative to the wheel platlib,
  so a plain `cmake --install` of a Python-enabled tree — a normal thing to do now that
  the C++ API installs — used to dump a stray `<prefix>/meshioplusplus/_core*.so` at the
  prefix root. Wheels and editable installs are unaffected.

## v8.8.0 (2026-07-25)

**Threaded (OpenMP) WebAssembly build.** The `@meshioplusplus/wasm` package now
ships **two** native artifacts: the sequential `meshioplusplus_wasm` (unchanged)
and a new threaded `meshioplusplus_wasm_mt`, compiled with the OpenMP parallel
backend over Emscripten's Wasm threads (pthreads/SharedArrayBuffer). The core's
`parallel_for` loops — every mesh operation the browser viewer runs through
`convertSurfaceOps` (quality/smooth/refine/decimate/partition/merge/slice/
isosurface) and VTU zlib compression — run multi-threaded when the threaded
build is loaded.

- The loader `loadMeshioPlusPlus()` **auto-selects** at runtime: the threaded
  build under Node and in a **cross-origin-isolated** browser context (COOP
  `same-origin` + COEP `require-corp`), the sequential build otherwise. A
  threaded module cannot instantiate without cross-origin isolation, and there
  is no in-artifact fallback, which is why both are shipped. Force one with the
  new `{ variant: 'mt' | 'seq' }` option.
- New `parallelBackend()` binding (JS/WASM) reports `"openmp"` or `"seq"` — the
  loaded artifact's parallel backend, alongside `meshBackend()`.
- The browser viewer picks the threaded build wherever it can: the Pages demo
  vendors a COOP/COEP service worker (`src/viewer/public/coi-serviceworker.js`)
  to become cross-origin isolated; the `file://` wheel-embedded viewer is
  unaffected (it uses no WASM at all).
- `build/configure-wasm.sh --build` now builds **both** variants by default
  (`--seq-only` skips the threaded one); new CMake option
  `MESHIOPLUSPLUS_WASM_THREADS`. Nothing outside the wasm build changes — the
  Python/native/C-API/Fortran builds are untouched.

**MCP server** (`meshioplusplus[mcp]`, `meshioplusplus-mcp`). Every operation
is now exposed to AI agents over the Model Context Protocol: 33 stateless,
file-path-based tools (inspection incl. `info`/`stats`/`quality`/`diff`/
`regions`/`data_preview`; `convert` subsuming the CLI's ascii/binary/compress/
decompress variants; all 16 path-based mesh operations; the data operations;
gated `data_export` → Parquet and `screenshot` → PNG image content) plus
`meshioplusplus://formats` and `://version` resources, all returning strict
JSON with truncation caps. New optional extra `mcp` (the official MCP Python
SDK, which itself needs Python ≥ 3.10) and console script `meshioplusplus-mcp`
(stdio; `--root DIR` / `MESHIOPLUSPLUS_MCP_ROOT` sandboxes every path). The
pure tool layer (`meshioplusplus/mcp/_tools.py`) works without the extra, and
a parity-guard test fails CI whenever a new public operation lacks a tool.
Docs: `doc/mcp.md`.

## v8.7.0 (2026-07-24)

Five improvements identified by an audit of the project's own documented gaps
and a direct check of the WASM artifact's behavior, all verified end to end
(not from source inspection alone):

1. **`read_metadata` reports a mesh's named regions**, closing the "enumerating
   regions costs a full read" gap. New `MeshMetadata::mRegions`
   (`RegionSummary{name, kind, dim, tag, num_entries}`, no entries) is
   populated from an already-in-memory mesh at essentially no extra cost
   (every fallback metadata path, and Exodus, which always falls back); a
   native metadata path (VTU/VTP/XDMF/Gmsh 4.1) reports none, since none of
   those formats currently map regions at all. Exposed identically on every
   binding: Python `read_metadata(...)["regions"]`, C API
   `mio_read_metadata_num_regions`/`_region_name`/`_region_info` (reusing
   `mio_region_info`'s shape), Fortran `mio_metadata%regions`, Julia/R the
   equivalent, WASM `readMetadata(...).regions`, and both CLIs' `info`.
2. **`split(mesh, by="regions")`** (plural — a new, additive criterion,
   distinct from the pre-existing singular `"region"`, which is unchanged) is
   one submesh per named **Cell** region, running in the C++ core and so
   reaching every binding through the existing shared `split_by_from_name`
   string dispatch with no further plumbing. Unlike every other criterion it
   is **not** a partition: a cell in several regions lands in several output
   pieces, a cell in none lands in none, and `Point`/`Side` regions produce no
   piece at all. A companion **`meshioplusplus regions FILE`** CLI verb (both
   CLIs) lists a mesh's regions using the `read_metadata` work above.
3. **`gmsh22` is now a selectable write format on every binding** (WASM, C
   API, Fortran, and both CLIs — Python already had it). `write_gmsh22`
   already synthesized `gmsh:physical` from named Cell regions when writing a
   mesh built from another format, so it was already the only Gmsh writer that
   round-trips region **membership**, not just the group name (which is all
   the registry-default 4.1 writer keeps) — it just wasn't reachable outside
   Python until now, since only `"gmsh"` → the 4.1 writer was registered in
   the shared dispatch tables. Read-side needs no new key: reading
   auto-detects a file's own `$MeshFormat` version.
4. **MED writes and reads ordinary fields (`point_data`/`cell_data`) for the
   single-timestep common case.** Previously any data-carrying mesh threw
   unconditionally ("fields handled by Python fallback") — fatal in WASM,
   which has no Python to fall back to, and the reason a MED export could not
   carry data out of a WASM-hosted tool at all. The new C++ path writes one
   `NOE`/`MAI.<type>` support subgroup per field with fixed `ndt=1`/`nor=-1`,
   blank units/component-names, and **no MED-4.1 optimization bitmask** — a
   deliberate scope cut, not an oversight: this project's own reader never
   reads the bitmask, so its absence costs nothing for a meshio++ round-trip,
   only for interoperability with external tools (Salome/MEDCoupling) that use
   it. The reader **declines** (defers the whole file to Python) rather than
   silently drop information whenever a field declares real units or
   non-default timestep metadata. Multi-timestep name-encoded arrays
   (`"Name[idx] - pdt"`) and the `med:field_units`/`med:step_meta` Python-only
   `field_data` conventions still defer to Python — the guard for the latter
   two necessarily lives in the Python shim rather than the C++ core, since
   those dict-valued conventions cannot survive the Python→C++ mesh conversion
   at all and so can never be observed on the C++ side by any binding.
5. **Ragged (polygon/polyhedron) cell blocks now cross the WASM/JS boundary**,
   on both read and write — previously rejected outright with "not supported
   by the JS API yet". Represented as flat CSR arrays instead of a nested
   array of arrays (which embind cannot represent efficiently):
   `{type, data, rowOffsets}` for 1-level ragged (jagged polygon rows) and
   `{type, data, faceOffsets, cellOffsets}` for 2-level ragged (polyhedron,
   cell → faces → node ids). MED is the ragged-**polygon**-capable writer
   (`POG`/`POG2`) this closes a real gap for; polyhedron blocks now cross the
   boundary correctly too (verified via `clean`, since geometry operations
   accept them), but **no C++ format writer accepts a polyhedron block yet** —
   a separate, pre-existing, documented gap this work does not (and could not)
   close, so writing one still throws naming the format rather than silently
   dropping data.

**Breaking:** none. `split`'s pre-existing `by="region"` (singular) keeps its
exact prior behavior; `mio_read_opts`'s and the Julia `_CReadOpts`'s ABI are
unaffected by this release (unlike v8.6.0, no new trailing field was added
here). The C++-side guard that used to check `HasFieldData("med:field_units")`
before writing MED fields is removed — it was structurally dead code (the
Python→C++ mesh conversion this project's own `med_write` binding uses always
drops non-numeric `field_data` entries before the C++ core ever sees them), so
its removal changes no observable behavior; the actual (now correctly enforced
in the Python shim) deferral rule is documented above.

## v8.6.0 (2026-07-24)

**Exodus II is usable on real files.** Three defects made the format unusable
for anything a mesher actually produces; each is fixed with its own test against
a hand-authored SEACAS/Cubit-shaped fixture (`tests/python/exodus_fixture.py` —
meshio++'s own writer emits no `qa_records`, no `eb_names` and no side sets, so a
round-trip test could never have caught the first of these).

1. **Ordinary metadata no longer fails the read.** The reader used to throw
   `ReadError("Exodus: <key> handled by Python fallback")` on `qa_records`,
   `info_records`, `ns_names` and any `node_ns*` variable. Every file SEACAS,
   Cubit or Sierra writes carries `qa_records`, so this made Exodus **entirely
   unreadable from WASM**, where there is no Python fallback to defer to
   (`readMesh(..., 'exodus')` threw on any real file; verified against the built
   artifact before and after). On the Python path the throw was invisible — the
   shim silently swallowed it — which is why it survived this long.
   `qa_records`/`info_records` are **preserved, not dropped**: they travel in a
   new `ExodusInfo` side-channel struct (the established `MedInfo`/`OpenFoamInfo`
   pattern) that the pybind binding attaches to the Python `Mesh` as `info`, so
   `mesh.info` is byte-identical to what the Python reference produced. `NDArray`
   has no string dtype, so they cannot ride on the mesh itself; as with `MedInfo`,
   the flat bindings (C, Fortran, Julia, R, WASM) construct one and drop it — a
   documented gap, not a silent loss.
2. **Element blocks, node sets and side sets become named `Region`s**, closing
   Exodus's share of the v8.1.0 "Deferred to Phase 2" list. One
   `RegionKind::Cell` per `connect{k}` named from `eb_names` (falling back to
   `"Block <id>"`) and tagged with its `eb_prop1` id — so two blocks of the
   **same** element type stay distinguishable rather than collapsing together;
   one `RegionKind::Point` per `node_ns{k}` from `ns_names`/`ns_prop1`; and one
   `RegionKind::Side` per `elem_ss{k}`/`side_ss{k}` from `ss_names`, as
   `(global cell, local facet)` pairs. Exodus numbers an element's sides in its
   own order, which is *not* `detail/cell_faces.hpp`'s, so the facet column is
   remapped through a new `exo_face_index` (mirroring `abq_face_index`); a gtest
   pins every entry against `cell_faces` by node set rather than trusting the
   transcription. Reading only — the **writer still emits no regions**, so Exodus
   is recorded as a read-only region source (`READ_ONLY_REGIONS` in
   `tests/python/test_region_roundtrip.py`) rather than a round-trip row.
3. **Time steps are selectable.** New `ReadOptions::mTimeStep` (0 = the first
   step, preserving today's behaviour exactly; negative counts from the end) and
   `MeshMetadata::mTimeValues` (from `time_whole`). Previously every reader
   meeting a multi-step file silently took the first step and warned "Skipping
   some time data"; now an out-of-range request is an error naming the available
   count, never a silent clamp. Exodus is registered as a `ReadExFn` +
   `MetadataFn` in `registry.cpp`, so `registry_reader_supports_options("exodus")`
   flips **false → true**.

Threaded through every binding surface the way `mPointsOnly` was: pybind
(`exodus_read(path, time_step=)`, `read(..., time_step=)`, `time_values` in
`read_metadata`), the C API (`mio_read_opts.time_step`, taking one of the six
former `reserved` slots so **the struct's size and every preceding field's offset
are unchanged**; plus `mio_read_metadata_num_time_values`/`_time_values`),
Fortran (`m%read(..., time_step=)`, `metadata%time_values`), Julia
(`ReadOptions(time_step=)`, `MeshMetadata.time_values`), R
(`mio_read(time_step=)`, `mio_read_metadata()$time_values`), WASM
(`readMeshSelective(path, {timeStep})`, `readMetadata(...).timeValues`) and both
CLIs (`convert --time-step=N`; `info --fast` now prints the available steps).

## v8.5.0 (2026-07-23)

**Parallelization pass over the newer operations** — an audit of every
operation's serial loops parallelized the remaining safe ones (independent
iterations writing disjoint slots), with output staying **byte-identical**
across backends and thread counts: `detail/subset.cpp`'s connectivity
remaps/index maps (benefits crop, split and partition at once), quality's
per-cell array assembly and histogram (fixed-chunk partials merged serially in
chunk order; the min/max/sum summary stays serial so reported values are
unchanged), reorder's per-cell sort keys and connectivity rebuilds, merge's
ragged builds and dedup-filtered gathers, crop's kept-cell test (phase-split:
parallel flag pass + serial compaction) and clean's rep→final remaps. The
deliberately-serial determinism passes (first-seen dedup, FP scatter
accumulation, greedy/BFS loops, stable argsorts) are untouched. New
`Quality.DeterministicAcrossRuns` gtest.

**KOKKOS parallel backend** (`-DMESHIOPLUSPLUS_PARALLEL_BACKEND=KOKKOS`) — a
fifth backend for `meshioplusplus::parallel_for`, running on
`Kokkos::DefaultHostExecutionSpace` (host deliberately: loop bodies capture
host pointers, so device offload is served by the DLPack/CuPy handoff, not by
this backend). Bring-your-own like KaHIP (never picked by `AUTO`; point
`Kokkos_DIR` at an installed Kokkos ≥ 3.4), lazily initialized only when the
embedding application hasn't initialized Kokkos itself, `parallel_for_bw`'s
4-thread bandwidth cap preserved by range partitioning. Building Kokkos
alongside `MESHIOPLUSPLUS_BUILD_C_API=ON` needs Kokkos built with
`-DCMAKE_POSITION_INDEPENDENT_CODE=ON` — its default static archives aren't
PIC and fail to link into the shared `libmeshioplusplus.so`. CI: the new
`kokkos` job (cached PIC source build of Kokkos 4.5.01, full gtest suite, plus
a 2-thread pool re-run).

**`NDArray` buffer-allocator hook** (`meshioplusplus::set_buffer_allocator`) —
the `doc/gpu.md` Phase-2 enabler: every owning `NDArray` buffer is now
allocated through an optional process-global `BufferAllocator` (plain C
callbacks), so readers can fill e.g. CUDA pinned memory directly, removing the
staging copy of a later host→device transfer. Each buffer keeps a
`shared_ptr` reference to the allocator it was born with, so uninstalling the
hook never orphans live buffers; content, zero-copy and determinism contracts
are unchanged (views are unaffected). The Python/CuPy wiring
(`pinned_reads()`) is the recorded follow-up in `doc/gpu.md`.

## v8.4.0 (2026-07-23)

**GPU handoff at the I/O boundary** (`to_dlpack` / `to_cupy` / `from_cupy`) —
move a mesh's arrays to and from device memory through the standard exchange
protocols, with no file round-trip. Python-only
(`src/python/meshioplusplus/_gpu.py`); the C++/WASM/C/Fortran core is
untouched and stays dependency-free. Stated plainly: **a host→device move is
always a bus transfer** — what this removes is the file round-trip and every
*extra* copy on either side of that one transfer; "zero-copy" applies only to
host buffer sharing and on-device adoption, never to the transfer itself.

- `to_dlpack(mesh)` returns a payload of host numpy arrays, each natively
  speaking `__dlpack__`/`__dlpack_device__` (`kDLCPU`) — consumable by
  `np.from_dlpack`, `torch.from_dlpack`, JAX, Numba, CuPy, … It honors the
  interop layer's `zero_copy_only` **host** buffer-sharing contract.
- `to_cupy(mesh, float32=…, int32=…, pinned=…, stream=…)` transfers the
  payload to the CUDA device: points, per-block connectivity, point/cell data,
  and named regions as device **index arrays** (`Side` `(cell, facet)` pairs
  included — the one interop target that keeps them). Optional pinned-memory
  staging with async DMA on a caller-supplied stream. Deliberately no
  `zero_copy_only` parameter here — it would be a lie on a 100%-copy path.
- `from_cupy(payload)` rebuilds an ordinary `Mesh` with one deliberate
  device→host copy per array; arrays may be anything exposing DLPack or
  `__cuda_array_interface__` (CuPy, torch, Numba). Host DLPack exporters are
  adopted zero-copy without CuPy installed.
- Dtypes stay canonical float64/int64 unless explicitly downcast
  (`float32=True` / `int32=True`) — recorded in the warned `notes`, and int32
  index casts are range-checked, never wrapped.
- Deliberately **no `[gpu]` pip extra**: CuPy wheels are CUDA-version-specific
  (`cupy-cuda13x` / `cupy-cuda12x` / `cupy-cuda11x` / ROCm), so a pinned extra would break for
  most users; the install error names the wheel recipe instead. Kept out of
  `[all]` and `[interop]`. Docs: [`doc/gpu.md`](doc/gpu.md).
- CI covers the pure payload layer and the DLPack **host** round-trip only;
  the CUDA device path runs under gated tests (`importorskip("cupy")` plus a
  real device check) and is **not covered by public CI** — its lines read as
  uncovered on the Codecov patch check by design.
- Housekeeping: the `CLAUDE.md` version-bump checklist grew from six to
  **eight** files — the Julia (`bindings/julia/MeshioPlusPlus/Project.toml`)
  and R (`bindings/r/meshioplusplus/DESCRIPTION`) manifests added in v8.3.0
  carry versions too and are exactly the kind that drift unnoticed.

## v8.3.0 (2026-07-23)

**Julia and R bindings** — the next two languages of the scientific-computing
audience after Python, C, Fortran and JavaScript. Both sit on the **existing**
C API (`libmeshioplusplus`, the installed pure-C99 header), exactly as the
Fortran module does: no new C++ is written, and the C++/WebAssembly/Python core
is untouched.

- **Julia** — `bindings/julia/MeshioPlusPlus/`, docs
  [`doc/julia.md`](doc/julia.md). `ccall` into the installed shared library,
  discovered through `MESHIOPLUSPLUS_LIB` or the standard loader path. A `Mesh`
  wraps the opaque handle with a GC **finalizer** (unlike Fortran, which frees
  explicitly). Genuine **zero-copy borrows** — `points_ptr`,
  `connectivity_ptr`, `point_data_ptr`, … — are returned as a `MeshBorrow`
  that both keeps its owning mesh alive and records the mesh's mutation
  generation, so using one after a mutating call raises `BorrowError` rather
  than reading stale memory. That is the C header's rule 3 *enforced*, not
  merely documented.
- **R** — `bindings/r/meshioplusplus/`, docs [`doc/r.md`](doc/r.md). Plain
  `.Call` over R's own C API rather than Rcpp, keeping the dependency
  footprint at zero; the handle is an external pointer with a registered
  finalizer. **R is copy-only**: R vectors are R-managed, so the C API's
  zero-copy borrow cannot survive into R without ALTREP machinery that is out
  of scope, and every accessor copies. There is therefore no `_ptr` accessor
  at all — the 0-based reader is named `mio_connectivity_raw`, deliberately
  not `_ptr`, so nobody reads it as a borrow. R also has no native 64-bit
  integer, so `int64` arrays arrive as `double` (exact to 2^53, no `bit64`
  dependency), with the stored dtype reported in a `"dtype"` attribute.

Both cover the full Fortran surface: lifecycle, read/write (including the
selective-read options and file metadata), every setter and getter, named
regions, and all ~24 operations — including the ones returning an opaque C
result, which are always drained through `_take_mesh` so every mesh handed to
the caller owns its handle and no piece can dangle when its result is freed.

Both follow the Fortran module's two conventions verbatim, because Julia and R
are column-major too: points shaped `(dim, n)` and connectivity
`(nodes_per_cell, n)` are the **same memory** as the C API's row-major shapes,
so **nothing is transposed anywhere**; and connectivity is **1-based**, with
the ±1 shift applied inside the copying accessors only. Index maps and
permutations shift the same way, with the C API's `-1` "pruned / absent"
sentinel becoming `0` — never a valid 1-based index. `partition_labels`
returns part *ids* rather than indices and is deliberately left unshifted.

> **Licence exception, deliberate and the one thing to know:** the Julia
> binding in `bindings/julia/` is **not MIT**. It is released under the
> **GNU General Public License, version 3 (GPL-3.0)** — a copyleft license,
> not a permission-required one: anyone, including a company, may use,
> modify or sell it commercially with **no permission needed**; the
> condition is on *conveying* (distributing) it — a distributed copy or
> modified version must be under GPL-3.0 too, with source available.
> Purely private/internal use that is never distributed carries **no
> obligation** at all. Because GPL-3.0 **is** OSI-approved, the package is
> eligible for Julia's General registry (registration itself is a separate
> follow-up, not done yet), and it still installs by path or URL in the
> meantime; there is also no BinaryBuilder JLL yet. Everything else — the
> C++ core, the C API the binding calls, and the R binding — remains MIT.

Neither binding invents a workaround for the C ABI's documented gaps (point and
cell sets beyond regions, the `frozen` pin masks, per-cell-type counts in the
statistics report, ragged block connectivity, the combined `data_manage`); each
repeats the same list in its own README and doc page. CI gains a `julia` and an
`r` job, both mirroring the existing external-consumer smoke test: build the C
API, `cmake --install` it, then consume it exactly as a user would — `Pkg.test`
and `R CMD check --as-cran` respectively — and each also asserts that a missing
library fails with a message naming how to build one.

**Julia and R example notebooks** — `example/julia/*.ipynb` and `example/r/*.ipynb`, the same
three-notebook tour (`01_read_and_visualize`, `02_convert_and_inspect`, `03_mesh_operations`) as the
existing C++ notebooks, called through the new bindings on their own Jupyter kernels ([IJulia](https://github.com/JuliaLang/IJulia.jl)
/ [IRkernel](https://irkernel.github.io/)). Since the flat C API these bindings ride on can't drive the SVG
writer's per-call data-driven colouring either (a gap the C++ notebooks don't hit, since they call
`write_svg` directly), quality/field renders that the C++ tour shows as a coloured mesh are shown as a
small chart instead — hand-rolled SVG bar/histogram charts in Julia, plain base-R graphics in R, neither
needing a plotting-library dependency. Writing these caught two real, now-fixed defects: `smooth()`'s `mu`
default had been hardcoded `-0.53` in **both** new bindings — an invented value, never checked against the
real default (`-0.34`, matching Fortran and the Python bindings) — which is wide enough to reliably degrade
a tangled mesh rather than recover it; and R's data setters were found to always write `Float64` regardless
of the R vector's storage mode, meaning `mio_split(by = "region")` cannot be driven by a tag built fresh in
R (only one already present in a read file, or produced by the C++ core itself), now a documented `doc/r.md`
gap distinct from Julia, which has no such restriction. Building the C API with HDF5 on for Julia notebooks
also surfaced a real Debian/Ubuntu + Julia interaction — an IJulia kernel's `dlopen` of a
`libhdf5_openmpi`-linked library can fail on a `libcurl` symbol-version mismatch against Julia's own bundled
`libcurl` — documented in `doc/julia.md` and fixed in the `julia` CI job (HDF5 off, not just netCDF).

## v8.2.0 (2026-07-23)

**In-memory interoperability with the wider ecosystem, without a file
round-trip.** A new Python-only module,
`src/python/meshioplusplus/_interop.py`, converts a `Mesh` to and from the
in-memory objects of its main consumers, sharing the underlying numpy buffers
wherever the target accepts them as they are. Nothing here touches the
C++/WASM/C/Fortran core, which stays dependency-free — this is pure Python over
the numpy the readers already return.

- **PyVista** — `to_pyvista(mesh, zero_copy_only=False)` /
  `from_pyvista(grid)`. Builds the VTK 9 `connectivity`/`offsets`/`celltypes`
  triple from the mesh's blocks in order, so mixed-type meshes are the normal
  case. This exists because PyVista's own `from_meshio` targets the upstream
  `meshio` package and does not recognize `meshioplusplus`.
- **trimesh** — `to_trimesh` / `from_trimesh`. trimesh holds triangles only, so
  non-triangle input is routed through meshio++'s *existing* operations rather
  than a reimplementation: volume meshes through `extract_surface`,
  higher-order cells through `convert_cells("linearize")`, quads and polygons
  through `convert_cells("simplexify")`.
- **Apache Arrow / Parquet** — `to_arrow` / `from_arrow` / `write_parquet` /
  `read_parquet`, plus a `meshioplusplus data export IN OUT.parquet
  [--location point|cell]` sub-verb in the Python CLI's `data` group (the
  native C++ CLI has no counterpart). This is a **tabular export of data
  arrays for analytics — not a mesh format**: it moves `point_data`/`cell_data`
  into pandas/polars/DuckDB, does not round-trip geometry, and is deliberately
  *not* registered in the format registry, so
  `meshioplusplus convert mesh.vtu out.parquet` does not work. Multi-component
  arrays become Arrow `fixed_size_list` columns rather than `_0`/`_1`/`_2`
  suffix columns, which would lose the shape, and the mesh's counts, cell
  types, version, location and region names ride along in the schema metadata.

**The zero-copy contract.** Buffers are shared when the target accepts the array
as-is (contiguous, supported dtype, right shape). Every `to_*` takes
`zero_copy_only`: `False` (default) records each copy in a `notes` list surfaced
as a warning; `True` raises instead, naming the array and the reason. It governs
arrays that exist in the `Mesh` — `points`, each data array, and single-block
connectivity — but not *derived* ones: VTK's `offsets`/`celltypes` have no
meshio++ counterpart and a multi-block mesh has no single connectivity array to
share, so making those fatal would reject every mixed-type mesh. Copies are
forced by an int32→int64 connectivity widening, a `wedge` block (whose meshio
node order differs from VTK's), a 2-D mesh's point padding, and any
cross-block concatenation. Returned wrappers hold references to every shared
array, so they stay valid after the source mesh is garbage-collected.

**Regions per target.** `Point` and `Cell` regions export as int8 `region:<name>`
mask arrays, plus a JSON sidecar (`meshioplusplus:regions` in PyVista's
`field_data`, trimesh's `metadata`) carrying each region's `dim`/`tag`, which a
mask alone cannot express — so a gmsh physical group's integer tag survives a
PyVista round-trip exactly. `Side` regions are dropped with a warning naming
them: neither target has a `(cell, local facet)` concept. For trimesh, regions
follow the composed operations' own documented behaviour and get no second
policy — a pure-triangle input keeps them, anything routed through
`extract_surface` loses them.

**New extras**, all pip-friendly and all Python-only: `[pyvista]`, `[trimesh]`,
`[arrow]`, and the aggregate `[interop]`. They are deliberately **not** in
`[all]`, which means "the optional deps the *formats* need" (h5py/netCDF4);
`[viewer]`/`[kahip]`/`[codecs]` stand apart for the same reason. Every
third-party import is lazy and raises a named `pip install meshioplusplus[...]`
error when absent.

**Open3D and DOLFINx are deferred to Phase 2.** `has_open3d()`/`has_dolfinx()`
ship returning `False` and `to_open3d`/`to_dolfinx` raise `NotImplementedError`
naming the phase. The constraints are recorded now in
[`doc/interop.md`](doc/interop.md): Open3D's `Vector3dVector` typically copies
(so the zero-copy claim differs per structure) and its wheel is ~400 MB, while
DOLFINx needs a single-cell-type mesh, a `ufl`/`basix` domain, an MPI
communicator and a VTK→basix node-ordering permutation, and is conda/apt-only so
it can never be a pip extra.

Docs: [`doc/interop.md`](doc/interop.md).

## v8.1.0 (2026-07-23)

**Named groups of entities are now a first-class part of the core.** A `Region`
— a named group of *points*, *cells* or, for the first time, *cell facets* —
lives in `meshioplusplus::Region`, is carried by all three mesh backends, crosses
the pybind11 boundary natively, and is visible from the C API, Fortran,
WebAssembly and the native CLI. This is Phase 1 of unifying what every format
spells differently: gmsh physical groups, Exodus blocks and sets, Abaqus
`*NSET`/`*ELSET`/`*SURFACE`, MED families, UNV groups, Ansys components,
OpenFOAM patches, Kratos SubModelParts.

What round-trips now:

- **Gmsh** (`.msh` 2.2) — physical groups map to `cell` regions carrying their
  dimension and their integer tag, derived from the `gmsh:physical` cell_data
  and `$PhysicalNames` on read and synthesized into both on write. A mesh whose
  groups came from another format now writes real physical groups.
- **Abaqus** (`.inp`) — `*NSET` → point regions, `*ELSET` → cell regions and
  `*SURFACE` → **side** regions, in both the C++ core and the Python reference.
  The C++ reader gained full parity with the Python one along the way
  (`GENERATE`, set-of-set references, `ELSET=` on an `*ELEMENT` line,
  `*INCLUDE`), so a set-carrying `.inp` no longer falls back.
- **Side sets have no precedent anywhere in meshio++** — they are only reachable
  through `.regions`, not through `point_sets`/`cell_sets`.

`point_sets` and `cell_sets` keep working exactly as they always have. They are
now views over the mesh's regions: reading one materializes the historical shape,
writing one creates or replaces the matching regions. Two accommodations make
that lossless rather than nearly so — a `cell_sets` entry that is not cell-index
data (gmsh's `gmsh:bounding_entities`, whose entity tags are signed) is kept
verbatim through a passthrough, and a `None` block becomes an empty array, which
every consumer already treated identically.

Also in this release:

- **`detail/region_remap.hpp`** — one shared remapper replaces nine near-identical
  per-operation Python implementations. `crop`, `split`, `merge` (with source-id
  namespacing), `reorder`, `clean`, `partition`, `convert_cells`, `refine` and
  `decimate` remap regions in C++; `transform`, `smooth`, `interpolate` and the
  data operations pass them through; `slice`, `isosurface` and surface extraction
  drop them with a warning naming the operation, never silently.
- **`detail/cell_index.hpp`** — the single owner of the global (block-major) cell
  index that `cell` and `side` regions are defined against.
- **KRATOS backend**: regions materialize as SubModelParts, taking precedence
  over the names the integer-tag inference would have claimed.
- **C API**: `mio_regions_create` / `_count` / `_name` / `_info` / `_entries` /
  `_free` plus `mio_mesh_add_region`. **Fortran**: `m%regions(...)` and
  `m%add_region(...)`. **WebAssembly**: regions travel on the mesh object, so
  `readMesh` / `writeMesh` / `convert` carry them with no new call.
- **Native CLI**: `info` prints point/cell/side sets, closing its documented
  omission, and `diff` compares regions in the core rather than only in the
  Python shim.

Deferred to Phase 2, explicitly: Exodus blocks/node sets/side sets, MED families
and groups (absorbing `MedInfo`), UNV and Ansys (absorbing `UnvInfo`/`AnsysInfo`),
OpenFOAM boundary patches, XDMF Sets, and VTU/VTP — which have no native set
concept, so a convention has to be chosen rather than invented silently. A
`regions` CLI group and a region-aware `split --by region` are deferred with them.

Two behaviour changes worth calling out. A remapped set now comes back **sorted**
rather than in the permutation's own order, because region entries are canonical
(sorted, de-duplicated) so that membership comparison is exact. And an `*ELSET`
declared inside an `*INCLUDE`d Abaqus file is now carried through the merge; it
used to be dropped under a TODO in `_abaqus.py`.

## v8.0.0 (2026-07-23)

**The WebAssembly build now ships every format the C++ core has.** `cgns`,
`h5m`, `hmf`, `med` and `exodus` — the five that need HDF5 or netCDF — are
readable and writable from `@meshioplusplus/wasm`, as is XDMF's `Format="HDF"`
data path. There is no longer a WASM-specific format gap: 41 formats, 40
readable, 41 writable. `availableFormats()` reports them, and every wasm-written
file was verified to read back correctly through the native h5py/netCDF4-backed
Python package.

- New `build/build-wasm-deps.sh` source-builds a wasm32 **HDF5 1.14.6** and
  **netcdf-c 4.9.3** (pinned, SHA256-checked) into a self-contained prefix.
  CMake still never downloads anything — it only *finds* the result, via
  `CMAKE_FIND_ROOT_PATH`. `build/configure-wasm.sh` runs it automatically the
  first time and gained `--with/--without-hdf5`, `--with/--without-netcdf` and
  `--deps-prefix`. `--without-hdf5` reproduces the old, smaller artifact.
- **Breaking:** writing `.xdmf` from the WASM build now emits an HDF companion
  `<base>.h5` beside the XML instead of inlining the heavy data, because the
  registry's XDMF writer default follows the build — the same rule every native
  build already obeyed. A JS caller must pull **two** files out of the virtual
  filesystem, not one. Reading all three XDMF data formats is unaffected.
- **Breaking:** the published `.wasm` grows from ~2.3 MB to ~5.5 MB (statically
  linked libhdf5 + libnetcdf). Build with `--without-hdf5` for the small
  artifact; the JS API is identical either way.
- MED **cannot write named fields** in the WASM build: the C++ MED writer defers
  a mesh carrying `point_data`/`cell_data` to the Python reference writer, and
  this build has no Python to defer to, so it throws by name. MED geometry,
  `point_tags`/`cell_tags` and families write normally.
- Fixed a latent, silent **stack overflow** in the WASM build, found while
  adding the above: HDF5's and netCDF-4's frames overrun Emscripten's default
  64 KiB stack, which grows down into the static data segment. One Exodus write
  clobbered libc++'s locale facets, after which every `istream >> number` in the
  module — i.e. every ASCII reader, `gmsh`/`obj`/`off`/`vtk` included — trapped.
  The wasm target now links with `-sSTACK_SIZE=4MB` and
  `-sSTACK_OVERFLOW_CHECK=1`, so a recurrence aborts loudly instead of
  corrupting unrelated state.
- CI: `wasm.yml` now also builds and smoke-tests on pull requests touching the
  wasm surface or `src/cpp/` (it was tag-only), with the dependency prefix
  cached.

## v7.16.0 (2026-07-22)

New **`decimate`** operation — reduce a surface mesh's face count by greedy
quadric-error-metric (Garland–Heckbert) edge collapse, preserving shape,
boundaries and features: the resolution-*reducing* inverse of `refine`,
completing the pair. Surface meshes only (`quad`/`polygon` blocks are
triangulated first, so the output is all-triangle with the block structure
kept 1:1); a volume mesh raises by name pointing at `extract_surface`.

- Stopping criteria (exactly one): `ratio` (fraction of faces to keep),
  `target_faces` (absolute, within one collapse), or `max_error` (collapse
  while the cheapest quadric error is below it).
- Placement `optimal` (quadric minimizer, midpoint fallback when the 3x3
  system is ill-conditioned) / `midpoint` / `endpoint`. Float `point_data` is
  blended along the collapsed edge (clamped parameter); integer arrays keep
  the survivor's value.
- Boundary vertices (once-used-edge test) and feature vertices (face normals
  differing by more than `feature_angle`, default 30°) are pinned by default;
  an optional `frozen` mask pins more. The link condition and a normal-flip
  guard reject any collapse that would change topology, create a non-manifold
  edge, or fold the surface — rejections are counted in the report.
- Deterministic: parallel setup with pinned FP order, serial greedy loop with
  a total heap ordering. Output is byte-identical across the three mesh
  backends, thread counts, and the C++/numpy-fallback boundary
  (`test_cpp_matches_python`).
- Exposed everywhere: Python `meshioplusplus.decimate` (with `return_report`),
  C API `mio_decimate` + opaque `mio_decimate_result` (maps + counters;
  `frozen` is a documented flat-ABI gap), Fortran `m%decimate(...)`, WASM
  `decimate(...)` (also a `convertSurfaceOps` pipeline op, so the browser
  viewer's worker can reach it), and a `decimate` verb in both CLIs.
- Docs: `doc/decimate.md`, README "Decimation" section, notebook demo cell.

Also: fixed two path strings in `CLAUDE.md` corrupted by the repository
restructure's `cpp/` → `src/cpp/` rewrite, and its "five version files"
sentence (there are six).

## v7.15.0 (2026-07-22)

New **`isosurface`** operation — the level set of a scalar field: the locus where
a `point_data` array equals a given isovalue, as a mesh one topological dimension
below the cut cells (a 3D volume mesh yields a `triangle`/`quad` surface, a 2D
surface mesh a `line` contour). This is the **data-driven sibling of `slice`**:
slice cuts where `dot(x - origin, normal) = 0`, isosurface where
`f(x) - isovalue = 0`.

- `isosurface(mesh, array, isovalues, component=, record_parent_ids=)`: the field
  must be `point_data` — `cell_data` is piecewise constant, so there is no
  crossing to locate and no level set to draw; naming one raises, pointing at
  `cell_data_to_point_data` (`meshioplusplus data to-point`) as the fix. A
  multi-component array reduces to `component`, or to the row magnitude when that
  is unset.
- **Several isovalues land in one mesh**, cut in ascending order (sorted, exact
  duplicates dropped) and concatenated with the section blocks merged by cell
  type, so a single-isovalue call has exactly slice's block structure. Each
  contour cell carries a Float64 `iso:value` (the level) and an Int64
  `iso:index` (its ordinal) — the latter because `split`'s tag criterion needs an
  integer array, which makes `split --by region --tag iso:index` the
  one-mesh-per-contour recipe.
- **The contoured field reads back as exactly the isovalue** on the cut points;
  every other `point_data` array is interpolated at the crossing (Float64, exact
  for a linear field). The exception is a magnitude-reduced multi-component
  array, where `|lerp(v)| != lerp(|v|)` mathematically and the value stays
  approximate.
- Degeneracy rule, uniform with slice: a node whose value is exactly the isovalue
  counts as being on the **positive** side, so a plateau lying at the isovalue
  emits its boundary once, not twice. Contours are watertight (crossings on
  shared edges dedupe to one node) and wound toward increasing field. An isovalue
  outside the field's range is an empty contour, not an error.
- `record_parent_ids` attaches an Int64 `iso:parent_cell`; each contour cell
  inherits its parent's `cell_data`. Contour points are all new, so
  `point_sets`/`cell_sets` are not carried.
- Output is byte-identical across the three mesh backends, thread counts and the
  C++/numpy boundary. Exposed on every binding surface (pybind `_core`, C API
  `mio_isosurface`, Fortran `m%isosurface`, WASM `isosurface`), as the CLI verb
  `isosurface IN OUT --array NAME --values v1,v2 [--component I]
  [--record-parent-ids]` in both CLIs, and as an operation chip in the browser
  viewer. Docs: [`doc/isosurface.md`](doc/isosurface.md).

Fixed: `src/viewer/package-lock.json` recorded `@meshioplusplus/wasm@7.10.0`
while the package itself was at 7.14.0, which makes `npm ci` hard-fail
(`does not satisfy`) and so broke the viewer jobs in `ci.yml` and `docs.yml`.
The lock's `"../wasm"` version is now part of the version bump — see the
"Version bumps" section of `CLAUDE.md`.

Internal, **no behaviour change**: slice's marching-tetrahedra cutter — the
simplexify, the sign-mask case table, the watertight edge dedup, the winding, the
degeneracy rule and the `point_data`/`cell_data` carry — was hoisted verbatim out
of `operations/slice.cpp` into the shared `detail/marching.hpp` (and out of
`_slice.py` into `_marching.py`), the way `spatial_hash.hpp` was hoisted from
merge in v7.13.0 and `space_filling.hpp` from reorder in v7.6.0. Slice's output is
byte-identical across the hoist and its test suites are unchanged; the only
parameterized rule is the winding, which slice resolves against its fixed plane
normal and isosurface against the local field gradient.

## v7.14.0 (2026-07-22)

New **`slice`** operation — the planar cross-section of a mesh: the actual
intersection of the mesh with a plane, one topological dimension below the cut
cells (a 3D volume mesh yields a `triangle`/`quad` surface, a 2D surface mesh a
`line` mesh). Unlike `crop` (plane mode), which keeps whole cells on one side,
`slice` computes the intersection and lowers the dimension.

- `slice(mesh, origin=, normal=, record_parent_ids=)`: robust marching
  tetrahedra on a simplexified input (every 3D cell becomes a tetra, every 2D
  cell a triangle), so each cell's cross-section is a well-defined convex
  primitive — a hex/wedge section is therefore the union of its simplices'
  sections. The signed-distance crossing `t = d_i/(d_i - d_j)` is computed from
  the sorted edge endpoints and deduped by that edge key, so shared edges yield
  a single output node (the section is watertight). Degeneracy rule: a node on
  the plane (`d == 0`) is classified positive, making the sign mask total, so a
  plane grazing a shared face is emitted exactly once (no double emission);
  collapsed primitives are dropped. Section faces are wound so their Newell
  normal points toward the `+normal` side. Each section cell inherits its
  parent's `cell_data`; `record_parent_ids` attaches an Int64
  `slice:parent_cell`. `point_data` is interpolated at the cut (Float64 output);
  `point_sets`/`cell_sets` are not carried (the section is new topology). Output
  is byte-identical across the three mesh backends, thread counts, and the
  C++-core/numpy-fallback boundary.
- Exposed on every surface: pybind `_core` + the numpy fallback (`slice` shadows
  the built-in only as a module attribute), C API `mio_slice`, Fortran
  type-bound `m%slice`, WASM `slice`, and the `slice IN OUT --origin --normal`
  verb in both CLIs.
- The browser viewer's planar "section" now routes through `slice` (the true
  cross-section) instead of the previous crop-half-space + re-skin.
- Docs: new `doc/slice.md`, CLI reference entry, README "Slicing /
  cross-sections" section, and a notebook demo in both
  `example/python/03_mesh_operations.ipynb` and the C++ mirror.

## v7.13.0 (2026-07-21)

New **`interpolate`** operation — cross-mesh field transfer, the first two-mesh
operation that moves data (diff compares, merge concatenates; neither
resamples): sample a source mesh's data arrays onto a target mesh, returning a
copy of the target with its geometry, connectivity, own data and sets preserved
exactly.

- `interpolate(source, target, method=, arrays=, extrapolate=, default_value=,
  on_conflict=)`: source `point_data` sampled at the target's points,
  `cell_data` by nearest source-cell centroid (always, whatever the method).
  `method="nearest"` (default) copies the nearest source point's value
  bit-for-bit (dtype-preserving); `method="barycentric"` simplexifies the
  source first and interpolates linearly — exact on a linear field, Float64
  output, with `default_value`/`extrapolate` covering target points outside the
  source domain. `on_conflict` is `error`/`overwrite`/`suffix` (`name +
  "_interp"`). Output is byte-identical across the three mesh backends, thread
  counts, and the C++-core/numpy-fallback boundary.
- Exposed on every surface: pybind `_core` + the numpy fallback, C API
  `mio_interpolate` (arrays as `char**` + count, `NULL`/`<= 0` = all
  point_data), Fortran module-level `mio_interpolate`, WASM `interpolate`, and
  the `interpolate SOURCE TARGET OUT` verb in both CLIs.
- New shared `detail/spatial_hash.hpp`: merge's weld bucket grid hoisted
  verbatim (merge's output stays byte-identical) and extended with the
  expanding-shell / box-insert queries interpolate needs.
- Docs: new `doc/interpolate.md`, CLI reference entry, README "Field transfer"
  section, and a notebook demo (`example/03_mesh_operations.ipynb`).

## v7.12.0 (2026-07-21)

The OFF reader/writer (C++ core and Python reference) gain **quad and polygon
face support** — [issue #35](https://github.com/loumalouomega/meshioplusplus/issues/35)
reported that a valid OFF file using quad faces was rejected outright with
"Can only read triangular faces". OFF's own spec allows faces of any vertex
count; only this implementation (and, as it turns out, upstream meshio too)
had hard-coded the triangle-only assumption.

- `read_off` now groups faces by vertex count into `triangle` (3), `quad` (4),
  or `polygon` (else) cell blocks, exactly like the sibling OBJ reader in the
  same file: a run of same-count faces stays in one block until the count
  changes. A leading count below 3 remains a hard `ReadError`.
- `write_off` now writes every `triangle`/`quad`/`polygon` cell block (in mesh
  order); any other cell type is skipped with a warning instead of silently
  dropped. A `polygon` block written by the C++ path must be rectangular; the
  Python reference writer also accepts a ragged `polygon` block.
- New fixtures `tests/meshes/off/cube_example.off` (6 quad faces) and
  `cube_example_as_triangs.off` (the same cube pre-triangulated) back the
  regression tests.

## v7.11.0 (2026-07-21)

The SVG and TikZ writers gain **data-driven colouring**: a `color_by` scalar
field mapped through a built-in colormap to per-face fills, with an optional
colorbar. They already rendered 3D meshes by projecting the extracted skin;
until now every face got the same flat fill, so the vector figures could show
shape but never a field. This is the vector complement to `screenshot()`
(v7.8.0), which covers the raster side.

- **`color_by` / `component` / `cmap` / `vmin` / `vmax` / `nan_color` /
  `colorbar`** on `write_svg` / `write_tikz`, the pybind bindings, the Python
  shims and both CLIs' `convert`. Appended after the camera arguments, in that
  order, everywhere.
  - **Point data** colours a face by the mean of its corner values; **cell
    data** by its owning cell's value. For a projected volume mesh the owner is
    found through the `"surface:parent_cell"` provenance, so a per-cell material
    or quality metric lands on the right skin facet.
  - Multi-component arrays reduce to `component` or to their row magnitude.
  - The range defaults to the finite range of the **drawn faces** — so the
    visible figure spans the whole colorbar. This differs from ParaView, which
    ranges over the whole array. Non-finite values are excluded from the range
    and drawn in `nan_color`.
  - `--color-by NAME [--component I] [--cmap …] [--vmin V] [--vmax V]
    [--nan-color C] [--colorbar]` on `meshioplusplus convert`, in **both** the
    Python and the native CLI, rejected for any output but `.svg`/`.tikz`.
- **Built-in colormaps** (`detail/colormap.{hpp,cpp}` + its `_colormap.py`
  twin, generated by `tools/gen_colormaps.py`): viridis, coolwarm and turbo as
  256-entry uint8 LUTs. **No new dependency** — matplotlib is needed only to
  *regenerate* the tables, never to use them. Storing full 256-entry tables
  rather than interpolated control points is what keeps the C++ and Python
  writers byte-identical: it removes every floating-point interpolation from the
  colour path, leaving a single index expression. The C++ side is
  declaration/definition-split like `projection.hpp`: the header declares the
  API only, and the table data plus function bodies live in `colormap.cpp`.
- **Colouring is a documented gap on the flat bindings.** The C API, Fortran and
  WebAssembly surfaces reach SVG/TikZ through the shared registry, whose
  `(path, mesh)` writer entries structurally cannot carry parameters, so they
  keep emitting the fixed default styling — as with the point/cell-set gaps in
  `diff`/`merge`/`split`.
- **`tools/gen_doc_images.py`** regenerates the committed doc/README figures
  from the bundled sample mesh: the coloured SVGs via this feature, the shaded
  screenshots via `screenshot()`. It complements the PyVista notebooks rather
  than replacing them — those still own the executable demonstrations.
- **Behaviour change (cosmetic):** the SVG writer's *default* stroke width now
  prints as `1` rather than `1.0`. The C++ core always emitted `%g`; the Python
  reference used `str()`, and that single token was the only thing keeping the
  two from being byte-identical. Fixing it lets `tests/test_svg.py` assert full
  byte equality, as `tests/test_tikz.py` already did — which is what now guards
  every fill, the `<style>` block and the colorbar. Rendered output is
  unaffected. Pass `stroke_width=` explicitly to pin an exact value.

With `color_by` unset, SVG and TikZ output is byte-identical to v7.10.0, and the
flat 2D path is physically untouched.

## v7.10.0 (2026-07-21)

The native CLI gets `view` and `screenshot`, restoring verb parity with the
Python one — which v7.9.0 had broken without saying so. The browser viewer
gains click-to-inspect.

- **`view` and `screenshot` in the C++ binary**, backed by
  [Polyscope](https://polyscope.run) (MIT), vendored as a git submodule.
  Volume rendering and slice planes: the things a surface renderer, and so the
  browser viewer, structurally cannot do.
  - **Optional and off by default** (`-DMESHIOPLUSPLUS_WITH_POLYSCOPE=ON`,
    `build/configure.sh --with-polyscope`). The prebuilt release binaries do
    **not** include it and are unchanged — their whole point is being
    dependency-free single files, and Polyscope needs OpenGL, GLFW and X11.
  - It attaches to the **CLI target only, never to the core**. Unlike KaHIP —
    which lives in the core because partitioning is a core operation — viewing
    is not, so `_core`, the C API, the Fortran module and the WebAssembly build
    cannot acquire an OpenGL dependency through it.
  - The verbs exist in **every** build and report the flag when it is off,
    rather than silently not existing.
  - Polyscope vendors its own submodules, so enabling it needs
    `git submodule update --init --recursive`.
- **Click-to-inspect in the browser viewer.** An **Inspect** toggle reports a
  clicked cell's id and type, every cell and point array's value there (all
  components), the nearest vertex, and the originating volume cell for a solid
  — with the picked cell outlined. On click only, never on hover, and disabled
  above two million cells rather than made slow.
- **Fixed:** `CLAUDE.md`'s "documented gaps vs the Python CLI" list omitted the
  two verbs v7.9.0 added, and its verb list was a release out of date.

Internal: `gather_cell_data_onto_surface` moves from the WebAssembly binding
into `operations/surface.cpp`, since the CLI now needs it too. The mesh →
Polyscope mapping is deliberately free of Polyscope headers, so it compiles and
is tested in the default build with no GL.

## v7.9.0 (2026-07-21)

The browser viewer stops being a generic vtk.js app and starts running meshio++
itself. It previously used 6 of the 40 functions the WASM package exposes.

- **Mesh operations in the browser.** Quality, clean, smooth, refine, partition
  and a sectioning cut, applied to the mesh you opened and re-rendered in
  place, with no server and no upload. They compose, each shows as a chip you
  can remove, and **undo is exact**: the worker keeps the original file bytes
  and replays the remaining pipeline, so nothing needs an inverse and nothing
  accumulates rounding.
- **WASM: `convertSurfaceOps`.** One binding applies an operation pipeline and
  writes the renderable surface, all inside C++. Chaining the individual
  operation bindings would route the mesh through the flat JS representation on
  every step and destroy every multi-component array — the thing
  `convertSurface` exists to prevent. An empty pipeline is byte-identical to
  `convertSurface`, so the plain and post-operation display paths cannot drift.
- **Viewer polish**: a DOM colour legend with an editable range replacing
  `vtkScalarBarActor`, an orientation cube whose faces snap the camera,
  surface/wireframe/points, an opacity slider, and Fit/PNG buttons.
- **`viewer/` is now TypeScript**, with `tsc --noEmit` in CI. The shared worker
  protocol means a mismatch between what the client sends and what the worker
  handles is a compile error rather than a runtime surprise.
- **A TikZ icon set** under `icons/`, built with the same `pdflatex` +
  `dvisvgm` pair `logo/build.sh` uses, generated into a typed module so every
  icon reference is checked.
- **The offline page carries results.** `view(backend="browser")` gained
  `quality=True` to bake in per-cell metrics, always embeds the volume mesh's
  geometric statistics (the page renders only the boundary, so it cannot derive
  them), and `color_by` now works instead of warning that it does not.

Fixed:

- Point-data colouring used only half the colormap. `interpolateScalarsBefore
  Mapping` renders a range spanning zero entirely in the warm half in vtk.js
  32.9.0 — measured at 0 blue pixels against 275k red ones on a symmetric
  field — so it is now off, with the evidence recorded beside the setting.
- Surface edges z-fought into faint dashes: the polygon offset was on the body
  mapper, but a negative offset moves *toward* the viewer, so the surface was
  drawn in front of its own wireframe.
- The `kahip` CI job failed at its Python step with `ImportError: libkahip.so`.
  scikit-build-core strips the RPATH from the installed extension so the wheel
  stays relocatable, so the loader needs the prefix on its path;
  [`doc/partition.md`](doc/partition.md) now warns about this for users too.

## v7.8.0 (2026-07-21)

meshio++ can now *show* you a mesh. One entry point, `view()`, with two
backends — a native desktop window and a browser — plus a hosted demo that
doubles as a client-side format converter.

- **`view(mesh, backend=...)` — interactive visualization.** `"polyscope"`
  opens a native window; `"browser"` renders with vtk.js, inline in a notebook
  or in your default browser; `"auto"` picks polyscope when it is installed and
  a display is available. Also `screenshot(mesh, path)`, which renders
  headlessly and so works from CI and a docs build, and `has_viewer()`. New CLI
  verbs `meshioplusplus view` and `meshioplusplus screenshot`.
  - **Polyscope is a Python-only optional dependency**, behind a new
    `[viewer]` extra (`pip install meshioplusplus[viewer]`). It never reaches
    the C++/WASM/C/Fortran core, and the browser backend needs nothing from it.
    A missing install raises naming the command that fixes it.
  - The mesh → renderer mapping is pure and separately tested: no renderer
    import, no display, no mutation of the input. Volume meshes route through
    `convert_cells(simplexify)` only where they must — polyscope holds
    tetrahedra and hexahedra directly, so a hexahedral mesh keeps its
    hexahedra. Every lossy step is reported rather than done quietly.
- **A browser viewer at `viewer/`**, deployed to GitHub Pages alongside the
  docs, that consumes the published `@meshioplusplus/wasm` package exactly as
  an external user would. Drag in any of the ~36 formats the WASM build
  supports, colour by point or cell data with a scalar bar, and convert and
  download to any writable format. Everything runs client-side: no server, no
  upload. The same bundle, built without the WASM, ships in the wheel as the
  offline render path for `view(backend="browser")`.
  - It renders **VTP**, not VTU, and shows a volume mesh by its boundary:
    vtk.js has no unstructured-grid model at all.
- **WASM: `availableFormats()`** returns the reader and writer names this build
  actually supports, so a consumer no longer has to hardcode a table that
  drifts from the build. **`convertSurface()`** reads, extracts the boundary,
  linearizes and writes in one call without materializing a JS mesh — which is
  what keeps multi-component (vector/tensor) arrays, since the flat JS mesh
  representation cannot carry them. Boundary facets also now inherit their
  owning cell's data.
  - Fixed: `index.d.ts` referenced a `MeshMetadata` type it never defined,
    a TS2304 for any consumer without `skipLibCheck`.

## v7.7.0 (2026-07-21)

A new dependency-free mesh operation that improves element *shape* in place — the
counterpart to `quality`, which only measures it, and the complement to `refine`, which
changes resolution without changing shape.

- **`smooth` — relax point coordinates toward their edge-neighbour centroids.** A pure
  coordinate move: connectivity, `cell_data`, `field_data` and `point_data` values all pass
  through unchanged, and the points array keeps its input dtype.
  - Two operators. **Laplacian** (`x <- x + lambda*L(x)`) smooths strongly but shrinks;
    **Taubin** alternates a `+lambda` pass with a larger `-mu` pass and does not, which is
    why it is the default. On a jittered 8x8 quad grid over 40 iterations, Laplacian
    contracts the bounding box by 57% where Taubin contracts it by 3.6%.
  - The neighbour graph is **edge** adjacency, not the element clique, so a structured
    hexahedron block is a fixed point rather than being bevelled toward a sphere.
  - Boundary nodes are pinned by default (`fix_boundary`), as are geometric corners and
    creases (`preserve_features`, `feature_angle_deg`, default 30°), nodes named in an
    optional `frozen` mask, and nodes of blocks whose edge topology is unknown — the
    higher-order family, the VTK-Lagrange types and `custom` — since an unknown
    neighbourhood gives no defined smoothing target.
  - An **inversion guard** (on by default) rejects any move that would turn a valid cell
    inverted, counting the rejections. It is "do no harm", not "preserve the sign": a cell
    that arrives already inverted imposes no constraint, so smoothing can still repair a
    tangled region rather than locking the tangle in.
  - Deterministic by construction: Jacobi updates from the previous pass's positions,
    neighbour sums in ascending node id, no hashing or sorting in the update loop, and the
    boundary set built with `surface.cpp`'s serial-dedup phase split. Output is
    byte-identical across the MESHIO/NATIVE/KRATOS backends and across thread counts.
  - On every surface: Python `smooth`, C API `mio_smooth` (plain mesh plus nullable counter
    out-params, like `mio_clean`), Fortran `m%smooth`, WASM `smooth`, and a `smooth` verb in
    both CLIs.
- **Internal:** the node-adjacency CSR moves to `detail/node_adjacency.hpp` with a
  `Clique | Edge` kind and is now shared with `reorder`, which keeps `Clique` and whose
  output is unchanged.

## v7.6.0 (2026-07-20)

A new mesh operation for domain decomposition, plus the repo's first optional
partitioning dependency.

- **`partition` — decompose a mesh into N balanced parts.** The count-driven
  complement to the criterion-driven `split`. Two methods:
  - **SFC** (always available, the default fallback): cells are ordered along a
    Hilbert space-filling curve of their centroids (the same key transforms
    `reorder` uses, now shared via `detail/space_filling.hpp`) and cut into
    `nparts` contiguous ranges — equal-weight part sizes differ by at most one
    cell, and with `weights=<cell_data>` the cut follows the weight prefix sum.
    Deterministic and byte-identical across mesh backends, thread counts, and
    the C++-core/numpy-fallback boundary (pinned by a test).
  - **KaHIP** (optional, the quality path): the shared-face dual graph is handed
    to KaHIP's serial `kaffpa()` with configurable `imbalance` (default 0.03),
    `mode` (`fast`/`eco`/`strong`, default `eco` — eco/strong carry the edge-cut
    wins) and `seed`. Ported from the Kratos KaHIPApplication
    (KratosMultiphysics/Kratos#14453).
  - `partition_labels` returns just the block-aligned Int64 assignment
    (`partition:part`); `record_ids` attaches `partition:original_point_id`/
    `partition:original_cell_id` to each piece. Pieces keep the input block
    structure 1:1 (unlike `split`), so they recombine into the input.
    `ghost_layers` is reserved (raises) in v1.
  - On every surface: Python `partition`/`partition_labels`, C API
    `mio_partition`/`mio_partition_labels` (opaque result handle with zero-copy
    map getters), Fortran `m%partition`/`m%partition_labels`, WASM
    `partition`/`partitionLabels`, and a `partition` verb in both CLIs
    (`OUT_{part}.vtu` expansion, `--labels-only`).
- **New optional dependency `MESHIOPLUSPLUS_WITH_KAHIP`** (OFF by default; MIT
  like meshio++, so no licensing implication). Located via the new
  `cmake/FindKaHIP.cmake` (`KAHIP_ROOT` prefix / pkg-config) — never vendored or
  auto-downloaded, and only the serial `kaffpa` interface is linked (no
  ParHIP/MPI). The installed library's 32/64-bit index width is detected at
  runtime via `kahip_sizeof_idx()`. Conan `with_kahip`, vcpkg feature `kahip`,
  pip extra `meshioplusplus[kahip]` (the MIT `kahip` wheel, which also gives
  pip-only installs the quality path). Requesting `method="kahip"` without any
  KaHIP backend fails with an error naming the option — never a silent
  downgrade to SFC.

## v7.5.0 (2026-07-20)

A new dependency-free mesh operation that *increases* resolution — the counterpart to
`convert_cells`, which preserves it, and `crop`/`clean`, which reduce it.

- **`refine` — uniform mesh refinement.** Subdivides every cell into congruent children of
  the **same** cell type, one fixed template per type: `line` → 2, `triangle` → 4,
  `quad` → 4, `tetra` → 8, `wedge` → 8, `hexahedron` → 8. `levels=n` applies the templates
  `n` times.
  - New nodes sit at the midpoints of the parent's edges, quad faces and (hexahedron only)
    body, and carry the mean of that entity's corner values for every `point_data` array, so
    a linear field is interpolated exactly. Each parent's `cell_data` row is replicated to
    its children, and block structure is preserved 1:1.
  - **The refined mesh has no hanging nodes.** Mid-edge *and* quad-face-centre nodes are
    shared between every cell touching the entity — only the hexahedron body node is
    per-cell. (Per-cell face centres would leave two adjacent hexahedra referencing distinct
    coincident nodes, splitting the mesh topologically along every interior face.)
  - Children inherit the parent's orientation, so a well-oriented input refines to zero
    newly-inverted cells. Volume is conserved exactly for `line`/`triangle`/`quad`/`tetra`
    always, and for `wedge`/`hexahedron` when the parent is affine — for a general trilinear
    hexahedron the children's volumes do not sum to the parent's, which is a property of the
    geometry rather than of the implementation.
  - The tetrahedron's interior diagonal is fixed at the opposite-edge pair `(0,1)`–`(2,3)`
    for determinism only; being strictly interior, it does not affect conformity — unlike
    `convert_cells`' hex-simplexify diagonal, whose endpoints lie on the boundary.
  - Higher-order cells (linearize first), `pyramid` (whose uniform refinement is 6 pyramids
    + 4 tetrahedra, breaking the same-type contract), and ragged polygon/polyhedron blocks
    raise by name rather than being silently passed through, which would produce hanging
    nodes next to refined neighbours.
  - Output is byte-identical across the MESHIO/NATIVE/KRATOS backends and any thread count:
    the templates are fixed and the new-node numbering comes from a serial dedup pass over a
    parallel-filled buffer, never a concurrent hash insert.

  Exposed as Python `meshioplusplus.refine(mesh, levels=1, record_parent_ids=False)`, C
  `mio_refine` (+ the `mio_refine_result` handle), Fortran `m%refine(levels, ...)`, WASM
  `refine(...)`, and a `refine` verb on both the Python and native CLIs.

- **`wedge18` is now skinnable.** `cell_faces` gained the `wedge18` row (completing the
  family alongside the existing `hexahedron27` and `pyramid14` entries), so
  `extract_surface`, `extract_skin` and `compute_quality` now handle `wedge18` meshes
  instead of warning and skipping them. Mirrored in the pure-Python `_skin.py` twin.

- Internal: the per-type edge tables that `convert_cells`' `elevate` mode and `refine` both
  need are now owned by one shared `detail/cell_subdivision.hpp` (which delegates the 2D
  rows to `detail/cell_edges.hpp`) rather than being transcribed independently. No
  behaviour change.

## v7.4.0 (2026-07-20)

A new dependency-free mesh operation, plus a WebAssembly fix that makes every
*existing* geometry operation reachable from the published package for the first time.

- **`convert_cells` — convert a mesh's element representation.** A mesh operation (not a
  file format), dependency-free and available on every binding surface, with three modes:
  - `linearize` — every higher-order cell becomes its linear base (`tetra10` → `tetra`,
    `hexahedron27` → `hexahedron`, ...), keeping the corner connectivity verbatim and
    pruning the nodes that become unreferenced (connectivity, `point_data` and
    `point_sets` are remapped). Cell count is unchanged, so `cell_data` passes through.
  - `simplexify` — every cell is decomposed into simplices of the same topological
    dimension: quad → 2 triangles, polygon(n) → (n−2)-triangle fan, hexahedron → 6 tetra
    (a canonical Freudenthal fan around the main diagonal 0–6), wedge → 3 tetra,
    pyramid → 2 tetra. The children reuse the parent's own corner nodes, so no points are
    added, and each parent's `cell_data` row is replicated to its children. Higher-order
    input is linearized first. Every emitted simplex is positively oriented for a
    well-oriented input, and volume is conserved — both pinned by tests.
  - `elevate` — linear cells are promoted to their serendipity quadratic counterpart
    (`triangle` → `triangle6`, `hexahedron` → `hexahedron20`, ...), creating one new node
    per unique edge at the edge midpoint with `point_data` set to the endpoint mean. The
    full-Lagrange targets that need face/body centres (`quad9`, `hexahedron27`) are an
    explicit non-goal of this version and raise by name.

  All three modes are idempotent on cells they do not apply to, so they are safe on a
  mixed-order mesh, and output is byte-identical across the MESHIO/NATIVE/KRATOS backends
  and any thread count (the mid-edge numbering is assigned by a serial pass over a
  parallel-filled buffer, never a concurrent hash insert).

  Exposed as Python `meshioplusplus.convert_cells(mesh, mode=..., record_parent_ids=...)`,
  C `mio_convert_cells` (+ the `mio_convert_cells_result` handle), Fortran
  `m%convert_cells(mode, ...)`, WASM `convertCells(...)`, and a `convert-cells` verb on
  both the Python and native CLIs.

- **WebAssembly: every geometry operation is now reachable from `loadMeshioPlusPlus()`.**
  `wasm/src/index.mjs` previously forwarded only file I/O and the five `data_*` operations,
  so `extractSurface`, `extractSkin`, `attachQuality`, `sniffFormat`, `reorder`,
  `computeBandwidth`, `diff`, `meshesEqual`, `merge`, `transform`, `clean`, `cropBbox`,
  `cropPlane`, `split`, `stats` and `meshBackend` were bound in `js_bindings.cpp` but
  unreachable through the package's own API — the same class of bug fixed for the data
  operations in v7.2.1. All of them (plus the new `convertCells`) are now forwarded by the
  wrapper, declared in `wasm/index.d.ts`, and exercised **through the wrapper** by
  `wasm/test/smoke.mjs`, which is what would have caught the original breakage.

## v7.3.0 (2026-07-20)

Three additive I/O performance features. **Default behaviour is unchanged**: `read()`,
`write()` and every existing file are byte-for-byte as before, and the default build gains
no new dependency.

- **Selective / partial reads and `read_metadata()`** — read only what you need.
  - `read(path, points_only=True)` returns geometry (points *and* connectivity) with no data
    arrays; `read(path, arrays=["u", "v"])` returns only the named ones. `arrays=None` means
    every array and `arrays=[]` means none — a deliberate distinction, preserved all the way
    down to the C ABI. Names absent from a file are ignored, not an error.
  - `read_metadata(path)` summarizes a file — point/cell counts, per-block cell types,
    data-array names — without loading the heavy arrays.
  - **VTU, VTP, XDMF and Gmsh** skip the unwanted array bodies outright, and all four plus
    **Gmsh 4.1** have native header-only metadata paths (Gmsh 2.2 declines and falls back, since
    it stores a type per element). Every other format is
    read in full and filtered, which is correct but not faster; `read_metadata`'s
    `fell_back_to_full_read` says which happened, so a summary never implies a saving that did
    not occur. XDMF is the cheapest case (every `<DataItem>` declares its shape, so counts are
    exact without touching any payload, and on the HDF path without opening the `.h5` at all).
  - Honest scope note: for VTU/VTP the file is still read and XML-parsed — pugixml always
    materializes PCDATA. What is skipped is base64 decoding, decompression, allocation and
    byte-swapping. That is a large constant factor, not an asymptotic change.
  - Exposed everywhere: Python `read`/`read_metadata`, pybind `points_only`/`arrays` kwargs,
    C `mio_read_ex` + `mio_read_opts` + the opaque `mio_read_metadata` handle (`mio_read` is
    unchanged), Fortran `mesh%read(..., points_only=, arrays=)` and `mio_read_metadata`, WASM
    `readMeshSelective`/`readMetadata`, and both CLIs (`info --fast`,
    `convert --points-only|--arrays a,b`).
- **Memory-mapped reading** — `detail::FileSource` maps whole files where that pays (POSIX
  `mmap`, Windows `MapViewOfFile`, always buffered under Emscripten), removing a full-file copy
  and the peak-RSS doubling it causes. `Auto` maps regular files at or above 16 MiB
  (`MESHIOPLUSPLUS_MMAP_THRESHOLD` overrides); anything unmappable falls back silently, so
  mapping is advisory and never fails a read. This is a memory-footprint feature more than a
  throughput one. All five whole-file readers use it — gmsh, vtk, ensight, ugrid's ASCII branch
  and openfoam, the last gaining most since it previously paid for two extra full-file copies.
- **Optional zstd and lz4 codecs** for VTK XML block compression — `MESHIOPLUSPLUS_WITH_ZSTD`
  / `_LZ4` (both **off** by default) → `_core.__has_zstd__` / `__has_lz4__`, plus Conan
  `with_zstd`/`with_lz4` and vcpkg `zstd`/`lz4` features. **zlib remains the default
  everywhere.**
  - `lz4` writes `vtkLZ4DataCompressor` in LZ4's raw block format — a real VTK compressor, so
    such files stay readable by VTK and ParaView. Verified both directions against VTK 9.6.
  - `zstd` writes `vtkZSTDDataCompressor`, which is a **meshio++ extension**: VTK ships no ZSTD
    compressor, so ParaView will report an unknown compressor rather than misread the file.
  - A build without a codec reports an error naming the CMake option to enable, and the
    pure-Python reference supports both via the new `codecs` extra
    (`pip install "meshioplusplus[codecs]"`). A file needing a codec available in *neither* is
    genuinely unreadable — a new failure class, and it fails by name.
  - `--codec zlib|lz4|zstd` on both CLIs' `compress`; **rejected** for formats with no block
    codec rather than silently ignored.

## v7.2.1 (2026-07-20)

- Fix: the eight v7.2.0 data-operation bindings were compiled into the WASM module but never forwarded by `wasm/src/index.mjs`'s ergonomic wrapper, so `dataInfo`/`dataCalc`/etc. were unreachable from `loadMeshioPlusPlus()`'s return value (`m.dataInfo is not a function`); also updates `wasm/index.d.ts`'s ambient TypeScript declarations, which had likewise never been extended. No other bindings affected.

## v7.2.0 (2026-07-19)

- **New data operations** — five dependency-free operations acting on a mesh's
  `point_data` / `cell_data` / `field_data` arrays rather than on its geometry, which none of
  them ever modifies:
  - **`data_manage`** (`data_drop` / `data_keep` / `data_rename`): rewrite which arrays a mesh
    carries and under what names. Values, dtypes and shapes are copied verbatim; an unknown key
    raises listing every available key. Phases apply in the order keep → drop → rename.
  - **`point_data_to_cell_data` / `cell_data_to_point_data`**: move data between locations by
    averaging. Point→cell is the mean over each cell's own nodes; cell→point is the mean over
    the incident cells, optionally weighted by each cell's |measure| (area/volume). Output is
    always `float64`, since a mean is not an integer.
  - **`data_calc`**: derive a new array from an elementwise expression over existing arrays at
    the same location. The evaluator is a hand-written tokenizer plus recursive-descent parser
    supporting `+ - * /`, unary minus, parentheses, numeric literals, array names, and
    `abs`/`sqrt`/`min`/`max`/`norm` — **no external parser library and no evaluation of
    arbitrary code**. Identifiers may contain `:` (so `gmsh:physical` works) and backtick
    quoting handles names with spaces.
  - **`data_condition`**: clamp to `[lo, hi]`, normalize onto a target range (default `[0, 1]`),
    or standardize to zero mean / unit standard deviation — per component or by row magnitude.
    For `cell_data` the statistics are computed jointly across all cell blocks.
  - **`data_info`**: a read-only per-array summary (location, dtype, shape, components, entry
    count, min/max/mean whole-array and per component, NaN/inf counts) — the data-side
    complement to the topological `info` and the geometric `stats`.
- **Documented NaN/inf policy**, shared by all five: non-finite values are always excluded from
  every reduction, and `nan_policy` (`ignore` / `replace` / `fail`) decides only what reaches
  the output. `data_info` never raises — it counts them.
- **New nested CLI group `meshioplusplus data <verb>`** with nine verbs (`info`, `rename`,
  `drop`, `keep`, `to-cell`, `to-point`, `calc`, `clamp`, `normalize`), in both the Python CLI
  and the Python-free native binary. This is the project's first two-level subcommand.
- Exposed across every binding surface: pybind (`meshioplusplus.data_*`), the C API
  (`mio_data_*` plus the opaque `mio_data_info` handle), the Fortran module (type-bound
  `data_*` procedures), and WASM (`dataCalc`, `dataInfo`, …). Fortran additionally exports
  `STRBUF_LEN`, which consumers need in order to declare the `keys` out-argument of `split`
  and `data_info`.
- Documented flat-ABI gap: the combined `data_manage` (keep + drop + rename in one call) is not
  exposed over the C ABI; the three primitives compose to the same effect.
- Documented at `doc/data_manage.md`, `doc/data_average.md`, `doc/data_calc.md`,
  `doc/data_condition.md` and `doc/data_info.md`. Not breaking.

## v7.1.0 (2026-07-19)

- **New CLI verbs for the editing and statistics operations**: `meshioplusplus transform`,
  `clean`, `crop`, `split` and `stats` in both the Python CLI and the native binary, with the
  matching README/`doc/cli.md` documentation.

## v7.0.0 (2026-07-19)

- **Breaking: the default branch moved from `main` to `master`** — CI workflow references and
  the README badges were updated accordingly; consumers pinning the branch in a URL need to
  follow.
- **Breaking: `find_package(meshioplusplus)` consumers must require version 7.0**; the packaged
  CMake config version was bumped with the release.
- **New mesh operations**: `transform` (affine transform of point coordinates, with
  translate/scale/rotate/matrix/units builders and optional vector/tensor rotation), `clean`
  (one-pass weld / drop-degenerate / drop-duplicate / remove-orphans), `crop` (subset by
  bounding box or half-space), `split` (partition by cell type, connected component, or integer
  tag), and `stats` (bounding box, centroid, per-cell-type counts, area, signed/unsigned volume,
  inverted-cell count). All are exposed across Python, the C API, Fortran, WASM and both CLIs,
  and share `detail/subset.hpp` for the prune-and-remap step.
- **arm64 support across the release artifacts**: Linux arm64 wheels, native CLI binaries and
  Conan packages are now built natively on GitHub's hosted arm64 runners (no QEMU).
- **Static runtime linking** (`MESHIOPLUSPLUS_STATIC_RUNTIME`, hoisted to a top-level option) so
  the prebuilt CLI binaries carry no `libstdc++`/`libgcc_s` dependency, including the MSVC
  static CRT on Windows.

## v6.9.0 (2026-07-19)

- **New `diff` operation and CLI verb**: compare two meshes with absolute/relative tolerances,
  reporting per-section differences (points, cells, each data map) with max abs/rel error and
  the worst index. An `unordered` mode matches points by proximity via a bucket-grid hash.
  `meshioplusplus diff A B` **exits non-zero when the meshes differ**, for use in CI and
  Makefiles. Named `point_sets`/`cell_sets` are compared in the Python shim only.
- **New `merge` operation and CLI verb**: combine two or more meshes, either concatenating or
  welding coincident points within a tolerance (spatial-hash bucket grid, no O(N²)), with
  `source_mesh_id` tagging, intersection/fill data policies, and optional duplicate-cell
  dropping.

## v6.8.0 (2026-07-19)

- **New `reorder` operation and CLI verb**: renumber nodes and elements by reverse Cuthill–McKee,
  Morton order, or Hilbert order, as a pure permutation that preserves all geometry and data and
  returns the applied permutations. `compute_bandwidth` reports the connectivity bandwidth.
- **New standalone, Python-free native CLI binary** (`meshioplusplus_cli`, installed as
  `meshioplusplus`), built over the shared format registry and the operations layer. Prebuilt
  statically-linked binaries for Linux x86_64/arm64, macOS universal and Windows x86_64 are
  attached to every `v*` tag's GitHub Release.
- **Doxygen API reference** for the C++/C headers, generated in CI and published alongside the
  VitePress site at `/api/`.

## v6.7.0 (2026-07-19)

- **New `extract_surface` operation**: the dimension-aware generalization of `extract_skin` —
  boundary faces of a 3D volume mesh, or boundary edges of a 2D surface mesh — sharing one
  implementation with the skin extractor. Optional `record_parent_ids` attaches the owning input
  cell index as `surface:parent_cell`.
- **New `sniff_format` operation**: content-based format detection from a file's leading bytes,
  wired in as a **read-only** fallback when the extension yields nothing. It returns a format
  only on a confident signature match, never guessing at ambiguous magics.

## v6.6.0 (2026-07-18)

- **New skin extraction** (`meshioplusplus.extract_skin(mesh, linearize=False)`): derives the boundary surface of a 3D volume mesh (tetra/hexahedron/wedge/pyramid + tetra10/hexahedron20/27/wedge15/pyramid13/14 → triangle/quad and their quadratic variants), implementing the face-hashing algorithm of Kratos Multiphysics' `SkinDetectionProcess` (credit: Kratos, BSD — algorithm reimplemented, no code copied). C++ core (uniform mesh API, all three backends) with a byte-equivalent numpy fallback. Points are compacted; `point_data` follows; `cell_data`/`field_data`/sets are dropped. Documented at `doc/extract_skin.md`.
- **Breaking: STL and PLY write the extracted skin of volume meshes by default.** `stl.write`/`ply.write` (and the C++ `write_stl`/`write_ply`, the flat registry, and therefore the WASM/C-API/Fortran surfaces) gain a `skin=True` parameter: a mesh containing supported volume cells now writes its boundary skin (STL triangulates quads; PLY compacts the vertex table) instead of silently dropping the volume cells and writing an empty/vertex-only file. Pre-existing surface blocks are dropped with a warning in that mode. Pass `skin=False` for the previous behavior.
- **SVG and TikZ render 3D meshes**: genuinely non-flat input no longer raises — the boundary skin (or the surface cells of a 3D shell mesh) is projected through an orthographic camera (`azimuth`/`elevation`/`roll` in degrees, default the classic CAD isometric view) and painted back-to-front (painter's algorithm). Flat 2D output is byte-identical to previous releases; the TikZ C++/Python byte-identity guarantee extends to the 3D path.
- **New logo**: the Stanford Bunny (`example/Bunny.stl`, "Stanford Bunny — Digitized!" by MakerBot, CC-BY, thingiverse thing:88208), decimated and rendered through meshio++'s own TikZ 3D machinery with the blue→teal palette (`logo/gen_logo_tikz.py`).

## v6.5.0 (2026-07-18)

- **New `ensight` format** (`.case`/`.geo`): EnSight Gold geometry, read **and** write, ASCII and C-binary (foreign byte order auto-detected on read; the writer ports Kratos's `EnSightOutput` Gold write logic onto the meshio++ mesh API). Multi-part files concatenate into one point array with the owning part tagged as `cell_data["ensight:part"]`; `nsided`/`nfaced` sections read into polygon/polyhedron blocks (write of ragged blocks raises). Backed by the C++ core with a full-fidelity pure-Python fallback.
- **New `vtp` format** (`.vtp`, VTK XML PolyData): read and write of surface meshes (`vertex`/`line`/`triangle`/`quad`/`polygon`), reusing the VTU base64/zlib stack (`binary`/`compression`/`header_type` parameters mirror `vtu`; lzma is Python-only). The shared VTK-XML `<DataArray>` helpers moved into `detail/vtk_xml.hpp` (VTU output is unchanged).
- **New `triangle` format** (`.node`/`.ele`/`.poly`): Shewchuk's Triangle, the 2D analogue of tetgen — `.node`/`.ele` pairs (`triangle`/`triangle6`) plus the `.poly` PSLG (segments as `line` cells; holes/regions skipped). `.node`/`.ele` still default to tetgen; the reader dispatcher falls through to `triangle` for 2D pairs, and writes need `file_format="triangle"` (only `.poly` defaults to triangle).
- All three formats are registered in the shared dispatch registry and therefore reachable from the WASM, C API, and Fortran flat bindings — WASM now ships 36 formats (35 readable / 36 writable). Not breaking.

## v6.4.0 (2026-07-18)

- **New `tikz` format** (`.tikz`): a write-only, 2D TikZ/PGF (LaTeX) writer, the LaTeX counterpart to `svg`. Emits a directly `pdflatex`-compilable `standalone` document by default (`standalone=False` for a bare `tikzpicture` snippet). **Both `svg` and `tikz` are now backed by the C++ core** (`write_svg`/`write_tikz`) with the pure-Python reference kept as fallback, and are registered in the shared dispatch registry, so they are additionally reachable (write-only, fixed default styling) from the WASM, C API, and Fortran flat bindings — WASM now ships 33 writable formats. No change to existing APIs; documented at `doc/formats/tikz.md`/`svg.md`. Not breaking.

## v6.3.2 (2026-07-17)

- **Coverage extended**: the `coverage` job now instruments the C API (`bindings_c/c_api.cpp` + its gtest suite, previously dark) and drops structurally-unreachable code (the non-MESHIO mesh-backend headers, covered by the separate `cpp-tests` matrix, and the generated `single_include/`) from the denominator. New tests lift the darkest paths — a `ply` C++ suite (the one format that had none), UGRID binary/endian flavours, malformed-input `ReadError` cases across ply/ugrid/su2/tetgen/vtk/xdmf/med, and Python public-API error paths + CLI edge cases. Tests/CI only; no API change. Not breaking.

## v6.3.1 (2026-07-17)

- **Coverage CI properly wired up**: the combined Python + C++ `coverage` job now runs (it was gated behind a red `lint` check and had never executed), uploads to Codecov under the `python`/`cpp` flags, and gates PRs (project/patch statuses flipped from informational to blocking). `pyproject.toml` gains `[tool.coverage.run]` (`relative_files`) so `coverage.xml` paths match Codecov's flag filters. Tooling/CI only; no API change. Not breaking.

## v6.3.0 (2026-07-17)

- **Single-header, header-only C++ distribution**: `single_include/meshioplusplus/meshioplusplus.hpp`, generated by `tools/amalgamate.sh` and kept up to date by CI. Declarations are always visible; `#define MESHIOPLUSPLUS_IMPLEMENTATION` before including it in one translation unit pulls in the implementation, pugixml bundled and no external dependencies by default (HDF5/netCDF/zlib/Eigen stay opt-in behind their existing `MESHIOPLUSPLUS_HAS_*` macros). No API change; documented at `doc/single_header.md`. Not breaking.

## v6.2.0 (2026-07-17)

- **New C API and Fortran interface** for HPC consumers: an installable
  `libmeshioplusplus` shared library with a stable pure-C99 header
  (`mio_*` functions; pkg-config + `find_package(meshioplusplus)` support)
  and a modern OO Fortran 2008 module (`type(mio_mesh)` with type-bound
  procedures) on top of it via ISO_C_BINDING. Off by default
  (`MESHIOPLUSPLUS_BUILD_C_API` / `MESHIOPLUSPLUS_BUILD_FORTRAN`, or
  `build/configure.sh --c-api` / `--fortran`); Python/WASM artifacts are
  unaffected. The WASM binding's format-dispatch tables moved into a shared
  core registry (`meshioplusplus/registry.hpp`) used by both flat bindings —
  no JS API change. Not breaking; documented at `doc/c_api.md` and
  `doc/fortran.md`.

## v6.0.0 (2026-07-14)

- **Default C++ parallel backend is now `AUTO`** (prefers OpenMP, then
  STL+TBB, then sequential) instead of `STL` — the old default silently ran
  sequentially on libstdc++ without TBB. Published wheels are now parallel.
  `meshioplusplus._core.__parallel_backend__` reports the active backend. The
  binary-format read/write paths were also optimised (bulk-buffered I/O); output
  is unchanged (byte-identical). **Source builds** should now run
  `git submodule update --init` to fetch the vendored **Eigen** (used for the
  MED transpose); it is optional — builds without it fall back to a plain loop.
- **Project renamed to meshio++** (machine identifier `meshioplusplus`, used
  wherever a literal `+` isn't valid: the Python package/import name, PyPI
  distribution, CLI entry point, C++ namespace, CMake project/targets, and
  build macros). This is a clean break with no compatibility shim — `import
  meshio` / `pip install meshio` no longer refer to this project; use `import
  meshioplusplus` / `pip install meshioplusplus` going forward. The public API
  surface, file formats, and behavior are otherwise unchanged from v5.x.
- Added two new formats, `ansysInp` (Ansys/APDL coded database, `.cdb`/`.inp`) and
  `openfoam` (OpenFOAM polyMesh, read-only), and significantly extended MED/Salome
  support: multi-mesh files (`meshioplusplus.med.read_med_multi`/`write_med_multi`), ragged
  polygon/Voronoi cell blocks, MED 4.1 bitmask metadata, node-orientation fixes,
  quadratic `triangle7`/`quad9`/polygon type support, mesh-level metadata
  (`mesh_name`/`description`/`unit_time`/`unit_coords`), and preserving Gmsh physical
  groups as MED families on write. `meshioplusplus.med.read`/`write` now always use the Python
  implementation (the C++ `meshioplusplus._core.med_read`/`med_write` bindings remain directly
  callable for the narrower/faster behavior). This work originates from
  [Simvia's `meshlane` fork](https://github.com/simvia-tech/meshlane) of meshio,
  contributed by Mariam Kesba, Fatima-Zahra Noussi, and Lucas Sovre, and has been
  brought back into this repository.
- The C++ core is now C++20 (previously C++17) with `std::format`-based logging
  (`MESHIOPLUSPLUS_LOG_LEVEL` env var) and a compile-time-selectable parallel
  backend for hot loops (`-DMESHIOPLUSPLUS_PARALLEL_BACKEND=SEQ|STL|OPENMP|TBB`,
  default STL).

## v5.1.0 (Dec 11, 2021)

- CellBlocks are no longer tuples, but classes. You can no longer iterate over them like
  ```python
  for cell_type, cell_data in cells:
      pass
  ```
  Instead, use
  ```python
  for cell_block in cells:
      cell_block.type
      cell_block.data
  ```

## v5.0.0 (Aug 06, 2021)

- meshio now only provides one command-line tool, `meshio`, with subcommands like
  `info`, `convert`, etc. This replaces the former `meshio-info`, `meshio-convert` etc.

## v4.4.0 (Apr 29, 2021)

- Polygons are now stored as `"polygon"` cell blocks, not `"polygonN"` (where `N` is the
  number of nodes per polygon). One can simply retrieve the number of points via
  `cellblock.data.shape[1]`.

## v4.0.0 (Feb 18, 2020)

- `mesh.cells` used to be a dictionary of the form

  ```python
  {
    "triangle": [[0, 1, 2], [0, 2, 3]],
    "quad": [[0, 7, 1, 10], ...]
  }
  ```

  From 4.0.0 on, `mesh.cells` is a list of tuples,

  ```python
  [
    ("triangle", [[0, 1, 2], [0, 2, 3]]),
    ("quad", [[0, 7, 1, 10], ...])
  ]
  ```

  This has the advantage that multiple blocks of the same cell type can be accounted
  for. Also, cell ordering can be preserved.

  You can now use the method `mesh.get_cells_type("triangle")` to get all cells of
  `"triangle"` type, or use `mesh.cells_dict` to build the old dictionary structure.

- `mesh.cell_data` used to be a dictionary of the form

  ```python
  {
    "triangle": {"a": [0.5, 1.3], "b": [2.17, 41.3]},
    "quad": {"a": [1.1, -0.3, ...], "b": [3.14, 1.61, ...]},
  }
  ```

  From 4.0.0 on, `mesh.cell_data` is a dictionary of lists,

  ```python
  {
    "a": [[0.5, 1.3], [1.1, -0.3, ...]],
    "b": [[2.17, 41.3], [3.14, 1.61, ...]],
  }
  ```

  Each data list, e.g., `mesh.cell_data["a"]`, can be `zip`ped with `mesh.cells`.

  An old-style `cell_data` dictionary can be retrieved via `mesh.cell_data_dict`.
