---
title: Data and datapipes
description: How simulation output becomes batched tensors, and the four contracts that bite when it does not.
---

# Data and datapipes

`physicsnemo.datapipes` is the answer to "I have simulation output on disk; how does it become batches?". It is a set of readers, transforms and dataset classes built on torch's `Dataset`/`DataLoader`, plus a handful of ready-made pipelines for specific model families.

## The pieces

**Readers** turn a file into a `TensorDict`: `NumpyReader`, `HDF5Reader`, `VTKReader`, `ZarrReader`, `TensorStoreZarrReader`, `MeshReader`, `DomainMeshReader`.

**Transforms** are composable operations on what a reader produced — `Compose`, `Normalize`, `Scale`, `Translate`, `Rename`, `ConcatFields`, `DropMeshFields`, `SubsampleMesh`, `SubsamplePoints`, `KNearestNeighbors`, `ComputeSDF`, `ComputeNormals`, `ComputeCellCentroids`, `CreateGrid`, `BoundingBoxFilter`, and the mesh augmentations `RandomRotateMesh`, `RandomScaleMesh`, `RandomTranslateMesh`.

**Datasets** hold it together: `DatasetBase`, `IterableDatasetBase`, `MeshDataset`, `MultiDataset` (mixing several series), plus `Collator` variants and physicsnemo's own `DataLoader`.

**Ready-made pipelines** under `datapipes.cae` for external aerodynamics (`domino_datapipe`, `transolver_datapipe`, `mesh_datapipe`, `cae_dataset`), and under `datapipes.gnn` for the published GNN benchmarks (Ahmed body, DrivAerNet, Stokes, vortex shedding, Lagrangian, HydroGraphNet). Also `datapipes.climate` and `datapipes.healpix`.

## Four contracts that bite

These are not obvious from the API, and every one of them has cost somebody a debugging session:

1. **`IterableDatasetBase` is an ABC, not a `torch.utils.data.IterableDataset`.** A bare subclass is rejected by torch's `DataLoader` for having no `len()`. Inherit both.
2. **physicsnemo's `DataLoader` unpacks each item as `(data, metadata)`** — an `(inputs, targets)` tuple would have its targets silently swallowed. Setting `yields_batches = True` passes items through verbatim.
3. **`num_workers > 0` duplicates an iterable stream across workers**, training on every sample twice.
4. **Output buffers must not be reused across yields** — the loader may still be reading the previous one.

## Where mesh data enters

Two routes, and the difference is whether the data touches disk.

![The datapipe stages, and the two places a solve feeds them](/images/physicsnemo/datapipe_pipeline.svg)

*Figure 1: The datapipe stages, and the two places a solve feeds them.*

**Through files.** The solve writes its output as it goes; a dataset reads it back afterwards. Samples are reusable, shuffleable and inspectable — the right default, and the only one that lets you look at a training sample after the fact and ask what went wrong.

**Streaming.** The solve pushes each step into a queue that an iterable dataset drains, and training starts before the solve finishes. Worth it only when the dataset genuinely does not fit on disk, and worth verifying against the file path first: the two must produce byte-identical items, which is a thing to assert rather than assume.

## On augmentation

Rotating a mesh must rotate its vector and rank-2 tensor fields with it. Upstream's transform defaults skip them, which quietly teaches the model that a rotated velocity field is the same velocity field — a model trained that way is not merely less accurate, it has learned a false invariance and will keep it.

meshio++'s [`transform`](../transform.md) takes `rotate_vector_data=True` and does the coherent thing: `R·v` for a trailing-dimension-3 array, `R·A·Rᵀ` for a trailing-dimension-9 one (as of v10.33.0 it rotates `cell_data` blocks of trailing dim 3/9 too, not point_data alone). The *dataset wrapper* around it shipped in the same release: `Augmentation` draws a per-epoch, seeded transform (`random.Random(f"{seed}/{epoch}/{index}")`, a pure function of the triple) and applies the identical draw to a paired input/target so augmenting them independently never teaches the model that a part rotates between one step and the next. See [temporal windows and augmentation](../physicsnemo.md#dataset-augmentation).

## In meshio++

meshio++ owns the step before a datapipe: turning forty-odd solver output formats into arrays that are already the right shape.

| Concern | meshio++ |
|---|---|
| One solve's steps as one ordered value | [`TimeSeries`](../sequences.md) — random access, no mesh cached between reads |
| Many solves catalogued, split and tagged | [`DatasetManifest`](../datasets.md) — a hand-editable JSON, the single source of truth |
| A whole campaign as one training table | [`write_dataset`](../ml.md) — Parquet, Zarr or HDF5, one mesh alive at a time |
| One mesh as one graph sample | [`graph_sample`](../physicsnemo.md) — `pos`, `x`, `y`, `edge_index`, edge features |
| The tensors, on the device, with no file | [`to_torch`, `to_dlpack`](../gpu.md) |
| A mesh in the format `MeshReader` globs for | [`pmsh`](../formats/pmsh.md) and [`zarr`](../formats/zarr.md) — the same two layouts, written in pure numpy |
| A case in the DoMINO/Transolver sample layout | [`cae`](../formats/cae.md) — one `.npz` per case, or `data export-cae` for a whole run |

The streaming invariant is a contract rather than an optimization: `read_sequence` is a generator, `write_dataset` reads one entry at a time, and `TimeSeries` holds only the plan. A dataset of a thousand transient solves never has more than one mesh in memory, which is what makes the file route practical at all.

The `Reader` ABC that `make_reader` returns plugs straight into physicsnemo's own Gen-2 datapipes, so none of the above replaces `physicsnemo.datapipes` — it feeds it.

Next: [Mesh and geometry](./mesh_and_geometry.md).
