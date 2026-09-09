---
title: Models
description: The 25 architecture families under physicsnemo.models, and how to pick one from the shape of your data.
---

# Models

`physicsnemo.models` holds 25 architecture families. You do not need to know them all; you need to know **which shape of data you have**, because that is what picks the family.

## Pick by data shape

| Your data is | Use | What meshio++ gives you |
|---|---|---|
| A regular grid, same resolution in and out | FNO, AFNO, UNet | `grid`, `voxelize` and `write_vti` produce the lattice |
| A coarse grid in, a fine grid out | SRResNet (`srrn`) | the [grid data path](../grids.md), a paired `Target` in the manifest, and the `srresnet` family in `TrainSpec` |
| An unstructured mesh with connectivity | MeshGraphNet family | `graph_sample` — nodes, edges, features, targets |
| An unordered point cloud | Transolver, FIGConvNet, FLARE | `feature_matrix` gives the node table, `select_points` the [token budget](../point_budgets.md) |
| A CAD surface plus a volume (external aero) | DoMINO | `extract_surface` plus `compute_sdf` as a nodal field |
| A time series of states | RNN, `mesh_reduced` | `TimeSeries` and the `target_offset` pairing |
| Particles with trajectories | MeshGraphNet, VFGN | mesh edges only; proximity graphs are a gap |
| A distribution, not a single answer | diffusion U-Nets, DiT | nothing yet |
| The globe | GraphCast, Pangu, FengWu, DLWP | nothing; no mesh counterpart at all |

The same choice as a chart — start from the shape of one sample and follow the arrows:

![A decision chart from the shape of one sample to an architecture family](/images/physicsnemo/model_choice.svg)

*Figure 1: What one sample looks like decides the family. Everything else is detail.*

## The families

### Neural operators on grids

| Family | Classes | Notes |
|---|---|---|
| `fno` | `FNO` | The reference neural operator. `dimension=1..4`; the 4-D case is a spatio-temporal block operator |
| `afno` | `AFNO`, `ModAFNO` | Fourier operator with an adaptive token mixer. `ModAFNO` takes a *timestep* input, so simulated time conditions it |
| `dpot` | `DPOTNet` | Denoising pre-trained operator transformer, 2-D and 3-D |
| `unet` | `UNet` | A plain 3-D convolutional U-Net |
| `pix2pix` | `Pix2Pix` | Image-to-image translation |
| `srrn` | `SRResNet` | Super-resolution residual network — coarse in, fine out |

### Graphs and meshes

| Family | Classes | Notes |
|---|---|---|
| `meshgraphnet` | `MeshGraphNet`, `BiStrideMeshGraphNet`, `HybridMeshGraphNet`, `MeshGraphKAN` | Encode–process–decode on an edge graph. The variants add a multiscale hierarchy, proximity "world" edges, and KAN layers |
| `mesh_reduced` | `Mesh_Reduced`, `Sequence_Model` | Reduce a mesh, then learn the dynamics in the reduced space with temporal attention |
| `graphcast` | `GraphCastNet` | The weather architecture, on an icosahedral grid |
| `vfgn` | `VFGNLearnedSimulator` and its encode/process/decode parts | Virtual Foundry GraphNet, for sintering and additive manufacturing |

### Point clouds and transformers

| Family | Classes | Notes |
|---|---|---|
| `transolver` | `Transolver` | Attention over learned "physics slices" of a point cloud |
| `geotransolver` | `GeoTransolver` | Geometry-aware successor |
| `figconvnet` | `FIGConvUNet` | Factorized implicit grids; per-point fields plus a scalar (drag-style) head |
| `flare` | `FLARE` | Experimental attention successor (also under `experimental.models`) |
| `domino` | `DoMINO` | External aerodynamics: CAD surface plus volume, with pretrained checkpoints |

### Generative

| Family | Classes | Notes |
|---|---|---|
| `diffusion_unets` | `SongUNet`, `SongUNetPosEmbd`, `DhariwalUNet`, `CorrDiffRegressionUNet`, `UNet`, `StormCastUNet` | The denoiser backbones. **2-D image oriented** |
| `dit` | `DiT` | Diffusion transformer — the denoiser to reach for on non-image data |
| `topodiff` | `TopoDiff` | Diffusion for topology optimization |

A **volumetric** 3-D denoiser exists too, under `physicsnemo.experimental.models.diffusion_unets.DiffusionUNet3D` — see [Companion packages](./companion_packages.md).

### Sequences and weather

`rnn` (`One2ManyRNN`, `Seq2SeqRNN`), `dlwp` (`DLWP`), `dlwp_healpix` (`HEALPixUNet`, `HEALPixRecUNet`), `pangu` (`Pangu`), `fengwu` (`Fengwu`), `swinvrnn` (`SwinRNN`).

### The plain one

`mlp` (`FullyConnected`) — a multilayer perceptron. It is the right first model far more often than it looks: if your input is a handful of case parameters and your output is a field, you want this, not a neural operator.

### Worth knowing about, on nobody's critical path

| Family | Class | What it would bring |
|---|---|---|
| `dpot` | `DPOTNet` | a PDE *foundation model* (AFNO mixing, pretrained across equation families) to fine-tune on your own grids |
| `topodiff` | `TopoDiff` | generative topology optimization with constraint channels, on compliance data |
| `pix2pix` | `Pix2Pix`, `Pix2PixUnet` | a plain convolutional image-to-image translator; fits any grid pipeline mechanically |
| `experimental.xdeeponet` | `DeepONet` | branch (parameters) plus trunk (coordinates) operator learning — parameters in, field at the mesh nodes out, without a POD basis |
| `experimental.globe` | `GLOBE` | boundary-driven elliptic problems, from named boundary meshes |
| `experimental.aerojepa` | `AeroJEPA` | self-supervised pretraining on geometry alone, before any labels exist |
| `experimental.strata`, `experimental.healda` | `Strata`, `VideoHealDA` | weather emulation on the sphere and HEALPix data assimilation; the assimilation idea matters for digital twins, the API is calendar-shaped |
| `pangu`, `fengwu`, `swinvrnn`, `dlwp_healpix` | as named | global weather architectures with no mesh counterpart |

## Building blocks

`physicsnemo.nn` holds the layers the models are made of — around 150 of them. Two are worth knowing by name:

- `ConcreteDropout` — dropout with a *learned* rate, which makes MC-dropout uncertainty estimates meaningfully calibrated instead of arbitrary;
- `physicsnemo.nn.functional.derivatives` — differential operators used by the physics-informed path.

## In meshio++

Two families are wired end to end: `MeshGraphNet`, through `TrainSpec`'s `Model.Name: "meshgraphnet"`, `graph_sample` for the tensors and `predict` for the write-back, and `SRResNet`, through `Model.Name: "srresnet"` over the [grid data path](../grids.md)'s coarse/fine pairs. The first choice is not arbitrary — a mesh with connectivity is the shape meshio++ natively holds, and PyG batches ragged graphs of different sizes natively, which is what makes a dataset of differently-sized meshes trainable with no padding convention of its own.

Everything else on the chart above is reachable by hand: `feature_matrix` gives you the node table any point-cloud model wants, `select_points` reduces it to a token budget, `to_torch`/`to_dlpack` move it to the device without a file round trip, and the model is then ordinary PyTorch. The rest of the *dataset* half has since shipped too — [`proximity_graph`](../proximity_graphs.md) (v10.31.0) builds the radius/kNN graphs a particle method needs, and [`make_window`/`iter_windows`/`rollout`](../physicsnemo.md#temporal-windows-and-rollout) (v10.33.0) give a sequence model its history and its autoregressive evaluation loop.

Next: [Data and datapipes](./data_and_datapipes.md) — feeding them.
