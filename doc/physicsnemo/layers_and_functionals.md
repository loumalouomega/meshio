---
title: Layers and functionals
description: physicsnemo.nn and physicsnemo.nn.functional - the blocks the architectures are built from, and the GPU operations you can call on mesh data with no model at all.
---

# Layers and functionals

`physicsnemo.models` is built from `physicsnemo.nn`. You rarely need the layers to *use* an architecture, but you need them for three things: building a small custom model that still travels in a `.mdlus`, calling a GPU-optimized operation on mesh data without any model at all, and understanding what a flag such as `use_te` or `mc_dropout` actually switches on.

![The layer families of physicsnemo.nn and the functional families of physicsnemo.nn.functional](/images/physicsnemo/layer_taxonomy.svg)

*Figure 1: The layer and functional families. Highlighted boxes are the ones a mesh pipeline reaches for directly.*

## Two kinds of building block

**Layers** are `torch.nn.Module` subclasses that carry parameters. Their base class, `physicsnemo.nn.Module`, is the same `physicsnemo.Module` the architectures use (see [Core and checkpoints](./core_and_checkpoints.md)): it records its constructor arguments, so a custom block written as one reconstructs itself from a checkpoint like any upstream model.

**Functionals** (`physicsnemo.nn.functional`) are stateless operations on plain tensors, autograd-aware, with Warp or CUDA kernels behind the expensive ones. They take and return tensors, so they compose with mesh arrays through `to_torch` or `to_dlpack` with nothing in between.

## The layer families

Upstream's API index groups them the same way; the names below are the ones you will meet in the model source or in a configuration.

| Family | Representative classes | Where you meet it |
|---|---|---|
| Fourier, FFT and spectral | `SpectralConv1d`/`2d`/`3d`, the FNO and AFNO blocks, `FourierEmbedding`, Fourier feature layers | every neural operator on a grid; `FNO(dimension=..)`'s spectral modes are these layers' `modes` |
| Attention and transformers | `TimmSelfAttention` (`is_causal` since 2.2), `DiTBlock`, the physics attention of Transolver (learned "slices"), FLARE attention, GALE, RoPE embeddings | Transolver, GeoTransolver, FLARE, DiT; the `use_te` flag swaps in TransformerEngine kernels (fp8) |
| Convolutional and U-Net | convolution blocks, U-Net encoder/decoder layers, resampling and interpolation layers, Apex-fused group norm | `UNet`, `SRResNet`, the diffusion U-Nets |
| Graph and geometry | the MeshGraphNet encoder/processor/decoder layers, ball query, point-transformer attention (2.2), neighborhood attention (`natten`) | `MeshGraphNet` and its variants, `FIGConvUNet`, DoMINO's local stencils |
| Fully connected and MLP | `FullyConnected`, MLP layers, weight factorization and weight normalization, SIREN, Pade, DGM and KAN layers | `physicsnemo.models.mlp.FullyConnected`, `MeshGraphKAN`, PINN networks |
| Embeddings and conditioning | `PositionalEmbedding`, `FourierPositionalEmbedding` (2.2), `ConditioningEmbedder`, the modulation embeddings of `ModAFNO` | diffusion conditioning, time conditioning of `ModAFNO` |
| Regularization | `ConcreteDropout`, `drop_path`, `collect_concrete_dropout_losses`, `get_concrete_dropout_rates` | calibrated MC-dropout uncertainty |
| Normalization, activations, pooling | LayerNorm variants, running norm, `get_activation("gelu")`, fused SiLU, `Sin` (2.2), `AttentionPooling`, Gumbel softmax | everywhere; `AttentionPooling` is how a per-point backbone feeds a scalar GP head |
| Specialized and experimental | HEALPix layers, transformer decoders, 3-D diffusion U-Net blocks, point tokenizers, SO(2)/SO(3) equivariant convolutions and norms (`experimental.nn.symmetry`) | weather models, `DiffusionUNet3D`, GeoTransolver's tokenizer |

Around 150 layer classes in total. The practical rule: if you want a small custom model, compose `FullyConnected` or a couple of `SpectralConv` blocks as a `physicsnemo.nn.Module` subclass, save it as a `.mdlus`, and it loads anywhere the upstream models do.

## The functional families

| Family | Functions | Notes |
|---|---|---|
| Neighbors | `knn`, `radius_search` (ball query), batched radius search | Warp-accelerated; upstream measured its ball query at up to 1384x faster and 249x less peak memory than a naive torch implementation |
| Derivatives | `uniform_grid_gradient`/`divergence`/`curl`/`laplacian`, the `rectilinear_grid_` family (2.2), spectral and finite-difference stencils | gradients take a bare scalar field and prepend a derivative axis; divergence and curl take a channels-first vector with channels equal to the spatial rank; the stencils are periodic unless you trim |
| Geometry | `signed_distance_field` (returns `(sdf, hit_points, hit_faces)` since 2.2), `free_form_deform_points`, `displace_points`, `morph_points`, `radial_basis_function_deform_points`, the strain / measure / inversion / bending / volume energies | the Warp backend computes in float32 and is auto-selected whenever CUDA exists |
| Sampling | `farthest_point_sampling` (2.2), `weighted_multinomial`, Poisson-disk sampling, voxelization | subsampling point clouds to a token budget |
| Interpolation and FFT | `grid_to_point_interpolation`, `irfft`/`irfft2` helpers, equivariant ops, regularization and parameterization functionals (`shrink_and_perturb_` lives one level up in `physicsnemo.nn`) | |
| Rendering | differentiable rendering functionals | geometry-from-image problems |

## Two traps worth knowing

The derivative functionals have an **inconsistent layout convention between gradients and vector operators** — a gradient prepends its derivative axis, a divergence expects channels first — so anything calling both has to normalize between them rather than assume one rule.

Several functionals **auto-select Warp on a CUDA machine and compute in float32**. That is fine for training and wrong by ten orders of magnitude when you are checking a float64 field against an analytic derivative, so pin the torch backend for float64 input.

## In meshio++

Several of these functionals have a meshio++ counterpart that needs no GPU, no torch and no framework at all, computed in the C++ core over the mesh you already have:

| PhysicsNeMo functional | meshio++ |
|---|---|
| grid derivatives | [`gradient`](../gradient.md) — Green-Gauss or least squares, on any cell type, plus [`hessian`](../hessian.md) |
| `signed_distance_field` | [`compute_sdf`, `sample_distance`](../sdf.md) — with the angle-weighted pseudonormal sign rule |
| voxelization | [`voxelize`, `grid`](../voxelize.md) — a real hexahedron mesh every writer already understands |
| `grid_to_point_interpolation` | [`interpolate`](../interpolate.md), and [`conservative_interpolate`](../conservative_interpolate.md) when the total has to be preserved |

They are not interchangeable: the meshio++ versions run on the host over an unstructured mesh and are exact for a linear field; the physicsnemo ones run on the device over a dense lattice and are differentiable. Use the framework's when the value has to carry a gradient back to the model, and meshio++'s for everything else — including preparing the features the model is trained on, which is the common case.

The sampling functionals have a host-side counterpart: [`select_points`](../point_budgets.md) is exact farthest-point sampling (plus a lattice-then-farthest `"grid"` method that scales to millions of points, and a random baseline), returning indices in selection order so one budget serves every smaller one by slicing — the same job as `farthest_point_sampling`, on numpy, before anything reaches a device. The neighbour functionals still have **none**: [`edge_index`](../ml.md) builds a graph from mesh edges or the cell dual, never from a radius or a k-nearest-neighbour search; proximity graphs are [on the roadmap](../roadmap.md).

Next: [Data and datapipes](./data_and_datapipes.md).
