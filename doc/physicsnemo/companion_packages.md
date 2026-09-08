---
title: Companion packages
description: active_learning, physicsnemo-cfd, physicsnemo-curator and the experimental namespace - what each needs installed.
---

# Companion packages

Four things ship alongside the core library. Two are inside the `physicsnemo` package; two are separate installs.

## `physicsnemo.active_learning` (bundled)

The loop that lets the model choose its own training solves. It has its own page: [Active learning](./active_learning.md).

## `physicsnemo.experimental` (bundled)

Not stable API, but several things here are the *only* implementation of what they do:

| Module | What it holds |
|---|---|
| `experimental.guardrails.embedded` | `OODGuard`, `OODGuardConfig` — out-of-distribution detection on a model's *inputs* |
| `experimental.guardrails.geometry` | `GeometryGuardrail`, `extract_features`, `validate_mesh` — out-of-distribution detection on the *shape* of a triangular surface mesh (2.2) |
| `experimental.uq` | variational GP heads (`variational_gp_head`, `field_variational_gp_head`) for calibrated posterior variance |
| `experimental.peft` | LoRA adapters — `apply`, `merge`, `io` |
| `experimental.models` | `flare`, `geotransolver`, `aerojepa`, `globe`, `healda`, `strata`, `xdeeponet`, `diffusion`, `diffusion_unets` |
| `experimental.nn` | FLARE attention, point tokenizers, RoPE, symmetry layers, 3-D diffusion U-Net blocks |
| `experimental.datapipes`, `experimental.metrics`, `experimental.utils` | HealDA pipeline, diffusion metrics, caching and prefetch |

**One note for anyone doing volumetric diffusion.** `physicsnemo.models.diffusion_unets` is 2-D-image oriented, but `experimental.models.diffusion_unets.DiffusionUNet3D` is a genuine volumetric denoiser: it implements the `physicsnemo.diffusion.base.DiffusionModel` protocol, so it composes with the same preconditioners, losses and samplers, and takes optional volume (`(B, C, D, H, W)`) and vector conditioning.

## `physicsnemo-cfd` (separate install, source only)

```bash
pip install git+https://github.com/NVIDIA/physicsnemo-cfd
```

Not on PyPI. Provides `physicsnemo.cfd`:

- `cfd.postprocessing_tools.metric_registry` — a domain-aware CFD metric registry (relative-L2, drag and lift, physics residuals, UQ metrics);
- `cfd.hybrid_initialization_tools` — blending a prediction into a solver's initial condition;
- `cfd.evaluation` — checkpoint-driven evaluation wrappers, benchmarks, datasets, reports and NIM clients. In the 0.0.3a0 release the benchmark engine (`run_benchmark`, `write_report`) is declared but does not import.

## `physicsnemo-curator` (separate install, git only)

An ETL framework for turning raw simulation output into AI-ready datasets (Zarr stores, VTU grids). A pipeline is `Source -> Filter -> Sink`, run sequentially or over a process pool; its **sinks** ship upstream, and the **source** side is what a solver has to supply.

Its build pulls a Rust toolchain, which is a real installation cost for what it does. Since 2.2 the mesh package's own `to_zarr` writes comparable AI-ready Zarr without it, and — for the columnar case — so does [`write_dataset`](../ml.md), in pure Python over pyarrow or zarr.

## Optional dependencies at a glance

| Package | Needed for | Without it |
|---|---|---|
| `torch` | everything ML | meshio++ still imports; ML entry points raise with an install hint |
| `nvidia-physicsnemo` | everything ML | same |
| `torch_geometric`, `torch_scatter` | the graph models and their batching | those paths raise |
| `torch_sparse` or `dgl` | GraphCast | that recipe is unavailable |
| `onnxruntime` / `onnxruntime-gpu` | ONNX inference | ONNX paths raise |
| `tritonclient` | Triton serving | Triton paths raise |
| `gpytorch` | the GP uncertainty head | that head raises |
| `pyvista` | mesh visualization | those paths raise |
| `nvidia-physicsnemo-cfd` | the CFD metric registry | its metrics are unavailable |
| `physicsnemo-curator` | its Zarr/VTU AI-ready export | that pipeline is unavailable |
| `onnxscript` | torch's ONNX exporter | export raises with an install hint |
| `cupy` | GPU array interchange | falls back to numpy |
| `usd-core` | OpenUSD export | that path raises (meshio++'s own [`usd`](../formats/usd.md) format needs it too, via the `[usd]` extra) |
| `tetgen` | exact boundary recovery in a tetrahedral fill | that backend raises; the default is unaffected |

## In meshio++

The same lazy-gate policy applies here, and it is enforced rather than intended: `import meshioplusplus` succeeds with **none** of the above installed, and the default CI matrix runs the whole test suite that way. Every gated surface raises a named error saying what to install.

Two of those messages are deliberately *not* `pip install meshioplusplus[...]`, and the reason is the same in both cases — the wheel is not pinnable for you. `torch`'s default Linux wheel bundles ~900 MB of CUDA, `nvidia-physicsnemo` hard-depends on a particular torch, and CuPy ships one wheel per CUDA major (`cupy-cuda13x`, `cupy-cuda12x`, ROCm). An extra that resolved to the wrong one would be worse than no extra, so those errors name the package and let you pick the build.

The extras that *do* exist are the ones with a single correct answer: `[arrow]`, `[pandas]`, `[polars]`, `[zarr]`, `[pyvista]`, `[trimesh]`, `[interop]`, `[viewer]`, `[mcp]`, `[kahip]`, `[codecs]`, and `[all]` for the optional dependencies the *formats* need. See [Installation](../installation.md).

Next: [Versions and compatibility](./versions_and_compatibility.md).
