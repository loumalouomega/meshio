---
title: Diffusion and deployment
description: Generative models, the sampler/preconditioner split, metrics, optimizers and ONNX export.
---

# Diffusion and deployment

## `physicsnemo.diffusion`

A diffusion model does not predict *an* answer; it samples from a distribution of plausible answers. That is the point: run it several times and the spread between the samples is a calibrated statement about what the data does not determine.

The subpackage splits the problem into three replaceable parts, and knowing the split is most of what you need:

| Part | Answers | In physicsnemo |
|---|---|---|
| **Denoiser** | What network removes the noise? | any model implementing `diffusion.base.DiffusionModel` — the U-Nets, or `DiT` |
| **Preconditioner** | How is noise scaled and the network conditioned on the noise level? | `EDMPrecond`, `VEPrecond`, `VPPrecond`, `iDDPMPrecond`, `EDMPrecondSuperResolution` |
| **Sampler** | How do we walk from noise to a sample? | `deterministic_sampler`, `stochastic_sampler`, `EDMStochasticHeunSolver`, `EulerSolver`, `HeunSolver` |

![Denoiser, preconditioner and sampler as three swappable boxes](/images/physicsnemo/diffusion_split.svg)

*Figure 1: The three replaceable parts, and CorrDiff's regression-plus-residual split.*

Also: `diffusion.guidance` (diffusion posterior sampling — `DataConsistencyDPSGuidance` steers samples toward masked observations, `ModelConsistencyDPSGuidance` toward a differentiable forward model, both at sampling time and without retraining), `diffusion.noise_schedulers` (EDM, VE, VP, iDDPM, one module per class since 2.2), `diffusion.multi_diffusion` (patch-based tiling of a 2-D domain larger than the training resolution), and `diffusion.metrics` with the EDM losses.

**Two APIs coexist in 2.2.** The *protocol* API — `DiffusionModel`, `Predictor` and `Denoiser` protocols, `EDMNoiseScheduler`, `EDMPreconditioner`, `MSEDSMLoss`, `samplers.sample(denoiser, latents, scheduler, solver=)` — is what guidance and multi-diffusion are written against. The *legacy* modules (`samplers.legacy_deterministic_sampler`, `metrics.legacy_losses`, `preconditioners.legacy`) each warn that they will be deprecated in a future release. New work should target the protocol API.

**CorrDiff** is the two-stage recipe that matters for physics: a *regression* model predicts the conditional mean, then a *residual* diffusion model learns what the regression could not. Predicting the mean with a deterministic model is much easier than making diffusion learn it, and the split shows.

## `physicsnemo.metrics`

- `metrics.general` — `crps` and `kcrps` (proper scoring rules for ensembles), `calibration`, `ensemble_metrics`, `entropy`, `histogram`, `mse`, `relative_error`, `power_spectrum`, `wasserstein`, `reduction`.
- `metrics.cae` — CFD-flavoured integrals and quantities.

CRPS is the one to know: it scores a whole *ensemble* against a single observation, rewarding both accuracy and honest spread. An over-confident ensemble scores badly even when its mean is right.

## `physicsnemo.optim`

`CombinedOptimizer` (different optimizers on different parameter groups) and `Muon`. Torch's own optimizers work fine; reach for these only when you need them.

## `physicsnemo.deploy.onnx`

`export_to_onnx_stream` and `run_onnx_inference`. ONNX is the portable artifact: a `.onnx` file plus ONNX Runtime needs **neither** physicsnemo **nor** torch to run, which is what makes it the right thing to hand to a production host.

Two caveats, both of which report success while being wrong. Not every operator exports — the FFTs inside FNO-style models are not supported by the CPU ONNX Runtime, while MLP and convolutional models export and run everywhere. And ONNX Runtime silently falls back to CPU when a CUDA build is missing or a device index does not exist, and silently drops the index when it does (`"cuda:1"` running on device 0). Both need an explicit check rather than trust.

## In meshio++

Nothing here is wired into meshio++, and only one piece of it is a gap worth naming.

`metrics.general.power_spectrum` is the honest measure of whether a super-resolved or generated field has the *right small-scale content*, as opposed to a plausible-looking one with the high wavenumbers smoothed out — pointwise error does not see the difference. meshio++'s [`power_spectrum`](../grids.md) (v10.27.0) computes exactly this on any lattice with nothing but an FFT, azimuthally averaged and Parseval-exact — the natural companion to the grid data path, and the number the `srresnet` superresolution family's own `spectrum_rel_l2` training rows are reported against.

For the rest, the boundary is clean: an ensemble's mean and per-point standard deviation are just two more `point_data` arrays, so anything meshio++ can write can carry them. Write the mean into the field's own name and the spread into a `<field>_std` sibling, and every viewer, every format and every downstream operation treats them as ordinary data — which is the whole reason to keep uncertainty in the mesh rather than in a framework-specific container.

Next: [Uncertainty and guardrails](./uncertainty_and_guardrails.md).
