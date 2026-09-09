---
title: Uncertainty and guardrails
description: The uncertainty methods physicsnemo provides, the one variance decomposition they all feed, the metrics that decide whether an error bar is honest, and the guardrails that catch inputs a surrogate should not be trusted on.
---

# Uncertainty and guardrails

A surrogate returns a number. Nothing in that number says whether the input was anything like the training data. Upstream treats this as two separate problems: **uncertainty** (attach a calibrated error bar to every prediction) and **guardrails** (refuse, or warn about, inputs the model was never trained for).

![Three routes to an error bar feeding one variance decomposition](/images/physicsnemo/gp_head.svg)

*Figure 1: Three ways to an error bar, one decomposition, three questions.*

## The methods

| Method | Where | Training cost | Inference cost | Gives | Distance-aware |
|---|---|---|---|---|---|
| **Concrete dropout** | `physicsnemo.nn.ConcreteDropout` | low — a regularizer term | 20 to 30 stochastic passes | epistemic only | no |
| **Ensembles** | no library needed | K trainings (deep), about one (snapshot, cyclic learning rate), free (checkpoint, last K epochs), K inferences (input ensemble over remeshed variants) | K passes | epistemic only | no |
| **`FieldVariationalGPHead`** | `physicsnemo.experimental.uq` (`uq-extras`, gpytorch) | moderate — a variational GP with deep kernel learning on the backbone features | one pass | mean, total variance **and** epistemic variance per point | yes |
| **`VariationalGPHead`** | same | same | one pass | the scalar version, for a drag coefficient rather than a field | yes |
| **Diffusion ensembles** | `physicsnemo.diffusion` | a generative model | M samples | a full conditional distribution | no |

Concrete dropout replaces a hand-tuned dropout rate with a learned one, which is what makes the resulting spread calibrated rather than arbitrary. Its documented trap: **a dropout layer in eval mode is a silent no-op** — forget to switch it back on for the sampling passes and you get zero uncertainty with no error.

The GP heads are closed form: one forward pass returns mean and variance, and the *epistemic* part grows with distance from the training features, which is the property the others lack. Upstream ships the class but not the training recipe, and without the recipe the variance collapses. The recipe is: seed the inducing points from real backbone features after a warm-up, anchor the posterior mean with an auxiliary MSE, ramp the KL term, set `n_train` to the number of training *points* (not geometries) for the field head, keep the features in float32, and use a radial feature norm so the feature norm still carries distance.

## One decomposition

Every method estimates the same thing — the total variance at a point splits into an epistemic part and a noise part:

```
σ²_total(x)  =  σ²_epistemic(x)  +  σ²_noise(x)
```

The epistemic term is what more training data would remove; it is the signal that rises where the surrogate extrapolates. The noise term is the input-dependent scatter the data itself carries (aleatoric noise, or model discrepancy). Dropout and ensembles estimate the first only; GP heads give both from one pass; a mean-variance network gives only the total.

## Is the error bar honest?

An error bar that is too narrow is worse than none. Three questions decide it, with the standardized residual as the common currency — for each observation `y_k` with predicted mean `μ_k` and standard deviation `σ_k`:

```
z_k  =  (y_k − μ_k) / σ_k
```

| Question | Metric | Target |
|---|---|---|
| Is the uncertainty the right size? (calibration) | z-RMS; coverage at 95 % (fraction within 1.96 sigma); negative log predictive density; sharpness (mean sigma, smaller is better *if* calibrated) | z-RMS near 1, coverage near 0.95 |
| Does high uncertainty mark high error? (discrimination) | rank correlation between sigma and the actual error; area under the sparsification-error curve (AUSE) | correlation high, AUSE near 0 |
| Does it grow off-distribution? | growth ratio (sigma_ood / sigma_id) divided by (rmse_ood / rmse_id) | near 1; below 1 means over-confident outside the training family |

Two measured facts worth carrying, because they are typical rather than pathological: a four-member ensemble's nominal 95 % bars covered **50 %** of the truth (textbook over-confidence, and exactly what the calibration metrics exist to catch); and below the training range all members were wrong *together*, so the spread stayed small while the error grew tenfold. The spread's direction was right; its magnitude was not guaranteed.

Calibration and error ranking peak at different checkpoints, so pick the checkpoint for the use you have; and normalizing features away removes the distance cue a GP head needs.

## Guardrails

Uncertainty answers "how sure is the model". A guardrail answers "should the model be asked at all".

- **`OODGuard`** (`physicsnemo.experimental.guardrails.embedded`) is calibrated on the training *inputs* — a kNN density over the model's input features — and checks each inference input against it. Its `check()` only logs, so turning it into a policy is the caller's job.
- **`GeometryGuardrail`** (`physicsnemo.experimental.guardrails.geometry`, new in 2.2) works on the *shape*: it extracts non-invariant descriptors of a triangular surface mesh (translation, rotation and scale are deliberately kept, because a scaled car is a different car) and fits a Gaussian-mixture or polynomial-chaos density with warn and reject percentiles.

Guardrails and uncertainty are complementary, not substitutes: a guard catches the input that is far from everything seen; a calibrated variance tells you how much to trust the answer on the inputs the guard lets through.

## In meshio++

Neither half is implemented, and they are gaps of different kinds.

**Uncertainty** needs no meshio++ support to work — an ensemble's mean and per-point standard deviation are ordinary `point_data` arrays, so `predict` writing a `<field>_std` sibling beside `<field>_pred` is a convention, not a feature. What is genuinely missing is only the training-side plumbing: the trainer builds one model per run and has no ensemble or dropout mode.

**The geometry guardrail is the more interesting case**, because it is the one a mesh library is unusually well placed to fill, and it shipped in v10.34.0. It asks a question about the *shape* rather than about any field: is this geometry like the ones the model was trained on? [`geometry_descriptors`](../geometry_guardrails.md)/`GeometryGuard` build exactly that over three existing operations — [`compute_stats`](../stats.md) for the bounding box, centroid, area and volume, [`compute_quality`](../mesh_quality.md) for the element-shape distribution, [`extract_surface`](../extract_surface.md) for the surface the descriptors are taken from — with the descriptors deliberately kept *non*-invariant (a scaled part is a different part), scored against a diagonal Gaussian fit over the training set's own worst case. It is wired advisorily into the training spec's `Guard` block, the model card, every `predict` path and `dataset_health`.

The nearest thing shipping today is [`dataset_health`](../dashboard.md), which reports per-entry summaries across a whole manifest — enough to *see* an outlier geometry in the dashboard, not enough to refuse one automatically.

Next: [Active learning](./active_learning.md).
