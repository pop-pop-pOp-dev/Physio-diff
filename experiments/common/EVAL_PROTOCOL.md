## Evaluation protocol (fair-comparison checklist)

This project includes multiple generative baselines (GANs, diffusion variants, transformers).
To make comparisons publishable (Q1-level), evaluation must be **consistent** across methods.

### 1) Data split (leakage control)

- Use **subject-wise split** (recommended) via `experiments/common/data.py::make_splits`.
- Fix `train_subjects`, `val_subjects`, `test_subjects` in config.

### 2) Synthetic post-processing (must be consistent)

Synthetic signals may require bounding for numerical stability and fair downstream evaluation.
All methods should apply the same post-processing:

- **Clipping**: `clip_min=-5`, `clip_max=5` (default).
- **Optional mean/std alignment**: `match_stats=True` aligns synth per-channel mean/std to a reference set.
  This can improve apparent fidelity, so it must be applied to **all methods** if enabled.

Implementation: `experiments/common/metrics.py::postprocess_synth`.

Recommended default: `match_stats=False`.

### 3) TSTR utility metric (primary)

Train a fixed classifier on synthetic data and evaluate on real **test split**.

Canonical implementation: `experiments/common/metrics.py::tstr_eval`

- Classifier: `SimpleCNN` (1D conv + pooling + GAP)
- Report: Accuracy, Macro-F1

### 4) Robustness (fair protocol)

Robustness should be defined on the **same real test split** for all methods.

Canonical implementation: `experiments/common/metrics.py::robustness_eval`

- Noise injection: accelerometer-scaled Gaussian noise (`inject_acc_noise`)
- Restoration: moving average denoiser (config: `eval.denoise_window`)
- Metric: accuracy of the synthetic-trained classifier on noisy and restored real test data

Note: model-specific denoising (e.g., SDEdit for diffusion) can be reported as an **additional** capability,
but should not be mixed into the main “fair” robustness table when comparing against GANs.

### 5) Fidelity metrics (unpaired generation)

Real and synthetic samples are **not paired**, so per-sample MAE is not meaningful.
Use distribution-level metrics:

- Feature distances (time + frequency features): `mmd_rbf`, `fid_like`
- PSD distance between mean waveforms
- LF/HF ratio (compatibility metric)

Canonical implementation: `experiments/common/metrics.py::time_freq_metrics`

`MAE*` in reports refers to **MAE between feature means**, not sample-wise alignment.

### 6) Multi-seed reporting (fair, robust, no cherry-picking)

To avoid outlier-sensitive means and ensure the same rule for every method:

- **Seeds**: All methods use the same seed set (e.g. 0–4); **no seeds are removed** for any method.
- **Central tendency**: Report **median** across seeds (robust to a single bad seed).
- **Spread**: Report **IQR** (Q3−Q1) so readers see variability without assuming normality.
- **Format**: `median (IQR)` or `median (Q1–Q3)` per metric.
- **Significance**: Paired comparison vs. Physio-Diff using **Wilcoxon signed-rank test** (same test for all baselines; paired by seed; non-parametric, no normality assumption).

This applies identically to Physio-Diff and every baseline: one rule, no double standard.

