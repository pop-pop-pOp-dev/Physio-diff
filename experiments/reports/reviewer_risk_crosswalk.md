# Reviewer Risk Crosswalk (Code-Aligned)

This note maps major rejection risks to manuscript revisions and implementation evidence.

## 1) Dataset scope and evaluation limitations

- **Risk**: single-dataset evidence is not convincing; tiny test subject set; weak robustness realism.
- **Paper-side revision**:
  - Emphasize unified multi-dataset protocol and cross-dataset transfer matrix workflow.
  - Replace single Gaussian-only robustness wording with multi-corruption suite wording.
  - Clarify subject split / LOSO / GroupKFold support.
- **Code evidence**:
  - `src/data/registry.py` (multi-dataset registry)
  - `src/data/datasets.py` (`subject`, `loso`, `groupkfold`)
  - `experiments/protocols/run_protocol_suite.py`
  - `experiments/protocols/run_cross_dataset_matrix.py`
  - `experiments/common/metrics.py::robustness_eval` (gaussian, motion, burst, baseline_wander, spike_dropout, time_jitter)

## 2) Method and loss-design rationality

- **Risk**: label permutation treated as a hack; kinematic smoothing may erase EDA peaks; spectral loss ignores phase; hard clipping may remove valid peaks.
- **Paper-side revision**:
  - State that label semantics are stabilized primarily in training (class/prototype/text anchoring), with label-flip as diagnostic only.
  - Explicitly describe EDA peak-preserving term and BVP phase-aware term.
  - Describe soft/hard/winsor clipping modes and clipping-fraction reporting.
- **Code evidence**:
  - `src/train/train_diffusion.py` (anchor labels, prototype loss, null-label CFG, semantic/text modules)
  - `src/losses/physio_losses.py` (`loss_kin` includes peak envelope; `loss_freq` includes phase term)
  - `src/data/wesad.py` (`clip_mode`, `clip_fraction_total`, `clip_fraction_per_channel`, normalization summary)

## 3) Baseline comparison and significance credibility

- **Risk**: method not clearly superior to older baselines (e.g., TimeGAN); missing significance support.
- **Paper-side revision**:
  - Shift claim style from absolute dominance to statistically grounded competitiveness.
  - Keep paired Wilcoxon + rank-biserial + BH-FDR as default significance language.
  - Clarify that overlapping uncertainty bars imply no strong superiority claim.
- **Code evidence**:
  - `experiments/common/stats.py` (Wilcoxon, rank-biserial, BH-FDR, bootstrap CI)
  - `experiments/reports/build_multiseed_report.py`
  - `experiments/reports/build_multi_dataset_closure_report.py`
  - `experiments/reports/build_cross_dataset_matrix_report.py`

## 4) Writing completeness and engineering practicality

- **Risk**: novelty questioned; missing latency/compute analysis; unclear metric definitions.
- **Paper-side revision**:
  - Keep novelty framing on coupled language-mechanistic stack (not isolated tricks).
  - Explicitly explain the added language module as a training-time conditioning and supervision path backed by compact transformer encoders, not a decorative explanation block.
  - Define `MAE_feat` as unpaired feature-space summary error (not reconstruction MAE).
  - Add/retain latency narrative as step-budget tradeoff and acceleration direction.
- **Code evidence**:
  - `src/models/signal_text_cycle.py`
  - `src/text/text_encoder.py`
  - `src/data/text_annotation.py`
  - `src/losses/language_grounding_losses.py`
  - `experiments/reports/benchmark_latency.py`
  - `experiments/reports/build_latency_table.py`
  - `experiments/common/metrics.py::time_freq_metrics`

## Recommended rebuttal language constraints

- Use: "competitive under paired multi-seed tests" instead of "significantly better" when p/q is not significant.
- Use: "protocol supports multi-dataset transfer and closure analysis" and tie to report scripts.
- Use: "diagnostic" for label-flip checks; reserve "core method" for training-time anchoring mechanisms.
