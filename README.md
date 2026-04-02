Physio-Diff (Language-Grounded Mechanistic Diffusion)
======================================================

This repository implements the revised Physio-Diff framework for wearable physiological signal synthesis.
The current version couples mechanistic priors (EDA/BVP physiology) with language-grounded conditioning,
cross-domain semantics, and artifact-aware robustness training.

Core capabilities
-----------------
- Multi-dataset unified data layer:
  - Stress-Predict, WESAD, SWELL-KW, CASE, UBFC-PHYS, MAHNOB-HCI
- Flexible split protocols:
  - Subject split, LOSO, GroupKFold
- Language-grounded generation:
  - Physiological text prototypes
  - Signal-text cycle consistency
  - Semantic alignment and artifact-text conditioning
  - Lightweight language encoder support (`local`, `MiniLM`, `BGE`)
- Physiology-aware objectives:
  - EDA kinematic smoothness + peak retention
  - BVP spectral magnitude + phase-aware alignment
- Fair evaluation and reporting:
  - TSTR utility (accuracy, macro-F1)
  - Unpaired fidelity (MAE_feat, PSD, MMD, FID-like)
  - Realistic robustness suite (gaussian, motion, burst, baseline wander, spike/dropout, time jitter)
  - Multi-seed robust statistics (median/IQR, bootstrap CI, paired Wilcoxon, rank-biserial, BH-FDR)

Project layout
--------------
- `configs/`
  - Main experiment configurations (`best_improved.yaml`, dataset-specific configs, ablation configs)
- `src/data/`
  - Unified dataset registry, manifests, adapters, cache building, normalization/clipping-rate reporting
- `src/models/`
  - Diffusion backbones, mechanistic latent modules, signal-text cycle modules
- `src/text/`
  - Physiological text templates, artifact text, semantic mapping, and text encoders
- `src/losses/`
  - Physiology losses and language-grounding losses
- `src/train/train_diffusion.py`
  - End-to-end training loop (CFG/null-label, EMA, P2, mechanistic branch, curriculum corruptions, domain/text conditioning)
- `src/scripts/run_pipeline.py`
  - Single-run pipeline: train/load checkpoint, generate, evaluate, save `physio_results.json`
- `experiments/`
  - Baselines, ablations, protocol suites, reports, and statistical closure scripts

Data layout
-----------
Expected root structure for each dataset:

```
data/
  <DATASET_NAME>/
    Raw_data/
      S01/
      S02/
      ...
```

Some datasets can also use manifest-based loading (`dataset_manifest.json`) via the unified registry.

Quick start
-----------
1) Install dependencies:
   - `pip install -r requirements.txt`
   - Install PyTorch for your CUDA/CPU environment.
2) Configure paths and hyperparameters in a config file (recommended: `configs/best_improved.yaml`).
3) Run one full pipeline:
   - `python -m src.scripts.run_pipeline --config configs/best_improved.yaml`

Recommended experiment flow
---------------------------
- SOTA multi-seed comparison:
  - `python experiments/sota_runs/run_multi_seed.py --seeds 0,1,2,3,4`
- Build multi-seed report:
  - `python experiments/reports/build_multiseed_report.py`
- Run protocol + closure suite:
  - `python experiments/protocols/run_evidence_closure_suite.py ...`

Notes
-----
- Per-subject Z-score normalization is supported with `hard/soft/winsor/none` clipping modes.
- Clipping ratio is logged in cache metadata (`dataset_meta.json` / `wesad_meta.json`).
- Label-semantics stability is handled primarily at training time (class/prototype/text anchoring); post-hoc label flipping is diagnostic only.
- The new paper-facing method description should treat the language module as a training-time conditioning and supervision path, not as a post-hoc explanation tool.
