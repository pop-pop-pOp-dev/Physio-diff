# Multi-seed runs (median + IQR + Wilcoxon)

This folder contains utilities to run baselines across multiple random seeds under a **unified evaluation protocol** (see `experiments/common/EVAL_PROTOCOL.md`). The report uses **median (Q1–Q3)** and **Wilcoxon signed-rank test** for all methods; no seeds are removed.

## Run multi-seed training/eval

From repo root (`Physio-Diff/`):

```bash
python experiments/sota_runs/run_multi_seed.py --seeds 0,1,2,3,4 --output_root experiments/sota_runs/outputs --physio train
```

Notes:
- `--physio train`: trains Physio-Diff per seed.
- `--physio reuse_ckpt`: uses an existing checkpoint path specified in `configs/best.yaml` (or set `eval.checkpoint_path`).
- `--physio skip`: skips Physio-Diff (useful if you only re-run baselines).

Outputs are written to:
- `experiments/sota_runs/outputs/<method>/seed_<k>/results.json` (baselines)
- `experiments/sota_runs/outputs/physio_diff/seed_<k>/physio_results.json` (Physio-Diff)

## Recompute metrics for existing synthetic outputs (no retraining)

If you already have `synthetic.npz` saved for each seed/method, you can recompute metrics under the unified protocol:

```bash
python experiments/sota_runs/recompute_metrics.py --config configs/best.yaml --root experiments/sota_runs/outputs
```

## Build reports

SOTA-only table:

```bash
python experiments/sota_reports/build_sota_report.py
```

Unified table — median (IQR) + Wilcoxon (all methods, same protocol):

```bash
PYTHONPATH=. python experiments/reports/build_multiseed_report.py
```

## Fill multi-seed gaps (n=5)

If some runs are missing (e.g. Physio-Diff seed_0 only trained, or DDPM/WGAN-GP not run yet):

1. **Physio-Diff seed_0** — generate + evaluate only (reuse `physio_diff_best.pt`, no retrain):

   ```bash
   PYTHONPATH=. python experiments/sota_runs/run_multi_seed.py \
     --methods physio_diff --seeds 0 --physio reuse_ckpt \
     --output_root experiments/sota_runs/outputs
   ```

2. **DDPM** (seeds 0..4):

   ```bash
   PYTHONPATH=. python experiments/sota_runs/run_multi_seed.py \
     --methods ddpm --seeds 0,1,2,3,4 --output_root experiments/sota_runs/outputs
   ```

3. **WGAN-GP** (seeds 0..4):

   ```bash
   PYTHONPATH=. python experiments/sota_runs/run_multi_seed.py \
     --methods wgan_gp --seeds 0,1,2,3,4 --output_root experiments/sota_runs/outputs
   ```

Or run all three in one go (from repo root, with `./data/WESAD` available):

```bash
PYTHONPATH=. bash experiments/sota_runs/run_multiseed_gaps.sh
```

