#!/usr/bin/env bash
# Fill multi-seed gaps:
#   1) Physio-Diff seed_0: generate + evaluate only (reuse existing checkpoint).
#   2) DDPM: run seeds 0..4 → outputs/ddpm/seed_*/results.json
#   3) WGAN-GP: run seeds 0..4 → outputs/wgan_gp/seed_*/results.json
#
# Run from repo root (Physio-Diff/) with data available at ./data/WESAD:
#   PYTHONPATH=. bash experiments/sota_runs/run_multiseed_gaps.sh

set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "=== 1/3 Physio-Diff seed_0 (reuse checkpoint, gen+eval only) ==="
python experiments/sota_runs/run_multi_seed.py \
  --methods physio_diff --seeds 0 --physio reuse_ckpt \
  --output_root experiments/sota_runs/outputs

echo "=== 2/3 DDPM seeds 0..4 ==="
python experiments/sota_runs/run_multi_seed.py \
  --methods ddpm --seeds 0,1,2,3,4 \
  --output_root experiments/sota_runs/outputs

echo "=== 3/3 WGAN-GP seeds 0..4 ==="
python experiments/sota_runs/run_multi_seed.py \
  --methods wgan_gp --seeds 0,1,2,3,4 \
  --output_root experiments/sota_runs/outputs

echo "Done. Regenerate report: python experiments/reports/build_multiseed_report.py"
