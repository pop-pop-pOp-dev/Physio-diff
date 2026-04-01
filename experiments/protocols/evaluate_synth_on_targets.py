import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.data import build_cache, load_config, load_cache, make_splits
from experiments.common.metrics import postprocess_synth, robustness_eval, time_freq_metrics, tstr_eval


def _is_physio_method(method_name: str) -> bool:
    return str(method_name).startswith("physio_diff")


def _load_synth(source_root: str, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, str] | None:
    seed_dir = os.path.join(source_root, method, f"seed_{int(seed)}")
    if _is_physio_method(method):
        synth_path = os.path.join(seed_dir, "synthetic_normalized.npz")
    else:
        synth_path = os.path.join(seed_dir, "synthetic.npz")
    if not os.path.exists(synth_path):
        return None
    payload = np.load(synth_path, allow_pickle=True)
    x = payload["x"].astype(np.float32)
    y = payload["y"].astype(np.int64)
    return x, y, synth_path


def _safe_device(cfg: Dict) -> torch.device:
    requested = str(cfg.get("project", {}).get("device", "cpu")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate_single(
    *,
    target_cfg: Dict,
    target_name: str,
    source_root: str,
    source_name: str,
    method: str,
    seed: int,
    out_root: str,
) -> None:
    cache_path, meta = build_cache(target_cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, target_cfg)
    device = _safe_device(target_cfg)

    loaded = _load_synth(source_root, method, seed)
    if loaded is None:
        print(f"[skip] missing synth -> method={method} seed={seed}")
        return
    synth_x, synth_y, synth_path = loaded

    eval_cfg = target_cfg.get("eval", {})
    match_stats = bool(eval_cfg.get("match_stats", False))
    clip_min = float(eval_cfg.get("clip_min", -5.0))
    clip_max = float(eval_cfg.get("clip_max", 5.0))
    if match_stats or clip_min is not None or clip_max is not None:
        synth_x = postprocess_synth(
            synth_x,
            splits["train"]["x"],
            match_stats=match_stats,
            clip_min=clip_min,
            clip_max=clip_max,
        )

    test_x = np.clip(splits["test"]["x"].astype(np.float32), clip_min, clip_max)
    tstr = tstr_eval(
        synth_x,
        synth_y,
        test_x,
        splits["test"]["y"],
        target_cfg,
        device,
        val_x=np.clip(splits["val"]["x"].astype(np.float32), clip_min, clip_max),
        val_y=splits["val"]["y"],
    )
    robust = robustness_eval(
        synth_x,
        synth_y,
        test_x,
        splits["test"]["y"],
        splits["test"]["acc"],
        target_cfg,
        device,
    )
    tfm = time_freq_metrics(
        test_x,
        synth_x[: len(test_x)],
        int(target_cfg["data"]["target_fs"]),
    )

    out_dir = os.path.join(
        out_root,
        target_name,
        "cross_target",
        "fold_0",
        method,
        f"seed_{int(seed)}",
    )
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "meta": {
            "source_dataset": source_name,
            "target_dataset": target_name,
            "method": method,
            "seed": int(seed),
            "synth_path": synth_path,
        },
        "tstr": tstr,
        "time_freq": tfm,
        "robust": robust,
    }
    result_name = "physio_results.json" if _is_physio_method(method) else "results.json"
    with open(os.path.join(out_dir, result_name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[ok] target={target_name} method={method} seed={seed} -> {out_dir}")


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate source-trained synthetic sets on multiple target datasets (no retraining)."
    )
    parser.add_argument("--source_root", required=True, help="Method/seed synthetic root (from run_multi_seed).")
    parser.add_argument("--source_name", required=True, help="Source dataset label for result metadata.")
    parser.add_argument("--target_configs", required=True, help="Comma-separated target config paths.")
    parser.add_argument("--methods", required=True, help="Comma-separated method names.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds to evaluate.")
    parser.add_argument(
        "--out_root",
        default="experiments/protocols/outputs_cross_target",
        help="Output root for cross-target evaluation results.",
    )
    args = parser.parse_args()

    methods = _parse_csv(args.methods)
    seeds = [int(v) for v in _parse_csv(args.seeds)]
    target_configs = _parse_csv(args.target_configs)

    for cfg_path in target_configs:
        cfg = load_config(cfg_path)
        target_name = str(cfg.get("data", {}).get("dataset_name", os.path.splitext(os.path.basename(cfg_path))[0]))
        for method in methods:
            for seed in seeds:
                _evaluate_single(
                    target_cfg=cfg,
                    target_name=target_name,
                    source_root=args.source_root,
                    source_name=args.source_name,
                    method=method,
                    seed=seed,
                    out_root=args.out_root,
                )


if __name__ == "__main__":
    main()
