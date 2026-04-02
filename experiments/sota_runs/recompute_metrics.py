import argparse
import json
import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from experiments.common.data import build_cache, load_cache, load_config, make_splits
from experiments.common.metrics import (
    postprocess_synth,
    robustness_eval,
    time_freq_metrics,
    tstr_eval,
)


def _load_npz_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (x, y) from an npz, with a clear error for Git LFS pointer files.
    """
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} appears to be a Git LFS pointer. Fetch LFS objects before recomputing metrics."
        )
    data = np.load(path, allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def _align_labels_binary_via_val_tstr(
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: Dict,
    device: torch.device,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """
    Align binary synth labels to real label semantics by choosing whether to flip synth_y
    based on which mapping yields higher TSTR accuracy on the REAL validation set.
    """
    cfg_align = deepcopy(cfg)
    cfg_align.setdefault("eval", {})
    cfg_align["eval"]["classifier_epochs"] = int(cfg_align["eval"].get("align_classifier_epochs", 5))
    if "align_classifier_lr" in cfg_align["eval"]:
        cfg_align["eval"]["classifier_lr"] = float(cfg_align["eval"]["align_classifier_lr"])

    out = tstr_eval(synth_x, synth_y, val_x, val_y, cfg_align, device=device)
    out_flip = tstr_eval(synth_x, 1 - synth_y, val_x, val_y, cfg_align, device=device)
    acc = float(out["accuracy"])
    acc_flip = float(out_flip["accuracy"])
    flipped = bool(acc_flip > acc)
    out_y = (1 - synth_y) if flipped else synth_y
    return out_y, flipped, {"val_acc": acc, "val_acc_flip": acc_flip}


def recompute_one(cfg: Dict, synth_path: str, out_path: str) -> Dict:
    device = torch.device(cfg["project"]["device"])

    cache_path, meta = build_cache(cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, cfg)

    synth_x, synth_y = _load_npz_xy(synth_path)

    match_stats = bool(cfg["eval"].get("match_stats", False))
    clip_min = float(cfg["eval"].get("clip_min", -5.0))
    clip_max = float(cfg["eval"].get("clip_max", 5.0))
    synth_x = postprocess_synth(
        synth_x, splits["train"]["x"], match_stats=match_stats, clip_min=clip_min, clip_max=clip_max
    )
    real_x = np.clip(splits["test"]["x"].astype(np.float32), clip_min, clip_max)
    real_val_x = np.clip(splits["val"]["x"].astype(np.float32), clip_min, clip_max)
    real_val_y = splits["val"]["y"]

    align = bool(cfg["eval"].get("align_labels", False))
    flipped = False
    align_info: Dict[str, float] = {}
    if align and int(cfg.get("model", {}).get("num_classes", 2)) == 2:
        synth_y, flipped, align_info = _align_labels_binary_via_val_tstr(
            synth_x, synth_y, real_val_x, real_val_y, cfg, device=device
        )

    tstr = tstr_eval(synth_x, synth_y, real_x, splits["test"]["y"], cfg, device=device)
    robust = robustness_eval(
        synth_x,
        synth_y,
        real_x,
        splits["test"]["y"],
        splits["test"]["acc"],
        cfg,
        device=device,
    )
    tfm = time_freq_metrics(real_x, synth_x[: len(real_x)], cfg["data"]["target_fs"])
    result = {"tstr": tstr, "time_freq": tfm, "robust": robust}
    if align:
        result["meta"] = {
            "diagnostic_label_alignment": True,
            "label_flipped": flipped,
            **align_info,
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best.yaml")
    parser.add_argument("--root", default="experiments/sota_runs/outputs")
    parser.add_argument(
        "--align_physio_labels",
        action="store_true",
        help="For physio_diff only: run the diagnostic binary label-alignment check before recomputing metrics.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.align_physio_labels:
        cfg.setdefault("eval", {})
        cfg["eval"]["align_labels"] = True

    methods = [
        ("physio_diff", "synthetic_normalized.npz", "physio_results.json"),
        ("timegan", "synthetic.npz", "results.json"),
        ("tsgm", "synthetic.npz", "results.json"),
        ("csdi", "synthetic.npz", "results.json"),
        ("tsdiff", "synthetic.npz", "results.json"),
    ]

    for method, synth_name, res_name in methods:
        method_dir = os.path.join(args.root, method)
        if not os.path.exists(method_dir):
            continue
        for seed_dir in sorted(os.listdir(method_dir)):
            seed_path = os.path.join(method_dir, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            synth_path = os.path.join(seed_path, synth_name)
            out_path = os.path.join(seed_path, res_name)
            if not os.path.exists(synth_path):
                print(f"[skip] missing {synth_path}")
                continue
            if args.dry_run:
                print(f"[dry-run] would recompute {method}/{seed_dir}")
                continue
            print(f"[recompute] {method}/{seed_dir}")
            recompute_one(cfg, synth_path, out_path)


if __name__ == "__main__":
    main()
