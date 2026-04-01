import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch

from experiments.common.data import build_cache, load_cache, load_config, make_splits
from experiments.common.metrics import robustness_eval, tstr_eval


def _median_q1_q3(values: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q1), float(q3)


def _fmt_median_iqr(values: List[float], decimals: int = 4) -> str:
    med, q1, q3 = _median_q1_q3(values)
    return f"{med:.{decimals}f} ({q1:.{decimals}f}--{q3:.{decimals}f})"


def _load_synth(method_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    cand = [
        os.path.join(method_dir, "synthetic_normalized.npz"),
        os.path.join(method_dir, "synthetic.npz"),
    ]
    path = None
    for p in cand:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Missing synthetic npz under: {method_dir}")
    data = np.load(path, allow_pickle=True)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)


def _eval_tstr(cfg: Dict, device: torch.device, synth_x, synth_y, test_x, test_y) -> Dict[str, float]:
    out = tstr_eval(synth_x, synth_y, test_x, test_y, cfg, device)
    acc = float(out.get("acc", out.get("accuracy", 0.0)) or 0.0)
    f1 = float(out.get("macro_f1", out.get("f1", 0.0)) or 0.0)
    return {"acc": acc, "macro_f1": f1}


def _eval_robust(cfg: Dict, device: torch.device, synth_x, synth_y, test_x, test_y, test_acc) -> Dict[str, float]:
    out = robustness_eval(synth_x, synth_y, test_x, test_y, test_acc, cfg, device)
    return {"noisy_acc": float(out["noisy_acc"]), "restored_acc": float(out["restored_acc"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best.yaml")
    parser.add_argument("--out_root", default="experiments/sota_runs/outputs_strong_r2_evalfix")
    parser.add_argument("--methods", default="physio_diff,csdi,cgan")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tstr_epochs", default="15,30,60")
    parser.add_argument("--noise_sigmas", default="0.25,0.5,1.0")
    parser.add_argument("--out_json", default="experiments/reports/sensitivity_results.json")
    parser.add_argument("--out_tex", default="experiments/reports/sensitivity_tables.tex")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    cache_path, meta = build_cache(base_cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, base_cfg)

    device = torch.device(base_cfg["project"]["device"])
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    tstr_epochs = [int(s.strip()) for s in str(args.tstr_epochs).split(",") if s.strip()]
    noise_sigmas = [float(s.strip()) for s in str(args.noise_sigmas).split(",") if s.strip()]

    test_x, test_y, test_acc = splits["test"]["x"], splits["test"]["y"], splits["test"]["acc"]

    tstr_grid = {}
    robust_grid = {}

    for method in methods:
        tstr_grid[method] = {e: {"acc": [], "macro_f1": []} for e in tstr_epochs}
        robust_grid[method] = {s: {"noisy_acc": [], "restored_acc": []} for s in noise_sigmas}

        for seed in seeds:
            method_dir = os.path.join(args.out_root, method, f"seed_{seed}")
            synth_x, synth_y = _load_synth(method_dir)

            for e in tstr_epochs:
                cfg = deepcopy(base_cfg)
                cfg.setdefault("eval", {})
                cfg["eval"]["classifier_epochs"] = int(e)
                r = _eval_tstr(cfg, device, synth_x, synth_y, test_x, test_y)
                tstr_grid[method][e]["acc"].append(r["acc"])
                tstr_grid[method][e]["macro_f1"].append(r["macro_f1"])

            for s in noise_sigmas:
                cfg = deepcopy(base_cfg)
                cfg.setdefault("eval", {})
                cfg["eval"]["noise_acc_std"] = float(s)
                r = _eval_robust(cfg, device, synth_x, synth_y, test_x, test_y, test_acc)
                robust_grid[method][s]["noisy_acc"].append(r["noisy_acc"])
                robust_grid[method][s]["restored_acc"].append(r["restored_acc"])

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump({"tstr": tstr_grid, "robust": robust_grid}, handle, indent=2)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Sensitivity analysis under evaluation hyperparameter changes on WESAD. Values are median (Q1--Q3) over $n=5$ seeds.}"
    )
    lines.append(r"\label{tab:sensitivity}")
    lines.append(r"\resizebox{\linewidth}{!}{")
    cols = "l" + "c" * (len(tstr_epochs) + len(noise_sigmas))
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    hdr = [r"\textbf{Method}"]
    hdr += [rf"\textbf{{TSTR macro-F1 @ {e} ep}} " for e in tstr_epochs]
    hdr += [rf"\textbf{{Robust Restored @ $\sigma$={s:g}}} " for s in noise_sigmas]
    lines.append(" & ".join(hdr) + r" \\")
    lines.append(r"\midrule")
    for method in methods:
        row = [method.replace("_", r"\_")]
        for e in tstr_epochs:
            row.append(_fmt_median_iqr(tstr_grid[method][e]["macro_f1"], decimals=4))
        for s in noise_sigmas:
            row.append(_fmt_median_iqr(robust_grid[method][s]["restored_acc"], decimals=4))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out_tex), exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as handle:
        handle.write(tex)


if __name__ == "__main__":
    main()
