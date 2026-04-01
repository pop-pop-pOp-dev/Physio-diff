import argparse
import json
import os
from typing import Dict

import numpy as np


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect(root: str) -> Dict[str, Dict[str, float]]:
    metric = {
        "tstr_acc": {},
        "tstr_f1": {},
        "mae": {},
        "psd": {},
        "lang_proto_sep": {},
        "recovered_text_consistency": {},
        "semantic_stability": {},
    }
    if not os.path.exists(root):
        return metric
    for seed_dir in sorted(os.listdir(root)):
        res = _load_json(os.path.join(root, seed_dir, "physio_results.json"))
        if not res:
            continue
        tstr = res.get("tstr", {})
        tfm = res.get("time_freq", {})
        lang = res.get("language_metrics", {})
        metric["tstr_acc"][seed_dir] = float(tstr.get("acc", tstr.get("accuracy", 0.0)) or 0.0)
        metric["tstr_f1"][seed_dir] = float(tstr.get("f1", 0.0) or 0.0)
        metric["mae"][seed_dir] = float(tfm.get("mae", 0.0) or 0.0)
        metric["psd"][seed_dir] = float(tfm.get("psd", 0.0) or 0.0)
        metric["lang_proto_sep"][seed_dir] = float(lang.get("text_prototype_separability", 0.0) or 0.0)
        metric["recovered_text_consistency"][seed_dir] = float(lang.get("recovered_text_consistency", 0.0) or 0.0)
        metric["semantic_stability"][seed_dir] = float(lang.get("semantic_stability", 0.0) or 0.0)
    return metric


def _median_iqr(vals: Dict[str, float]) -> str:
    if not vals:
        return "--"
    arr = np.array(list(vals.values()), dtype=float)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return f"{med:.4f} ({q1:.4f}-{q3:.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown report for architecture ablation matrix.")
    parser.add_argument("--root", default="experiments/ablations/llm_outputs")
    parser.add_argument("--out_path", default="experiments/reports/llm_ablation_report.md")
    parser.add_argument("--full_variant", default="main_competitive")
    args = parser.parse_args()

    variants = [d for d in sorted(os.listdir(args.root)) if os.path.isdir(os.path.join(args.root, d))]
    if args.full_variant not in variants:
        raise RuntimeError(f"Full variant {args.full_variant} not found in {args.root}")
    data = {variant: _collect(os.path.join(args.root, variant)) for variant in variants}

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as handle:
        handle.write("# Architecture-first ablation matrix\n\n")
        handle.write(
            "| Variant | TSTR Acc | TSTR F1 | MAE_feat | PSD | Text Prototype Sep. | Recovered Text Consistency | Semantic Stability |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for variant in variants:
            metric = data[variant]
            handle.write(
                f"| {variant} | {_median_iqr(metric['tstr_acc'])} | {_median_iqr(metric['tstr_f1'])} | {_median_iqr(metric['mae'])} | {_median_iqr(metric['psd'])} | {_median_iqr(metric['lang_proto_sep'])} | {_median_iqr(metric['recovered_text_consistency'])} | {_median_iqr(metric['semantic_stability'])} |\n"
            )
        handle.write(
            "\nText Prototype Sep. measures label-wise language prototype separability; recovered-text consistency evaluates signal-to-text cycle closure; semantic stability measures cross-text lexical consistency.\n"
        )
    print(f"Wrote LLM ablation report to {args.out_path}")


if __name__ == "__main__":
    main()
