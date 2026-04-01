import argparse
import json
import os
from typing import Dict, List

import numpy as np

from experiments.common.stats import benjamini_hochberg, bootstrap_ci, paired_wilcoxon, rank_biserial


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _median_iqr(vals: List[float]) -> str:
    if not vals:
        return "--"
    arr = np.asarray(vals, dtype=np.float64)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    d = {
        str(i): float(v)
        for i, v in enumerate(arr.tolist())
    }
    ci_lo, ci_hi = bootstrap_ci(d)
    return f"{med:.4f} ({q1:.4f}--{q3:.4f}); 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"


def _seed_dict(vals: List[float]) -> Dict[str, float]:
    return {f"seed_{i}": float(v) for i, v in enumerate(vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown report for cross-dataset source-target matrix.")
    parser.add_argument("--matrix_dir", default="experiments/reports/cross_dataset_matrix")
    parser.add_argument("--out_md", default="experiments/reports/cross_dataset_matrix_report.md")
    args = parser.parse_args()

    files = [os.path.join(args.matrix_dir, x) for x in sorted(os.listdir(args.matrix_dir)) if x.endswith(".json")]
    rows = []
    pvals_f1 = []
    pvals_acc = []
    pvals_rob = []
    for path in files:
        payload = _load_json(path)
        methods = payload.get("methods", {})
        phys = methods.get("physio_diff")
        if not phys:
            continue
        for method, metrics in methods.items():
            if method == "physio_diff":
                continue
            phys_f1 = _seed_dict(phys.get("tstr_f1", []))
            base_f1 = _seed_dict(metrics.get("tstr_f1", []))
            phys_acc = _seed_dict(phys.get("tstr_acc", []))
            base_acc = _seed_dict(metrics.get("tstr_acc", []))
            phys_rob = _seed_dict(phys.get("robust_restored", []))
            base_rob = _seed_dict(metrics.get("robust_restored", []))
            p_f1 = paired_wilcoxon(phys_f1, base_f1)
            p_acc = paired_wilcoxon(phys_acc, base_acc)
            p_rob = paired_wilcoxon(phys_rob, base_rob)
            pvals_f1.append(p_f1)
            pvals_acc.append(p_acc)
            pvals_rob.append(p_rob)
            rows.append(
                {
                    "pair": os.path.basename(path).replace(".json", ""),
                    "method": method,
                    "physio_acc": _median_iqr(phys.get("tstr_acc", [])),
                    "physio_f1": _median_iqr(phys.get("tstr_f1", [])),
                    "physio_rob": _median_iqr(phys.get("robust_restored", [])),
                    "baseline_acc": _median_iqr(metrics.get("tstr_acc", [])),
                    "baseline_f1": _median_iqr(metrics.get("tstr_f1", [])),
                    "baseline_rob": _median_iqr(metrics.get("robust_restored", [])),
                    "p_acc_raw": p_acc,
                    "p_f1_raw": p_f1,
                    "p_rob_raw": p_rob,
                    "r_acc": rank_biserial(phys_acc, base_acc),
                    "r_f1": rank_biserial(phys_f1, base_f1),
                    "r_rob": rank_biserial(phys_rob, base_rob),
                }
            )
    qvals_f1 = benjamini_hochberg(pvals_f1)
    qvals_acc = benjamini_hochberg(pvals_acc)
    qvals_rob = benjamini_hochberg(pvals_rob)
    for i in range(len(rows)):
        rows[i]["q_f1_fdr"] = qvals_f1[i]
        rows[i]["q_acc_fdr"] = qvals_acc[i]
        rows[i]["q_rob_fdr"] = qvals_rob[i]

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write("# Cross-dataset matrix report\n\n")
        handle.write(
            "| Pair | Baseline | Physio Acc | Base Acc | Physio F1 | Base F1 | Physio Robust | Base Robust | p(Acc)/r/q | p(F1)/r/q | p(Robust)/r/q |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            p_acc = "--" if row["p_acc_raw"] is None else f"{row['p_acc_raw']:.4f}"
            p_f1 = "--" if row["p_f1_raw"] is None else f"{row['p_f1_raw']:.4f}"
            p_rob = "--" if row["p_rob_raw"] is None else f"{row['p_rob_raw']:.4f}"
            q_acc = "--" if row["q_acc_fdr"] is None else f"{row['q_acc_fdr']:.4f}"
            q_f1 = "--" if row["q_f1_fdr"] is None else f"{row['q_f1_fdr']:.4f}"
            q_rob = "--" if row["q_rob_fdr"] is None else f"{row['q_rob_fdr']:.4f}"
            r_acc = "--" if row["r_acc"] is None else f"{row['r_acc']:.3f}"
            r_f1 = "--" if row["r_f1"] is None else f"{row['r_f1']:.3f}"
            r_rob = "--" if row["r_rob"] is None else f"{row['r_rob']:.3f}"
            handle.write(
                f"| {row['pair']} | {row['method']} | {row['physio_acc']} | {row['baseline_acc']} | {row['physio_f1']} | {row['baseline_f1']} | {row['physio_rob']} | {row['baseline_rob']} | {p_acc}/{r_acc}/{q_acc} | {p_f1}/{r_f1}/{q_f1} | {p_rob}/{r_rob}/{q_rob} |\n"
            )


if __name__ == "__main__":
    main()
