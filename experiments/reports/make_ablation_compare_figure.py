import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _median_iqr(vals: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    return float(med), float(q1), float(q3)


def _vivid_palette(n: int) -> List[str]:
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    reps = (n + len(base) - 1) // len(base)
    return (base * reps)[:n]



def _collect(root: str, method: str, metric_path: List[str], seeds: List[int]) -> List[float]:
    out: List[float] = []
    for s in seeds:
        p = os.path.join(root, method, f"seed_{s}", "physio_results.json")
        if not os.path.exists(p):
            continue
        j = _read_json(p)
        cur = j
        for k in metric_path:
            cur = cur[k]
        out.append(float(cur))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Make ablation comparison figure from existing results.")
    parser.add_argument(
        "--root",
        default="experiments/sota_runs/outputs_strong_r2_evalfix",
        help="Root containing physio_diff and ablation outputs.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seed list.")
    parser.add_argument(
        "--out_pdf",
        default="pictures/summary/fig_ablation_compare.pdf",
        help="Output PDF path (relative to Physio-Diff repo root).",
    )
    parser.add_argument(
        "--also_copy_to",
        default="",
        help="Optional second output PDF path (absolute or relative to Physio-Diff repo root).",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    root = os.path.join(repo_root, args.root)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    variants = [
        ("Full-Stack", "physio_diff"),
        ("Mechanistic-only", "physio_diff_mechanistic_only"),
        ("Language-only", "physio_diff_language_only"),
        ("-Cycle", "physio_diff_no_cycle"),
        ("-Semantic Align", "physio_diff_no_semantic_align"),
        ("-Artifact Text", "physio_diff_no_artifact_text"),
    ]
    metrics = [
        ("TSTR Acc", ["tstr", "accuracy"]),
        ("TSTR F1", ["tstr", "f1"]),
        ("Robust Restored", ["robust", "restored_acc"]),
    ]

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    colors = _vivid_palette(len(variants))
    for ax, (mname, mpath) in zip(axes, metrics, strict=True):
        xs = np.arange(len(variants))
        meds: List[float] = []
        q1s: List[float] = []
        q3s: List[float] = []
        for _, vdir in variants:
            vals = _collect(root, vdir, mpath, seeds=seeds)
            if not vals:
                meds.append(np.nan)
                q1s.append(np.nan)
                q3s.append(np.nan)
                continue
            med, q1, q3 = _median_iqr(vals)
            meds.append(med)
            q1s.append(q1)
            q3s.append(q3)
        for i in range(len(xs)):
            if not np.isfinite(meds[i]):
                continue
            yerr = [[meds[i] - q1s[i]], [q3s[i] - meds[i]]]
            ax.errorbar(
                [xs[i]],
                [meds[i]],
                yerr=yerr,
                fmt="o",
                capsize=3.0,
                lw=1.2,
                color=colors[i],
                ecolor=colors[i],
                markerfacecolor=colors[i],
                markeredgecolor="black",
                markeredgewidth=0.6,
            )
            ax.annotate(
                f"{q1s[i]:.3f}–{q3s[i]:.3f}",
                (xs[i], q3s[i]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([v[0] for v in variants], rotation=20, ha="right")
        ax.set_title(mname)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_pdf = os.path.join(repo_root, args.out_pdf)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    if args.also_copy_to:
        out2 = args.also_copy_to
        if not os.path.isabs(out2):
            out2 = os.path.join(repo_root, out2)
        os.makedirs(os.path.dirname(out2), exist_ok=True)
        fig.savefig(out2)
    plt.close(fig)


if __name__ == "__main__":
    main()
