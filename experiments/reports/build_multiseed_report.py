"""
Build multi-seed report using a single, fair evaluation rule for all methods:
  - Central tendency: median across seeds (robust to outliers).
  - Spread: IQR (Q3−Q1); displayed as median (Q1–Q3).
  - Significance vs. Physio-Diff: Wilcoxon signed-rank test (paired by seed), same for every baseline.
No seeds are removed; the same protocol applies to every method (see experiments/common/EVAL_PROTOCOL.md §6).
"""
import argparse
import json
import os
from typing import Dict

import numpy as np

from experiments.common.stats import benjamini_hochberg, bootstrap_ci, median_iqr, paired_wilcoxon, rank_biserial


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect(root: str, physio: bool = False) -> Dict[str, Dict[str, float]]:
    metric = {
        "tstr_acc": {},
        "tstr_f1": {},
        "robust_noisy": {},
        "robust_restored": {},
        "mae": {},
        "psd": {},
        "lfhf": {},
        "mmd": {},
        "fid": {},
        "lang_proto_sep": {},
        "lang_cycle": {},
        "lang_semantic": {},
    }
    if not os.path.exists(root):
        return metric

    for seed_dir in sorted(os.listdir(root)):
        res_path = os.path.join(root, seed_dir, "physio_results.json" if physio else "results.json")
        res = load_json(res_path)
        if not res:
            continue
        tstr = res.get("tstr", {})
        robust = res.get("robust", {})
        tfm = res.get("time_freq", {})
        lang = res.get("language_metrics", {})
        metric["tstr_acc"][seed_dir] = float(tstr.get("acc", tstr.get("accuracy", 0.0)) or 0.0)
        metric["tstr_f1"][seed_dir] = float(tstr.get("macro_f1", tstr.get("f1", 0.0)) or 0.0)
        metric["robust_noisy"][seed_dir] = robust.get("noisy_acc", 0.0)
        metric["robust_restored"][seed_dir] = robust.get("restored_acc", 0.0)
        metric["mae"][seed_dir] = tfm.get("mae", 0.0)
        metric["psd"][seed_dir] = tfm.get("psd", 0.0)
        metric["lfhf"][seed_dir] = tfm.get("lf_hf", 0.0)
        if "mmd_rbf" in tfm:
            metric["mmd"][seed_dir] = tfm["mmd_rbf"]
        if "fid_like" in tfm:
            metric["fid"][seed_dir] = tfm["fid_like"]
        metric["lang_proto_sep"][seed_dir] = float(lang.get("text_prototype_separability", 0.0) or 0.0)
        metric["lang_cycle"][seed_dir] = float(lang.get("recovered_text_consistency", 0.0) or 0.0)
        metric["lang_semantic"][seed_dir] = float(lang.get("semantic_stability", 0.0) or 0.0)
    return metric


def median_iqr_str(vals: Dict[str, float]) -> str:
    """Format: median (Q1–Q3). Same rule for all methods."""
    if not vals:
        return "--"
    arr = np.array(list(vals.values()), dtype=float)
    if arr.size == 0:
        return "--"
    med, q1, q3 = median_iqr(vals)
    ci_lo, ci_hi = bootstrap_ci(vals)
    scale = max(abs(med), abs(q1), abs(q3))
    if scale != 0.0 and scale < 1e-3:
        return f"{med:.2e} ({q1:.2e}–{q3:.2e}); 95% CI [{ci_lo:.2e}, {ci_hi:.2e}]"
    return f"{med:.4f} ({q1:.4f}–{q3:.4f}); 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"


def wilcoxon_p(base: Dict[str, float], other: Dict[str, float]) -> str:
    """Paired Wilcoxon signed-rank test (same test for all baselines)."""
    p = paired_wilcoxon(base, other)
    return "--" if p is None else f"{p:.4f}"


def _fmt_q(p: float | None) -> str:
    return "--" if p is None else f"{p:.4f}"


def _is_physio_method(name: str) -> bool:
    return str(name).startswith("physio_diff")


def _label_for_method(name: str) -> str:
    labels = {
        "physio_diff": "Physio-Diff (Full-Stack)",
        "physio_diff_main": "Physio-Diff (Main)",
        "physio_diff_legacy_fullstack": "Physio-Diff (Legacy Full-Stack)",
        "physio_diff_main_additive_cond": "Physio-Diff (Main, Additive Cond)",
        "physio_diff_main_no_multiband": "Physio-Diff (Main, No Multiband)",
        "physio_diff_mechanistic_only": "Physio-Diff (Mechanistic-only)",
        "physio_diff_language_only": "Physio-Diff (Language-only)",
        "physio_diff_no_cycle": "Physio-Diff (-Cycle)",
        "physio_diff_no_semantic_align": "Physio-Diff (-Semantic Align)",
        "physio_diff_no_artifact_text": "Physio-Diff (-Artifact Text)",
        "csdi": "CSDI",
        "ddpm": "DDPM",
        "cgan": "cGAN",
        "wgan_gp": "WGAN-GP",
        "timegan": "TimeGAN",
        "tsgm": "TSGM",
        "tsdiff": "TS-Diff",
    }
    return labels.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-seed report (median + IQR + Wilcoxon).")
    parser.add_argument("--root", default="experiments/sota_runs/outputs", help="Root dir of method/seed outputs")
    parser.add_argument(
        "--primary_method",
        default="physio_diff",
        help="Primary method used as statistical reference.",
    )
    parser.add_argument(
        "--methods",
        default="physio_diff,physio_diff_mechanistic_only,physio_diff_language_only,physio_diff_no_cycle,physio_diff_no_semantic_align,physio_diff_no_artifact_text,timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm",
        help="Comma-separated methods to include.",
    )
    parser.add_argument(
        "--out_path",
        default="experiments/reports/multiseed_report.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write a compact table without optional distribution distances (LF/HF, MMD, FID-like).",
    )
    parser.add_argument(
        "--expected_seeds",
        type=int,
        default=5,
        help="Expected number of seeds per method (used for report text only).",
    )
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=2,
        help="Minimum paired seeds required to compute Wilcoxon p-values.",
    )
    args = parser.parse_args()
    root = args.root
    os.makedirs("experiments/reports", exist_ok=True)
    method_names = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if args.primary_method not in method_names:
        method_names = [args.primary_method] + method_names
    all_metrics = {
        name: collect(
            os.path.join(root, name),
            physio=_is_physio_method(name),
        )
        for name in method_names
    }
    primary = all_metrics.get(args.primary_method)
    if primary is None:
        raise RuntimeError(f"Primary method not found: {args.primary_method}")

    out_path = args.out_path
    raw_p_acc = []
    raw_p_f1 = []
    method_list = []
    for name in method_names:
        if name == args.primary_method:
            continue
        m = all_metrics[name]
        raw_p_acc.append(paired_wilcoxon(primary["tstr_acc"], m["tstr_acc"]))
        raw_p_f1.append(paired_wilcoxon(primary["tstr_f1"], m["tstr_f1"]))
        method_list.append(name)
    adj_p_acc = benjamini_hochberg(raw_p_acc)
    adj_p_f1 = benjamini_hochberg(raw_p_f1)
    idx_lookup = {name: idx for idx, name in enumerate(method_list)}
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("# Multi-seed comparison (median (Q1–Q3) + Wilcoxon)\n\n")
        n_primary = len(primary["tstr_acc"])
        expected = int(args.expected_seeds)
        handle.write(
            "**Evaluation protocol (same for all methods):** "
            f"n={n_primary}/{expected} seeds available, no seeds removed; central tendency = **median**, spread = **IQR (Q1–Q3)**; "
            f"significance vs. `{_label_for_method(args.primary_method)}` = **Wilcoxon signed-rank test** (paired by seed) with BH-adjusted q-values, "
            "plus rank-biserial effect size. "
            "See `experiments/common/EVAL_PROTOCOL.md` §6.\n\n"
        )
        if args.compact:
            handle.write(
                "| Method | TSTR Acc | TSTR F1 | Robust Restored | MAE_feat | PSD | p(TSTR Acc) | p(TSTR F1) |\n"
            )
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            handle.write(
                f"| {_label_for_method(args.primary_method)} | {median_iqr_str(primary['tstr_acc'])} | {median_iqr_str(primary['tstr_f1'])} | {median_iqr_str(primary['robust_restored'])} | {median_iqr_str(primary['mae'])} | {median_iqr_str(primary['psd'])} | -- | -- |\n"
            )
        else:
            handle.write(
                "| Method | TSTR Acc | TSTR F1 | Robust Restored | MAE_feat | PSD | LF/HF | MMD | FID-like | Text Proto Sep. | Cycle Consistency | Semantic Stability | p(TSTR Acc) | p(TSTR F1) |\n"
            )
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            handle.write(
                f"| {_label_for_method(args.primary_method)} | {median_iqr_str(primary['tstr_acc'])} | {median_iqr_str(primary['tstr_f1'])} | {median_iqr_str(primary['robust_restored'])} | {median_iqr_str(primary['mae'])} | {median_iqr_str(primary['psd'])} | {median_iqr_str(primary['lfhf'])} | {median_iqr_str(primary['mmd'])} | {median_iqr_str(primary['fid'])} | {median_iqr_str(primary['lang_proto_sep'])} | {median_iqr_str(primary['lang_cycle'])} | {median_iqr_str(primary['lang_semantic'])} | -- | -- |\n"
            )
        for name in method_names:
            if name == args.primary_method:
                continue
            m = all_metrics[name]
            if args.compact:
                handle.write(
                    f"| {_label_for_method(name)} | {median_iqr_str(m['tstr_acc'])} | {median_iqr_str(m['tstr_f1'])} | {median_iqr_str(m['robust_restored'])} | {median_iqr_str(m['mae'])} | {median_iqr_str(m['psd'])} | {wilcoxon_p(primary['tstr_acc'], m['tstr_acc']) if len(set(primary['tstr_acc'].keys()) & set(m['tstr_acc'].keys())) >= int(args.min_pairs) else '--'} / q={_fmt_q(adj_p_acc[idx_lookup[name]])} | {wilcoxon_p(primary['tstr_f1'], m['tstr_f1']) if len(set(primary['tstr_f1'].keys()) & set(m['tstr_f1'].keys())) >= int(args.min_pairs) else '--'} / q={_fmt_q(adj_p_f1[idx_lookup[name]])} |\n"
                )
            else:
                handle.write(
                    f"| {_label_for_method(name)} | {median_iqr_str(m['tstr_acc'])} | {median_iqr_str(m['tstr_f1'])} | {median_iqr_str(m['robust_restored'])} | {median_iqr_str(m['mae'])} | {median_iqr_str(m['psd'])} | {median_iqr_str(m['lfhf'])} | {median_iqr_str(m['mmd'])} | {median_iqr_str(m['fid'])} | {median_iqr_str(m['lang_proto_sep'])} | {median_iqr_str(m['lang_cycle'])} | {median_iqr_str(m['lang_semantic'])} | {wilcoxon_p(primary['tstr_acc'], m['tstr_acc'])} / r={rank_biserial(primary['tstr_acc'], m['tstr_acc']) or 0.0:.3f} / q={adj_p_acc[idx_lookup[name]] if adj_p_acc[idx_lookup[name]] is not None else '--'} | {wilcoxon_p(primary['tstr_f1'], m['tstr_f1'])} / r={rank_biserial(primary['tstr_f1'], m['tstr_f1']) or 0.0:.3f} / q={adj_p_f1[idx_lookup[name]] if adj_p_f1[idx_lookup[name]] is not None else '--'} |\n"
                )
        handle.write(
            "\nMAE_feat is computed on distribution summary features (unpaired), not sample-wise alignment.\n"
        )

    print(f"Multi-seed report written to {out_path}")


if __name__ == "__main__":
    main()
