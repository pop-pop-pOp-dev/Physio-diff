import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Tuple

from experiments.common.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    median_iqr,
    paired_wilcoxon,
    rank_biserial,
)


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_result_file(seed_dir: str) -> str | None:
    phys = os.path.join(seed_dir, "physio_results.json")
    base = os.path.join(seed_dir, "results.json")
    if os.path.exists(phys):
        return phys
    if os.path.exists(base):
        return base
    return None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_protocol_outputs(root: str) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "tstr_acc": {},
                "tstr_f1": {},
                "robust_restored": {},
                "mae": {},
                "psd": {},
            }
        )
    )
    if not os.path.exists(root):
        return out

    for dataset in sorted(os.listdir(root)):
        dataset_dir = os.path.join(root, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for split in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for fold in sorted(os.listdir(split_dir)):
                if not fold.startswith("fold_"):
                    continue
                fold_dir = os.path.join(split_dir, fold)
                if not os.path.isdir(fold_dir):
                    continue
                for method in sorted(os.listdir(fold_dir)):
                    method_dir = os.path.join(fold_dir, method)
                    if not os.path.isdir(method_dir):
                        continue
                    for seed in sorted(os.listdir(method_dir)):
                        if not seed.startswith("seed_"):
                            continue
                        seed_dir = os.path.join(method_dir, seed)
                        result_file = _pick_result_file(seed_dir)
                        if result_file is None:
                            continue
                        payload = _load_json(result_file)
                        if not payload:
                            continue
                        tstr = payload.get("tstr", {})
                        robust = payload.get("robust", {})
                        tfm = payload.get("time_freq", {})
                        key = f"{fold}__{seed}"
                        out[dataset][method]["tstr_acc"][key] = _safe_float(
                            tstr.get("acc", tstr.get("accuracy", 0.0))
                        )
                        out[dataset][method]["tstr_f1"][key] = _safe_float(
                            tstr.get("macro_f1", tstr.get("f1", 0.0))
                        )
                        out[dataset][method]["robust_restored"][key] = _safe_float(
                            robust.get("restored_acc", 0.0)
                        )
                        out[dataset][method]["mae"][key] = _safe_float(tfm.get("mae", 0.0))
                        out[dataset][method]["psd"][key] = _safe_float(tfm.get("psd", 0.0))
    return out


def _fmt_med_iqr_ci(vals: Dict[str, float]) -> str:
    if not vals:
        return "--"
    med, q1, q3 = median_iqr(vals)
    lo, hi = bootstrap_ci(vals)
    return f"{med:.4f} ({q1:.4f}-{q3:.4f}); 95% CI [{lo:.4f}, {hi:.4f}]"


def _primary_method_name(methods: Dict[str, Dict[str, Dict[str, float]]]) -> str | None:
    for candidate in ("physio_diff_main", "physio_diff", "physio_diff_legacy_fullstack"):
        if candidate in methods:
            return candidate
    return None


def _collect_global_tests(
    data: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    baseline_names: Iterable[str],
) -> Tuple[list[float | None], list[float | None], list[float | None], list[Tuple[str, str, str]]]:
    p_acc: list[float | None] = []
    p_f1: list[float | None] = []
    p_rob: list[float | None] = []
    keys: list[Tuple[str, str, str]] = []
    for dataset, methods in data.items():
        primary_name = _primary_method_name(methods)
        if primary_name is None:
            continue
        for baseline in baseline_names:
            if baseline not in methods:
                continue
            p_acc.append(
                paired_wilcoxon(
                    methods[primary_name]["tstr_acc"],
                    methods[baseline]["tstr_acc"],
                )
            )
            p_f1.append(
                paired_wilcoxon(
                    methods[primary_name]["tstr_f1"],
                    methods[baseline]["tstr_f1"],
                )
            )
            p_rob.append(
                paired_wilcoxon(
                    methods[primary_name]["robust_restored"],
                    methods[baseline]["robust_restored"],
                )
            )
            keys.append((dataset, primary_name, baseline))
    return p_acc, p_f1, p_rob, keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-dataset evidence-closure report.")
    parser.add_argument("--root", default="experiments/protocols/outputs")
    parser.add_argument("--out_md", default="experiments/reports/multi_dataset_closure_report.md")
    parser.add_argument("--out_json", default="experiments/reports/multi_dataset_closure_report.json")
    parser.add_argument(
        "--methods",
        default="timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm",
        help="Baselines compared against the detected primary Physio-Diff method",
    )
    args = parser.parse_args()

    baselines = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    data = _collect_protocol_outputs(args.root)
    p_acc_all, p_f1_all, p_rob_all, test_keys = _collect_global_tests(data, baselines)
    q_acc_all = benjamini_hochberg(p_acc_all)
    q_f1_all = benjamini_hochberg(p_f1_all)
    q_rob_all = benjamini_hochberg(p_rob_all)
    global_q_lookup = {
        key: {
            "q_acc_global": q_acc_all[i],
            "q_f1_global": q_f1_all[i],
            "q_rob_global": q_rob_all[i],
        }
        for i, key in enumerate(test_keys)
    }

    report_rows = []
    for dataset, methods in data.items():
        primary_name = _primary_method_name(methods)
        if primary_name is None:
            continue
        local_p_acc = []
        local_p_f1 = []
        local_p_rob = []
        local_keys = []
        for baseline in baselines:
            if baseline not in methods:
                continue
            local_p_acc.append(paired_wilcoxon(methods[primary_name]["tstr_acc"], methods[baseline]["tstr_acc"]))
            local_p_f1.append(paired_wilcoxon(methods[primary_name]["tstr_f1"], methods[baseline]["tstr_f1"]))
            local_p_rob.append(
                paired_wilcoxon(methods[primary_name]["robust_restored"], methods[baseline]["robust_restored"])
            )
            local_keys.append(baseline)
        q_acc_local = benjamini_hochberg(local_p_acc)
        q_f1_local = benjamini_hochberg(local_p_f1)
        q_rob_local = benjamini_hochberg(local_p_rob)
        q_local_lookup = {
            b: {
                "q_acc_local": q_acc_local[i],
                "q_f1_local": q_f1_local[i],
                "q_rob_local": q_rob_local[i],
            }
            for i, b in enumerate(local_keys)
        }

        for baseline in baselines:
            if baseline not in methods:
                continue
            phys = methods[primary_name]
            base = methods[baseline]
            pair_key = (dataset, primary_name, baseline)
            row = {
                "dataset": dataset,
                "primary_method": primary_name,
                "baseline": baseline,
                "physio_tstr_acc": _fmt_med_iqr_ci(phys["tstr_acc"]),
                "base_tstr_acc": _fmt_med_iqr_ci(base["tstr_acc"]),
                "physio_tstr_f1": _fmt_med_iqr_ci(phys["tstr_f1"]),
                "base_tstr_f1": _fmt_med_iqr_ci(base["tstr_f1"]),
                "physio_robust": _fmt_med_iqr_ci(phys["robust_restored"]),
                "base_robust": _fmt_med_iqr_ci(base["robust_restored"]),
                "p_acc": paired_wilcoxon(phys["tstr_acc"], base["tstr_acc"]),
                "p_f1": paired_wilcoxon(phys["tstr_f1"], base["tstr_f1"]),
                "p_rob": paired_wilcoxon(phys["robust_restored"], base["robust_restored"]),
                "r_acc": rank_biserial(phys["tstr_acc"], base["tstr_acc"]),
                "r_f1": rank_biserial(phys["tstr_f1"], base["tstr_f1"]),
                "r_rob": rank_biserial(phys["robust_restored"], base["robust_restored"]),
            }
            row.update(q_local_lookup.get(baseline, {}))
            row.update(global_q_lookup.get(pair_key, {}))
            report_rows.append(row)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump({"root": os.path.abspath(args.root), "rows": report_rows}, handle, indent=2)

    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write("# Multi-dataset evidence and statistical-closure report\n\n")
        handle.write(
            "| Dataset | Primary | Baseline | Primary Acc | Base Acc | Primary F1 | Base F1 | Primary Robust | Base Robust | p(Acc)/r/q_local/q_global | p(F1)/r/q_local/q_global | p(Rob)/r/q_local/q_global |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in report_rows:
            p_acc = "--" if row["p_acc"] is None else f"{row['p_acc']:.4f}"
            p_f1 = "--" if row["p_f1"] is None else f"{row['p_f1']:.4f}"
            p_rob = "--" if row["p_rob"] is None else f"{row['p_rob']:.4f}"
            r_acc = "--" if row["r_acc"] is None else f"{row['r_acc']:.3f}"
            r_f1 = "--" if row["r_f1"] is None else f"{row['r_f1']:.3f}"
            r_rob = "--" if row["r_rob"] is None else f"{row['r_rob']:.3f}"
            q_acc_l = "--" if row.get("q_acc_local") is None else f"{row['q_acc_local']:.4f}"
            q_f1_l = "--" if row.get("q_f1_local") is None else f"{row['q_f1_local']:.4f}"
            q_rob_l = "--" if row.get("q_rob_local") is None else f"{row['q_rob_local']:.4f}"
            q_acc_g = "--" if row.get("q_acc_global") is None else f"{row['q_acc_global']:.4f}"
            q_f1_g = "--" if row.get("q_f1_global") is None else f"{row['q_f1_global']:.4f}"
            q_rob_g = "--" if row.get("q_rob_global") is None else f"{row['q_rob_global']:.4f}"
            handle.write(
                f"| {row['dataset']} | {row['primary_method']} | {row['baseline']} | {row['physio_tstr_acc']} | {row['base_tstr_acc']} | {row['physio_tstr_f1']} | {row['base_tstr_f1']} | {row['physio_robust']} | {row['base_robust']} | {p_acc}/{r_acc}/{q_acc_l}/{q_acc_g} | {p_f1}/{r_f1}/{q_f1_l}/{q_f1_g} | {p_rob}/{r_rob}/{q_rob_l}/{q_rob_g} |\n"
            )
        handle.write(
            "\nq_local 为每个数据集内多重比较 (BH-FDR)；q_global 为跨数据集全体比较后的 BH-FDR，二者共同用于统计闭环。\n"
        )
    print(f"Wrote markdown report to {args.out_md}")
    print(f"Wrote json report to {args.out_json}")


if __name__ == "__main__":
    main()
