import argparse
import os
import subprocess
import sys
from copy import deepcopy

from experiments.common.data import build_cache, load_config
from experiments.sota_runs.run_multi_seed import METHODS


def _parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _resolve_loso_fold_count(base_cfg: dict, fallback: int) -> int:
    subjects = [str(s) for s in base_cfg.get("data", {}).get("subjects", []) if str(s).strip()]
    if subjects:
        return max(1, len(set(subjects)))
    try:
        _, meta = build_cache(base_cfg)
        meta_subjects = [str(s) for s in meta.get("subjects", []) if str(s).strip()]
        if meta_subjects:
            return max(1, len(set(meta_subjects)))
    except Exception:
        pass
    return max(1, int(fallback))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run subject-wise CV protocols across datasets and methods.")
    parser.add_argument("--configs", default="configs/best.yaml")
    parser.add_argument("--methods", default="physio_diff")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output_root", default="experiments/protocols/outputs")
    parser.add_argument("--split_strategy", default=None, choices=[None, "subject", "loso", "groupkfold"])
    parser.add_argument("--cv_n_splits", type=int, default=None)
    parser.add_argument("--folds", default=None, help="Comma-separated fold indices; defaults to all folds.")
    parser.add_argument("--run_cross_matrix", action="store_true")
    parser.add_argument("--cross_matrix_targets", default="")
    parser.add_argument("--cross_matrix_source_roots", default="")
    parser.add_argument("--llm_variants", default="", help="Optional comma-separated LLM ablation variants.")
    parser.add_argument("--build_closure_report", action="store_true")
    parser.add_argument("--closure_report_out_md", default="experiments/reports/multi_dataset_closure_report.md")
    parser.add_argument("--closure_report_out_json", default="experiments/reports/multi_dataset_closure_report.json")
    args = parser.parse_args()

    configs = _parse_csv_list(args.configs)
    methods = _parse_csv_list(args.methods)
    seeds = [int(v) for v in _parse_csv_list(args.seeds)]
    os.makedirs(args.output_root, exist_ok=True)

    for config_path in configs:
        base_cfg = load_config(config_path)
        data_name = base_cfg.get("data", {}).get("dataset_name", os.path.splitext(os.path.basename(config_path))[0])
        split_strategy = args.split_strategy or base_cfg.get("data", {}).get("split_strategy", "subject")
        cv_n_splits = int(args.cv_n_splits or base_cfg.get("data", {}).get("cv_n_splits", 5))
        if split_strategy == "loso":
            total_folds = _resolve_loso_fold_count(base_cfg, fallback=cv_n_splits)
        elif split_strategy == "groupkfold":
            total_folds = cv_n_splits
        else:
            total_folds = 1
        fold_indices = [int(v) for v in _parse_csv_list(args.folds)] if args.folds else list(range(total_folds))

        for method in methods:
            if method not in METHODS:
                raise ValueError(f"Unknown method: {method}")
            _, runner = METHODS[method]
            for fold_index in fold_indices:
                for seed in seeds:
                    cfg = deepcopy(base_cfg)
                    cfg.setdefault("project", {})
                    cfg.setdefault("data", {})
                    cfg["project"]["seed"] = int(seed)
                    cfg["data"]["split_strategy"] = split_strategy
                    cfg["data"]["cv_n_splits"] = cv_n_splits
                    cfg["data"]["cv_fold_index"] = int(fold_index)
                    cfg["project"]["output_dir"] = os.path.join(
                        args.output_root,
                        data_name,
                        split_strategy,
                        f"fold_{fold_index}",
                        method,
                        f"seed_{seed}",
                    )
                    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
                    print(
                        f"Running dataset={data_name} method={method} split={split_strategy} fold={fold_index} seed={seed}",
                        flush=True,
                    )
                    runner(cfg)

        if args.llm_variants:
            cmd = [
                sys.executable,
                "experiments/ablations/run_llm_ablation_matrix.py",
                "--config",
                config_path,
                "--variants",
                args.llm_variants,
                "--seeds",
                args.seeds,
                "--output_root",
                os.path.join(args.output_root, data_name, "llm_ablations"),
            ]
            print("Running LLM ablation matrix:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    if args.run_cross_matrix:
        targets = args.cross_matrix_targets or args.configs
        sources = args.cross_matrix_source_roots or args.output_root
        cmd = [
            sys.executable,
            "experiments/protocols/run_cross_dataset_matrix.py",
            "--source_roots",
            sources,
            "--target_configs",
            targets,
            "--methods",
            args.methods,
            "--seeds",
            args.seeds,
        ]
        print("Running cross-dataset matrix:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    if args.build_closure_report:
        primary_candidates = {"physio_diff_main", "physio_diff", "physio_diff_legacy_fullstack"}
        baseline_methods = [m for m in methods if m not in primary_candidates]
        cmd = [
            sys.executable,
            "experiments/reports/build_multi_dataset_closure_report.py",
            "--root",
            args.output_root,
            "--out_md",
            args.closure_report_out_md,
            "--out_json",
            args.closure_report_out_json,
            "--methods",
            ",".join(baseline_methods),
        ]
        print("Building multi-dataset closure report:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
