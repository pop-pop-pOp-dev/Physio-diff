import argparse
import os
import subprocess
import sys


def _csv_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on one source dataset and evaluate generalization on multiple target datasets."
    )
    parser.add_argument("--source_config", required=True, help="Source dataset config used for all methods.")
    parser.add_argument(
        "--source_name",
        default="source_domain",
        help="Short source label in metadata/report (e.g. wesad or case).",
    )
    parser.add_argument(
        "--target_configs",
        default="configs/stress_predict.yaml,configs/swell_kw.yaml,configs/case.yaml",
        help="Comma-separated target configs for cross-target evaluation.",
    )
    parser.add_argument(
        "--methods",
        default="physio_diff_main,timegan,csdi,tsdiff,ddpm",
        help="Comma-separated methods to run under the same source domain.",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--train_output_root",
        default="experiments/sota_runs/source_outputs",
        help="Outputs for source-domain training runs.",
    )
    parser.add_argument(
        "--target_eval_output_root",
        default="experiments/protocols/outputs_cross_target",
        help="Outputs for target-domain evaluation runs.",
    )
    parser.add_argument("--skip_source_train", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.train_output_root, exist_ok=True)
    os.makedirs(args.target_eval_output_root, exist_ok=True)

    methods = _csv_list(args.methods)
    primary_method = "physio_diff_main" if "physio_diff_main" in methods else ("physio_diff" if "physio_diff" in methods else None)
    baseline_methods = [m for m in methods if m != primary_method]

    if not args.skip_source_train:
        train_cmd = [
            sys.executable,
            "-m",
            "experiments.sota_runs.run_multi_seed",
            "--source_config",
            args.source_config,
            "--methods",
            args.methods,
            "--seeds",
            args.seeds,
            "--device",
            args.device,
            "--output_root",
            args.train_output_root,
        ]
        print("Running source-domain training:", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, check=True)

    source_report_cmd = [
        sys.executable,
        "-m",
        "experiments.reports.build_multiseed_report",
        "--root",
        args.train_output_root,
        "--primary_method",
        "physio_diff_main",
        "--methods",
        args.methods,
        "--out_path",
        "experiments/reports/multiseed_report_source.md",
    ]
    print("Building source-domain report:", " ".join(source_report_cmd), flush=True)
    subprocess.run(source_report_cmd, check=True)

    target_eval_cmd = [
        sys.executable,
        "-m",
        "experiments.protocols.evaluate_synth_on_targets",
        "--source_root",
        args.train_output_root,
        "--source_name",
        args.source_name,
        "--target_configs",
        args.target_configs,
        "--methods",
        args.methods,
        "--seeds",
        args.seeds,
        "--out_root",
        args.target_eval_output_root,
    ]
    print("Running target-domain evaluation:", " ".join(target_eval_cmd), flush=True)
    subprocess.run(target_eval_cmd, check=True)

    closure_cmd = [
        sys.executable,
        "-m",
        "experiments.reports.build_multi_dataset_closure_report",
        "--root",
        args.target_eval_output_root,
        "--out_md",
        "experiments/reports/multi_dataset_closure_report_target_eval.md",
        "--out_json",
        "experiments/reports/multi_dataset_closure_report_target_eval.json",
        "--methods",
        ",".join(baseline_methods),
    ]
    print("Building target-domain closure report:", " ".join(closure_cmd), flush=True)
    subprocess.run(closure_cmd, check=True)


if __name__ == "__main__":
    main()
