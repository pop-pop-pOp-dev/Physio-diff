import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot runner for multi-dataset experiments and statistical-closure reports."
    )
    parser.add_argument("--configs", required=True, help="Comma-separated dataset config paths.")
    parser.add_argument(
        "--methods",
        default="physio_diff_main,physio_diff_legacy_fullstack,physio_diff_main_additive_cond,physio_diff_main_no_multiband,timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--split_strategy", default="groupkfold")
    parser.add_argument("--cv_n_splits", type=int, default=5)
    parser.add_argument("--output_root", default="experiments/protocols/outputs")
    parser.add_argument("--run_cross_matrix", action="store_true")
    parser.add_argument("--cross_matrix_out_dir", default="experiments/reports/cross_dataset_matrix")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    suite_cmd = [
        sys.executable,
        "experiments/protocols/run_protocol_suite.py",
        "--configs",
        args.configs,
        "--methods",
        args.methods,
        "--seeds",
        args.seeds,
        "--output_root",
        args.output_root,
        "--split_strategy",
        args.split_strategy,
        "--cv_n_splits",
        str(int(args.cv_n_splits)),
    ]
    if args.run_cross_matrix:
        suite_cmd.append("--run_cross_matrix")
        suite_cmd.extend(["--cross_matrix_targets", args.configs])
        suite_cmd.extend(["--cross_matrix_source_roots", args.output_root])
    print("Running protocol suite:", " ".join(suite_cmd), flush=True)
    subprocess.run(suite_cmd, check=True)

    closure_cmd = [
        sys.executable,
        "experiments/reports/build_multi_dataset_closure_report.py",
        "--root",
        args.output_root,
        "--out_md",
        "experiments/reports/multi_dataset_closure_report.md",
        "--out_json",
        "experiments/reports/multi_dataset_closure_report.json",
        "--methods",
        args.methods.replace("physio_diff,", "").replace(",physio_diff", "").replace("physio_diff", ""),
    ]
    print("Building multi-dataset closure report:", " ".join(closure_cmd), flush=True)
    subprocess.run(closure_cmd, check=True)

    if args.run_cross_matrix:
        cross_cmd = [
            sys.executable,
            "experiments/reports/build_cross_dataset_matrix_report.py",
            "--matrix_dir",
            args.cross_matrix_out_dir,
            "--out_md",
            "experiments/reports/cross_dataset_matrix_report.md",
        ]
        print("Building cross-dataset matrix report:", " ".join(cross_cmd), flush=True)
        subprocess.run(cross_cmd, check=True)


if __name__ == "__main__":
    main()
