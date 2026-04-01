import argparse
import os
import subprocess
import sys


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-target cross-dataset matrix evaluation.")
    parser.add_argument("--source_roots", required=True, help="Comma-separated roots of generated synthetic outputs.")
    parser.add_argument("--target_configs", required=True, help="Comma-separated target dataset configs.")
    parser.add_argument("--methods", default="physio_diff,timegan,csdi,tsdiff,cgan,wgan_gp,ddpm,tsgm")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max_per_class", default="200")
    parser.add_argument("--align_labels", type=int, default=0)
    parser.add_argument("--align_epochs", type=int, default=5)
    parser.add_argument("--out_dir", default="experiments/reports/cross_dataset_matrix")
    parser.add_argument("--build_report", action="store_true")
    parser.add_argument("--report_out_md", default="experiments/reports/cross_dataset_matrix_report.md")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    source_roots = _split_csv(args.source_roots)
    target_configs = _split_csv(args.target_configs)
    script_path = os.path.join("experiments", "reports", "cross_dataset_eval.py")
    for source in source_roots:
        source_name = os.path.basename(source.rstrip("/"))
        for cfg in target_configs:
            target_name = os.path.splitext(os.path.basename(cfg))[0]
            out_json = os.path.join(args.out_dir, f"{source_name}__to__{target_name}.json")
            out_tex = os.path.join(args.out_dir, f"{source_name}__to__{target_name}.tex")
            cmd = [
                sys.executable,
                script_path,
                "--source_root",
                source,
                "--target_config",
                cfg,
                "--methods",
                args.methods,
                "--seeds",
                args.seeds,
                "--max_per_class",
                args.max_per_class,
                "--align_labels",
                str(args.align_labels),
                "--align_epochs",
                str(args.align_epochs),
                "--out_json",
                out_json,
                "--out_tex",
                out_tex,
            ]
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    if args.build_report:
        report_cmd = [
            sys.executable,
            os.path.join("experiments", "reports", "build_cross_dataset_matrix_report.py"),
            "--matrix_dir",
            args.out_dir,
            "--out_md",
            args.report_out_md,
        ]
        print("Building matrix report:", " ".join(report_cmd), flush=True)
        subprocess.run(report_cmd, check=True)


if __name__ == "__main__":
    main()
