import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Export latency benchmark to markdown and tex.")
    parser.add_argument("--in_json", default="experiments/reports/latency_results.json")
    parser.add_argument("--out_md", default="experiments/reports/latency_table.md")
    parser.add_argument("--out_tex", default="experiments/reports/latency_table.tex")
    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("results", [])
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as handle:
        handle.write("| sample_steps | batch_size | latency_sec_mean | throughput_win_s | peak_memory_mb |\n")
        handle.write("|---:|---:|---:|---:|---:|\n")
        for r in rows:
            handle.write(
                f"| {r['sample_steps']} | {r['batch_size']} | {r['latency_sec_mean']:.4f} | "
                f"{r['throughput_windows_per_sec']:.2f} | {r['peak_memory_mb']:.1f} |\n"
            )

    with open(args.out_tex, "w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(
                f"{r['sample_steps']} & {r['batch_size']} & {r['latency_sec_mean']:.4f} & "
                f"{r['throughput_windows_per_sec']:.2f} & {r['peak_memory_mb']:.1f} \\\\\n"
            )


if __name__ == "__main__":
    main()
