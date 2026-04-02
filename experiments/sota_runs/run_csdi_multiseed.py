import argparse
import os
from copy import deepcopy

import yaml

from experiments.csdi_baseline.run_csdi import main as csdi_main


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/csdi_baseline/configs/base.yaml")
    parser.add_argument("--out_root", default="experiments/sota_runs/outputs_strong_r2_evalfix/csdi")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    base = _load_yaml(args.config)
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    for seed in seeds:
        cfg = deepcopy(base)
        cfg.setdefault("project", {})
        cfg["project"]["seed"] = seed
        cfg["project"]["output_dir"] = os.path.join(args.out_root, f"seed_{seed}")
        os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
        csdi_main(cfg)


if __name__ == "__main__":
    main()

