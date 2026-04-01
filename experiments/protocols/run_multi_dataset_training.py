import argparse
import os
from copy import deepcopy

from experiments.common.data import load_config
from src.scripts.run_pipeline import run


def _csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential multi-dataset training launcher.")
    parser.add_argument("--configs", default="configs/best_improved.yaml,configs/swell_kw.yaml,configs/case.yaml,configs/ubfc_phys.yaml,configs/mahnob_hci.yaml,configs/stress_predict.yaml")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output_root", default="experiments/protocols/multi_dataset_outputs")
    args = parser.parse_args()

    configs = _csv(args.configs)
    seeds = [int(x) for x in _csv(args.seeds)]
    for cfg_path in configs:
        base_cfg = load_config(cfg_path)
        dname = base_cfg.get("data", {}).get("dataset_name", os.path.splitext(os.path.basename(cfg_path))[0])
        for seed in seeds:
            cfg = deepcopy(base_cfg)
            cfg["project"]["seed"] = int(seed)
            cfg["project"]["output_dir"] = os.path.join(args.output_root, dname, f"seed_{seed}")
            os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
            print(f"Running dataset={dname} seed={seed}", flush=True)
            run(cfg)


if __name__ == "__main__":
    main()
