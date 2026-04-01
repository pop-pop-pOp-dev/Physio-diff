import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.common.data import load_config
from src.models.physio_diff import PhysioDiffusion


def _build_model(cfg, device: torch.device) -> PhysioDiffusion:
    return PhysioDiffusion(
        in_channels=len(cfg["data"]["channels"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        depth=cfg["model"]["depth"],
        kernel_size=cfg["model"]["kernel_size"],
        dilation_cycle=cfg["model"]["dilation_cycle"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_classes=cfg["model"]["num_classes"],
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        dropout=cfg["model"]["dropout"],
        denoiser_type=cfg.get("model", {}).get("denoiser_type", "dilated"),
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Physio-Diff sampling latency.")
    parser.add_argument("--config", default="configs/best_improved.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_steps", default="1000,250,100,50")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out_json", default="experiments/reports/latency_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["project"]["device"])
    model = _build_model(cfg, device=device)
    model.eval()
    window_length = int(cfg["data"]["window_length"])
    in_channels = len(cfg["data"]["channels"])
    y = torch.zeros((args.batch_size,), device=device, dtype=torch.long)
    results = []

    for steps in [int(v) for v in str(args.sample_steps).split(",") if v.strip()]:
        for _ in range(args.warmup):
            with torch.no_grad():
                _ = model.sample(
                    (args.batch_size, in_channels, window_length),
                    y,
                    device=device,
                    sample_steps=steps,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model.sample(
                    (args.batch_size, in_channels, window_length),
                    y,
                    device=device,
                    sample_steps=steps,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
        mem_mb = 0.0
        if torch.cuda.is_available():
            mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        results.append(
            {
                "sample_steps": int(steps),
                "batch_size": int(args.batch_size),
                "latency_sec_mean": float(np.mean(timings)),
                "latency_sec_std": float(np.std(timings)),
                "throughput_windows_per_sec": float(args.batch_size / max(np.mean(timings), 1e-8)),
                "peak_memory_mb": mem_mb,
            }
        )

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump({"config": os.path.abspath(args.config), "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
