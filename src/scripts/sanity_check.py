import argparse
import os

import numpy as np
import torch
import yaml

from src.data.datasets import WESADDataset, make_dataloaders
from src.data.wesad import build_dataset_cache
from src.eval.metrics import evaluate_comprehensive, train_classifier
from src.models.physio_diff import PhysioDiffusion


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(cfg):
    device = torch.device(cfg["project"]["device"])
    cache_path = build_dataset_cache(
        root_dir=cfg["data"]["root_dir"],
        cache_dir=cfg["data"]["cache_dir"],
        subjects=cfg["data"]["subjects"],
        channels=cfg["data"]["channels"],
        include_acc=cfg["data"]["include_acc"],
        target_fs=cfg["data"]["target_fs"],
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["window_stride"],
        zscore=cfg["data"]["zscore"],
        processed_label_path=cfg["data"].get("processed_label_path"),
    )
    dataset = WESADDataset(cache_path)
    train_loader, val_loader, test_loader = make_dataloaders(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        train_ratio=cfg["data"].get("train_ratio", 0.8),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        seed=cfg["project"]["seed"],
        split_strategy=cfg["data"].get("split_strategy", "random"),
        train_subjects=cfg["data"].get("train_subjects"),
        val_subjects=cfg["data"].get("val_subjects"),
        test_subjects=cfg["data"].get("test_subjects"),
    )

    print("Sanity Check A: Real->Real classifier")
    classifier = train_classifier(
        train_loader.dataset,
        in_channels=len(cfg["data"]["channels"]),
        epochs=cfg["eval"]["classifier_epochs"],
        lr=cfg["eval"]["classifier_lr"],
        device=device,
    )
    acc, f1 = evaluate_comprehensive(classifier, test_loader, device=device)
    print(f"Real->Real Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    print("Sanity Check B: Condition effect in synthetic")
    ckpt_path = cfg["eval"].get("checkpoint_path") or os.path.join(
        cfg["project"]["output_dir"], "physio_diff_best.pt"
    )
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    model = PhysioDiffusion(
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
    ).to(device)
    # Backward-compatible load for checkpoints saved before CFG support.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "denoiser.label_embed.weight"
    if isinstance(state, dict) and key in state and key in model.state_dict():
        w = state[key]
        want = model.state_dict()[key].shape
        if isinstance(w, torch.Tensor):
            have = w.shape
            if len(have) == 2 and len(want) == 2 and have[0] + 1 == want[0] and have[1] == want[1]:
                w_new = torch.zeros(want, dtype=w.dtype)
                w_new[: have[0]] = w
                state[key] = w_new
    model.load_state_dict(state, strict=True)
    model.eval()

    n = 256
    channels = len(cfg["data"]["channels"])
    length = cfg["data"]["window_length"]
    cfg_scale = float(cfg["eval"].get("cfg_scale", 1.0))
    with torch.no_grad():
        y0 = torch.zeros(n, device=device, dtype=torch.long)
        y1 = torch.ones(n, device=device, dtype=torch.long)
        x0 = model.sample((n, channels, length), y0, device=device, cfg_scale=cfg_scale).cpu().numpy()
        x1 = model.sample((n, channels, length), y1, device=device, cfg_scale=cfg_scale).cpu().numpy()
    mean0 = x0.mean(axis=0)
    mean1 = x1.mean(axis=0)
    diff = np.mean(np.abs(mean0 - mean1))
    print(f"Synthetic class mean abs diff: {diff:.6f}")
    print(f"Synthetic range: min {x0.min():.3f}, max {x0.max():.3f}, std {x0.std():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
