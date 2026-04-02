import argparse
import os

import numpy as np
import torch
from torch import nn

from experiments.common.data import build_cache, load_cache, load_config, make_dataloaders, make_splits
from experiments.common.metrics import postprocess_synth, robustness_eval, tstr_eval, time_freq_metrics
from experiments.common.visuals import save_bvp_spectrum, save_class_compare


class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, out_dim: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, z_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.label_emb(y)
        return self.net(torch.cat([z, y_emb], dim=1))


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim + in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.label_emb(y)
        return self.net(torch.cat([x, y_emb], dim=1))


def main(cfg):
    device = torch.device(cfg["project"]["device"])
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    np.random.seed(int(cfg["project"].get("seed", 42)))
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    cache_path, meta = build_cache(cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, cfg)
    train_loader = make_dataloaders(splits["train"], cfg["train"]["batch_size"])

    x_dim = x.shape[1] * x.shape[2]
    g = Generator(cfg["model"]["z_dim"], cfg["model"]["hidden_dim"], x_dim, cfg["model"]["num_classes"]).to(
        device
    )
    d = Discriminator(x_dim, cfg["model"]["hidden_dim"], cfg["model"]["num_classes"]).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg["train"]["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg["train"]["lr"], betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(cfg["train"]["epochs"]):
        for xb, yb in train_loader:
            xb = xb.to(device).view(xb.size(0), -1)
            yb = yb.to(device)
            z = torch.randn(xb.size(0), cfg["model"]["z_dim"], device=device)

            fake = g(z, yb)
            d_real = d(xb, yb)
            d_fake = d(fake.detach(), yb)
            real_loss = loss_fn(d_real, torch.ones_like(d_real))
            fake_loss = loss_fn(d_fake, torch.zeros_like(d_fake))
            d_loss = (real_loss + fake_loss) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            d_fake = d(fake, yb)
            g_loss = loss_fn(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")

    # generate synthetic
    samples_per_class = cfg["eval"]["synth_samples_per_class"]
    gen = []
    labels = []
    with torch.no_grad():
        for cls in range(cfg["model"]["num_classes"]):
            yb = torch.full((samples_per_class,), cls, device=device, dtype=torch.long)
            z = torch.randn(samples_per_class, cfg["model"]["z_dim"], device=device)
            fake = g(z, yb).cpu().numpy()
            gen.append(fake)
            labels.append(np.full((samples_per_class,), cls))
    synth = np.concatenate(gen, axis=0).reshape(-1, x.shape[1], x.shape[2])
    synth_y = np.concatenate(labels, axis=0)
    match_stats = bool(cfg["eval"].get("match_stats", False))
    clip_min = float(cfg["eval"].get("clip_min", -5.0))
    clip_max = float(cfg["eval"].get("clip_max", 5.0))
    synth = postprocess_synth(synth, splits["train"]["x"], match_stats=match_stats, clip_min=clip_min, clip_max=clip_max)
    np.savez_compressed(os.path.join(cfg["project"]["output_dir"], "synthetic.npz"), x=synth, y=synth_y)
    save_class_compare(synth, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "class_compare.png")
    save_bvp_spectrum(synth, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "bvp_spectrum.png")

    tstr = tstr_eval(
        synth,
        synth_y,
        splits["test"]["x"],
        splits["test"]["y"],
        cfg,
        device,
    )
    robust = robustness_eval(
        synth,
        synth_y,
        splits["test"]["x"],
        splits["test"]["y"],
        splits["test"]["acc"],
        cfg,
        device,
    )
    tfm = time_freq_metrics(splits["test"]["x"], synth[: len(splits["test"]["x"])], cfg["data"]["target_fs"])
    out_path = os.path.join(cfg["project"]["output_dir"], "results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        import json

        json.dump({"tstr": tstr, "time_freq": tfm, "robust": robust}, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/cgan_baseline/configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
