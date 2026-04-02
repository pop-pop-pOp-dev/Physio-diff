import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.data import build_cache, load_cache, load_config, make_splits
from experiments.common.metrics import postprocess_synth, robustness_eval, tstr_eval, time_freq_metrics
from experiments.common.visuals import save_bvp_spectrum, save_class_compare


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerGenerator(nn.Module):
    def __init__(self, z_dim: int, d_model: int, nhead: int, num_layers: int, out_channels: int, num_classes: int):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.in_proj = nn.Linear(z_dim + num_classes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = torch.nn.functional.one_hot(y, num_classes=self.label_embed.num_embeddings).float()
        y_oh = y_oh.unsqueeze(1).expand(-1, z.size(1), -1)
        h = self.in_proj(torch.cat([z, y_oh], dim=-1))
        h = self.pos(h)
        h = self.encoder(h)
        return self.out_proj(h)


class TransformerDiscriminator(nn.Module):
    def __init__(self, in_channels: int, d_model: int, nhead: int, num_layers: int, num_classes: int):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, num_classes)
        self.in_proj = nn.Linear(in_channels + num_classes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = torch.nn.functional.one_hot(y, num_classes=self.label_embed.num_embeddings).float()
        y_oh = y_oh.unsqueeze(1).expand(-1, x.size(1), -1)
        h = self.in_proj(torch.cat([x, y_oh], dim=-1))
        h = self.pos(h)
        h = self.encoder(h)
        return self.fc(h[:, -1]).squeeze(-1)


def main(cfg):
    device = torch.device(cfg["project"]["device"])
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    np.random.seed(int(cfg["project"].get("seed", 42)))
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    cache_path, meta = build_cache(cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, cfg)
    x_train = splits["train"]["x"].transpose(0, 2, 1)
    y_train = splits["train"]["y"]
    ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    gen = TransformerGenerator(
        z_dim=cfg["model"]["z_dim"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        out_channels=x.shape[1],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    disc = TransformerDiscriminator(
        in_channels=x.shape[1],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg["train"]["lr"])
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg["train"]["lr"])

    log_every = cfg["train"].get("log_every", 50)
    for epoch in range(cfg["train"]["epochs"]):
        d_losses = []
        g_losses = []
        for step, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            z = torch.randn(xb.size(0), xb.size(1), cfg["model"]["z_dim"], device=device)
            fake = gen(z, yb)
            d_real = disc(xb, yb)
            d_fake = disc(fake.detach(), yb)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            d_fake = disc(fake, yb)
            g_loss = bce(d_fake, torch.ones_like(d_fake))
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            if log_every and step % log_every == 0:
                print(
                    f"Epoch {epoch+1}/{cfg['train']['epochs']} step {step}/{len(loader)} "
                    f"d_loss={np.mean(d_losses):.4f} g_loss={np.mean(g_losses):.4f}"
                )
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} d_loss={np.mean(d_losses):.4f} g_loss={np.mean(g_losses):.4f}")

    print("Training done. Generating synthetic samples...")
    gen.eval()
    samples_per_class = cfg["eval"]["synth_samples_per_class"]
    gen_list = []
    labels = []
    with torch.no_grad():
        for cls in range(cfg["model"]["num_classes"]):
            print(f"Sampling class {cls+1}/{cfg['model']['num_classes']}...")
            yb = torch.full((samples_per_class,), cls, device=device, dtype=torch.long)
            z = torch.randn(samples_per_class, x.shape[2], cfg["model"]["z_dim"], device=device)
            fake = gen(z, yb).cpu().numpy()
            gen_list.append(fake.transpose(0, 2, 1))
            labels.append(np.full((samples_per_class,), cls))
    print("Saving synthetic outputs and visuals...")
    synth_x = np.concatenate(gen_list, axis=0)
    synth_y = np.concatenate(labels, axis=0)
    match_stats = bool(cfg["eval"].get("match_stats", False))
    clip_min = float(cfg["eval"].get("clip_min", -5.0))
    clip_max = float(cfg["eval"].get("clip_max", 5.0))
    synth_x = postprocess_synth(synth_x, splits["train"]["x"], match_stats=match_stats, clip_min=clip_min, clip_max=clip_max)
    np.savez_compressed(os.path.join(cfg["project"]["output_dir"], "synthetic.npz"), x=synth_x, y=synth_y)
    save_class_compare(synth_x, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "class_compare.png")
    save_bvp_spectrum(synth_x, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "bvp_spectrum.png")

    print("Running evaluation metrics (TSTR, robustness, time-frequency)...")
    tstr = tstr_eval(
        synth_x,
        synth_y,
        splits["test"]["x"],
        splits["test"]["y"],
        cfg,
        device,
    )
    robust = robustness_eval(
        synth_x,
        synth_y,
        splits["test"]["x"],
        splits["test"]["y"],
        splits["test"]["acc"],
        cfg,
        device,
    )
    tfm = time_freq_metrics(splits["test"]["x"], synth_x[: len(splits["test"]["x"])], cfg["data"]["target_fs"])
    result = {"tstr": tstr, "time_freq": tfm, "robust": robust}
    out_path = os.path.join(cfg["project"]["output_dir"], "results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(result, handle, indent=2)
    print(f"TSGM results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/tsgm_baseline/configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
