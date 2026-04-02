import argparse
import os

import numpy as np
import torch
from torch import nn

from experiments.common.data import build_cache, load_cache, load_config, make_dataloaders, make_splits
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


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return self.mlp(emb)


class TransformerDenoiser(nn.Module):
    def __init__(self, in_channels: int, d_model: int, nhead: int, num_layers: int, num_classes: int):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.pos = PositionalEncoding(d_model)
        self.time_embed = TimeEmbedding(d_model)
        self.label_embed = nn.Embedding(num_classes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.pos(h)
        cond = self.time_embed(t, h.size(-1)) + self.label_embed(y)
        h = h + cond.unsqueeze(1)
        h = self.encoder(h)
        return self.out_proj(h)


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.timesteps = cfg["diffusion"]["timesteps"]
        self.denoiser = TransformerDenoiser(
            in_channels=len(cfg["data"]["channels"]),
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            num_classes=mcfg["num_classes"],
        )
        betas = torch.linspace(cfg["diffusion"]["beta_start"], cfg["diffusion"]["beta_end"], self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x_t.transpose(1, 2), t, y).transpose(1, 2)

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = self.predict_eps(x_t, t, y)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar) * eps)
        if (t == 0).all():
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)

    def sample(self, shape, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y)
        return x


def main(cfg):
    device = torch.device(cfg["project"]["device"])
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    np.random.seed(int(cfg["project"].get("seed", 42)))
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    cache_path, meta = build_cache(cfg)
    x, y, acc = load_cache(cache_path)
    splits = make_splits(x, y, acc, meta, cfg)
    train_loader = make_dataloaders(splits["train"], cfg["train"]["batch_size"])
    val_loader = make_dataloaders(splits["val"], cfg["train"]["batch_size"], shuffle=False)

    model = Diffusion(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            t = torch.randint(0, model.timesteps, (xb.size(0),), device=device)
            noise = torch.randn_like(xb)
            x_t = model.q_sample(xb, t, noise)
            eps_pred = model.predict_eps(x_t, t, yb)
            loss = torch.mean((noise - eps_pred) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = []
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                t = torch.randint(0, model.timesteps, (xb.size(0),), device=device)
                noise = torch.randn_like(xb)
                x_t = model.q_sample(xb, t, noise)
                eps_pred = model.predict_eps(x_t, t, yb)
                val_loss.append(torch.mean((noise - eps_pred) ** 2).item())
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} loss={np.mean(val_loss):.4f}")

    model.eval()
    samples_per_class = cfg["eval"]["synth_samples_per_class"]
    gen = []
    labels = []
    with torch.no_grad():
        for cls in range(cfg["model"]["num_classes"]):
            yb = torch.full((samples_per_class,), cls, device=device, dtype=torch.long)
            shape = (samples_per_class, x.shape[1], x.shape[2])
            gen.append(model.sample(shape, yb, device=device).cpu().numpy())
            labels.append(np.full((samples_per_class,), cls))
    synth_x = np.concatenate(gen, axis=0)
    synth_y = np.concatenate(labels, axis=0)
    match_stats = bool(cfg["eval"].get("match_stats", False))
    clip_min = float(cfg["eval"].get("clip_min", -5.0))
    clip_max = float(cfg["eval"].get("clip_max", 5.0))
    synth_x = postprocess_synth(synth_x, splits["train"]["x"], match_stats=match_stats, clip_min=clip_min, clip_max=clip_max)
    np.savez_compressed(os.path.join(cfg["project"]["output_dir"], "synthetic.npz"), x=synth_x, y=synth_y)
    save_class_compare(synth_x, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "class_compare.png")
    save_bvp_spectrum(synth_x, synth_y, cfg["data"]["channels"], cfg["project"]["output_dir"], "bvp_spectrum.png")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/tsdiff_baseline/configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
