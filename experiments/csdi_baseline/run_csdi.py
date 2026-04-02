import argparse
import os

import numpy as np
import torch
from torch import nn

from experiments.common.data import build_cache, load_cache, load_config, make_dataloaders, make_splits
from experiments.common.metrics import postprocess_synth, robustness_eval, tstr_eval, time_freq_metrics
from experiments.common.visuals import save_bvp_spectrum, save_class_compare


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.cond = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = torch.relu(h + self.cond(cond).unsqueeze(-1))
        h = self.conv2(h)
        h = self.norm2(h)
        return x + torch.relu(h)


class Denoiser1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        kernel_size: int,
        dilation_cycle: int,
        embedding_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.label_embed = nn.Embedding(num_classes, embedding_dim)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** (i % dilation_cycle)
            self.blocks.append(ResidualBlock1D(hidden_channels, embedding_dim, kernel_size, dilation))
        self.output_proj = nn.Conv1d(hidden_channels, in_channels - 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        cond = self.time_mlp(self.time_embed(t)) + self.label_embed(y)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)


class Diffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.timesteps = cfg["diffusion"]["timesteps"]
        self.denoiser = Denoiser1D(
            in_channels=len(cfg["data"]["channels"]) + 1,
            hidden_channels=mcfg["hidden_channels"],
            depth=mcfg["depth"],
            kernel_size=mcfg["kernel_size"],
            dilation_cycle=mcfg["dilation_cycle"],
            embedding_dim=mcfg["embedding_dim"],
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

    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Mask is a single channel so input matches (channels + 1).
        x_in = torch.cat([x_t * mask, mask], dim=1)
        return self.denoiser(x_in, t, y)

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eps = self.predict_eps(x_t, t, y, mask)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar) * eps)
        if (t == 0).all():
            return mean
        return mean + torch.sqrt(beta_t) * torch.randn_like(x_t)

    def sample(self, shape, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        mask = torch.zeros((shape[0], 1, shape[2]), device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, y, mask)
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
    mask_prob = cfg["train"].get("mask_prob", 0.2)
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            t = torch.randint(0, model.timesteps, (xb.size(0),), device=device)
            noise = torch.randn_like(xb)
            x_t = model.q_sample(xb, t, noise)
            mask = (torch.rand(xb.size(0), 1, xb.size(2), device=xb.device) > mask_prob).float()
            eps_pred = model.predict_eps(x_t, t, yb, mask)
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
                mask = (torch.rand(xb.size(0), 1, xb.size(2), device=xb.device) > mask_prob).float()
                eps_pred = model.predict_eps(x_t, t, yb, mask)
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
    parser.add_argument("--config", default="experiments/csdi_baseline/configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
