import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common.data import build_cache, load_cache, load_config, make_splits
from experiments.common.metrics import postprocess_synth, robustness_eval, tstr_eval, time_freq_metrics
from experiments.common.visuals import save_bvp_spectrum, save_class_compare


class TimeGAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedder = nn.LSTM(input_dim + num_classes, hidden_dim, batch_first=True)
        self.recovery = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.recovery_fc = nn.Linear(hidden_dim, input_dim)
        self.generator = nn.LSTM(z_dim + num_classes, hidden_dim, batch_first=True)
        self.supervisor = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.discriminator = nn.LSTM(hidden_dim + num_classes, hidden_dim, batch_first=True)
        self.disc_fc = nn.Linear(hidden_dim, 1)

    def _label_onehot(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()

    def embed(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = self._label_onehot(y).unsqueeze(1).expand(-1, x.size(1), -1)
        h, _ = self.embedder(torch.cat([x, y_oh], dim=-1))
        return h

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        r, _ = self.recovery(h)
        return self.recovery_fc(r)

    def generate(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = self._label_onehot(y).unsqueeze(1).expand(-1, z.size(1), -1)
        g, _ = self.generator(torch.cat([z, y_oh], dim=-1))
        s, _ = self.supervisor(g)
        return s

    def discriminate(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_oh = self._label_onehot(y).unsqueeze(1).expand(-1, h.size(1), -1)
        d, _ = self.discriminator(torch.cat([h, y_oh], dim=-1))
        return self.disc_fc(d[:, -1]).squeeze(-1)


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

    input_dim = x.shape[1]
    model = TimeGAN(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        z_dim=cfg["model"]["z_dim"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    opt_embed = torch.optim.Adam(
        list(model.embedder.parameters())
        + list(model.recovery.parameters())
        + list(model.recovery_fc.parameters()),
        lr=cfg["train"]["lr"],
    )
    opt_sup = torch.optim.Adam(model.supervisor.parameters(), lr=cfg["train"]["lr"])
    opt_gen = torch.optim.Adam(model.generator.parameters(), lr=cfg["train"]["lr"])
    opt_disc = torch.optim.Adam(model.discriminator.parameters(), lr=cfg["train"]["lr"])

    for epoch in range(cfg["train"]["pretrain_epochs"]):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            h = model.embed(xb, yb)
            x_tilde = model.recover(h)
            loss = mse(x_tilde, xb)
            opt_embed.zero_grad()
            loss.backward()
            opt_embed.step()
            losses.append(loss.item())
        print(f"Pretrain {epoch+1}/{cfg['train']['pretrain_epochs']} loss={np.mean(losses):.4f}")

    for epoch in range(cfg["train"]["sup_epochs"]):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.no_grad():
                h = model.embed(xb, yb)
            s, _ = model.supervisor(h)
            loss = mse(h[:, 1:], s[:, :-1])
            opt_sup.zero_grad()
            loss.backward()
            opt_sup.step()
            losses.append(loss.item())
        print(f"Supervisor {epoch+1}/{cfg['train']['sup_epochs']} loss={np.mean(losses):.4f}")

    for epoch in range(cfg["train"]["gan_epochs"]):
        model.train()
        d_losses = []
        g_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            h_real = model.embed(xb, yb)
            z = torch.randn(xb.size(0), xb.size(1), cfg["model"]["z_dim"], device=device)
            h_fake = model.generate(z, yb)

            d_real = model.discriminate(h_real.detach(), yb)
            d_fake = model.discriminate(h_fake.detach(), yb)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

            d_fake = model.discriminate(h_fake, yb)
            g_adv = bce(d_fake, torch.ones_like(d_fake))
            g_sup = mse(h_real[:, 1:], h_fake[:, :-1])
            g_loss = g_adv + 0.1 * g_sup
            opt_gen.zero_grad()
            opt_sup.zero_grad()
            g_loss.backward()
            opt_gen.step()
            opt_sup.step()
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
        print(
            f"GAN {epoch+1}/{cfg['train']['gan_epochs']} d_loss={np.mean(d_losses):.4f} g_loss={np.mean(g_losses):.4f}"
        )

    model.eval()
    samples_per_class = cfg["eval"]["synth_samples_per_class"]
    gen = []
    labels = []
    with torch.no_grad():
        for cls in range(cfg["model"]["num_classes"]):
            yb = torch.full((samples_per_class,), cls, device=device, dtype=torch.long)
            z = torch.randn(samples_per_class, x.shape[2], cfg["model"]["z_dim"], device=device)
            h = model.generate(z, yb)
            x_hat = model.recover(h).cpu().numpy()
            gen.append(x_hat.transpose(0, 2, 1))
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
    parser.add_argument("--config", default="experiments/timegan_baseline/configs/base.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
