import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from experiments.common.data import build_cache, load_cache, load_config, make_splits


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pick_class_samples(x: np.ndarray, y: np.ndarray, n: int = 5):
    idx0 = np.where(y == 0)[0][:n]
    idx1 = np.where(y == 1)[0][:n]
    return x[idx0], x[idx1]


def plot_waveform_grid(real_x, real_y, synth_x, synth_y, channels, out_dir):
    _ensure_dir(out_dir)
    r0, r1 = _pick_class_samples(real_x, real_y)
    s0, s1 = _pick_class_samples(synth_x, synth_y)
    for i, ch in enumerate(channels):
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes[0, 0].plot(r0[0, i], label="Real C0")
        axes[0, 1].plot(r1[0, i], label="Real C1")
        axes[1, 0].plot(s0[0, i], label="Synth C0")
        axes[1, 1].plot(s1[0, i], label="Synth C1")
        axes[0, 0].legend(); axes[0, 1].legend()
        axes[1, 0].legend(); axes[1, 1].legend()
        fig.suptitle(f"Waveforms: {str(ch).upper()}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"waveform_grid_{ch}.png"), dpi=150)
        plt.close(fig)


def plot_amplitude_hist(real_x, synth_x, channels, out_dir):
    _ensure_dir(out_dir)
    for i, ch in enumerate(channels):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(real_x[:, i, :].ravel(), bins=80, alpha=0.5, label="Real", density=True)
        ax.hist(synth_x[:, i, :].ravel(), bins=80, alpha=0.5, label="Synth", density=True)
        ax.set_title(f"Amplitude Distribution: {str(ch).upper()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"amplitude_hist_{ch}.png"), dpi=150)
        plt.close(fig)


def plot_psd_overlay(real_x, synth_x, channels, out_dir):
    _ensure_dir(out_dir)
    for i, ch in enumerate(channels):
        r = np.abs(np.fft.rfft(real_x[:, i, :].mean(axis=0)))
        s = np.abs(np.fft.rfft(synth_x[:, i, :].mean(axis=0)))
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(r, label="Real")
        ax.plot(s, label="Synth")
        ax.set_title(f"PSD (Mean) {str(ch).upper()}")
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Magnitude")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"psd_overlay_{ch}.png"), dpi=150)
        plt.close(fig)


def plot_pca_scatter(real_x, real_y, synth_x, synth_y, out_dir):
    _ensure_dir(out_dir)
    def feats(x):
        mean = x.mean(axis=2)
        std = x.std(axis=2)
        return np.concatenate([mean, std], axis=1)
    r = feats(real_x)
    s = feats(synth_x)
    X = np.concatenate([r, s], axis=0)
    X -= X.mean(axis=0, keepdims=True)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    pca = U[:, :2]
    r_p = pca[: len(r)]
    s_p = pca[len(r) :]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(r_p[:, 0], r_p[:, 1], s=8, alpha=0.4, label="Real")
    ax.scatter(s_p[:, 0], s_p[:, 1], s=8, alpha=0.4, label="Synth")
    ax.set_title("PCA Scatter (Mean/Std features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=150)
    plt.close(fig)


def main(cfg_path: str, synth_path: str, out_dir: str):
    cfg = load_config(cfg_path)
    cache_path, meta = build_cache(cfg)
    real_x, real_y, _ = load_cache(cache_path)
    splits = make_splits(real_x, real_y, None, meta, cfg)
    real_x = splits["test"]["x"]
    real_y = splits["test"]["y"]
    synth = np.load(synth_path)
    synth_x = synth["x"]
    synth_y = synth["y"]
    channels = cfg["data"]["channels"]
    plot_waveform_grid(real_x, real_y, synth_x, synth_y, channels, out_dir)
    plot_amplitude_hist(real_x, synth_x, channels, out_dir)
    plot_psd_overlay(real_x, synth_x, channels, out_dir)
    plot_pca_scatter(real_x, real_y, synth_x, synth_y, out_dir)
    print(f"Fancy plots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best.yaml")
    parser.add_argument("--synth", default="outputs/synthetic_normalized.npz")
    parser.add_argument("--out_dir", default="experiments/reports/fancy_plots")
    args = parser.parse_args()
    main(args.config, args.synth, args.out_dir)
