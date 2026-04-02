import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_class_compare(x: np.ndarray, y: np.ndarray, channels, out_dir: str, fname: str):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return
    x0 = x[idx0[0]]
    x1 = x[idx1[0]]
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    for i, ch in enumerate(channels):
        axes[i].plot(x0[i], label="Class 0", alpha=0.8)
        axes[i].plot(x1[i], label="Class 1", alpha=0.8)
        axes[i].set_title(str(ch).upper())
        axes[i].legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def save_bvp_spectrum(x: np.ndarray, y: np.ndarray, channels, out_dir: str, fname: str):
    bvp_idx = None
    for i, ch in enumerate(channels):
        if str(ch).lower() == "bvp":
            bvp_idx = i
            break
    if bvp_idx is None:
        return
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return
    sig0 = x[idx0[0], bvp_idx]
    sig1 = x[idx1[0], bvp_idx]
    fft0 = np.abs(np.fft.rfft(sig0))
    fft1 = np.abs(np.fft.rfft(sig1))
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(fft0, label="Class 0")
    ax.plot(fft1, label="Class 1")
    ax.set_title("BVP Spectrum (Magnitude)")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Magnitude")
    ax.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
