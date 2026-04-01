import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _vivid_palette(n: int) -> List[str]:
    base = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    if n <= len(base):
        return base[:n]
    reps = (n + len(base) - 1) // len(base)
    return (base * reps)[:n]


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _median_iqr(arr: np.ndarray) -> Tuple[float, float, float]:
    q1, med, q3 = np.percentile(arr.astype(float), [25, 50, 75])
    return float(med), float(q1), float(q3)


def _collect_metric(
    root: str, method_dir: str, fname: str, key_path: List[str]
) -> np.ndarray:
    vals: List[float] = []
    method_root = os.path.join(root, method_dir)
    if not os.path.exists(method_root):
        return np.array([], dtype=float)
    for seed_dir in sorted(os.listdir(method_root)):
        p = os.path.join(method_root, seed_dir, fname)
        j = _load_json(p)
        if not j:
            continue
        cur = j
        for k in key_path:
            cur = cur.get(k, {})
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    return np.asarray(vals, dtype=float)


def _save_sota_plots(root: str, out_dir: str) -> None:
    methods = [
        ("Physio-Diff (Full-Stack)", "physio_diff", "physio_results.json"),
        ("Physio-Diff (Mechanistic-only)", "physio_diff_mechanistic_only", "physio_results.json"),
        ("Physio-Diff (Language-only)", "physio_diff_language_only", "physio_results.json"),
        ("Physio-Diff (-Cycle)", "physio_diff_no_cycle", "physio_results.json"),
        ("Physio-Diff (-Semantic Align)", "physio_diff_no_semantic_align", "physio_results.json"),
        ("Physio-Diff (-Artifact Text)", "physio_diff_no_artifact_text", "physio_results.json"),
        ("CSDI", "csdi", "results.json"),
        ("DDPM", "ddpm", "results.json"),
        ("cGAN", "cgan", "results.json"),
        ("WGAN-GP", "wgan_gp", "results.json"),
        ("TimeGAN", "timegan", "results.json"),
        ("TSGM", "tsgm", "results.json"),
        ("TS-Diff", "tsdiff", "results.json"),
    ]

    labels: List[str] = []
    acc_med: List[float] = []
    acc_q1: List[float] = []
    acc_q3: List[float] = []
    rob_med: List[float] = []
    rob_q1: List[float] = []
    rob_q3: List[float] = []
    f1_med: List[float] = []
    mae_med: List[float] = []

    for label, d, f in methods:
        acc = _collect_metric(root, d, f, ["tstr", "accuracy"])
        f1 = _collect_metric(root, d, f, ["tstr", "f1"])
        rob = _collect_metric(root, d, f, ["robust", "restored_acc"])
        mae = _collect_metric(root, d, f, ["time_freq", "mae"])

        labels.append(label)
        if acc.size:
            m, q1, q3 = _median_iqr(acc)
            acc_med.append(m)
            acc_q1.append(q1)
            acc_q3.append(q3)
        else:
            acc_med.append(np.nan)
            acc_q1.append(np.nan)
            acc_q3.append(np.nan)

        if rob.size:
            m, q1, q3 = _median_iqr(rob)
            rob_med.append(m)
            rob_q1.append(q1)
            rob_q3.append(q3)
        else:
            rob_med.append(np.nan)
            rob_q1.append(np.nan)
            rob_q3.append(np.nan)

        f1_med.append(float(np.median(f1)) if f1.size else np.nan)
        mae_med.append(float(np.median(mae)) if mae.size else np.nan)

    x = np.arange(len(labels))
    colors = _vivid_palette(len(labels))

    def _finish(ax):
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for i in range(len(labels)):
        if not np.isfinite(acc_med[i]):
            continue
        yerr = [[acc_med[i] - acc_q1[i]], [acc_q3[i] - acc_med[i]]]
        ax.errorbar(
            [x[i]],
            [acc_med[i]],
            yerr=yerr,
            fmt="o",
            capsize=3.0,
            lw=1.2,
            color=colors[i],
            ecolor=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        ax.annotate(
            f"{acc_q1[i]:.3f}–{acc_q3[i]:.3f}",
            (x[i], acc_q3[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("TSTR accuracy")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sota_tstr_acc.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    for i in range(len(labels)):
        if not np.isfinite(rob_med[i]):
            continue
        yerr = [[rob_med[i] - rob_q1[i]], [rob_q3[i] - rob_med[i]]]
        ax.errorbar(
            [x[i]],
            [rob_med[i]],
            yerr=yerr,
            fmt="o",
            capsize=3.0,
            lw=1.2,
            color=colors[i],
            ecolor=colors[i],
            markerfacecolor=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        ax.annotate(
            f"{rob_q1[i]:.3f}–{rob_q3[i]:.3f}",
            (x[i], rob_q3[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Robust restored accuracy")
    _finish(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sota_robust_restored.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for i in range(len(labels)):
        if np.isfinite(mae_med[i]) and np.isfinite(f1_med[i]):
            ax.scatter(
                [mae_med[i]],
                [f1_med[i]],
                s=42,
                color=colors[i],
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
    for i, lab in enumerate(labels):
        if np.isfinite(mae_med[i]) and np.isfinite(f1_med[i]):
            ax.annotate(lab, (mae_med[i], f1_med[i]), xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"MAE$_{\mathrm{feat}}$ (median)")
    ax.set_ylabel("TSTR macro-F1 (median)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tradeoff_f1_mae.pdf"))
    plt.close(fig)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Xz = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xz, full_matrices=False)
    return Xz @ Vt.T[:, :2]


def _pick_best_synth_npz(path: str, *, bvp_channel: int = 1) -> str:
    if "seed_" not in path:
        return path
    base = os.path.dirname(os.path.dirname(path))
    fname = os.path.basename(path)
    if not os.path.isdir(base):
        return path
    best = None
    for d in sorted(os.listdir(base)):
        if not d.startswith("seed_"):
            continue
        p = os.path.join(base, d, fname)
        if not os.path.exists(p):
            continue
        try:
            syn = np.load(p)
            x = syn["x"].astype(np.float32)
            ch = x[:, bvp_channel, :]
            wstd = ch.std(axis=1)
            frac0 = float((wstd < 1e-3).mean())
            frac_bounds = float(((np.isclose(ch, 5.0) | np.isclose(ch, -5.0)).mean()))
            score = frac0 * 10.0 + frac_bounds
            if best is None or score < best[0]:
                best = (score, p, frac0, frac_bounds)
        except Exception:
            continue
    if best is None:
        return path
    return best[1]

def _sample_balanced(real_x, real_y, synth_x, synth_y, n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_r0 = np.where(real_y == 0)[0]
    idx_r1 = np.where(real_y == 1)[0]
    idx_s0 = np.where(synth_y == 0)[0]
    idx_s1 = np.where(synth_y == 1)[0]
    n = min(len(idx_r0), len(idx_r1), len(idx_s0), len(idx_s1), int(n_per_class))
    r_idx = np.concatenate(
        [rng.choice(idx_r0, n, replace=False), rng.choice(idx_r1, n, replace=False)]
    )
    s_idx = np.concatenate(
        [rng.choice(idx_s0, n, replace=False), rng.choice(idx_s1, n, replace=False)]
    )
    return real_x[r_idx], synth_x[s_idx]


def _save_pca(real_x, real_y, synth_x, synth_y, out_path: str) -> None:
    r_sel, s_sel = _sample_balanced(real_x, real_y, synth_x, synth_y, n_per_class=300, seed=0)
    r_feat = r_sel.reshape(len(r_sel), -1)
    s_feat = s_sel.reshape(len(s_sel), -1)
    Z = _pca_2d(np.concatenate([r_feat, s_feat], axis=0))
    Zr = Z[: len(r_feat)]
    Zs = Z[len(r_feat) :]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(Zr[:, 0], Zr[:, 1], s=10, alpha=0.45, label="Real", color="#1f77b4")
    ax.scatter(Zs[:, 0], Zs[:, 1], s=10, alpha=0.45, label="Synth", color="#ff7f0e")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_waveform_grid(real_x, real_y, synth_x, synth_y, channels: List[str], out_path: str) -> None:
    rng = np.random.default_rng(0)
    def pick(x, y, cls):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            return None
        return x[int(rng.choice(idx, 1)[0])]

    r0 = pick(real_x, real_y, 0)
    r1 = pick(real_x, real_y, 1)
    s0 = pick(synth_x, synth_y, 0)
    s1 = pick(synth_x, synth_y, 1)
    if r0 is None or r1 is None or s0 is None or s1 is None:
        return

    cols = [("Real C0", r0), ("Real C1", r1), ("Synth C0", s0), ("Synth C1", s1)]
    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 4, figsize=(7.6, 1.45 * n_ch), sharex=True)
    if n_ch == 1:
        axes = np.expand_dims(axes, axis=0)

    for j, (title, _) in enumerate(cols):
        axes[0, j].set_title(title)

    for i, ch in enumerate(channels):
        ys = np.concatenate([c[1][i].ravel() for c in cols], axis=0)
        lo = float(np.percentile(ys, 1))
        hi = float(np.percentile(ys, 99))
        pad = 0.05 * (hi - lo + 1e-6)
        ylim = (lo - pad, hi + pad)
        for j, (_, sample) in enumerate(cols):
            ax = axes[i, j]
            ax.plot(sample[i], lw=1.0)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.18, linewidth=0.6)
            if j == 0:
                ax.set_ylabel(str(ch).upper())

    for ax in axes[-1, :]:
        ax.set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_psd_bvp(real_x, real_y, synth_x, synth_y, channels: List[str], fs: int, out_path: str) -> None:
    try:
        from scipy.signal import welch
    except Exception:
        return

    bvp_idx = None
    for i, ch in enumerate(channels):
        if str(ch).lower() == "bvp":
            bvp_idx = i
            break
    if bvp_idx is None:
        return

    def _pick_windows(x, y, cls, max_n: int = 256, seed: int = 0) -> np.ndarray | None:
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            return None
        rng = np.random.default_rng(seed + int(cls))
        n = min(len(idx), int(max_n))
        pick = rng.choice(idx, size=n, replace=False)
        return x[pick, bvp_idx, :]

    def _welch_iqr(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, pxx = welch(windows, fs=fs, axis=-1, nperseg=min(256, windows.shape[-1]))
        pxx_db = 10.0 * np.log10(pxx + 1e-12)
        q1 = np.percentile(pxx_db, 25, axis=0)
        med = np.percentile(pxx_db, 50, axis=0)
        q3 = np.percentile(pxx_db, 75, axis=0)
        return f, med, np.stack([q1, q3], axis=0)

    r0 = _pick_windows(real_x, real_y, 0)
    r1 = _pick_windows(real_x, real_y, 1)
    s0 = _pick_windows(synth_x, synth_y, 0)
    s1 = _pick_windows(synth_x, synth_y, 1)
    if any(v is None for v in [r0, r1, s0, s1]):
        return

    f0, r0_med, r0_iqr = _welch_iqr(r0)
    _, s0_med, s0_iqr = _welch_iqr(s0)
    f1, r1_med, r1_iqr = _welch_iqr(r1)
    _, s1_med, s1_iqr = _welch_iqr(s1)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    axes[0].plot(f0, r0_med, lw=1.8, label="Real", color="#1f77b4")
    axes[0].fill_between(f0, r0_iqr[0], r0_iqr[1], color="#1f77b4", alpha=0.18, linewidth=0)
    axes[0].plot(f0, s0_med, lw=1.8, label="Synth", color="#ff7f0e")
    axes[0].fill_between(f0, s0_iqr[0], s0_iqr[1], color="#ff7f0e", alpha=0.18, linewidth=0)
    axes[0].set_title("Class 0")

    axes[1].plot(f1, r1_med, lw=1.8, label="Real", color="#1f77b4")
    axes[1].fill_between(f1, r1_iqr[0], r1_iqr[1], color="#1f77b4", alpha=0.18, linewidth=0)
    axes[1].plot(f1, s1_med, lw=1.8, label="Synth", color="#ff7f0e")
    axes[1].fill_between(f1, s1_iqr[0], s1_iqr[1], color="#ff7f0e", alpha=0.18, linewidth=0)
    axes[1].set_title("Class 1")
    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("PSD (dB)")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/best_improved.yaml")
    parser.add_argument(
        "--root",
        default="experiments/sota_runs/outputs_strong_r2_evalfix",
        help="Root containing method/seed outputs for SOTA plots.",
    )
    parser.add_argument(
        "--synth",
        default="experiments/sota_runs/outputs_strong_r2_evalfix/physio_diff/seed_0/synthetic_normalized.npz",
    )
    parser.add_argument(
        "--auto_pick_synth_seed",
        action="store_true",
        help="If set, pick a non-degenerate synth seed from the same folder for qualitative plots.",
    )
    parser.add_argument("--out_dir", default="../Figures_Real")
    parser.add_argument("--clip_min", type=float, default=-5.0)
    parser.add_argument("--clip_max", type=float, default=5.0)
    args = parser.parse_args()

    _set_style()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)
    from experiments.common.data import build_cache, load_cache, load_config, make_splits

    cfg = load_config(os.path.join(repo_root, args.config))
    cache_path, meta = build_cache(cfg)
    real_x, real_y, acc = load_cache(cache_path)
    splits = make_splits(real_x, real_y, acc, meta, cfg)
    real_test = splits["test"]["x"].astype(np.float32)
    real_test_y = splits["test"]["y"].astype(int)

    synth_path = os.path.join(repo_root, args.synth)
    bvp_channel = 1
    for i, ch in enumerate(cfg["data"]["channels"]):
        if str(ch).lower() == "bvp":
            bvp_channel = i
            break
    if args.auto_pick_synth_seed:
        synth_path = _pick_best_synth_npz(synth_path, bvp_channel=bvp_channel)
    synth = np.load(synth_path)
    synth_x = synth["x"].astype(np.float32)
    synth_y = synth["y"].astype(int)

    clip_min = float(args.clip_min)
    clip_max = float(args.clip_max)
    real_test = np.clip(real_test, clip_min, clip_max)
    synth_x = np.clip(synth_x, clip_min, clip_max)

    out_dir = os.path.abspath(os.path.join(repo_root, args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    src_pipeline = os.path.join(repo_root, "pictures/summary/fig_pipeline.pdf")
    if os.path.exists(src_pipeline):
        dst_pipeline = os.path.join(out_dir, "fig_pipeline.pdf")
        if os.path.abspath(src_pipeline) != os.path.abspath(dst_pipeline):
            shutil.copy2(src_pipeline, dst_pipeline)

    _save_sota_plots(os.path.join(repo_root, args.root), out_dir)
    _save_pca(real_test, real_test_y, synth_x, synth_y, os.path.join(out_dir, "fig_pca.pdf"))

    channels = list(cfg["data"]["channels"])
    _save_waveform_grid(
        real_test,
        real_test_y,
        synth_x,
        synth_y,
        channels=channels,
        out_path=os.path.join(out_dir, "fig_waveform_grid.pdf"),
    )
    _save_psd_bvp(
        real_test,
        real_test_y,
        synth_x,
        synth_y,
        channels=channels,
        fs=int(cfg["data"]["target_fs"]),
        out_path=os.path.join(out_dir, "fig_psd_comparison.pdf"),
    )


if __name__ == "__main__":
    main()
