from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import random
import torch
from scipy.signal import welch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.text.text_encoder import tokenize_text


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class StrongCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


def train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    cfg: Dict,
    device: torch.device,
    *,
    model_cls: type[nn.Module] = SimpleCNN,
    epochs: int | None = None,
    lr: float | None = None,
) -> nn.Module:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=int(cfg["eval"].get("classifier_batch", 64)), shuffle=True)
    model = model_cls(in_channels=x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["eval"]["classifier_lr"] if lr is None else lr))
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(int(cfg["eval"].get("classifier_epochs", 30) if epochs is None else epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_classifier(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device) -> Tuple[float, float]:
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=cfg_eval_batch(model), shuffle=False)
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(yb.numpy().tolist())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1


def cfg_eval_batch(model: nn.Module) -> int:
    return 64


def _set_eval_seed(cfg: Dict, salt: int) -> None:
    seed = int(cfg.get("project", {}).get("seed", 42))
    seed = (seed * 1009 + int(salt)) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def psd_distance(x: np.ndarray, y: np.ndarray, fs: int) -> float:
    _, pxx = welch(x, fs=fs)
    _, pyy = welch(y, fs=fs)
    return float(np.mean((pxx - pyy) ** 2))


def lf_hf_ratio(x: np.ndarray, fs: int) -> float:
    freqs, pxx = welch(x, fs=fs)
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)
    lf = np.trapz(pxx[lf_band], freqs[lf_band])
    hf = np.trapz(pxx[hf_band], freqs[hf_band])
    if hf == 0:
        return 0.0
    return float(lf / hf)


def postprocess_synth(
    synth_x: np.ndarray,
    ref_x: np.ndarray,
    *,
    match_stats: bool = False,
    clip_min: float = -5.0,
    clip_max: float = 5.0,
) -> np.ndarray:
    """
    Post-process synthetic signals for evaluation fairness.

    Notes:
    - `match_stats` aligns per-channel mean/std to a reference set. This can improve apparent
      fidelity but should be applied consistently across methods (or disabled).
    - Clipping bounds extreme outliers and stabilizes downstream evaluation.
    """
    x = synth_x.astype(np.float32, copy=True)
    if match_stats:
        ref_mean = ref_x.mean(axis=(0, 2), keepdims=True)
        ref_std = ref_x.std(axis=(0, 2), keepdims=True) + 1e-6
        syn_mean = x.mean(axis=(0, 2), keepdims=True)
        syn_std = x.std(axis=(0, 2), keepdims=True) + 1e-6
        x = (x - syn_mean) / syn_std * ref_std + ref_mean
    x = np.clip(x, clip_min, clip_max)
    return x


def _safe_skew(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    z = (x - mean) / std
    return (z**3).mean(axis=-1)


def _safe_kurt(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-8
    z = (x - mean) / std
    return (z**4).mean(axis=-1) - 3.0


def _band_powers(signal: np.ndarray, n_bands: int = 5) -> np.ndarray:
    """Relative band powers over equally spaced frequency bins. signal: (N, L) -> (N, n_bands)."""
    fft = np.fft.rfft(signal, axis=-1)
    psd = (np.abs(fft) ** 2).astype(np.float32)
    total = psd.sum(axis=-1, keepdims=True) + 1e-8
    psd = psd / total
    n_freq = psd.shape[-1]
    bands = []
    for i in range(n_bands):
        lo = int(i * n_freq / n_bands)
        hi = int((i + 1) * n_freq / n_bands)
        bands.append(psd[..., lo:hi].sum(axis=-1))
    return np.stack(bands, axis=-1)


def _spectral_centroid(signal: np.ndarray) -> np.ndarray:
    fft = np.fft.rfft(signal, axis=-1)
    mag = np.abs(fft).astype(np.float32)
    freqs = np.linspace(0.0, 1.0, mag.shape[-1], dtype=np.float32)
    num = (mag * freqs).sum(axis=-1)
    den = mag.sum(axis=-1) + 1e-8
    return num / den


def time_freq_features(x: np.ndarray, fs: int, n_bands: int = 5) -> np.ndarray:
    """
    Distribution-level features for unpaired generation evaluation.

    We do NOT assume sample-wise alignment between real and synthetic data.
    """
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    rms = np.sqrt((x**2).mean(axis=-1))
    ptp = x.max(axis=-1) - x.min(axis=-1)
    skew = _safe_skew(x)
    kurt = _safe_kurt(x)

    band = []
    centroid = []
    for c in range(x.shape[1]):
        band.append(_band_powers(x[:, c, :], n_bands=n_bands))
        centroid.append(_spectral_centroid(x[:, c, :]))
    band = np.concatenate(band, axis=-1)  # (N, C*n_bands)
    centroid = np.stack(centroid, axis=-1)  # (N, C)

    return np.concatenate([mean, std, rms, ptp, skew, kurt, band, centroid], axis=1).astype(
        np.float32
    )


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """RBF-MMD between feature sets x and y."""
    if gamma is None:
        z = np.vstack([x, y])
        dists = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
        med = float(np.median(dists))
        gamma = 1.0 / (med + 1e-8)
    Kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    Kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    Kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def frechet_like(x: np.ndarray, y: np.ndarray) -> float:
    """Frechet distance on features (FID-like)."""
    mu1 = x.mean(axis=0)
    mu2 = y.mean(axis=0)
    cov1 = np.cov(x, rowvar=False)
    cov2 = np.cov(y, rowvar=False)

    def sqrtm(mat: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eigh(mat)
        vals = np.clip(vals, 0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T

    covmean = sqrtm(cov1) @ sqrtm(cov2)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))


def time_freq_metrics(real_x: np.ndarray, gen_x: np.ndarray, fs: int) -> Dict:
    """
    Metrics for unpaired generation.

    IMPORTANT: Real and synthetic samples are not aligned one-to-one. Per-sample MAE is not
    meaningful. We report:
    - `mae`: MAE between feature-means (summary statistics)
    - `psd`: PSD distance between mean waveforms
    - `lf_hf`: LF/HF ratio on synthetic mean waveform (compat)
    - `mmd_rbf`, `fid_like`: distribution distances on time+freq features
    """
    n = min(len(real_x), len(gen_x))
    real_x = real_x[:n]
    gen_x = gen_x[:n]

    r_feat = time_freq_features(real_x, fs=fs)
    g_feat = time_freq_features(gen_x, fs=fs)
    mae_feat = float(np.mean(np.abs(r_feat.mean(axis=0) - g_feat.mean(axis=0))))

    ch = real_x.shape[1]
    psd_vals = []
    lf_hf_vals = []
    for i in range(ch):
        psd_vals.append(psd_distance(real_x[:, i, :].mean(axis=0), gen_x[:, i, :].mean(axis=0), fs))
        lf_hf_vals.append(lf_hf_ratio(gen_x[:, i, :].mean(axis=0), fs))

    return {
        "mae": mae_feat,
        "psd": float(np.mean(psd_vals)),
        "lf_hf": float(np.mean(lf_hf_vals)),
        "mmd_rbf": mmd_rbf(r_feat, g_feat),
        "fid_like": frechet_like(r_feat, g_feat),
    }


def text_prototype_separability(texts: list[str], labels: np.ndarray) -> float:
    if len(texts) == 0 or labels is None or len(texts) != len(labels):
        return 0.0
    label_tokens = {}
    for text, label in zip(texts, labels):
        label_tokens.setdefault(int(label), []).append(set(tokenize_text(text)))
    if len(label_tokens) < 2:
        return 0.0
    centroids = {}
    vocab = sorted({tok for groups in label_tokens.values() for doc in groups for tok in doc})
    tok_map = {tok: i for i, tok in enumerate(vocab)}
    for label, docs in label_tokens.items():
        arr = np.zeros((len(docs), len(vocab)), dtype=np.float32)
        for i, doc in enumerate(docs):
            for tok in doc:
                arr[i, tok_map[tok]] = 1.0
        centroids[label] = arr.mean(axis=0)
    labels_sorted = sorted(centroids)
    inter = []
    intra = []
    for label in labels_sorted:
        intra.append(float(np.mean(np.abs(centroids[label] - centroids[label].mean()))))
    for i, li in enumerate(labels_sorted):
        for lj in labels_sorted[i + 1 :]:
            inter.append(float(np.mean(np.abs(centroids[li] - centroids[lj]))))
    return float(np.mean(inter) / (np.mean(intra) + 1e-6))


def recovered_text_consistency(reference_texts: list[str], recovered_texts: list[str]) -> float:
    if not reference_texts or not recovered_texts:
        return 0.0
    n = min(len(reference_texts), len(recovered_texts))
    scores = []
    for ref, rec in zip(reference_texts[:n], recovered_texts[:n]):
        a = set(tokenize_text(ref))
        b = set(tokenize_text(rec))
        union = len(a | b)
        scores.append(1.0 if union == 0 else len(a & b) / union)
    return float(np.mean(scores))


def cross_dataset_semantic_stability(source_texts: list[str], target_texts: list[str]) -> float:
    return recovered_text_consistency(source_texts, target_texts)


def _extract_features(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x).float())
    loader = DataLoader(ds, batch_size=cfg_eval_batch(model), shuffle=False)
    model.eval()
    feats = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            if hasattr(model, 'extract_features'):
                h = model.extract_features(xb)
            else:
                h = model.net(xb).squeeze(-1)
            feats.append(h.cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)


def diversity_diagnostics(
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    real_x: np.ndarray,
    real_y: np.ndarray,
    device: torch.device,
) -> Dict:
    feat_model = StrongCNN(in_channels=real_x.shape[1]).to(device)
    real_feat = _extract_features(feat_model, real_x, device)
    synth_feat = _extract_features(feat_model, synth_x, device)

    def _class_stats(feats: np.ndarray, labels: np.ndarray, cls: int):
        mask = labels == cls
        if not np.any(mask):
            return None, 0.0
        sub = feats[mask]
        centroid = sub.mean(axis=0)
        var = float(np.mean(np.var(sub, axis=0)))
        return centroid, var

    c0_syn, v0_syn = _class_stats(synth_feat, synth_y, 0)
    c1_syn, v1_syn = _class_stats(synth_feat, synth_y, 1)
    c0_real, v0_real = _class_stats(real_feat, real_y, 0)
    c1_real, v1_real = _class_stats(real_feat, real_y, 1)

    def _dist(a, b):
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a - b))

    collapse_ratio = None
    real_var = (v0_real + v1_real) / 2.0 if (v0_real > 0 and v1_real > 0) else None
    syn_var = (v0_syn + v1_syn) / 2.0 if (v0_syn > 0 and v1_syn > 0) else None
    if real_var and real_var > 0 and syn_var is not None:
        collapse_ratio = float(syn_var / real_var)

    return {
        'syn_var_class0': v0_syn,
        'syn_var_class1': v1_syn,
        'real_var_class0': v0_real,
        'real_var_class1': v1_real,
        'syn_real_centroid_dist_class0': _dist(c0_syn, c0_real),
        'syn_real_centroid_dist_class1': _dist(c1_syn, c1_real),
        'syn_interclass_centroid_dist': _dist(c0_syn, c1_syn),
        'real_interclass_centroid_dist': _dist(c0_real, c1_real),
        'collapse_ratio': collapse_ratio,
    }


def tstr_strong_eval(
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    real_x: np.ndarray,
    real_y: np.ndarray,
    cfg: Dict,
    device: torch.device,
) -> Dict:
    _set_eval_seed(cfg, salt=101)
    model = train_classifier(
        synth_x,
        synth_y,
        cfg,
        device=device,
        model_cls=StrongCNN,
        epochs=int(cfg.get('eval', {}).get('strong_classifier_epochs', cfg['eval'].get('classifier_epochs', 30))),
        lr=float(cfg.get('eval', {}).get('strong_classifier_lr', cfg['eval'].get('classifier_lr', 1e-3))),
    )
    acc, f1 = evaluate_classifier(model, real_x, real_y, device=device)
    return {'accuracy': acc, 'f1': f1}


def tstr_eval(
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    real_x: np.ndarray,
    real_y: np.ndarray,
    cfg: Dict,
    device: torch.device,
    *,
    val_x: np.ndarray | None = None,
    val_y: np.ndarray | None = None,
) -> Dict:
    _set_eval_seed(cfg, salt=0)
    y_train = synth_y
    flipped = False
    align_meta = None
    align = bool(cfg.get("eval", {}).get("align_labels", False))
    num_classes = int(cfg.get("model", {}).get("num_classes", len(np.unique(real_y))))
    if align and num_classes == 2 and val_x is not None and val_y is not None:
        model = train_classifier(synth_x, synth_y, cfg, device=device)
        acc, _ = evaluate_classifier(model, val_x, val_y, device=device)
        model_flip = train_classifier(synth_x, 1 - synth_y, cfg, device=device)
        acc_flip, _ = evaluate_classifier(model_flip, val_x, val_y, device=device)
        if acc_flip > acc:
            y_train = 1 - synth_y
            flipped = True
        align_meta = {
            "ran": True,
            "flipped": flipped,
            "val_acc": float(acc),
            "val_acc_flip": float(acc_flip),
        }
    model = train_classifier(synth_x, y_train, cfg, device=device)
    acc, f1 = evaluate_classifier(model, real_x, real_y, device=device)
    out = {"accuracy": acc, "f1": f1}
    if align_meta is not None:
        out["label_alignment"] = align_meta
    return out


def inject_acc_noise(x: np.ndarray, acc: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.randn(*x.shape) * std
    if acc is None:
        return x + noise
    if acc.ndim == 4 and acc.shape[1] == 1 and acc.shape[3] == 3:
        acc = acc.squeeze(1).transpose(0, 2, 1)
    if acc.ndim == 3 and acc.shape[1] == 3:
        acc_mag = np.linalg.norm(acc, axis=1)
    else:
        acc_mag = acc
    acc_mag = (acc_mag - acc_mag.mean()) / (acc_mag.std() + 1e-6)
    scale = acc_mag / (np.max(np.abs(acc_mag)) + 1e-6)
    if scale.ndim == 2:
        scale = scale[:, np.newaxis, :]
    return x + noise * scale


def inject_burst_noise(x: np.ndarray, severity: float) -> np.ndarray:
    out = x.copy()
    burst_len = max(4, int(x.shape[-1] * max(0.02, 0.06 * severity)))
    n_bursts = max(1, int(round(1 + 3 * severity)))
    for i in range(out.shape[0]):
        for _ in range(n_bursts):
            start = np.random.randint(0, max(1, out.shape[-1] - burst_len))
            amp = (0.5 + np.random.rand()) * severity
            out[i, :, start : start + burst_len] += np.random.randn(out.shape[1], burst_len) * amp
    return out


def inject_spike_dropout(x: np.ndarray, severity: float) -> np.ndarray:
    out = x.copy()
    spike_count = max(1, int(round(5 * severity)))
    drop_count = max(1, int(round(2 * severity)))
    for i in range(out.shape[0]):
        for _ in range(spike_count):
            t = np.random.randint(0, out.shape[-1])
            out[i, :, t] += np.random.randn(out.shape[1]) * (1.5 * severity)
        for _ in range(drop_count):
            start = np.random.randint(0, max(1, out.shape[-1] - 8))
            width = np.random.randint(4, 12)
            out[i, :, start : start + width] = 0.0
    return out


def inject_baseline_wander(x: np.ndarray, severity: float, fs: int) -> np.ndarray:
    out = x.copy()
    t = np.arange(out.shape[-1], dtype=np.float32) / float(fs)
    freq = 0.05 + 0.25 * severity
    drift = np.sin(2.0 * np.pi * freq * t)[None, None, :]
    scale = (0.1 + 0.4 * severity) * np.std(out, axis=-1, keepdims=True)
    return out + drift * scale


def inject_time_jitter(x: np.ndarray, severity: float) -> np.ndarray:
    out = x.copy()
    max_shift = max(1, int(round(6 * severity)))
    for i in range(out.shape[0]):
        shifts = np.random.randint(-max_shift, max_shift + 1, size=out.shape[1])
        for c, shift in enumerate(shifts):
            out[i, c] = np.roll(out[i, c], shift)
    return out


def inject_motion_artifact_surrogate(x: np.ndarray, acc: np.ndarray, severity: float, fs: int) -> np.ndarray:
    base = inject_acc_noise(x, acc, std=max(severity, 1e-4))
    burst = inject_burst_noise(base, severity=severity)
    wander = inject_baseline_wander(burst, severity=0.5 * severity, fs=fs)
    return wander


def apply_corruption(
    x: np.ndarray,
    acc: np.ndarray,
    kind: str,
    severity: float,
    fs: int,
) -> np.ndarray:
    name = str(kind).lower()
    if name == "gaussian":
        return inject_acc_noise(x, acc, std=severity)
    if name == "burst":
        return inject_burst_noise(x, severity)
    if name == "spike_dropout":
        return inject_spike_dropout(x, severity)
    if name == "baseline_wander":
        return inject_baseline_wander(x, severity, fs)
    if name == "time_jitter":
        return inject_time_jitter(x, severity)
    if name == "motion":
        return inject_motion_artifact_surrogate(x, acc, severity, fs)
    raise ValueError(f"Unknown corruption kind: {kind}")


def _moving_average_denoise(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        for c in range(x.shape[1]):
            out[i, c] = np.convolve(x_pad[i, c], kernel, mode="valid")
    return out


def robustness_eval(
    synth_x: np.ndarray,
    synth_y: np.ndarray,
    real_x: np.ndarray,
    real_y: np.ndarray,
    acc: np.ndarray,
    cfg: Dict,
    device: torch.device,
) -> Dict:
    _set_eval_seed(cfg, salt=1)
    classifier = train_classifier(synth_x, synth_y, cfg, device=device)
    fs = int(cfg.get("data", {}).get("target_fs", 64))
    corruption_kinds = cfg["eval"].get(
        "robustness_corruptions",
        ["gaussian", "motion", "burst", "baseline_wander", "spike_dropout", "time_jitter"],
    )
    severities = cfg["eval"].get("robustness_severities", [cfg["eval"].get("noise_acc_std", 0.5), 1.0])
    window = int(cfg["eval"].get("denoise_window", 9))
    per_corruption = {}
    noisy_scores = []
    restored_scores = []
    for kind in corruption_kinds:
        curve = []
        for severity in severities:
            noisy = apply_corruption(real_x, acc, kind=kind, severity=float(severity), fs=fs)
            acc_noisy, _ = evaluate_classifier(classifier, noisy, real_y, device=device)
            restored = _moving_average_denoise(noisy, window)
            acc_restored, _ = evaluate_classifier(classifier, restored, real_y, device=device)
            curve.append(
                {
                    "severity": float(severity),
                    "noisy_acc": float(acc_noisy),
                    "restored_acc": float(acc_restored),
                }
            )
            noisy_scores.append(float(acc_noisy))
            restored_scores.append(float(acc_restored))
        per_corruption[str(kind)] = curve
    default_curve = per_corruption.get("gaussian") or next(iter(per_corruption.values()))
    return {
        "noisy_acc": float(default_curve[0]["noisy_acc"]),
        "restored_acc": float(default_curve[0]["restored_acc"]),
        "mean_noisy_acc": float(np.mean(noisy_scores)) if noisy_scores else 0.0,
        "mean_restored_acc": float(np.mean(restored_scores)) if restored_scores else 0.0,
        "corruptions": per_corruption,
    }
