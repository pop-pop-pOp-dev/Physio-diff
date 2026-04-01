from typing import Tuple

import numpy as np
import torch
from scipy.signal import welch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(x - y)))


def dtw_distance(x: np.ndarray, y: np.ndarray, downsample: int = 10) -> float:
    try:
        from fastdtw import fastdtw
    except ImportError:
        fastdtw = None
    if downsample > 1:
        x = x[::downsample]
        y = y[::downsample]
    if fastdtw is not None:
        dist, _ = fastdtw(x, y)
        return float(dist)
    n, m = len(x), len(y)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.fc(h)


def train_classifier(
    dataset: Dataset,
    in_channels: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleCNN(in_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_classifier_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += y.numel()
    return correct / max(1, total)


def evaluate_classifier(model: nn.Module, dataset: Dataset, device: torch.device) -> float:
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return evaluate_classifier_on_loader(model, loader, device)


def evaluate_comprehensive(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1
