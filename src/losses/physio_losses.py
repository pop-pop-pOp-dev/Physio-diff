from typing import Optional

import torch
import torch.nn.functional as F


def loss_simple(eps: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean((eps - eps_pred) ** 2)


def _diff(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    out = x
    for _ in range(int(order)):
        out = out[..., 1:] - out[..., :-1]
    return out


def _avg_pool_same(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)


def _peak_envelope(x: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    pad = kernel_size // 2
    max_env = F.max_pool1d(x, kernel_size=kernel_size, stride=1, padding=pad)
    mean_env = _avg_pool_same(x, kernel_size=kernel_size)
    return torch.relu(max_env - mean_env)


def multi_scale_time_loss(
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    channel_index: int,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    losses = []
    x_true = x0[:, channel_index : channel_index + 1, :]
    x_pred = x0_hat[:, channel_index : channel_index + 1, :]
    for scale in scales:
        if scale <= 1:
            losses.append(torch.mean(torch.abs(x_true - x_pred)))
        else:
            losses.append(
                torch.mean(
                    torch.abs(
                        F.avg_pool1d(x_true, kernel_size=scale, stride=scale)
                        - F.avg_pool1d(x_pred, kernel_size=scale, stride=scale)
                    )
                )
            )
    return torch.stack(losses).mean()


def loss_kin(
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    eda_index: int = 0,
    smooth_weight: float = 1.0,
    peak_weight: float = 1.0,
    multi_scale_weight: float = 0.5,
    clamp_max: float = 1000.0,
) -> torch.Tensor:
    eda_true = x0[:, eda_index : eda_index + 1, :]
    eda_pred = x0_hat[:, eda_index : eda_index + 1, :]
    accel = _diff(eda_pred, order=2)
    smooth = torch.mean(accel ** 2)
    peak_true = _peak_envelope(eda_true)
    peak_pred = _peak_envelope(eda_pred)
    peak = torch.mean(torch.abs(peak_true - peak_pred))
    ms = multi_scale_time_loss(x0, x0_hat, channel_index=eda_index)
    total = smooth_weight * smooth + peak_weight * peak + multi_scale_weight * ms
    return torch.clamp(total, max=clamp_max)


def loss_freq(
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    bvp_index: int = 1,
    eps: float = 1e-6,
    phase_weight: float = 1.0,
    time_weight: float = 0.5,
    feature_weight: float = 0.25,
) -> torch.Tensor:
    x_true = x0[:, bvp_index, :]
    x_pred = x0_hat[:, bvp_index, :]
    fft_true = torch.fft.rfft(x_true, dim=-1)
    fft_pred = torch.fft.rfft(x_pred, dim=-1)
    mag_true = torch.abs(fft_true)[..., 1:]
    mag_pred = torch.abs(fft_pred)[..., 1:]
    log_true = torch.log(mag_true + eps)
    log_pred = torch.log(mag_pred + eps)
    mag_loss = torch.mean((log_true - log_pred) ** 2)
    phase_true = torch.angle(fft_true)[..., 1:]
    phase_pred = torch.angle(fft_pred)[..., 1:]
    phase_diff = 1.0 - torch.cos(phase_true - phase_pred)
    phase_w = mag_true / (mag_true.mean(dim=-1, keepdim=True) + eps)
    phase_loss = torch.mean(phase_diff * phase_w)
    time_align = multi_scale_time_loss(x0, x0_hat, channel_index=bvp_index)
    feat_true = torch.stack(
        [
            x_true.mean(dim=-1),
            x_true.std(dim=-1),
            torch.sqrt(torch.mean(x_true**2, dim=-1) + eps),
            mag_true.mean(dim=-1),
        ],
        dim=-1,
    )
    feat_pred = torch.stack(
        [
            x_pred.mean(dim=-1),
            x_pred.std(dim=-1),
            torch.sqrt(torch.mean(x_pred**2, dim=-1) + eps),
            mag_pred.mean(dim=-1),
        ],
        dim=-1,
    )
    feat_loss = torch.mean((feat_true - feat_pred) ** 2)
    return mag_loss + phase_weight * phase_loss + time_weight * time_align + feature_weight * feat_loss


def feature_anchor_loss(
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    channel_index: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    x_true = x0[:, channel_index, :]
    x_pred = x0_hat[:, channel_index, :]
    true_feats = torch.stack(
        [
            x_true.mean(dim=-1),
            x_true.std(dim=-1),
            x_true.max(dim=-1).values,
            x_true.min(dim=-1).values,
            torch.sqrt(torch.mean(x_true**2, dim=-1) + eps),
        ],
        dim=-1,
    )
    pred_feats = torch.stack(
        [
            x_pred.mean(dim=-1),
            x_pred.std(dim=-1),
            x_pred.max(dim=-1).values,
            x_pred.min(dim=-1).values,
            torch.sqrt(torch.mean(x_pred**2, dim=-1) + eps),
        ],
        dim=-1,
    )
    return torch.mean((true_feats - pred_feats) ** 2)


def total_loss(
    eps: torch.Tensor,
    eps_pred: torch.Tensor,
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    w_simple: float,
    w_kin: float,
    w_freq: float,
    w_feat: float = 0.0,
    eda_index: int = 0,
    bvp_index: int = 1,
) -> torch.Tensor:
    l_simple = loss_simple(eps, eps_pred)
    l_kin = loss_kin(x0, x0_hat, eda_index=eda_index)
    l_freq = loss_freq(x0, x0_hat, bvp_index=bvp_index)
    l_feat = feature_anchor_loss(x0, x0_hat, channel_index=eda_index)
    return w_simple * l_simple + w_kin * l_kin + w_freq * l_freq + w_feat * l_feat
