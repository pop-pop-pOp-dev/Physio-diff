from typing import Optional

import numpy as np
import torch


def inject_acc_noise(
    x: np.ndarray,
    acc: Optional[np.ndarray],
    std: float,
) -> np.ndarray:
    """
    Inject noise scaled by accelerometer magnitude.

    Args:
        x: Input signal tensor (N, C, L)
        acc: Accelerometer signal tensor (N, 3, L)
        std: Standard deviation of the base noise
    """
    noise = np.random.randn(*x.shape) * std

    if acc is None:
        return x + noise

    if acc.ndim == 3 and acc.shape[1] == 3:
        acc_mag = np.linalg.norm(acc, axis=1)
    elif acc.ndim == 2 and acc.shape[1] == 3:
        acc_mag = np.linalg.norm(acc, axis=1)
    else:
        acc_mag = acc

    mean = acc_mag.mean()
    std_dev = acc_mag.std() + 1e-6
    acc_mag_norm = (acc_mag - mean) / std_dev

    scale = acc_mag_norm / (np.max(np.abs(acc_mag_norm)) + 1e-6)

    if scale.ndim == 2:
        scale = scale[:, np.newaxis, :]
    elif scale.ndim == 1:
        scale = scale[np.newaxis, np.newaxis, :]

    return x + noise * scale


def denoise_with_model(
    model,
    x_noisy: torch.Tensor,
    y: torch.Tensor,
    t_start: int = 400,
) -> torch.Tensor:
    """
    Denoise a signal using SDEdit strategy (diffuse to t_start then reverse).
    """
    device = next(model.parameters()).device
    x_noisy = x_noisy.to(device)
    y = y.to(device)
    batch_size = x_noisy.shape[0]
    t_start = int(min(max(t_start, 0), model.timesteps - 1))
    t_tensor = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
    noise = torch.randn_like(x_noisy)
    x_t = model.q_sample(x_noisy, t_tensor, noise)
    for step in reversed(range(t_start + 1)):
        t_step = torch.full((batch_size,), step, device=device, dtype=torch.long)
        x_t = model.p_sample(x_t, t_step, y)
    return x_t
