from __future__ import annotations

import torch

from src.text.artifact_text import build_artifact_text


def _rand_like(x: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(x)


def apply_corruptions(x: torch.Tensor, severity: float, kinds: list[str]) -> torch.Tensor:
    out = x.clone()
    sev = float(max(0.0, severity))
    for kind in kinds:
        name = str(kind).lower()
        if name == "gaussian":
            out = out + _rand_like(out) * (0.1 * sev)
        elif name == "burst":
            length = out.shape[-1]
            width = max(4, int(0.04 * length))
            start = torch.randint(0, max(1, length - width), (out.shape[0],), device=out.device)
            for i in range(out.shape[0]):
                out[i, :, start[i] : start[i] + width] += _rand_like(out[i, :, :width]) * (0.2 * sev)
        elif name == "spike_dropout":
            mask = torch.rand_like(out) < (0.02 * sev)
            out = torch.where(mask, torch.zeros_like(out), out)
        elif name == "baseline_wander":
            t = torch.linspace(0.0, 1.0, out.shape[-1], device=out.device).view(1, 1, -1)
            drift = torch.sin(2.0 * torch.pi * (0.5 + sev) * t)
            out = out + drift * out.std(dim=-1, keepdim=True) * (0.1 + 0.3 * sev)
        elif name == "time_jitter":
            max_shift = max(1, int(round(5 * sev)))
            if max_shift > 0:
                shifts = torch.randint(-max_shift, max_shift + 1, (out.shape[0], out.shape[1]), device=out.device)
                for i in range(out.shape[0]):
                    for c in range(out.shape[1]):
                        out[i, c] = torch.roll(out[i, c], shifts=int(shifts[i, c].item()), dims=-1)
        elif name == "motion":
            out = out + _rand_like(out) * (0.15 * sev)
    return out


def describe_corruption(kind: str, severity: float) -> str:
    mapping = {
        "spike_dropout": "dropout",
        "baseline_wander": "wander",
        "time_jitter": "jitter",
    }
    sev = float(max(0.0, severity))
    severity_word = "mild" if sev < 0.35 else "moderate" if sev < 0.7 else "severe"
    return build_artifact_text(
        {
            "artifact_kind": mapping.get(str(kind).lower(), str(kind).lower()),
            "artifact_severity": severity_word,
        }
    )


def apply_corruption_with_text(x: torch.Tensor, severity: float, kind: str) -> tuple[torch.Tensor, str]:
    return apply_corruptions(x, severity=severity, kinds=[kind]), describe_corruption(kind, severity)


def severity_schedule(epoch: int, total_epochs: int, start: float = 0.1, end: float = 1.0) -> float:
    progress = min(max((float(epoch) - 1.0) / max(1.0, float(total_epochs) - 1.0), 0.0), 1.0)
    return float(start + (end - start) * progress)
