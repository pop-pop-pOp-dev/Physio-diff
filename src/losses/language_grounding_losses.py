from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor(0.0, device=x.device if x.numel() else y.device)
    return 1.0 - F.cosine_similarity(F.normalize(x, dim=-1), F.normalize(y, dim=-1), dim=-1).mean()


def prototype_alignment_loss(features: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if features.numel() == 0 or prototypes is None or prototypes.numel() == 0:
        return torch.tensor(0.0, device=features.device)
    picked = prototypes[labels]
    return cosine_alignment_loss(features, picked)


def cycle_reconstruction_loss(pred_text_embed: torch.Tensor, target_text_embed: torch.Tensor) -> torch.Tensor:
    return cosine_alignment_loss(pred_text_embed, target_text_embed)


def semantic_consistency_loss(physio_embed: torch.Tensor, semantic_embed: torch.Tensor) -> torch.Tensor:
    return cosine_alignment_loss(physio_embed, semantic_embed)


def artifact_text_loss(pred_artifact_embed: torch.Tensor, target_artifact_embed: torch.Tensor) -> torch.Tensor:
    return cosine_alignment_loss(pred_artifact_embed, target_artifact_embed)
