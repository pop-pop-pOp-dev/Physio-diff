from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.models.physio_diff import PhysioDiffusion
from src.models.physio_renderer import PhysioRenderer


class MechanisticPhysioDiffusion(PhysioDiffusion):
    """
    PhysioDiffusion with an additional mechanistic latent projection branch.
    The diffusion state remains in signal space for compatibility, while
    mechanistic consistency is enforced through renderer-guided losses.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        kernel_size: int,
        dilation_cycle: int,
        embedding_dim: int,
        num_classes: int,
        timesteps: int,
        schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        dropout: float = 0.0,
        denoiser_type: str = "dilated",
        conditioning_mode: str = "additive",
        use_condition_tokens: bool = False,
        band_split_mode: str = "none",
        branch_hidden_multiplier: float = 2.0,
        patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            depth=depth,
            kernel_size=kernel_size,
            dilation_cycle=dilation_cycle,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            timesteps=timesteps,
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            dropout=dropout,
            denoiser_type=denoiser_type,
            conditioning_mode=conditioning_mode,
            use_condition_tokens=use_condition_tokens,
            band_split_mode=band_split_mode,
            branch_hidden_multiplier=branch_hidden_multiplier,
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
        )
        self.renderer = PhysioRenderer(smooth_kernel=31)
        self.latent_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1),
        )

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = super().predict_eps(
            x_t,
            t,
            y,
            domain_ids=domain_ids,
            subject_ids=subject_ids,
            text_embedding=text_embedding,
        )
        mech = self.latent_proj(x_t)
        return base + 0.1 * mech

    def mechanistic_loss(self, x_true: torch.Tensor, x_pred: torch.Tensor, eda_index: int, bvp_index: int) -> torch.Tensor:
        return self.renderer.consistency_loss(
            x_true,
            x_pred,
            eda_index=eda_index,
            bvp_index=bvp_index,
        )
