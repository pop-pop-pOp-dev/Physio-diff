import math
from typing import Optional

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device).float() / (half - 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class AdaGN(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, num_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        x = self.norm(x)
        return x * (1 + scale) + shift


class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.adagn1 = AdaGN(channels, cond_dim)
        self.adagn2 = AdaGN(channels, cond_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = torch.relu(self.adagn1(h, cond))
        h = self.dropout(h)
        h = self.conv2(h)
        h = torch.relu(self.adagn2(h, cond))
        return x + h


class Denoiser1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        kernel_size: int,
        dilation_cycle: int,
        embedding_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Reserve an extra "null" label id (= num_classes) for classifier-free guidance (CFG).
        # This enables per-sample conditional dropout without changing tensor shapes.
        self.num_classes = int(num_classes)
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.time_embed = SinusoidalTimeEmbedding(embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.label_embed = nn.Embedding(num_classes + 1, embedding_dim)
        self.domain_embed = nn.Embedding(33, embedding_dim)
        self.subject_embed = nn.Embedding(513, embedding_dim)
        self.text_proj = nn.Linear(embedding_dim, embedding_dim)
        self.cond_proj = nn.Linear(embedding_dim, embedding_dim)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** (i % dilation_cycle)
            self.blocks.append(
                ResidualBlock1D(
                    channels=hidden_channels,
                    cond_dim=embedding_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        t_emb = self.time_mlp(self.time_embed(t))
        if y is None:
            # unconditional branch (CFG): all samples use the reserved null label id
            y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
        y_emb = self.label_embed(y)
        if domain_ids is None:
            domain_ids = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        if subject_ids is None:
            subject_ids = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        d_emb = self.domain_embed(torch.clamp(domain_ids, min=0, max=self.domain_embed.num_embeddings - 1))
        s_emb = self.subject_embed(torch.clamp(subject_ids, min=0, max=self.subject_embed.num_embeddings - 1))
        txt_emb = torch.zeros_like(t_emb) if text_embedding is None else self.text_proj(text_embedding)
        cond = self.cond_proj(t_emb + y_emb + d_emb + s_emb + txt_emb)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)
