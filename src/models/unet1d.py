from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import AdaGN, SinusoidalTimeEmbedding


class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = AdaGN(out_channels, cond_dim)
        self.norm2 = AdaGN(out_channels, cond_dim)
        self.dropout = nn.Dropout(dropout)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = torch.relu(self.norm1(h, cond))
        h = self.dropout(h)
        h = self.conv2(h)
        h = torch.relu(self.norm2(h, cond))
        return self.skip(x) + h


class ConditionTokenMixer(nn.Module):
    def __init__(self, embedding_dim: int, num_tokens: int = 5):
        super().__init__()
        self.token_norm = nn.LayerNorm(embedding_dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(embedding_dim * num_tokens),
            nn.Linear(embedding_dim * num_tokens, embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.residual = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, tokens: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack(tokens, dim=1)
        normed = self.token_norm(stacked)
        flat = normed.reshape(normed.shape[0], -1)
        pooled = normed.mean(dim=1)
        return self.mix(flat) + self.residual(pooled), normed


class ConditionedAttention1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = max(1, min(int(num_heads), self.channels))
        while self.channels % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels, affine=False)
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Linear(cond_dim, channels)
        self.v_proj = nn.Linear(cond_dim, channels)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor | None) -> torch.Tensor:
        if cond_tokens is None:
            return x
        b, c, seq_len = x.shape
        heads = self.num_heads
        head_dim = c // heads
        q = self.q_proj(self.norm(x)).transpose(1, 2).reshape(b, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(cond_tokens).reshape(b, cond_tokens.shape[1], heads, head_dim).transpose(1, 2)
        v = self.v_proj(cond_tokens).reshape(b, cond_tokens.shape[1], heads, head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(max(head_dim, 1)))
        attn = torch.softmax(attn, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).reshape(b, seq_len, c).transpose(1, 2)
        return x + self.out_proj(context)


class NoOpAttention1D(nn.Module):
    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor | None = None) -> torch.Tensor:
        return x


class MultiBandStem1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, *, branch_hidden_multiplier: float = 2.0, dropout: float = 0.0):
        super().__init__()
        branch_channels = max(channels, int(round(channels * float(branch_hidden_multiplier))))
        self.lowpass = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.low_proj = nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1)
        self.high_proj = nn.Conv1d(channels, branch_channels, kernel_size=3, padding=1)
        self.low_block = ResBlock1D(branch_channels, branch_channels, cond_dim, dropout=dropout)
        self.high_block = ResBlock1D(branch_channels, branch_channels, cond_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        low = self.lowpass(x)
        high = x - low
        low = self.low_block(self.low_proj(low), cond)
        high = self.high_block(self.high_proj(high), cond)
        return self.fuse(torch.cat([low, high], dim=1))


class MultiBandHead1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.lowpass = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
        self.low_block = ResBlock1D(channels, channels, cond_dim, dropout=dropout)
        self.high_block = ResBlock1D(channels, channels, cond_dim, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        low = self.lowpass(x)
        high = x - low
        low = self.low_block(low, cond)
        high = self.high_block(high, cond)
        return self.fuse(torch.cat([low, high], dim=1))


class _ConditionedUNetBase(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        conditioning_mode: str = "additive",
        use_condition_tokens: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.conditioning_mode = str(conditioning_mode or "additive").lower()
        self.use_condition_tokens = bool(use_condition_tokens)
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
        self.token_mixer = ConditionTokenMixer(embedding_dim, num_tokens=5)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def _condition(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        domain_ids: torch.Tensor | None,
        subject_ids: torch.Tensor | None,
        text_embedding: torch.Tensor | None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        t_emb = self.time_mlp(self.time_embed(t))
        if y is None:
            y = torch.full((x.shape[0],), self.num_classes, device=x.device, dtype=torch.long)
        y_emb = self.label_embed(y)
        if domain_ids is None:
            domain_ids = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        if subject_ids is None:
            subject_ids = torch.zeros((x.shape[0],), device=x.device, dtype=torch.long)
        d_emb = self.domain_embed(torch.clamp(domain_ids, min=0, max=self.domain_embed.num_embeddings - 1))
        s_emb = self.subject_embed(torch.clamp(subject_ids, min=0, max=self.subject_embed.num_embeddings - 1))
        txt_emb = torch.zeros_like(t_emb) if text_embedding is None else self.text_proj(text_embedding)
        pooled_tokens = [t_emb, y_emb, d_emb, s_emb, txt_emb]
        if self.conditioning_mode == "additive":
            cond = self.cond_proj(t_emb + y_emb + d_emb + s_emb + txt_emb)
            return cond, None, text_tokens, text_mask
        cond, cond_tokens = self.token_mixer(pooled_tokens)
        cond = self.cond_proj(cond)
        if self.use_condition_tokens:
            return cond, cond_tokens, text_tokens, text_mask
        return cond, None, text_tokens, text_mask


class UNetDenoiser1D(_ConditionedUNetBase):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
        conditioning_mode: str = "additive",
        use_condition_tokens: bool = False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            conditioning_mode=conditioning_mode,
            use_condition_tokens=use_condition_tokens,
        )
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        ch1 = hidden_channels
        ch2 = hidden_channels * 2
        ch3 = hidden_channels * 4

        self.down1 = ResBlock1D(ch1, ch1, embedding_dim, dropout=dropout)
        self.downsample1 = nn.Conv1d(ch1, ch2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock1D(ch2, ch2, embedding_dim, dropout=dropout)
        self.downsample2 = nn.Conv1d(ch2, ch3, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock1D(ch3, ch3, embedding_dim, dropout=dropout)
        self.mid_attn = ConditionedAttention1D(ch3, embedding_dim) if self.conditioning_mode == "crossattn" else NoOpAttention1D()

        self.upsample2 = nn.ConvTranspose1d(ch3, ch2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock1D(ch2 + ch2, ch2, embedding_dim, dropout=dropout)
        self.upsample1 = nn.ConvTranspose1d(ch2, ch1, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock1D(ch1 + ch1, ch1, embedding_dim, dropout=dropout)

        self.output_proj = nn.Conv1d(ch1, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        domain_ids: torch.Tensor | None = None,
        subject_ids: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        cond, cond_tokens, _, _ = self._condition(x, t, y, domain_ids, subject_ids, text_embedding, text_tokens, text_mask)

        h1 = self.down1(x, cond)
        d1 = self.downsample1(h1)
        h2 = self.down2(d1, cond)
        d2 = self.downsample2(h2)

        m = self.mid(d2, cond)
        m = self.mid_attn(m, cond_tokens)

        u2 = self.upsample2(m)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up2(u2, cond)
        u1 = self.upsample1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1(u1, cond)
        return self.output_proj(u1)


class PhysioUNetDenoiser1D(_ConditionedUNetBase):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
        conditioning_mode: str = "adagn",
        use_condition_tokens: bool = True,
        band_split_mode: str = "multires",
        branch_hidden_multiplier: float = 2.0,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            conditioning_mode=conditioning_mode,
            use_condition_tokens=use_condition_tokens,
        )
        self.band_split_mode = str(band_split_mode or "multires").lower()
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        ch1 = hidden_channels
        ch2 = hidden_channels * 2
        ch3 = hidden_channels * 4

        if self.band_split_mode == "multires":
            self.band_stem = MultiBandStem1D(
                ch1,
                embedding_dim,
                branch_hidden_multiplier=branch_hidden_multiplier,
                dropout=dropout,
            )
            self.band_head = MultiBandHead1D(ch1, embedding_dim, dropout=dropout)
        else:
            self.band_stem = nn.Identity()
            self.band_head = nn.Identity()

        self.down1 = ResBlock1D(ch1, ch1, embedding_dim, dropout=dropout)
        self.downsample1 = nn.Conv1d(ch1, ch2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock1D(ch2, ch2, embedding_dim, dropout=dropout)
        self.downsample2 = nn.Conv1d(ch2, ch3, kernel_size=4, stride=2, padding=1)

        self.mid = ResBlock1D(ch3, ch3, embedding_dim, dropout=dropout)
        self.mid_attn = ConditionedAttention1D(ch3, embedding_dim) if self.conditioning_mode == "crossattn" else NoOpAttention1D()

        self.upsample2 = nn.ConvTranspose1d(ch3, ch2, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock1D(ch2 + ch2, ch2, embedding_dim, dropout=dropout)
        self.upsample1 = nn.ConvTranspose1d(ch2, ch1, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock1D(ch1 + ch1, ch1, embedding_dim, dropout=dropout)

        self.output_proj = nn.Conv1d(ch1, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        domain_ids: torch.Tensor | None = None,
        subject_ids: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        cond, cond_tokens, _, _ = self._condition(x, t, y, domain_ids, subject_ids, text_embedding, text_tokens, text_mask)
        x = self.band_stem(x, cond) if isinstance(self.band_stem, MultiBandStem1D) else self.band_stem(x)

        h1 = self.down1(x, cond)
        d1 = self.downsample1(h1)
        h2 = self.down2(d1, cond)
        d2 = self.downsample2(h2)

        m = self.mid(d2, cond)
        m = self.mid_attn(m, cond_tokens)

        u2 = self.upsample2(m)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up2(u2, cond)
        u1 = self.upsample1(u2)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up1(u1, cond)
        u1 = self.band_head(u1, cond) if isinstance(self.band_head, MultiBandHead1D) else self.band_head(u1)
        return self.output_proj(u1)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock1D(nn.Module):
    def __init__(self, dim: int, cond_dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaptiveLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveLayerNorm(dim, cond_dim)
        hidden = int(dim * float(mlp_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.cond_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        cond_tokens: torch.Tensor | None,
        text_tokens: torch.Tensor | None,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x, cond)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        if cond_tokens is not None:
            cross_in = self.norm1(x, cond)
            cross_out, _ = self.cond_cross_attn(cross_in, cond_tokens, cond_tokens, need_weights=False)
            x = x + cross_out
        if text_tokens is not None:
            cross_in = self.norm1(x, cond)
            key_padding_mask = None if text_mask is None else ~text_mask.bool()
            cross_out, _ = self.text_cross_attn(
                cross_in,
                text_tokens,
                text_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + cross_out
        h = self.norm2(x, cond)
        x = x + self.mlp(h)
        return x


class PatchDiTDenoiser1D(_ConditionedUNetBase):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
        conditioning_mode: str = "crossattn",
        use_condition_tokens: bool = True,
        patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            conditioning_mode=conditioning_mode,
            use_condition_tokens=use_condition_tokens,
        )
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.patch_embed = nn.Linear(self.in_channels * self.patch_size, self.d_model)
        self.cond_to_model = nn.Linear(embedding_dim, self.d_model)
        self.token_to_model = nn.Linear(embedding_dim, self.d_model)
        self.text_token_to_model = nn.Linear(embedding_dim, self.d_model)
        self.pos_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DiTBlock1D(self.d_model, self.d_model, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = AdaptiveLayerNorm(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.in_channels * self.patch_size)
        self.register_parameter("pos_embed", nn.Parameter(torch.zeros(1, 256, self.d_model)))
        nn.init.normal_(self.pos_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, c, l = x.shape
        patch = self.patch_size
        pad_len = (patch - (l % patch)) % patch
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        n = x.shape[-1] // patch
        x = x.view(b, c, n, patch).permute(0, 2, 1, 3).reshape(b, n, c * patch)
        return x, pad_len

    def _unpatchify(self, tokens: torch.Tensor, pad_len: int, orig_len: int) -> torch.Tensor:
        b, n, _ = tokens.shape
        patch = self.patch_size
        x = tokens.view(b, n, self.in_channels, patch).permute(0, 2, 1, 3).reshape(b, self.in_channels, n * patch)
        if pad_len > 0:
            x = x[..., :orig_len]
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None,
        domain_ids: torch.Tensor | None = None,
        subject_ids: torch.Tensor | None = None,
        text_embedding: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        orig_len = x.shape[-1]
        patches, pad_len = self._patchify(x)
        cond, cond_tokens, text_tokens, text_mask = self._condition(
            x,
            t,
            y,
            domain_ids,
            subject_ids,
            text_embedding,
            text_tokens,
            text_mask,
        )
        cond = self.cond_to_model(cond)
        patch_tokens = self.patch_embed(patches)
        pos = self.pos_embed[:, : patch_tokens.shape[1], :]
        h = self.pos_dropout(patch_tokens + pos)
        if cond_tokens is not None:
            cond_tokens = self.token_to_model(cond_tokens)
        if text_tokens is not None:
            text_tokens = self.text_token_to_model(text_tokens)
        for block in self.blocks:
            h = block(h, cond, cond_tokens, text_tokens, text_mask)
        h = self.final_norm(h, cond)
        out = self.out_proj(h)
        return self._unpatchify(out, pad_len=pad_len, orig_len=orig_len)
