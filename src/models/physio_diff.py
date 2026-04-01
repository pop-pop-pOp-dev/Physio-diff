from typing import Optional

import torch
from torch import nn

from .blocks import Denoiser1D
from .unet1d import PatchDiTDenoiser1D, PhysioUNetDenoiser1D, UNetDenoiser1D


def _linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class PhysioDiffusion(nn.Module):
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
        super().__init__()
        self.timesteps = timesteps
        dtype = str(denoiser_type or "dilated").lower()
        if dtype in {"dilated", "wavenet", "residual"}:
            self.denoiser = Denoiser1D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                depth=depth,
                kernel_size=kernel_size,
                dilation_cycle=dilation_cycle,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif dtype in {"unet", "unet1d"}:
            self.denoiser = UNetDenoiser1D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                dropout=dropout,
                conditioning_mode=conditioning_mode,
                use_condition_tokens=use_condition_tokens,
            )
        elif dtype in {"physio_unet", "multiband_unet"}:
            self.denoiser = PhysioUNetDenoiser1D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                dropout=dropout,
                conditioning_mode=conditioning_mode,
                use_condition_tokens=use_condition_tokens,
                band_split_mode=band_split_mode,
                branch_hidden_multiplier=branch_hidden_multiplier,
            )
        elif dtype in {"patch_dit", "dit", "patch_transformer"}:
            self.denoiser = PatchDiTDenoiser1D(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                dropout=dropout,
                conditioning_mode=conditioning_mode,
                use_condition_tokens=use_condition_tokens,
                patch_size=patch_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mlp_ratio=mlp_ratio,
            )
        else:
            raise ValueError(f"Unknown denoiser_type: {denoiser_type}")
        if schedule == "linear":
            betas = _linear_beta_schedule(beta_start, beta_end, timesteps)
        else:
            betas = _cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.denoiser(
            x_t,
            t,
            y,
            domain_ids=domain_ids,
            subject_ids=subject_ids,
            text_embedding=text_embedding,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_omb = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_omb * eps) / sqrt_ab

    def _x0_dynamic_clip(self, x0_hat: torch.Tensor, quantile: float | None) -> torch.Tensor:
        """
        Dynamic thresholding / clipping to prevent sampling blow-ups.
        If quantile is None, falls back to fixed clamp [-10, 10] (previous behavior).
        """
        if quantile is None:
            return torch.clamp(x0_hat, -10.0, 10.0)
        q = float(quantile)
        if not (0.0 < q < 1.0):
            return torch.clamp(x0_hat, -10.0, 10.0)
        b = x0_hat.shape[0]
        flat = x0_hat.detach().abs().reshape(b, -1)
        s = torch.quantile(flat, q, dim=1).clamp_min(1.0).view(b, 1, 1)
        return torch.clamp(x0_hat, -s, s)

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        *,
        x0_clip_quantile: float | None = None,
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        eps = self.predict_eps(
            x_t,
            t,
            y,
            domain_ids=domain_ids,
            subject_ids=subject_ids,
            text_embedding=text_embedding,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        x0_hat = self.predict_x0(x_t, t, eps)
        x0_hat = self._x0_dynamic_clip(x0_hat, x0_clip_quantile)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar) * eps)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise

    def p_sample_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
        *,
        cfg_rescale: bool = True,
        x0_clip_quantile: float | None = None,
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classifier-free guidance (CFG) sampling step.

        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        where eps_uncond is predicted with a reserved null-label id.
        """
        if cfg_scale is None or float(cfg_scale) == 1.0:
            return self.p_sample(
                x_t,
                t,
                y,
                x0_clip_quantile=x0_clip_quantile,
                domain_ids=domain_ids,
                subject_ids=subject_ids,
                text_embedding=text_embedding,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
        # unconditional: use reserved null label id (= num_classes) for all samples
        y_null = torch.full_like(y, self.denoiser.num_classes)
        eps_uncond = self.predict_eps(
            x_t,
            t,
            y_null,
            domain_ids=domain_ids,
            subject_ids=subject_ids,
            text_embedding=text_embedding,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        eps_cond = self.predict_eps(
            x_t,
            t,
            y,
            domain_ids=domain_ids,
            subject_ids=subject_ids,
            text_embedding=text_embedding,
            text_tokens=text_tokens,
            text_mask=text_mask,
        )
        eps = eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)
        if cfg_rescale:
            # Rescale guided eps to match the conditional branch magnitude
            # (stabilizes sampling when cfg_scale>1).
            std_cond = eps_cond.std(dim=(1, 2), keepdim=True)
            std_guided = eps.std(dim=(1, 2), keepdim=True)
            eps = eps * (std_cond / (std_guided + 1e-6))

        x0_hat = self.predict_x0(x_t, t, eps)
        x0_hat = self._x0_dynamic_clip(x0_hat, x0_clip_quantile)
        beta_t = self.betas[t].view(-1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar) * eps)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(beta_t) * noise

    def sample(
        self,
        shape,
        y: Optional[torch.Tensor],
        device: torch.device,
        *,
        cfg_scale: float = 1.0,
        cfg_rescale: bool = True,
        x0_clip_quantile: float | None = 0.995,
        x_t_clip: float | None = 20.0,
        sample_steps: int | None = None,
        ddim_eta: float = 0.0,
        domain_ids: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=device)
        if sample_steps is None or int(sample_steps) >= int(self.timesteps):
            for step in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), step, device=device, dtype=torch.long)
                if y is None:
                    x = self.p_sample(
                        x,
                        t,
                        None,
                        x0_clip_quantile=x0_clip_quantile,
                        domain_ids=domain_ids,
                        subject_ids=subject_ids,
                        text_embedding=text_embedding,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                    )
                else:
                    x = self.p_sample_cfg(
                        x,
                        t,
                        y,
                        cfg_scale=cfg_scale,
                        cfg_rescale=cfg_rescale,
                        x0_clip_quantile=x0_clip_quantile,
                        domain_ids=domain_ids,
                        subject_ids=subject_ids,
                        text_embedding=text_embedding,
                        text_tokens=text_tokens,
                        text_mask=text_mask,
                    )
                if x_t_clip is not None:
                    x = torch.clamp(x, -float(x_t_clip), float(x_t_clip))
            return x

        steps = int(sample_steps)
        if steps < 2:
            steps = 2
        t_seq = torch.linspace(
            float(self.timesteps - 1),
            0.0,
            steps,
            device=device,
        )
        t_seq = torch.round(t_seq).to(torch.long)
        t_seq = torch.unique_consecutive(t_seq)
        if int(t_seq[-1].item()) != 0:
            t_seq = torch.cat([t_seq, torch.zeros((1,), device=device, dtype=torch.long)], dim=0)

        for i in range(len(t_seq)):
            t_now = int(t_seq[i].item())
            t_prev = int(t_seq[i + 1].item()) if i + 1 < len(t_seq) else -1
            t = torch.full((shape[0],), t_now, device=device, dtype=torch.long)

            if y is None or cfg_scale is None or float(cfg_scale) == 1.0:
                eps = self.predict_eps(
                    x,
                    t,
                    y if y is not None else None,
                    domain_ids=domain_ids,
                    subject_ids=subject_ids,
                    text_embedding=text_embedding,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
            else:
                y_null = torch.full_like(y, self.denoiser.num_classes)
                eps_uncond = self.predict_eps(
                    x,
                    t,
                    y_null,
                    domain_ids=domain_ids,
                    subject_ids=subject_ids,
                    text_embedding=text_embedding,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
                eps_cond = self.predict_eps(
                    x,
                    t,
                    y,
                    domain_ids=domain_ids,
                    subject_ids=subject_ids,
                    text_embedding=text_embedding,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
                eps = eps_uncond + float(cfg_scale) * (eps_cond - eps_uncond)
                if cfg_rescale:
                    std_cond = eps_cond.std(dim=(1, 2), keepdim=True)
                    std_guided = eps.std(dim=(1, 2), keepdim=True)
                    eps = eps * (std_cond / (std_guided + 1e-6))

            x0_hat = self.predict_x0(x, t, eps)
            x0_hat = self._x0_dynamic_clip(x0_hat, x0_clip_quantile)
            if t_prev < 0:
                x = x0_hat
            else:
                ab_t = self.alphas_cumprod[t_now].view(1, 1, 1)
                ab_prev = self.alphas_cumprod[t_prev].view(1, 1, 1)
                eta = float(ddim_eta)
                if eta < 0.0:
                    eta = 0.0
                if eta > 0.0:
                    sigma = eta * torch.sqrt(
                        (1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / (ab_prev + 1e-8))
                    )
                    noise = torch.randn_like(x)
                    x = (
                        torch.sqrt(ab_prev) * x0_hat
                        + torch.sqrt(torch.clamp(1.0 - ab_prev - sigma**2, min=0.0)) * eps
                        + sigma * noise
                    )
                else:
                    x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps

            if x_t_clip is not None:
                x = torch.clamp(x, -float(x_t_clip), float(x_t_clip))
        return x
