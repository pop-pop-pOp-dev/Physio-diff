from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class MechanisticComponents:
    eda_tonic: torch.Tensor
    eda_phasic: torch.Tensor
    bvp_rhythm: torch.Tensor
    bvp_morph: torch.Tensor


class PhysioRenderer(torch.nn.Module):
    """
    Lightweight differentiable decomposition/reconstruction block.
    It provides mechanistic factors used for auxiliary constraints.
    """

    def __init__(self, smooth_kernel: int = 31):
        super().__init__()
        self.smooth_kernel = int(max(3, smooth_kernel // 2 * 2 + 1))

    def _smooth(self, x: torch.Tensor, kernel: int) -> torch.Tensor:
        kernel = int(max(3, kernel // 2 * 2 + 1))
        pad = kernel // 2
        return F.avg_pool1d(x, kernel_size=kernel, stride=1, padding=pad)

    def decompose(self, x: torch.Tensor, eda_index: int, bvp_index: int) -> MechanisticComponents:
        eda = x[:, eda_index : eda_index + 1, :]
        bvp = x[:, bvp_index : bvp_index + 1, :]
        eda_tonic = self._smooth(eda, self.smooth_kernel)
        eda_phasic = eda - eda_tonic
        bvp_rhythm = self._smooth(bvp, max(9, self.smooth_kernel // 2))
        bvp_morph = bvp - bvp_rhythm
        return MechanisticComponents(
            eda_tonic=eda_tonic,
            eda_phasic=eda_phasic,
            bvp_rhythm=bvp_rhythm,
            bvp_morph=bvp_morph,
        )

    def reconstruct(self, comp: MechanisticComponents) -> Dict[str, torch.Tensor]:
        eda = comp.eda_tonic + comp.eda_phasic
        bvp = comp.bvp_rhythm + comp.bvp_morph
        return {"eda": eda, "bvp": bvp}

    def consistency_loss(self, x_true: torch.Tensor, x_pred: torch.Tensor, eda_index: int, bvp_index: int) -> torch.Tensor:
        c_true = self.decompose(x_true, eda_index=eda_index, bvp_index=bvp_index)
        c_pred = self.decompose(x_pred, eda_index=eda_index, bvp_index=bvp_index)
        loss = (
            torch.mean(torch.abs(c_true.eda_tonic - c_pred.eda_tonic))
            + torch.mean(torch.abs(c_true.eda_phasic - c_pred.eda_phasic))
            + torch.mean(torch.abs(c_true.bvp_rhythm - c_pred.bvp_rhythm))
            + torch.mean(torch.abs(c_true.bvp_morph - c_pred.bvp_morph))
        )
        return loss
