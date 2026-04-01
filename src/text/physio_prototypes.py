from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def encode_text_batch(text_encoder, texts: Iterable[str], device: torch.device) -> torch.Tensor:
    text_list = [str(t) for t in texts]
    if not text_list:
        dim = int(getattr(text_encoder, "proj_dim", 128))
        return torch.zeros((0, dim), device=device)
    return text_encoder(text_list, device=device)


def build_label_text_prototypes(
    text_encoder,
    texts: List[str],
    labels: List[int],
    num_classes: int,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    dim = int(getattr(text_encoder, "proj_dim", 128))
    proto = torch.zeros((int(num_classes), dim), device=device)
    counts = torch.zeros((int(num_classes),), device=device)
    if not texts:
        return proto
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(texts), step):
            end = min(len(texts), start + step)
            emb = encode_text_batch(text_encoder, texts[start:end], device=device)
            for rel_idx, label in enumerate(labels[start:end]):
                cls = int(label)
                if 0 <= cls < int(num_classes):
                    proto[cls] += emb[rel_idx]
                    counts[cls] += 1.0
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    return torch.nn.functional.normalize(proto / counts, dim=-1)


def summarize_text_prototypes(texts: List[str], labels: List[int]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for text, label in zip(texts, labels):
        key = str(int(label))
        if key not in summary:
            summary[key] = str(text)
    return summary
