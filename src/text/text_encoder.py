from __future__ import annotations

import re
from typing import Iterable, List

import torch
from torch import nn


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize_text(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


class LocalTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        embedding_dim: int = 128,
        proj_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.proj_dim = int(proj_dim)
        self.token_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.token_norm = nn.LayerNorm(self.embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.proj_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.proj_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.proj_dim, self.proj_dim),
        )

    def _hash_tokens(self, texts: Iterable[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        max_len = 1
        for text in texts:
            tokens = tokenize_text(text)
            ids = [hash(tok) % self.vocab_size for tok in tokens] or [0]
            sequences.append(ids)
            max_len = max(max_len, len(ids))
        token_ids = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.float32)
        for i, ids in enumerate(sequences):
            token_ids[i, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
            mask[i, : len(ids)] = 1.0
        return token_ids, mask

    def encode_tokens(self, texts: List[str], device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.token_embed.weight.device
        token_ids, mask = self._hash_tokens(texts, device=device)
        embedded = self.token_embed(token_ids)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.norm(pooled)
        pooled_out = nn.functional.normalize(self.proj(pooled), dim=-1)
        token_hidden = self.token_norm(embedded)
        token_out = nn.functional.normalize(self.token_proj(token_hidden), dim=-1)
        return pooled_out, token_out, mask

    def forward(self, texts: List[str], device: torch.device | None = None) -> torch.Tensor:
        pooled, _, _ = self.encode_tokens(texts, device=device)
        return pooled


class PretrainedTextEncoder(nn.Module):
    """
    HuggingFace-backed text encoder with projection to diffusion embedding dim.
    Supports local model paths or hub model ids.
    """

    def __init__(
        self,
        model_name_or_path: str,
        proj_dim: int,
        *,
        pooling: str = "mean",
        max_length: int = 128,
        dropout: float = 0.1,
        trainable: bool = False,
        local_files_only: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "transformers is required for PretrainedTextEncoder. "
                "Please install `transformers` and `sentence-transformers`."
            ) from exc

        self.model_name_or_path = str(model_name_or_path)
        self.pooling = str(pooling).lower()
        self.max_length = int(max_length)
        self.trainable = bool(trainable)
        self.local_files_only = bool(local_files_only)
        self.proj_dim = int(proj_dim)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
        )
        hidden_size = int(getattr(self.backbone.config, "hidden_size"))
        self.norm = nn.LayerNorm(hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, self.proj_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_size, self.proj_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.proj_dim, self.proj_dim),
        )
        if not self.trainable:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _move_backbone_if_needed(self, device: torch.device) -> None:
        current_device = next(self.backbone.parameters()).device
        if current_device != device:
            self.backbone.to(device)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def encode_tokens(self, texts: List[str], device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.proj.parameters()).device
        self._move_backbone_if_needed(device)
        encoded = self.tokenizer(
            [str(t) for t in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if self.trainable:
            outputs = self.backbone(**encoded)
        else:
            with torch.no_grad():
                outputs = self.backbone(**encoded)
        hidden = outputs.last_hidden_state
        pooled = self._pool(hidden, encoded["attention_mask"])
        pooled = self.norm(pooled)
        pooled_out = nn.functional.normalize(self.proj(pooled), dim=-1)
        token_hidden = self.token_norm(hidden)
        token_out = nn.functional.normalize(self.token_proj(token_hidden), dim=-1)
        token_mask = encoded["attention_mask"].to(torch.float32)
        return pooled_out, token_out, token_mask

    def forward(self, texts: List[str], device: torch.device | None = None) -> torch.Tensor:
        pooled, _, _ = self.encode_tokens(texts, device=device)
        return pooled


def build_text_encoder(text_cfg: dict, proj_dim: int) -> nn.Module:
    encoder_type = str(text_cfg.get("encoder_type", "local")).lower()
    if encoder_type in {"local", "hash"}:
        return LocalTextEncoder(
            vocab_size=int(text_cfg.get("vocab_size", 4096)),
            embedding_dim=int(text_cfg.get("hidden_dim", proj_dim)),
            proj_dim=int(proj_dim),
            dropout=float(text_cfg.get("dropout", 0.1)),
        )

    common = dict(
        proj_dim=int(proj_dim),
        max_length=int(text_cfg.get("max_length", 128)),
        dropout=float(text_cfg.get("dropout", 0.1)),
        trainable=bool(text_cfg.get("trainable_pretrained", False)),
        local_files_only=bool(text_cfg.get("local_files_only", True)),
    )
    if encoder_type in {"minilm", "all-minilm-l6-v2", "all_minilm_l6_v2"}:
        model_name = text_cfg.get(
            "minilm_model_name_or_path",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        return PretrainedTextEncoder(
            model_name_or_path=str(model_name),
            pooling="mean",
            **common,
        )
    if encoder_type in {"bge", "bge-small-en-v1.5", "bge_small_en_v1_5"}:
        model_name = text_cfg.get(
            "bge_model_name_or_path",
            "BAAI/bge-small-en-v1.5",
        )
        return PretrainedTextEncoder(
            model_name_or_path=str(model_name),
            pooling="cls",
            **common,
        )
    if encoder_type in {"hf", "pretrained"}:
        model_name = text_cfg.get("model_name_or_path")
        if not model_name:
            raise ValueError("text.model_name_or_path must be set when text.encoder_type=hf")
        pooling = str(text_cfg.get("pooling", "mean"))
        return PretrainedTextEncoder(
            model_name_or_path=str(model_name),
            pooling=pooling,
            **common,
        )
    raise ValueError(f"Unknown text.encoder_type: {encoder_type}")
