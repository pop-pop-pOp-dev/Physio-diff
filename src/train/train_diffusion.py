import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import random

from src.data.datasets import WESADDataset, make_dataloaders
from src.data.wesad import build_dataset_cache
from src.losses.language_grounding_losses import (
    artifact_text_loss,
    cycle_reconstruction_loss,
    prototype_alignment_loss,
    semantic_consistency_loss,
)
from src.losses.physio_losses import feature_anchor_loss, loss_freq, loss_kin
from src.models.mech_latent_diff import MechanisticPhysioDiffusion
from src.models.physio_diff import PhysioDiffusion
from src.models.signal_text_cycle import SignalTextCycle
from src.text.artifact_text import build_artifact_text
from src.text.physio_prototypes import build_label_text_prototypes
from src.text.text_encoder import build_text_encoder
from src.train.corruption_curriculum import apply_corruptions, severity_schedule
from src.utils.project_paths import resolve_project_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _schedule_weight(epoch: int, start: float, end: float, warmup: int, total: int) -> float:
    saturation_epoch = 30
    if epoch <= warmup:
        return start
    progress = min((epoch - warmup) / max(1, saturation_epoch), 1.0)
    return start + (end - start) * progress


class AuxClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.extract_features(x))


class EMA:
    """Simple exponential moving average (EMA) for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        d = self.decay
        state = model.state_dict()
        for k, v in state.items():
            v_cpu = v.detach().to("cpu")
            self.shadow[k].mul_(d).add_(v_cpu, alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def _train_anchor_classifier(
    loader,
    *,
    in_channels: int,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
) -> nn.Module:
    model = AuxClassifier(in_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(int(epochs)):
        for batch in loader:
            x0, y, _ = _resolve_batch_indices(batch)
            x0 = x0.to(device)
            y = y.to(device)
            logits = model(x0)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _compute_class_prototypes(loader, model: AuxClassifier, device: torch.device, num_classes: int) -> torch.Tensor:
    feats = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for batch in loader:
            x0, y, _ = _resolve_batch_indices(batch)
            x0 = x0.to(device)
            y = y.to(device)
            h = model.extract_features(x0)
            for cls in range(num_classes):
                mask = y == cls
                if bool(mask.any()):
                    feats[cls].append(h[mask])
    dim = model.fc.in_features
    proto = torch.zeros((num_classes, dim), device=device)
    for cls in range(num_classes):
        if feats[cls]:
            proto[cls] = torch.cat(feats[cls], dim=0).mean(dim=0)
    return proto


def _prototype_nce_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: Optional[torch.Tensor],
    temperature: float = 0.2,
) -> torch.Tensor:
    if prototypes is None or prototypes.numel() == 0 or features.numel() == 0:
        return torch.tensor(0.0, device=features.device)
    f = torch.nn.functional.normalize(features, dim=-1)
    p = torch.nn.functional.normalize(prototypes, dim=-1).to(dtype=f.dtype)
    logits = f @ p.t() / max(float(temperature), 1e-4)
    return torch.nn.functional.cross_entropy(logits, labels)


def _label_embedding_margin(model: PhysioDiffusion, margin: float = 1.0) -> torch.Tensor:
    weights = model.denoiser.label_embed.weight
    class_w = weights[: model.denoiser.num_classes]
    if class_w.shape[0] < 2:
        return torch.tensor(0.0, device=weights.device)
    dist = torch.cdist(class_w, class_w, p=2)
    eye = torch.eye(dist.shape[0], device=dist.device, dtype=torch.bool)
    pair_loss = torch.relu(float(margin) - dist[~eye]).mean()
    null_w = weights[model.denoiser.num_classes : model.denoiser.num_classes + 1]
    null_dist = torch.cdist(class_w, null_w, p=2)
    null_loss = torch.relu(float(margin) - null_dist).mean()
    return pair_loss + 0.5 * null_loss


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, num_domains: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def _grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, float(lambd))


def _apply_x0_constraint(x0_hat: torch.Tensor, cfg: Dict) -> torch.Tensor:
    if cfg["train"].get("disable_x0_clamp", False):
        return x0_hat
    clip_mode = str(cfg["train"].get("x0_constraint_mode", "soft")).lower()
    clip_value = float(cfg["train"].get("x0_constraint_value", 10.0))
    if clip_mode == "none":
        return x0_hat
    if clip_mode == "hard":
        return torch.clamp(x0_hat, -clip_value, clip_value)
    return torch.tanh(x0_hat / max(clip_value, 1e-6)) * clip_value


def _build_model(cfg: Dict) -> PhysioDiffusion:
    common = dict(
        in_channels=len(cfg["data"]["channels"]),
        hidden_channels=cfg["model"]["hidden_channels"],
        depth=cfg["model"]["depth"],
        kernel_size=cfg["model"]["kernel_size"],
        dilation_cycle=cfg["model"]["dilation_cycle"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_classes=cfg["model"]["num_classes"],
        timesteps=cfg["diffusion"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        dropout=cfg["model"]["dropout"],
        denoiser_type=cfg.get("model", {}).get("denoiser_type", "dilated"),
        conditioning_mode=cfg.get("model", {}).get("conditioning_mode", "additive"),
        use_condition_tokens=bool(cfg.get("model", {}).get("use_condition_tokens", False)),
        band_split_mode=cfg.get("model", {}).get("band_split_mode", "none"),
        branch_hidden_multiplier=float(cfg.get("model", {}).get("branch_hidden_multiplier", 2.0)),
        patch_size=int(cfg.get("model", {}).get("patch_size", 16)),
        d_model=int(cfg.get("model", {}).get("d_model", 256)),
        nhead=int(cfg.get("model", {}).get("nhead", 8)),
        num_layers=int(cfg.get("model", {}).get("num_layers", 6)),
        mlp_ratio=float(cfg.get("model", {}).get("mlp_ratio", 4.0)),
    )
    model_type = str(cfg.get("model", {}).get("model_type", "standard")).lower()
    if model_type in {"mechanistic", "mech", "mech_latent"}:
        return MechanisticPhysioDiffusion(**common)
    return PhysioDiffusion(**common)


def _attach_language_modules(model: PhysioDiffusion, cfg: Dict) -> PhysioDiffusion:
    embed_dim = int(cfg["model"]["embedding_dim"])
    text_cfg = dict(cfg.get("text", {}))
    for key in ("minilm_model_name_or_path", "bge_model_name_or_path", "model_name_or_path"):
        raw = text_cfg.get(key)
        if not raw:
            continue
        candidate = resolve_project_path(str(raw))
        if os.path.exists(candidate):
            text_cfg[key] = candidate
    if not hasattr(model, "text_encoder"):
        model.text_encoder = build_text_encoder(text_cfg=text_cfg, proj_dim=embed_dim)
    if not hasattr(model, "signal_text_cycle"):
        model.signal_text_cycle = SignalTextCycle(
            in_channels=len(cfg["data"]["channels"]),
            text_dim=embed_dim,
        )
    return model


def _severity_word(value: float) -> str:
    if value < 0.35:
        return "mild"
    if value < 0.7:
        return "moderate"
    return "severe"


def _corruption_key(kind: str) -> str:
    mapping = {
        "spike_dropout": "dropout",
        "baseline_wander": "wander",
        "time_jitter": "jitter",
    }
    return mapping.get(str(kind), str(kind))


def _lookup_text_conditioning(
    dataset: WESADDataset,
    indices: Optional[torch.Tensor],
    model: PhysioDiffusion,
    device: torch.device,
    fallback_labels: Optional[torch.Tensor] = None,
    artifact_kind: str = "clean",
    artifact_severity: str = "mild",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if indices is None:
        bsz = int(fallback_labels.shape[0]) if fallback_labels is not None else 1
        physio_texts = [""] * bsz
        semantic_texts = [""] * bsz
    else:
        idx_list = [int(i) for i in indices.detach().cpu().tolist()]
        physio_texts = dataset.get_text_batch(idx_list, "physio_text")
        semantic_texts = dataset.get_text_batch(idx_list, "semantic_text")
    if indices is None:
        artifact_texts = [build_artifact_text({"artifact_kind": artifact_kind, "artifact_severity": artifact_severity})]
        artifact_texts = artifact_texts * len(physio_texts)
    else:
        artifact_texts = [
            build_artifact_text({"artifact_kind": artifact_kind, "artifact_severity": artifact_severity})
            for _ in range(len(physio_texts))
        ]
    physio_embed, physio_tokens, physio_mask = model.text_encoder.encode_tokens(physio_texts, device=device)
    semantic_embed = model.text_encoder(semantic_texts, device=device)
    artifact_embed = model.text_encoder(artifact_texts, device=device)
    return physio_embed, semantic_embed, artifact_embed, physio_tokens, physio_mask


def _lookup_text_embeddings(
    dataset: WESADDataset,
    indices: Optional[torch.Tensor],
    model: PhysioDiffusion,
    device: torch.device,
    fallback_labels: Optional[torch.Tensor] = None,
    artifact_kind: str = "clean",
    artifact_severity: str = "mild",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    physio_embed, semantic_embed, artifact_embed, _, _ = _lookup_text_conditioning(
        dataset,
        indices,
        model,
        device,
        fallback_labels=fallback_labels,
        artifact_kind=artifact_kind,
        artifact_severity=artifact_severity,
    )
    return physio_embed, semantic_embed, artifact_embed


def _resolve_batch_indices(batch) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _subject_and_domain_ids(dataset: WESADDataset, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if indices is None or dataset.subjects is None:
        b = int(indices.shape[0]) if indices is not None else 1
        return (
            torch.zeros((b,), device=device, dtype=torch.long),
            torch.zeros((b,), device=device, dtype=torch.long),
        )
    subject_names = [str(dataset.subjects[int(i)]) for i in indices.detach().cpu().tolist()]
    uniq = sorted(set(subject_names))
    subj_map = {name: idx for idx, name in enumerate(uniq)}
    subject_ids = torch.tensor([subj_map[s] for s in subject_names], device=device, dtype=torch.long)
    dataset_name = str(getattr(dataset, "dataset_name", "wesad")).lower()
    domain_id_map = {
        "stress_predict": 0,
        "wesad": 1,
        "swell_kw": 2,
        "case": 3,
        "ubfc_phys": 4,
        "mahnob_hci": 5,
    }
    domain_ids = torch.full(
        (subject_ids.shape[0],),
        int(domain_id_map.get(dataset_name, 15)),
        device=device,
        dtype=torch.long,
    )
    return subject_ids, domain_ids


def _should_attach_language_modules(cfg: Dict) -> bool:
    train_cfg = cfg.get("train", {})
    return bool(
        train_cfg.get("enable_text_modules", False)
        or train_cfg.get("use_language_conditioning", False)
        or train_cfg.get("use_text_prototypes", False)
        or train_cfg.get("use_signal_text_cycle", False)
        or train_cfg.get("use_semantic_alignment", False)
        or train_cfg.get("use_artifact_text_conditioning", False)
    )


def _needs_aux_classifier(cfg: Dict) -> bool:
    train_cfg = cfg.get("train", {})
    loss_cfg = cfg.get("loss", {})
    if bool(train_cfg.get("anchor_labels", False)):
        return True
    weight_keys = ["w_feat", "w_proto", "w_domain", "w_cls", "w_cls_start", "w_cls_end"]
    return any(float(loss_cfg.get(key, 0.0)) > 0.0 for key in weight_keys)


def _normalize_training_recipe(cfg: Dict) -> Dict:
    train_cfg = cfg.setdefault("train", {})
    loss_cfg = cfg.setdefault("loss", {})
    recipe = str(train_cfg.get("training_recipe", "legacy_fullstack")).lower()
    if recipe != "main_competitive":
        return cfg

    train_cfg.setdefault("anchor_labels", False)
    train_cfg.setdefault("domain_generalization", False)
    train_cfg.setdefault("artifact_aware", False)
    train_cfg.setdefault("use_language_conditioning", False)
    train_cfg.setdefault("use_text_prototypes", False)
    train_cfg.setdefault("use_signal_text_cycle", False)
    train_cfg.setdefault("use_semantic_alignment", False)
    train_cfg.setdefault("use_artifact_text_conditioning", False)
    train_cfg.setdefault("enable_aux_classifier", False)
    train_cfg.setdefault("enable_domain_disc", False)
    train_cfg.setdefault("enable_mechanistic_branch", False)
    train_cfg.setdefault("enable_text_modules", False)

    loss_cfg.setdefault("w_kin_start", 0.0)
    loss_cfg.setdefault("w_kin_end", 0.0)
    loss_cfg.setdefault("w_freq_start", 0.0)
    loss_cfg.setdefault("w_freq_end", 0.0)
    loss_cfg.setdefault("w_cls_start", 0.0)
    loss_cfg.setdefault("w_cls_end", 0.0)
    loss_cfg.setdefault("w_cls", 0.0)
    loss_cfg.setdefault("w_proto", 0.0)
    loss_cfg.setdefault("w_embed_margin", 0.0)
    loss_cfg.setdefault("w_feat", 0.0)
    loss_cfg.setdefault("w_domain", 0.0)
    loss_cfg.setdefault("w_mech", 0.0)
    loss_cfg.setdefault("w_consistency", 0.0)
    loss_cfg.setdefault("w_text_proto", 0.0)
    loss_cfg.setdefault("w_cycle", 0.0)
    loss_cfg.setdefault("w_artifact_text", 0.0)
    loss_cfg.setdefault("w_semantic_align", 0.0)
    return cfg


def train(cfg: Dict) -> Tuple[PhysioDiffusion, str]:
    cfg = _normalize_training_recipe(cfg)
    set_seed(cfg["project"]["seed"])
    device = torch.device(cfg["project"]["device"])
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out_dir = cfg["project"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    cache_path = build_dataset_cache(
        root_dir=cfg["data"]["root_dir"],
        cache_dir=cfg["data"]["cache_dir"],
        subjects=cfg["data"]["subjects"],
        channels=cfg["data"]["channels"],
        include_acc=cfg["data"]["include_acc"],
        target_fs=cfg["data"]["target_fs"],
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["window_stride"],
        zscore=cfg["data"]["zscore"],
        processed_label_path=cfg["data"].get("processed_label_path"),
        dataset_name=cfg["data"].get("dataset_name"),
        clip_mode=cfg["data"].get("clip_mode", "hard"),
        clip_value=float(cfg["data"].get("clip_value", 5.0)),
        prebuilt_cache_path=cfg["data"].get("prebuilt_cache_path"),
        force_rebuild=bool(cfg["data"].get("force_rebuild_cache", False)),
        drop_ambiguous_windows=bool(cfg["data"].get("drop_ambiguous_windows", False)),
        label_purity_threshold=float(cfg["data"].get("label_purity_threshold", 0.8)),
        window_label_mode=str(cfg["data"].get("window_label_mode", "majority")),
        label_center_ratio=float(cfg["data"].get("label_center_ratio", 0.5)),
        positive_labels=cfg["data"].get("positive_labels"),
        negative_labels=cfg["data"].get("negative_labels"),
        ignore_labels=cfg["data"].get("ignore_labels"),
    )
    dataset = WESADDataset(cache_path)
    dataset.return_index = True
    train_loader, val_loader, _ = make_dataloaders(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        train_ratio=cfg["data"].get("train_ratio", 0.8),
        val_ratio=cfg["data"].get("val_ratio", 0.1),
        test_ratio=cfg["data"].get("test_ratio", 0.1),
        seed=cfg["project"]["seed"],
        split_strategy=cfg["data"].get("split_strategy", "random"),
        train_subjects=cfg["data"].get("train_subjects"),
        val_subjects=cfg["data"].get("val_subjects"),
        test_subjects=cfg["data"].get("test_subjects"),
        weighted_sampling=cfg["train"].get("weighted_sampling", False),
        cv_n_splits=int(cfg["data"].get("cv_n_splits", 5)),
        cv_fold_index=int(cfg["data"].get("cv_fold_index", 0)),
        val_subject_count=int(cfg["data"].get("val_subject_count", 1)),
        pin_memory=bool(cfg["train"].get("pin_memory", use_cuda)),
        persistent_workers=bool(cfg["train"].get("persistent_workers", int(cfg["train"]["num_workers"]) > 0)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)),
    )

    model = _build_model(cfg)
    if _should_attach_language_modules(cfg):
        model = _attach_language_modules(model, cfg)
    model = model.to(device)
    use_ema = bool(cfg["train"].get("use_ema", True))
    ema_decay = float(cfg["train"].get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay) if use_ema else None
    aux_classifier_enabled = _needs_aux_classifier(cfg)
    anchor_labels = bool(cfg["train"].get("anchor_labels", False)) and aux_classifier_enabled
    prototypes = None
    dg_enabled = bool(cfg["train"].get("domain_generalization", False)) and bool(cfg["train"].get("enable_domain_disc", True))
    domain_disc = None
    if dg_enabled:
        domain_disc = DomainDiscriminator(
            in_dim=128,
            num_domains=int(cfg["train"].get("num_domains", 16)),
        ).to(device)
    if anchor_labels:
        anchor_epochs = int(cfg["train"].get("anchor_classifier_epochs", 10))
        anchor_lr = float(cfg["train"].get("anchor_classifier_lr", 1e-3))
        anchor_clf = _train_anchor_classifier(
            train_loader,
            in_channels=len(cfg["data"]["channels"]),
            num_classes=cfg["model"]["num_classes"],
            device=device,
            epochs=anchor_epochs,
            lr=anchor_lr,
        )
        prototypes = _compute_class_prototypes(
            train_loader,
            anchor_clf,
            device=device,
            num_classes=cfg["model"]["num_classes"],
        )
        cls_head = None
        params = list(model.parameters()) + (list(domain_disc.parameters()) if domain_disc is not None else [])
        optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    else:
        anchor_clf = None
        cls_head = AuxClassifier(len(cfg["data"]["channels"]), cfg["model"]["num_classes"]).to(device) if aux_classifier_enabled else None
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + (list(cls_head.parameters()) if cls_head is not None else [])
            + (list(domain_disc.parameters()) if domain_disc is not None else []),
            lr=cfg["train"]["lr"],
        )
    text_prototypes = None
    should_build_text_prototypes = bool(cfg["train"].get("use_text_prototypes", True)) and hasattr(model, "text_encoder")
    if (
        should_build_text_prototypes
        and getattr(dataset, "physio_text", None) is not None
        and getattr(dataset, "y", None) is not None
    ):
        text_prototypes = build_label_text_prototypes(
            model.text_encoder,
            [str(t) for t in dataset.physio_text.tolist()],
            [int(v) for v in dataset.y.tolist()],
            num_classes=int(cfg["model"]["num_classes"]),
            device=device,
        )
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tensorboard"))

    channel_names = [c.lower() for c in cfg["data"]["channels"]]
    eda_idx = channel_names.index("eda") if "eda" in channel_names else -1
    bvp_idx = channel_names.index("bvp") if "bvp" in channel_names else -1
    best_val = float("inf")
    no_improve_epochs = 0
    early_stop = cfg["train"].get("early_stop", False)
    early_stop_patience = cfg["train"].get("early_stop_patience", 5)
    early_stop_min_delta = cfg["train"].get("early_stop_min_delta", 0.0)

    history = {
        "train_loss": [],
        "val_loss": [],
        "l_kin": [],
        "l_freq": [],
        "l_cls": [],
        "l_feat": [],
        "l_embed": [],
        "l_proto": [],
        "l_cycle": [],
        "l_semantic": [],
        "l_artifact_text": [],
    }
    cond_drop_prob = float(cfg["train"].get("cond_drop_prob", 0.1))
    null_label = int(cfg["model"]["num_classes"])
    p2_gamma = float(cfg.get("loss", {}).get("p2_gamma", 0.0))
    p2_k = float(cfg.get("loss", {}).get("p2_k", 1.0))
    w_proto = float(cfg.get("loss", {}).get("w_proto", 0.3))
    w_embed = float(cfg.get("loss", {}).get("w_embed_margin", 0.05))
    w_feat = float(cfg.get("loss", {}).get("w_feat", 0.1))
    proto_temp = float(cfg.get("loss", {}).get("proto_temperature", 0.2))
    w_domain = float(cfg.get("loss", {}).get("w_domain", 0.2))
    w_mech = float(cfg.get("loss", {}).get("w_mech", 0.2))
    w_consistency = float(cfg.get("loss", {}).get("w_consistency", 0.15))
    w_text_proto = float(cfg.get("loss", {}).get("w_text_proto", 0.15))
    w_cycle = float(cfg.get("loss", {}).get("w_cycle", 0.15))
    w_artifact_text = float(cfg.get("loss", {}).get("w_artifact_text", 0.1))
    w_semantic_align = float(cfg.get("loss", {}).get("w_semantic_align", 0.15))
    artifact_aware = bool(cfg["train"].get("artifact_aware", False))
    use_language_conditioning = bool(cfg["train"].get("use_language_conditioning", True))
    use_text_prototypes = bool(cfg["train"].get("use_text_prototypes", True))
    use_signal_text_cycle = bool(cfg["train"].get("use_signal_text_cycle", True))
    use_semantic_alignment = bool(cfg["train"].get("use_semantic_alignment", True))
    use_artifact_text_conditioning = bool(cfg["train"].get("use_artifact_text_conditioning", artifact_aware))
    corruption_kinds = cfg["train"].get(
        "artifact_corruptions",
        ["motion", "burst", "spike_dropout", "baseline_wander", "time_jitter"],
    )

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        use_amp = bool(cfg["train"].get("use_amp", use_cuda))
        amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "fp16")).lower() == "bf16" else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and use_cuda)
        w_kin = _schedule_weight(
            epoch,
            cfg["loss"]["w_kin_start"],
            cfg["loss"]["w_kin_end"],
            cfg["loss"]["warmup_epochs"],
            cfg["train"]["epochs"],
        )
        w_freq = _schedule_weight(
            epoch,
            cfg["loss"]["w_freq_start"],
            cfg["loss"]["w_freq_end"],
            cfg["loss"]["warmup_epochs"],
            cfg["train"]["epochs"],
        )
        if "w_cls_start" in cfg.get("loss", {}) and "w_cls_end" in cfg.get("loss", {}):
            w_cls = _schedule_weight(
                epoch,
                float(cfg["loss"]["w_cls_start"]),
                float(cfg["loss"]["w_cls_end"]),
                int(cfg["loss"].get("warmup_epochs", 0)),
                int(cfg["train"]["epochs"]),
            )
        else:
            w_cls = float(cfg["loss"].get("w_cls", 0.0))
        epoch_losses = []
        epoch_kin = []
        epoch_freq = []
        epoch_proto = []
        epoch_cls = []
        epoch_feat = []
        epoch_embed = []
        epoch_cycle = []
        epoch_semantic = []
        epoch_artifact = []
        for step, batch in enumerate(train_loader, start=1):
            x0, y, sample_idx = _resolve_batch_indices(batch)
            x0 = x0.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            sample_idx = sample_idx.to(device, non_blocking=use_cuda) if sample_idx is not None else None
            subject_ids, domain_ids = _subject_and_domain_ids(dataset, sample_idx, device=device)
            selected_corruption = "clean"
            artifact_severity = "mild"
            if artifact_aware:
                sev = severity_schedule(
                    epoch,
                    total_epochs=int(cfg["train"]["epochs"]),
                    start=float(cfg["train"].get("artifact_severity_start", 0.1)),
                    end=float(cfg["train"].get("artifact_severity_end", 1.0)),
                )
                selected_corruption = str(corruption_kinds[(epoch + step - 2) % len(corruption_kinds)])
                artifact_severity = _severity_word(float(sev))
                x0_aug = apply_corruptions(x0, severity=sev, kinds=[selected_corruption])
            else:
                x0_aug = x0
            if hasattr(model, "text_encoder"):
                physio_text_embed, semantic_text_embed, artifact_text_embed, physio_text_tokens, physio_text_mask = _lookup_text_conditioning(
                    dataset,
                    sample_idx,
                    model,
                    device,
                    fallback_labels=y,
                    artifact_kind=_corruption_key(selected_corruption),
                    artifact_severity=artifact_severity,
                )
            else:
                text_dim = int(cfg["model"]["embedding_dim"])
                physio_text_embed = torch.zeros((y.shape[0], text_dim), device=device)
                semantic_text_embed = torch.zeros_like(physio_text_embed)
                artifact_text_embed = torch.zeros_like(physio_text_embed)
                physio_text_tokens = None
                physio_text_mask = None
            semantic_term = semantic_text_embed if use_semantic_alignment else torch.zeros_like(semantic_text_embed)
            artifact_term = artifact_text_embed if use_artifact_text_conditioning else torch.zeros_like(artifact_text_embed)
            fused_text_embed = F.normalize(physio_text_embed + semantic_term + artifact_term, dim=-1)
            if not use_language_conditioning:
                fused_text_embed = torch.zeros_like(fused_text_embed)
            # Classifier-free guidance training trick:
            # randomly replace some labels with the reserved null label id (= num_classes).
            if cond_drop_prob > 0:
                drop = torch.rand((y.shape[0],), device=device) < cond_drop_prob
                y_cond = y.clone()
                y_cond[drop] = null_label
            else:
                drop = None
                y_cond = y
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                t = torch.randint(0, model.timesteps, (x0.size(0),), device=device)
                noise = torch.randn_like(x0)
                x_t = model.q_sample(x0_aug, t, noise)
                eps_pred = model.predict_eps(
                    x_t,
                    t,
                    y_cond,
                    domain_ids=domain_ids if dg_enabled else None,
                    subject_ids=subject_ids if dg_enabled else None,
                    text_embedding=fused_text_embed,
                    text_tokens=physio_text_tokens if use_language_conditioning else None,
                    text_mask=physio_text_mask if use_language_conditioning else None,
                )
                x0_hat = model.predict_x0(x_t, t, eps_pred)
                x0_hat_clamped = _apply_x0_constraint(x0_hat, cfg)
                mse = (noise - eps_pred) ** 2
                mse_per = mse.mean(dim=(1, 2))
                if p2_gamma > 0:
                    alpha_bar = model.alphas_cumprod[t]
                    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
                    w = (p2_k + snr) ** (-p2_gamma)
                    l_simple = (mse_per * w).mean()
                else:
                    l_simple = mse_per.mean()
                l_kin = torch.tensor(0.0, device=device)
                if eda_idx >= 0:
                    l_kin = loss_kin(x0, x0_hat_clamped, eda_index=eda_idx)
                l_freq = torch.tensor(0.0, device=device)
                if bvp_idx >= 0:
                    l_freq = loss_freq(x0, x0_hat_clamped, bvp_index=bvp_idx)
                classifier = anchor_clf if anchor_clf is not None else cls_head
                if classifier is not None:
                    logits = classifier(x0_hat_clamped)
                    if drop is None:
                        l_cls = torch.nn.functional.cross_entropy(logits, y)
                        keep = None
                    else:
                        keep = ~drop
                        if bool(keep.any()):
                            l_cls = torch.nn.functional.cross_entropy(logits[keep], y[keep])
                        else:
                            l_cls = torch.tensor(0.0, device=device)
                    feat_anchor = torch.tensor(0.0, device=device)
                    if eda_idx >= 0:
                        feat_anchor = feature_anchor_loss(x0, x0_hat_clamped, channel_index=eda_idx)
                    features = classifier.extract_features(x0_hat_clamped)
                else:
                    l_cls = torch.tensor(0.0, device=device)
                    keep = None
                    feat_anchor = torch.tensor(0.0, device=device)
                    features = x0_hat_clamped.mean(dim=-1)
            if keep is not None and bool(keep.any()):
                proto_features = features[keep]
                proto_labels = y[keep]
            elif keep is not None:
                proto_features = features[:0]
                proto_labels = y[:0]
            else:
                proto_features = features
                proto_labels = y
            l_proto = _prototype_nce_loss(
                proto_features,
                proto_labels,
                prototypes=prototypes,
                temperature=proto_temp,
            )
            l_embed = _label_embedding_margin(model, margin=float(cfg.get("loss", {}).get("embed_margin", 1.0)))
            l_mech = torch.tensor(0.0, device=device)
            if hasattr(model, "mechanistic_loss") and eda_idx >= 0 and bvp_idx >= 0:
                l_mech = model.mechanistic_loss(x0, x0_hat_clamped, eda_index=eda_idx, bvp_index=bvp_idx)
            l_domain = torch.tensor(0.0, device=device)
            if dg_enabled and domain_disc is not None:
                # Domain discriminator runs outside autocast; keep activations in fp32.
                rev_feat = _grad_reverse(features.float(), lambd=float(cfg["train"].get("dg_grl_lambda", 1.0)))
                dom_logits = domain_disc(rev_feat)
                l_domain = torch.nn.functional.cross_entropy(dom_logits, domain_ids)
            if hasattr(model, "signal_text_cycle"):
                pred_text_embed, pred_artifact_embed = model.signal_text_cycle(x0_hat_clamped)
            else:
                pred_text_embed = torch.zeros_like(physio_text_embed)
                pred_artifact_embed = torch.zeros_like(artifact_text_embed)
            l_text_proto = prototype_alignment_loss(physio_text_embed, text_prototypes, y) if use_text_prototypes else torch.tensor(0.0, device=device)
            l_cycle = cycle_reconstruction_loss(pred_text_embed, physio_text_embed) if use_signal_text_cycle else torch.tensor(0.0, device=device)
            l_art_text = artifact_text_loss(pred_artifact_embed, artifact_text_embed) if use_artifact_text_conditioning else torch.tensor(0.0, device=device)
            l_semantic = semantic_consistency_loss(physio_text_embed, semantic_text_embed) if use_semantic_alignment else torch.tensor(0.0, device=device)
            l_consistency = torch.tensor(0.0, device=device)
            if artifact_aware:
                x_t_alt = model.q_sample(x0, t, noise)
                eps_alt = model.predict_eps(
                    x_t_alt,
                    t,
                    y_cond,
                    domain_ids=domain_ids if dg_enabled else None,
                    subject_ids=subject_ids if dg_enabled else None,
                    text_embedding=fused_text_embed,
                    text_tokens=physio_text_tokens if use_language_conditioning else None,
                    text_mask=physio_text_mask if use_language_conditioning else None,
                )
                l_consistency = torch.mean((eps_pred - eps_alt) ** 2)
            loss = (
                cfg["loss"]["w_simple"] * l_simple
                + w_kin * l_kin
                + w_freq * l_freq
                + w_cls * l_cls
                + w_feat * feat_anchor
                + w_proto * l_proto
                + w_embed * l_embed
                + w_domain * l_domain
                + w_mech * l_mech
                + w_consistency * l_consistency
                + w_text_proto * l_text_proto
                + w_cycle * l_cycle
                + w_artifact_text * l_art_text
                + w_semantic_align * l_semantic
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            if step % cfg["train"]["log_every"] == 0:
                writer.add_scalar("train/loss", loss.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_kin", l_kin.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_freq", l_freq.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_cls", l_cls.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_feat", feat_anchor.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_embed", l_embed.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_proto", l_proto.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_domain", l_domain.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_mech", l_mech.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_consistency", l_consistency.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_text_proto", l_text_proto.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_cycle", l_cycle.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_artifact_text", l_art_text.item(), epoch * 1000 + step)
                writer.add_scalar("train/l_semantic", l_semantic.item(), epoch * 1000 + step)
                print(
                    f"[Epoch {epoch} Step {step}] "
                    f"loss={loss.item():.4f} l_kin={l_kin.item():.4f} "
                    f"l_freq={l_freq.item():.4f} l_cls={l_cls.item():.4f} "
                    f"l_feat={feat_anchor.item():.4f} l_embed={l_embed.item():.4f} "
                    f"l_proto={l_proto.item():.4f} l_domain={l_domain.item():.4f} "
                    f"l_mech={l_mech.item():.4f} l_cycle={l_cycle.item():.4f} "
                    f"l_sem={l_semantic.item():.4f}"
                )
            epoch_losses.append(loss.item())
            epoch_kin.append(l_kin.item())
            epoch_freq.append(l_freq.item())
            epoch_proto.append(l_proto.item())
            epoch_cls.append(l_cls.item())
            epoch_feat.append(feat_anchor.item())
            epoch_embed.append(l_embed.item())
            epoch_cycle.append(l_cycle.item())
            epoch_semantic.append(l_semantic.item())
            epoch_artifact.append(l_art_text.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x0, y, sample_idx = _resolve_batch_indices(batch)
                x0 = x0.to(device, non_blocking=use_cuda)
                y = y.to(device, non_blocking=use_cuda)
                sample_idx = sample_idx.to(device, non_blocking=use_cuda) if sample_idx is not None else None
                subject_ids, domain_ids = _subject_and_domain_ids(dataset, sample_idx, device=device)
                if hasattr(model, "text_encoder"):
                    physio_text_embed, semantic_text_embed, artifact_text_embed, physio_text_tokens, physio_text_mask = _lookup_text_conditioning(
                        dataset,
                        sample_idx,
                        model,
                        device,
                        fallback_labels=y,
                        artifact_kind="clean",
                        artifact_severity="mild",
                    )
                else:
                    text_dim = int(cfg["model"]["embedding_dim"])
                    physio_text_embed = torch.zeros((y.shape[0], text_dim), device=device)
                    semantic_text_embed = torch.zeros_like(physio_text_embed)
                    artifact_text_embed = torch.zeros_like(physio_text_embed)
                    physio_text_tokens = None
                    physio_text_mask = None
                semantic_term = semantic_text_embed if use_semantic_alignment else torch.zeros_like(semantic_text_embed)
                artifact_term = artifact_text_embed if use_artifact_text_conditioning else torch.zeros_like(artifact_text_embed)
                fused_text_embed = F.normalize(physio_text_embed + semantic_term + artifact_term, dim=-1)
                if not use_language_conditioning:
                    fused_text_embed = torch.zeros_like(fused_text_embed)
                if cond_drop_prob > 0:
                    drop = torch.rand((y.shape[0],), device=device) < cond_drop_prob
                    y_cond = y.clone()
                    y_cond[drop] = null_label
                else:
                    drop = None
                    y_cond = y
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    t = torch.randint(0, model.timesteps, (x0.size(0),), device=device)
                    noise = torch.randn_like(x0)
                    x_t = model.q_sample(x0, t, noise)
                    eps_pred = model.predict_eps(
                        x_t,
                        t,
                        y_cond,
                        domain_ids=domain_ids if dg_enabled else None,
                        subject_ids=subject_ids if dg_enabled else None,
                        text_embedding=fused_text_embed,
                        text_tokens=physio_text_tokens if use_language_conditioning else None,
                        text_mask=physio_text_mask if use_language_conditioning else None,
                    )
                    x0_hat = model.predict_x0(x_t, t, eps_pred)
                    x0_hat_clamped = _apply_x0_constraint(x0_hat, cfg)
                    mse = (noise - eps_pred) ** 2
                    mse_per = mse.mean(dim=(1, 2))
                    if p2_gamma > 0:
                        alpha_bar = model.alphas_cumprod[t]
                        snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
                        w = (p2_k + snr) ** (-p2_gamma)
                        l_simple = (mse_per * w).mean()
                    else:
                        l_simple = mse_per.mean()
                    l_kin = torch.tensor(0.0, device=device)
                    if eda_idx >= 0:
                        l_kin = loss_kin(x0, x0_hat_clamped, eda_index=eda_idx)
                    l_freq = torch.tensor(0.0, device=device)
                    if bvp_idx >= 0:
                        l_freq = loss_freq(x0, x0_hat_clamped, bvp_index=bvp_idx)
                    classifier = anchor_clf if anchor_clf is not None else cls_head
                    if classifier is not None:
                        logits = classifier(x0_hat_clamped)
                if classifier is not None:
                    if drop is None:
                        l_cls = torch.nn.functional.cross_entropy(logits, y)
                        keep = None
                    else:
                        keep = ~drop
                        if bool(keep.any()):
                            l_cls = torch.nn.functional.cross_entropy(logits[keep], y[keep])
                        else:
                            l_cls = torch.tensor(0.0, device=device)
                    feat_anchor = torch.tensor(0.0, device=device)
                    if eda_idx >= 0:
                        feat_anchor = feature_anchor_loss(x0, x0_hat_clamped, channel_index=eda_idx)
                    features = classifier.extract_features(x0_hat_clamped)
                else:
                    l_cls = torch.tensor(0.0, device=device)
                    keep = None
                    feat_anchor = torch.tensor(0.0, device=device)
                    features = x0_hat_clamped.mean(dim=-1)
                if keep is not None and bool(keep.any()):
                    proto_features = features[keep]
                    proto_labels = y[keep]
                elif keep is not None:
                    proto_features = features[:0]
                    proto_labels = y[:0]
                else:
                    proto_features = features
                    proto_labels = y
                l_proto = _prototype_nce_loss(
                    proto_features,
                    proto_labels,
                    prototypes=prototypes,
                    temperature=proto_temp,
                )
                l_embed = _label_embedding_margin(model, margin=float(cfg.get("loss", {}).get("embed_margin", 1.0)))
                l_mech = torch.tensor(0.0, device=device)
                if hasattr(model, "mechanistic_loss") and eda_idx >= 0 and bvp_idx >= 0:
                    l_mech = model.mechanistic_loss(x0, x0_hat_clamped, eda_index=eda_idx, bvp_index=bvp_idx)
                l_domain = torch.tensor(0.0, device=device)
                if dg_enabled and domain_disc is not None:
                    rev_feat = _grad_reverse(features, lambd=float(cfg["train"].get("dg_grl_lambda", 1.0)))
                    dom_logits = domain_disc(rev_feat)
                    l_domain = torch.nn.functional.cross_entropy(dom_logits, domain_ids)
                if hasattr(model, "signal_text_cycle"):
                    pred_text_embed, pred_artifact_embed = model.signal_text_cycle(x0_hat_clamped)
                else:
                    pred_text_embed = torch.zeros_like(physio_text_embed)
                    pred_artifact_embed = torch.zeros_like(artifact_text_embed)
                l_text_proto = prototype_alignment_loss(physio_text_embed, text_prototypes, y) if use_text_prototypes else torch.tensor(0.0, device=device)
                l_cycle = cycle_reconstruction_loss(pred_text_embed, physio_text_embed) if use_signal_text_cycle else torch.tensor(0.0, device=device)
                l_art_text = artifact_text_loss(pred_artifact_embed, artifact_text_embed) if use_artifact_text_conditioning else torch.tensor(0.0, device=device)
                l_semantic = semantic_consistency_loss(physio_text_embed, semantic_text_embed) if use_semantic_alignment else torch.tensor(0.0, device=device)
                val_loss = (
                    cfg["loss"]["w_simple"] * l_simple
                    + w_kin * l_kin
                    + w_freq * l_freq
                    + w_cls * l_cls
                    + w_feat * feat_anchor
                    + w_proto * l_proto
                    + w_embed * l_embed
                    + w_domain * l_domain
                    + w_mech * l_mech
                    + w_text_proto * l_text_proto
                    + w_cycle * l_cycle
                    + w_artifact_text * l_art_text
                    + w_semantic_align * l_semantic
                )
                val_losses.append(val_loss.item())
        val_mean = float(np.mean(val_losses)) if val_losses else 0.0
        writer.add_scalar("val/loss", val_mean, epoch)
        history["train_loss"].append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
        history["val_loss"].append(val_mean)
        history["l_kin"].append(float(np.mean(epoch_kin)) if epoch_kin else 0.0)
        history["l_freq"].append(float(np.mean(epoch_freq)) if epoch_freq else 0.0)
        history["l_proto"].append(float(np.mean(epoch_proto)) if epoch_proto else 0.0)
        history["l_cls"].append(float(np.mean(epoch_cls)) if epoch_cls else 0.0)
        history["l_feat"].append(float(np.mean(epoch_feat)) if epoch_feat else 0.0)
        history["l_embed"].append(float(np.mean(epoch_embed)) if epoch_embed else 0.0)
        history["l_cycle"].append(float(np.mean(epoch_cycle)) if epoch_cycle else 0.0)
        history["l_semantic"].append(float(np.mean(epoch_semantic)) if epoch_semantic else 0.0)
        history["l_artifact_text"].append(float(np.mean(epoch_artifact)) if epoch_artifact else 0.0)
        if val_mean < best_val - early_stop_min_delta:
            best_val = val_mean
            best_ckpt = os.path.join(out_dir, "physio_diff_best.pt")
            if ema is not None:
                torch.save(ema.state_dict(), best_ckpt)
            else:
                torch.save(model.state_dict(), best_ckpt)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        scheduler.step(val_mean)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/lr", current_lr, epoch)
        if epoch % cfg["train"]["save_every"] == 0:
            ckpt = os.path.join(out_dir, f"physio_diff_epoch_{epoch}.pt")
            if ema is not None:
                torch.save(ema.state_dict(), ckpt)
            else:
                torch.save(model.state_dict(), ckpt)
            _save_curves(history, out_dir)
        if early_stop and no_improve_epochs >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} (no improvement for {no_improve_epochs} epochs)."
            )
            break
    return model, cache_path


def _save_curves(history: Dict[str, list], out_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if history["train_loss"] or history["val_loss"]:
        axes[0].set_yscale("log")

    axes[1].plot(history["l_kin"], label="l_kin")
    axes[1].plot(history["l_freq"], label="l_freq")
    axes[1].plot(history.get("l_proto", []), label="l_proto")
    axes[1].plot(history.get("l_cycle", []), label="l_cycle")
    axes[1].plot(history.get("l_semantic", []), label="l_semantic")
    axes[1].plot(history.get("l_artifact_text", []), label="l_artifact_text")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Aux Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def load_config(path: str) -> Dict:
    cfg_path = resolve_project_path(path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
