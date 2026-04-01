import os

import matplotlib
import numpy as np
import torch
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

from src.data.datasets import WESADDataset, load_meta, make_dataloaders
from src.data.wesad import build_dataset_cache, build_subject_windows
from experiments.common.metrics import (
    cross_dataset_semantic_stability,
    postprocess_synth,
    text_prototype_separability,
    robustness_eval,
    recovered_text_consistency,
    time_freq_metrics,
    tstr_eval,
)
from src.models.mech_latent_diff import MechanisticPhysioDiffusion
from src.models.physio_diff import PhysioDiffusion
from src.train.train_diffusion import _attach_language_modules, _should_attach_language_modules, train as train_diffusion
from src.utils.project_paths import resolve_project_path


def load_config(path: str):
    cfg_path = resolve_project_path(path)
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if "data" in cfg:
        for key in ["root_dir", "cache_dir", "processed_label_path"]:
            value = cfg["data"].get(key)
            if value:
                cfg["data"][key] = resolve_project_path(value)
    if "project" in cfg and cfg["project"].get("output_dir"):
        cfg["project"]["output_dir"] = resolve_project_path(cfg["project"]["output_dir"])
    return cfg

def _label_counts(loader):
    from torch.utils.data import Subset
    ds = loader.dataset
    labels = []
    if isinstance(ds, Subset):
        base = ds.dataset
        indices = ds.indices
        if hasattr(base, "y") and base.y is not None:
            labels = [int(base.y[i]) for i in indices]
    else:
        if hasattr(ds, "y") and ds.y is not None:
            labels = [int(v) for v in ds.y]
    if not labels:
        return None
    counts = {0: 0, 1: 0}
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _load_state_dict_compat(model: torch.nn.Module, state: dict) -> None:
    """
    Backward-compatible loader for checkpoints saved before CFG support.

    Older checkpoints had `denoiser.label_embed.weight` shaped [num_classes, D].
    Newer models reserve an extra null-label row: [num_classes+1, D].
    """
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict (dict), got: {type(state)}")
    key = "denoiser.label_embed.weight"
    if key in state and key in model.state_dict():
        w = state[key]
        want = model.state_dict()[key].shape
        if isinstance(w, torch.Tensor):
            have = w.shape
            if len(have) == 2 and len(want) == 2 and have[0] + 1 == want[0] and have[1] == want[1]:
                w_new = torch.zeros(want, dtype=w.dtype)
                w_new[: have[0]] = w
                state[key] = w_new
    model.load_state_dict(state, strict=False)


def _build_model(cfg: dict) -> torch.nn.Module:
    kwargs = dict(
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
    if str(cfg.get("model", {}).get("model_type", "standard")).lower() in {"mechanistic", "mech", "mech_latent"}:
        model = MechanisticPhysioDiffusion(**kwargs)
    else:
        model = PhysioDiffusion(**kwargs)
    if _should_attach_language_modules(cfg):
        model = _attach_language_modules(model, cfg)
    return model


def _class_condition_texts(dataset: WESADDataset, class_idx: int) -> tuple[str, str, str]:
    default = (
        "physiology profile: eda tonic moderate, eda phasic density moderate, eda peak sharpness balanced, eda recovery moderate; bvp rhythm moderately variable, bvp amplitude moderate, bvp morphology balanced.",
        "shared physiology state with lower sympathetic arousal and steadier autonomic balance."
        if int(class_idx) == 0
        else "shared physiology state with elevated sympathetic arousal and stronger stress reactivity.",
        "mild clean wearable signal",
    )
    if getattr(dataset, "y", None) is None or getattr(dataset, "physio_text", None) is None:
        return default
    idx = np.where(np.asarray(dataset.y) == int(class_idx))[0]
    if idx.size == 0:
        return default
    pos = int(idx[0])
    phys = str(dataset.physio_text[pos]) if dataset.physio_text is not None else default[0]
    sem = str(dataset.semantic_text[pos]) if dataset.semantic_text is not None else default[1]
    art = str(dataset.artifact_text[pos]) if dataset.artifact_text is not None else default[2]
    return phys, sem, art


def _subset_arrays(loader):
    """Return (x, y, acc) numpy arrays for a DataLoader's dataset/subset."""
    ds = loader.dataset
    # Subset from torch.utils.data.Subset
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        base = ds.dataset
        idx = np.asarray(ds.indices)
        x = np.asarray(base.x)[idx]
        y = np.asarray(base.y)[idx] if getattr(base, "y", None) is not None else None
        acc = None
        if getattr(base, "acc", None) is not None:
            acc_arr = np.asarray(base.acc)
            # include_acc=False can yield a 0-d object array; treat as absent ACC.
            if acc_arr.ndim > 0:
                acc = acc_arr[idx]
        return x, y, acc
    # Direct dataset
    x = np.asarray(getattr(ds, "x", None))
    y = np.asarray(getattr(ds, "y", None)) if getattr(ds, "y", None) is not None else None
    acc = None
    if getattr(ds, "acc", None) is not None:
        acc_arr = np.asarray(getattr(ds, "acc", None))
        if acc_arr.ndim > 0:
            acc = acc_arr
    return x, y, acc


def _split_summary(split: dict) -> dict:
    y = split.get("y")
    if y is None:
        return {"num_windows": int(len(split.get("x", [])))}
    unique, counts = np.unique(y, return_counts=True)
    class_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    return {
        "num_windows": int(len(split.get("x", []))),
        "class_counts": class_counts,
    }


def run(cfg):
    os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
    device = torch.device(cfg["project"]["device"])
    _set_seed(int(cfg["project"].get("seed", 42)))

    use_existing = cfg["eval"].get("use_existing_model", False)
    if use_existing:
        print("Building dataset cache...")
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
        )
        model = _build_model(cfg).to(device)
        ckpt_path = cfg["eval"].get("checkpoint_path") or os.path.join(
            cfg["project"]["output_dir"], "physio_diff_best.pt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Loading model from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _load_state_dict_compat(model, state)
    else:
        print("Training diffusion model...")
        model, cache_path = train_diffusion(cfg)
        best_ckpt = os.path.join(cfg["project"]["output_dir"], "physio_diff_best.pt")
        if os.path.exists(best_ckpt):
            print(f"Loading best model from {best_ckpt}...")
            state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            _load_state_dict_compat(model, state)
        else:
            print("Warning: best model checkpoint not found, using last epoch weights.")

    dataset = WESADDataset(cache_path)
    _check_channel_alignment(cache_path, cfg["data"]["channels"])
    train_loader, val_loader, test_loader = make_dataloaders(
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
        cv_n_splits=int(cfg["data"].get("cv_n_splits", 5)),
        cv_fold_index=int(cfg["data"].get("cv_fold_index", 0)),
        val_subject_count=int(cfg["data"].get("val_subject_count", 1)),
        pin_memory=bool(cfg["train"].get("pin_memory", torch.cuda.is_available())),
        persistent_workers=bool(cfg["train"].get("persistent_workers", int(cfg["train"]["num_workers"]) > 0)),
        prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)),
    )
    train_x, train_y, train_acc = _subset_arrays(train_loader)
    val_x, val_y, val_acc = _subset_arrays(val_loader)
    test_x, test_y, test_acc = _subset_arrays(test_loader)
    splits = {
        "train": {"x": train_x, "y": train_y, "acc": train_acc},
        "val": {"x": val_x, "y": val_y, "acc": val_acc},
        "test": {"x": test_x, "y": test_y, "acc": test_acc},
    }

    if cfg["data"].get("split_strategy", "random") != "subject":
        print("Warning: split_strategy is not 'subject'; risk of data leakage.")

    model.eval()
    synth_samples = []
    synth_labels = []
    samples_per_class = cfg["eval"].get("synth_samples_per_class", 1000)
    batch_size_gen = cfg["eval"].get("gen_batch_size", 50)
    cfg_scale = float(cfg["eval"].get("cfg_scale", 1.0))
    cfg_rescale = bool(cfg["eval"].get("cfg_rescale", True))
    x0_clip_q = cfg["eval"].get("x0_clip_quantile", 0.995)
    if x0_clip_q is not None:
        x0_clip_q = float(x0_clip_q)
    sample_steps = cfg["eval"].get("sample_steps", None)
    if sample_steps is not None:
        sample_steps = int(sample_steps)
    sample_eta = cfg["eval"].get("sample_eta", 0.0)
    if sample_eta is not None:
        sample_eta = float(sample_eta)
    x_t_clip = cfg["eval"].get("x_t_clip", 20.0)
    if x_t_clip is not None:
        x_t_clip = float(x_t_clip)
    print(f"Generating synthetic data: {samples_per_class} per class.")
    with torch.no_grad():
        for class_idx in range(cfg["model"]["num_classes"]):
            class_samples = []
            num_batches = (samples_per_class + batch_size_gen - 1) // batch_size_gen
            if hasattr(model, "text_encoder"):
                phys_text, sem_text, art_text = _class_condition_texts(dataset, class_idx)
                # Precompute class text embeddings once to avoid repeated encoder calls per batch.
                full_text_embedding = model.text_encoder([phys_text] * samples_per_class, device=device)
                full_text_embedding = torch.nn.functional.normalize(
                    full_text_embedding
                    + model.text_encoder([sem_text] * samples_per_class, device=device)
                    + model.text_encoder([art_text] * samples_per_class, device=device),
                    dim=-1,
                )
                _, full_text_tokens, full_text_mask = model.text_encoder.encode_tokens([phys_text] * samples_per_class, device=device)
            else:
                full_text_embedding = None
                full_text_tokens = None
                full_text_mask = None
            for i in range(num_batches):
                current_batch = min(batch_size_gen, samples_per_class - i * batch_size_gen)
                y = torch.full((current_batch,), class_idx, device=device, dtype=torch.long)
                start = i * batch_size_gen
                end = start + current_batch
                text_embedding = None if full_text_embedding is None else full_text_embedding[start:end]
                text_tokens = None if full_text_tokens is None else full_text_tokens[start:end]
                text_mask = None if full_text_mask is None else full_text_mask[start:end]
                shape = (
                    current_batch,
                    len(cfg["data"]["channels"]),
                    cfg["data"]["window_length"],
                )
                x_gen = model.sample(
                    shape,
                    y,
                    device=device,
                    cfg_scale=cfg_scale,
                    cfg_rescale=cfg_rescale,
                    x0_clip_quantile=x0_clip_q,
                    x_t_clip=x_t_clip,
                    sample_steps=sample_steps,
                    ddim_eta=sample_eta,
                    text_embedding=text_embedding,
                    text_tokens=text_tokens,
                    text_mask=text_mask,
                )
                class_samples.append(x_gen.cpu().numpy())
                print(f"  Class {class_idx}: Batch {i + 1}/{num_batches} done.")
            synth_samples.append(np.concatenate(class_samples, axis=0))
            synth_labels.append(np.full((samples_per_class,), class_idx))
    synth_x = np.concatenate(synth_samples, axis=0)
    synth_y = np.concatenate(synth_labels, axis=0)
    # Fair post-processing (must be consistent across methods)
    match_stats = bool(cfg["eval"].get("match_stats", False))
    clip_min = float(cfg["eval"].get("clip_min", -5.0))
    clip_max = float(cfg["eval"].get("clip_max", 5.0))
    synth_x = postprocess_synth(synth_x, splits["train"]["x"], match_stats=match_stats, clip_min=clip_min, clip_max=clip_max)

    tstr_cache = os.path.join(cfg["project"]["output_dir"], "synthetic_normalized.npz")
    np.savez_compressed(tstr_cache, x=synth_x, y=synth_y)
    _save_visual_check(tstr_cache, cfg["project"]["output_dir"], cfg["data"]["channels"])
    _save_tstr_visual(tstr_cache, test_loader, cfg["data"]["channels"], cfg["project"]["output_dir"])
    _save_class_comparison(tstr_cache, cfg["data"]["channels"], cfg["project"]["output_dir"])
    _save_bvp_spectrum(tstr_cache, cfg["data"]["channels"], cfg["project"]["output_dir"])

    if cfg["eval"].get("unnormalize", True) and dataset.global_mean is not None:
        mean = np.asarray(dataset.global_mean).reshape(1, -1, 1)
        std = np.asarray(dataset.global_std).reshape(1, -1, 1)
        synth_x_unnorm = synth_x * std + mean
        np.savez_compressed(
            os.path.join(cfg["project"]["output_dir"], "synthetic_original_units.npz"),
            x=synth_x_unnorm,
            y=synth_y,
        )

    meta_path = os.path.join(cfg["data"]["cache_dir"], "dataset_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(cfg["data"]["cache_dir"], "wesad_meta.json")
    synth_dataset = WESADDataset(tstr_cache, meta_path=meta_path if os.path.exists(meta_path) else None)
    counts = _label_counts(test_loader)
    if counts is not None:
        print(f"Test Set Class 0 count: {counts.get(0, 0)}")
        print(f"Test Set Class 1 count: {counts.get(1, 0)}")
    # Clamp test data to the same range used for synth post-processing
    test_x_clamped = np.clip(splits["test"]["x"].astype(np.float32), clip_min, clip_max)
    tstr = tstr_eval(
        synth_x,
        synth_y,
        test_x_clamped,
        splits["test"]["y"],
        cfg,
        device=device,
        val_x=np.clip(splits["val"]["x"].astype(np.float32), clip_min, clip_max),
        val_y=splits["val"]["y"],
    )
    acc, f1 = float(tstr["accuracy"]), float(tstr["f1"])

    tfm = time_freq_metrics(test_x_clamped, synth_x[: len(test_x_clamped)], cfg["data"]["target_fs"])
    language_metrics = {}
    if getattr(dataset, "physio_text", None) is not None and getattr(dataset, "y", None) is not None:
        language_metrics["text_prototype_separability"] = text_prototype_separability(
            [str(t) for t in dataset.physio_text.tolist()],
            np.asarray(dataset.y),
        )
    if getattr(dataset, "physio_text", None) is not None and getattr(dataset, "semantic_text", None) is not None:
        language_metrics["semantic_stability"] = cross_dataset_semantic_stability(
            [str(t) for t in dataset.physio_text.tolist()],
            [str(t) for t in dataset.semantic_text.tolist()],
        )
    if hasattr(model, "signal_text_cycle") and hasattr(model, "text_encoder"):
        with torch.no_grad():
            n_eval = min(128, synth_x.shape[0])
            synth_tensor = torch.from_numpy(synth_x[:n_eval]).float().to(device)
            pred_text_embed, _ = model.signal_text_cycle(synth_tensor)
            target_texts = []
            for label in synth_y[:n_eval]:
                phys_text, _, _ = _class_condition_texts(dataset, int(label))
                target_texts.append(phys_text)
            target_embed = model.text_encoder(target_texts, device=device)
            language_metrics["recovered_text_consistency"] = float(
                torch.nn.functional.cosine_similarity(pred_text_embed, target_embed, dim=-1).mean().item()
            )
    print(f"TSTR Result -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    meta = {
        "dataset_name": cfg["data"].get("dataset_name", "wesad"),
        "seed": int(cfg["project"].get("seed", -1)),
        "cfg_scale": float(cfg["eval"].get("cfg_scale", 1.0)),
        "cond_drop_prob": float(cfg["train"].get("cond_drop_prob", 0.0)),
        "timesteps": int(cfg["diffusion"].get("timesteps", -1)),
        "sample_steps": None if sample_steps is None else int(sample_steps),
        "sample_eta": None if sample_eta is None else float(sample_eta),
        "split_strategy": cfg["data"].get("split_strategy", "random"),
        "splits": {name: _split_summary(split) for name, split in splits.items()},
        "normalization_summary": load_meta(cfg["data"]["cache_dir"]).get("normalization_summary", {}),
        "label_alignment": tstr.get("label_alignment"),
    }
    _save_physio_results(
        cfg["project"]["output_dir"],
        acc,
        f1,
        time_freq=tfm,
        meta=meta,
        language_metrics=language_metrics,
    )

    # Robustness evaluation (fair, comparable across methods)
    print("Running Robustness Evaluation (fair protocol)...")
    robust = robustness_eval(
        synth_x,
        synth_y,
        test_x_clamped,
        splits["test"]["y"],
        splits["test"].get("acc"),
        cfg,
        device=device,
    )
    _save_physio_results(
        cfg["project"]["output_dir"],
        acc,
        f1,
        time_freq=tfm,
        meta=meta,
        robust=robust,
        language_metrics=language_metrics,
    )


def _standardize_signal(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    if std < 1e-6:
        return x - mean
    return (x - mean) / std


def _save_visual_check(cache_path: str, out_dir: str, channels):
    data = np.load(cache_path)
    x = data["x"]
    y = data["y"]
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    for i, ch in enumerate(channels):
        axes[i].plot(_standardize_signal(x[0, i, :]), label=f"Class {int(y[0])}")
        axes[i].set_title(str(ch).upper())
        axes[i].legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "check_generated.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def _save_tstr_visual(cache_path: str, test_loader, channels, out_dir: str) -> None:
    data = np.load(cache_path)
    synth_x = data["x"]
    synth_y = data["y"]
    real_x = None
    real_y = None
    ds = test_loader.dataset
    if hasattr(ds, "indices"):
        base = ds.dataset
        idx = ds.indices[0]
        if hasattr(base, "x") and hasattr(base, "y"):
            real_x = base.x[idx]
            real_y = int(base.y[idx])
    else:
        if hasattr(ds, "x") and hasattr(ds, "y"):
            real_x = ds.x[0]
            real_y = int(ds.y[0])
    if real_x is None:
        return
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    for i, ch in enumerate(channels):
        axes[i].plot(
            _standardize_signal(synth_x[0, i, :]),
            label=f"Synth Class {int(synth_y[0])}",
            alpha=0.8,
        )
        axes[i].plot(
            _standardize_signal(real_x[i, :]),
            label=f"Real Class {real_y}",
            alpha=0.8,
        )
        axes[i].set_title(str(ch).upper())
        axes[i].legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "tstr_compare.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def _check_channel_alignment(cache_path: str, cfg_channels) -> None:
    meta = load_meta(os.path.dirname(cache_path))
    meta_channels = meta.get("channels", [])
    if meta_channels and [c.lower() for c in meta_channels] != [c.lower() for c in cfg_channels]:
        print(f"Warning: cache channels {meta_channels} != cfg channels {cfg_channels}")


def _make_clamped_test_loader(test_loader, clamp_min: float, clamp_max: float):
    from torch.utils.data import DataLoader, TensorDataset

    ds = test_loader.dataset
    if hasattr(ds, "indices"):
        base = ds.dataset
        idx = ds.indices
        x = base.x[idx]
        y = base.y[idx]
    else:
        x = ds.x
        y = ds.y
    x = np.clip(x, clamp_min, clamp_max)
    tensor_ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(tensor_ds, batch_size=32, shuffle=False)


def _save_class_comparison(cache_path: str, channels, out_dir: str) -> None:
    data = np.load(cache_path)
    x = data["x"]
    y = data["y"]
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return
    x0 = x[idx0[0]]
    x1 = x[idx1[0]]
    fig, axes = plt.subplots(len(channels), 1, figsize=(10, 2.5 * len(channels)))
    if len(channels) == 1:
        axes = [axes]
    for i, ch in enumerate(channels):
        axes[i].plot(_standardize_signal(x0[i]), label="Class 0", alpha=0.8)
        axes[i].plot(_standardize_signal(x1[i]), label="Class 1", alpha=0.8)
        axes[i].set_title(str(ch).upper())
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "synth_class_compare.png"), dpi=150)
    plt.close(fig)


def _save_bvp_spectrum(cache_path: str, channels, out_dir: str) -> None:
    data = np.load(cache_path)
    x = data["x"]
    y = data["y"]
    bvp_idx = None
    for i, ch in enumerate(channels):
        if str(ch).lower() == "bvp":
            bvp_idx = i
            break
    if bvp_idx is None:
        return
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return
    sig0 = x[idx0[0], bvp_idx]
    sig1 = x[idx1[0], bvp_idx]
    fft0 = np.abs(np.fft.rfft(sig0))
    fft1 = np.abs(np.fft.rfft(sig1))
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(fft0, label="Class 0")
    ax.plot(fft1, label="Class 1")
    ax.set_title("BVP Spectrum (Magnitude)")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Magnitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "synth_bvp_spectrum.png"), dpi=150)
    plt.close(fig)


def _save_physio_results(
    out_dir: str,
    acc: float,
    f1: float,
    acc_noisy=None,
    acc_restored=None,
    time_freq=None,
    meta=None,
    robust=None,
    language_metrics=None,
):
    import json

    result_path = os.path.join(out_dir, "physio_results.json")
    payload = {
        "meta": meta or {},
        "tstr": {"accuracy": acc, "f1": f1},
        "time_freq": time_freq or {},
        "language_metrics": language_metrics or {},
    }
    if robust is not None:
        payload["robust"] = robust
    elif acc_noisy is not None:
        payload["robust"] = {"noisy_acc": acc_noisy, "restored_acc": acc_restored}
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run(load_config(args.config))
