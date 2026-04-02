import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.wesad import build_dataset_cache
from src.utils.project_paths import resolve_project_path


def load_config(path: str) -> Dict:
    import yaml

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


def build_cache(cfg: Dict) -> Tuple[str, Dict]:
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
    meta_path = os.path.join(cfg["data"]["cache_dir"], "dataset_meta.json")
    legacy_meta_path = os.path.join(cfg["data"]["cache_dir"], "wesad_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    elif os.path.exists(legacy_meta_path):
        with open(legacy_meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    return cache_path, meta


def load_cache(cache_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    data = np.load(cache_path, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    acc = data["acc"] if "acc" in data else None
    if acc is not None and acc.dtype == object:
        acc = None
    return x, y, acc


def _split_indices_by_subject(
    subjects: list,
    train_subjects: list,
    val_subjects: list,
    test_subjects: list,
):
    train_idx = [i for i, s in enumerate(subjects) if s in train_subjects]
    val_idx = [i for i, s in enumerate(subjects) if s in val_subjects]
    test_idx = [i for i, s in enumerate(subjects) if s in test_subjects]
    return train_idx, val_idx, test_idx


def make_splits(x, y, acc, meta: Dict, cfg: Dict):
    subjects = meta.get("subjects", [])
    train_subjects = cfg["data"].get("train_subjects", [])
    val_subjects = cfg["data"].get("val_subjects", [])
    test_subjects = cfg["data"].get("test_subjects", [])
    split_strategy = str(cfg["data"].get("split_strategy", "random")).lower()
    if split_strategy in {"subject", "loso", "groupkfold"} and subjects:
        if split_strategy == "loso":
            from src.data.datasets import loso_subject_split

            held_out = test_subjects[0] if test_subjects else None
            train_subjects, val_subjects, test_subjects = loso_subject_split(
                subjects,
                fold_index=int(cfg["data"].get("cv_fold_index", 0)),
                seed=int(cfg["project"]["seed"]),
                val_subject_count=int(cfg["data"].get("val_subject_count", 1)),
                test_subject=held_out,
            )
        elif split_strategy == "groupkfold":
            from src.data.datasets import group_kfold_subject_split

            train_subjects, val_subjects, test_subjects = group_kfold_subject_split(
                subjects,
                n_splits=int(cfg["data"].get("cv_n_splits", 5)),
                fold_index=int(cfg["data"].get("cv_fold_index", 0)),
                seed=int(cfg["project"]["seed"]),
                val_subject_count=int(cfg["data"].get("val_subject_count", 1)),
            )
        elif not (train_subjects and val_subjects and test_subjects):
            from src.data.datasets import split_subjects

            train_subjects, val_subjects, test_subjects = split_subjects(
                subjects,
                val_ratio=float(cfg["data"].get("val_ratio", 0.1)),
                test_ratio=float(cfg["data"].get("test_ratio", 0.1)),
                seed=int(cfg["project"]["seed"]),
            )
        train_idx, val_idx, test_idx = _split_indices_by_subject(
            subjects, train_subjects, val_subjects, test_subjects
        )
    else:
        rng = np.random.default_rng(cfg["project"]["seed"])
        idx = np.arange(len(x))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * cfg["data"].get("train_ratio", 0.8))
        n_val = int(n * cfg["data"].get("val_ratio", 0.1))
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
    splits = {}
    for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        splits[name] = {
            "x": x[indices],
            "y": y[indices],
            "acc": acc[indices] if acc is not None else None,
        }
    return splits


def make_dataloaders(split: Dict, batch_size: int, shuffle: bool = True):
    ds = TensorDataset(
        torch.from_numpy(split["x"]).float(),
        torch.from_numpy(split["y"]).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
