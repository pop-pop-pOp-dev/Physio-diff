import argparse
import os
from copy import deepcopy

from experiments.common.data import load_config
from src.scripts.run_pipeline import run as physio_main


VARIANTS = {
    "main_competitive": {},
    "legacy_fullstack": {
        "train.training_recipe": "legacy_fullstack",
        "model.model_type": "mechanistic",
        "model.denoiser_type": "dilated",
        "model.conditioning_mode": "additive",
        "model.band_split_mode": "none",
        "model.use_condition_tokens": False,
        "train.enable_text_modules": True,
        "train.use_language_conditioning": True,
        "train.use_text_prototypes": True,
        "train.use_signal_text_cycle": True,
        "train.use_semantic_alignment": True,
        "train.use_artifact_text_conditioning": True,
        "train.anchor_labels": True,
        "train.domain_generalization": True,
        "train.artifact_aware": True,
    },
    "physio_unet": {
        "model.denoiser_type": "physio_unet",
        "model.conditioning_mode": "adagn",
        "model.band_split_mode": "multires",
    },
    "additive_conditioning": {
        "model.conditioning_mode": "additive",
        "model.use_condition_tokens": False,
    },
    "no_multiband": {
        "model.band_split_mode": "none",
    },
    "cross_attention": {
        "model.conditioning_mode": "crossattn",
        "model.use_condition_tokens": True,
    },
    "with_light_freq_prior": {
        "loss.w_freq_end": 0.0002,
    },
    "mechanistic_only": {
        "train.training_recipe": "legacy_fullstack",
        "train.use_language_conditioning": False,
        "train.use_text_prototypes": False,
        "train.use_signal_text_cycle": False,
        "train.use_semantic_alignment": False,
        "train.use_artifact_text_conditioning": False,
        "loss.w_text_proto": 0.0,
        "loss.w_cycle": 0.0,
        "loss.w_artifact_text": 0.0,
        "loss.w_semantic_align": 0.0,
    },
    "language_only": {
        "train.training_recipe": "legacy_fullstack",
        "model.model_type": "standard",
        "loss.w_mech": 0.0,
        "loss.w_kin_start": 0.0,
        "loss.w_kin_end": 0.0,
        "loss.w_freq_start": 0.0,
        "loss.w_freq_end": 0.0,
    },
}


def _set_nested(cfg: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    ref = cfg
    for key in keys[:-1]:
        ref = ref.setdefault(key, {})
    ref[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM ablation matrix for Physio-Diff.")
    parser.add_argument("--config", default="configs/best_improved.yaml")
    parser.add_argument(
        "--source_config",
        default=None,
        help="Optional config whose `data` section overrides base config dataset/split fields.",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS.keys()))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output_root", default="experiments/ablations/llm_outputs")
    parser.add_argument(
        "--reuse_ckpt_path",
        default=None,
        help="Optional checkpoint path to reuse for all variants/seeds (evaluation-only mode).",
    )
    parser.add_argument("--eval_synth_samples_per_class", type=int, default=None)
    parser.add_argument("--eval_gen_batch_size", type=int, default=None)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.source_config:
        source_cfg = load_config(args.source_config)
        base_cfg.setdefault("data", {})
        base_cfg["data"].update(deepcopy(source_cfg.get("data", {})))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.output_root, exist_ok=True)

    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        for seed in seeds:
            cfg = deepcopy(base_cfg)
            cfg.setdefault("project", {})
            cfg["project"]["seed"] = int(seed)
            cfg["project"]["output_dir"] = os.path.join(args.output_root, variant, f"seed_{seed}")
            cfg.setdefault("eval", {})
            if args.reuse_ckpt_path:
                cfg["eval"]["use_existing_model"] = True
                cfg["eval"]["checkpoint_path"] = args.reuse_ckpt_path
            if args.eval_synth_samples_per_class is not None:
                cfg["eval"]["synth_samples_per_class"] = int(args.eval_synth_samples_per_class)
            if args.eval_gen_batch_size is not None:
                cfg["eval"]["gen_batch_size"] = int(args.eval_gen_batch_size)
            for dotted_key, value in VARIANTS[variant].items():
                _set_nested(cfg, dotted_key, value)
            os.makedirs(cfg["project"]["output_dir"], exist_ok=True)
            print(f"Running variant={variant} seed={seed}", flush=True)
            physio_main(cfg)


if __name__ == "__main__":
    main()
