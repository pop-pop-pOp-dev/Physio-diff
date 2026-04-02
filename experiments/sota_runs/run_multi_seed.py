import argparse
import os
from copy import deepcopy
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.common.data import load_config
from experiments.ddpm_baseline.run_ddpm import main as ddpm_main
from experiments.cgan_baseline.run_cgan import main as cgan_main
from experiments.wgan_gp_baseline.run_wgan_gp import main as wgan_gp_main
from experiments.timegan_baseline.run_timegan import main as timegan_main
from experiments.tsgm_baseline.run_tsgm import main as tsgm_main
from experiments.csdi_baseline.run_csdi import main as csdi_main
from experiments.tsdiff_baseline.run_tsdiff import main as tsdiff_main
from src.scripts.run_pipeline import run as physio_main


METHODS = {
    "physio_diff_main": ("configs/main_competitive.yaml", physio_main),
    "physio_diff_legacy_fullstack": ("configs/best_improved.yaml", physio_main),
    "physio_diff_physio_unet": ("configs/main_competitive.yaml", physio_main),
    "physio_diff_main_additive_cond": ("configs/main_competitive.yaml", physio_main),
    "physio_diff_main_no_multiband": ("configs/main_competitive.yaml", physio_main),
    "physio_diff_mechanistic_only": ("configs/best_improved.yaml", physio_main),
    "physio_diff_language_only": ("configs/best_improved.yaml", physio_main),
    "physio_diff_no_cycle": ("configs/best_improved.yaml", physio_main),
    "physio_diff_no_semantic_align": ("configs/best_improved.yaml", physio_main),
    "physio_diff_no_artifact_text": ("configs/best_improved.yaml", physio_main),
    "ddpm": ("experiments/ddpm_baseline/configs/base.yaml", ddpm_main),
    "cgan": ("experiments/cgan_baseline/configs/base.yaml", cgan_main),
    "wgan_gp": ("experiments/wgan_gp_baseline/configs/base.yaml", wgan_gp_main),
    "timegan": ("experiments/timegan_baseline/configs/base.yaml", timegan_main),
    "tsgm": ("experiments/tsgm_baseline/configs/base.yaml", tsgm_main),
    "csdi": ("experiments/csdi_baseline/configs/base.yaml", csdi_main),
    "tsdiff": ("experiments/tsdiff_baseline/configs/base.yaml", tsdiff_main),
}


def _is_physio_method(method: str) -> bool:
    return str(method).startswith("physio_diff")


def _apply_physio_variant(cfg: dict, method_name: str) -> None:
    cfg.setdefault("train", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("model", {})
    if method_name in {"physio_diff_main", "physio_diff_legacy_fullstack"}:
        return
    if method_name == "physio_diff_physio_unet":
        cfg["model"]["denoiser_type"] = "physio_unet"
        cfg["model"]["conditioning_mode"] = "adagn"
        cfg["model"]["band_split_mode"] = "multires"
        cfg["model"]["use_condition_tokens"] = True
        return
    if method_name == "physio_diff_main_additive_cond":
        cfg["model"]["conditioning_mode"] = "additive"
        cfg["model"]["use_condition_tokens"] = False
        return
    if method_name == "physio_diff_main_no_multiband":
        cfg["model"]["denoiser_type"] = "physio_unet"
        cfg["model"]["conditioning_mode"] = "adagn"
        cfg["model"]["band_split_mode"] = "none"
        return
    if method_name == "physio_diff_mechanistic_only":
        cfg["train"]["use_language_conditioning"] = False
        cfg["train"]["use_text_prototypes"] = False
        cfg["train"]["use_signal_text_cycle"] = False
        cfg["train"]["use_semantic_alignment"] = False
        cfg["train"]["use_artifact_text_conditioning"] = False
        cfg["train"]["enable_text_modules"] = False
        cfg["loss"]["w_text_proto"] = 0.0
        cfg["loss"]["w_cycle"] = 0.0
        cfg["loss"]["w_artifact_text"] = 0.0
        cfg["loss"]["w_semantic_align"] = 0.0
        return
    if method_name == "physio_diff_language_only":
        cfg["model"]["model_type"] = "standard"
        cfg["loss"]["w_mech"] = 0.0
        cfg["loss"]["w_kin_start"] = 0.0
        cfg["loss"]["w_kin_end"] = 0.0
        cfg["loss"]["w_freq_start"] = 0.0
        cfg["loss"]["w_freq_end"] = 0.0
        return
    if method_name == "physio_diff_no_cycle":
        cfg["train"]["use_signal_text_cycle"] = False
        cfg["loss"]["w_cycle"] = 0.0
        return
    if method_name == "physio_diff_no_semantic_align":
        cfg["train"]["use_semantic_alignment"] = False
        cfg["loss"]["w_semantic_align"] = 0.0
        return
    if method_name == "physio_diff_no_artifact_text":
        cfg["train"]["use_artifact_text_conditioning"] = False
        cfg["loss"]["w_artifact_text"] = 0.0
        return
    raise ValueError(f"Unknown Physio-Diff variant: {method_name}")


def _has_synth_artifact(method: str, out_dir: str) -> bool:
    if _is_physio_method(method):
        return os.path.exists(os.path.join(out_dir, "synthetic_normalized.npz"))
    return os.path.exists(os.path.join(out_dir, "synthetic.npz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output_root", default="experiments/sota_runs/outputs")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated subset of methods to run (e.g. 'ddpm,cgan,wgan_gp'). Use 'all' to run everything.",
    )
    parser.add_argument(
        "--physio",
        default="train",
        choices=["train", "reuse_ckpt", "skip"],
        help="How to handle Physio-Diff in multi-seed runs.",
    )
    parser.add_argument(
        "--eval_synth_samples_per_class",
        type=int,
        default=None,
        help="Override eval.synth_samples_per_class for all methods (if applicable).",
    )
    parser.add_argument(
        "--eval_classifier_epochs",
        type=int,
        default=None,
        help="Override eval.classifier_epochs for all methods (if applicable).",
    )
    parser.add_argument(
        "--eval_classifier_lr",
        type=float,
        default=None,
        help="Override eval.classifier_lr for all methods (if applicable).",
    )
    parser.add_argument(
        "--eval_gen_batch_size",
        type=int,
        default=None,
        help="Override eval.gen_batch_size for all methods (if applicable).",
    )
    parser.add_argument(
        "--physio_sample_steps",
        type=int,
        default=None,
        help="Override eval.sample_steps for Physio-Diff (enables DDIM if < timesteps).",
    )
    parser.add_argument(
        "--physio_epochs",
        type=int,
        default=None,
        help="Override train.epochs for Physio-Diff only.",
    )
    parser.add_argument(
        "--physio_ckpt_root",
        default=None,
        help=(
            "Optional root dir containing per-seed Physio-Diff checkpoints when using --physio reuse_ckpt. "
            "Expected layout: <root>/seed_<k>/physio_diff_best.pt"
        ),
    )
    parser.add_argument(
        "--source_config",
        default=None,
        help=(
            "Optional source-domain config path. If set, its `data` section is used to override "
            "dataset/split/cache/path fields for all methods to ensure same training domain."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional override for project.device (e.g. cuda or cpu) for all methods.",
    )
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.output_root, exist_ok=True)
    source_cfg = load_config(args.source_config) if args.source_config else None

    if args.methods.strip().lower() == "all":
        selected = list(METHODS.keys())
    else:
        selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    for name in selected:
        if name not in METHODS:
            raise ValueError(f"Unknown method: {name}. Available: {sorted(METHODS.keys())}")
        cfg_path, runner = METHODS[name]
        for seed in seeds:
            cfg = load_config(cfg_path)
            cfg = deepcopy(cfg)
            if source_cfg is not None:
                cfg.setdefault("data", {})
                cfg["data"].update(deepcopy(source_cfg.get("data", {})))
                if "project" in source_cfg and "device" in source_cfg["project"]:
                    cfg.setdefault("project", {})
                    cfg["project"]["device"] = source_cfg["project"]["device"]
            if args.device:
                cfg.setdefault("project", {})
                cfg["project"]["device"] = str(args.device)
            cfg["project"]["seed"] = seed
            out_dir = os.path.join(args.output_root, name, f"seed_{seed}")
            cfg["project"]["output_dir"] = out_dir
            cfg.setdefault("eval", {})
            # Enforce consistent evaluation knobs across methods.
            cfg["eval"].setdefault("match_stats", False)
            cfg["eval"].setdefault("clip_min", -5.0)
            cfg["eval"].setdefault("clip_max", 5.0)
            if args.eval_synth_samples_per_class is not None:
                cfg["eval"]["synth_samples_per_class"] = int(args.eval_synth_samples_per_class)
            if args.eval_classifier_epochs is not None:
                cfg["eval"]["classifier_epochs"] = int(args.eval_classifier_epochs)
            if args.eval_classifier_lr is not None:
                cfg["eval"]["classifier_lr"] = float(args.eval_classifier_lr)
            if args.eval_gen_batch_size is not None:
                cfg["eval"]["gen_batch_size"] = int(args.eval_gen_batch_size)

            if _is_physio_method(name):
                if args.physio == "skip":
                    continue
                if args.physio == "reuse_ckpt":
                    cfg["eval"]["use_existing_model"] = True
                else:
                    cfg["eval"]["use_existing_model"] = False
                cfg["eval"].setdefault("checkpoint_path", None)
                if args.physio_sample_steps is not None:
                    cfg["eval"]["sample_steps"] = int(args.physio_sample_steps)
                if args.physio_epochs is not None:
                    cfg.setdefault("train", {})
                    cfg["train"]["epochs"] = int(args.physio_epochs)
                if args.physio == "reuse_ckpt" and not cfg["eval"].get("checkpoint_path"):
                    if args.physio_ckpt_root:
                        candidate = os.path.join(
                            args.physio_ckpt_root, f"seed_{seed}", "physio_diff_best.pt"
                        )
                        if os.path.exists(candidate):
                            cfg["eval"]["checkpoint_path"] = candidate
                _apply_physio_variant(cfg, name)
            else:
                cfg["eval"]["use_existing_model"] = False
                cfg["eval"]["checkpoint_path"] = None
            os.makedirs(out_dir, exist_ok=True)
            result_path = os.path.join(out_dir, "results.json")
            if _is_physio_method(name):
                result_path = os.path.join(out_dir, "physio_results.json")
            if os.path.exists(result_path) and _has_synth_artifact(name, out_dir):
                print(f"[{name}] seed {seed} already exists, skipping.", flush=True)
                continue
            print(f"Running {name} seed={seed}...", flush=True)
            try:
                runner(cfg)
                print(f"Completed {name} seed={seed}.", flush=True)
            except Exception as exc:
                err_path = os.path.join(out_dir, "error.txt")
                with open(err_path, "w", encoding="utf-8") as handle:
                    handle.write(f"{type(exc).__name__}: {exc}\n")
                print(f"Failed {name} seed={seed}: {exc}", flush=True)


if __name__ == "__main__":
    main()
