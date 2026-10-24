"""Disco engine wrapper -- the app-facing half of the Gradio UI.

Companion to disco_engine_core.py (verbatim engine functions extracted from
disco.py, Disco Diffusion v5.7). This file holds everything the notebook
used to keep as top-level script state, rewritten as explicit functions:

- NOTEBOOK_DEFAULTS: every setting the notebook assigned, with the
  notebook's own default values (taken from disco.py, not invented).
- load_models(): the model-settings section of the notebook, as a
  function -- downloads (no-op if files exist), model_config branches,
  secondary model, CLIP list, normalize, LPIPS. Results are cached:
  CLIP/secondary/normalize/LPIPS load ONCE and are reused until the
  diffusion model or CLIP set changes (the app equivalent of the
  notebook's kernel-restart rule).
- build_args(): the "Do the Run" cell's args dict, key for key, including
  timestep_respacing = f'ddim{steps}' and
  diffusion_steps = (1000//steps)*steps when steps < 1000.
- sync_settings_globals(): sets every bare module global that do_run()
  and save_settings() read (the notebook's globals trap, made explicit).
- run_generation(): one full still-image generation: sync, prepare,
  build args, call do_run(), collect the saved frames.

What this file deliberately does NOT do (phase-1 scope):
- Animation modes (2D/3D/Video Input), video stitching, RAFT/flow.
- OpenCLIP (laion) models: their flags exist and default False, but
  loading them is a TODO (see below).
- The 'custom' diffusion model path and the KaliYuga pixelart /
  watercolor / pulp models: model_config branches for those were not
  ported; choosing one raises a clear error (TODO).

TODOs needing Alan's PC to resolve (do not guess these):
- TODO(verify): IPython display calls inside do_run() (clear_output /
  display / Output widget) running headless -- they should degrade to
  no-ops/prints, but this must be confirmed on his machine. The app
  does not depend on them: it watches progress.png and batchFolder.
- TODO(verify): reusing a cached diffusion model object across
  generations with the SAME steps-derived schedule key. The code
  recreates the model whenever the schedule key changes, which is the
  notebook-faithful behaviour; same-key reuse is the one new behaviour
  and needs his acceptance run to confirm.
- TODO(phase-2): OpenCLIP laion models, 'custom' model path, KaliYuga
  model_config branches, animation modes, video stitching, Stop button
  (stop_on_next_loop exists in the engine but has no UI control yet).
"""

import gc
import math
import os
import random
from glob import glob
from types import SimpleNamespace

import torch

import disco_engine_core as core
from CLIP import clip
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# Engine globals that must exist before any cache check touches them.
# (The notebook created these by running its settings cells; the app
# creates them here, empty, and load_models()/sync fills them in.)
core.model = None
core.diffusion = None
core.clip_models = []
core.secondary_model = None
core.normalize = None
core.lpips_model = None
core.args = None

# ---------------------------------------------------------------------------
# Notebook defaults (verbatim values from disco.py)
# ---------------------------------------------------------------------------

# All 24 CLIP flags the notebook knows, in the order its CLIP section lists
# them. Phase-1 UI offers the first six; the rest stay False.
CLIP_FLAGS = [
    "ViTB32", "ViTB16", "ViTL14", "ViTL14_336px", "RN101", "RN50",
    "RN50x4", "RN50x16", "RN50x64",
    "ViTB32_laion2b_e16", "ViTB32_laion400m_e31", "ViTB32_laion400m_32",
    "ViTB32quickgelu_laion400m_e31", "ViTB32quickgelu_laion400m_e32",
    "ViTB16_laion400m_e31", "ViTB16_laion400m_e32",
    "RN50_yffcc15m", "RN50_cc12m", "RN50_quickgelu_yfcc15m",
    "RN50_quickgelu_cc12m", "RN101_yfcc15m", "RN101_quickgelu_yfcc15m",
]

# Diffusion models the notebook's dropdown offers. Phase 1 supports the
# first three (their model_config branches are ported verbatim below).
# The rest raise a clear error until their branches are ported (TODO).
SUPPORTED_MODELS = [
    "512x512_diffusion_uncond_finetune_008100",
    "256x256_diffusion_uncond",
    "portrait_generator_v001",
]
ALL_NOTEBOOK_MODELS = SUPPORTED_MODELS + [
    "pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256",
    "pixel_art_diffusion_soft_256", "pixelartdiffusion4k",
    "watercolordiffusion_2", "watercolordiffusion", "PulpSciFiDiffusion",
    "custom",
]

NOTEBOOK_DEFAULTS = {
    # model settings (notebook section 2)
    "diffusion_model": "512x512_diffusion_uncond_finetune_008100",
    "use_secondary_model": True,
    "diffusion_sampling_mode": "ddim",
    "custom_path": "",
    "use_checkpoint": True,
    "check_model_SHA": False,
    # basic settings (notebook section 3)
    "batch_name": "TimeToDisco",
    "steps": 250,
    "width_height": [1280, 768],
    "clip_guidance_scale": 5000,
    "tv_scale": 0,
    "range_scale": 150,
    "sat_scale": 0,
    "cutn_batches": 4,
    "skip_augs": False,
    "init_image": None,
    "init_scale": 1000,
    "skip_steps": 10,
    # animation settings (still-image mode uses the defaults only)
    "animation_mode": "None",
    "key_frames": True,
    "max_frames": 10000,
    "interp_spline": "Linear",
    "angle": "0:(0)",
    "zoom": "0: (1), 10: (1.05)",
    "translation_x": "0: (0)",
    "translation_y": "0: (0)",
    "translation_z": "0: (10.0)",
    "rotation_3d_x": "0: (0)",
    "rotation_3d_y": "0: (0)",
    "rotation_3d_z": "0: (0)",
    "midas_depth_model": "dpt_large",
    "midas_weight": 0.3,
    "near_plane": 200,
    "far_plane": 10000,
    "fov": 40,
    "padding_mode": "border",
    "sampling_mode": "bicubic",
    "turbo_mode": False,
    "turbo_steps": "3",
    "turbo_preroll": 10,
    "frames_scale": 1500,
    "frames_skip_steps": "60%",
    "vr_mode": False,
    "vr_eye_angle": 0.5,
    "vr_ipd": 5.0,
    # saving / advanced settings
    "intermediate_saves": 0,
    "intermediates_in_subfolder": True,
    "perlin_init": False,
    "perlin_mode": "mixed",
    "set_seed": "random_seed",
    "eta": 0.8,
    "clamp_grad": True,
    "clamp_max": 0.05,
    "randomize_class": True,
    "clip_denoised": False,
    "fuzzy_prompt": False,
    "rand_mag": 0.05,
    # cutn scheduling (strings; the run cell eval()s them)
    "cut_overview": "[12]*400+[4]*600",
    "cut_innercut": "[4]*400+[12]*600",
    "cut_ic_pow": "[1]*1000",
    "cut_icgray_p": "[0.2]*400+[0]*600",
    # transformation settings
    "use_vertical_symmetry": False,
    "use_horizontal_symmetry": False,
    "transformation_percent": [0.09],
    # video-init settings (unused in phase-1 stills, but save_settings()
    # reads them, so they must exist)
    "video_init_path": "init.mp4",
    "extract_nth_frame": 2,
    "video_init_seed_continuity": False,
    "video_init_flow_warp": True,
    "video_init_flow_blend": 0.999,
    "video_init_check_consistency": False,
    "video_init_blend_mode": "optical flow",
    "video_init_steps": 100,
    "video_init_clip_guidance_scale": 1000,
    "video_init_tv_scale": 0.1,
    "video_init_range_scale": 150,
    "video_init_sat_scale": 300,
    "video_init_cutn_batches": 4,
    "video_init_skip_steps": 50,
    "video_init_frames_scale": 15000,
    "video_init_frames_skip_steps": "70%",
    # run-cell settings
    "display_rate": 20,
    "n_batches": 1,  # notebook default is 50; phase-1 stills make 1 image
    "batch_size": 1,
}

# Alan's proven portrait recipe (Oct 24, 2026 runs) -- these are the UI
# defaults in app.py, kept here so the recipe lives in one place.
ALAN_PORTRAIT_RECIPE = {
    "diffusion_model": "portrait_generator_v001",
    "width_height": [512, 768],
    "steps": 300,
    "clip_guidance_scale": 10000,
    "tv_scale": 200,
    "sat_scale": 10110,
    "cut_ic_pow_number": 10,
    "use_horizontal_symmetry": True,
    "transformation_percent": [0.09],
    "clip_on": ["ViTB32", "ViTB16", "ViTL14", "RN101", "RN50", "RN50x4"],
}

# ---------------------------------------------------------------------------
# Load-once state
# ---------------------------------------------------------------------------

# Set by request_stop(), read by the app's queue loop between prompts.
_STOP_REQUESTED = False

_STATE = {
    "clip_key": None,
    "diff_key": None,
    "device": None,
    "model_path": None,
    "out_dir": None,
}


def _clip_key(diffusion_model, clip_on, use_secondary_model, use_checkpoint):
    return (
        diffusion_model,
        tuple(sorted(clip_on)),
        bool(use_secondary_model),
        bool(use_checkpoint),
    )


def _diff_key(diffusion_model, timestep_respacing, diffusion_steps, use_fp16):
    return (diffusion_model, timestep_respacing, diffusion_steps, use_fp16)


def _model_config_for(diffusion_model, use_checkpoint, use_cpu=False):
    """The notebook's model_config branches, verbatim, for supported models."""
    model_config = model_and_diffusion_defaults()
    if diffusion_model == "512x512_diffusion_uncond_finetune_008100":
        model_config.update({
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,  # taken care of later, per notebook
            "rescale_timesteps": True,
            "timestep_respacing": 250,  # taken care of later, per notebook
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": not use_cpu,
            "use_scale_shift_norm": True,
        })
    elif diffusion_model == "256x256_diffusion_uncond":
        model_config.update({
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": 250,
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": not use_cpu,
            "use_scale_shift_norm": True,
        })
    elif diffusion_model == "portrait_generator_v001":
        model_config.update({
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 128,
            "num_heads": 4,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        })
    elif diffusion_model == "custom":
        # Notebook's Custom Model Settings cell, verbatim. The checkpoint
        # itself comes from custom_path (see _prepare_diffusion).
        model_config.update({
            "attention_resolutions": "16",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "ddim100",
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 128,
            "num_heads": 1,
            "num_res_blocks": 2,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": False,
        })
    else:
        # TODO(phase-2): port the notebook's remaining model_config branches
        # (KaliYuga pixelart/watercolor/pulp, 'custom').
        raise ValueError(
            f"Diffusion model '{diffusion_model}' is not wired in phase 1. "
            f"Supported: {', '.join(SUPPORTED_MODELS)}."
        )
    return model_config


# ---------------------------------------------------------------------------
# Model loading (load-once, with reload on change)
# ---------------------------------------------------------------------------

def init_paths(install_dir):
    """Point the engine at Alan's install (read-only use).

    Mirrors the notebook's local branch: root_path is the working
    directory, models live in <root>/models, outputs in <root>/images_out.
    """
    model_path = os.path.join(install_dir, "models")
    out_dir = os.path.join(install_dir, "images_out")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    _STATE["model_path"] = model_path
    _STATE["out_dir"] = out_dir
    core.model_path = model_path
    core.check_model_SHA = False
    core.is_colab = False
    # MiDaS model map (notebook's default_models, verbatim): only read by
    # init_midas_depth_model, i.e. 3D paths, but it must exist on core.
    core.default_models = {
        "midas_v21_small": f"{model_path}/midas_v21_small-70d6b9c8.pt",
        "midas_v21": f"{model_path}/midas_v21-f6b98070.pt",
        "dpt_large": f"{model_path}/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": f"{model_path}/dpt_hybrid-midas-501f0c75.pt",
        "dpt_hybrid_nyu": f"{model_path}/dpt_hybrid_nyu-2ce69ec7.pt",
    }
    return model_path, out_dir


def _device():
    if _STATE["device"] is None:
        _STATE["device"] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        core.device = _STATE["device"]
        core.DEVICE = _STATE["device"]
    return _STATE["device"]


def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_models(diffusion_model, clip_on, use_secondary_model=True,
                use_checkpoint=True):
    """Load (or reuse) CLIP models, secondary model, normalize, LPIPS.

    This is the notebook's model-settings section as a function. The
    diffusion UNet itself is created in prepare_run(), because the
    notebook creates it in the run cell -- but it is cached there by
    schedule key, so repeat generations with the same settings reuse it.
    Returns a status string for the UI ('loaded' vs 'reused').
    """
    if _STATE["model_path"] is None:
        raise RuntimeError("init_paths() must run before load_models().")
    device = _device()
    key = _clip_key(diffusion_model, clip_on, use_secondary_model,
                    use_checkpoint)
    if _STATE["clip_key"] == key and core.clip_models:
        return "reused loaded models"

    # Reload: drop the old models first (kernel-restart rule, app form).
    _free_gpu()
    core.clip_models = []
    core.secondary_model = None

    # Diffusion model file + secondary file (no-op prints if present,
    # exactly like the notebook's download_model()).
    core.download_model(diffusion_model)
    if use_secondary_model:
        core.download_model("secondary")
        core.secondary_model = core.SecondaryDiffusionImageNet2()
        core.secondary_model.load_state_dict(torch.load(
            os.path.join(_STATE["model_path"],
                         "secondary_model_imagenet_2.pth"),
            map_location="cpu"))
        core.secondary_model.eval().requires_grad_(False).to(device)

    # CLIP list, in the notebook's exact order and call shapes.
    clip_on_set = set(clip_on)
    for flag in ("ViTB32", "ViTB16", "ViTL14", "ViTL14_336px", "RN101",
                 "RN50", "RN50x4", "RN50x16", "RN50x64"):
        if flag in clip_on_set:
            openai_name = {
                "ViTB32": "ViT-B/32", "ViTB16": "ViT-B/16",
                "ViTL14": "ViT-L/14", "ViTL14_336px": "ViT-L/14@336px",
                "RN101": "RN101", "RN50": "RN50", "RN50x4": "RN50x4",
                "RN50x16": "RN50x16", "RN50x64": "RN50x64",
            }[flag]
            core.clip_models.append(
                clip.load(openai_name, jit=False)[0]
                .eval().requires_grad_(False).to(device))
    laion_on = [f for f in clip_on_set if f not in
                ("ViTB32", "ViTB16", "ViTL14", "ViTL14_336px", "RN101",
                 "RN50", "RN50x4", "RN50x16", "RN50x64")]
    if laion_on:
        # TODO(phase-2): OpenCLIP loading (open_clip.create_model calls
        # from the notebook's CLIP section) is not wired yet.
        raise ValueError(
            f"OpenCLIP models are not wired in phase 1: {sorted(laion_on)}")

    core.normalize = core.T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    import lpips as _lpips
    core.lpips_model = _lpips.LPIPS(net="vgg").to(device)

    _STATE["clip_key"] = key
    return "loaded models"


def _prepare_diffusion(diffusion_model, timestep_respacing,
                       diffusion_steps, use_checkpoint, custom_path=""):
    """Create (or reuse) the diffusion UNet + schedule, run-cell style."""
    model_config = _model_config_for(diffusion_model, use_checkpoint)
    model_config.update({
        "timestep_respacing": timestep_respacing,
        "diffusion_steps": diffusion_steps,
    })
    use_fp16 = bool(model_config.get("use_fp16"))
    key = _diff_key(diffusion_model, timestep_respacing, diffusion_steps,
                    use_fp16)
    if _STATE["diff_key"] == key and core.model is not None:
        return "reused diffusion model"
    _free_gpu()
    device = _device()
    model, diffusion = create_model_and_diffusion(**model_config)
    if diffusion_model == "custom":
        if not custom_path or not os.path.isfile(custom_path):
            raise ValueError(
                "Custom model needs a custom_path pointing at a .pt file, "
                f"got: {custom_path!r}")
        model.load_state_dict(torch.load(custom_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(
            os.path.join(_STATE["model_path"],
                         core.get_model_filename(diffusion_model)),
            map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    core.model = model
    core.diffusion = diffusion
    _STATE["diff_key"] = key
    return "loaded diffusion model"


# ---------------------------------------------------------------------------
# Args building (the "Do the Run" cell, key for key)
# ---------------------------------------------------------------------------

def _series_for(settings):
    """Keyframe/inbetween series, exactly like the notebook's settings cell."""
    core.max_frames = 10000
    core.interp_spline = settings["interp_spline"]
    core.key_frames = settings["key_frames"]
    out = {}
    if settings["key_frames"]:
        out["angle_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["angle"]))
        out["zoom_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["zoom"]))
        out["translation_x_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["translation_x"]))
        out["translation_y_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["translation_y"]))
        out["translation_z_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["translation_z"]))
        out["rotation_3d_x_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["rotation_3d_x"]))
        out["rotation_3d_y_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["rotation_3d_y"]))
        out["rotation_3d_z_series"] = core.get_inbetweens(
            core.parse_key_frames(settings["rotation_3d_z"]))
    else:
        # Notebook behaviour with key_frames off is undefined for the
        # series; phase 1 always runs with key_frames on (the default).
        raise ValueError("key_frames off is not supported in phase 1.")
    return out


def build_args(settings, batch_num, seed):
    """Build the run-cell args dict from a settings dict.

    `settings` is NOTEBOOK_DEFAULTS overlaid with the UI's values.
    Returns a SimpleNamespace, like the notebook's `args`.
    """
    steps = settings["steps"]
    # The run cell's own derivation (this is why steps and
    # diffusion_steps never need to match by hand).
    timestep_respacing = f"ddim{steps}"
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps

    width_height = settings["width_height"]
    side_x = (width_height[0] // 64) * 64
    side_y = (width_height[1] // 64) * 64

    skip_step_ratio = int(
        settings["frames_skip_steps"].rstrip("%")) / 100
    calc_frames_skip_steps = math.floor(steps * skip_step_ratio)
    if steps <= calc_frames_skip_steps:
        raise ValueError(
            "You can't skip more steps than your total steps")

    series = _series_for(settings)
    text_prompts = {0: [settings["prompt"]]}
    image_prompts = {}

    args = {
        "batchNum": batch_num,
        "prompts_series": core.split_prompts(text_prompts),
        "image_prompts_series": core.split_prompts(image_prompts)
        if image_prompts else None,
        "seed": seed,
        "display_rate": settings["display_rate"],
        "n_batches": settings["n_batches"],
        "batch_size": settings["batch_size"],
        "batch_name": settings["batch_name"],
        "steps": steps,
        "diffusion_sampling_mode": settings["diffusion_sampling_mode"],
        "width_height": width_height,
        "clip_guidance_scale": settings["clip_guidance_scale"],
        "tv_scale": settings["tv_scale"],
        "range_scale": settings["range_scale"],
        "sat_scale": settings["sat_scale"],
        "cutn_batches": settings["cutn_batches"],
        "init_image": settings["init_image"],
        "init_scale": settings["init_scale"],
        "skip_steps": settings["skip_steps"],
        "side_x": side_x,
        "side_y": side_y,
        "timestep_respacing": timestep_respacing,
        "diffusion_steps": diffusion_steps,
        "animation_mode": "None",
        "video_init_path": settings["video_init_path"],
        "extract_nth_frame": settings["extract_nth_frame"],
        "video_init_seed_continuity": settings["video_init_seed_continuity"],
        "key_frames": settings["key_frames"],
        "max_frames": 1,  # animation_mode 'None' makes exactly 1 frame
        "interp_spline": settings["interp_spline"],
        "start_frame": 0,
        "angle": settings["angle"],
        "zoom": settings["zoom"],
        "translation_x": settings["translation_x"],
        "translation_y": settings["translation_y"],
        "translation_z": settings["translation_z"],
        "rotation_3d_x": settings["rotation_3d_x"],
        "rotation_3d_y": settings["rotation_3d_y"],
        "rotation_3d_z": settings["rotation_3d_z"],
        "midas_depth_model": settings["midas_depth_model"],
        "midas_weight": settings["midas_weight"],
        "near_plane": settings["near_plane"],
        "far_plane": settings["far_plane"],
        "fov": settings["fov"],
        "padding_mode": settings["padding_mode"],
        "sampling_mode": settings["sampling_mode"],
        "angle_series": series["angle_series"],
        "zoom_series": series["zoom_series"],
        "translation_x_series": series["translation_x_series"],
        "translation_y_series": series["translation_y_series"],
        "translation_z_series": series["translation_z_series"],
        "rotation_3d_x_series": series["rotation_3d_x_series"],
        "rotation_3d_y_series": series["rotation_3d_y_series"],
        "rotation_3d_z_series": series["rotation_3d_z_series"],
        "frames_scale": settings["frames_scale"],
        "skip_step_ratio": skip_step_ratio,
        "calc_frames_skip_steps": calc_frames_skip_steps,
        "text_prompts": text_prompts,
        "image_prompts": image_prompts,
        "cut_overview": eval(settings["cut_overview"]),
        "cut_innercut": eval(settings["cut_innercut"]),
        "cut_ic_pow": eval(settings["cut_ic_pow"]),
        "cut_icgray_p": eval(settings["cut_icgray_p"]),
        "intermediate_saves": settings["intermediate_saves"],
        "intermediates_in_subfolder": settings["intermediates_in_subfolder"],
        "steps_per_checkpoint": _steps_per_checkpoint(settings),
        "perlin_init": settings["perlin_init"],
        "perlin_mode": settings["perlin_mode"],
        "set_seed": settings["set_seed"],
        "eta": settings["eta"],
        "clamp_grad": settings["clamp_grad"],
        "clamp_max": settings["clamp_max"],
        "skip_augs": settings["skip_augs"],
        "randomize_class": settings["randomize_class"],
        "clip_denoised": settings["clip_denoised"],
        "fuzzy_prompt": settings["fuzzy_prompt"],
        "rand_mag": settings["rand_mag"],
        "turbo_mode": settings["turbo_mode"],
        "turbo_steps": settings["turbo_steps"],
        "turbo_preroll": settings["turbo_preroll"],
        "use_vertical_symmetry": settings["use_vertical_symmetry"],
        "use_horizontal_symmetry": settings["use_horizontal_symmetry"],
        "transformation_percent": settings["transformation_percent"],
        "video_init_steps": settings["video_init_steps"],
        "video_init_clip_guidance_scale":
            settings["video_init_clip_guidance_scale"],
        "video_init_tv_scale": settings["video_init_tv_scale"],
        "video_init_range_scale": settings["video_init_range_scale"],
        "video_init_sat_scale": settings["video_init_sat_scale"],
        "video_init_cutn_batches": settings["video_init_cutn_batches"],
        "video_init_skip_steps": settings["video_init_skip_steps"],
        "video_init_frames_scale": settings["video_init_frames_scale"],
        "video_init_frames_skip_steps":
            settings["video_init_frames_skip_steps"],
        "video_init_flow_warp": settings["video_init_flow_warp"],
        "video_init_flow_blend": settings["video_init_flow_blend"],
        "video_init_check_consistency":
            settings["video_init_check_consistency"],
        "video_init_blend_mode": settings["video_init_blend_mode"],
    }
    return SimpleNamespace(**args), timestep_respacing, diffusion_steps


def _steps_per_checkpoint(settings):
    """The notebook's saving-settings logic, verbatim."""
    intermediate_saves = settings["intermediate_saves"]
    steps = settings["steps"]
    skip_steps = settings["skip_steps"]
    if type(intermediate_saves) is not list:
        if intermediate_saves:
            steps_per_checkpoint = math.floor(
                (steps - skip_steps - 1) // (intermediate_saves + 1))
            return steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        return steps + 10
    return None


# ---------------------------------------------------------------------------
# Settings globals (the notebook's globals trap, made explicit)
# ---------------------------------------------------------------------------

def request_stop():
    """Stop the running generation after the current step.

    do_run() checks this flag each step (the notebook's stop), so the
    run ends cleanly and still saves what it finished. The queue loop
    also checks stop_requested() between prompts, so Stop ends the
    whole queue, not just the current prompt.
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    core.stop_on_next_loop = True


def stop_requested():
    return _STOP_REQUESTED


def clear_stop():
    global _STOP_REQUESTED
    _STOP_REQUESTED = False


def sync_settings_globals(settings, batch_folder, batch_num, seed,
                          width_height, side_x, side_y, diffusion_steps):
    """Set every bare global do_run()/save_settings() read on core.

    Without this, do_run() would silently use stale values from a
    previous generation -- the exact trap the build plan warns about.
    """
    g = core.__dict__
    g["batchFolder"] = batch_folder
    g["batch_name"] = settings["batch_name"]
    g["batchNum"] = batch_num
    g["batch_size"] = settings["batch_size"]
    g["stop_on_next_loop"] = False
    g["resume_run"] = False
    g["start_frame"] = 0
    g["animation_mode"] = "None"
    g["turbo_mode"] = settings["turbo_mode"]
    g["turbo_steps"] = settings["turbo_steps"]
    g["turbo_preroll"] = settings["turbo_preroll"]
    g["vr_mode"] = settings["vr_mode"]
    g["vr_eye_angle"] = settings["vr_eye_angle"]
    g["vr_ipd"] = settings["vr_ipd"]
    g["TRANSLATION_SCALE"] = 1.0 / 200.0
    g["partialFolder"] = None  # intermediates off in phase 1
    g["flo_folder"] = None  # video-input only
    g["videoFramesFolder"] = None  # video-input only
    g["perlin_init"] = settings["perlin_init"]
    g["perlin_mode"] = settings["perlin_mode"]
    g["clip_denoised"] = settings["clip_denoised"]
    g["randomize_class"] = settings["randomize_class"]
    g["eta"] = settings["eta"]
    g["skip_steps"] = settings["skip_steps"]
    g["skip_augs"] = settings["skip_augs"]
    g["fuzzy_prompt"] = settings["fuzzy_prompt"]
    g["rand_mag"] = settings["rand_mag"]
    g["clamp_grad"] = settings["clamp_grad"]
    g["clamp_max"] = settings["clamp_max"]
    g["text_prompts"] = {0: [settings["prompt"]]}
    g["image_prompts"] = {}
    g["clip_guidance_scale"] = settings["clip_guidance_scale"]
    g["tv_scale"] = settings["tv_scale"]
    g["range_scale"] = settings["range_scale"]
    g["sat_scale"] = settings["sat_scale"]
    g["cutn_batches"] = settings["cutn_batches"]
    g["init_image"] = settings["init_image"]
    g["init_scale"] = settings["init_scale"]
    g["side_x"] = side_x
    g["side_y"] = side_y
    g["width_height"] = width_height
    g["steps"] = settings["steps"]
    g["diffusion_steps"] = diffusion_steps
    g["diffusion_model"] = settings["diffusion_model"]
    g["diffusion_sampling_mode"] = settings["diffusion_sampling_mode"]
    g["use_secondary_model"] = settings["use_secondary_model"]
    g["seed"] = seed
    g["set_seed"] = settings["set_seed"]
    g["key_frames"] = settings["key_frames"]
    g["max_frames"] = 10000
    g["interp_spline"] = settings["interp_spline"]
    g["angle"] = settings["angle"]
    g["zoom"] = settings["zoom"]
    g["translation_x"] = settings["translation_x"]
    g["translation_y"] = settings["translation_y"]
    g["translation_z"] = settings["translation_z"]
    g["rotation_3d_x"] = settings["rotation_3d_x"]
    g["rotation_3d_y"] = settings["rotation_3d_y"]
    g["rotation_3d_z"] = settings["rotation_3d_z"]
    g["midas_depth_model"] = settings["midas_depth_model"]
    g["midas_weight"] = settings["midas_weight"]
    g["near_plane"] = settings["near_plane"]
    g["far_plane"] = settings["far_plane"]
    g["fov"] = settings["fov"]
    g["padding_mode"] = settings["padding_mode"]
    g["sampling_mode"] = settings["sampling_mode"]
    g["frames_scale"] = settings["frames_scale"]
    g["frames_skip_steps"] = settings["frames_skip_steps"]
    g["cut_overview"] = settings["cut_overview"]
    g["cut_innercut"] = settings["cut_innercut"]
    g["cut_ic_pow"] = settings["cut_ic_pow"]
    g["cut_icgray_p"] = settings["cut_icgray_p"]
    g["intermediate_saves"] = settings["intermediate_saves"]
    g["intermediates_in_subfolder"] = settings["intermediates_in_subfolder"]
    g["use_vertical_symmetry"] = settings["use_vertical_symmetry"]
    g["use_horizontal_symmetry"] = settings["use_horizontal_symmetry"]
    g["transformation_percent"] = settings["transformation_percent"]
    for key in ("video_init_path", "extract_nth_frame",
                "video_init_seed_continuity", "video_init_flow_warp",
                "video_init_flow_blend", "video_init_check_consistency",
                "video_init_blend_mode", "video_init_steps",
                "video_init_clip_guidance_scale", "video_init_tv_scale",
                "video_init_range_scale", "video_init_sat_scale",
                "video_init_cutn_batches", "video_init_skip_steps",
                "video_init_frames_scale", "video_init_frames_skip_steps"):
        g[key] = settings[key]
    # All 24 CLIP flags (save_settings() reads every one).
    for flag in CLIP_FLAGS:
        g[flag] = flag in settings["clip_on_set"]
    # The notebook's do_run() also reads steps_per_checkpoint bare (not
    # just as args.steps_per_checkpoint); mirror the args value here.
    g["steps_per_checkpoint"] = _steps_per_checkpoint(settings)


# ---------------------------------------------------------------------------
# One full generation
# ---------------------------------------------------------------------------

def _next_batch_num(batch_folder, batch_name):
    """The run cell's fresh-run batch numbering, verbatim."""
    batch_num = len(glob(os.path.join(batch_folder, "*.txt")))
    while (os.path.isfile(
            os.path.join(batch_folder,
                         f"{batch_name}({batch_num})_settings.txt"))
           or os.path.isfile(
            os.path.join(batch_folder,
                         f"{batch_name}-{batch_num}_settings.txt"))):
        batch_num += 1
    return batch_num


def run_generation(ui):
    """Run one still-image generation from UI values.

    `ui` is a dict with the app.py field names (see app.py). Returns a
    dict: status lines, seed used, batch folder/number, saved frames,
    settings file path. Raises ValueError on bad input, RuntimeError
    if the engine is not initialised.
    """
    if _STATE["model_path"] is None or _STATE["out_dir"] is None:
        raise RuntimeError("init_paths() must run before run_generation().")

    # Resolve the seed the notebook's way: blank means random.
    seed_text = (ui.get("seed") or "").strip()
    if seed_text == "":
        random.seed()
        seed = random.randint(0, 2 ** 32)
        set_seed = "random_seed"
    else:
        try:
            seed = int(seed_text)
        except ValueError:
            raise ValueError(f"Seed must be a whole number, got: {seed_text!r}")
        set_seed = str(seed)

    # Settings: notebook defaults, overlaid with UI values + recipe fixes.
    settings = dict(NOTEBOOK_DEFAULTS)
    settings["prompt"] = ui["prompt"]
    settings["diffusion_model"] = ui["diffusion_model"]
    settings["use_secondary_model"] = True
    settings["steps"] = int(ui["steps"])
    settings["width_height"] = [int(ui["width"]), int(ui["height"])]
    settings["clip_guidance_scale"] = int(ui["clip_guidance_scale"])
    # Formerly fixed recipe values; now UI fields (defaults = Alan's
    # recipe values, so existing runs are unchanged unless he edits them).
    settings["tv_scale"] = int(ui.get("tv_scale", 200))
    settings["sat_scale"] = int(ui.get("sat_scale", 10110))
    settings["range_scale"] = int(ui.get("range_scale", 150))
    settings["cutn_batches"] = int(ui.get("cutn_batches", 4))
    settings["skip_steps"] = int(ui.get("skip_steps", 10))
    settings["eta"] = float(ui.get("eta", 0.8))
    settings["clamp_max"] = float(ui.get("clamp_max", 0.05))
    settings["cut_overview"] = ui.get("cut_overview", "[12]*400+[4]*600")
    settings["cut_innercut"] = ui.get("cut_innercut", "[4]*400+[12]*600")
    settings["cut_icgray_p"] = ui.get("cut_icgray_p", "[0.2]*400+[0]*600")
    settings["cut_ic_pow"] = f"[{int(ui['cut_ic_pow'])}]*1000"
    settings["use_horizontal_symmetry"] = bool(ui["use_horizontal_symmetry"])
    settings["transformation_percent"] = [float(ui["transformation_percent"])]
    settings["batch_name"] = ui["batch_name"] or "TimeToDisco"
    settings["set_seed"] = set_seed
    clip_on = list(ui["clip_on"])
    settings["clip_on_set"] = set(clip_on)
    settings["clip_on"] = clip_on
    # Model & toggles (defaults = previous behavior).
    settings["use_secondary_model"] = bool(ui.get("use_secondary_model", True))
    settings["custom_path"] = ui.get("custom_path", "")
    settings["use_checkpoint"] = bool(ui.get("use_checkpoint", True))
    settings["diffusion_sampling_mode"] = ui.get("diffusion_sampling_mode", "ddim")
    settings["clamp_grad"] = bool(ui.get("clamp_grad", True))
    settings["fuzzy_prompt"] = bool(ui.get("fuzzy_prompt", False))
    settings["clip_denoised"] = bool(ui.get("clip_denoised", False))
    settings["randomize_class"] = bool(ui.get("randomize_class", True))
    settings["skip_augs"] = bool(ui.get("skip_augs", False))
    settings["perlin_init"] = bool(ui.get("perlin_init", False))
    settings["perlin_mode"] = ui.get("perlin_mode", "mixed")
    settings["rand_mag"] = float(ui.get("rand_mag", 0.05))
    settings["display_rate"] = int(ui.get("display_rate", 20))
    settings["intermediate_saves"] = int(ui.get("intermediate_saves", 0))
    settings["intermediates_in_subfolder"] = bool(
        ui.get("intermediates_in_subfolder", True))
    # Batches = frames per run (do_run loops n_batches); default 1.
    settings["n_batches"] = max(1, int(ui.get("n_batches") or 1))

    # Batch folder + number, run-cell style.
    batch_folder = os.path.join(_STATE["out_dir"], settings["batch_name"])
    os.makedirs(batch_folder, exist_ok=True)
    batch_num = _next_batch_num(batch_folder, settings["batch_name"])

    width_height = settings["width_height"]
    side_x = (width_height[0] // 64) * 64
    side_y = (width_height[1] // 64) * 64
    if side_x != width_height[0] or side_y != width_height[1]:
        # Same warning the notebook prints; sizes must be multiples of 64.
        print(f"Changing output size to {side_x}x{side_y}. "
              f"Dimensions must by multiples of 64.")

    # Model prep (cached; reloads when model/schedule/CLIP key changes).
    steps = settings["steps"]
    timestep_respacing = f"ddim{steps}"
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    load_status = load_models(settings["diffusion_model"], clip_on,
                              use_secondary_model=settings["use_secondary_model"],
                              use_checkpoint=settings["use_checkpoint"])
    diff_status = _prepare_diffusion(settings["diffusion_model"],
                                     timestep_respacing, diffusion_steps,
                                     use_checkpoint=settings["use_checkpoint"],
                                     custom_path=settings["custom_path"])

    # Sync every global, build args, run.
    sync_settings_globals(settings, batch_folder, batch_num, seed,
                          width_height, side_x, side_y, diffusion_steps)
    args, _, _ = build_args(settings, batch_num, seed)
    core.args = args
    core.do_run()

    # Collect the saved frames for this batch (final frames only).
    prefix = f"{settings['batch_name']}({batch_num})_"
    frames = sorted(
        p for p in glob(os.path.join(batch_folder, prefix + "*.png"))
        if "-" not in os.path.basename(p) or p.endswith("_0000.png"))
    # Final still frames are named <batch>_<0000>.png (n_batches == 1);
    # anything with a -NN% or -NNN step suffix is an intermediate.
    finals = [p for p in frames
              if os.path.basename(p) == f"{prefix}0000.png"]
    settings_path = os.path.join(
        batch_folder, f"{settings['batch_name']}({batch_num})_settings.txt")
    return {
        "status": [load_status, diff_status],
        "seed": seed,
        "batch_folder": batch_folder,
        "batch_num": batch_num,
        "frames": finals if finals else frames,
        "settings_path": settings_path,
    }


def progress_path():
    """Where do_run() writes progress.png (the app's working directory)."""
    return os.path.join(os.getcwd(), "progress.png")
