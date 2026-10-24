"""Disco Diffusion Gradio UI -- phase 1 (still images only).

Standalone app: no Jupyter, no Colab. Run it, the browser opens, type a
prompt, hit Generate, pictures appear underneath.

How to run (on Alan's Windows PC, in his existing Python 3.10.6 setup --
NOT a new environment):

    python app.py

from this folder, with the install folder next to it. The app reads
Alan's install at C:\\AI\\Disco\\disco-diffusion (models, code, and
outputs) and never writes to it except the usual outputs: frames and
_settings.txt land in <install>\\images_out\\<batch name>, exactly like
the notebook, and progress.png lands in the install folder while a run
is going (do_run() writes it there; the app copies it for preview).

If the install lives somewhere else, set DISCO_INSTALL_DIR before
launching, e.g.:

    set DISCO_INSTALL_DIR=C:\\AI\\Disco\\disco-diffusion
    python app.py

First launch takes a while: the diffusion model and the six CLIP models
load once, then stay loaded. Changing the diffusion model or the CLIP
checkboxes reloads the affected models (the app form of the notebook's
kernel-restart rule); repeat generations with the same settings reuse
the loaded models.

Defaults are the proven Oct 24 lighthouse run on the stock 512 model:
1280x768, 100 steps, guidance 10000, all six CLIP models on, cut_ic_pow
10, H symmetry OFF. tv_scale (200) and sat_scale (10110) are the
recipe values carried over from the author's setup.
"""

import os
import queue
import random
import re
import shutil
import sys
import threading
import time

# --- install location (read-only for code/models; outputs go to its
# images_out folder, exactly like the notebook) ---
INSTALL_DIR = os.environ.get(
    "DISCO_INSTALL_DIR", r"C:\AI\Disco\disco-diffusion")
UI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UI_DIR)
sys.path.insert(0, INSTALL_DIR)
# The install keeps some dependencies as clone folders whose names differ
# from the package inside them: guided-diffusion/ holds the
# guided_diffusion package, MiDaS/ holds the midas package. The notebook
# imported those because its setup put the clone roots on the path; the
# app must do the same, or the import fails with "No module named
# 'guided_diffusion'" even though the code is sitting in the install.
for _clone in ("guided-diffusion", "MiDaS"):
    _clone_path = os.path.join(INSTALL_DIR, _clone)
    if os.path.isdir(_clone_path):
        sys.path.insert(0, _clone_path)
# The notebook ran with the install folder as its working directory, so
# root_path / model_path / images_out all resolve from there. The app
# does the same thing, deliberately.
os.chdir(INSTALL_DIR)

import gradio as gr

import disco_engine
from disco_engine import ALAN_PORTRAIT_RECIPE

# ---------------------------------------------------------------------------
# UI defaults: Alan's proven portrait recipe
# ---------------------------------------------------------------------------

RECIPE = ALAN_PORTRAIT_RECIPE
# Public release defaults: the stock 512 model. No third-party portrait
# model ships in this repo (see MODELS.md), so the defaults follow the
# proven Oct 24 lighthouse run on the 512 model, not the portrait recipe.
RECIPE = dict(RECIPE)
RECIPE["diffusion_model"] = "512x512_diffusion_uncond_finetune_008100"
RECIPE["width_height"] = [1280, 768]
RECIPE["steps"] = 100
DEFAULT_PROMPT = (
    "A beautiful painting of a singular lighthouse, shining its light "
    "across a tumultuous sea of blood by greg rutkowski and thomas "
    "kinkade, Trending on artstation."
)
CANVAS_CHOICES = ["512x512", "512x768", "768x768", "768x1024",
                  "1280x768", "1664x768"]
CLIP_CHOICES = ["ViTB32", "ViTB16", "ViTL14", "RN101", "RN50", "RN50x4"]
MODEL_CHOICES = [
    "512x512_diffusion_uncond_finetune_008100",
    "256x256_diffusion_uncond",
    "custom",
]
# NOTE: the model dropdown is an addition to the plan's field list, and
# it is required by acceptance test 2: the yellow spaceship batch runs
# on the 512 model, not on v001. Other notebook models (pixelart,
# watercolor, pulp, custom) raise a clear error until phase 2 ports
# their model_config branches.


def _canvas_to_wh(canvas):
    width, height = canvas.split("x")
    return int(width), int(height)


def _snapshot_progress():
    """Copy progress.png (rewritten mid-run by do_run()) to a stable file
    the gallery can read without catching a half-written image."""
    src = disco_engine.progress_path()
    dst = os.path.join(UI_DIR, "progress_preview.png")
    try:
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
            return dst
    except OSError:
        pass
    return None


RANDOM_PROMPTS = [
    "A floating city above the clouds at sunset, golden light, intricate architecture, Greg Rutkowski, trending on artstation",
    "A beautiful female elf warrior with silver hair, ornate armor, forest background, Frank Frazetta, oil painting",
    "A futuristic yellow spaceship speeding toward an exploding planet, deep space, James Gurney, science fiction",
    "A lighthouse on a rocky cliff during a storm, waves crashing, dramatic sky, Thomas Kinkade, Greg Rutkowski",
    "A cyberpunk street market at night in the rain, neon signs, crowds, blade runner atmosphere, detailed",
    "A majestic dragon perched on a castle tower, moonlight, fantasy art, Greg Rutkowski, dramatic",
    "An underwater city with glowing jellyfish and ancient ruins, bioluminescent, ethereal light",
    "A steam-powered airship sailing through golden clouds, brass and copper details, adventure",
    "A lone astronaut standing on an alien planet with two moons, purple sky, science fiction",
    "A haunted Victorian mansion on a hill, twisted trees, fog, full moon, gothic horror",
    "A samurai standing in cherry blossom rain, mountains behind, ukiyo-e style, dramatic",
    "A futuristic robot city at dawn, towering spires, flying vehicles, Syd Mead style",
    "A mermaid queen on a coral throne, deep ocean, shimmering scales, fantasy portrait",
    "A wizard's tower filled with floating books and glowing runes, magical atmosphere",
    "A desert oasis with a giant sandworm in the distance, twin suns, Dune inspired",
    "A cozy cabin in snowy woods, warm light in windows, aurora borealis above",
    "A pirate ship battling a sea serpent in a storm, huge waves, dramatic oil painting",
    "A biomechanical cyborg portrait, intricate machinery fused with skin, symmetrical, dark background",
    "A golden reactor sun rising over a floating citadel, radiant light, epic scale",
    "A skull figure in a Halloween carnival, candy colors, spooky fun, detailed",
]


def _expand_templates(text):
    """Dynamic prompts: each {a|b|c} group becomes one random pick.

    Single-level groups only; unmatched braces are left untouched.
    """

    def _pick(match):
        options = [o.strip() for o in match.group(1).split("|")]
        options = [o for o in options if o]
        return random.choice(options) if options else match.group(0)

    return re.sub(r"\{([^{}]*)\}", _pick, text)


def _prompts_for_run(prompt_text, queue_text, n_batches):
    """Expand the run into (prompt, batches) jobs.

    Queue lines run in order, Batches frames each. A prompt containing
    {a|b} templates expands to one fresh variant per frame (each variant
    its own single-frame run, since do_run() holds one prompt per run).
    """
    lines = [ln.strip() for ln in (queue_text or "").splitlines()
             if ln.strip()]
    if lines:
        jobs = []
        for line in lines:
            if "{" in line and "}" in line:
                jobs.extend([(_expand_templates(line), 1)
                             for _ in range(n_batches)])
            else:
                jobs.append((line, n_batches))
        return jobs
    prompt_text = (prompt_text or "").strip()
    if "{" in prompt_text and "}" in prompt_text:
        return [(_expand_templates(prompt_text), 1)
                for _ in range(n_batches)]
    return [(prompt_text, n_batches)]


def _random_prompt():
    return random.choice(RANDOM_PROMPTS)


def generate(prompt, diffusion_model, steps, guidance, canvas, custom_w,
             custom_h, clip_on,
             seed, batch_name, use_h_symmetry, transformation_percent,
             cut_ic_pow, n_batches, tv_scale, sat_scale, range_scale,
             cutn_batches, skip_steps, eta, clamp_max, cut_overview,
             cut_innercut, cut_icgray_p, use_secondary_model, custom_path,
             use_checkpoint, diffusion_sampling_mode, clamp_grad,
             fuzzy_prompt, clip_denoised, randomize_class, skip_augs,
             perlin_init, perlin_mode, rand_mag, display_rate,
             intermediate_saves, intermediates_in_subfolder, queue_text,
             progress=gr.Progress(track_tqdm=True)):
    """Gradio callback: yields (status, gallery, latest) updates while running.

    do_run() blocks on the GPU, so it runs in a worker thread and this
    generator polls progress.png + the batch folder for updates. The
    progress bar tracks do_run()'s own step counter (tqdm) while it runs.
    """
    if (not prompt or not prompt.strip()) and (
            not queue_text or not queue_text.strip()):
        yield "Type a prompt first.", [], None
        return
    width, height = _canvas_to_wh(canvas)
    # Custom size wins when both boxes are filled (any positive numbers).
    if custom_w and custom_h:
        width, height = int(custom_w), int(custom_h)
    ui = {
        "prompt": prompt.strip(),
        "diffusion_model": diffusion_model,
        "steps": steps,
        "clip_guidance_scale": guidance,
        "width": width,
        "height": height,
        "clip_on": list(clip_on),
        "seed": seed,
        "batch_name": batch_name,
        "use_horizontal_symmetry": use_h_symmetry,
        "transformation_percent": transformation_percent,
        "cut_ic_pow": cut_ic_pow,
        "n_batches": max(1, int(n_batches or 1)),
        "tv_scale": tv_scale,
        "sat_scale": sat_scale,
        "range_scale": range_scale,
        "cutn_batches": cutn_batches,
        "skip_steps": skip_steps,
        "eta": eta,
        "clamp_max": clamp_max,
        "cut_overview": cut_overview,
        "cut_innercut": cut_innercut,
        "cut_icgray_p": cut_icgray_p,
        "use_secondary_model": use_secondary_model,
        "custom_path": custom_path,
        "use_checkpoint": use_checkpoint,
        "diffusion_sampling_mode": diffusion_sampling_mode,
        "clamp_grad": clamp_grad,
        "fuzzy_prompt": fuzzy_prompt,
        "clip_denoised": clip_denoised,
        "randomize_class": randomize_class,
        "skip_augs": skip_augs,
        "perlin_init": perlin_init,
        "perlin_mode": perlin_mode,
        "rand_mag": rand_mag,
        "display_rate": display_rate,
        "intermediate_saves": intermediate_saves,
        "intermediates_in_subfolder": intermediates_in_subfolder,
    }

    errors = queue.Queue()
    result = {}
    progress_info = {}

    # Clear the previous run's preview first: do_run() rewrites
    # progress.png as it goes, but until it does, the old file is still
    # on disk and _snapshot_progress() would serve it as the "new"
    # preview -- that is why the last run's image showed on start.
    for stale in (disco_engine.progress_path(),
                  os.path.join(UI_DIR, "progress_preview.png")):
        try:
            os.remove(stale)
        except OSError:
            pass
    disco_engine.clear_stop()

    def _work():
        try:
            jobs = _prompts_for_run(
                prompt, queue_text, max(1, int(n_batches or 1)))
            all_frames = []
            last = {}
            for idx, (job_prompt, job_batches) in enumerate(jobs):
                if disco_engine.stop_requested():
                    break
                progress_info.update({
                    "current": idx + 1,
                    "total": len(jobs),
                    "prompt": job_prompt,
                })
                ui_job = dict(ui)
                ui_job["prompt"] = job_prompt
                ui_job["n_batches"] = job_batches
                last = disco_engine.run_generation(ui_job)
                all_frames.extend(last.get("frames", []))
            result.update(last)
            result["frames"] = all_frames
        except Exception as exc:  # noqa: BLE001 -- surfaced in the UI
            errors.put(exc)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    status = "Starting... (first run loads the models, please wait)"
    gallery = []
    latest = None
    yield status, gallery, latest
    last_preview = None
    while thread.is_alive():
        preview = _snapshot_progress()
        if preview and preview != last_preview:
            last_preview = preview
            gallery = [preview]
            latest = preview
        if progress_info.get("total"):
            status = (f"Prompt {progress_info['current']}/"
                      f"{progress_info['total']}: "
                      f"{progress_info['prompt'][:80]}")
        yield status, gallery, latest
        time.sleep(2)
    thread.join()
    if not errors.empty():
        exc = errors.get()
        yield f"Error: {exc}", gallery, latest
        return
    status_lines = result.get("status", [])
    frames = result.get("frames", [])
    status = (
        f"Done. Seed {result.get('seed')}, "
        f"batch {batch_name}({result.get('batch_num')}). "
        f"Models: {'; '.join(status_lines)}. "
        f"Settings saved to {result.get('settings_path')}."
    )
    yield status, frames, (frames[-1] if frames else latest)


def build_ui():
    with gr.Blocks(title="Disco Diffusion") as demo:
        gr.Markdown("# Disco Diffusion\n"
                    "Still images only (phase 1). No Jupyter, no Colab.")
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=3)
                queue_text = gr.Textbox(
                    label="Prompt queue (one per line — runs in order, "
                          "Batches frames each; when filled, the single "
                          "prompt above is ignored)",
                    lines=4)
                diffusion_model = gr.Dropdown(
                    label="Diffusion model",
                    choices=MODEL_CHOICES,
                    value=RECIPE["diffusion_model"])
                with gr.Row():
                    steps = gr.Slider(
                        label="Steps", minimum=25, maximum=1000,
                        step=25, value=RECIPE["steps"])
                    guidance = gr.Slider(
                        label="Guidance (clip_guidance_scale)",
                        minimum=1000, maximum=100000, step=1000,
                        value=RECIPE["clip_guidance_scale"])
                canvas = gr.Radio(
                    label="Canvas (width x height)",
                    choices=CANVAS_CHOICES,
                    value=f"{RECIPE['width_height'][0]}"
                           f"x{RECIPE['width_height'][1]}")
                with gr.Row():
                    custom_w = gr.Number(
                        label="Custom width (optional, overrides canvas)",
                        value=None, precision=0)
                    custom_h = gr.Number(
                        label="Custom height (optional, overrides canvas)",
                        value=None, precision=0)
                clip_on = gr.CheckboxGroup(
                    label="CLIP models",
                    choices=CLIP_CHOICES,
                    value=RECIPE["clip_on"])
                with gr.Row():
                    seed = gr.Textbox(
                        label="Seed (blank = random)",
                        value="")
                    batch_name = gr.Textbox(
                        label="Batch name",
                        value="TimeToDisco")
                with gr.Row():
                    n_batches = gr.Number(
                        label="Batches (frames per run)",
                        value=1, precision=0)
                with gr.Accordion("More settings (tv, sat, range, cuts)",
                                  open=False):
                    with gr.Row():
                        tv_scale = gr.Number(
                            label="tv_scale", value=200, precision=0)
                        sat_scale = gr.Number(
                            label="sat_scale", value=10110, precision=0)
                        range_scale = gr.Number(
                            label="range_scale", value=150, precision=0)
                    with gr.Row():
                        cutn_batches = gr.Number(
                            label="cutn_batches", value=4, precision=0)
                        skip_steps = gr.Number(
                            label="skip_steps", value=10, precision=0)
                        eta = gr.Number(
                            label="eta", value=0.8)
                        clamp_max = gr.Number(
                            label="clamp_max", value=0.05)
                    with gr.Row():
                        cut_overview = gr.Textbox(
                            label="cut_overview",
                            value="[12]*400+[4]*600")
                        cut_innercut = gr.Textbox(
                            label="cut_innercut",
                            value="[4]*400+[12]*600")
                        cut_icgray_p = gr.Textbox(
                            label="cut_icgray_p",
                            value="[0.2]*400+[0]*600")
                with gr.Accordion("Model & toggles", open=False):
                    with gr.Row():
                        use_secondary_model = gr.Checkbox(
                            label="Use secondary model",
                            value=True)
                        use_checkpoint = gr.Checkbox(
                            label="Use checkpoint (use_checkpoint)",
                            value=True)
                        diffusion_sampling_mode = gr.Dropdown(
                            label="Sampling mode",
                            choices=["ddim", "plms"],
                            value="ddim")
                    custom_path = gr.Textbox(
                        label="Custom model path (only when model = custom)",
                        value="")
                    with gr.Row():
                        clamp_grad = gr.Checkbox(
                            label="clamp_grad", value=True)
                        fuzzy_prompt = gr.Checkbox(
                            label="fuzzy_prompt", value=False)
                        clip_denoised = gr.Checkbox(
                            label="clip_denoised", value=False)
                        randomize_class = gr.Checkbox(
                            label="randomize_class", value=True)
                        skip_augs = gr.Checkbox(
                            label="skip_augs", value=False)
                    with gr.Row():
                        perlin_init = gr.Checkbox(
                            label="perlin_init", value=False)
                        perlin_mode = gr.Dropdown(
                            label="perlin_mode",
                            choices=["mixed", "gray", "color"],
                            value="mixed")
                        rand_mag = gr.Number(
                            label="rand_mag", value=0.05)
                    with gr.Row():
                        display_rate = gr.Number(
                            label="display_rate", value=20, precision=0)
                        intermediate_saves = gr.Number(
                            label="intermediate_saves", value=0,
                            precision=0)
                        intermediates_in_subfolder = gr.Checkbox(
                            label="intermediates_in_subfolder",
                            value=True)
                with gr.Row():
                    use_h_symmetry = gr.Checkbox(
                        label="H symmetry (use_horizontal_symmetry)",
                        value=False)
                    transformation_percent = gr.Slider(
                        label="Transformation percent",
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=RECIPE["transformation_percent"][0])
                    cut_ic_pow = gr.Number(
                        label="cut_ic_pow",
                        value=RECIPE["cut_ic_pow_number"],
                        precision=0)
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    stop_btn = gr.Button("Stop")
                    random_btn = gr.Button("Random prompt")
            with gr.Column(scale=3):
                status = gr.Textbox(label="Status", interactive=False)
                latest = gr.Image(label="Latest frame (full, uncropped)")
                gallery = gr.Gallery(label="Results")
        generate_btn.click(
            generate,
            inputs=[prompt, diffusion_model, steps, guidance, canvas,
                    custom_w, custom_h,
                    clip_on, seed, batch_name, use_h_symmetry,
                    transformation_percent, cut_ic_pow, n_batches,
                    tv_scale, sat_scale, range_scale, cutn_batches,
                    skip_steps, eta, clamp_max, cut_overview,
                    cut_innercut, cut_icgray_p, use_secondary_model,
                    custom_path, use_checkpoint, diffusion_sampling_mode,
                    clamp_grad, fuzzy_prompt, clip_denoised,
                    randomize_class, skip_augs, perlin_init, perlin_mode,
                    rand_mag, display_rate, intermediate_saves,
                    intermediates_in_subfolder, queue_text],
            outputs=[status, gallery, latest])
        stop_btn.click(disco_engine.request_stop, inputs=[], outputs=[])
        random_btn.click(_random_prompt, inputs=[], outputs=[prompt])
    return demo


if __name__ == "__main__":
    disco_engine.init_paths(INSTALL_DIR)
    # allowed_paths: the progress preview snapshot lives in this folder
    # (UI_DIR), but the app chdirs to the install folder, so Gradio
    # would otherwise refuse to serve it (InvalidPathError) even though
    # the generation itself runs fine. Final frames come from the
    # install's images_out folder, which is already allowed.
    build_ui().launch(
        server_name="127.0.0.1", inbrowser=True,
        allowed_paths=[UI_DIR])
