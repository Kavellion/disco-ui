"""Disco Diffusion Gradio UI.

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

Defaults are Alan's proven portrait recipe (Oct 24, 2026): v001 model,
512x768, 300 steps, guidance 10000, all six CLIP models on, cut_ic_pow
10, H symmetry ON at 0.09. tv_scale (200) and sat_scale (10110) are
fixed at his recipe values in phase 1 -- they are not UI fields yet.

The Prompt Storage tab is a notebook for favorite prompts (one
prompt per line), kept in prompt_storage.txt next to app.py, so
the prompts are still there when the app is reopened.

The improvements round (Oct 31, 2026) adds the rest of the
family features to this app, all in this one file:

- Saved setups: the whole Generate recipe under a name, kept
  in setups.json next to app.py.
- Seed variations: the Generate settings frozen, the seed
  walked forward -- one frame per seed.
- A type-in box beside the Steps slider (the Batches box was
  already a type-in number).
- An Animation tab: Disco's own 2D/3D animation (the frame
  chain / latent walk from the notebook), driving the engine's
  do_run() in animation mode, frames in the batch folder, and
  the movie assembled in the page.
- An Upscale tab: 2x/4x enlargement (PIL Lanczos) of any saved
  frame, written to an upscaled folder next to app.py.
"""

import json
import os
import queue
import random
import re
import shutil
import subprocess
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
DEFAULT_PROMPT = (
    "a face portrait of a beautiful female elf, symmetrical, "
    "filigree dress, Frank Frazetta, Manuel Sanjulian"
)
CANVAS_CHOICES = ["512x512", "512x768", "768x768", "768x1024",
                  "1280x768", "1664x768"]
CLIP_CHOICES = ["ViTB32", "ViTB16", "ViTL14", "RN101", "RN50", "RN50x4"]
MODEL_CHOICES = [
    "portrait_generator_v001",
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


RANDOM_PROMPTS_FILE = os.path.join(UI_DIR, "random_prompts.txt")
PROMPT_STORAGE_FILE = os.path.join(UI_DIR, "prompt_storage.txt")
SETUPS_FILE = os.path.join(UI_DIR, "setups.json")
UPSCALE_DIR = os.path.join(UI_DIR, "upscaled")


def _random_prompt():
    """Pick from random_prompts.txt (one prompt per line) when it
    exists and is non-empty, so the list is editable without touching
    code; otherwise fall back to the built-in list. The file is read
    on every click, so edits apply immediately with no restart."""
    try:
        with open(RANDOM_PROMPTS_FILE, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return random.choice(lines)
    except OSError:
        pass
    return random.choice(RANDOM_PROMPTS)


# ---------------------------------------------------------------------------
# Prompt Storage -- a notebook tab for favorite prompts. The
# notebook is one plain-text box in its own tab, labelled
# "Prompt Storage", holding one prompt per line. It is kept in
# prompt_storage.txt next to app.py, so the prompts are still
# there when the app is reopened -- that persistence is the
# whole point. Save writes the whole notebook; "Add current
# prompt" appends the Prompt box's text (and saves at once);
# the dropdown lists the stored lines and "Use this prompt"
# loads the picked one back into the Prompt box. A missing or
# unreadable file simply means "no prompts stored yet"; it
# never stops the app.
# ---------------------------------------------------------------------------


def _load_prompt_storage():
    """The stored notebook text. A missing or unreadable
    prompt_storage.txt means an empty notebook."""
    try:
        with open(PROMPT_STORAGE_FILE, "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _prompt_storage_lines(text=None):
    """The stored prompts: the notebook's non-empty lines,
    stripped, in file order, duplicates kept out (first copy
    wins). Reads the file when no text is passed."""
    if text is None:
        text = _load_prompt_storage()
    lines = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line and line not in lines:
            lines.append(line)
    return lines


def _write_prompt_storage(text):
    """Write the notebook text to prompt_storage.txt, ending
    with a single trailing newline when it has content.
    Raises OSError when the write fails; the handlers turn
    that into a plain status line."""
    text = (text or "").strip("\n")
    if text:
        text += "\n"
    with open(PROMPT_STORAGE_FILE, "w", encoding="utf-8") as handle:
        handle.write(text)


def _prompt_storage_choices(text=None):
    """The stored-prompts dropdown choices."""
    return _prompt_storage_lines(text)


def _save_prompt_storage(text):
    """Save the notebook box exactly as typed. Returns
    (status, stored-prompts dropdown update)."""
    import gradio as gr
    try:
        _write_prompt_storage(text)
    except OSError as exc:  # noqa: BLE001 -- plain status text
        return ("Could not save Prompt Storage: {}".format(exc),
                gr.update())
    count = len(_prompt_storage_lines(text))
    if count == 1:
        message = "Prompt Storage saved -- 1 prompt stored."
    else:
        message = "Prompt Storage saved -- {} prompts stored.".format(count)
    return message, gr.update(choices=_prompt_storage_lines(text),
                              value=None)


def _add_current_prompt(current, text):
    """Append the Prompt box's current text to the notebook
    and save at once. Returns (status, notebook update,
    stored-prompts dropdown update). An empty prompt box, or
    a prompt already stored, is a plain status and nothing
    is written."""
    import gradio as gr
    current = (current or "").strip()
    if not current:
        return ("The Prompt box is empty -- nothing to add.",
                gr.update(), gr.update())
    if "\n" in current:
        return ("That prompt has line breaks in it -- Prompt "
                "Storage keeps one prompt per line, so it was "
                "not added.", gr.update(), gr.update())
    lines = _prompt_storage_lines(text)
    if current in lines:
        return ("That prompt is already in Prompt Storage.",
                gr.update(), gr.update())
    lines.append(current)
    new_text = "\n".join(lines) + "\n"
    try:
        _write_prompt_storage(new_text)
    except OSError as exc:  # noqa: BLE001 -- plain status text
        return ("Could not save Prompt Storage: {}".format(exc),
                gr.update(), gr.update())
    return ("Added to Prompt Storage -- {} prompts stored now."
            .format(len(lines)),
            gr.update(value=new_text),
            gr.update(choices=lines, value=None))


def _use_stored_prompt(choice):
    """Load one stored prompt into the Prompt box. Returns a
    single update for the Prompt box; no pick leaves it
    untouched."""
    import gradio as gr
    choice = (choice or "").strip()
    if not choice:
        return gr.update()
    return gr.update(value=choice)


# ---------------------------------------------------------------------------
# Saved setups -- the whole Generate recipe under a name, kept
# in setups.json next to app.py (stdlib json only). Save reads
# every Generate control; Load writes them all back; Delete
# removes one. The prompt queue is NOT part of a setup (it is a
# per-run choice, not recipe -- same call as the SD 3.5 build).
# A missing, unreadable, corrupt, or non-object setups.json
# means "no setups yet"; the app never crashes over this file.
# ---------------------------------------------------------------------------

# The recipe fields, in generate()'s argument order (the queue
# text is the one generate() input left out).
_SETUP_FIELDS = [
    "prompt", "diffusion_model", "steps", "guidance", "canvas",
    "custom_w", "custom_h", "clip_on", "seed", "batch_name",
    "use_h_symmetry", "transformation_percent", "cut_ic_pow",
    "n_batches", "tv_scale", "sat_scale", "range_scale",
    "cutn_batches", "skip_steps", "eta", "clamp_max",
    "cut_overview", "cut_innercut", "cut_icgray_p",
    "use_secondary_model", "custom_path", "use_checkpoint",
    "diffusion_sampling_mode", "clamp_grad", "fuzzy_prompt",
    "clip_denoised", "randomize_class", "skip_augs",
    "perlin_init", "perlin_mode", "rand_mag", "display_rate",
    "intermediate_saves", "intermediates_in_subfolder",
]

# What a field falls back to when a saved setup does not carry
# it (a hand-edited file): the controls' own starting values.
_SETUP_DEFAULTS = {
    "prompt": DEFAULT_PROMPT,
    "diffusion_model": RECIPE["diffusion_model"],
    "steps": RECIPE["steps"],
    "guidance": RECIPE["clip_guidance_scale"],
    "canvas": "{}x{}".format(RECIPE["width_height"][0],
                             RECIPE["width_height"][1]),
    "custom_w": None,
    "custom_h": None,
    "clip_on": list(RECIPE["clip_on"]),
    "seed": "",
    "batch_name": "TimeToDisco",
    "use_h_symmetry": True,
    "transformation_percent": RECIPE["transformation_percent"][0],
    "cut_ic_pow": RECIPE["cut_ic_pow_number"],
    "n_batches": 1,
    "tv_scale": 200,
    "sat_scale": 10110,
    "range_scale": 150,
    "cutn_batches": 4,
    "skip_steps": 10,
    "eta": 0.8,
    "clamp_max": 0.05,
    "cut_overview": "[12]*400+[4]*600",
    "cut_innercut": "[4]*400+[12]*600",
    "cut_icgray_p": "[0.2]*400+[0]*600",
    "use_secondary_model": True,
    "custom_path": "",
    "use_checkpoint": True,
    "diffusion_sampling_mode": "ddim",
    "clamp_grad": True,
    "fuzzy_prompt": False,
    "clip_denoised": False,
    "randomize_class": True,
    "skip_augs": False,
    "perlin_init": False,
    "perlin_mode": "mixed",
    "rand_mag": 0.05,
    "display_rate": 20,
    "intermediate_saves": 0,
    "intermediates_in_subfolder": True,
}


def _load_setups():
    """All saved setups, as a {name: recipe} dict. A missing
    setups.json, an unreadable or corrupt file, or JSON that
    is not an object all mean "no setups yet": an empty dict,
    never an exception. Entries whose value is not an object
    are skipped."""
    try:
        with open(SETUPS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(name): recipe for name, recipe in data.items()
            if isinstance(recipe, dict)}


def _write_setups(setups):
    """Write the setups dict to setups.json (sorted keys and a
    trailing newline, so the file stays readable by hand).
    Raises OSError when the write fails; the Save/Delete
    handlers turn that into a plain status line."""
    with open(SETUPS_FILE, "w", encoding="utf-8") as handle:
        json.dump(setups, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _setup_choices():
    """The saved-setups dropdown choices: the setup names,
    sorted."""
    return sorted(_load_setups())


def _save_setup(name, *values):
    """Save the current Generate controls as a named setup.
    `values` are the controls in _SETUP_FIELDS order. Returns
    (status, setups dropdown update). An empty name is refused
    with a plain status and nothing is written. Saving over
    an existing name replaces that setup, and the status
    says so."""
    import gradio as gr
    name = (name or "").strip()
    if not name:
        return ("Type a name in the setup name box first -- "
                "nothing was saved.", gr.update())
    setups = _load_setups()
    replaced = name in setups
    recipe = {}
    for field, value in zip(_SETUP_FIELDS, values):
        if field == "clip_on":
            value = list(value or [])
        recipe[field] = value
    setups[name] = recipe
    try:
        _write_setups(setups)
    except OSError as exc:  # noqa: BLE001 -- plain status text
        return ("Could not save setup '{}': {}".format(name, exc),
                gr.update())
    if replaced:
        message = ("Setup '{}' saved -- it replaced the old "
                   "setup with that name.".format(name))
    else:
        message = "Setup '{}' saved.".format(name)
    return message, gr.update(choices=sorted(setups), value=name)


def _setup_load_fallback(message):
    """A load that cannot run: every recipe control left
    untouched (empty updates), the reason as the status."""
    import gradio as gr
    return tuple(gr.update() for _ in _SETUP_FIELDS) + (message,)


def _load_setup(name):
    """Load one saved setup into every Generate control.
    Returns one update per field in _SETUP_FIELDS order, plus
    a status line. Fields a setup does not have fall back to
    _SETUP_DEFAULTS; a model / canvas / sampling / perlin
    choice the app no longer offers falls back the same way;
    CLIP picks are filtered to the models the app offers."""
    import gradio as gr
    name = (name or "").strip()
    if not name:
        return _setup_load_fallback(
            "Pick a saved setup to load first.")
    setups = _load_setups()
    if name not in setups:
        return _setup_load_fallback(
            "No setup named '{}' is saved.".format(name))
    recipe = dict(_SETUP_DEFAULTS)
    saved = setups[name]
    for field in _SETUP_FIELDS:
        if field in saved:
            recipe[field] = saved[field]
    if recipe["diffusion_model"] not in MODEL_CHOICES:
        recipe["diffusion_model"] = _SETUP_DEFAULTS["diffusion_model"]
    if recipe["canvas"] not in CANVAS_CHOICES:
        recipe["canvas"] = _SETUP_DEFAULTS["canvas"]
    if recipe["diffusion_sampling_mode"] not in ("ddim", "plms"):
        recipe["diffusion_sampling_mode"] = \
            _SETUP_DEFAULTS["diffusion_sampling_mode"]
    if recipe["perlin_mode"] not in ("mixed", "gray", "color"):
        recipe["perlin_mode"] = _SETUP_DEFAULTS["perlin_mode"]
    if isinstance(recipe["clip_on"], (list, tuple)):
        recipe["clip_on"] = [pick for pick in recipe["clip_on"]
                             if pick in CLIP_CHOICES]
    else:
        recipe["clip_on"] = list(_SETUP_DEFAULTS["clip_on"])
    updates = tuple(gr.update(value=recipe[field])
                    for field in _SETUP_FIELDS)
    return updates + ("Setup '{}' loaded.".format(name),)


def _delete_setup(name):
    """Delete one saved setup from setups.json. Returns
    (status, setups dropdown update). No pick, or a name that
    is not saved, is a plain status and nothing is written."""
    import gradio as gr
    name = (name or "").strip()
    setups = _load_setups()
    if not name or name not in setups:
        return ("Pick a saved setup to delete first.", gr.update())
    del setups[name]
    try:
        _write_setups(setups)
    except OSError as exc:  # noqa: BLE001 -- plain status text
        return ("Could not delete setup '{}': {}".format(
            name, exc), gr.update())
    return ("Setup '{}' deleted.".format(name),
            gr.update(choices=sorted(setups), value=None))


# ---------------------------------------------------------------------------
# Steps type-in box -- the Steps slider gets a number box
# beside it, two-way synced: the slider updates the box live,
# and typing a number and pressing Enter moves the slider
# (clamped to the slider's range and snapped to its step).
# The Batches box was already a type-in number in this app.
# ---------------------------------------------------------------------------


def _steps_echo(value):
    """Slider -> type-in box: the slider's value as a whole
    number (None passes through as None)."""
    if value is None:
        return None
    return int(value)


def _steps_typein(value):
    """Type-in box -> slider update. The typed value is
    clamped to the slider's 25..1000 range and snapped to
    its 25 step. A blank or non-numeric box leaves the
    slider untouched."""
    import gradio as gr
    if value is None:
        return gr.update()
    try:
        typed = float(value)
    except (TypeError, ValueError):
        return gr.update()
    typed = max(25.0, min(1000.0, typed))
    snapped = int(round(typed / 25.0)) * 25
    snapped = max(25, min(1000, snapped))
    return gr.update(value=snapped)


# ---------------------------------------------------------------------------
# Animation -- Disco's own 2D/3D animation, the notebook's
# frame chain: frame 1 diffuses in full, every later frame
# starts from the previous frame (warped by the movement
# keyframes) and only diffuses the remaining steps
# (frames_skip_steps). The stills engine wrapper deliberately
# forces animation_mode "None"; this section builds the
# settings the way run_generation() does, then sets the
# animation fields the wrapper leaves off -- args.animation_
# mode / max_frames / start_frame and the matching engine
# globals -- and calls the engine's do_run() directly. That
# is the notebook's own run, driven from the tab.
#
# Defaults are Alan's proven animation recipe (his refresher
# + the drift-fix run, Oct 2026): 512x512 model, 512x512,
# 150 steps, guidance LOW (1000 -- high guidance flickers),
# tv 0, sat 300, cutn_batches 1, frames_skip_steps 60%,
# 360 frames at 12 fps, gentle constant drift (translation_z
# 0.2, zoom 1.0). 2D first; turbo is 3D only.
# ---------------------------------------------------------------------------

ANIM_DEFAULT_PROMPT = (
    "A Post apocalyptic alien electric robot stunning dense "
    "city by john atkinson grimshaw, dramatic, highly detailed, "
    "vibrant"
)
ANIM_CLIP_DEFAULT = ["ViTB32", "ViTB16", "RN50"]


def _parse_prompt_keyframes(text, base_prompt):
    """The prompt-keyframes box -> the notebook's text_prompts
    dict {frame: [prompt]}. One line per keyframe, as
    "frame: prompt". Blank box (or no line for frame 0) means
    frame 0 carries the Prompt box's text. A line that is not
    "a whole number, a colon, a prompt" raises ValueError
    with a plain message. A repeated frame: the last line
    wins, like a dict."""
    prompts = {0: [(base_prompt or "").strip()]}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        head, sep, tail = line.partition(":")
        if not sep:
            raise ValueError(
                "Prompt keyframe line is not 'frame: prompt': "
                "{!r}".format(line))
        try:
            frame = int(head.strip())
        except ValueError:
            raise ValueError(
                "Prompt keyframe frame is not a whole number: "
                "{!r}".format(line))
        if frame < 0:
            raise ValueError(
                "Prompt keyframe frame cannot be negative: "
                "{!r}".format(line))
        prompt = tail.strip()
        if not prompt:
            raise ValueError(
                "Prompt keyframe has no prompt after the colon: "
                "{!r}".format(line))
        prompts[frame] = [prompt]
    return prompts


def _anim_frames_in(batch_folder, batch_name, batch_num):
    """This run's saved frames in order: the files do_run()
    writes as <batch name>(<batchNum>)_<frame:04>.png."""
    pattern = re.compile(
        r"^{}\({}\)_(\d{{4}})\.png$".format(
            re.escape(batch_name), int(batch_num)))
    found = []
    try:
        names = os.listdir(batch_folder)
    except OSError:
        return []
    for name in names:
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)),
                          os.path.join(batch_folder, name)))
    found.sort()
    return [path for _, path in found]


def _anim_plan(batch_folder, batch_name, resume):
    """Where an animation run starts: (batch_num, start_frame).

    Fresh run: the run cell's next batch number, frame 0.
    Resume: the batch number in this folder holding the most
    frames, starting after its last unbroken frame -- the
    notebook's resume_run rule (same settings, pick up where
    it stopped). Resume with nothing to resume raises
    ValueError with a plain message."""
    if not resume:
        return disco_engine._next_batch_num(batch_folder,
                                            batch_name), 0
    pattern = re.compile(
        r"^{}\((\d+)\)_(\d{{4}})\.png$".format(
            re.escape(batch_name)))
    by_batch = {}
    try:
        names = os.listdir(batch_folder)
    except OSError:
        names = []
    for name in names:
        match = pattern.match(name)
        if match:
            by_batch.setdefault(int(match.group(1)),
                                set()).add(int(match.group(2)))
    best = None
    for batch_num, frames in by_batch.items():
        start = 0
        while start in frames:
            start += 1
        if start == 0:
            continue
        if best is None or start > best[1] or (
                start == best[1] and batch_num > best[0]):
            best = (batch_num, start)
    if best is None:
        raise ValueError(
            "Nothing to resume in the '{}' batch folder -- no "
            "finished frames yet.".format(batch_name))
    return best


def _anim_settings(ui_anim):
    """The notebook settings dict for one animation run:
    NOTEBOOK_DEFAULTS overlaid with the Animation tab's
    values, mirroring how run_generation() overlays the
    stills values. Raises ValueError on a bad movement /
    skip-steps value, with a plain message."""
    settings = dict(disco_engine.NOTEBOOK_DEFAULTS)
    settings["prompt"] = ui_anim["prompt"]
    settings["diffusion_model"] = ui_anim["diffusion_model"]
    settings["use_secondary_model"] = True
    settings["steps"] = int(ui_anim["steps"])
    settings["width_height"] = [int(ui_anim["width"]),
                                int(ui_anim["height"])]
    settings["clip_guidance_scale"] = int(ui_anim["guidance"])
    settings["tv_scale"] = int(ui_anim["tv_scale"])
    settings["sat_scale"] = int(ui_anim["sat_scale"])
    settings["range_scale"] = int(ui_anim["range_scale"])
    settings["cutn_batches"] = int(ui_anim["cutn_batches"])
    settings["skip_steps"] = 10
    settings["eta"] = float(ui_anim["eta"])
    settings["clamp_max"] = float(ui_anim["clamp_max"])
    # Alan's proven animation cut schedules (run #2, verbatim).
    settings["cut_overview"] = "[12]*400+[4]*600"
    settings["cut_innercut"] = "[4]*400+[12]*600"
    settings["cut_icgray_p"] = "[0.2]*400+[0]*600"
    settings["cut_ic_pow"] = "[1]*1000"
    settings["use_horizontal_symmetry"] = False
    settings["transformation_percent"] = [0.09]
    settings["batch_name"] = ui_anim["batch_name"] or "DiscoAnim"
    settings["set_seed"] = ui_anim["set_seed"]
    clip_on = list(ui_anim["clip_on"])
    settings["clip_on_set"] = set(clip_on)
    settings["clip_on"] = clip_on
    settings["custom_path"] = ""
    settings["use_checkpoint"] = True
    settings["diffusion_sampling_mode"] = "ddim"
    settings["clamp_grad"] = True
    settings["fuzzy_prompt"] = False
    settings["clip_denoised"] = False
    settings["randomize_class"] = True
    settings["skip_augs"] = False
    settings["perlin_init"] = False
    settings["perlin_mode"] = "mixed"
    settings["rand_mag"] = 0.05
    settings["display_rate"] = 20
    settings["intermediate_saves"] = 0
    settings["intermediates_in_subfolder"] = True
    settings["n_batches"] = 1
    settings["batch_size"] = 1
    settings["init_image"] = None
    settings["init_scale"] = 1000
    settings["frames_scale"] = int(ui_anim["frames_scale"])
    skip_pct = int(ui_anim["frames_skip_pct"])
    if skip_pct < 0 or skip_pct > 99:
        raise ValueError(
            "frames_skip_steps must be between 0% and 99%, "
            "got {}%.".format(skip_pct))
    settings["frames_skip_steps"] = "{}%".format(skip_pct)
    # Movement keyframe strings (Disco syntax "0: (value)"),
    # each falling back to its recipe default when left blank.
    settings["angle"] = ui_anim["angle"] or "0: (0)"
    settings["zoom"] = ui_anim["zoom"] or "0: (1.0)"
    settings["translation_x"] = ui_anim["translation_x"] or "0: (0)"
    settings["translation_y"] = ui_anim["translation_y"] or "0: (0)"
    settings["translation_z"] = ui_anim["translation_z"] or "0: (0.2)"
    settings["rotation_3d_x"] = ui_anim["rotation_3d_x"] or "0: (0)"
    settings["rotation_3d_y"] = ui_anim["rotation_3d_y"] or "0: (0)"
    settings["rotation_3d_z"] = ui_anim["rotation_3d_z"] or "0: (0)"
    settings["turbo_mode"] = bool(ui_anim["turbo_mode"])
    settings["turbo_steps"] = str(int(ui_anim["turbo_steps"]))
    settings["turbo_preroll"] = int(ui_anim["turbo_preroll"])
    settings["key_frames"] = True
    settings["interp_spline"] = "Linear"
    return settings


def _run_animation_engine(settings, batch_num, seed, text_prompts,
                          mode, max_frames, start_frame, resume):
    """One full animation run on the engine: model prep and
    global sync exactly as run_generation() does them, then
    the animation fields set on args + the engine globals,
    then do_run(). Returns the same result dict shape as
    run_generation()."""
    if (disco_engine._STATE["model_path"] is None
            or disco_engine._STATE["out_dir"] is None):
        raise RuntimeError(
            "init_paths() must run before animation.")
    batch_folder = os.path.join(
        disco_engine._STATE["out_dir"], settings["batch_name"])
    os.makedirs(batch_folder, exist_ok=True)
    width_height = settings["width_height"]
    side_x = (width_height[0] // 64) * 64
    side_y = (width_height[1] // 64) * 64
    steps = settings["steps"]
    timestep_respacing = "ddim{}".format(steps)
    diffusion_steps = (1000 // steps) * steps if steps < 1000 \
        else steps
    load_status = disco_engine.load_models(
        settings["diffusion_model"], settings["clip_on"],
        use_secondary_model=settings["use_secondary_model"],
        use_checkpoint=settings["use_checkpoint"])
    diff_status = disco_engine._prepare_diffusion(
        settings["diffusion_model"], timestep_respacing,
        diffusion_steps,
        use_checkpoint=settings["use_checkpoint"],
        custom_path=settings["custom_path"])
    disco_engine.sync_settings_globals(
        settings, batch_folder, batch_num, seed, width_height,
        side_x, side_y, diffusion_steps)
    args, _, _ = disco_engine.build_args(settings, batch_num, seed)
    core = disco_engine.core
    # The prompt series is built while the engine's max_frames
    # global is still the notebook's series length (10000);
    # only afterwards does the global take the run's real
    # frame count, so save_settings() writes the honest one.
    args.prompts_series = core.split_prompts(text_prompts)
    args.animation_mode = mode
    args.max_frames = int(max_frames)
    args.start_frame = int(start_frame)
    core.args = args
    core.text_prompts = text_prompts
    core.animation_mode = mode
    core.max_frames = int(max_frames)
    core.start_frame = int(start_frame)
    core.resume_run = bool(resume)
    core.do_run()
    frames = _anim_frames_in(batch_folder, settings["batch_name"],
                             batch_num)
    settings_path = os.path.join(
        batch_folder, "{}({})_settings.txt".format(
            settings["batch_name"], batch_num))
    return {
        "status": [load_status, diff_status],
        "seed": seed,
        "batch_folder": batch_folder,
        "batch_num": batch_num,
        "frames": frames,
        "settings_path": settings_path,
    }


def _find_ffmpeg():
    """Locate an ffmpeg executable, best source first:

    1. a bundled copy at tools/ffmpeg.exe next to app.py;
    2. the static ffmpeg binary the imageio-ffmpeg package
       carries, when that package is installed;
    3. whatever shutil.which finds on PATH.

    None when all three come up empty -- the caller then
    keeps the frames and says so plainly. (Same search as
    the SD-family Animation tabs.)"""
    bundled = os.path.join(UI_DIR, "tools", "ffmpeg.exe")
    if os.path.isfile(bundled):
        return bundled
    try:
        import imageio_ffmpeg
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and os.path.isfile(candidate):
            return candidate
    except Exception:  # noqa: BLE001 -- package missing or
        pass               # broken: fall through to PATH
    return shutil.which("ffmpeg")


def _fps_text(value):
    """'12' rather than '12.0' for ffmpeg arguments."""
    return ("%f" % float(value)).rstrip("0").rstrip(".")


def _anim_video_command(ffmpeg, batch_folder, batch_name,
                        batch_num, first_frame, fps_value,
                        out_path):
    """The ffmpeg command for the movie, as an argument list
    (never a shell string): this run's frames in the batch
    folder, aimed by -start_number at its first frame."""
    pattern = os.path.join(
        batch_folder, "{}({})_%04d.png".format(batch_name,
                                               int(batch_num)))
    fps = _fps_text(fps_value)
    return ([ffmpeg, "-nostdin", "-y", "-r", fps,
             "-start_number", str(int(first_frame)),
             "-i", pattern, "-c:v", "libx264", "-vf",
             "fps=" + fps, "-pix_fmt", "yuv420p", "-crf", "17",
             "-preset", "veryfast", out_path])


def _assemble_movie(frames, fps_value, batch_folder, batch_name,
                    batch_num, want_mp4, want_gif):
    """Assemble the run's movie(s) in the batch folder.
    Returns (movie_path, notes): movie_path is the MP4 when
    one was made (the file the in-page player gets), else
    None; notes are plain status fragments. The GIF is made
    with PIL (always available -- the engine needs it too);
    the MP4 needs ffmpeg, and its absence is a note, never
    an error: the frames are the run's real output."""
    notes = []
    made = []
    if not frames:
        return None, notes
    first_frame = 0
    match = re.search(r"_(\d{4})\.png$", frames[0])
    if match:
        first_frame = int(match.group(1))
    if want_gif:
        gif_path = os.path.join(
            batch_folder, "{}({})_movie.gif".format(
                batch_name, int(batch_num)))
        try:
            from PIL import Image
            images = [Image.open(path).convert("RGB")
                      for path in frames]
            duration = max(1, int(round(1000.0 / float(fps_value))))
            images[0].save(
                gif_path, save_all=True, append_images=images[1:],
                duration=duration, loop=0)
            made.append(gif_path)
        except Exception as exc:  # noqa: BLE001 -- plain note
            notes.append("GIF assembly failed: {}".format(exc))
    movie = None
    if want_mp4:
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            notes.append(
                "no ffmpeg could be found (no bundled copy in "
                "tools/, the imageio-ffmpeg package unavailable, "
                "and nothing on PATH), so no MP4 was assembled "
                "-- the frames are all in the batch folder.")
        else:
            mp4_path = os.path.join(
                batch_folder, "{}({})_movie.mp4".format(
                    batch_name, int(batch_num)))
            cmd = _anim_video_command(
                ffmpeg, batch_folder, batch_name, batch_num,
                first_frame, fps_value, mp4_path)
            try:
                result = subprocess.run(
                    cmd, capture_output=True,
                    stdin=subprocess.DEVNULL,
                    timeout=max(120, min(1800,
                                         5 * len(frames))))
                if result.returncode == 0 and \
                        os.path.exists(mp4_path):
                    made.append(mp4_path)
                    movie = mp4_path
                else:
                    tail = (result.stderr or b"").decode(
                        "utf-8", errors="ignore").strip()
                    notes.append(
                        "MP4 assembly failed (ffmpeg exit {}): "
                        "{}".format(
                            result.returncode,
                            tail.splitlines()[-1]
                            if tail else "no output"))
            except Exception as exc:  # noqa: BLE001 -- plain note
                notes.append(
                    "MP4 assembly failed: {}".format(exc))
    if movie is None and not want_gif:
        # Safety net: the MP4 path produced nothing (no ffmpeg,
        # a failed encode, a timeout). Still hand back a movie
        # -- the PIL GIF needs nothing external. Skipped when
        # the GIF box was ticked: that GIF was already attempted
        # above, and its own note already stands.
        gif_path = os.path.join(
            batch_folder, "{}({})_movie.gif".format(
                batch_name, int(batch_num)))
        try:
            from PIL import Image
            images = [Image.open(path).convert("RGB")
                      for path in frames]
            duration = max(1, int(round(1000.0 / float(fps_value))))
            images[0].save(
                gif_path, save_all=True, append_images=images[1:],
                duration=duration, loop=0)
            made.append(gif_path)
            movie = gif_path
            notes.append(
                "Made a GIF instead, from the same frames.")
        except Exception as exc:  # noqa: BLE001 -- plain note
            notes.append(
                "GIF fallback failed too: {}".format(exc))
    if made:
        notes.append("Movie file(s): {}".format(
            ", ".join(os.path.basename(path) for path in made)))
    return movie, notes


def generate_animation(prompt, keyframes_text, mode,
                       diffusion_model, canvas, clip_on, steps,
                       guidance, max_frames, fps_value,
                       frames_skip_pct, angle, zoom, translation_x,
                       translation_y, translation_z,
                       rotation_3d_x, rotation_3d_y, rotation_3d_z,
                       turbo_mode, turbo_steps, turbo_preroll,
                       tv_scale, sat_scale, range_scale,
                       cutn_batches, eta, clamp_max, frames_scale,
                       seed, batch_name, resume, want_mp4, want_gif,
                       progress=gr.Progress(track_tqdm=True)):
    """Gradio callback for the Animation tab: yields (status,
    gallery, latest, movie) while the run goes, in the same
    worker-thread + progress-poll shape as generate()."""
    if not prompt or not prompt.strip():
        yield ("Type a prompt first.", [], None, None)
        return
    try:
        text_prompts = _parse_prompt_keyframes(keyframes_text,
                                               prompt)
    except ValueError as exc:
        yield (str(exc), [], None, None)
        return
    if disco_engine._STATE["out_dir"] is None:
        yield ("The engine is not initialised yet -- the app "
               "is still starting.", [], None, None)
        return
    width, height = _canvas_to_wh(canvas)
    seed_text = (seed or "").strip()
    if seed_text == "":
        random.seed()
        seed_value = random.randint(0, 2 ** 32)
        set_seed = "random_seed"
    else:
        try:
            seed_value = int(seed_text)
        except ValueError:
            yield ("Seed must be a whole number, got: {!r}"
                   .format(seed), [], None, None)
            return
        set_seed = str(seed_value)
    batch_label = (batch_name or "").strip() or "DiscoAnim"
    batch_folder = os.path.join(disco_engine._STATE["out_dir"],
                                batch_label)
    try:
        os.makedirs(batch_folder, exist_ok=True)
        batch_num, start_frame = _anim_plan(
            batch_folder, batch_label, bool(resume))
        settings = _anim_settings({
            "prompt": prompt.strip(),
            "diffusion_model": diffusion_model,
            "steps": steps,
            "width": width,
            "height": height,
            "guidance": guidance,
            "tv_scale": tv_scale,
            "sat_scale": sat_scale,
            "range_scale": range_scale,
            "cutn_batches": cutn_batches,
            "eta": eta,
            "clamp_max": clamp_max,
            "frames_scale": frames_scale,
            "frames_skip_pct": frames_skip_pct,
            "angle": (angle or "").strip(),
            "zoom": (zoom or "").strip(),
            "translation_x": (translation_x or "").strip(),
            "translation_y": (translation_y or "").strip(),
            "translation_z": (translation_z or "").strip(),
            "rotation_3d_x": (rotation_3d_x or "").strip(),
            "rotation_3d_y": (rotation_3d_y or "").strip(),
            "rotation_3d_z": (rotation_3d_z or "").strip(),
            "turbo_mode": turbo_mode,
            "turbo_steps": turbo_steps,
            "turbo_preroll": turbo_preroll,
            "batch_name": batch_label,
            "set_seed": set_seed,
            "clip_on": list(clip_on),
        })
    except ValueError as exc:
        yield (str(exc), [], None, None)
        return
    max_frames = max(1, int(max_frames or 1))

    errors = queue.Queue()
    result = {}

    for stale in (disco_engine.progress_path(),
                  os.path.join(UI_DIR, "progress_preview.png")):
        try:
            os.remove(stale)
        except OSError:
            pass
    disco_engine.clear_stop()

    def _work():
        try:
            result.update(_run_animation_engine(
                settings, batch_num, seed_value, text_prompts,
                mode, max_frames, start_frame, bool(resume)))
        except Exception as exc:  # noqa: BLE001 -- surfaced in UI
            errors.put(exc)

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    status = ("Starting animation... (first run loads the "
              "models, please wait)")
    yield status, [], None, None
    last_preview = None
    gallery = []
    latest = None
    while thread.is_alive():
        preview = _snapshot_progress()
        if preview and preview != last_preview:
            last_preview = preview
            latest = preview
        gallery = _anim_frames_in(batch_folder, batch_label,
                                  batch_num)
        if gallery:
            status = ("Animating ({}): {} of {} frames saved "
                      "in {}.".format(
                          mode, len(gallery), max_frames,
                          batch_label))
        yield status, gallery, latest, None
        time.sleep(2)
    thread.join()
    if not errors.empty():
        exc = errors.get()
        yield ("Error: {}".format(exc), gallery, latest, None)
        return
    frames = result.get("frames", [])
    movie, notes = _assemble_movie(
        frames, fps_value, batch_folder, batch_label, batch_num,
        bool(want_mp4), bool(want_gif))
    status = (
        "Done. {} frames, seed {}, batch {}({}). "
        "Models: {}. Settings saved to {}.".format(
            len(frames), result.get("seed"), batch_label,
            result.get("batch_num"),
            "; ".join(result.get("status", [])),
            result.get("settings_path")))
    for note in notes:
        status += " " + note
    if frames and not movie and want_mp4:
        status += (" The frames are in the batch folder; the "
                   "movie player only takes the MP4.")
    yield status, frames, (frames[-1] if frames else latest), movie


# ---------------------------------------------------------------------------
# Upscale -- 2x/4x enlargement of a saved image, PIL Lanczos
# (Pillow is already the engine's image library; no new
# dependency). The result lands in an "upscaled" folder next
# to app.py. This is a plain resample upscale, not an AI
# upscaler -- the tab says so.
# ---------------------------------------------------------------------------


def _upscale_output_path(image_path, factor):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(
        UPSCALE_DIR, "{}_x{}.png".format(stem, int(factor)))


def upscale_image(image_path, factor):
    """Upscale one image file by 2x or 4x. Returns
    (output path or None, status). A missing pick, a bad
    factor, or an unreadable image is a plain status and
    nothing is written."""
    if not image_path:
        return None, "Pick an image to upscale first."
    try:
        factor = int(factor)
    except (TypeError, ValueError):
        return None, "Upscale factor must be 2 or 4."
    if factor not in (2, 4):
        return None, "Upscale factor must be 2 or 4."
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            big = img.resize((width * factor, height * factor),
                             Image.LANCZOS)
    except Exception as exc:  # noqa: BLE001 -- plain status text
        return None, "Could not open that image: {}".format(exc)
    try:
        os.makedirs(UPSCALE_DIR, exist_ok=True)
        out_path = _upscale_output_path(image_path, factor)
        big.save(out_path)
    except OSError as exc:  # noqa: BLE001 -- plain status text
        return None, "Could not save the upscale: {}".format(exc)
    return out_path, (
        "Upscaled {}x{} -> {}x{} ({}x, Lanczos). "
        "Saved to {}.".format(width, height, width * factor,
                              height * factor, factor, out_path))


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


# ---------------------------------------------------------------------------
# Seed variations -- the planner's definition: "walk a liked
# frame's seed forward, settings frozen." Every other Generate
# control stays exactly as set; the run is one ordinary
# generate() per seed (seed, seed+1, ...), frames gathered
# into one gallery. A blank Seed box picks the starting seed
# at random, once, and walks from there.
# ---------------------------------------------------------------------------


def _variation_seeds(seed_text, count, rng=None):
    """The seed strings for a variations run: the base seed
    walked forward, one per variation, wrapping at 2**32
    (the engine's own seed range). The count is clamped to
    1..10. A blank seed box starts from a random base (the
    rng is injectable for tests). A seed that is not a whole
    number raises ValueError with a plain message."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    count = max(1, min(10, count))
    text = (seed_text or "").strip()
    if text == "":
        rng = rng if rng is not None else random
        base = rng.randint(0, 2 ** 32 - 1)
    else:
        try:
            base = int(text)
        except ValueError:
            raise ValueError(
                "Seed must be a whole number, got: {!r}".format(
                    seed_text))
    return [str((base + offset) % (2 ** 32))
            for offset in range(count)]


def generate_variations(prompt, diffusion_model, steps, guidance,
                        canvas, custom_w, custom_h, clip_on, seed,
                        batch_name, use_h_symmetry,
                        transformation_percent, cut_ic_pow,
                        n_batches, tv_scale, sat_scale, range_scale,
                        cutn_batches, skip_steps, eta, clamp_max,
                        cut_overview, cut_innercut, cut_icgray_p,
                        use_secondary_model, custom_path,
                        use_checkpoint, diffusion_sampling_mode,
                        clamp_grad, fuzzy_prompt, clip_denoised,
                        randomize_class, skip_augs, perlin_init,
                        perlin_mode, rand_mag, display_rate,
                        intermediate_saves,
                        intermediates_in_subfolder, queue_text,
                        variation_count):
    """Gradio callback: yields (status, gallery, latest) like
    generate(), one single-frame generate() per walked seed.
    The prompt queue is ignored here (variations are about
    the one prompt), and Batches is forced to 1 per seed --
    the count box is the frame count."""
    if not prompt or not prompt.strip():
        yield "Type a prompt first.", [], None
        return
    try:
        seeds = _variation_seeds(seed, variation_count)
    except ValueError as exc:
        yield str(exc), [], None
        return
    all_frames = []
    latest = None
    total = len(seeds)
    for index, seed_text in enumerate(seeds):
        status = ""
        frames = []
        inner_latest = None
        for status, gallery, inner_latest in generate(
                prompt, diffusion_model, steps, guidance, canvas,
                custom_w, custom_h, clip_on, seed_text, batch_name,
                use_h_symmetry, transformation_percent, cut_ic_pow,
                1, tv_scale, sat_scale, range_scale, cutn_batches,
                skip_steps, eta, clamp_max, cut_overview,
                cut_innercut, cut_icgray_p, use_secondary_model,
                custom_path, use_checkpoint,
                diffusion_sampling_mode, clamp_grad, fuzzy_prompt,
                clip_denoised, randomize_class, skip_augs,
                perlin_init, perlin_mode, rand_mag, display_rate,
                intermediate_saves, intermediates_in_subfolder, ""):
            frames = gallery
            yield ("Variation {}/{} (seed {}): {}".format(
                       index + 1, total, seed_text, status),
                   all_frames + list(gallery),
                   inner_latest or latest)
        if status.startswith("Error:") or \
                status == "Type a prompt first.":
            yield status, all_frames, latest
            return
        all_frames.extend(frames)
        if inner_latest is not None:
            latest = inner_latest
        yield ("Variation {}/{} done (seed {}).".format(
                   index + 1, total, seed_text),
               all_frames, latest)
    yield ("Done. {} variation(s), seeds {} to {}.".format(
               total, seeds[0], seeds[-1]),
           all_frames, latest)


def build_ui():
    with gr.Blocks(title="Disco Diffusion") as demo:
        gr.Markdown("# Disco Diffusion\n"
                    "No Jupyter, no Colab. Generate is the stills "
                    "page (with saved setups and seed variations); "
                    "Animation runs Disco's own 2D/3D frame chains "
                    "with the movie in the page; Upscale enlarges "
                    "a saved frame 2x/4x; Prompt Storage is a "
                    "notebook for your favorite prompts -- they "
                    "persist in prompt_storage.txt next to app.py.")
        with gr.Tabs():
            with gr.Tab("Generate"):
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
                            steps_num = gr.Number(
                                label="Steps (type-in, Enter applies)",
                                value=RECIPE["steps"], precision=0)
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
                                value=True)
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
                        gr.Markdown(
                            "**Saved setups** -- the whole recipe "
                            "above under a name (kept in setups.json "
                            "next to app.py). **Seed variations** -- "
                            "the settings stay frozen and the seed "
                            "walks forward from the Seed box, one "
                            "frame per seed.")
                        with gr.Row():
                            setup_name = gr.Textbox(
                                label="Setup name", value="")
                            setup_save_btn = gr.Button("Save setup")
                        with gr.Row():
                            setup_dd = gr.Dropdown(
                                label="Saved setups",
                                choices=_setup_choices(),
                                value=None)
                            setup_load_btn = gr.Button("Load setup")
                            setup_del_btn = gr.Button("Delete setup")
                        with gr.Row():
                            variation_count = gr.Number(
                                label="Seed variations (count)",
                                value=3, precision=0)
                            variation_btn = gr.Button(
                                "Generate variations")
                    with gr.Column(scale=3):
                        status = gr.Textbox(label="Status", interactive=False)
                        latest = gr.Image(label="Latest frame (full, uncropped)")
                        gallery = gr.Gallery(label="Results")
            with gr.Tab("Prompt Storage"):
                gr.Markdown(
                    "## Prompt Storage\n"
                    "This is where you put your favorite prompts "
                    "and it stores them. One prompt per line. They "
                    "are saved in prompt_storage.txt next to "
                    "app.py, so they are still here when you "
                    "reopen the app. Edit the notebook, then hit "
                    "Save Prompt Storage. Pick a stored prompt "
                    "below and hit Use this prompt to load it "
                    "into the Prompt box.")
                storage_box = gr.Textbox(
                    label="Prompt Storage notebook "
                          "(one prompt per line)",
                    value=_load_prompt_storage(), lines=12)
                with gr.Row():
                    storage_save_btn = gr.Button(
                        "Save Prompt Storage")
                    storage_add_btn = gr.Button(
                        "Add current prompt")
                with gr.Row():
                    storage_dd = gr.Dropdown(
                        label="Stored prompts",
                        choices=_prompt_storage_choices(),
                        value=None)
                    storage_use_btn = gr.Button("Use this prompt")
            with gr.Tab("Animation"):
                gr.Markdown(
                    "## Animation\n"
                    "Disco's own animation, the notebook's frame "
                    "chain: frame 1 diffuses in full, then every "
                    "frame starts from the previous one -- warped "
                    "by the movement below -- and diffuses only "
                    "the remaining steps (frames_skip_steps). "
                    "Alan's recipe: guidance LOW (high guidance "
                    "flickers), 150 steps, 60% skip, gentle drift. "
                    "A 360-frame run is an overnight job -- set "
                    "Frames low (8) for a first test. Frames land "
                    "in the batch folder; the movie is assembled "
                    "there and plays below. Resume picks up the "
                    "newest run in the batch folder where it "
                    "stopped -- same settings, don't change them "
                    "between the stop and the resume.")
                with gr.Row():
                    with gr.Column(scale=2):
                        anim_prompt = gr.Textbox(
                            label="Prompt",
                            value=ANIM_DEFAULT_PROMPT,
                            lines=3)
                        anim_keyframes = gr.Textbox(
                            label="Prompt keyframes (optional, one "
                                  "per line as 'frame: prompt' -- "
                                  "the prompt morphs at that frame)",
                            lines=3, value="")
                        anim_mode = gr.Radio(
                            label="Animation mode",
                            choices=["2D", "3D"], value="2D")
                        anim_model = gr.Dropdown(
                            label="Diffusion model",
                            choices=MODEL_CHOICES,
                            value="512x512_diffusion_uncond_finetune_008100")
                        anim_canvas = gr.Radio(
                            label="Canvas (width x height)",
                            choices=CANVAS_CHOICES,
                            value="512x512")
                        anim_clip_on = gr.CheckboxGroup(
                            label="CLIP models",
                            choices=CLIP_CHOICES,
                            value=ANIM_CLIP_DEFAULT)
                        with gr.Row():
                            anim_steps = gr.Slider(
                                label="Steps", minimum=25,
                                maximum=1000, step=25, value=150)
                            anim_guidance = gr.Slider(
                                label="Guidance (clip_guidance_scale) "
                                      "-- keep it LOW for animation",
                                minimum=100, maximum=100000,
                                step=100, value=1000)
                        with gr.Row():
                            anim_max_frames = gr.Number(
                                label="Frames (max_frames)",
                                value=360, precision=0)
                            anim_fps = gr.Number(
                                label="Movie fps", value=12,
                                precision=0)
                            anim_skip_pct = gr.Number(
                                label="frames_skip_steps (%)",
                                value=60, precision=0)
                        with gr.Row():
                            anim_seed = gr.Textbox(
                                label="Seed (blank = random)",
                                value="")
                            anim_batch = gr.Textbox(
                                label="Animation batch name",
                                value="DiscoAnim")
                        anim_resume = gr.Checkbox(
                            label="Resume (continue the newest run "
                                  "in this batch folder)",
                            value=False)
                        with gr.Accordion("Movement (Disco keyframes)",
                                          open=True):
                            anim_angle = gr.Textbox(
                                label="angle", value="0: (0)")
                            anim_zoom = gr.Textbox(
                                label="zoom", value="0: (1.0)")
                            anim_tx = gr.Textbox(
                                label="translation_x", value="0: (0)")
                            anim_ty = gr.Textbox(
                                label="translation_y", value="0: (0)")
                            anim_tz = gr.Textbox(
                                label="translation_z",
                                value="0: (0.2)")
                        with gr.Accordion("3D movement & turbo "
                                          "(3D mode only)",
                                          open=False):
                            anim_rx = gr.Textbox(
                                label="rotation_3d_x", value="0: (0)")
                            anim_ry = gr.Textbox(
                                label="rotation_3d_y", value="0: (0)")
                            anim_rz = gr.Textbox(
                                label="rotation_3d_z", value="0: (0)")
                            with gr.Row():
                                anim_turbo = gr.Checkbox(
                                    label="turbo_mode", value=False)
                                anim_turbo_steps = gr.Number(
                                    label="turbo_steps", value=3,
                                    precision=0)
                                anim_turbo_preroll = gr.Number(
                                    label="turbo_preroll", value=10,
                                    precision=0)
                        with gr.Accordion("Animation recipe "
                                          "(tv, sat, cuts)",
                                          open=False):
                            with gr.Row():
                                anim_tv = gr.Number(
                                    label="tv_scale", value=0,
                                    precision=0)
                                anim_sat = gr.Number(
                                    label="sat_scale", value=300,
                                    precision=0)
                                anim_range = gr.Number(
                                    label="range_scale", value=150,
                                    precision=0)
                            with gr.Row():
                                anim_cutn = gr.Number(
                                    label="cutn_batches", value=1,
                                    precision=0)
                                anim_eta = gr.Number(
                                    label="eta", value=0.8)
                                anim_clamp = gr.Number(
                                    label="clamp_max", value=0.05)
                                anim_frames_scale = gr.Number(
                                    label="frames_scale", value=1500,
                                    precision=0)
                        with gr.Row():
                            anim_mp4 = gr.Checkbox(
                                label="Make MP4 (needs ffmpeg)",
                                value=True)
                            anim_gif = gr.Checkbox(
                                label="Make GIF", value=False)
                        with gr.Row():
                            anim_btn = gr.Button(
                                "Generate animation",
                                variant="primary")
                            anim_stop_btn = gr.Button("Stop")
                    with gr.Column(scale=3):
                        anim_status = gr.Textbox(
                            label="Animation status",
                            interactive=False)
                        anim_latest = gr.Image(
                            label="Latest frame (full, uncropped)")
                        anim_gallery = gr.Gallery(
                            label="Animation frames")
                        anim_movie = gr.Video(label="Movie")
            with gr.Tab("Upscale"):
                gr.Markdown(
                    "## Upscale\n"
                    "Enlarge a saved image 2x or 4x (PIL Lanczos "
                    "resampling -- a plain, honest upscale, not "
                    "an AI upscaler). The result is written to "
                    "an 'upscaled' folder next to app.py. Drop "
                    "any frame from a batch folder here.")
                up_image = gr.Image(
                    label="Image to upscale", type="filepath")
                up_factor = gr.Radio(
                    label="Factor", choices=[2, 4], value=2)
                up_btn = gr.Button("Upscale", variant="primary")
                up_status = gr.Textbox(
                    label="Upscale status", interactive=False)
                up_out = gr.Image(label="Upscaled image")
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
        storage_save_btn.click(
            _save_prompt_storage, inputs=[storage_box],
            outputs=[status, storage_dd])
        storage_add_btn.click(
            _add_current_prompt, inputs=[prompt, storage_box],
            outputs=[status, storage_box, storage_dd])
        storage_use_btn.click(
            _use_stored_prompt, inputs=[storage_dd], outputs=[prompt])
        steps.change(
            _steps_echo, inputs=[steps], outputs=[steps_num])
        steps_num.submit(
            _steps_typein, inputs=[steps_num], outputs=[steps])
        _recipe_components = [
            prompt, diffusion_model, steps, guidance, canvas,
            custom_w, custom_h, clip_on, seed, batch_name,
            use_h_symmetry, transformation_percent, cut_ic_pow,
            n_batches, tv_scale, sat_scale, range_scale,
            cutn_batches, skip_steps, eta, clamp_max, cut_overview,
            cut_innercut, cut_icgray_p, use_secondary_model,
            custom_path, use_checkpoint, diffusion_sampling_mode,
            clamp_grad, fuzzy_prompt, clip_denoised,
            randomize_class, skip_augs, perlin_init, perlin_mode,
            rand_mag, display_rate, intermediate_saves,
            intermediates_in_subfolder]
        setup_save_btn.click(
            _save_setup, inputs=[setup_name] + _recipe_components,
            outputs=[status, setup_dd])
        setup_load_btn.click(
            _load_setup, inputs=[setup_dd],
            outputs=_recipe_components + [status])
        setup_del_btn.click(
            _delete_setup, inputs=[setup_dd],
            outputs=[status, setup_dd])
        variation_btn.click(
            generate_variations,
            inputs=_recipe_components + [queue_text,
                                         variation_count],
            outputs=[status, gallery, latest])
        anim_btn.click(
            generate_animation,
            inputs=[anim_prompt, anim_keyframes, anim_mode,
                    anim_model, anim_canvas, anim_clip_on,
                    anim_steps, anim_guidance, anim_max_frames,
                    anim_fps, anim_skip_pct, anim_angle, anim_zoom,
                    anim_tx, anim_ty, anim_tz, anim_rx, anim_ry,
                    anim_rz, anim_turbo, anim_turbo_steps,
                    anim_turbo_preroll, anim_tv, anim_sat,
                    anim_range, anim_cutn, anim_eta, anim_clamp,
                    anim_frames_scale, anim_seed, anim_batch,
                    anim_resume, anim_mp4, anim_gif],
            outputs=[anim_status, anim_gallery, anim_latest,
                     anim_movie])
        anim_stop_btn.click(disco_engine.request_stop,
                            inputs=[], outputs=[])
        up_btn.click(
            upscale_image, inputs=[up_image, up_factor],
            outputs=[up_out, up_status])
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
