# Disco UI
Easiest install: grab the bundled zip from the Releases tab — no Python setup needed.

**Disco Diffusion, without Jupyter and without Colab.**

Disco Diffusion is a beautiful, weird, powerful image generator — but
running it used to mean notebooks, cells, and cloud accounts. This is a
plain app: run it, a page opens in your browser, type a prompt, pictures
appear. Built for everyone, free forever.
<img width="1280" height="720" alt="disco-ui-screenshot" src="https://github.com/user-attachments/assets/e8cf4c02-3493-40e0-be2a-7d33c4bbce3d" />
*Disco UI with generated images*
<img width="1280" height="768" alt="14-cyberpunk-market" src="https://github.com/user-attachments/assets/84538608-94ab-475d-9c82-479ceffa5cdc" />
*Example image made with Disco UI: cyberpunk street market*
<img width="1280" height="768" alt="59-lighthouse-red-sea-towers" src="https://github.com/user-attachments/assets/8c4e5822-fbc8-4bf4-b361-22a642011d80" />
*Example image made with Disco UI: red-sea lighthouse*

## What it does

- Still-image generation on your own GPU — no cloud, no account
- Prompt queue: one prompt per line, runs through them in order
- Dynamic prompts: `{cat|dog|owl}` picks one at random per frame
- Random prompt button for when you're stuck
- Batches: walk away, come back to finished frames
- Stop button that ends a run cleanly
- Model choice (512, 256, custom), CLIP selection, symmetry,
  TV/sat/range/cut controls, and the rest of the notebook's settings
- Every run saves its frames and a `_settings.txt` file next to them

## Requirements

- Windows, an NVIDIA GPU, and a recent driver
- Python 3.10 or newer, with "Add python.exe to PATH" ticked
- Disk space: the app itself is small; the models download on first run
  (several GB total) and your frames live in `images_out/`

### Install Python first
If you don't already have Python 3.10 or newer, this is the one thing you install yourself — install.bat handles everything else.
1. Download Python 3.10 from https://www.python.org/downloads/release/python-31022/ — pick the Windows installer (64-bit).
2. Run it. On the first screen, tick "Add python.exe to PATH" at the bottom, then Install.
3. Then continue with Install below.
## Install

1. Download this repo (Code → Download ZIP) or `git clone` it.
2. Double-click `install.bat` and wait — it builds a virtual
   environment, installs PyTorch with CUDA, and fetches the CLIP and
   guided-diffusion code folders. Run it again if anything interrupts it;
   it resumes where it stopped.
3. Double-click `run.bat`. A page opens in your browser. Type a prompt
   and hit Generate.

First run takes a while: the models load once, then stay loaded. Picking
a different diffusion model or CLIP set reloads what's needed; repeat
runs reuse what's already loaded.

## Models and permissions

See [MODELS.md](MODELS.md). Stock models download automatically from
their public sources. Other creators' checkpoints are not bundled — if
you have your own model file, use the custom path box or drop it in
`models/`.

## Credits

- The engine core is derived from [Disco Diffusion](https://github.com/alembics/disco-diffusion) (MIT license).
- CLIP and guided-diffusion code are by OpenAI (MIT license).
- This UI (the Gradio app, the wrapper, the installer) is by Alan Carmack.

## License

MIT — free to use, share, and change. See [LICENSE](LICENSE).
