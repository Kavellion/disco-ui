# Disco UI

**Disco Diffusion, without Jupyter and without Colab.**

Disco Diffusion is a beautiful, weird, powerful image generator — but
running it used to mean notebooks, cells, and cloud accounts. This is a
plain app: run it, a page opens in your browser, type a prompt, pictures
appear. Built for everyone, free forever.

## What it does

- Still-image generation on your own GPU — no cloud, no account
- Prompt queue: one prompt per line, runs through them in order
- Dynamic prompts: `{cat|dog|owl}` picks one at random per frame
- Random prompt button for when you're stuck (it draws from `disco-ui/random_prompts.txt` — one prompt per line — so you can add your own; without the file it uses a built-in list)
- Batches: walk away, come back to finished frames
- Stop button that ends a run cleanly
- Model choice (512, 256, custom), CLIP selection, symmetry,
  TV/sat/range/cut controls, and the rest of the notebook's settings
- Every run saves its frames and a `_settings.txt` file next to them

## Requirements

- Windows, an NVIDIA GPU, and a recent driver
- Python: nothing to install — Python 3.10 is included in this package, bundled in `python_embeded\`. Any Python you already have is left completely alone
- Disk space: the app itself is small; the models download on first run
  (several GB total) and your frames live in `images_out/`

## Install

1. Download this repo (Code → Download ZIP) or `git clone` it, and unzip it. Python is included — there is nothing to install first.
2. Double-click `install.bat` and wait — it sets up pip in the bundled
   Python, installs PyTorch with CUDA, and fetches the CLIP and
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
