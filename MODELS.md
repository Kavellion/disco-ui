# Models — what downloads, what doesn't, and why

Disco UI itself is free and open (MIT). The AI models it runs belong to
their creators, so this page says plainly where each one comes from.
Nothing here is sold, and no model file is bundled in this repo.

## Downloaded automatically on first run (free, public sources)

These come from their original public homes the first time you pick them.
You don't have to fetch anything by hand.

| Model | Source | Used for |
|---|---|---|
| 256x256 diffusion unconditional | OpenAI public weights | The 256 model |
| 512x512 diffusion unconditional (finetune 008100) | Hugging Face `lowlevelware` (public) | The 512 model |
| Secondary diffusion (ImageNet 2) | Hugging Face `spaces/huggi` (public) | Secondary model, on by default |
| CLIP models (ViT-B/32, ViT-B/16, ViT-L/14, RN50, RN101, RN50x4) | OpenAI CLIP (MIT) | Text guidance — downloaded by the CLIP library itself |

## Code folders fetched by install.bat (free, open source)

| Folder | Source | License |
|---|---|---|
| `CLIP/` | github.com/openai/CLIP | MIT |
| `guided-diffusion/` | github.com/openai/guided-diffusion | MIT |

## Your own model files — not bundled, not downloaded

Checkpoints made by other creators are **not** bundled here and the
installer does not download them.

If you have your own model file, you have two routes: drop it in the
`models/` folder and pick it if it appears in the dropdown, or use the
custom path box (model: custom) and point it at the file. The app uses
a file already in `models/` as-is, no download attempted.

Do not redistribute someone else's checkpoint without their permission.

## Future models

Pixel-art, watercolor, and pulp-style checkpoints (public Hugging Face
repos, for example KaliYuga's) are not wired into this UI yet. When they
are added, they will be credited here the same way.
