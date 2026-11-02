# Project 27 ? Data, masks, and OpenCV baselines

Student A's part of the image-inpainting project: select a reproducible dataset
subset, generate rectangular/free-form masks, and run Telea/Navier?Stokes baselines.

## Setup

Run from the repository root. Python 3.13 is required (tested with 3.13.18).
Dependencies are PyTorch/torchvision (installed by a helper script) plus the
packages in `requirements.txt`: NumPy, OpenCV, Pillow, and the Hugging Face
diffusion stack.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/install_torch.py
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`install_torch.py` must run first, because no single PyTorch wheel supports every
GPU generation. It installs PyTorch 2.9.1 and torchvision 0.24.1, choosing the
build from the compute capability that `nvidia-smi` reports (oldest GPU if several):

| Build | GPUs |
|---|---|
| `cu128` | Compute capability ≥ 7.0: Volta, Turing (GTX 16 / RTX 20), and newer, including Blackwell |
| `cu126` | Pascal (GTX 10-series) and older |
| `cpu` | No NVIDIA GPU detected |

On macOS it installs the standard PyPI build, which supports Apple MPS.

The wheels are several GB. They download in resumable chunks into `.cache/wheels/`
and are checked against SHA-256. If the download stops, rerun the script to
continue. Options: `--variant cu128|cu126|cpu` to override detection, `--dry-run`
to print the wheels and pip command only, and `--chunk-mb` / `--retries` to tune
the download.

## Configuration

Edit `configs/default.json`:

| Section | Purpose |
|---|---|
| `data` | Dataset root, subset size/seed, image size, manifest path |
| `masks` | Mask seed, types, and nominal hole fraction |
| `opencv` | Methods (`telea`, `ns`) and inpainting radius |
| `run` | Preview image count and output directory |

Paths resolve from the repository root; `--image` resolves from the terminal's
working directory. Current defaults are 512?512 RGB, nominal 25% holes, radius 3,
and both baseline methods. The preview uses three images.

## Dataset subset

The current manifest selects 2,000 relative paths from 36,500 local Places365
validation images. Its keys are `source`, `seed`, and `images`. The source
release's validation label is independent of future project train/test splits.

Candidates are sorted, then sampled with `data.seed=42`. To create a manifest:

```powershell
python -m inpainting prepare-data
```

The current manifest already exists. This command preserves it; choose a new
`data.manifest_path` before deliberately selecting another subset. Raw images
stay outside Git. Document the exact source release and its usage terms.

## Run baselines

```powershell
python -m inpainting run --name baseline-preview
python -m inpainting run --image "path/to/image.jpg" --name single-image
```

Set `opencv.methods` to `["telea"]`, `["ns"]`, or `["telea", "ns"]`.
Every method receives the same prepared image and binary mask.

`run.image_count` limits how many subset images are processed. Remove this key
to use `data.subset_size` and process the full configured subset.

Each run saves original/masked images, masks, baseline predictions, comparison
strips, two labeled overview sheets, and one final `run.json`. Each overview shows
only the first three images; individual outputs are saved for all processed images.
The report records configuration, image IDs, effective mask seeds, actual
coverage, and timings.
Existing output folders are preserved; use a fresh run name.

## Shared image and mask conventions

Images are RGB `uint8`, center-cropped/resized to the configured square size.
Masks are `uint8`: 0 means preserve, 255 means reconstruct. Structured masks
contain one rectangle; random masks contain rounded free-form brush strokes.

`masks.seed=123` controls masks independently of subset selection. Effective mask
seeds depend on the relative image ID, type, and generator version, so changing
processing order or moving the dataset preserves each mask. Coverage is
approximate and recorded. Saved image/mask pairs can be reused by teammates.

M1 uses one fixed subset without training splits. The proposal should plan the
later evaluation separation before any training or tuning.
