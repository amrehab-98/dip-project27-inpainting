# Diffusion-Based Image Inpainting

Digital Image Processing — Project 27.

Compare OpenCV Telea and Navier–Stokes inpainting with a pretrained diffusion
inpainting model on structured and random masks. The diffusion approach will use
a frozen backbone and a lightweight trainable mask-conditioning head.

**Status:** Development environment configured. Dataset preparation, baseline,
training, and evaluation scripts are not implemented yet.

## Setup

Tested on Windows with Python 3.13.18 and an NVIDIA GPU using CUDA 12.6.

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` lists the direct project dependencies. Use
`requirements-lock.txt` instead to reproduce the exact runtime dependency versions.
The virtual environment, datasets, and model weights are excluded from Git.

## Dataset

The project targets a 2,000–5,000-image Places subset with synthetic masks.
The current download is the Places365 validation split (36,500 images), stored at:

```text
../Project/dataset/val_large (1)/val_large/
```

Subset selection and train/validation/test manifests are pending. Masks will use
`0` for known pixels and `255` for regions to reconstruct.

## Run

Experiment commands will be added with the implementation. To check the environment:

```powershell
python -m pip check
python -c "import cv2, torch; print('OpenCV:', cv2.__version__); print('CUDA:', torch.cuda.is_available())"
```
