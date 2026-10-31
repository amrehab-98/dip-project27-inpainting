"""Shared image convention: RGB uint8 arrays shaped (height, width, 3)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path, size: int) -> np.ndarray:
    """Apply EXIF orientation, convert to RGB, then center-crop/resize to square."""
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        return np.array(ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS))


def save_image(path: str | Path, pixels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
