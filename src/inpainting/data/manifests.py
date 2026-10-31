"""Seeded file-level subset selection; no image copies or downloads."""

import random
from pathlib import Path


def build_manifest(root: Path, count: int, seed: int, source: str) -> dict:
    """Save a reproducible subset of relative paths for M1 inference and samples.

    This does not detect duplicate image content or verify Places2 provenance.
    Perform those checks before adopting a manifest for experiments.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    if type(count) is not int or count < 1:
        raise ValueError("Subset count must be a positive integer")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    if count > len(files):
        raise ValueError(f"Requested {count} images, found only {len(files)}")
    selected = random.Random(seed).sample(files, count)
    return {
        "source": source,
        "seed": seed,
        "images": selected,
    }
