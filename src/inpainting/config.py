"""JSON settings for subset selection, masks, and OpenCV baselines."""

import json
import math
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Load the shared data, mask, and baseline settings."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    data, masks, run = config["data"], config["masks"], config["run"]
    for label, seed in (("data.seed", data["seed"]), ("masks.seed", masks["seed"])):
        if type(seed) is not int or seed < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
    if type(data["subset_size"]) is not int or data["subset_size"] < 1:
        raise ValueError("data.subset_size must be a positive integer")
    if type(data["image_size"]) is not int or data["image_size"] < 8:
        raise ValueError("data.image_size must be an integer of at least 8")
    if type(masks["hole_fraction"]) not in (int, float) or not 0 < masks["hole_fraction"] < 1:
        raise ValueError("masks.hole_fraction must lie strictly between 0 and 1")
    mask_types = masks["types"]
    if not isinstance(mask_types, list) or not mask_types or any(
        kind not in ("structured", "random") for kind in mask_types
    ) or len(set(mask_types)) != len(mask_types):
        raise ValueError("masks.types must contain unique structured and/or random entries")
    run.setdefault("image_count", data["subset_size"])
    if type(run["image_count"]) is not int or run["image_count"] < 1:
        raise ValueError("run.image_count must be a positive integer")
    if not isinstance(run["output_dir"], str) or not run["output_dir"]:
        raise ValueError("run.output_dir must be a nonempty path string")

    opencv = config["opencv"]
    methods = opencv["methods"]
    if not isinstance(methods, list) or not methods or any(
        method not in ("telea", "ns") for method in methods
    ) or len(set(methods)) != len(methods):
        raise ValueError("opencv.methods must contain unique telea and/or ns entries")
    radius = opencv["radius"]
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("opencv.radius must be positive")
    return config


def resolve_path(config_path: str | Path, value: str) -> Path:
    """Configs live in <repo>/configs; absolute user paths are also accepted."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent.parent / path).resolve()
