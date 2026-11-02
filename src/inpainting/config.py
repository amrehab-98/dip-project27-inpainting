"""Shared JSON settings with validation for the methods selected for this run."""

import json
import math
from pathlib import Path


def load_config(path: str | Path, *, methods: list[str] | None = None) -> dict:
    """Load one workflow config; validate only the selected method sections.

    A method override replaces run.methods. An explicit empty list is used by
    prepare-data, which needs shared settings but no inpainting method.
    """
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

    selected = run["methods"] if methods is None else methods
    if not isinstance(selected, list) or (not selected and methods is None) or any(
        method not in ("opencv", "diffusion") for method in selected
    ) or len(set(selected)) != len(selected):
        raise ValueError("run.methods must contain unique opencv and/or diffusion entries")
    run["methods"] = selected.copy()

    if "opencv" in selected:
        opencv = config["opencv"]
        classical = opencv["methods"]
        if not isinstance(classical, list) or not classical or any(
            method not in ("telea", "ns") for method in classical
        ) or len(set(classical)) != len(classical):
            raise ValueError("opencv.methods must contain unique telea and/or ns entries")
        radius = opencv["radius"]
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("opencv.radius must be positive")

    if "diffusion" in selected:
        diffusion = config["diffusion"]
        if data["image_size"] % 8:
            raise ValueError("data.image_size must be divisible by 8 for diffusion")
        if not isinstance(diffusion["checkpoint"], str) or not diffusion["checkpoint"].strip():
            raise ValueError("diffusion.checkpoint must name a public checkpoint")
        revision = diffusion["revision"]
        if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in revision
        ):
            raise ValueError("diffusion.revision must be a full 40-character hexadecimal commit")
        seed = diffusion["generation_seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("diffusion.generation_seed must be a nonnegative integer")
        steps = diffusion["inference_steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("diffusion.inference_steps must be a positive integer")
        guidance = diffusion["guidance_scale"]
        if type(guidance) not in (int, float) or not math.isfinite(guidance) or guidance <= 0:
            raise ValueError("diffusion.guidance_scale must be positive")
        if not isinstance(diffusion["prompt"], str):
            raise ValueError("diffusion.prompt must be a string")
        if not isinstance(diffusion["cache_dir"], str) or not diffusion["cache_dir"]:
            raise ValueError("diffusion.cache_dir must be a nonempty path string")
    return config


def resolve_path(config_path: str | Path, value: str) -> Path:
    """Configs live in <repo>/configs; absolute user paths are also accepted."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config_path).resolve().parent.parent / path).resolve()
