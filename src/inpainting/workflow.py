"""Prepare images and masks, then run the OpenCV inpainting baselines."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from .baselines.opencv import inpaint
from .config import resolve_path
from .data.images import load_rgb, save_image
from .data.masks import MASK_GENERATORS, MASK_SEED_SCHEME, derive_mask_seed, make_mask


OVERVIEW_IMAGE_LIMIT = 3


def select_images(manifest: dict, count: int) -> list[str]:
    """Preview the first count images of the fixed M1 subset."""
    files = manifest["images"]
    if type(count) is not int or not 1 <= count <= len(files):
        raise ValueError("image_count must fit the selected subset")
    selected = files[:count]
    if len(set(selected)) != len(selected):
        raise ValueError("Selected manifest paths contain duplicates")
    return selected


def write_overviews(destination: Path, report: dict) -> list[str]:
    """Show the first few OpenCV comparisons in one sheet per mask type."""
    names = []
    columns = report["comparison_panels"]
    font = ImageFont.load_default(size=17)
    for kind in report["config"]["masks"]["types"]:
        cases = [case for case in report["cases"] if case["mask_type"] == kind][:OVERVIEW_IMAGE_LIMIT]
        sheet = Image.new("RGB", (256 * len(columns), 36 + 280 * len(cases)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, label in enumerate(columns):
            draw.text((256 * index + 8, 8), label, fill="black", font=font)
        for index, case in enumerate(cases):
            top = 36 + 280 * index
            label = f"{case['image_id']} | {case['actual_coverage']:.1%} | {case['status']}"
            draw.text((8, top + 2), label, fill="black", font=font)
            with Image.open(destination / case["artifact_dir"] / "comparison.png") as strip:
                sheet.paste(strip.resize((sheet.width, 256), Image.Resampling.LANCZOS), (0, top + 24))
        name = f"overview-{kind}.png"
        sheet.save(destination / name)
        names.append(name)
    return names


def run_workflow(config: dict, config_path: Path, name: str, *,
                 image_path: str | None = None) -> Path:
    """Prepare each image/mask once and reuse it across the OpenCV methods."""
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("--name must be a single run-folder name")
    settings = config["run"]
    methods = config["opencv"]["methods"]
    root = resolve_path(config_path, config["data"]["root"])
    size = config["data"]["image_size"]
    if image_path:
        source = Path(image_path).resolve()
        image_id = source.relative_to(root).as_posix() if source.is_relative_to(root) else source.name
        inputs = [(image_id, source)]
    else:
        manifest_path = resolve_path(config_path, config["data"]["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = select_images(manifest, settings["image_count"])
        inputs = [(image_id, (root / image_id).resolve()) for image_id in files]
        for image_id, source in inputs:
            if not source.is_relative_to(root) or not source.is_file():
                raise ValueError(f"Image must exist inside the dataset root: {image_id}")

    columns = ["original", "masked", *methods]
    destination = resolve_path(config_path, settings["output_dir"]) / name
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "M1 OpenCV baseline samples",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "methods": methods,
        "selected_images": [image_id for image_id, _ in inputs],
        "selection": "single image" if image_path else "first image_count entries of subset manifest",
        "mask_seed_derivation": MASK_SEED_SCHEME,
        "versions": {"python": sys.version, "opencv": cv2.__version__,
                     "numpy": np.__version__, "Pillow": pillow_version},
        "comparison_panels": columns,
        "cases": [],
    }

    for image_id, source in inputs:
        original = load_rgb(source, size)
        image_key = Path(image_id).stem + "-" + hashlib.sha256(image_id.encode()).hexdigest()[:8]
        for kind in config["masks"]["types"]:
            # All methods share this exact prepared input and binary mask.
            mask_seed = derive_mask_seed(config["masks"]["seed"], image_id, kind)
            mask = make_mask(size, size, kind, mask_seed, config["masks"]["hole_fraction"])
            masked = original.copy()
            masked[mask == 255] = 0
            case_dir = destination / image_key / kind
            save_image(case_dir / "original.png", original)
            save_image(case_dir / "mask.png", mask)
            save_image(case_dir / "masked.png", masked)
            record = {
                "image_id": image_id, "mask_type": kind,
                "mask_generator": MASK_GENERATORS[kind], "mask_seed": mask_seed,
                "actual_coverage": float(np.count_nonzero(mask) / mask.size),
                "artifact_dir": case_dir.relative_to(destination).as_posix(),
                "baseline_seconds": {},
            }
            report["cases"].append(record)
            panels = [original, masked]
            for method in methods:
                start = time.perf_counter()
                prediction = inpaint(masked, mask, method, config["opencv"]["radius"])
                record["baseline_seconds"][method] = time.perf_counter() - start
                save_image(case_dir / f"{method}.png", prediction)
                panels.append(prediction)
            save_image(case_dir / "comparison.png", np.concatenate(panels, axis=1))
            record["status"] = "complete"
            print(f"{image_id} / {kind}: {record['status']}", flush=True)
    report["status"] = "complete"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["overviews"] = write_overviews(destination, report)
    (destination / "run.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return destination
