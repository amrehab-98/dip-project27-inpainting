"""M1 subset preparation, mask generation, and OpenCV inpainting samples."""

import argparse
import json
from pathlib import Path
import sys

from .config import load_config, resolve_path
from .data.manifests import build_manifest
from .workflow import run_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", help="Write a seeded subset manifest; no image copies")
    run = sub.add_parser("run", help="Run OpenCV baselines on shared images and masks")
    run.add_argument("--image", help="Use one local image instead of the configured manifest preview")
    run.add_argument("--name", default="preview", help="New output folder; existing runs are preserved")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare-data":
            data = config["data"]
            target = resolve_path(args.config, data["manifest_path"])
            if target.exists():
                raise FileExistsError(f"Manifest already exists; choose a new manifest_path: {target}")
            manifest = build_manifest(
                resolve_path(args.config, data["root"]), data["subset_size"],
                data["seed"], data["source"],
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            print(f"Saved {data['subset_size']} selected paths to {target}")
        else:
            destination = run_workflow(
                config, args.config, args.name, image_path=args.image,
            )
            print(f"Run artifacts: {destination}")
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        print(f"{args.command}: {error}", file=sys.stderr)
        return 2
    return 0
