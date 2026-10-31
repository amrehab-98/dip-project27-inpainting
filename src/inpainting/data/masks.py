"""Seeded rectangular and free-form masks. 0 = known, 255 = hole."""

import hashlib
import json

import cv2
import numpy as np


MASK_GENERATORS = {"structured": "rectangle-v1", "random": "free-form-brush-v1"}
MASK_SEED_SCHEME = "sha256-json-v1"


def derive_mask_seed(mask_seed: int, image_id: str, kind: str) -> int:
    """Give each relative image ID/mask kind its own repeatable random sequence.

    JSON separates the fields unambiguously; SHA-256 is stable across processes.
    Include the generator version so deliberate generator changes get new seeds.
    """
    payload = json.dumps(
        [mask_seed, image_id, kind, MASK_GENERATORS[kind]],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _brush_mask(height: int, width: int, rng: np.random.Generator,
                hole_fraction: float) -> np.ndarray:
    """Accumulate rounded brush segments; keep the closest coverage at crossing.

    Coverage is approximate so we never add scattered pixels to hit an exact count.
    Widths are 4-10% and segment lengths 5-20% of the shorter image dimension.
    A stroke has 2-6 segments, with a fresh random start for each stroke.
    """
    # Start with all pixels known; brush strokes will mark missing pixels as 255.
    mask = np.zeros((height, width), dtype=np.uint8)
    # Scale brush geometry by the shorter side so it fits rectangular images too.
    short_side = min(height, width)
    # Convert the requested fraction to pixels, leaving at least one hole and one known pixel.
    target = min(height * width - 1, max(1, round(height * width * hole_fraction)))
    # Track the current hole area so overlapping strokes are not counted twice.
    count = 0
    # Bound work even for unusually high coverage or tiny images.
    max_strokes = 2048
    for _ in range(max_strokes):
        # Choose a new stroke's starting point in OpenCV's (x, y) coordinate order.
        start = (int(rng.integers(width)), int(rng.integers(height)))
        # Choose a brush width of 4-10% of the shorter side, with a one-pixel minimum.
        thickness = max(1, round(short_side * rng.uniform(0.04, 0.10)))
        # Avoid a first brush dab much larger than a very small requested hole.
        thickness = min(thickness, max(1, int((target / np.pi) ** 0.5)))
        # Pick the initial direction in radians; a full turn is 2*pi.
        angle = rng.uniform(0, 2 * np.pi)
        # Draw 2-6 connected segments; the upper bound of rng.integers is exclusive.
        for _ in range(int(rng.integers(2, 7))):
            # Turn by up to 90 degrees either way to imitate a free-hand stroke.
            angle += rng.uniform(-np.pi / 2, np.pi / 2)
            # Choose this segment's length as 5-20% of the shorter image side.
            length = short_side * rng.uniform(0.05, 0.20)
            # Convert direction and length to an endpoint, rounded and clipped inside the image.
            end = (
                int(np.clip(round(start[0] + length * np.cos(angle)), 0, width - 1)),
                int(np.clip(round(start[1] + length * np.sin(angle)), 0, height - 1)),
            )
            # Draw on a copy so we can keep the previous mask if this segment overshoots.
            candidate = mask.copy()
            # Use a solid line without antialiasing to preserve binary values of 0 and 255.
            cv2.line(candidate, start, end, 255, thickness, lineType=cv2.LINE_8)
            # Fill circles at both ends for rounded caps; -1 means a filled circle.
            for point in (start, end):
                cv2.circle(candidate, point, thickness // 2, 255, -1,
                           lineType=cv2.LINE_8)
            # Count the union of all missing pixels, including this proposed segment.
            new_count = np.count_nonzero(candidate)
            # Stop when this segment reaches or crosses the requested hole area.
            if new_count >= target:
                # Keep a nonempty previous mask if the candidate hides everything or is no closer.
                if count and (new_count == height * width
                              or target - count <= new_count - target):
                    return mask
                # Otherwise keep the candidate, whose area is closer to the target.
                return candidate
            # Accept the segment and continue this stroke from its endpoint.
            mask, count = candidate, new_count
            start = end
    # Report failure rather than silently returning a mask far below the target area.
    raise ValueError("Could not reach requested brush-mask coverage")


def make_mask(
    height: int,
    width: int,
    kind: str,
    seed: int,
    hole_fraction: float = 0.25,
) -> np.ndarray:
    """Create a rectangle (structured) or random free-form brush strokes.

    hole_fraction is nominal: rectangles round to pixels, brushes stop at the
    closest area before/after crossing the target. Save actual coverage per mask.
    """
    # Require room for the bounded hole dimensions and a fraction between empty and full.
    if height < 2 or width < 2 or not 0 < hole_fraction < 1:
        raise ValueError("Mask needs dimensions >= 2 and 0 < hole_fraction < 1")
    # Seed a local generator so the same inputs reproduce the same random choices.
    rng = np.random.default_rng(seed)
    # Start with all pixels known; uint8 stores our mask values of 0 and 255.
    mask = np.zeros((height, width), dtype=np.uint8)
    if kind == "structured":
        # Scale height by sqrt(area fraction), round to pixels, and keep 1 <= hole_h < height.
        hole_h = min(height - 1, max(1, round(height * hole_fraction**0.5)))
        # Apply the same sqrt scaling and bounds to width, targeting the requested area fraction.
        hole_w = min(width - 1, max(1, round(width * hole_fraction**0.5)))
        # Include the last valid top position using +1 because the random upper bound is exclusive.
        top = int(rng.integers(0, height - hole_h + 1))
        # Include the last valid left position while keeping the entire hole inside the image.
        left = int(rng.integers(0, width - hole_w + 1))
        # Mark the chosen rectangle as missing; each slice stops just before its end index.
        mask[top : top + hole_h, left : left + hole_w] = 255
    elif kind == "random":
        mask = _brush_mask(height, width, rng, hole_fraction)
    else:
        raise ValueError(f"Unknown mask type: {kind}")
    return mask
