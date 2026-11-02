"""Frozen SD inpainting; GPU dependencies are loaded only on construction."""

from pathlib import Path

import numpy as np
from PIL import Image


class FrozenDiffusionInpainter:
    def __init__(self, checkpoint: str, revision: str, *, cache_dir: Path,
                 steps: int = 20, guidance_scale: float = 7.5, prompt: str = ""):
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("The diffusion method requires a CUDA GPU.")
        if not revision or len(revision) != 40:
            raise ValueError("Pin the checkpoint to a full 40-character commit revision.")
        self.checkpoint = checkpoint
        self.revision = revision
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            checkpoint, revision=revision, variant="fp16", dtype=torch.float16,
            use_safetensors=True, cache_dir=str(cache_dir), token=False,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        for component in self.pipe.components.values():
            if isinstance(component, torch.nn.Module):
                component.requires_grad_(False)
                component.eval()
        self.trainable_parameters = sum(
            p.numel() for component in self.pipe.components.values()
            if isinstance(component, torch.nn.Module)
            for p in component.parameters() if p.requires_grad
        )
        self.last_safety_filtered = None

    def predict(self, image: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
        """Return RGB uint8, preserving known pixels; use the shared mask convention."""
        import torch

        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be RGB uint8 with shape (H, W, 3)")
        if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
            raise ValueError("mask must be uint8 with shape (H, W)")
        if not np.isin(mask, [0, 255]).all():
            raise ValueError("mask must use 0 for known pixels and 255 for holes")
        result = self.pipe(
            prompt=self.prompt, negative_prompt="", image=Image.fromarray(image),
            mask_image=Image.fromarray(mask), height=image.shape[0], width=image.shape[1],
            num_inference_steps=self.steps, guidance_scale=self.guidance_scale,
            strength=1.0, generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        prediction = np.array(result.images[0].convert("RGB"))
        flags = result.nsfw_content_detected
        self.last_safety_filtered = bool(flags[0]) if flags is not None else None
        return np.where(mask[..., None] == 255, prediction, image)
