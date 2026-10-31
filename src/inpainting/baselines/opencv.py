"""OpenCV Telea and Navier-Stokes with explicit RGB and binary-mask contracts."""

import cv2
import numpy as np


def inpaint(
    image: np.ndarray, mask: np.ndarray, method: str = "telea", radius: float = 3.0
) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB uint8 with shape (H, W, 3)")
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
        raise ValueError("mask must be uint8 with shape (H, W)")
    if not np.isin(mask, [0, 255]).all():
        raise ValueError("mask must use 0 for known pixels and 255 for holes")
    flags = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
    if method not in flags or radius <= 0:
        raise ValueError("method must be telea or ns and radius must be positive")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    prediction = cv2.cvtColor(
        cv2.inpaint(bgr, mask, radius, flags[method]), cv2.COLOR_BGR2RGB
    )
    return prediction
