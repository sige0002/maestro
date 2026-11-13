"""Depth Anything RGB-only client; simulator depth is never a fallback."""

import io

import numpy as np
import requests
from PIL import Image


def infer_depth(rgb, settings):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    response = requests.post(
        settings.get("url", "http://127.0.0.1:8007") + "/depth",
        files={"image": ("rgb.png", buffer.getvalue(), "image/png")},
        timeout=(10, 180),
    )
    response.raise_for_status()
    with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
        depth = data["depth"].astype(np.float32)
        metadata = {
            "source": "Depth_Anything_V2_metric_RGB_only",
            "model": str(data["model"]),
            "revision": str(data["revision"]),
            "inference_s": float(data["elapsed_s"]),
        }
    if (
        depth.shape != rgb.shape[:2]
        or not np.isfinite(depth).all()
        or np.any(depth <= 0)
    ):
        raise ValueError("Depth Anything returned invalid metric depth")
    scale = float(settings.get("metric_scale", 1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("metric_scale must be positive and finite")
    metadata["metric_scale"] = scale
    metadata["calibration_note"] = settings.get("calibration_note", "uncalibrated")
    return depth * scale, metadata
