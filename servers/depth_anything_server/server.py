"""RGB-only metric Depth Anything V2 service. No simulator/depth inputs."""

import io
import os
import threading
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
REVISION = "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL, revision=REVISION, use_fast=False)
model = (
    AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION)
    .to(device)
    .eval()
)
lock = threading.Lock()
app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model": MODEL,
        "revision": REVISION,
        "device": device,
        "units": "metres",
        "input": "RGB_only",
    }


@app.post("/depth")
def infer(image: UploadFile):
    rgb = Image.open(io.BytesIO(image.file.read())).convert("RGB")
    start = time.monotonic()
    with lock, torch.inference_mode():
        inputs = processor(images=rgb, return_tensors="pt").to(device)
        output = model(**inputs).predicted_depth
        depth = (
            torch.nn.functional.interpolate(
                output.unsqueeze(1),
                size=rgb.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .float()
            .cpu()
            .numpy()
        )
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        depth=depth,
        model=MODEL,
        revision=REVISION,
        elapsed_s=time.monotonic() - start,
    )
    return Response(buffer.getvalue(), media_type="application/octet-stream")


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ.get("DEPTH_ANYTHING_PORT", "8007"))
    )
