"""Offline diagnostic for the fixed mixed scene only; truth is evaluation-only.
Usage: python scripts/so101-depth-diagnostic.py SAVED_TRIAL OUTPUT_DIRECTORY
Table pixels and plane height are specific to the existing fixed overview camera.
The fitted scale is NOT written to runtime configuration automatically.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maestro.so101.monocular_depth import infer_depth

root = Path(sys.argv[1])
p = min(root.glob("scan-*.npz"))
s = np.load(p)
rgb = np.array(Image.open(p.with_suffix(".png")))
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
pred, meta = infer_depth(rgb, {})
np.save(output / "raw.npy", pred)
Image.fromarray(rgb).save(output / "rgb.png")
k = np.array(s["intrinsics"]).reshape(3, 3)
cam = s["camera_pose"]
refs = [
    [300, 430],
    [450, 420],
    [520, 340],
    [550, 250],
    [200, 280],
    [220, 320],
    [350, 450],
    [460, 340],
]
ratios = []
expected = []
observed = []
for x, y in refs:
    ray = cam[:3, :3] @ np.linalg.solve(k, [x, y, 1])
    z = (-0.00240025973226284 - cam[2, 3]) / ray[2]
    d = float(np.median(pred[y - 2 : y + 3, x - 2 : x + 3]))
    ratios.append(z / d)
    expected.append(z)
    observed.append(d)
scale = float(np.median(ratios))
cal = pred * scale
report = {
    "model": meta,
    "raw_range_m": [float(pred.min()), float(pred.max())],
    "table_reference_pixels": refs,
    "scale": scale,
    "table_reference_residual_m": (np.array(observed) * scale - expected).tolist(),
    "object_depth_error_m": {},
}
for name, (x, y) in {
    "red": (440, 291),
    "green": (280, 355),
    "blue": (382, 362),
}.items():
    report["object_depth_error_m"][name] = {
        "raw": float(pred[y, x] - s["depth"][y, x]),
        "scaled": float(cal[y, x] - s["depth"][y, x]),
        "truth": float(s["depth"][y, x]),
        "predicted": float(cal[y, x]),
    }
np.savez(
    output / "comparison.npz",
    prediction=cal,
    raw=pred,
    truth=s["depth"],
    camera=cam,
    intrinsics=k,
)
(output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
