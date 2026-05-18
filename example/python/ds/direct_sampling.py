import numpy as np
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s


ti = np.array(Image.open(BytesIO(requests.get(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
).content)), dtype=np.float32)

result = g2s(
    "-a", "ds",
    "-ti", ti,
    "-di", np.full(ti.shape, np.nan, dtype=np.float32),
    "-dt", [0],
    "-th", 0.05,
    "-f", 0.30,
    "-n", 30,
    "-s", 100,
    "-j", 1.00001,
)

print("job", result["job_id"], "seconds", result.get("time"))
print("simulation", result["simulation"].shape, "indexmap", result["indexmap"].shape)
