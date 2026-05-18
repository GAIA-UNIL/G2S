import numpy as np
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s


ti = np.array(Image.open(BytesIO(requests.get(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/Strebelle.tiff"
).content)), dtype=np.float32)

result = g2s(
    "-a", "snesim",
    "-ti", ti,
    "-di", np.full(ti.shape, np.nan, dtype=np.float32),
    "-dt", [1],
    "-n", 24,
    "-s", 100,
)

print("job", result["job_id"], "seconds", result.get("time"))
print("simulation", result["simulation"].shape)
