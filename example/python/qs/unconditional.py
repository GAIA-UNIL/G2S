import numpy as np
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s


ti = np.array(Image.open(BytesIO(requests.get(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
).content)), dtype=np.float32)

result = g2s(
    "-a", "qs",
    "-ti", ti,
    "-di", np.full((200, 200), np.nan, dtype=np.float32),
    "-dt", [0],
    "-k", 1.2,
    "-n", 50,
    "-j", 0.5,
)

simulation = result["simulation"]
indexmap = result["indexmap"]
print("job", result["job_id"], "seconds", result.get("time"))
print("simulation shape", simulation.shape, "index shape", indexmap.shape)
