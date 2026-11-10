import numpy as np
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s


ti = np.array(Image.open(BytesIO(requests.get(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
).content)), dtype=np.float32)

submitted = g2s(
    "-a", "qs",
    "-submitOnly",
    "-ti", ti,
    "-di", np.full((200, 200), np.nan, dtype=np.float32),
    "-dt", [0],
    "-k", 1.2,
    "-n", 50,
    "-j", 0.5,
)

job_id = submitted["job_id"]
status = g2s("-statusOnly", job_id)
result = g2s("-waitAndDownload", job_id)

print("submitted", job_id)
print("status", status.get("status"), status.get("progress"))
print("finished", result["status"], result["simulation"].shape)
