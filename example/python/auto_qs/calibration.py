import numpy as np
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s


ti = np.array(Image.open(BytesIO(requests.get(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
).content)), dtype=np.float32)

result = g2s(
    "-a", "autoQS",
    "-ti", ti,
    "-dt", [0],
    "-n", 40,
    "-k", 1.2,
    "-j", 0.5,
)

print("job", result["job_id"])
print("mean error", result["mean_error"].shape)
print("deviation error", result["deviation_error"].shape)
print("sample count", result["sample_count"].shape)
