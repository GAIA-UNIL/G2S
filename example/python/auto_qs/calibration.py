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
mean_error = result.get("mean_error", result.get("simulation"))
if mean_error is not None:
    print("mean error", mean_error.shape)
if "deviation_error" in result:
    print("deviation error", result["deviation_error"].shape)
if "sample_count" in result:
    print("sample count", result["sample_count"].shape)
