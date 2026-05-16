import numpy as np
from PIL import Image
from urllib.request import urlopen
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


ti = load_tiff(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff"
)
di = np.full((180, 180), np.nan, dtype=np.float32)

yy, xx = np.indices(di.shape, dtype=np.float32)
center_y = (di.shape[0] - 1) / 2.0
center_x = (di.shape[1] - 1) / 2.0
radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
radius = radius / np.nanmax(radius)

rotation_map = (radius * np.pi / 2.0).astype(np.float32)
scale_map = (0.85 + 0.35 * radius).astype(np.float32)

simulation, *_ = g2s(
    "-a", "qs",
    "-ti", ti,
    "-di", di,
    "-dt", [0],
    "-k", 1.2,
    "-n", 40,
    "-s", 123,
    "-rmi", rotation_map,
    "-smi", scale_map,
)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(ti, cmap="gray")
axes[0].set_title("Training image")
axes[1].imshow(rotation_map, cmap="twilight")
axes[1].set_title("Rotation")
axes[2].imshow(scale_map, cmap="viridis")
axes[2].set_title("Scale")
axes[3].imshow(simulation, cmap="gray")
axes[3].set_title("Simulation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
