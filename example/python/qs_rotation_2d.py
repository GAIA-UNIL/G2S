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

y = np.linspace(-1.0, 1.0, di.shape[0], dtype=np.float32)[:, None]
rotation_map = (0.5 * np.pi * y) * np.ones(di.shape, dtype=np.float32)

simulation, *_ = g2s(
    "-a", "qs",
    "-ti", ti,
    "-di", di,
    "-dt", [0],
    "-k", 1.2,
    "-n", 40,
    "-s", 123,
    "-rmi", rotation_map,
)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(ti, cmap="gray")
axes[0].set_title("Training image")
axes[1].imshow(rotation_map, cmap="twilight")
axes[1].set_title("Rotation radians")
axes[2].imshow(simulation, cmap="gray")
axes[2].set_title("Simulation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
