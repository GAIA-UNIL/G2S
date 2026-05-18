from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from g2s import g2s


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


ti = load_tiff("https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff")
di = np.full(ti.shape, np.nan, dtype=np.float32)

yy, xx = np.indices(di.shape, dtype=np.float32)
cy = (di.shape[0] - 1) / 2.0
cx = (di.shape[1] - 1) / 2.0
radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
radius = radius / np.nanmax(radius)

rotation_map = (radius * np.pi / 3.0).astype(np.float32)
rotation_tolerance = np.full(di.shape, np.deg2rad(5.0), dtype=np.float32)
scale_map = (0.9 + 0.25 * radius).astype(np.float32)
scale_tolerance = np.full(di.shape, 0.05, dtype=np.float32)

simulation, index, *_ = g2s(
    "-a",
    "DS",
    "-ti",
    ti,
    "-di",
    di,
    "-dt",
    [0],
    "-th",
    0.08,
    "-f",
    0.35,
    "-n",
    40,
    "-j",
    1.00001,
    "-s",
    456,
    "-rmi",
    rotation_map,
    "-rti",
    rotation_tolerance,
    "-smi",
    scale_map,
    "-sti",
    scale_tolerance,
)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, image, title, cmap in zip(
    axes,
    [ti, rotation_map, scale_map, simulation],
    ["Stone TI", "Rotation center", "Scale center", "Native DS"],
    ["gray", "twilight", "viridis", "gray"],
):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
