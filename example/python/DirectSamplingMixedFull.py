from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from g2s import g2s


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


categorical = load_tiff("https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff")
continuous = (
    categorical
    + np.roll(categorical, 1, axis=0)
    + np.roll(categorical, -1, axis=0)
    + np.roll(categorical, 1, axis=1)
    + np.roll(categorical, -1, axis=1)
) / 5.0
continuous = (continuous - np.nanmin(continuous)) / (np.nanmax(continuous) - np.nanmin(continuous))

ti = np.stack([categorical, continuous.astype(np.float32)], axis=-1).astype(np.float32)
di = np.full((180, 180, 2), np.nan, dtype=np.float32)
di[::45, ::45, :] = ti[: di.shape[0] : 45, : di.shape[1] : 45, :]

simulation, index, *_ = g2s(
    "-a",
    "ds",
    "-ti",
    ti,
    "-di",
    di,
    "-dt",
    [1, 0],
    "-th",
    0.15,
    "-f",
    0.4,
    "-n",
    48,
    "-cnorm",
    2.0,
    "-fs",
    "-s",
    789,
)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
images = [ti[..., 0], ti[..., 1], simulation[..., 0], simulation[..., 1]]
titles = ["Categorical TI", "Derived continuous TI", "DS category", "DS continuous"]
cmaps = ["tab20", "viridis", "tab20", "viridis"]
for ax, image, title, cmap in zip(axes, images, titles, cmaps):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
