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

simulation, index, *_ = g2s(
    "-a",
    "ds",
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
    123,
)

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, image, title in zip(axes, [ti, simulation], ["Stone TI", "Native DS"]):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
