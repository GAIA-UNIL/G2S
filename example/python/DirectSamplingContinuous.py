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
di = np.full((180, 180), np.nan, dtype=np.float32)
di[::40, ::40] = ti[: di.shape[0] : 40, : di.shape[1] : 40]

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
    "-s",
    123,
)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, image, title in zip(axes, [ti, di, simulation], ["Stone TI", "Conditioning", "Native DS"]):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
