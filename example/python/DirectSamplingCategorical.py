from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from g2s import g2s


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


ti = load_tiff("https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff")
di = np.full((180, 180), np.nan, dtype=np.float32)
di[::45, ::45] = ti[: di.shape[0] : 45, : di.shape[1] : 45]

simulation, index, *_ = g2s(
    "-a",
    "DirectSampling",
    "-ti",
    ti,
    "-di",
    di,
    "-dt",
    [1],
    "-th",
    0.12,
    "-f",
    0.4,
    "-n",
    48,
    "-s",
    321,
)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, image, title in zip(axes, [ti, di, simulation], ["Strebelle TI", "Conditioning", "Native DS"]):
    ax.imshow(image, cmap="tab20")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
