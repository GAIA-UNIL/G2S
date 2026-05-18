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
di = np.full(ti.shape, np.nan, dtype=np.float32)

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
    "-j",
    1.00001,
    "-s",
    321,
)

fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, image, title in zip(axes, [ti, simulation], ["Strebelle TI", "Native DS"]):
    ax.imshow(image, cmap="tab20")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
