from io import BytesIO
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from g2s import g2s


SIM_SHAPE = (500, 500)
SEED = 123
OUTPUT_FIGURE = "qs_scale_2d_500.png"


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


ti = load_tiff(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff"
)
di = np.full(SIM_SHAPE, np.nan, dtype=np.float32)

x = np.linspace(0.5, 2, di.shape[1], dtype=np.float32)[None, :]
scale_map = np.repeat(x, di.shape[0], axis=0)

base_args = (
    "-a", "qs",
    "-ti", ti,
    "-di", di,
    "-dt", [1],
    "-k", 2.4,
    "-n", 30,
    "-j", 1.0001,
    "-s", SEED,
)

print("Running 500x500 baseline QS simulation...")
_schema_result = g2s(*base_args)
baseline = _schema_result["simulation"]
print("Running 500x500 QS simulation with scale range 0.25 to 3.5...")
_schema_result = g2s(
    *base_args,
    "-smi", scale_map)
simulation = _schema_result["simulation"]

changed = np.asarray(simulation != baseline, dtype=np.float32)
change_ratio = float(np.mean(changed))

fig, axes_grid = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
axes = axes_grid.ravel()
axes[0].imshow(ti, cmap="gray")
axes[0].set_title("Training image")
axes[1].imshow(scale_map, cmap="viridis")
axes[1].set_title("Scale")
axes[2].imshow(baseline, cmap="gray")
axes[2].set_title("Baseline")
axes[3].imshow(simulation, cmap="gray")
axes[3].set_title("Scaled search pattern")
axes[4].imshow(changed, cmap="magma", vmin=0, vmax=1)
axes[4].set_title(f"Changed cells: {change_ratio:.1%}")
axes[5].axis("off")
for ax in axes:
    ax.axis("off")
plt.savefig(OUTPUT_FIGURE, dpi=160)
print(f"Saved comparison figure to {OUTPUT_FIGURE}")
plt.show()
