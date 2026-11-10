from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from g2s import g2s


SIM_SHAPE = (240, 240)
SEED = 123
ANGLE = 0.5 * np.pi
OUTPUT_FIGURE = "qs_rotation_equivalence_2d.png"


def load_tiff(url):
    with urlopen(url) as response:
        return np.array(Image.open(BytesIO(response.read())), dtype=np.float32)


def run_qs(ti, rotation_map=None):
    args = [
        "-a", "qs",
        "-ti", ti,
        "-di", np.full(SIM_SHAPE, np.nan, dtype=np.float32),
        "-dt", [1],
        "-k", 2.4,
        "-n", 80,
        "-j", 1.0001,
        "-s", SEED,
    ]
    if rotation_map is not None:
        args.extend(["-rmi", rotation_map])
    _schema_result = g2s(*args)
    simulation = _schema_result["simulation"]
    return simulation


ti = load_tiff(
    "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff"
)
rotation_map = np.full(SIM_SHAPE, ANGLE, dtype=np.float32)

print("Running QS with constant +pi/2 search-pattern rotation...")
sim_rmi = run_qs(ti, rotation_map)

print("Running QS with counter-clockwise rotated TI and no rotation map...")
ti_ccw = np.rot90(ti, k=1)
sim_ti_ccw = run_qs(ti_ccw)

print("Running QS with clockwise rotated TI and no rotation map...")
ti_cw = np.rot90(ti, k=-1)
sim_ti_cw = run_qs(ti_cw)

diff_ccw = np.asarray(sim_rmi != sim_ti_ccw, dtype=np.float32)
diff_cw = np.asarray(sim_rmi != sim_ti_cw, dtype=np.float32)
ratio_ccw = float(np.mean(diff_ccw))
ratio_cw = float(np.mean(diff_cw))
closer = "counter-clockwise" if ratio_ccw < ratio_cw else "clockwise"

print(f"Difference to counter-clockwise TI run: {ratio_ccw:.1%}")
print(f"Difference to clockwise TI run: {ratio_cw:.1%}")
print(f"Closest rotated-TI comparison: {closer}")

fig, axes_grid = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
axes = axes_grid.ravel()
axes[0].imshow(ti, cmap="gray")
axes[0].set_title("Original TI")
axes[1].imshow(ti_ccw, cmap="gray")
axes[1].set_title("TI rotated CCW")
axes[2].imshow(ti_cw, cmap="gray")
axes[2].set_title("TI rotated CW")
axes[3].imshow(sim_rmi, cmap="gray")
axes[3].set_title("QS with +pi/2 RMI")
axes[4].imshow(sim_ti_ccw, cmap="gray")
axes[4].set_title(f"CCW TI run, diff {ratio_ccw:.1%}")
axes[5].imshow(sim_ti_cw, cmap="gray")
axes[5].set_title(f"CW TI run, diff {ratio_cw:.1%}")
for ax in axes:
    ax.axis("off")
plt.savefig(OUTPUT_FIGURE, dpi=160)
print(f"Saved comparison figure to {OUTPUT_FIGURE}")
plt.show()
