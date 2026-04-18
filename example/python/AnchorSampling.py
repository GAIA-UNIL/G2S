import numpy
import matplotlib.pyplot as plt
from g2s import g2s

# Two aligned training-image stacks with different center values.
ti1 = numpy.ones((21, 21), dtype=float)
ti2 = 2.0 * numpy.ones((21, 21), dtype=float)
ti1[10, 10] = 5.0
ti2[10, 10] = 9.0

# Conditioning data around the center matches TI 1.
di = numpy.nan * numpy.ones((21, 21), dtype=float)
di[9, 10] = 1.0
di[10, 9] = 1.0
di[10, 11] = 1.0
di[11, 10] = 1.0

# Force the center to be simulated first.
path = 100.0 + numpy.arange(di.size, dtype=float).reshape(di.shape)
path[10, 10] = 0.0

# Optional TI-selection mask with one weight per TI at each location.
mask = numpy.zeros(di.shape + (2,), dtype=float)
mask[..., 0] = 1.0
mask[..., 1] = 0.2

simulation, selected_ti, *_ = g2s(
    "-a", "as",
    "-ti", [ti1, ti2],
    "-di", di,
    "-sp", path,
    "-mi", mask,
    "-dt", [0],
    "-k", 2,
    "-n", 20,
    "-j", 0.5,
)

fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True, sharey=True)
fig.suptitle("Anchor Sampling")
for ax, image, title in zip(
    axes,
    [ti1, ti2, di, simulation, selected_ti],
    ["TI 1", "TI 2", "Conditioning", "AS result", "Selected TI id"],
):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
plt.show()
