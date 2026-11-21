# A covariance model whose parameters vary in space: the range grows from
# left to right and the anisotropy direction turns from top to bottom.
# Each node uses the model with the parameter values at that node (local
# stationarity).
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

ny, nx = 200, 300
rows, columns = np.mgrid[0:ny, 0:nx]
range_map = 10 + 40*columns/(nx - 1)    # practical range of the major axis, 10 to 50 cells
angle_map = -60 + 120*rows/(ny - 1)     # degrees, -60 at the top to +60 at the bottom

L = cov.var('L', range_map)
model = 0.05*cov.nugget() + 0.95*cov.exponential(range=[L, L/4], angle=angle_map)
print(str(model))  # 0.05*nugget + 0.95*exponential(range=[L, L/4], angle=_a1)

result = g2s('-a', 'sgs',
             '-di', np.full((ny, nx), np.nan),
             '-dt', [0],
             *model.args(),        # -cov text and -cvar L range_map -cvar _a1 angle_map
             '-n', 40,
             '-s', 42,
             '-j', 0.5)
simulation = result['simulation']

# The covariance at one node, as covEval resolves it (pos = [column, row])
print(model([[0, 0], [10, 0]], pos=[nx - 1, ny // 2]))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, image, title in zip(axes, [range_map, angle_map, simulation], ['range (cells)', 'angle (degrees)', 'sgs']):
    im = ax.imshow(image)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.show()
