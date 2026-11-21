# Kriging of the empty cells of a grid from scattered data: estimate and
# kriging variance, with the direct (unique neighbourhood) and the local
# (moving neighbourhood) strategies.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

rng = np.random.default_rng(1)
model = 0.05*cov.nugget() + 0.95*cov.spherical(range=[80, 30], angle=-20)

reference = g2s('-a', 'fftma', '-di', np.full((200, 250), np.nan), '-dt', [0], *model.args(), '-m', 10, '-s', 5)['simulation']
data = np.full(reference.shape, np.nan)
cells = rng.choice(data.size, 300, replace=False)
data.flat[cells] = reference.flat[cells]

direct = g2s('-a', 'kriging', '-di', data, '-dt', [0], *model.args(), '-method', 'direct', '-var')
local = g2s('-a', 'kriging', '-di', data, '-dt', [0], *model.args(), '-method', 'local', '-n', 32, '-os', 8, '-var')
print('largest difference between direct and local:', np.abs(direct['estimate'] - local['estimate']).max())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, image, title in zip(axes, [reference, direct['estimate'], direct['variance']],
                            ['reference', 'ordinary kriging (direct)', 'kriging variance']):
    im = ax.imshow(image)
    ax.plot(cells % data.shape[1], cells // data.shape[1], 'k.', markersize=2)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.show()
