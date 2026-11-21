# FFT-MA conditioned to hard data by kriging of the residuals, with the
# direct (all data) and the local (moving neighbourhood) kriging methods.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

rng = np.random.default_rng(2)
model = 0.1*cov.nugget() + 0.9*cov.exponential(range=60)
grid = np.full((200, 200), np.nan)

reference = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-s', 9)['simulation']
data = grid.copy()
cells = rng.choice(data.size, 400, replace=False)
data.flat[cells] = reference.flat[cells]

direct = g2s('-a', 'fftma', '-di', data, '-dt', [0], *model.args(), '-nr', 20, '-s', 1, '-j', 0.5)
local = g2s('-a', 'fftma', '-di', data, '-dt', [0], *model.args(), '-nr', 20, '-s', 1,
            '-km', 'local', '-n', 40, '-j', 0.5)
print('kriging method chosen by auto:', direct['kriging_method'])
print('data honoured:', np.allclose(direct['simulation'].reshape(-1, 20)[cells], data.flat[cells][:, None]))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, image, title in zip(axes, [reference, direct['simulation'][:, :, 0], direct['simulation'].std(axis=2)],
                            ['reference', 'conditional fftma', 'std of 20 realisations']):
    ax.imshow(image)
    ax.plot(cells % 200, cells // 200, 'k.', markersize=1)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
