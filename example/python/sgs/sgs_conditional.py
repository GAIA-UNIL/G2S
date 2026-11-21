# Sequential Gaussian simulation conditioned to hard data, with simple and
# ordinary kriging, compared with the kriging estimate.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

rng = np.random.default_rng(0)
model = cov.exponential(range=40)

# Hard data: 150 values taken from a reference field (normal scores, mean 0)
reference = g2s('-a', 'fftma', '-di', np.full((150, 150), np.nan), '-dt', [0], *model.args(), '-s', 3)['simulation']
data = np.full((150, 150), np.nan)
cells = rng.choice(data.size, 150, replace=False)
data.flat[cells] = reference.flat[cells]

sk = g2s('-a', 'sgs', '-di', data, '-dt', [0], *model.args(), '-kt', 'sk', '-m', 0,
         '-n', 30, '-nr', 50, '-s', 1, '-j', 0.5)['simulation']
ok = g2s('-a', 'sgs', '-di', data, '-dt', [0], *model.args(), '-kt', 'ok',
         '-n', 30, '-nr', 50, '-s', 1, '-j', 0.5)['simulation']
estimate = g2s('-a', 'kriging', '-di', data, '-dt', [0], *model.args(), '-kt', 'sk')['estimate']

print('hard data honoured:', np.allclose(sk.reshape(-1, 50)[cells], data.flat[cells][:, None]))

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, image, title in zip(axes, [reference, sk[:, :, 0], sk.mean(axis=2), estimate],
                            ['reference', 'SGS (SK), one realisation', 'mean of 50 SGS (SK)', 'simple kriging']):
    ax.imshow(image, vmin=-2.5, vmax=2.5)
    ax.plot(cells % 150, cells // 150, 'k.', markersize=2)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()
