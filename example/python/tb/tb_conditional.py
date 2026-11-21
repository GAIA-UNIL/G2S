# Turning bands in 3D, conditioned to vertical wells by kriging of the residuals.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

rng = np.random.default_rng(3)
nz, ny, nx = 40, 100, 100
# range=[x, y, z]: component 0 is the last numpy axis (x), component 2 the first (z)
model = cov.exponential(range=[60, 30, 8], angle=[20, 0, 0])

data = np.full((nz, ny, nx), np.nan)
for _ in range(12):  # 12 vertical wells
    y, x = rng.integers(0, ny), rng.integers(0, nx)
    data[:, y, x] = rng.normal(size=nz)

result = g2s('-a', 'tb', '-di', data, '-dt', [0], *model.args(),
             '-nr', 2, '-km', 'local', '-n', 40, '-s', 1, '-j', 0.5)
simulation = result['simulation']  # 40 x 100 x 100 x 2
print('data honoured:', np.allclose(simulation[..., 0][~np.isnan(data)], data[~np.isnan(data)]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(simulation[nz // 2, :, :, 0])
axes[0].set_title('horizontal slice')
axes[1].imshow(simulation[:, ny // 2, :, 0], aspect=2)
axes[1].set_title('vertical section')
plt.tight_layout()
plt.show()
