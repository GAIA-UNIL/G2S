# Unconditional simulation with FFT-MA and with the spectral method, and a
# periodic field (-csim) that tiles without seams.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

model = cov.matern(range=[50, 20], angle=60, nu=1.5)
grid = np.full((256, 384), np.nan)

fftma = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-nr', 2, '-s', 1, '-j', 0.5)
spectral = g2s('-a', 'spectral', '-di', grid, '-dt', [0], *model.args(), '-nr', 2, '-s', 1, '-j', 0.5)
periodic = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-csim', '-s', 1, '-j', 0.5)
print('extended grid used by fftma:', fftma['extended_grid'])

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
axes[0].imshow(fftma['simulation'][:, :, 0])
axes[0].set_title('fftma')
axes[1].imshow(spectral['simulation'][:, :, 0])
axes[1].set_title('spectral')
axes[2].imshow(np.tile(periodic['simulation'], (1, 2)))
axes[2].set_title('fftma -csim, tiled twice')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
