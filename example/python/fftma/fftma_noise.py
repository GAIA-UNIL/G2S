# FFT-MA with a given white noise: the noise of two realisations is returned
# (-on) and combined, as in gradual deformation, to make a continuous family
# of realisations that all have the model covariance.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

model = cov.spherical(range=40)
grid = np.full((150, 150), np.nan)

result = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-nr', 2, '-on', '-s', 4)
noise = result['noise']  # white noise on the extended grid, one layer per realisation
print('simulation', result['simulation'].shape, 'noise', noise.shape)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, t in zip(axes, np.linspace(0, np.pi/2, 5)):
    w = np.cos(t)*noise[:, :, 0] + np.sin(t)*noise[:, :, 1]  # still N(0, 1) white noise
    field = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-ni', w)['simulation']
    ax.imshow(field, vmin=-2.5, vmax=2.5)
    ax.set_title('t = %.2f' % t)
    ax.axis('off')
plt.tight_layout()
plt.show()

# t = 0 gives back the first realisation
first = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-ni', noise[:, :, 0])['simulation']
print('identical to realisation 0:', np.array_equal(first, result['simulation'][:, :, 0]))

# the whole noise (both realisations, one per layer) can be given back at once
both = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-ni', noise)['simulation']
print('both realisations identical:', np.array_equal(both, result['simulation']))
