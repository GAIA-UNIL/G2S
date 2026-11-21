# Turning bands: a nested model with a nugget, a spherical and a Gaussian
# structure (each structure is simulated on its own lines).
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

model = 0.1*cov.nugget() + 0.6*cov.spherical(range=[80, 30], angle=30) + 0.3*cov.gaussian(range=15)

result = g2s('-a', 'tb',
             '-di', np.full((200, 300), np.nan),
             '-dt', [0],
             *model.args(),
             '-nb', 1000,   # lines per structure
             '-nr', 3,
             '-m', 5,       # mean
             '-s', 7,
             '-j', 0.5)
simulation = result['simulation']  # 200 x 300 x 3

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for r, ax in enumerate(axes):
    ax.imshow(simulation[:, :, r])
    ax.set_title('realisation %d' % r)
    ax.axis('off')
plt.tight_layout()
plt.show()
