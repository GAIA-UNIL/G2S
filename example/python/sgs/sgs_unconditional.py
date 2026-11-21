# Sequential Gaussian simulation: four unconditional realisations in one run.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

model = 0.1*cov.nugget() + 0.9*cov.spherical(range=[60, 20], angle=30)

result = g2s('-a', 'sgs',
             '-di', np.full((200, 300), np.nan),  # the grid: every cell is simulated
             '-dt', [0],
             *model.args(),                       # '-cov', '0.1*nugget + 0.9*spherical(...)'
             '-n', 30,                            # neighbours per node
             '-nr', 4,                            # realisations on the same path
             '-s', 100,
             '-j', 0.5)
simulation = result['simulation']    # 200 x 300 x 4
neighbours = result['indexmap']      # neighbours used at each node

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for r, ax in enumerate(axes):
    ax.imshow(simulation[:, :, r])
    ax.set_title('realisation %d' % r)
    ax.axis('off')
plt.tight_layout()
plt.show()
