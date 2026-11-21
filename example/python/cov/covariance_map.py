# A covariance given as a table (map): a damped hole effect, which the
# parametric structures cannot express. The table is a valid (positive
# definite) covariance: the product of an exponential covariance and the
# covariance of a random-phase cosine wave, sampled far enough for the
# exponential to have vanished at the edge of the table.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

half = 80                                     # lags -80..80; lag 0 at [80, 80] = shape//2
dy, dx = np.mgrid[-half:half + 1, -half:half + 1]  # dx: column shift, dy: row shift
direction = np.radians(30)
table = np.exp(-np.hypot(dx, dy)/10) * np.cos(2*np.pi*(dx*np.cos(direction) + dy*np.sin(direction))/25)

model = cov.map(cov.var('C', table))
print(str(model))                              # 1*map(C)
print(model([[0, 0], [12, 0], [25, 0]]))       # read from the table by covEval

grid = np.full((200, 200), np.nan)
sgs = g2s('-a', 'sgs', '-di', grid, '-dt', [0], *model.args(), '-n', 40, '-s', 2, '-j', 0.5)
fftma = g2s('-a', 'fftma', '-di', grid, '-dt', [0], *model.args(), '-s', 2, '-j', 0.5)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, image, title in zip(axes, [table, sgs['simulation'], fftma['simulation']],
                            ['covariance map C', 'sgs with map(C)', 'fftma with map(C)']):
    ax.imshow(image)
    ax.set_title(title)
plt.tight_layout()
plt.show()
