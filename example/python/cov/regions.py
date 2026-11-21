# Several covariance models, one per region, selected for each cell by an
# index map (-ii): region 0 is isotropic and smooth, region 1 anisotropic.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

ny, nx = 200, 200
rows, columns = np.mgrid[0:ny, 0:nx]
index = (rows > 80 + 30*np.sin(columns/25)).astype(int)  # 0 above the boundary, 1 below

smooth = cov.gaussian(range=30)
layered = 0.1*cov.nugget() + 0.9*cov.exponential(range=[60, 6], angle=10)
regions = cov.regions([smooth, layered], index)
print(regions.args()[:3])  # ['-cov', '1*gaussian(range=30)', '0.1*nugget + 0.9*exponential(range=[60, 6], angle=10)']

result = g2s('-a', 'sgs',
             '-di', np.full((ny, nx), np.nan),
             '-dt', [0],
             *regions.args(),      # -cov model0 model1 -ii index
             '-n', 40,
             '-s', 7,
             '-j', 0.5)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(index)
axes[0].set_title('model index (-ii)')
axes[1].imshow(result['simulation'])
axes[1].set_title('sgs')
plt.tight_layout()
plt.show()
