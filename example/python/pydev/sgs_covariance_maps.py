# pyDev non-stationary SGS: covariance maps are given as training images and
# -ii selects which map is used at each node (here a gradual change of range from
# top to bottom). See build/PyDevAlgo/sgsTiCP.py.
import numpy
from g2s import g2s
import matplotlib.pyplot as plt

shape = (200, 200)
destination = numpy.full(shape, numpy.nan)
path = numpy.random.default_rng(0).permutation(destination.size).reshape(shape).astype(float)

# covariance maps centred on lag 0: exp(-alpha * |h|) for increasing ranges
distance = numpy.hypot(*numpy.mgrid[-100:101, -100:101])
alphas = 2.0 ** numpy.arange(-2, -7.5, -0.5)
covariance_maps = [numpy.exp(-alpha * distance) for alpha in alphas]

# map index per node: from the shortest range (top) to the longest (bottom)
map_index = numpy.floor(numpy.arange(shape[0]) / shape[0] * len(covariance_maps))[:, None] * numpy.ones(shape)

result = g2s('-a', 'pyDev', '-pa', 'sgsTiCP', '-di', destination, '-dt', [0], '-n', 50,
             '-sp', path, '-ti', *covariance_maps, '-ii', map_index, '-j', 0.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(map_index)
ax1.set_title('covariance map index (-ii)')
ax2.imshow(result["simulation"])
ax2.set_title('simulation')
for ax in (ax1, ax2):
    ax.axis('off')
plt.show()
