# pyDev with a constant simulation path: sgsCP caches the kriging weights of each
# path index, so a second realisation with the same path only replays them.
# See build/PyDevAlgo/sgsCP.py.
import numpy
from g2s import g2s
import matplotlib.pyplot as plt

shape = (200, 200)
destination = numpy.full(shape, numpy.nan)
path = numpy.random.default_rng(0).permutation(destination.size).reshape(shape).astype(float)

realisations = []
for seed in (1, 2):
    result = g2s('-a', 'pyDev', '-pa', 'sgsCP', '-di', destination, '-dt', [0],
                 '-n', 50, '-sp', path, '-s', seed, '-j', 0.5)
    replayed = numpy.mean(result["indexmap"] == 2) * 100
    print(f"seed {seed}: {float(result['duration_ms']) / 1000:.2f} s, {replayed:.0f}% of the weights replayed from the cache")
    realisations.append(result["simulation"])

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, sim, seed in zip(axes, realisations, (1, 2)):
    ax.imshow(sim)
    ax.set_title(f'seed {seed}')
    ax.axis('off')
fig.suptitle('pyDev SGS with a constant path')
plt.show()
