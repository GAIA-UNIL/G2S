import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)))

maxK = 100
maxN = 100
size = 1000

y = numpy.linspace(numpy.sqrt(2), numpy.sqrt(maxN), size) ** 2
x = numpy.exp(numpy.linspace(0, numpy.log(maxK), size))
xv, yv = numpy.meshgrid(x, y)

simulation, *_ = g2s(
    '-a', 'qs',
    '-ti', ti,
    '-di', numpy.zeros((size, size)) * numpy.nan,
    '-dt', [0],
    '-kvi', xv,
    '-ni', yv,
    '-j'
)

plt.imshow(simulation)

x_labels = [f"{val:.1f}" for val in x]
y_labels = [f"{val:.0f}" for val in y]
spacingLabelX = simulation.shape[0] // 20
spacingLabelY = simulation.shape[1] // 20

plt.xticks(
    ticks=numpy.linspace(0, simulation.shape[1] - 1, len(x))[::spacingLabelX],
    labels=x_labels[::spacingLabelX]
)
plt.yticks(
    ticks=numpy.linspace(0, simulation.shape[0] - 1, len(y))[::spacingLabelY],
    labels=y_labels[::spacingLabelY]
)

plt.xlabel('k')
plt.ylabel('n')
plt.show()
