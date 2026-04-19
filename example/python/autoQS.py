import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt


def first_2d_view(array):
    view = numpy.asarray(array)
    view = numpy.squeeze(view)
    while view.ndim > 2:
        view = view[..., 0]
    return view


# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)))

# simple calibration kernel
kernel = numpy.ones((15, 15), dtype=float)

# AutoQS call using G2S
result, *_ = g2s(
    '-a', 'autoQS',
    '-ti', ti,
    '-ki', kernel,
    '-dt', [0],
    '-maxk', 2,
    '-maxn', 80,
    '-density', 0.0312, 0.0625, 0.125, 0.25,
    '-maxIter', 5000,
    '-minIter', 200,
    '-mpow', 2,
    '-j', 0.5
)

result2d = first_2d_view(result)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('AutoQS calibration', size='xx-large')
ax1.imshow(ti)
ax1.set_title('Training image')
ax1.axis('off')
ax2.imshow(result2d)
ax2.set_title('Calibration result')
ax2.axis('off')
plt.show()
