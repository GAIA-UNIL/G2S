# pyDev batch mode with a machine-learning model: a small neural network
# (build/PyDevAlgo/mlpTemplate.py, NumPy only) is trained on the Strebelle
# training image, then predicts category probabilities for a whole batch of
# nodes in one forward pass. The same structure works with PyTorch or JAX.
# The first run trains the network (a few seconds); later runs reuse it.
import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff').content)))

result = g2s('-a', 'pyDev',
             '-pa', 'mlpTemplate',
             '-ti', ti,
             '-di', numpy.full((200, 200), numpy.nan),
             '-dt', [1],        # categorical
             '-n', 40,          # informed neighbours given to the model
             '-bs', 64)         # up to 64 nodes per forward pass (128 threads)
print(f"{result['python_calls']} Python calls, {float(result['batch_average_size']):.1f} nodes per batch on average")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(ti)
ax1.set_title('Training image')
ax2.imshow(result['simulation'])
ax2.set_title('Neural network, batch mode')
for ax in (ax1, ax2):
    ax.axis('off')
plt.show()
