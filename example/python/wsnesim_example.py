import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

# load example training image ('strebelle') from same repo path style
url = "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff"
resp = requests.get(url, timeout=30)
resp.raise_for_status()
ti_raw = numpy.array(Image.open(BytesIO(resp.content)))

# WSNESIM call using G2S
# 5 grid levels total: 4 -> 3 -> 2 -> 1 -> 0 (because -mg is the max level)
# --wd is wildcard prefix depth (here: wildcard branches active only for first 2 tree levels)
simulation, t, *_ = g2s(
    '-a', 'wsnesim',
    '-ti', ti_raw,
    '-di', numpy.zeros((1000, 1000)) * numpy.nan,
    '-dt', [1],                   # 1 => categorical
    '-j', 1.001,
    '-mg', 4,
    '-tpl', 3,
    '--wd', 1
)

print("WSNESIM duration:", t)

# display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
fig.suptitle('WSNESIM Unconditional simulation (wildcard depth = 2)', size='x-large')
ax1.imshow(ti_raw, cmap='tab20')
ax1.set_title('Training image (categorical)')
ax1.axis('off')
ax2.imshow(simulation, cmap='tab20')
ax2.set_title('Simulation')
ax2.axis('off')
plt.show()
