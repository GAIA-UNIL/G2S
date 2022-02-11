import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

# load example training image ('stone') and cut out a part of it
tiWithGap = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)));
tiWithGap[60:140,60:140]=numpy.nan;

# QS call using G2S
simulation,_=g2s('-a','qs',
                 '-ti',tiWithGap,
                 '-di',tiWithGap,
                 '-dt',[0],
                 '-k',1.2,
                 '-n',25,
                 '-j',0.5);

#Display results 
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(7,4))
fig.suptitle('QS Gap filling',size='xx-large')
ax1.imshow(tiWithGap)
ax1.set_title('Training image');
ax1.axis('off');
ax2.imshow(simulation)
ax2.set_title('Simulation');
ax2.axis('off');
plt.show()