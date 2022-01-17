# coming soon

import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)));

# create empty grid and add conditioning data
conditioning = numpy.zeros((200,200))*numpy.nan; 
# fill the grid with 50 random points
conditioning.flat[numpy.random.permutation(conditioning.size)[:50]]=ti.flat[numpy.random.permutation(ti.size)[:50]];

# QS call using G2S
simulation,_=g2s('-a','qs', 
                 '-ti',ti,
                 '-di',conditioning,
                 '-dt',numpy.zeros((1,)),
                 '-k',1.2,
                 '-n',30,
                 '-j',0.5);

# Display results 
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(7,4))
fig.suptitle('QS Conditional simulation',size='xx-large',y=0.9)
ax1.imshow(ti)
ax1.set_title('Training image');
ax1.axis('off');
ax2.imshow(conditioning)
ax2.set_title('Conditioning');
ax2.axis('off');
ax3.imshow(simulation)
ax3.set_title('Simulation');
ax3.axis('off');
plt.show()