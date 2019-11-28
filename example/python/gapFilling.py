import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

#load data
tiWithGap = numpy.array(Image.open(BytesIO(requests.get('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)));
tiWithGap[60:140,60:140]=numpy.nan;

# QS call using G2S
simulation,_=g2s('-a','qs','-ti',tiWithGap,'-di',tiWithGap,'-dt',numpy.zeros((1,)),'-k',1.2,'-n',25,'-j',0.5);

#Display results 
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Gap filling')
ax1.imshow(tiWithGap)
ax1.set_title('Training image');
ax1.axis('off');
ax2.imshow(simulation)
ax2.set_title('Simulation');
ax2.axis('off');
plt.show()