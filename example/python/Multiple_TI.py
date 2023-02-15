import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

#ti1 contains horizontal lines and ti2 vertical lines
ti1 = numpy.tile(numpy.sin(range(150)),(150,1))
ti2 = ti1.transpose()

# QS call using only the horizontal lines as TI
simulation1,index1,*_ = g2s('-a','qs',
                         '-ti',ti1,
                         '-di',numpy.zeros((150,150))*numpy.nan,
                         '-dt',[1], #1 for categorical variable
                         '-k',1.2,
                         '-n',25,
                         '-j',0.5);

# QS call using both horizontal and vertical lines as TI's
simulation2,index2,*_ = g2s('-a','qs',
                         '-ti',[ti1,ti2],
                         '-di',numpy.zeros((150,150))*numpy.nan,
                         '-dt',[1],
                         '-k',1.2,
                         '-n',25,
                         '-j',0.5);

#Display results
fig, ([[ax1,ax2],[ax3,ax4]]) = plt.subplots(2,2,figsize=(10,10),sharex = True,sharey = True)
fig.suptitle('QS Multiple TI simulation',size='xx-large')
ax1.imshow(ti1)
ax1.set_title('Training image 1');
ax1.axis('off');
ax2.imshow(ti2)
ax2.set_title('Training image 2');
ax2.axis('off');
ax3.imshow(simulation1)
ax3.set_title('Simulation 1');
ax3.axis('off');
ax4.imshow(simulation2)
ax4.set_title('Simulation 2');
ax4.axis('off');
