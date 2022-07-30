import numpy
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt
import tifffile as tf

ti = tf.imread(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/ti_3_variables.tiff').content))

# QS call using G2S (with dt set to two continuous and one categorical variable)
simulation,index,_=g2s('-a','qs',
                 '-ti',ti,
                 '-di',numpy.zeros([400,400,3])*numpy.nan,
                 '-dt',[0,0,1],
                 '-k',1.2,
                 '-n',50,
                 '-j',0.5);

#Display results 
fig, ([[ax1,ax2,ax3],[ax4,ax5,ax6]]) = plt.subplots(2,3,figsize=(15,10),sharex = True,sharey = True)
fig.suptitle('QS Multivariate simulation',size='xx-large')
ax1.imshow(ti[:,:,0])
ax1.set_title('Training image dim 1');
ax1.axis('off');
ax2.imshow(ti[:,:,1])
ax2.set_title('Training image dim 2');
ax2.axis('off');
ax3.imshow(ti[:,:,2])
ax3.set_title('Training image dim 3');
ax3.axis('off');
ax4.imshow(simulation[:,:,0])
ax4.set_title('Simulation dim 1');
ax4.axis('off');
ax5.imshow(simulation[:,:,1])
ax5.set_title('Simulation dim 2');
ax5.axis('off');
ax6.imshow(simulation[:,:,2])
ax6.set_title('Simulation dim 3');
ax6.axis('off');