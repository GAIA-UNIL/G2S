import numpy
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt
import tifffile as tf

ti = tf.imread(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/ti_3_variables.tiff').content))

###Example 1 - Complete multivariate simulation
#Here all three variables are entirely unkwown and will be simulated simultaneously

# QS call using G2S (with dt set to two continuous and one categorical variable)
simulation1,index1,*_=g2s('-a','qs',
                 '-ti',ti,
                 '-di',numpy.zeros([400,400,3])*numpy.nan,
                 '-dt',[0,0,1],
                 '-k',1.2,
                 '-n',50,
                 '-j',0.5);

#Display results 
fig, ([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]) = plt.subplots(3,3,figsize=(15,10),sharex = True,sharey = True)
fig.suptitle('QS Multivariate simulation - example 2',size='xx-large')
ax1.imshow(ti[:,:,0])
ax1.set_title('Training image dim 1');
ax1.axis('off');
ax2.imshow(ti[:,:,1])
ax2.set_title('Training image dim 2');
ax2.axis('off');
ax3.imshow(ti[:,:,2])
ax3.set_title('Training image dim 3');
ax3.axis('off');
ax4.imshow(numpy.zeros([400,400])*numpy.nan)
ax4.set_title('Destination image dim 1')
ax4.axis('off')
ax5.imshow(numpy.zeros([400,400])*numpy.nan)
ax5.set_title('Destination image dim 2')
ax5.set_aspect('equal')
ax5.axis('off')
ax6.imshow(numpy.zeros([400,400])*numpy.nan)
ax6.set_title('Destination image dim 3')
ax6.axis('off')
ax7.imshow(simulation1[:,:,0])
ax7.set_title('Simulation dim 1');
ax7.axis('off');
ax8.imshow(simulation1[:,:,1])
ax8.set_title('Simulation dim 2');
ax8.axis('off');
ax9.imshow(simulation1[:,:,2])
ax9.set_title('Simulation dim 3');
ax9.axis('off');



### Example 2 - Multivariate simulation with partially informed covariables
#In many situations we will have fully and/or partially informed covariables (e.g. topography + point measurements)
#To simulate a different environment than the training images, we will use the results from example 1 above
#Variable 2 is partially informed and variable 3 is fully informed

#Take random values from the previous simulation of variable 2 and add them to the di
di_var2 = numpy.zeros([400,400])*numpy.nan
di_var2.flat[numpy.random.permutation(di_var2.size)[:500]] = ti[:,:,1].flat[numpy.random.permutation(ti[:,:,1].size)[:500]]
di_example2 = numpy.stack([numpy.zeros([400,400])*numpy.nan,
                                 di_var2,
                                 simulation1[:,:,2]],axis = 2)

# QS call using G2S (with dt set to two continuous and one categorical variable)
simulation2,index,*_=g2s('-a','qs',
                 '-ti',ti,
                 '-di',di_example2,
                 '-dt',[0,0,1],
                 '-k',1.2,
                 '-n',50,
                 '-j',0.5);

#Display results 
fig, ([[ax1,ax2,ax3],[ax4,ax5,ax6],[ax7,ax8,ax9]]) = plt.subplots(3,3,figsize=(15,10),sharex = True,sharey = True)
fig.suptitle('QS Multivariate simulation - example 2',size='xx-large')
ax1.imshow(ti[:,:,0])
ax1.set_title('Training image dim 1');
ax1.axis('off');
ax2.imshow(ti[:,:,1])
ax2.set_title('Training image dim 2');
ax2.axis('off');
ax3.imshow(ti[:,:,2])
ax3.set_title('Training image dim 3');
ax3.axis('off');
ax4.imshow(di_example2[:,:,0])
ax4.set_title('Destination image dim 1')
ax4.axis('off')
ax5.scatter(*numpy.meshgrid(numpy.arange(400),numpy.arange(400,0,-1)),s=5,c=di_example2[:,:,1],marker='.')
ax5.set_title('Destination image dim 2')
ax5.set_aspect('equal')
ax5.axis('off')
ax6.imshow(di_example2[:,:,2])
ax6.set_title('Destination image dim 3')
ax6.axis('off')
ax7.imshow(simulation2[:,:,0])
ax7.set_title('Simulation dim 1');
ax7.axis('off');
ax8.imshow(simulation2[:,:,1])
ax8.set_title('Simulation dim 2');
ax8.axis('off');
ax9.imshow(simulation2[:,:,2])
ax9.set_title('Simulation dim 3');
ax9.axis('off');

