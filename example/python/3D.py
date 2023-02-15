import numpy as np
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt
import tifffile as tf
from matplotlib import cm

#Load example boolean training image "Damier 3D" and crop it to reduce computational requirements
dim = 30
ti = tf.imread(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/Damier3D.tiff').content))[:dim,:dim,:dim]

# QS call using G2S
simulation,index,time,*_=g2s('-a','qs',
                             '-ti',ti,
                             '-di',np.zeros_like(ti)*np.nan,
                             '-dt',[1], #1 for categorical variables
                             '-k',1.2,
                             '-n',30)

#Display results
fig = plt.figure(figsize=(20,15))
ax1=fig.add_subplot(121,projection='3d')
ax1.voxels(ti,alpha=0.9,facecolor='tab:blue',edgecolor='black')
ax1.view_init(azim=45)
ax1.set_title('3D Training image')
ax2=fig.add_subplot(122,projection='3d')
ax2.voxels(simulation,alpha=0.9,facecolor='tab:blue',edgecolor='black')
ax2.view_init(azim=45)
ax2.set_title('QS 3D simulation')

#Display the indices of the ti-values found in the di-image
viridis = cm.get_cmap('viridis',index.max())
colormap = viridis(index)
fig = plt.figure(figsize=(10,10))
ax1=fig.add_subplot(111,projection='3d')
ax1.voxels(index,facecolors=colormap)
ax1.view_init(azim=45);
