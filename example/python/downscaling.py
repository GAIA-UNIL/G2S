import numpy as np
from PIL import Image
from g2s import g2s
import matplotlib.pyplot as plt
from matplotlib import colors
from io import BytesIO
import requests

#Load example training image "Dunes gobi"
ti_full = Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/dunes_gobi.tiff').content))
#Keep one version at fine resolution
ti_fine = np.array(ti_full)
#Make a synthetic coarse resolution image by upscaling and reprojecting to the original grid 
ti_coarse = np.array(ti_full.resize(
    (int(ti_full.width/5),int(ti_full.height/5))).resize(
    (ti_full.width,ti_full.height),resample=Image.NEAREST))

#Display the full training image at both resolutions
norm =colors.Normalize(vmin=ti_fine.min(),vmax=ti_fine.max())
f,(ax1,ax2) = plt.subplots(2,1,figsize=(12,12))
ax1.imshow(ti_fine,norm=norm)
ax1.set_title('Fine resolution training image')
ax2.imshow(ti_coarse, norm =norm)
ax2.set_title('Coarse resolution training image')

#Crop half of the image to be used as ti
ti = np.stack((ti_fine[:500,:500],ti_coarse[:500,:500]),axis=2)
#Crop upper right corner to be used as di
di_coarse = ti_coarse[:200,-200:]
di_fine   = np.zeros((200,200))*np.nan
di=np.stack((di_fine,di_coarse),axis=2)
#dt consists of two zeros representing two continuous variables
dt = [0]*ti.shape[-1]

# QS call using G2S
simulation,index,_=g2s('-a','qs',
                         '-ti',ti,
                         '-di',di,
                         '-dt',dt,
                         '-k',1.0,
                         '-n',30,
                         '-j',0.5)

#Display results 
norm =colors.Normalize(vmin=ti_fine.min(),vmax=ti_fine.max())
f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,12),sharex=True,sharey=True)
plt.subplots_adjust(wspace=0.1,hspace=0.1)
plt.suptitle('QS Downscaling',size='xx-large')
ax1.imshow(di_coarse,norm=norm)
ax1.set_title('Coarse res di')
ax2.imshow(simulation[:,:,0], norm =norm)
ax2.set_title('Simulation')
ax3.imshow(index)
ax3.set_title('Index')
ax4.imshow(ti_fine[:200,-200:],norm=norm)
ax4.set_title('True image')
ax4.set_xticks(np.arange(0,200,25),np.arange(ti_full.width-200,ti_full.width,25));
