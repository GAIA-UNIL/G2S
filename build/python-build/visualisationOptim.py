#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy
import math
from variogram import variogram
from g2s import run as g2s
import sys
import os

def computeConnectivity(image,steps):
	result=numpy.zeros([len(steps)]);
	for i in range(len(steps)):
		bwImage=image < steps[i]
		labeledImage1,NumberFeature1=ndimage.label(bwImage);
		labeledImage2,NumberFeature2=ndimage.label(numpy.logical_not(bwImage));
		hist, bin_edges=numpy.histogram(labeledImage1-labeledImage2,numpy.arange(-NumberFeature2,NumberFeature1+1),density=False);
		result[i]= (hist*(hist-1)).sum()/(image.size*(image.size-1))
	return result
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy
import math
from variogram import variogram
from g2s import run as g2s
import sys
import os  

## extra function
def mesureQualitry( vario1, vario2 ):
	return numpy.sum(numpy.abs((vario1[0]-vario2[0])*vario1[1]));


## main code
source=numpy.single(misc.imread('source.png'))/255
serverAddressList=['localhost'];


##
#list Of kernel

# possible alphas
alpha=[0,0.01,0.05,0.2,0.5,0.7,0.8,0.9,0.95,0.98,0.995,1];

distanceMtrix=numpy.zeros([81,81]);
distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
distanceMtrix=ndimage.distance_transform_edt(numpy.logical_not(distanceMtrix))
kernel=[];

# create kernels

for alpha_value in alpha:
	print(alpha_value)
	type=[];
	functionName=[];
	loaclDistance=distanceMtrix*alpha_value;
	#uniform
	functionName.append("Uniform");
	type.append(numpy.single(loaclDistance<1));
	#triangular
	functionName.append("Triangular");
	type.append(numpy.maximum(1-loaclDistance,0));
	#Epanechnikov
	functionName.append("Epanechnikov");
	type.append(numpy.maximum(3/4*(1-numpy.square(loaclDistance)),0));
	#Quartic
	functionName.append("Quartic");
	type.append(numpy.maximum(15/16*numpy.square(1-numpy.square(loaclDistance)),0));
	#Triweight
	functionName.append("Triweight");
	type.append(numpy.maximum(35/32*numpy.power(1-numpy.square(loaclDistance),3),0));
	#Tricube
	functionName.append("Tricube");
	type.append(numpy.maximum(70/81*numpy.power(1-numpy.power(loaclDistance,3),3),0));
	#Gaussian
	functionName.append("Gaussian");
	type.append(1/math.sqrt(2*math.pi)*numpy.exp(-1/2*numpy.square(loaclDistance)));
	#Cosine
	functionName.append("Cosine");
	type.append(math.pi/4*numpy.cos(math.pi/2*loaclDistance));
	#Logistic
	functionName.append("Logistic");
	type.append(1./(numpy.exp(loaclDistance)+2+numpy.exp(-loaclDistance)));
	#Sigmoid function
	functionName.append("Sigmoid");
	type.append(math.pi/2*1./(numpy.exp(loaclDistance)+numpy.exp(-loaclDistance)));
	#Silverman kernel
	functionName.append("Silverman");
	type.append(1/2*numpy.multiply(numpy.exp(-loaclDistance/math.sqrt(2)),numpy.sin(loaclDistance/math.sqrt(2))+math.pi/4));
	# #variogram kernel
	# functionName.append("variogram");
	# type.append(1/2*exp(-loaclDistance/sqrt(2)).*sin(loaclDistance/sqrt(2)+pi/4));
	# #entropy kernel
	# functionName.append("entropy");
	# type.append(1/2*exp(-loaclDistance/sqrt(2)).*sin(loaclDistance/sqrt(2)+pi/4));
	#kernel=cat(2,kernel,type);
	kernel.append(type)

numberOfSimulation=50
numberOfThreadProJob=2

val=numpy.full([len(kernel), len(kernel[0]), numberOfSimulation],numpy.nan);


import colorsys
import matplotlib.colors as colors
spacing=numpy.linspace(0,1,256)
colorsList=plt.cm.nipy_spectral(spacing);
for x in range(colorsList.shape[0]):
	r,g,b,alpha=colorsList[x,:];
	h,l,s=colorsys.rgb_to_hls(r,g,b)
	l=1-l;
	r,g,b=colorsys.hls_to_rgb(h,l,s)
	colorsList[x,:]=r,g,b,alpha;



#print(numpy.stack([spacing,colorsList[:,0],colorsList[:,0]],1));
cdict1 = {'red':   numpy.stack([spacing,colorsList[:,0],colorsList[:,0]],1),

         'green': numpy.stack([spacing,colorsList[:,1],colorsList[:,1]],1),

         'blue':  numpy.stack([spacing,colorsList[:,2],colorsList[:,2]],1)
        }
cmapPerso=colors.LinearSegmentedColormap('nipy_spectral_LInv',cdict1);
cmapPerso.set_bad('black',1.)


fileName='./simErrorMap.npy';


if len(sys.argv)>1 :
	fileName = sys.argv[1]
	#print(fileName)
if os.path.exists(fileName) :
	val=numpy.load(fileName)
	#print(val)
varioRef=variogram(source);

#val=val.take([0, 2, 3, 4],1)

# print("min : ",val.mean(2)[:,:,0].min(), "- ", val.mean(2)[:,:,1].min())

# print(val.mean(2)[30:40,1,0])

from matplotlib.colors import LogNorm
kernelName=functionName;#["Gaussian","Cosinus","Logistic","Sigmoid","Silverman"]
print(functionName)
plt.imshow(val.mean(2)[:,:,0].transpose(),vmin=val.mean(2)[:,:,0].min(), vmax=val.mean(2)[:,:,0].max(),norm=LogNorm(), aspect='auto').set_cmap(cmapPerso)
plt.yticks(numpy.arange(len(functionName)), kernelName,fontsize=15)
plt.xticks(numpy.arange(9)*5+4.5, (numpy.arange(10)+1)/10,fontsize=15)
plt.xlabel('$\\alpha$ kernel parameter', fontsize=16)
print(val.mean(2)[:,:,0].transpose().argmin()/val.shape[0])
print(val.mean(2)[:,:,0].transpose().argmin()%val.shape[0])
cb=plt.colorbar(orientation="horizontal",ticks=[val.mean(2)[:,:,0].min(), (val.mean(2)[:,:,0].min() + val.mean(2)[:,:,0].max())/2, val.mean(2)[:,:,0].max()])
cb.ax.tick_params(labelsize=15);
cb.ax.set_xticklabels(['Low', 'Medium', 'High'])
plt.tight_layout()
plt.savefig('../figures/static_vario.png', transparent=True)
plt.show(block=False)
plt.figure()
plt.imshow(val.mean(2)[:,:,1].transpose(),vmin=val.mean(2)[:,:,1].min(), vmax=val.mean(2)[:,:,1].max(),norm=LogNorm(), aspect='auto').set_cmap(cmapPerso)
plt.yticks(numpy.arange(len(functionName)), kernelName,fontsize=15)
plt.xticks(numpy.arange(9)*5+4.5, (numpy.arange(10)+1)/10,fontsize=15)
plt.xlabel('$\\alpha$ kernel parameter', fontsize=16)
print(val.mean(2)[:,:,1].transpose().argmin()/val.shape[0])
print(val.mean(2)[:,:,1].transpose().argmin()%val.shape[0])
cb=plt.colorbar(orientation="horizontal",ticks=[val.mean(2)[:,:,1].min(), (val.mean(2)[:,:,1].min() + val.mean(2)[:,:,1].max())/2, val.mean(2)[:,:,1].max()])
cb.ax.tick_params(labelsize=15);
cb.ax.set_xticklabels(['Low', 'Medium', 'High'])
plt.tight_layout()
plt.savefig('../figures/static_connect.png', transparent=True)
plt.show(block=True)

