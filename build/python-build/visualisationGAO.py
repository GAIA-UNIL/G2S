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

def extractCenter(matrix):
	size=numpy.ceil(numpy.array(matrix.shape)/4).astype('int');
	return matrix[size[0]:-size[0],size[1]:-size[1]];

## extra function
def mesureQualitry( vario1, vario2, conectivity1, conectivity2 ):
	connectError=numpy.sum(numpy.abs(conectivity1-conectivity2));
	varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0])*vario1[1])));
	#print(connectError," -", varioError);
	return connectError


## main code
source=numpy.single(misc.imread('source.png'))/255
serverAddressList=['localhost'];

maxIteration=10000;

if len(sys.argv)>1 :
	maxIteration = int(sys.argv[1])

if len(sys.argv)>2 :
	file_name = sys.argv[2]
	fp = open(file_name)
	serverAddressList = fp.read().splitlines()

print(serverAddressList)


sizeKernel=7;

distanceMtrix=numpy.zeros([sizeKernel,sizeKernel]);
distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
distanceMtrix=ndimage.distance_transform_edt(distanceMtrix)
kernels=[];

NumberOfKernel=128;
numberOfSimulation=10
saveRate=5;
iteration=0
probPower=1/2
mixingRatio=0.3
muationfactor=0.02;
muationRatio=0.3;
ratioSelection=0.25;
idValue=1;
convergance=numpy.full([maxIteration,NumberOfKernel],numpy.nan)

if os.path.exists('./kernelSet.npz') :
	data = numpy.load('kernelSet.npz')
	kernels=data['kernels']
	iteration= data['iteration']
	numberOfSimulation=data['numberOfSimulation']
	probPower=data['probPower']
	convergance=data['convergance'];
	idValue=data['idValue']

nanList=numpy.argwhere(numpy.logical_not(numpy.isnan(convergance)))
maxId=nanList[:,0].max().astype('int');

print(maxId)

from matplotlib.colors import LogNorm

plt.plot(	range(maxId),convergance[:maxId].min(1),':',
			range(maxId),numpy.percentile(convergance[:maxId], 5,1),'-.',
			range(maxId),numpy.percentile(convergance[:maxId],50,1),'-',
			range(maxId),numpy.percentile(convergance[:maxId],95,1),'-.',
			range(maxId),convergance[:maxId].max(1),':');
plt.show()

maxValue=numpy.percentile(numpy.dstack(kernels),99)

print(maxValue)

for x in range(len(kernels)):
	plt.close()
	# plt.plot(range(sizeKernel*sizeKernel),kernels[x].flatten())
	plt.imshow(kernels[x],vmin=0.00001, vmax=maxValue, norm=LogNorm()).set_cmap('nipy_spectral')
	plt.colorbar()
	plt.show(block=False)
	plt.pause(0.01)

plt.imshow(numpy.median(numpy.dstack(kernels),2),vmin=0.0001, vmax=maxValue,norm=LogNorm()).set_cmap('nipy_spectral');
plt.show()

plt.imshow(numpy.percentile(numpy.dstack(kernels),10,2),vmin=0.0001, vmax=maxValue,norm=LogNorm()).set_cmap('nipy_spectral');
plt.show()

plt.imshow(numpy.mean(numpy.dstack(kernels),2),vmin=0.0001, vmax=maxValue,norm=LogNorm()).set_cmap('nipy_spectral');
plt.show()


