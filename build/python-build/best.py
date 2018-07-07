#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
from scipy import random
import matplotlib.pyplot as plt
import numpy
import math
from variogram import variogram
from g2s import run as g2s
import sys
import os
from multiprocessing import sharedctypes
from scipy.spatial.distance import euclidean
from skimage.measure import compare_ssim as ssim
from fastdtw import fastdtw

# from osgeo import gdal
# import osgeo.gdalnumeric as gdn

# def img_to_array(input_file, dim_ordering="channels_last", dtype='float32'):
# 	file  = gdal.Open(input_file)
# 	bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
# 	arr = numpy.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
# 	if dim_ordering=="channels_last":
# 		arr = numpy.transpose(arr, [1, 2, 0])  # Reorders dimensions, so that channels are last
# 	return arr


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
	connectError,path=fastdtw(conectivity1, conectivity2, dist=euclidean)
	#numpy.sum(numpy.abs(conectivity1-conectivity2));
	varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0]))));
	#print(connectError," -", varioError);
	return connectError

## extra function
def mesureQualitry2( vario1, vario2, conectivity1, conectivity2 ):
	connectError,path=fastdtw(conectivity1, conectivity2, dist=euclidean)
	#numpy.sum(numpy.abs(conectivity1-conectivity2));
	varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0]))));
	#print(connectError," -", varioError);
	return varioError

def bestStaticKernelLogistic(alpha):
	distanceMtrix=numpy.zeros([sizeKernel,sizeKernel])
	distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
	distanceMtrix=ndimage.distance_transform_edt(numpy.logical_not(distanceMtrix))
	loaclDistance=distanceMtrix*alpha
	ker=1./(numpy.exp(loaclDistance)+2+numpy.exp(-loaclDistance))
	norm=numpy.sum(ker)
	ker=ker/norm
	print(ker.shape)
	return ker

def bestStaticKernelSilverman(alpha):
	distanceMtrix=numpy.zeros([sizeKernel,sizeKernel])
	distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
	distanceMtrix=ndimage.distance_transform_edt(numpy.logical_not(distanceMtrix))
	loaclDistance=distanceMtrix*alpha
	ker=1/2*numpy.multiply(numpy.exp(-loaclDistance/math.sqrt(2)),numpy.sin(loaclDistance/math.sqrt(2))+math.pi/4)
	norm=numpy.sum(ker)
	ker=ker/norm
	print(ker.shape)
	return ker

def bestStaticKernelSigmoid(alpha):
	distanceMtrix=numpy.zeros([sizeKernel,sizeKernel])
	distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
	distanceMtrix=ndimage.distance_transform_edt(numpy.logical_not(distanceMtrix))
	loaclDistance=distanceMtrix*alpha
	ker=math.pi/2*1./(numpy.exp(loaclDistance)+numpy.exp(-loaclDistance))
	norm=numpy.sum(ker)
	ker=ker/norm
	print(ker.shape)
	return ker


## main code
if __name__ == "__main__":
	source=numpy.single(misc.imread('source.png'))/255
	
	address='localhost';

	sizeKernel=41;

	varioRef=variogram(source);
	connectStep=(numpy.arange(255)+0.5)/255;
	connectivityRef=computeConnectivity(source,connectStep);

	fileName='./kernelSet.npz';


	if len(sys.argv)>1 :
		fileName = sys.argv[1]
		print(fileName)
	if os.path.exists(fileName) :
		data = numpy.load(fileName)
		kernels=data['oldKernel']
		iteration= data['iteration']
		numberOfSimulation=data['numberOfSimulation']
		probPower=data['probPower']
		convergance=data['convergance'];
		idValue=data['idValue']

	nanList=numpy.argwhere(numpy.logical_not(numpy.isnan(convergance)))
	maxId=nanList[:,0].max().astype('int');

	sizeKernel=9;
	
	numberOfThreadProJob=2
	numberOfSimulation=200;
	ratioSelection=0.25;
	meanQualityPosition=convergance[maxId,:].argsort();
	maxPosition=int(ratioSelection*len(kernels));

	# create kernels
	ker=[numpy.median(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1), 
			numpy.mean(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1),
			bestStaticKernelLogistic(0.39),
			bestStaticKernelSigmoid(0.34)]

	qualityBests=numpy.full([len(ker)*2,numberOfSimulation],numpy.nan);

	requestId=0;
	for z in range(numberOfSimulation):
		for i in range(len(ker)):
			result=g2s('-sa',address,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',ker[i],'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);
			qualityBests[2*i+0,z]=mesureQualitry(varioRef, variogram(result[0]), connectivityRef, computeConnectivity(result[0],connectStep));
			qualityBests[2*i+1,z]=mesureQualitry2(varioRef, variogram(result[0]), connectivityRef, computeConnectivity(result[0],connectStep));
			requestId+=1
			if i==0 :
				misc.imsave('../figures/p50-Simulation'+str(z)+'.png', result[0])
			if i==1 :
				misc.imsave('../figures/meanSimulation'+str(z)+'.png', result[0])
			if i==2 :
				misc.imsave('../figures/staticLogisticSimulation'+str(z)+'.png', result[0])
			if i==3 :
				misc.imsave('../figures/staticSigmoidSimulation'+str(z)+'.png', result[0])

	

	numpy.savez('best.npz', qualityBests=qualityBests);
