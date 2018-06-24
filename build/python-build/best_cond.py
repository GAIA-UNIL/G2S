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

# from osgeo import gdal
# import osgeo.gdalnumeric as gdn

# def img_to_array(input_file, dim_ordering="channels_last", dtype='float32'):
# 	file  = gdal.Open(input_file)
# 	bands = [file.GetRasterBand(i) for i in range(1, file.RasterCount + 1)]
# 	arr = numpy.array([gdn.BandReadAsArray(band) for band in bands]).astype(dtype)
# 	if dim_ordering=="channels_last":
# 		arr = numpy.transpose(arr, [1, 2, 0])  # Reorders dimensions, so that channels are last
# 	return arr


## extra function
def mesureQualitry( ref, sim):
	errorSimSSIM=0;
	numberOfBand=min(ref.shape[2],sim.shape[2])
	for x in range(numberOfBand):
		errorSimSSIM+=ssim(ref[:,:,x], sim[:,:,x])
	# connectError,path=fastdtw(conectivity1, conectivity2, dist=euclidean)
	#numpy.sum(numpy.abs(conectivity1-conectivity2));
	# varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0])*vario1[1])));
	#print(connectError," -", varioError);
	errorSimSSIM/=numberOfBand;
	return errorSimSSIM

def bestStaticKernel(alpha):
	distanceMtrix=numpy.zeros([sizeKernel,sizeKernel,1])
	distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
	distanceMtrix=ndimage.distance_transform_edt(numpy.logical_not(distanceMtrix))
	loaclDistance=distanceMtrix*alpha
	ker=numpy.repeat(1/math.sqrt(2*math.pi)*numpy.exp(-loaclDistance),4, axis=2)
	norm=numpy.sum(ker)
	ker=ker/norm
	print(ker.shape)
	return ker

## main code
if __name__ == "__main__":
	source=numpy.load('training.npy');#img_to_array('training.tiff')
	destinationRef=numpy.load('target.npy');#img_to_array('target.tiff')
	source=source[:200,:200,:]
	destinationRef=destinationRef[:200,:200,:]

	sourceReshaped=numpy.reshape(source,[source.shape[0]*source.shape[1],source.shape[2]]);

	minValue=numpy.percentile(sourceReshaped, 2, axis=0)
	maxValue=numpy.percentile(sourceReshaped, 98, axis=0)

	destination=destinationRef.copy();
	destination[:,:,3:]=numpy.nan;
	address='localhost';

	fileName='./kernelSet.npz';


	if len(sys.argv)>1 :
		fileName = sys.argv[1]
		print(fileName)
	if os.path.exists(fileName) :
		data = numpy.load(fileName)
		kernels=data['kernels']
		iteration= data['iteration']
		numberOfSimulation=data['numberOfSimulation']
		probPower=data['probPower']
		convergance=data['convergance'];
		idValue=data['idValue']

	nanList=numpy.argwhere(numpy.logical_not(numpy.isnan(convergance)))
	maxId=nanList[:,0].max().astype('int');

	sizeKernel=9;
	dimProblem=min(source.shape[2],destinationRef.shape[2])

	numberOfThreadProJob=2
	numberOfSimulation=1;
	ratioSelection=0.25;
	meanQualityPosition=convergance[maxId,:].argsort();
	maxPosition=int(ratioSelection*len(kernels));

	# create kernels
	ker=[numpy.median(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1), 
			numpy.mean(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1),
			bestStaticKernel(4.5)]

	qualityBests=numpy.full([len(ker),numberOfSimulation],numpy.nan);

	requestId=0;
	for i in range(len(ker)):
		for z in range(numberOfSimulation):
			result=g2s('-sa',address,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,source.shape[2])),'-ki',ker[i],'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);#,'-sp',path
			qualityBests[i,z]=mesureQualitry(destinationRef,result[0]);
			requestId+=1
			result[0]=numpy.clip(result[0],minValue,maxValue);
			print(qualityBests[i,z])
			if i==0 and z==0 :
				misc.imsave('../figures/p50-Simulation_cond.png', result[0][:,:,-1])
			if i==1 and z==0 :
				misc.imsave('../figures/meanSimulation_cond.png', result[0][:,:,-1])
			if i==2 and z==0 :
				misc.imsave('../figures/staticSimulation_cond.png', result[0][:,:,-1])

	

	# numpy.savez('best_cond.npz', qualityBests=qualityBests);
