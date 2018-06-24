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

def uniformKernel():
	ker=numpy.ones([sizeKernel,sizeKernel,dimProblem])
	norm=numpy.sum(ker)
	ker=ker/norm
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
	print(minValue)
	print(maxValue)

	
	destination=destinationRef.copy();
	destination[:,:,3:]=numpy.nan;
	address='localhost';

	sizeKernel=9;
	dimProblem=min(source.shape[2],destinationRef.shape[2])

	uniKernel=uniformKernel()
	# create kernels
	
	numberOfThreadProJob=2
	numberOfSimulation=100;

	qualityUniform=numpy.full([1,numberOfSimulation],numpy.nan);

	requestId=0;
	for z in range(numberOfSimulation):
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,source.shape[2])),'-ki',uniKernel,'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);#,'-sp',path
		qualityUniform[0,z]=mesureQualitry(destinationRef,result[0]);
		requestId+=1
		result[0]=numpy.clip(result[0],minValue,maxValue);
		misc.imsave('../figures/uniformSimulation_cond.png', result[0][:,:,-1])


	meanQuality=qualityUniform.mean(1);

	numpy.savez('uniformKer_cond.npz', qualityUniform=qualityUniform);

	print(meanQuality)

	source=numpy.clip(source,minValue,maxValue);
	destinationRef=numpy.clip(destinationRef,minValue,maxValue);

	misc.imsave('../figures/ti_cond.png', (source[:,:,-1]-minValue[-1])/(maxValue[-1]-minValue[-1]))
	misc.imsave('../figures/ti_cv_cond.png', (source[:,:,:-1]-minValue[:-1])/(maxValue[:-1]-minValue[:-1]))
	misc.imsave('../figures/sg_cond.png', (destinationRef[:,:,-1]-minValue[-1])/(maxValue[-1]-minValue[-1]))
	misc.imsave('../figures/sg_cv_cond.png', (destinationRef[:,:,:-1]-minValue[:-1])/(maxValue[:-1]-minValue[:-1]))

