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
	#connectError,path=fastdtw(conectivity1, conectivity2, dist=euclidean)
	#numpy.sum(numpy.abs(conectivity1-conectivity2));
	varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0]))));
	#print(connectError," -", varioError);
	return varioError

## extra function
def mesureQualitry2( vario1, vario2, conectivity1, conectivity2 ):
	connectError,path=fastdtw(conectivity1, conectivity2, dist=euclidean)
	#numpy.sum(numpy.abs(conectivity1-conectivity2));
	varioError=numpy.sum(numpy.abs(extractCenter((vario1[0]-vario2[0]))));
	#print(connectError," -", varioError);
	return varioError

def uniformKernel():
	ker=numpy.ones([sizeKernel,sizeKernel])
	norm=numpy.sum(ker)
	ker=ker/norm
	return ker

## main code
if __name__ == "__main__":
	source=numpy.single(misc.imread('source.png'))/255
	
	address='localhost';

	sizeKernel=41;

	varioRef=variogram(source);
	connectStep=(numpy.arange(255)+0.5)/255;
	connectivityRef=computeConnectivity(source,connectStep);

	uniKernel=uniformKernel()
	# create kernels
	
	numberOfThreadProJob=4
	numberOfSimulation=200;

	qualityUniform=numpy.full([2,numberOfSimulation],numpy.nan);

	requestId=0;
	for z in range(numberOfSimulation):
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',uniKernel,'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);
		qualityUniform[0,z]=mesureQualitry(varioRef, variogram(result[0]), connectivityRef, computeConnectivity(result[0],connectStep));
		qualityUniform[1,z]=mesureQualitry2(varioRef, variogram(result[0]), connectivityRef, computeConnectivity(result[0],connectStep));
		requestId+=1
		misc.imsave('../figures/uniformSimulation'+str(z)+'.png', result[0])


	meanQuality=qualityUniform.mean(1);

	numpy.savez('uniformKer.npz', qualityUniform=qualityUniform);

	print(meanQuality)

	misc.imsave('../figures/source.png', source)

