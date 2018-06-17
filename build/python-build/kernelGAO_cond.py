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

def randonKernel():
	ker=numpy.random.rand(sizeKernel,sizeKernel,dimProblem)
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker

def mutateKernel(ker1, ratio):
	ker=numpy.maximum(ker1.copy()+numpy.random.randn(sizeKernel,sizeKernel,dimProblem)*ratio,0);
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker

def mergeKernel(ker1, ker2, ratio):
	places=numpy.random.rand(sizeKernel,sizeKernel,dimProblem)<ratio
	ker=ker1.copy()
	ker[places]=ker2[places];
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker


## main code
if __name__ == "__main__":
	source=numpy.load('training.npy');#img_to_array('training.tiff')
	destinationRef=numpy.load('target.npy');#img_to_array('target.tiff')
	source=source[:200,:200,:]
	destinationRef=destinationRef[:200,:200,:]

	destination=destinationRef.copy();
	destination[:,:,3:]=numpy.nan;

	serverAddressList=['localhost'];

	maxIteration=10000;

	if len(sys.argv)>1 :
		maxIteration = int(sys.argv[1])

	if len(sys.argv)>2 :
		file_name = sys.argv[2]
		fp = open(file_name)
		serverAddressList = fp.read().splitlines()

	print(serverAddressList)


	sizeKernel=9;
	dimProblem=min(source.shape[2],destinationRef.shape[2])

	print(dimProblem)

	distanceMtrix=numpy.zeros([sizeKernel,sizeKernel]);
	distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
	distanceMtrix=ndimage.distance_transform_edt(distanceMtrix)
	kernels=[];

	NumberOfKernel=10;
	numberOfSimulation=2
	saveRate=1;


	# create kernels
	for x in range(0,NumberOfKernel):
		kernels.append(randonKernel())
	iteration=0
	probPower=1/2
	mixingRatio=0.3
	muationfactor=0.5;
	muationRatio=0.3;
	ratioSelection=0.25;
	idValue=2000000;
	convergance=numpy.full([maxIteration,NumberOfKernel],numpy.nan)

	if os.path.exists('./kernelSet_cond.npz') :
		data = numpy.load('kernelSet_cond.npz')
		kernels=data['kernels']
		iteration= data['iteration']
		numberOfSimulation=data['numberOfSimulation']
		probPower=data['probPower']
		convergance=data['convergance'];
		idValue=int(data['idValue'])

	if maxIteration>convergance.shape[0]:
		convergance.resize(maxIteration,NumberOfKernel)

	numberOfThreadProJob=1

	quality=numpy.full([NumberOfKernel,numberOfSimulation],numpy.nan);
	probability=numpy.power(probPower,range(NumberOfKernel))

	resultPtr = numpy.ctypeslib.as_ctypes(quality)
	shared_array = sharedctypes.RawArray(resultPtr._type_, resultPtr)

	from itertools import product
	from multiprocessing import JoinableQueue
	from multiprocessing import Process
	import time

	def saveData():
		print("save")
		numpy.savez('kernelSet_cond.npz', kernels=kernels, iteration=iteration, numberOfSimulation=numberOfSimulation, probPower=probPower, convergance=convergance, idValue=idValue)

	#  worker
	def worker(queue, address):
		qualityLocal = numpy.ctypeslib.as_array(shared_array)
		random.seed()
		while True:
			item = queue.get()
			if item is None:
				print("died")
				break
			z,x ,requestId, kernel= item
			# time.sleep(0.1)
			#path=numpy.random.permutation(source.shape[0]*source.shape[1])
			#path=path.reshape((source.shape[0],source.shape[1])).astype('float');
			result=g2s('-sa',address,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,source.shape[2])),'-ki',kernel,'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);#,'-sp',path
			#result=g2s('-sa',address,'-silent','-waitAndDownload',requestId) #debug
			qualityLocal[x,z]=mesureQualitry(destinationRef,result[0]);
			#quality[x,z]=numpy.random.rand();
			queue.task_done()

	queue = JoinableQueue()

	# create threads
	threads = [Process(target=worker, args=(queue, address)) for address in serverAddressList]

	# run jobs
	for t in threads:
		print("start process")
		t.start()

	quality = numpy.ctypeslib.as_array(shared_array)

	while maxIteration>iteration :
		if iteration%(saveRate)==0 :
			saveData()
		for t in product( range(0, numberOfSimulation),range(0,len(kernels))):
			queue.put(t + (idValue,)+ (kernels[t[1]],))
			idValue=idValue+1;
		queue.join()

		meanQuality=quality.mean(1);
		convergance[iteration,:]=meanQuality

		meanQualityPosition=meanQuality.argsort();
		newKernels=[];
		#(p)^k ==> klog(p)
		position=numpy.empty([1, 1])
		while position.shape[0]<=len(kernels) :
			#selection=numpy.random.rand(len(kernels)*10,2)*((1-math.pow(probPower,len(kernels)))/(1-probPower))
			#selection=numpy.random.rand(len(kernels)*10,2)*((1-math.pow(probPower,len(kernels)))/(1-probPower))
			#position=numpy.floor(numpy.log(selection*(probPower-1)+1)/math.log(probPower)).astype('int');
			position=numpy.floor(numpy.random.rand(len(kernels)*10,2)*ratioSelection*len(kernels)).astype('int')
			survivor=(numpy.unique(position[numpy.where(position[:len(kernels),0]==position[:len(kernels),1]),0]))
			position=numpy.delete(position, numpy.where(position[:,0]==position[:,1]),0);
		survivor=[]
		position=position[:len(kernels)-len(survivor)]
		#print(position)
		#print(survivor)
		for x in range(len(survivor)):
			newKernels.append(kernels[meanQualityPosition[survivor[x]]].copy());
		for x in range(len(kernels)-len(survivor)):
			newKernels.append(mergeKernel(kernels[meanQualityPosition[position[x,0]]],kernels[meanQualityPosition[position[x,1]]],mixingRatio))
		perm=numpy.random.permutation(len(kernels))
		offset=0;
		print(numpy.stack(kernels,axis=-1).shape)
		variability=numpy.stack(kernels,axis=-1)[:,:,:,meanQualityPosition[:math.floor(ratioSelection*len(kernels))]].std(3);
		for _ in range(int(math.ceil(len(kernels)*muationfactor))):
			newKernels[perm[offset]]=mutateKernel(newKernels[perm[offset]],numpy.maximum(variability,1/(1+iteration/10)))
			offset+=1
		# for _ in range(int(math.ceil(len(kernels)*muationfactor))):
		# 	newKernels[perm[offset]]=randonKernel()
		# 	offset+=1
		print("new generation :", iteration)
		kernels=newKernels;
		iteration=iteration+1

	queue.join()
	saveData()

	print(quality)
	# stop workers
	for i in range(len(serverAddressList)):
		queue.put(None) 
	for t in threads:
		t.join()

