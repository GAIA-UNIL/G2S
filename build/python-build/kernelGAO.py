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
saveRate=1;

def randonKernel():
	ker=numpy.random.rand(sizeKernel,sizeKernel)
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker

def mutateKernel(ker1, ratio):
	ker=numpy.maximum(ker1.copy()+numpy.random.randn(sizeKernel,sizeKernel)*variability,0);
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker

def mergeKernel(ker1, ker2, ratio):
	places=numpy.random.rand(sizeKernel,sizeKernel)<ratio
	ker=ker1.copy()
	ker[places]=ker2[places];
	norm=numpy.sum(ker)
	if norm<0.001:
		ker=randomKernel()
		norm=1;
	ker=ker/norm
	return ker


# create kernels
for x in range(0,NumberOfKernel):
	kernels.append(randonKernel())
iteration=0
probPower=1/2
mixingRatio=0.3
muationfactor=0.5;
muationRatio=0.3;
ratioSelection=0.15;
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

if maxIteration>convergance.shape[0]:
	convergance.resize(maxIteration,NumberOfKernel)

numberOfThreadProJob=2

quality=numpy.full([NumberOfKernel,numberOfSimulation],numpy.nan);
probability=numpy.power(probPower,range(NumberOfKernel))
varioRef=variogram(source);
connectStep=(numpy.arange(255)+0.5)/255;
connectivityRef=computeConnectivity(source,connectStep);


from itertools import product
from queue import Queue
from threading import Thread
import time

def saveData():
	print("save")
	numpy.savez('kernelSet.npz', kernels=kernels, iteration=iteration, numberOfSimulation=numberOfSimulation, probPower=probPower, convergance=convergance, idValue=idValue)

#  worker
def worker(queue, address):
	while True:
		item = queue.get()
		if item is None:
			print("died")
			break
		x,z ,requestId = item
		# time.sleep(0.1)
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',kernels[x],'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);
		quality[x,z]=mesureQualitry(varioRef, variogram(result[0]), connectivityRef, computeConnectivity(result[0],connectStep));
		#quality[x,z]=numpy.random.rand();
		queue.task_done()

queue = Queue()

# create threads
threads = [Thread(target=worker, args=(queue, address)) for address in serverAddressList]

# run jobs
for t in threads:
	t.start()


wwhile maxIteration>iteration :
	if iteration%(saveRate)==0 :
		saveData()
	for t in product(range(0,len(kernels)), range(0, numberOfSimulation)):
		queue.put(t + (idValue,))
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
	variability=numpy.dstack(kernels)[:,:,meanQualityPosition[:math.floor(ratioSelection*len(kernels))]].std(2);
	for _ in range(int(math.ceil(len(kernels)*muationfactor))):
		newKernels[perm[offset]]=mutateKernel(newKernels[perm[offset]],numpy.maximum(variability,1/(1+iteration)))
		offset+=1
	# for _ in range(int(math.ceil(len(kernels)*muationfactor))):
	# 	newKernels[perm[offset]]=randonKernel()
	# 	offset+=1
	print("new generation :", iteration)
	kernels=newKernels;
	iteration=iteration+1

queue.join()
saveData()
# stop workers
for i in range(len(serverAddressList)):
    queue.put(None) 
for t in threads:
    t.join()

