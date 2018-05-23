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

maxIteration=10000;

if len(sys.argv)>1 :
	maxIteration = sys.argv[1]

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
saveRate=20;

def randonKernel():
	ker=numpy.random.rand(sizeKernel,sizeKernel)
	ker=ker/numpy.sum(ker)
	return ker

def mutateKernel(ker, ratio):
	places=numpy.random.rand(sizeKernel,sizeKernel)<ratio
	randKer=numpy.random.rand(sizeKernel,sizeKernel)
	ker[places]=randKer[places];
	ker=ker/numpy.sum(ker)
	return ker

def mergeKernel(ker1, ker2, ratio):
	places=numpy.random.rand(sizeKernel,sizeKernel)<ratio
	randKer=numpy.random.rand(sizeKernel,sizeKernel)
	ker1[places]=ker2[places];
	ker1=ker1/numpy.sum(ker1)
	return ker1


# create kernels
for x in range(0,NumberOfKernel):
	kernels.append(randonKernel())
iteration=0
probPower=1/2
mixingRatio=0.3
muationfactor=0.02;
muationRatio=0.3;
convergance=numpy.full([maxIteration,NumberOfKernel],numpy.nan)

if os.path.exists('./kernelSet.npz') :
	data = numpy.load('kernelSet.npz')
	kernels=data['kernels']
	iteration= data['iteration']
	numberOfSimulation=data['numberOfSimulation']
	probPower=data['probPower']
	convergance=data['convergance'];

if maxIteration>convergance.shape[0]:
	convergance.resize(maxIteration,NumberOfKernel)

numberOfThreadProJob=2

quality=numpy.full([NumberOfKernel,numberOfSimulation],numpy.nan);
probability=numpy.power(probPower,range(NumberOfKernel))
varioRef=variogram(source);

from itertools import product
from queue import Queue
from threading import Thread
import time

def saveData():
	print("save")
	numpy.savez('kernelSet.npz', kernels=kernels, iteration=iteration, numberOfSimulation=numberOfSimulation, probPower=probPower, convergance=convergance)

#  worker
def worker(queue, address):
	while True:
		item = queue.get()
		if item is None:
			print("died")
			break
		x,z = item
		# time.sleep(0.1)
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',kernels[x],'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob);
		quality[x,z]=mesureQualitry(varioRef,variogram(result[0]));
		queue.task_done()

queue = Queue()

# create threads
threads = [Thread(target=worker, args=(queue, address)) for address in serverAddressList]

# run jobs
for t in threads:
	t.start()


while maxIteration>iteration :
	if iteration%(saveRate)==0 :
		saveData()
	for t in product(range(0,len(kernels)), range(0, numberOfSimulation)):
		queue.put(t)
	queue.join()

	meanQuality=quality.mean(1);
	convergance[iteration,:]=meanQuality

	meanQualityPosition=meanQuality.argsort();
	newKernels=[];
	#(p)^k ==> klog(p)
	position=numpy.empty([1, 1])
	while position.shape[0]<=len(kernels) :
		selection=numpy.random.rand(len(kernels)*10,2)*((1-math.pow(probPower,len(kernels)))/(1-probPower))
		position=numpy.floor(numpy.log(selection*(probPower-1)+1)/math.log(probPower)).astype('int');
		survivor=(numpy.unique(position[numpy.where(position[:len(kernels),0]==position[:len(kernels),1]),0]))
		position=numpy.delete(position, numpy.where(position[:,0]==position[:,1]),0);
	position=position[:len(kernels)-len(survivor)]
	#print(position)
	#print(survivor)
	for x in range(len(survivor)):
		newKernels.append(kernels[meanQualityPosition[survivor[x]]]);
	for x in range(len(kernels)-len(survivor)):
		newKernels.append(mergeKernel(kernels[meanQualityPosition[position[x,0]]],kernels[meanQualityPosition[position[x,1]]],mixingRatio))
	perm=numpy.random.permutation(len(kernels))
	offset=0;
	for _ in range(int(math.ceil(len(kernels)*muationfactor))):
		newKernels[perm[offset]]=mutateKernel(newKernels[x],muationRatio)
		offset+=1
	for _ in range(int(math.ceil(len(kernels)*muationfactor))):
		newKernels[perm[offset]]=mutateKernel(newKernels[x],muationRatio)
		offset+=1
	kernels=newKernels;
	iteration=iteration+1

queue.join()
saveData()
# stop workers
for i in range(len(serverAddressList)):
    queue.put(None) 
for t in threads:
    t.join()

