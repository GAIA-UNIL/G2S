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


sizeKernel=5;

distanceMtrix=numpy.zeros([sizeKernel,sizeKernel]);
distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
distanceMtrix=ndimage.distance_transform_edt(distanceMtrix)
kernels=[];

NumberOfKernel=2;
numberOfSimulation=10

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

if os.path.exists('./kernelSet.npz') :
	data = np.load('kernelSet.npz')
	kernels=data['kernels']
	iteration= data['iteration']
	numberOfSimulation=data['numberOfSimulation']
	probPower=data['probPower']

numberOfThreadProJob=2

quality=numpy.full([NumberOfKernel,numberOfSimulation],numpy.nan);
probability=numpy.power(probPower,range(NumberOfKernel))
varioRef=variogram(source);

from itertools import product
from queue import Queue
from threading import Thread
import time

#  worker
def worker(queue, address):
	while True:
		item = queue.get()
		if item is None:
			print("died")
			break
		x,z = item 
		if not numpy.isnan(val[item]):
			queue.task_done()
			continue
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


while(maxIteration>iteration){
	if cpt%(iteration)==0 :
		saveData()
	for t in product(range(0,len(kernels)), range(0, lnumberOfSimulation)):
		queue.put(t)
	queue.join()

	meanQualityPosition=quality.mean(1).argsort();
	newKernels=[];
	selection=numpy.random.rand(len(kernels),2)*((1-math.power(probPower,len(kernels)+1))/(1-probPower))
	position=numpy.floor(numpy.log(selection)/math.log(probPower)).astype('int');
	for _ in range(len(kernels)):
		newKernels.append(mergeKernel(kernels[position[x][0]],kernels[position[x][0]],mixingRatio))
	np.random.permutation(len(kernels))
	for x in range(int(ceil(len(kernels)*muationfactor))):
		newKernels[x]=mutateKernel(newKernels[x],muationRatio)

	iteration=iteration+1
}


# from itertools import product
# from multiprocessing import Pool
# def worker(t):
# 	x, y, z = t
# 	id=multiprocessing.current_process();
# 	print(id)
# 	# result=g2s('-sa',serverAddressList[prosId],'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',kernel[x][y],'-k',1.5,'-n',50,'-s',numberOfSimulation,'-j',4);
# 	# val[x,y,z]=mesureQualitry(varioRef,variogram(result[0]));
	# print(multiprocessing.some)




# prepare jobs






# save results

def saveData():
	print("save")
	np.savez('kernelSet.npz', kernels=kernels, iteration=iteration, numberOfSimulation=numberOfSimulation, probPower=probPower)

cpt=0;
while not queue.empty():
	time.sleep(1)
	cpt=cpt+1;
	if cpt%(1*10)==0 :
		saveData()

queue.join()
saveData()
# stop workers
for i in range(len(serverAddressList)):
    queue.put(None) 
for t in threads:
    t.join()

