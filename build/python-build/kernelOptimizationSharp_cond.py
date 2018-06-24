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

from scipy.spatial.distance import euclidean
from skimage.measure import compare_ssim as ssim

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



## main code
source=numpy.load('training.npy');#img_to_array('training.tiff')
destinationRef=numpy.load('target.npy');#img_to_array('target.tiff')
source=source[:200,:200,:]
destinationRef=destinationRef[:200,:200,:]
destination=destinationRef.copy();
destination[:,:,3:]=numpy.nan;
serverAddressList=['localhost'];

print(len(sys.argv))

if len(sys.argv)>1 :
	file_name = sys.argv[1]
	fp = open(file_name)
	serverAddressList = fp.read().splitlines()

print(serverAddressList)

##
#list Of kernel

# possible alphas
alpha=numpy.arange(100)/100.;

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
	# #uniform
	# functionName.append("uniform");
	# type.append(numpy.single(loaclDistance<1));
	# #triangular
	# functionName.append("triangular");
	# type.append(numpy.maximum(1-loaclDistance,0));
	# #Epanechnikov
	# functionName.append("Epanechnikov");
	# type.append(numpy.maximum(3/4*(1-numpy.square(loaclDistance)),0));
	# #Quartic
	# functionName.append("Quartic");
	# type.append(numpy.maximum(15/16*numpy.square(1-numpy.square(loaclDistance)),0));
	# #Triweight
	# functionName.append("Triweight");
	# type.append(numpy.maximum(35/32*numpy.power(1-numpy.square(loaclDistance),3),0));
	# #Tricube
	# functionName.append("Tricube");
	# type.append(numpy.maximum(70/81*numpy.power(1-numpy.power(loaclDistance,3),3),0));
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

numberOfSimulation=25
numberOfThreadProJob=2

val=numpy.full([len(kernel), len(kernel[0]), numberOfSimulation,1],numpy.nan);
if os.path.exists('./simErrorMap_cond.npy') :
	val=numpy.load('simErrorMap_cond.npy')
	print(val)

from itertools import product
from queue import Queue
from threading import Thread
import time

idValue=4000000;

#  worker

def worker(queue, address):
	while True:
		item = queue.get()
		if item is None:
			print("died")
			break
		x, y, z, requestId = item 
		if not numpy.isnan(val[x, y, z]).any():
			queue.task_done()
			continue
		# time.sleep(0.1)
		path=numpy.random.permutation(source.shape[0]*source.shape[1])
		path=path.reshape((source.shape[0],source.shape[1])).astype('float');
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,source.shape[2])),'-ki',kernel[x][y],'-k',1.5,'-n',50,'-s',z,'-sp',path,'-j',numberOfThreadProJob,'-silent','-noTO','-id',requestId);
		val[x, y, z,:]=mesureQualitry(destinationRef,result[0]);
		queue.task_done()

# prepare jobs

queue = Queue()
for t in product(range(0,len(kernel)), range(0, len(kernel[0])), range(0,numberOfSimulation)):
	queue.put(t+ (idValue,))
	idValue=idValue+1;

# create threads

threads = [Thread(target=worker, args=(queue, address)) for address in serverAddressList]


# run jobs
for t in threads:
	t.start()

# save results

def saveData():
	print("save")
	numpy.save("simErrorMap_cond",val)

cpt=0;
while not queue.empty():
	time.sleep(1)
	cpt=cpt+1;
	if cpt%(1*60)==0 :
		saveData()

queue.join()
saveData()
# stop workers
for i in range(len(serverAddressList)):
    queue.put(None) 
for t in threads:
    t.join()

