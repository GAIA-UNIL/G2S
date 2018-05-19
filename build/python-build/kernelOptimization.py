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

print(len(sys.argv))

if len(sys.argv)>1 :
	file_name = sys.argv[1]
	fp = open(file_name)
	serverAddressList = fp.read().splitlines()

print(serverAddressList)

##
#list Of kernel

# possible alphas
alpha=[0,0.01,0.05,0.2,0.5,0.7,0.8,0.9,0.95,0.98,0.995,1];

distanceMtrix=numpy.zeros([81,81]);
distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
distanceMtrix=ndimage.distance_transform_edt(distanceMtrix)
kernel=[];

# create kernels

for alpha_value in alpha:
	print(alpha_value)
	type=[];
	functionName=[];
	loaclDistance=distanceMtrix*alpha_value;
	#uniform
	functionName.append("uniform");
	type.append(numpy.single(loaclDistance<1));
	#triangular
	functionName.append("triangular");
	type.append(numpy.maximum(1-loaclDistance,0));
	#Epanechnikov
	functionName.append("Epanechnikov");
	type.append(numpy.maximum(3/4*(1-numpy.square(loaclDistance)),0));
	#Quartic
	functionName.append("Quartic");
	type.append(numpy.maximum(15/16*numpy.square(1-numpy.square(loaclDistance)),0));
	#Triweight
	functionName.append("Triweight");
	type.append(numpy.maximum(35/32*numpy.power(1-numpy.square(loaclDistance),3),0));
	#Tricube
	functionName.append("Tricube");
	type.append(numpy.maximum(70/81*numpy.power(1-numpy.power(loaclDistance,3),3),0));
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

numberOfSimulation=20
numberOfThreadProJob=4

val=numpy.full([len(kernel), len(kernel[0]), numberOfSimulation],numpy.nan);
if os.path.exists('./simErrorMap.npy') :
	val=numpy.load('simErrorMap.npy')
	print(val)
varioRef=variogram(source);

# from itertools import product
# from multiprocessing import Pool
# def worker(t):
# 	x, y, z = t
# 	id=multiprocessing.current_process();
# 	print(id)
# 	# result=g2s('-sa',serverAddressList[prosId],'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',kernel[x][y],'-k',1.5,'-n',50,'-s',numberOfSimulation,'-j',4);
# 	# val[x,y,z]=mesureQualitry(varioRef,variogram(result[0]));
	# print(multiprocessing.some)
from itertools import product
from queue import Queue
from threading import Thread
import time

#  worker

def worker(queue, address):
	while True:
		item = queue.get()
		print("get job")
		if item is None:
			print("died")
			break
		x, y, z = item 
		if not numpy.isnan(val[item]):
			queue.task_done()
			continue
		print(item,address)
		# time.sleep(0.1)
		result=g2s('-sa',address,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-ki',kernel[x][y],'-k',1.5,'-n',50,'-s',z,'-j',numberOfThreadProJob);
		val[item]=mesureQualitry(varioRef,variogram(result[0]));
		queue.task_done()

# prepare jobs

queue = Queue()
for t in product(range(0,len(kernel)), range(0, len(kernel[0])), range(0,numberOfSimulation)):
	queue.put(t)

# create threads

threads = [Thread(target=worker, args=(queue, address)) for address in serverAddressList]


# run jobs
for t in threads:
	t.start()

# save results

def saveData():
	print("save")
	numpy.save("simErrorMap",val)

cpt=0;
while not queue.empty():
	time.sleep(1)
	cpt=cpt+1;
	if cpt%(1*60)==0 :
		saveData()

queue.join()
saveData()
