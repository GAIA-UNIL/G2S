#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy
import math
from g2s import run as g2s

source=misc.imread('source.png')/255.
destination=numpy.nan*numpy.ones(shape=(200,200));
serverAddress='localhost';
verbose = True;

# conditional
pourcantage=0.25; # 0.25%
conDestination=destination;
sizeDest=numpy.size(conDestination);
position1=numpy.random.permutation(sizeDest)[1:math.ceil(sizeDest*pourcantage/100)];
position2=numpy.random.permutation(sizeDest)[1:math.ceil(sizeDest*pourcantage/100)];
conDestination.reshape(sizeDest,1)[position1]=source.reshape(sizeDest,1)[position2];

# simple echo
data=g2s('-sa',serverAddress,'-a','echo','-ti',source,'-dt',numpy.zeros(shape=(1,1)))
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# simple unconditional simulation with QS
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# simple unconditional simulation with QS with GPU if integrated GPU avaible
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-W_GPU');
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# simulation with random value at random position
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',conDestination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# simple non conditional simulation with multi TI of differente size
data=g2s('-sa',serverAddress,'-a','qs','-ti',source[:,0:149].copy(),numpy.rot90(source,1).copy(),numpy.rot90(source[:,0:174],2).copy(),numpy.rot90(source[:,0:149],3).copy(),'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# simulation with a fixed path, row path
path= numpy.arange(0,sizeDest).reshape((200,200));
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-sp',path);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

#simulation with a fixed path, partial random path
path=numpy.random.permutation(sizeDest).astype(float);
path[path>math.ceil(sizeDest/2)]=-numpy.inf;
path=path.reshape((200,200));
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-sp',path);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# specifing a kernel
kernel=numpy.zeros(shape=(101,101));
kernel[51,51]=1;
kernel=numpy.exp(-0.1*ndimage.morphology.distance_transform_edt(1-kernel));
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-ki',kernel);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# Multi variete
source3=numpy.stack([source,source,source],2);
destination3=numpy.stack([destination,destination,destination],2);
data=g2s('-sa',serverAddress,'-a','qs','-ti',source3,'-di',destination3,'-dt',numpy.zeros(shape=(1,3)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# Multi-threaded, if suported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# ds mode
data=g2s('-sa',serverAddress,'-a','ds-l','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-th',0.05,'-f',0.3,'-n',50,'-s',100);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# Categorical Mode
# creation of the image
sourceCat=numpy.matlib.repmat(numpy.eye(2),25,25);
sourceCat=numpy.hstack([numpy.vstack([sourceCat*1,sourceCat*2]),numpy.vstack([sourceCat*3,sourceCat*4])]);
plt.imshow(sourceCat)
plt.show(block=False)
plt.pause(0.1)
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,'-di',numpy.nan*numpy.ones(shape=(100,100)),'-dt',numpy.ones(shape=(1,1)),'-k',1,'-n',50,'-s',100,'-j',1);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,numpy.rot90(sourceCat,1).copy(),numpy.rot90(sourceCat,2).copy(),numpy.rot90(sourceCat,3).copy(),'-di',numpy.nan*numpy.ones(shape=(100,100)),'-dt',numpy.ones(shape=(1,1)),'-k',1,'-n',50,'-s',100,'-j',1);
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)


# G2S interface options
# async submission
id=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-submitOnly');
if verbose:
	print("id is :",id)
plt.pause(5)
# progression check
progression=g2s('-sa',serverAddress,'-statusOnly',id);
if verbose:
	print("progression is  :",progression, " %")
# Download data
data=g2s('-sa',serverAddress,'-waitAndDownload',id);  # '-kill' to interrupt a job
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# silent mode
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-silent');
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# without timeout
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',destination,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-noTO');
plt.imshow(data[0])
if verbose:
	print(data)
plt.show(block=False)
plt.pause(0.1)

# shutdown the server
g2s('-sa',serverAddress,'-shutdown');

plt.show()
