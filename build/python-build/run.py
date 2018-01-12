#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy
import math
from g2s import run as g2s

source=misc.imread('source.png')
serverAddress='localhost';
# simple echo
data=g2s('-sa',serverAddress,'-a','echo','-ti',source,'-dt',numpy.zeros(shape=(1,1)))
plt.imshow(data[0])

# simple unconditional simulation with QS
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])

# %with GPU is integrated GPU avaible
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-W_GPU');
plt.imshow(data[0])

# simple conditional simulation with QS
pourcantage=0.25;#  0.25%
dest=numpy.nan*numpy.ones(shape=(200,200));
sizeDest=numpy.size(dest);

position=numpy.random.permutation(sizeDest)[1:math.ceil(sizeDest*pourcantage/100)];

dest.reshape(sizeDest,1)[position]=source.reshape(sizeDest,1)[position];

# % simuulation of the source
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',dest,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])

# % simulation with random value at random position
dest=dest*numpy.nan;
position2=numpy.random.permutation(sizeDest)[1:math.ceil(sizeDest*pourcantage/100)];
dest.reshape(sizeDest,1)[position2]=source.reshape(sizeDest,1)[position];

# % simuulation of the source
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',dest,'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])

# simple non conditional simulation with multi TI of differente size
data=g2s('-sa',serverAddress,'-a','qs','-ti',source[:,0:149],numpy.rot90(source,1),numpy.rot90(source[:,0:174],2),numpy.rot90(source[:,0:149],3),'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])

# simulation with a fixed path
# % row path
path= numpy.arange(0,sizeDest).reshape((200,200));
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-sp',path);
plt.imshow(data[0])

# % partial random path
path=numpy.random.permutation(sizeDest)
path[math.ceil(sizeDest/2):-1]=-numpy.inf;
path=path.reshape((200,200));
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-sp',path);
plt.imshow(data[0])

# specifing a kernel
# % ? kernel need to be define for each variable 

kernel=numpy.zeros(shape=(101,101));
kernel[51,51]=1;
kernel=numpy.exp(-0.1*ndimage.morphology.distance_transform_edt(1-kernel));

data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-ki',kernel);
plt.imshow(data[0])

# Multi variete
source3=numpy.stack([source,source,source],2); # need to fine better example
data=g2s('-sa',serverAddress,'-a','qs','-ti',source3,'-di',numpy.nan*numpy.ones(shape=(200,200,3)),'-dt',numpy.zeros(shape=(1,3)),'-k',2,'-n',50,'-s',100);
plt.imshow(data[0])

# Multi-threaded, if suported
nbThreads=4;
data=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-j',nbThreads);
plt.imshow(data[0])

# ds mode
data=g2s('-sa',serverAddress,'-a','ds-l','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-th',10,'-f',0.3,'-n',50,'-s',100);
plt.imshow(data[0])

# Categorical Mode
# % creation of the image
sourceCat=numpy.matlib.repmat(numpy.eye(2),25,25);
sourceCat=numpy.hstack([numpy.vstack([sourceCat*1,sourceCat*2]),numpy.vstack([sourceCat*3,sourceCat*4])]);
plt.imshow(sourceCat)
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,'-di',numpy.nan*numpy.ones(shape=(100,100)),'-dt',numpy.ones(shape=(1,1)),'-k',1,'-n',50,'-s',100,'-j',1);
plt.imshow(data[0])
data=g2s('-sa',serverAddress,'-a','qs','-ti',sourceCat,numpy.rot90(sourceCat,1),numpy.rot90(sourceCat,2),numpy.rot90(sourceCat,3),'-di',numpy.nan*numpy.ones(shape=(100,100)),'-dt',numpy.ones(shape=(1,1)),'-k',1,'-n',50,'-s',100,'-j',1);
plt.imshow(data[0])
