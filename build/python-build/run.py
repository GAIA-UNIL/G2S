#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
import matplotlib.pyplot as plt
import numpy
from g2s import run as g2s

image=misc.imread('source.png')

print(type(image))
serverAddress='localhost';
test= g2s('-sa',serverAddress,'-a','echo','-ti',image,'-dt',numpy.zeros(shape=(1,1)));

plt.imshow(test[0])
plt.show()

data=g2s('-sa',serverAddress,'-a','qs','-ti',image,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100);
plt.imshow(data[0])
plt.show()

data=g2s('-sa',serverAddress,'-a','qs','-ti',image,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-W_GPU');

plt.imshow(data[0])
plt.show()
