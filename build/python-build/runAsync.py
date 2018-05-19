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

id=g2s('-sa',serverAddress,'-a','qs','-ti',source,'-di',numpy.nan*numpy.ones(shape=(200,200)),'-dt',numpy.zeros(shape=(1,1)),'-k',1.5,'-n',50,'-s',100,'-submitOnly');
print("id is :",id)
data=g2s('-sa',serverAddress,'-statusOnly',id);
print("progression is  :",data, " %")
data=g2s('-sa',serverAddress,'-waitAndDownload',id);
plt.imshow(data[0])
plt.show()