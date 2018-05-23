#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy
from scipy import misc
from scipy import ndimage
import math
import matplotlib.pyplot as plt

def variogram( x1 ):
	n=numpy.size(x1,0)
	p=numpy.size(x1,1)
	nr2 = math.ceil((2*n-1)/8)*8;
	nc2 = math.ceil((2*p-1)/8)*8;

	x1id		= numpy.logical_not(numpy.isnan(x1))
	x1[numpy.isnan(x1)]  = 0;
	fx1 		= numpy.fft.fft2(numpy.pad(x1,((0, nr2-n),(0, nc2-p)), 'constant', constant_values=(0)));
	fx1_x1		= numpy.fft.fft2(numpy.pad(numpy.square(x1), ((0, nr2-n),(0, nc2-p)), 'constant', constant_values=(0)));
	fx1id 		= numpy.fft.fft2(numpy.pad(x1id.astype('float'),((0, nr2-n),(0, nc2-p)), 'constant', constant_values=(0)));
	nh11 		= numpy.round(numpy.real(numpy.fft.ifft2(numpy.multiply(numpy.conj(fx1id),fx1id))));
	gh11 		= numpy.real(numpy.fft.ifft2(numpy.add(numpy.add(
		numpy.multiply(numpy.conj(fx1id), fx1_x1),
		numpy.multiply(numpy.conj(fx1_x1),fx1id)),
		(-2 * numpy.multiply(numpy.conj(fx1),fx1)))));
	gh11 		= numpy.divide(gh11,numpy.maximum(nh11,1)) / 2;
	# nh11 		= numpy.concatenate([[nh11[:n-1,:p-1], nh11[:n-1,-p+1:]],[nh11[-n+1:,:p-1], nh11[-n+1:,-p+1:]]]);
	# gh11 		= numpy.concatenate([[gh11[:n-1,:p-1], gh11[:n-1,-p+1:]],[gh11[-n+1:,:p-1], gh11[-n+1:,-p+1:]]]);
	gh11 		= numpy.fft.fftshift(gh11);
	nh11 		= numpy.fft.fftshift(nh11);

	return [gh11[int(numpy.size(gh11,0)/2-numpy.size(x1,0)):int(numpy.size(gh11,0)/2+numpy.size(x1,0))-1,
				 int(numpy.size(gh11,1)/2-numpy.size(x1,1)):int(numpy.size(gh11,1)/2+numpy.size(x1,1))-1],
	 		nh11[int(numpy.size(nh11,0)/2-numpy.size(x1,0)):int(numpy.size(nh11,0)/2+numpy.size(x1,0))-1,
				 int(numpy.size(nh11,1)/2-numpy.size(x1,1)):int(numpy.size(nh11,1)/2+numpy.size(x1,1))-1]];

# def omniDirectionalvariogram( x1 ):
# 	vario=variogram(x1)
# 	dist=numpy.zeros(vario[0].shape);
# 	dist[int(math.ceil(numpy.size(dist,0)/2)),int(math.ceil(numpy.size(dist,1)/2))]=1;
# 	dist=numpy.round(ndimage.distance_transform_edt(dist),0);

# 	x=numpy.unique(dist[:]);
# 	alpha=(dist[:]==x);
# 	print(alpha)
# 	y=numpy.matmul(vario[:],numpy.single(alpha))#numpy.multiply(vario[:]*numpy.single(alpha),numpy.sum(alpha));


# 	return [x,y];

# source=numpy.single(misc.imread('source.png'))/256
# vario=variogram(source);

# print(vario[0].shape)
# print(vario[1].shape)

# plt.imshow(vario[0])
# print(vario[0])
# plt.show()

# plt.imshow(vario[1])
# print(vario[1])
# plt.show()

# vario=omniDirectionalvariogram(source);

# plt.plot(vario[0],vario[1])
# plt.show();
