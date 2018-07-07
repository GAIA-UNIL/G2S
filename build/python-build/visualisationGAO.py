#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.colors as colors
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
fileName='./kernelSet.npz';


if len(sys.argv)>1 :
	fileName = sys.argv[1]
	print(fileName)


print(serverAddressList)


sizeKernel=7;

distanceMtrix=numpy.zeros([sizeKernel,sizeKernel]);
distanceMtrix[int(numpy.ceil(numpy.size(distanceMtrix,0)/2)),int(numpy.ceil(numpy.size(distanceMtrix,1)/2))]=1;
distanceMtrix=ndimage.distance_transform_edt(distanceMtrix)
kernels=[];

NumberOfKernel=128;
numberOfSimulation=10
saveRate=5;
iteration=0
probPower=1/2
mixingRatio=0.3
muationfactor=0.02;
muationRatio=0.3;
ratioSelection=0.25;
idValue=1;
convergance=numpy.full([maxIteration,NumberOfKernel],numpy.nan)

if os.path.exists(fileName) :
	data = numpy.load(fileName)
	kernels=data['oldKernel']
	iteration= data['iteration']
	numberOfSimulation=data['numberOfSimulation']
	probPower=data['probPower']
	convergance=data['convergance'];
	idValue=data['idValue']

nanList=numpy.argwhere(numpy.logical_not(numpy.isnan(convergance)))
maxId=nanList[:,0].max().astype('int');

print(maxId)

import colorsys
spacing=numpy.linspace(0,1,256)
colorsList=plt.cm.nipy_spectral(spacing);
for x in range(colorsList.shape[0]):
	r,g,b,alpha=colorsList[x,:];
	h,l,s=colorsys.rgb_to_hls(r,g,b)
	l=1-l;
	r,g,b=colorsys.hls_to_rgb(h,l,s)
	colorsList[x,:]=r,g,b,alpha;



#print(numpy.stack([spacing,colorsList[:,0],colorsList[:,0]],1));
cdict1 = {'red':   numpy.stack([spacing,colorsList[:,0],colorsList[:,0]],1),

         'green': numpy.stack([spacing,colorsList[:,1],colorsList[:,1]],1),

         'blue':  numpy.stack([spacing,colorsList[:,2],colorsList[:,2]],1)
        }
cmapPerso=colors.LinearSegmentedColormap('nipy_spectral_LInv',cdict1);
cmapPerso.set_bad('black',1.)
#scmapPersoRegister=plt.register_cmap(cmap=cmapPerso);

## load extra data
maxId=maxId

if len(sys.argv)>2 and os.path.exists(sys.argv[2]) :
	extraData = numpy.load(sys.argv[2])
	qualityUniform=extraData['qualityUniform'];

	tmp=int(math.floor(qualityUniform.shape[1]/numberOfSimulation));
	qualityUniform=numpy.reshape(qualityUniform[:,:tmp*numberOfSimulation],[qualityUniform.shape[0],tmp,numberOfSimulation]);

	qualityUniform=qualityUniform.mean(2);
	print(qualityUniform.shape)

if len(sys.argv)>3 and os.path.exists(sys.argv[3]) :
	extraData = numpy.load(sys.argv[3])
	qualityBest=extraData['qualityBests'];
	print(qualityBest.shape)


	tmp=int(math.floor(qualityBest.shape[1]/numberOfSimulation));
	qualityBest=numpy.reshape(qualityBest[:,:tmp*numberOfSimulation],[qualityBest.shape[0],tmp,numberOfSimulation]);
	print(qualityBest.shape)
	qualityBest=qualityBest.mean(2);
	print(qualityBest.shape)



from matplotlib.colors import LogNorm
figureSize= (16,9)
#figureSize= (8,5)
plt.figure(num=None, figsize=figureSize, dpi=120)
amountSkiped=14;
#plt.plot( range(maxId-amountSkiped), convergance[:maxId-amountSkiped].min(1),':',label='p0',lw=2);
plt.plot( range(maxId-amountSkiped), numpy.percentile(convergance[:maxId-amountSkiped], 5,1),'-.',label='p5',lw=2);
plt.plot( range(maxId-amountSkiped), numpy.percentile(convergance[:maxId-amountSkiped],50,1),'-',label='p50',lw=2);
plt.plot( range(maxId-amountSkiped), numpy.percentile(convergance[:maxId-amountSkiped],95,1),'-.',label='p95',lw=2);
#plt.plot( range(maxId-amountSkiped), convergance[:maxId-amountSkiped].max(1),':',label='p100',lw=2);
			
if 'qualityUniform' in locals():
	plt.axhline(y=numpy.percentile(qualityUniform[0,:],50,0), xmin=0.0, xmax=1.0, linewidth=1, color='grey',label='Uniform')
	plt.fill_between([-20, maxId+20], qualityUniform[0,:].min(0), qualityUniform[0,:].max(0), color='grey', alpha='0.3')
if 'qualityBest' in locals():
	if qualityBest.shape[0]>0:
		plt.axhline(y=numpy.percentile(qualityBest[1,:],50,0), xmin=0.0, xmax=1.0, linewidth=1, color='lime',label='Trained')
		plt.fill_between([-20, maxId+20], qualityBest[1,:].min(0), qualityBest[1,:].max(0), color='lime', alpha='0.3')

	if qualityBest.shape[0]>2:
		plt.axhline(y=numpy.percentile(qualityBest[2,:],50,0), xmin=0.0, xmax=1.0, linewidth=1, color='cyan', label='static')
		plt.fill_between([-20, maxId+20], qualityBest[2,:].min(0), qualityBest[2,:].max(0), color='cyan', alpha='0.5')
		
	# if qualityBest.shape[0]>4:
	# 	plt.axhline(y=numpy.percentile(qualityBest[4+1,:],50,0), xmin=0.0, xmax=1.0, linewidth=1, color='cyan', label='static')
	# 	plt.fill_between([-20, maxId+20], qualityBest[4+1,:].min(0), qualityBest[4+1,:].max(0), color='cyan', alpha='0.5')

	if qualityBest.shape[0]>6:
		plt.axhline(y=numpy.percentile(qualityBest[6+1,:],50,0), xmin=0.0, xmax=1.0, linewidth=1, color='cyan', label='static')
		plt.fill_between([-20, maxId+20], qualityBest[6+1,:].min(0), qualityBest[6+1,:].max(0), color='cyan', alpha='0.5')


plt.xlim(-2, maxId+10)
plt.legend(loc=1, fontsize=20)

plt.xlabel('Iterations', fontsize=26)
if kernels[0].ndim < 3:
	plt.ylabel('Variogram - $\epsilon$', fontsize=26)
else:
	plt.ylabel('SSIM', fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=20)

plt.savefig('../figures/conv_cond.png', transparent=True)
plt.show()


meanQualityPosition=convergance[maxId,:].argsort();
maxPosition=int(ratioSelection*len(kernels));
print(maxPosition)
print(meanQualityPosition[:maxPosition])
maxValue=numpy.percentile(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),99.9)*2;
minValue=0.001;#numpy.percentile(numpy.stack(kernels,axis=-1),10)/2;

for kernel in kernels:
	kernel[kernel>maxValue]=maxValue;
	kernel[kernel<minValue]=minValue;
	if kernel.ndim>2 :
		kernel[int(math.ceil(sizeKernel/2)),int(math.ceil(sizeKernel/2)),-1]=numpy.inf;
	else :
		kernel[int(math.ceil(sizeKernel/2)),int(math.ceil(sizeKernel/2))]=numpy.inf;
# print(maxValue)


titles=['Blue', 'Green', 'Red', 'Near Infrared'];
import matplotlib
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=1, bitrate=-1, metadata=metadata);

if kernels[0].ndim < 3:

	figureSize=(10,8)

	# for x in range(len(kernels)):
	# 	plt.close()
	# 	# plt.plot(range(sizeKernel*sizeKernel),kernels[x].flatten())
	# 	plt.figure(num=None,figsize=figureSize, dpi=120)
	# 	plt.imshow(kernels[x],vmin=minValue, vmax=maxValue, norm=LogNorm()).set_cmap(cmapPerso)
	# 	plt.colorbar()
	# 	plt.tick_params(
	# 			# axis='both',         # changes apply to the x-axis
	# 			which='both',      # both major and minor ticks are affected
	# 			bottom=False,      # ticks along the bottom edge are off
	# 			top=False,         # ticks along the top edge are off
	# 			left=False,
	# 			right=False,
	# 			labelbottom=False,
	# 			labelleft=False) # labels along the bottom edge are off
	# 	plt.pause(0.01)

	plt.figure(num=None,figsize=figureSize, dpi=120)
	plt.imshow(numpy.median(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1),vmin=minValue, vmax=maxValue,norm=LogNorm()).set_cmap(cmapPerso)
	plt.colorbar().ax.tick_params(labelsize=20)
	plt.tick_params(
				# axis='both',         # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
	plt.savefig('../figures/p50_connected.png', transparent=True)
	plt.show()

	# plt.figure(num=None,figsize=figureSize, dpi=120)
	# plt.imshow(numpy.percentile(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),10,-1),vmin=minValue, vmax=maxValue,norm=LogNorm()).set_cmap(cmapPerso)
	# fig.colorbar()
	# plt.tick_params(
	# 			# axis='both',         # changes apply to the x-axis
	# 			which='both',      # both major and minor ticks are affected
	# 			bottom=False,      # ticks along the bottom edge are off
	# 			top=False,         # ticks along the top edge are off
	# 			left=False,
	# 			right=False,
	# 			labelbottom=False,
	# 			labelleft=False) # labels along the bottom edge are off
	# plt.savefig('../figures/p10_connected.png', transparent=True)
	# plt.show()

	plt.figure(num=None,figsize=figureSize, dpi=120)
	plt.imshow(numpy.mean(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1),vmin=minValue, vmax=maxValue,norm=LogNorm()).set_cmap(cmapPerso)
	plt.colorbar().ax.tick_params(labelsize=20)
	plt.tick_params(
				# axis='both',         # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
	plt.savefig('../figures/mean_connected.png', transparent=True)
	plt.show(block=False)
	plt.show()

else :

	numberOfRows=1;
	figureSize=(16,5)

	# for x in range(len(kernels)):
	# 	plt.close()
	# 	fig, axes = plt.subplots(nrows=numberOfRows, ncols=int(math.ceil(kernels[0].shape[-1]/numberOfRows)),figsize=, dpi=120)
	# 	y=0
	# 	for ax in axes.flat:
	# 		im = ax.imshow(kernels[x].take(y,axis=2),vmin=minValue, vmax=maxValue, norm=LogNorm())
	# 		im.set_cmap(cmapPerso)
	# 		ax.set_title( titles[y], fontsize=20)
	# 		ax.tick_params(
	# 			# axis='both',         # changes apply to the x-axis
	# 			which='both',      # both major and minor ticks are affected
	# 			bottom=False,      # ticks along the bottom edge are off
	# 			top=False,         # ticks along the top edge are off
	# 			left=False,
	# 			right=False,
	# 			labelbottom=False,
	# 			labelleft=False) # labels along the bottom edge are off
	# 		y+=1

	# 	fig.subplots_adjust(right=0.8)
	# 	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	# 	fig.colorbar(im, cax=cbar_ax)
	# 	# plt.plot(range(sizeKernel*sizeKernel),kernels[x].flatten())
	# 	plt.show(block=False)
	# 	plt.pause(0.01)
	# plt.close()


	### video

	# fig=plt.figure(figsize=figureSize, dpi=120)

	# with writer.saving(fig, "writer_test.mp4", 120):
	# 	for x in range(len(kernels)):
			
	# 		axes = fig.subplots(nrows=numberOfRows, ncols=int(math.ceil(kernels[0].shape[-1]/numberOfRows)))
	# 		y=0
	# 		for ax in axes.flat:
	# 			im = ax.imshow(kernels[x].take(y,axis=2),vmin=minValue, vmax=maxValue, norm=LogNorm())
	# 			im.set_cmap(cmapPerso)
	# 			ax.set_title( titles[y], fontsize=20)
	# 			ax.tick_params(
	# 				# axis='both',         # changes apply to the x-axis
	# 				which='both',      # both major and minor ticks are affected
	# 				bottom=False,      # ticks along the bottom edge are off
	# 				top=False,         # ticks along the top edge are off
	# 				left=False,
	# 				right=False,
	# 				labelbottom=False,
	# 				labelleft=False) # labels along the bottom edge are off
	# 			y+=1

	# 		fig.subplots_adjust(right=0.8)
	# 		cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	# 		fig.colorbar(im, cax=cbar_ax)
	# 		writer.grab_frame()


	fig, axes = plt.subplots(nrows=numberOfRows, ncols=int(math.ceil(kernels[0].shape[-1]/numberOfRows)),figsize=figureSize, dpi=120)
	y=0
	for ax in axes.flat:
		im = ax.imshow(numpy.median(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1).take(y,axis=2),vmin=minValue, vmax=maxValue,norm=LogNorm());
		im.set_cmap(cmapPerso)
		ax.set_title( titles[y], fontsize=20)
		ax.tick_params(
				# axis='both',         # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
		y+=1

	fig.subplots_adjust(top=0.95, bottom=0.15,right=0.95,left=0.05)
	cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.05])
	cbar_ax.tick_params(labelsize=20)
	fig.colorbar(im, orientation="horizontal", cax=cbar_ax)
	plt.savefig('../figures/p50_cond.png', transparent=True)
	plt.show()


	# fig, axes = plt.subplots(nrows=numberOfRows, ncols=int(math.ceil(kernels[0].shape[-1]/numberOfRows)),figsize=figureSize, dpi=120)
	# y=0
	# for ax in axes.flat:
	# 	im = ax.imshow(numpy.percentile(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),10,-1).take(y,axis=2),vmin=minValue, vmax=maxValue,norm=LogNorm());
	# 	im.set_cmap(cmapPerso)
	# 	ax.set_title( titles[y], fontsize=20)
	# 	ax.tick_params(
	# 			# axis='both',         # changes apply to the x-axis
	# 			which='both',      # both major and minor ticks are affected
	# 			bottom=False,      # ticks along the bottom edge are off
	# 			top=False,         # ticks along the top edge are off
	# 			left=False,
	# 			right=False,
	# 			labelbottom=False,
	# 			labelleft=False) # labels along the bottom edge are off
	# 	y+=1

	# fig.subplots_adjust(top=0.95, bottom=0.15,right=0.95,left=0.05)
	# cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.05])
	# cbar_ax.tick_params(labelsize=20)
	# fig.colorbar(im, orientation="horizontal", cax=cbar_ax)
	# plt.savefig('../figures/p10_cond.png', transparent=True)
	# plt.show()

	fig, axes = plt.subplots(nrows=numberOfRows, ncols=int(math.ceil(kernels[0].shape[-1]/numberOfRows)),figsize=figureSize, dpi=120)
	y=0
	for ax in axes.flat:
		im = ax.imshow(numpy.mean(numpy.stack(kernels[meanQualityPosition[:maxPosition]],axis=-1),-1).take(y,axis=2),vmin=minValue, vmax=maxValue,norm=LogNorm());
		im.set_cmap(cmapPerso)
		ax.set_title( titles[y], fontsize=20)
		ax.tick_params(
				# axis='both',         # changes apply to the x-axis
				which='both',      # both major and minor ticks are affected
				bottom=False,      # ticks along the bottom edge are off
				top=False,         # ticks along the top edge are off
				left=False,
				right=False,
				labelbottom=False,
				labelleft=False) # labels along the bottom edge are off
		y+=1

	fig.subplots_adjust(top=0.95, bottom=0.15,right=0.95,left=0.05)
	cbar_ax = fig.add_axes([0.05, 0.1, 0.9, 0.05])
	cbar_ax.tick_params(labelsize=20)
	fig.colorbar(im, orientation="horizontal", cax=cbar_ax)
	plt.savefig('../figures/mean_cond.png', transparent=True)
	plt.show()

