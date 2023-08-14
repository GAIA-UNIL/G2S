---
title: QuickSampling
author:
  - Mathieu Gravey
date: 2023-03-13
toc: true
redirect_from:
  - /qs
---

# QuickSampling QS

## Parameters for QS

Usage: `[sim,index,time,finalprogress,jobid] = g2s(flag1,value1, flag2,value2, ...)`
Outputs: sim = simulation, index = index of the simulated values in the flattened TI, time = simulation time, finalprogress = final progression of the simulation (normally 100), jobid = job ID.

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ti` | Training images (one or more images). If multivariate, the last dimension should be the same size as the number of variables, and should also match the size of the array given for the parameter dt. NaN values in the training image will be ignored. Unlike other MPS-algorithms, if there are multiple variables they will not be automatically normalized to be in the same range. | &#x2714; |
| `-di` | Destination image (one image, aka simulation grid). The size of di will be the size of the simulation grid. di can be identical as ti for gap-filing. NaN values will be simulated. Non-NaN values will be considered as conditioning data. | &#x2714; |
| `-dt` | Data type. 0 &rarr; continuous and 1 &rarr; categorical. <red>This is where the number of variables is defined.</red> Provide a list containing zeros or ones representing the data type of each variable. | &#x2714; |
| `-k` | Number of best candidates to consider &#x2208; [1 &infin;]. | &#x2714; |
| `-n` | N closest neighbors to consider. If multiple variables: use a single N value if identical for all variables or use an array of N values if each variable has a different N. | &#x2714; |
| `-ki` | Image of the weighting kernel. Can be used to normalize the variables. If multiple variables: use a single kernel value if identical for all variables or if each variable has a different kernel, stack all kernels along the last dimension. The number of kernels should then match the size of the array given for the parameter dt. | |
| `-f` | f=1/k equivalent to f of DS with a threshold to 0. | |
| `-j` | To run in parallel (default is single core). To use as follows '-j', N1, N2, N3 (all three are optional but N3 needs N2, which in turn needs N1). Use integer values to specify a number of threads (or logical cores). Use decimal values &isin;]0,1[ to indicate fraction of the maximum number of available logical cores (e.g., 0.5=50% of all available logical cores). N1 threads used to parallelize the path (path-level) Default: the maximum number of threads available. N2 threads used to parallelize over training images (if many TIs are available, each is scanned on a different core). Default: 1. N3 threads used to parallelize FFTs (node-level). Default: 1. N1 and N2 are recommended over N3. N1 is usually more efficient than N2, but requires more memory. | |
| `-sp` | Simulation path, array of the size of the simulation grid containing values that specify the simulation path (from low to high). Default is a random path. Equal values are accepted but will always be ordered in the same way (i.e. not random). -&infin; values are not simulated. In case of multiple variables, a vector simulation is default (same path for all variables) and the simulation path should be one dimension less than the number of variables. If you prefer a full simulation, provide an array containing the path for each variable and use the "-fs" flag below. | |
| `-s` | Random seed value. | |
| `-W_GPU` | Use integrated GPU if available. | |
| `-W_CUDA` | Use Nvidia Cuda compatible GPU: specify the device id. (e.g., <code>'-W_CUDA',0,1</code> to use the first two GPUs). | |

### Advanced parameters

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ii` | Array that specifies for each pixel which training image to sample from. Default: all training images are searched for the best match. | |
| `-far` | Fast and risky &#x1f604;, like -ii but with a random input (experimental). | |
| `-cti` | With this flag QS will treat the training image(s) as periodic (aka circular or cyclic) over each dimension. | |
| `-csim` | With this flag QS will make sure to create a periodic (aka circular or cyclic) simulation over each dimension. | |
| `-adsim` | Augmented dimentionality simulation: allows for 3D simulation using 2D training image, only for categories (Coming maybe some day!). | |
| `-fs` | Full simulation: follows a different simulation path for each variable (as opposed to vector simulation, where the same simulation path is used for all variables). | |
| `-nV` | No Verbatim, i.e. prohibits neighbors in the training image to be neighbors in the simulation. (experimental). | |

## Examples
Below are several examples showcasing different applications of QS. For these examples the G2S server should be installed and running, either on your own machine or remotely. A Google Colab notebook with more examples and an automatic installation of G2S can be found [here](https://github.com/GAIA-UNIL/Short-course-MPS/blob/main/MPS_SC_with_QS_Online.ipynb).

### Unconditional simulation
{% include include_code.md exampleName="unconditionalSimulation" %}

### Conditional simulation
{% include include_code.md exampleName="conditionalSimulation" %}

### Simulation with multiple Training Images
{% include include_code.md exampleName="Multiple_TI" %}

### Multivariate simulation
{% include include_code.md exampleName="multivariateSimulation" %}

### Gap filling
{% include include_code.md exampleName="gapFilling" %}

### Downscaling
{% include include_code.md exampleName="downscaling" %}

### 3D simulation
{% include include_code.md exampleName="3D" %}

### Asynchronous mode 
{% include include_code.md exampleName="async_mode" %}

## Publication

*Gravey, M., & Mariethoz, G. (2020). QuickSampling v1.0: a robust and simplified pixel-based multiple-point simulation approach. Geoscientific Model Development, 13(6), 2611–2630. https://doi.org/10.5194/gmd-13-2611-2020*

## Benchmarking

This code CAN be used for benchmarking (and I invite you to do so 😉), the code needs to run natively on macOS or on Linux using the Intel Compiler with MKL library. The version needs to be reported, and the time needs to be the time reported by the algorithm (that is the true execution time without taking into account interfaces overhead).

When benchmarking, the code should NOT be used inside a Virtual Machine or through WSL on Windows 10+.

## Advanced

This page provides some additional information on how to do common tasks. It also serves as personal note to myself 😁

### Multiple realization

A question that is frequently asked is how to do multiple realizations. Currently there are three different ways to do it, each one of them has pros and cons.

#### The lazy solution

The simplest solution is to do a for-loop over the realizations. However, in each step the algorithm needs to wait for the data to load, and this creates an overhead

```python
sims1=numpy.empty((250,250,numberOfSimulation));
for i in range(numberOfSimulation):
  sims1[:,:,i],*_=g2s('-sa',computationServer,'-a','qs','-ti',ti,'-di',numpy.zeros((250,250))*numpy.nan,'-dt',numpy.ones((1,)),'-k',1.2,'-n',50,'-j',0.5,'-s',100+i);
```

#### For-loop without overhead

By using the G2S submission queue, it's possible to remove most of the overhead. This is the most versatile solution and it is recommended over the first solution, especially in case the computations are run on another machine. Furthermore, in this solution the parameters can be changed for each realization.

```python
ds=numpy.empty((numberOfSimulation,), dtype='long');
for i in range(numberOfSimulation):
  ids[i]=g2s('-sa',computationServer,'-a','qs','-ti',ti,'-di',numpy.zeros((250,250))*numpy.nan,'-dt',numpy.ones((1,)),'-k',1.2,'-n',50,'-j',0.5,'-s',100+i,'-submitOnly');

sims2=numpy.empty((250,250,numberOfSimulation));
for i in range(numberOfSimulation):
  sims2[:,:,i],*_=g2s('-sa',computationServer,'-waitAndDownload',ids[i]);
```

#### Overdimensioning of the destination image

This is the third solution to get multiple simulations at once. Although this solution looks easier, it has more limitations (e.g., being limited same size, same parameters, etc.), and therefore it is ⚠️ **not guaranteed** to stay functional in the future. This last solution should only be considered in case of extremely parallelized simulations (number of threads >50) and/or extremely small simulations (less than 1e4 pixels per simulation)

```python
sims3,*_=g2s('-sa',computationServer,'-a','qs','-ti',ti,'-di',numpy.zeros((numberOfSimulation,250,250))*numpy.nan,'-dt',numpy.ones((1,)),'-k',1.2,'-n',50,'-j',0.5);
```

### How to do a simulation by segment

Although QS tends to reduce issues related to non-stationarity, it won't remove them completely. Therefore in cases with high non-stationarity, the challenge is to respect each zone separately. Here we assume that we have different training images for each specific zone.

#### Using a secondary variable

The first solution for doing a simulation by segment is to add a secondary variable with the information of non-stationarity. In this case the setting of the algorithm has to be identical for both training images.

```python
ti1=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/bangladesh.tiff');
ti2=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff');

dt=[0];
N=300;

ti1_bis=cat(3,ti1,zeros(size(ti1)));
ti2_bis=cat(3,ti2,ones(size(ti2)));

localKernel=cat(3,kernel,padarray(15,25*[1,1]));
% localKernel=cat(3,kernel,kernel);

sim=g2s('-a','qs','-ti',ti1_bis,ti2_bis,'-di',cat(3,nan(size(tiSelection)),tiSelection),'-dt',[0,1],'-n',50,'-k',1.5,'-j',0.5,'-ki',localKernel);
imshow(medfilt2(sim(:,:,1)))
drawnow;
pause(5)
```

#### Using training image indices
The second solution for doing a simulation by segment is to use a training image index map. This requires that each training image represents a stationary subdomain. Also in this case the setting of the algorithm needs to be identical between training images.

```python

ti1=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/bangladesh.tiff');
ti2=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff');

dt=[0];
N=300;

sim=g2s('-a','qs','-ti',ti1,ti2,'-di',nan(size(tiSelection)),'-dt',dt,'-n',50,'-k',1.5,'-j',0.5,'-ki',kernel,'-ii',tiSelection);
imshow(medfilt2(sim))
pause(5)
```

#### The optimal solution

Using the `-ni` and `-kii` ...
TODO!!

#### Simulating each subdomain sequentially
The last solution for doing a simulation by segment is to do simulations per subdomain. This allows for changing the settings between each subdomain. However each subdomain is simulated sequentially and therefore can be at the origin of some artifacts, regarding the order of simulations.

```python
ti1=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/bangladesh.tiff');
ti2=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff');

dt=[0];
N=300;

path=reshape(randperm(numel(tiSelection)),size(tiSelection));
p1=path;
p1(tiSelection==0)=-inf;
p2=path;
p2(tiSelection==1)=-inf;
sim=g2s('-a','qs','-ti',ti1,'-di',nan(size(tiSelection)),'-dt',dt,'-n',50,'-k',1.5,'-j',0.5,'-ki',kernel,'-sp',p1);
sim=g2s('-a','qs','-ti',ti2,'-di',sim,'-dt',dt,'-n',50,'-k',1.5,'-j',0.5,'-ki',kernel,'-sp',p2);
imshow(medfilt2(sim))
pause(5)
```





