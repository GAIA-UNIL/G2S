---
title: QS Calibration
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
---

# QS Calibration

In most cases, it is enough to explore the sensitivity of QS directly instead of launching a full AutoQS calibration.

This approach is usually much cheaper than AutoQS and is often sufficient to identify a good region for `k` and `n`.

## Principle

A practical approach is to run a single QS simulation where the parameters vary across the simulation grid:

- use `-kvi` to vary `k` over the image
- use `-ni` to vary `n` over the image
- inspect the resulting map to identify stable ranges of parameters

The idea is to sample the parameter space non-linearly:

- for `n`, in 2D it is often useful to explore the range with a squared progression
- for `k`, an exponential progression is often more informative than a linear one
- in 3D, the same idea generally suggests a cubic progression for `n`

## Examples

<div class="tab code">
  <button class="tablinks python" onclick="openTab(event, 'python', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks matlab" onclick="openTab(event, 'matlab', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Matlab.png" alt="Matlab">
  </button>
</div>

<div class="langcontent code interface python">

```python
#This code requires the G2S server to be running
import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s
import matplotlib.pyplot as plt

# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)))

maxK = 100
maxN = 100
size = 1000

y = numpy.linspace(numpy.sqrt(2), numpy.sqrt(maxN), size) ** 2
x = numpy.exp(numpy.linspace(0, numpy.log(maxK), size))
xv, yv = numpy.meshgrid(x, y)

simulation, *_ = g2s(
    '-a', 'qs',
    '-ti', ti,
    '-di', numpy.zeros((size, size)) * numpy.nan,
    '-dt', [0],
    '-kvi', xv,
    '-ni', yv,
    '-j'
)

plt.imshow(simulation)

x_labels = [f"{val:.1f}" for val in x]
y_labels = [f"{val:.0f}" for val in y]
spacingLabelX = simulation.shape[0] // 20
spacingLabelY = simulation.shape[1] // 20

plt.xticks(
    ticks=numpy.linspace(0, simulation.shape[1] - 1, len(x))[::spacingLabelX],
    labels=x_labels[::spacingLabelX]
)
plt.yticks(
    ticks=numpy.linspace(0, simulation.shape[0] - 1, len(y))[::spacingLabelY],
    labels=y_labels[::spacingLabelY]
)

plt.xlabel('k')
plt.ylabel('n')
plt.show()
```

</div>

<div class="langcontent code interface matlab">

```matlab
%This code requires the G2S server to be running
% load example training image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

maxK=100;
maxN=100;
size=1000;

y=linspace(sqrt(2),sqrt(maxN),size).^2;
x=exp(linspace(0,log(maxK),size));
[xv,yv]=meshgrid(x,y);

simulation=g2s('-a','qs',...
               '-ti',ti,...
               '-di',nan(size,size),...
               '-dt',[0],...
               '-kvi',xv,...
               '-ni',yv,...
               '-j');

imagesc(simulation);
axis image;
colormap parula;

x_labels=arrayfun(@(v) sprintf('%.1f',v),x,'UniformOutput',false);
y_labels=arrayfun(@(v) sprintf('%.0f',v),y,'UniformOutput',false);
spacingLabelX=max(1,floor(size/20));
spacingLabelY=max(1,floor(size/20));

xticks(round(linspace(1,size,length(x))));
yticks(round(linspace(1,size,length(y))));
xticklabels(x_labels);
yticklabels(y_labels);
xticks(xticks(1:spacingLabelX:end));
yticks(yticks(1:spacingLabelY:end));
xticklabels(x_labels(1:spacingLabelX:end));
yticklabels(y_labels(1:spacingLabelY:end));

xlabel('k');
ylabel('n');
title('QS calibration map');
colorbar;
```

</div>
