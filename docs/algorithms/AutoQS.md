---
title: AutoQS
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
toc-depth: 3
redirect_from:
  - /autoqs
---

# AutoQS

> Warning:
> in about 95% of practical cases, [QS Calibration]({{ site.baseurl }}/algorithms/QSCalibration.html) is enough.
> AutoQS is very computationally expensive and should be reserved for extremely challenging cases.

## Parameters for AutoQS

AutoQS is a calibration workflow for QuickSampling. Instead of generating a single simulation, it evaluates combinations of neighborhood sizes, candidate counts, densities, and calibration settings in order to help select QS parameters.

Usage: `result = g2s(flag1, value1, flag2, value2, ...)`

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ti` | Training image or list of training images used for calibration. | &#x2714; |
| `-ki` | Kernel image or list of kernel images used during calibration. | &#x2714; |
| `-dt` | Data type definition passed through the standard interface upload logic. Use `0` for continuous and `1` for categorical variables. | &#x2714; |
| `-maxk` | Maximum number of best candidates explored during calibration. |  |
| `-maxn` | Maximum number of neighbors explored during calibration. Multiple values can be provided for multivariate cases. |  |
| `-nl` | File containing an explicit list of neighbor configurations to test. When this flag is provided, AutoQS uses that list instead of building one from `-maxn`. |  |
| `-maxIter` | Maximum number of calibration iterations. |  |
| `-minIter` | Minimum number of calibration iterations before early stopping is allowed. |  |
| `-maxT` | Maximum calibration time. |  |
| `-mpow` | Power used in the calibration error metric. |  |
| `-density` | Sampling densities tested during calibration. Multiple values can be provided. |  |
| `-cti` | Treat the training image as circular over each dimension. |  |
| `-ln` | Noise level used during calibration. |  |
| `-j` | Parallel execution settings. Use `-j`, `N1`, `N2`, `N3`, where all three values are optional but `N3` requires `N2`, and `N2` requires `N1`. Decimal values in `]0,1[` represent a fraction of the available logical cores. |  |
| `-W_GPU` | Use an integrated GPU if available. |  |
| `-W_CUDA` | Use one or more CUDA devices by id. |  |

If `-density` is omitted, AutoQS uses a default density grid internally. If `-nl` is omitted, AutoQS builds a list of neighbor configurations from `-maxn`.

## Examples

AutoQS is a calibration workflow rather than a normal simulation call. The goal is to explore QS settings and inspect the calibration result returned by the interface.

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


def first_2d_view(array):
    view = numpy.asarray(array)
    view = numpy.squeeze(view)
    while view.ndim > 2:
        view = view[..., 0]
    return view


# load example training image ('stone')
ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)))

# simple calibration kernel
kernel = numpy.ones((15, 15), dtype=float)

# AutoQS call using G2S
result, *_ = g2s(
    '-a', 'autoQS',
    '-ti', ti,
    '-ki', kernel,
    '-dt', [0],
    '-maxk', 2,
    '-maxn', 80,
    '-density', 0.0312, 0.0625, 0.125, 0.25,
    '-maxIter', 5000,
    '-minIter', 200,
    '-mpow', 2,
    '-j', 0.5
)

result2d = first_2d_view(result)

# Display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('AutoQS calibration', size='xx-large')
ax1.imshow(ti)
ax1.set_title('Training image')
ax1.axis('off')
ax2.imshow(result2d)
ax2.set_title('Calibration result')
ax2.axis('off')
plt.show()
```

</div>

<div class="langcontent code interface matlab">

```matlab
%This code requires the G2S server to be running
% load example training image ('stone')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff');

% simple calibration kernel
kernel=ones(15,15);

% AutoQS call using G2S
result=g2s('-a','autoQS',...
           '-ti',ti,...
           '-ki',kernel,...
           '-dt',[0],...
           '-maxk',2,...
           '-maxn',80,...
           '-density',0.0312,0.0625,0.125,0.25,...
           '-maxIter',5000,...
           '-minIter',200,...
           '-mpow',2,...
           '-j',0.5);

result2d=squeeze(result);
while ndims(result2d)>2
    result2d=result2d(:,:,1);
end

% Display results
sgtitle('AutoQS calibration');
subplot(1,2,1);
imshow(ti);
title('Training image');
subplot(1,2,2);
imagesc(result2d);
axis image;
title('Calibration result');
colorbar;
```

</div>

For a lighter and usually sufficient alternative to AutoQS, see [QS Calibration]({{ site.baseurl }}/algorithms/QSCalibration.html).

## Publication

AutoQS is a calibration utility built around the QuickSampling workflow.
