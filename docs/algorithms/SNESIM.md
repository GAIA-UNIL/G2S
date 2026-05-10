---
title: SNESIM
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
---

# SNESIM

SNESIM is a categorical multiple-point simulation algorithm based on a search-tree representation of training-image patterns. In G2S it is exposed as a separate algorithm intended for categorical simulation problems and multi-grid workflows.

## Parameters for SNESIM

Usage: `[sim,time,...] = g2s(flag1,value1, flag2,value2, ...)`

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ti` | Training image or list of training images. SNESIM currently expects categorical data. | &#x2714; |
| `-di` | Destination image (simulation grid). NaN values are simulated and non-NaN values are used as conditioning data. | &#x2714; |
| `-dt` | Data type. For SNESIM this should describe categorical variables, typically `1` for each variable. | &#x2714; |
| `-ii` | Training-image index image. Use this to select which training image is used at each simulated node. |  |
| `-j` | Number of worker threads. Decimal values in `]0,1[` represent a fraction of the available logical cores. |  |
| `-mg` | Maximum multigrid level. The execution proceeds from this level down to `0`. |  |
| `-tpl` | Template radius. A value of `3` means offsets in `[-3,+3]`. Provide one value per dimension if needed. |  |
| `--tree-strategy` | Tree-selection strategy. Supported values are `first`, `ii`, and `merged`. Default: `merged`. |  |
| `-tree-root` | Root directory used for the SNESIM tree cache. |  |
| `-force-tree` | Force a rebuild of the tree cache instead of reusing cached trees. |  |
| `-s` | Random seed. |  |

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

# load example training image ('strebelle') from same repo path style
url = "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff"
resp = requests.get(url, timeout=30)
resp.raise_for_status()
ti_raw = numpy.array(Image.open(BytesIO(resp.content)))

# SNESIM call using G2S
simulation, *_ = g2s(
    '-a', 'snesim',
    '-ti', ti_raw,
    '-di', numpy.zeros((1000, 1000)) * numpy.nan,
    '-dt', [1],
    '-j', 0.5,
    '-mg', 4,
    '-tpl', 3
)

# display results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
fig.suptitle('SNESIM Unconditional simulation', size='xx-large')
ax1.imshow(ti_raw, cmap='tab20')
ax1.set_title('Training image (categorical)')
ax1.axis('off')
ax2.imshow(simulation, cmap='tab20')
ax2.set_title('Simulation')
ax2.axis('off')
plt.show()
```

</div>

<div class="langcontent code interface matlab">

```matlab
%This code requires the G2S server to be running
% load example training image ('strebelle')
ti=imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/strebelle.tiff');

% SNESIM call using G2S
simulation=g2s('-a','snesim',...
               '-ti',ti,...
               '-di',nan(1000,1000),...
               '-dt',[1],...
               '-j',0.5,...
               '-mg',4,...
               '-tpl',3);

% Display results
sgtitle('SNESIM unconditional simulation');
subplot(1,2,1);
imshow(ti,[]);
title('Training image');
subplot(1,2,2);
imshow(simulation,[]);
title('Simulation');
```

</div>

## Notes

- SNESIM is intended for categorical variables.
- The multigrid controls are specific to this algorithm.
- Tree caching is managed internally by the executable and can be controlled with `-tree-root` and `-force-tree`.
