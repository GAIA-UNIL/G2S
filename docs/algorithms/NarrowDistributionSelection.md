---
title: Narrow Distribution Selection
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 3
toc: true
redirect_from:
  - /nds
---

## Parameters for NDS

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ti` | Training images (one or more images). If multivariate, the last dimension should match the number of variables and the size of `-dt`. NaN values in the training image are ignored. | &#x2714; |
| `-di` | Destination image (simulation grid). NaN values are simulated and non-NaN values are treated as conditioning data. | &#x2714; |
| `-dt` | Data type. `0` for continuous and `1` for categorical. This also defines the number of variables. | &#x2714; |
| `-k` | Number of best candidates used to evaluate the narrowness. | &#x2714; |
| `-ki` | Weighting kernel. This defines the search neighborhood and can also be used to normalize variables. If omitted, NDS generates a default kernel internally. |  |
| `-nw` | Narrowness range. `0` corresponds to max-min, `1` approaches the median, default is `0.5` (interquartile range). |  |
| `-nwv` | Number of variables used to compute the narrowness, counted from the end of the variable list. Default: `1`. |  |
| `-cs` | Chunk size: number of pixels simulated together at each iteration. Default: `1`. |  |
| `-uds` | Update radius: number of nearby pixels updated around each simulated pixel. Default: `10`. |  |
| `-mp` | Partial simulation ratio. `0` means empty and `1` means a full simulation. Default: `1`. |  |
| `-s` | Random seed. |  |
| `-j` | Parallel execution settings. Use `-j`, `N1`, `N2`, `N3`, where all three values are optional but `N3` requires `N2`, and `N2` requires `N1`. Decimal values in `]0,1[` represent a fraction of the available logical cores. |  |
| `-wd` | Use the kernel distance. |  |
| `-ed` | Use the Euclidean distance. This is the default. |  |
| `-md` | Use the Manhattan distance. |  |
| `-W_GPU` | Use an integrated GPU if available. |  |
| `-nV` | No Verbatim mode (experimental). |  |

## Examples

NDS is primarily intended for spectrally guided simulation and colorization tasks. The active implementation is controlled by the narrowness parameters `-nw` and `-nwv`, together with the chunking and update controls `-cs`, `-uds`, and `-mp`.

## Publication

*Gravey, M., Rasera, L. G., & Mariethoz, G. (2019). Analogue-based colorization of remote sensing images using textural information. ISPRS Journal of Photogrammetry and Remote Sensing, 147, 242–254. https://doi.org/10.1016/j.isprsjprs.2018.11.003*


