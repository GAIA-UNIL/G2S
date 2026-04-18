---
title: QuickSampling
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
| `-ti` | Training images (one or more images). If multivariate, the last dimension should be the same size as the number of variables, and should also match the size of the array given for the parameter dt. NaN values in the training image will be ignored. Unlike other MPS-algorithms, if there are multiple variables they will not be automatically normalized to be in the same range. | &#x2714; |
| `-di` | Destination image (one image, aka simulation grid). The size of di will be the size of the simulation grid. di can be identical as ti for gap-filing. NaN values will be simulated. Non-NaN values will be considered as conditioning data. | &#x2714; |
| `-dt` | Data type. 0 -> continuous and 1 -> categorical. This is where the number of variables is defined. If multiple variables, use a single N value if identical for all variables or use an array of N values if each variable has a different N. | &#x2714; |
| `-k` | Number of best candidates to consider to compute the narrowness, range [5, inf]. | &#x2714; |
| `-n` | N closest neighbors to consider. | &#x2714; |
| `-ki` | Image of the weighting kernel. Can be used to normalize the variables. If multiple variables, use a single kernel value if identical for all variables. If each variable has a different kernel, stack all kernels along the last dimension. The number of kernels should then match the size of the array given for the parameter dt. |  |
| `-nw` | Narrowness range: 0 -> max-min, 1 -> median, default IQR -> 0.5. |  |
| `-nwv` | Number of variables to consider in the narrowness (start from the end), default: 1. |  |
| `-cs` | Chunk size, the number of pixels to simulate at the same time, at each iteration, default: 1. |  |
| `-uds` | Area to update around each simulated pixel, the M closest pixel default: 10. |  |
| `-mp` | Partial simulation, 0 -> empty, 1 -> 100%. |  |
| `-s` | Seed value. |  |
| `-j` | To run in parallel (default is single core). To use as follows: `-j`, N1, N2, N3 (all three are optional but N3 needs N2, which in turn needs N1). Use integer values to specify a number of threads (or logical cores). Use decimal values ∈ ]0,1[ to indicate fraction of the maximum number of available logical cores (e.g., 0.5=50% of all available logical cores). N1 threads used to parallelize the path (path-level) Default: the maximum number of threads available. N2 threads used to parallelize over training images (if many TIs are available, each is scanned on a different core). Default: 1. N3 threads used to parallelize FFTs (node-level). Default: 1. N1 and N2 are recommended over N3. N1 is usually more efficient than N2, but requires more memory. |  |
| `-W_GPU` | Use integrated GPU if available. |  |
| `-nV` | No Verbatim (experimental). |  |

## Examples

## Publication

*Gravey, M., Rasera, L. G., & Mariethoz, G. (2019). Analogue-based colorization of remote sensing images using textural information. ISPRS Journal of Photogrammetry and Remote Sensing, 147, 242–254. https://doi.org/10.1016/j.isprsjprs.2018.11.003*



