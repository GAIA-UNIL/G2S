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

{% include include_code.md examplePath="auto_qs/autoQS" %}

For a lighter and usually sufficient alternative to AutoQS, see [QS Calibration]({{ site.baseurl }}/algorithms/QSCalibration.html).

## Publication

AutoQS is a calibration utility built around the QuickSampling workflow.
