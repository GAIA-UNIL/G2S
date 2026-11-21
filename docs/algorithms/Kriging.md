---
title: Kriging
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /kriging
---

# Kriging

`kriging` (also `Kriging`, `krige`) estimates the empty cells of a grid from its informed cells by simple or ordinary kriging, and optionally returns the kriging variance. The informed (non-NaN) cells of `-di` are the data and keep their value; the NaN cells are estimated. The model is any model of the [covariance grammar]({{ site.baseurl }}/algorithms/CovarianceModels.html).

Kriging gives the best linear estimate, which is smoother than reality. To produce realisations that honour the data and reproduce the variability of the model, use a simulation: [sgs]({{ site.baseurl }}/algorithms/SGS.html), or [fftma, spectral]({{ site.baseurl }}/algorithms/FFTMA.html) and [tb]({{ site.baseurl }}/algorithms/TurningBands.html), which condition their realisations with the same two kriging strategies as described here.

## Strategies

| `-method` | What it does | Models | Exact |
| --- | --- | --- | --- |
| `direct` | Unique neighbourhood: every datum is used at every cell. One Cholesky factorisation of the *n* × *n* data covariance, then the dual form of the kriging estimate. | stationary only | yes |
| `local` | Moving neighbourhood: the `-n` nearest data of each cell, one small system per cell. Cells with the same neighbour layout and the same resolved model share one solved system. | all, including spatially varying models and regions | approximate where the neighbourhood leaves out data that matter |
| `auto` | `direct` when the model is stationary, there are at most 10000 data, the *n* × *n* matrix fits in `-maxmem`, and, with `-var`, *n*² × (number of cells to estimate) ≤ 10¹²; `local` otherwise. The choice and its reason are written to the job log. | | |

The cost of `direct` grows as *n*³ for the factorisation and *n* per estimated cell (*n*² per cell for the variance); the cost of `local` grows with `-n`³ per distinct system and does not depend on the total number of data.

## Parameters

Usage: `result = g2s('-a', 'kriging', flag1, value1, flag2, value2, ...)`

Outputs: `result["estimate"]` / `result.estimate`, and with `-var`, `result["variance"]` / `result.variance`.

| Flag | Description | Default | Mandatory |
| ---- | ----------- | ------- | --------- |
| `-di` | Data image, one variable, 1 to 4 dimensions: NaN cells are estimated, the other cells are data. Through the interfaces, give `-dt` as well (`'-dt', [0]`). | | &#x2714; |
| `-cov` | Covariance model(s), see [Covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html). | | &#x2714; |
| `-cvar` | `name value`: binds a model variable; repeat as needed. | | |
| `-ii` | Model index image (0-based) when several models are given. | model 0 | |
| `-kt` | `sk` (simple kriging, known mean `-m`) or `ok` (ordinary kriging, unknown mean). | `ok` | |
| `-m` | Simple-kriging mean: a number or an image on the grid. Ordinary kriging only uses it at cells without any neighbour. | 0 | |
| `-method` | `direct`, `local` or `auto` (see [Strategies](#strategies)). | `auto` | |
| `-var` | Also compute the kriging variance (output 2). | off | |
| `-n` | Local: number of neighbours. Ignored (with a warning) by `direct`. | 50 | |
| `-ki` | Local: kernel image; its size is the search window, and cells with a zero or NaN weight are excluded. | unrestricted | |
| `-ks` | Local: kernel size (search window), in cells. | unrestricted | |
| `-os` | Local: at most `k` neighbours per quadrant (2D) or octant (3D). | off | |
| `-csim` | Local: periodic (circular) grid. | off | |
| `-maxmem` | Direct: largest *n* × *n* matrix allowed, in GiB. | 2 | |
| `-j` | Number of threads; a fraction is a share of the cores. | 1 | |

`-ki`, `-ks`, `-os` and `-csim` are ignored, with a warning, by `direct`.

## Outputs

- `estimate`: the grid of `-di`; data cells keep their value.
- `variance` (with `-var`): the kriging variance, 0 at data cells.
- A cell without any datum in its neighbourhood gets the mean `-m` and the sill as variance, for SK and OK alike (a warning gives their number). A kriging system that needs a diagonal jitter to be solved (duplicate or nearly collinear data, very smooth models) is reported as a warning.
- Job metrics: `kriging_method`, and for `local` the number of systems solved and shared.

With `direct` and ordinary kriging, the generalised-least-squares estimate of the mean is written to the log (`ok_gls_mean`).

## Examples

These examples need a running G2S server. They are in `example/python/kriging/` and `example/matlab/kriging/`. Conditional simulations compared with the kriging estimate are in the [sgs]({{ site.baseurl }}/algorithms/SGS.html#conditional-simulation) examples.

### Direct and local kriging with variance
{% include include_code.md examplePath="kriging/kriging" %}
