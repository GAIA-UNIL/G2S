---
title: Turning bands
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /tb
---

# Turning bands (tb)

`tb` (also `TB`, `turningBands`, `TurningBands`) simulates a stationary Gaussian random field on a 1D, 2D or 3D grid by the turning bands method (Matheron, 1973), in the formulation of Emery & Lantuéjoul (2006). The field is a sum of one-dimensional random processes along many lines of different directions; every cell is computed independently from the others, so there is no edge effect, no periodicity and no constraint on the grid size. Hard data in `-di` are honoured by conditioning each realisation by kriging (see [Conditioning to hard data](#conditioning-to-hard-data)).

The model must be stationary and made of parametric structures: `nugget`, `spherical`, `exponential`, `gaussian`, `cubic`, `matern` and `stable` (see [Covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html)). Spatially varying parameters, several models with `-ii`, and covariance maps (`map`) are refused. To compare with the other methods, see [Choosing a Gaussian simulation method]({{ site.baseurl }}/algorithms/GaussianSimulation.html).

## Method

Each nested structure of the model is simulated separately, in its own space where it is isotropic with a unit practical range, on its own `-nb` lines, and the structures are added:

*Z*(*x*) = mean + Σ<sub>s</sub> √(*c<sub>s</sub>* / *L*) Σ<sub>l</sub> *Y<sub>sl</sub>*(⟨*A<sub>s</sub> x*, *u<sub>sl</sub>*⟩) + nugget,

where *c<sub>s</sub>* is the partial sill of structure *s*, *A<sub>s</sub>* its anisotropy transform, *u<sub>sl</sub>* the direction of line *l* and *Y<sub>sl</sub>* independent one-dimensional processes of unit variance.

- **Directions**: an evenly spread set (in 2D, directions regularly spaced on the half circle; in 3D, a Fibonacci lattice on the half sphere), turned by a uniform random rotation drawn for each realisation and each structure. The random rotation makes the covariance exact in expectation for any number of lines; the evenly spread set avoids banding in each realisation.
- **Line processes**, chosen per structure (the job log lists them):

| Structure | Line process |
| --- | --- |
| `spherical`, `cubic` | Dilution process (piecewise, random sign per unit segment), exact for these models. |
| `exponential`, `gaussian`, `matern`, `stable` | Continuous spectral method: a cosine with a random phase and a random frequency drawn from the spectral measure of the structure in the dimension of the grid. |
| `nugget` | Independent normal noise at each cell. |
| `map` | Not supported. |

The number of lines `-nb` controls how close each realisation is to a Gaussian field with the model covariance: with few lines a realisation is a sum of few line processes, far from Gaussian; the default is 1000 lines per structure. The cost of a realisation is proportional to the number of cells times the number of lines times the number of non-nugget structures.

## Parameters

Usage: `result = g2s('-a', 'tb', flag1, value1, flag2, value2, ...)`

Output: `result["simulation"]` / `result.simulation`: the grid of `-di` with one variable per realisation.

| Flag | Description | Default | Mandatory |
| ---- | ----------- | ------- | --------- |
| `-di` | Destination image: gives the grid (1D, 2D or 3D). Non-NaN cells are hard data, honoured exactly; an all-NaN grid gives an unconditional simulation. Through the interfaces, give `-dt` as well (`'-dt', [0]`). | | &#x2714; |
| `-cov` | Stationary covariance model made of `nugget`, `spherical`, `exponential`, `gaussian`, `cubic`, `matern` and `stable` structures. | | &#x2714; |
| `-cvar` | `name value`: binds a model variable to a constant; spatially varying values are refused. | | |
| `-nb` | Number of lines per structure, 1 to 16777215. | 1000 | |
| `-nr` | Number of realisations, 1 to 100000, output as variables. | 1 | |
| `-ri` | Index of the first realisation: realisation `r` of a run with `-ri 0` equals realisation 0 of a run with `-ri r`. | 0 | |
| `-m` | Mean. | 0 | |
| `-s` | Seed. | from the clock | |
| `-j` | Number of threads; a fraction is a share of the cores. | 1 | |
| `-kt` | Conditioning: kriging type of the residuals, `sk` or `ok`. | `sk` | |
| `-km` | Conditioning: kriging method, `direct`, `local` or `auto`. | `auto` | |
| `-n` | Conditioning, local: number of neighbours. | 50 | |
| `-ki` | Conditioning, local: kernel image (search window); cells with a zero or NaN weight are excluded. | unrestricted | |
| `-ks` | Conditioning, local: kernel size (search window), in cells. | unrestricted | |
| `-os` | Conditioning, local: at most `k` neighbours per quadrant (2D) or octant (3D). | off | |
| `-maxmem` | Conditioning, direct: largest *n* × *n* matrix allowed, in GiB. | 2 | |

`-csim` is refused: turning-bands fields are not periodic.

## Conditioning to hard data

The non-NaN cells of `-di` are hard data. Each unconditional realisation *z*<sub>u</sub> is conditioned by kriging of its residuals at the data, *z*<sub>c</sub> = *z*<sub>u</sub> + *K*[*d* − *z*<sub>u</sub>], with the same covariance model; the result honours the data. With simple kriging (`-kt sk`, the default) the conditioned field follows the conditional distribution given the data, and far from the data it is the unconditional field; with ordinary kriging (`-kt ok`) its mean is replaced by the mean estimated from the data. The kriging methods (`-km direct | local | auto`) and their options are those of the [kriging]({{ site.baseurl }}/algorithms/Kriging.html) executable; `auto` takes `direct` up to 10000 data while the matrix fits in `-maxmem`, `local` beyond. The [FFT-MA page]({{ site.baseurl }}/algorithms/FFTMA.html#conditioning-to-hard-data) describes the same conditioning in more detail.

## Reproducibility

For a given seed and inputs, the result does not depend on the number of threads (`-j`), and realisation `r` of a run with `-ri 0` equals realisation 0 of a run with `-ri r`, so realisations can be computed in several runs. The seed defaults to the clock: give `-s` to reproduce a run.

## Examples

These examples need a running G2S server. They are in `example/python/tb/` and `example/matlab/tb/`.

### Unconditional simulation with nested structures
{% include include_code.md examplePath="tb/tb_unconditional" %}

### Conditional simulation in 3D
{% include include_code.md examplePath="tb/tb_conditional" %}

## References

- Matheron, G. (1973). The intrinsic random functions and their applications. *Advances in Applied Probability*.
- Emery, X., & Lantuéjoul, C. (2006). TBSIM: A computer program for conditional simulation of three-dimensional Gaussian random fields via the turning bands method. *Computers & Geosciences*, 32, 1615–1628.
- Lantuéjoul, C. (2002). *Geostatistical Simulation: Models and Algorithms*. Springer.
