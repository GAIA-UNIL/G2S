---
title: Choosing a Gaussian simulation method
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /gaussian
---

# Choosing a Gaussian simulation method

G2S has four algorithms that simulate Gaussian random fields with a covariance model: [sgs]({{ site.baseurl }}/algorithms/SGS.html), [fftma and spectral]({{ site.baseurl }}/algorithms/FFTMA.html), and [tb]({{ site.baseurl }}/algorithms/TurningBands.html). They read the same [covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html), honour hard data exactly, and, for a given seed, give results that do not depend on the number of threads (for `fftma` and `spectral`, with a given FFTW library on a given machine; see the reproducibility section of each page). [kriging]({{ site.baseurl }}/algorithms/Kriging.html) gives the estimate rather than realisations.

## Summary

| | `sgs` | `fftma`, `spectral` | `tb` |
| --- | --- | --- | --- |
| Principle | Sequential: each cell drawn from its kriging distribution given the data and the cells simulated before it | Product of the spectrum of the covariance with random numbers, by FFT, on a periodic grid | Sum of one-dimensional processes along many lines |
| Grid | 1D to 4D | 1D to 3D | 1D to 3D |
| Hard data | Inside the simulation (simple or ordinary kriging) | Kriging of the residuals of each realisation (`-km`) | Kriging of the residuals of each realisation (`-km`) |
| Spatially varying parameters, regions (`-ii`) | yes | no | no |
| Covariance maps (`map`) | yes | yes | no |
| Covariance of the result | approximate: each cell only sees its `-n` nearest informed cells | exact on the output grid when the buffer covers the model (the clipped spectral energy is reported) | exact in expectation; each realisation closer to Gaussian with more lines (`-nb`) |
| Periodic field | `-csim` | `-csim` | not available |
| Cost of one more realisation (same grid) | small: the kriging weights are computed once per path and reused | about one FFT of the extended grid | proportional to cells × lines × structures |
| Plan or noise reuse | plan files (`-wo`, `-wi`) | white noise in and out (`-ni`, `-on`, `fftma` only) | |

## Which one to use

- **Stationary model, regular grid, many realisations**: `fftma` or `spectral`. They are by far the fastest, and their covariance is exact. Use `fftma` when you need the noise (gradual deformation, local perturbation); otherwise the two are equivalent.
- **Non-stationary model** (range, anisotropy or sill varying in space, or different models by region): `sgs`, which is the only one of the four that accepts them. Keep in mind the [local-stationarity caveat]({{ site.baseurl }}/algorithms/CovarianceModels.html#validity-of-spatially-varying-parameters).
- **Stationary model, but the buffer of the FFT methods would be too large** (ranges that are large compared with the grid, especially in 3D): `tb`, which needs no buffer and computes each cell independently, at a higher cost per realisation.
- **Many conditional realisations with the same data locations**: `sgs` with one path and many realisations (`-nr`), or a plan file replayed later with new seeds; or `fftma`/`spectral` with direct conditioning, whose factorisation is shared by all realisations.
- **A 4D grid**: `sgs`.

## Measured cost and covariance

The numbers below come from [example/python/gaussian_comparison/compare_methods.py](https://github.com/GAIA-UNIL/G2S/blob/master/example/python/gaussian_comparison/compare_methods.py), with the model `spherical(range=30)`, default parameters of each method (`sgs`: 50 neighbours; `tb`: 1000 lines) and `-j 8`, on an Apple M1 Pro (8 cores). The machine was running other work at the same time, and repeated runs of `sgs` varied by a factor of two: read the times as orders of magnitude. They are the computing times reported by the jobs (`result["time"]`), without the transfer of the data.

Computing time for 16 realisations of a 1000 × 1000 grid:

| | `sgs` | `fftma` | `spectral` | `tb` |
| --- | --- | --- | --- | --- |
| unconditional | 3.7 s | 0.06 s | 0.05 s | 4.0 s |
| 2000 hard data (`-km auto`, here `direct`) | 6.5 s | 1.7 s | 1.6 s | 5.4 s |

How the time grows with the number of realisations (1000 × 1000, unconditional):

| realisations | 1 | 16 | 64 |
| --- | --- | --- | --- |
| `sgs` | 3.3 s | 3.7 s | 4.0 s |
| `fftma` | 0.02 s | 0.10 s | 0.31 s |
| `tb` | 0.26 s | 4.0 s | 14.7 s |

`sgs` spends almost all its time in the plan (neighbour search and kriging systems), which is shared by all realisations of a path; `tb` costs about the same for every realisation; the FFT methods are cheap per realisation. With hard data, the FFT methods and `tb` add the conditioning, here one factorisation of the 2000 × 2000 data covariance and 2000 multiply-adds per cell and realisation.

Covariance for horizontal lags (along the rows) of 500 unconditional realisations of a 128 × 128 grid (the mean is known, 0), with twice the standard error of the mean over realisations:

| lag (cells) | 0 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| model | 1 | 0.752 | 0.519 | 0.148 | 0 |
| `sgs` | 0.997 | 0.750 | 0.519 | 0.160 | 0.006 |
| `fftma` | 0.999 | 0.752 | 0.518 | 0.151 | 0.005 |
| `spectral` | 0.989 | 0.742 | 0.510 | 0.144 | −0.003 |
| `tb` | 0.996 | 0.748 | 0.514 | 0.144 | 0.001 |
| ± (2 standard errors) | 0.015 | 0.014 | 0.013 | 0.012 | 0.012 |

At this precision all four reproduce the model; the largest deviation, `sgs` at a lag of 20 cells, is at the limit of the sampling error. The test does not resolve smaller differences.
