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
| GPU (`-W_GPU`, `-W_CUDA`) | no (CPU only) | yes, but on Apple silicon slower than the CPU (OpenCL only with `-gpuforce`) | yes, at par with the CPU on Apple silicon (large jobs with `exponential`, `gaussian`, `matern`, `stable`) |

## Which one to use

- **Stationary model, regular grid, many realisations**: `fftma` or `spectral`. They are by far the fastest, and their covariance is exact. Use `fftma` when you need the noise (gradual deformation, local perturbation); otherwise the two are equivalent.
- **Non-stationary model** (range, anisotropy or sill varying in space, or different models by region): `sgs`, which is the only one of the four that accepts them. Keep in mind the [local-stationarity caveat]({{ site.baseurl }}/algorithms/CovarianceModels.html#validity-of-spatially-varying-parameters).
- **Stationary model, but the buffer of the FFT methods would be too large** (ranges that are large compared with the grid, especially in 3D): `tb`, which needs no buffer and computes each cell independently, at a somewhat higher cost per realisation.
- **Many conditional realisations with the same data locations**: `sgs` with one path and many realisations (`-nr`), or a plan file replayed later with new seeds; or `fftma`/`spectral` with direct conditioning, whose factorisation is shared by all realisations.
- **A 4D grid**: `sgs`.

## Measured cost and covariance

The numbers below come from [example/python/gaussian_comparison/compare_methods.py](https://github.com/GAIA-UNIL/G2S/blob/master/example/python/gaussian_comparison/compare_methods.py), with the model `spherical(range=30)`, default parameters of each method (`sgs`: 50 neighbours; `tb`: 1000 lines) and `-j 8`, on an Apple M1 Pro (8 cores: 6 performance, 2 efficiency; 32 GB), measured on 21 November 2026 with the current build (SIMD turning-bands kernels, parallel output compression). Each time is the median of three runs. The 1-minute load average was 2.2 when the script started; the script's own 8-thread jobs raise it, and other work on the machine pushed it to 12–20 during the second table, so read the times as indicative: in another run `sgs` took up to twice as long. They are the computing times reported by the jobs (`result["time"]`), without the transfer of the data. Everything runs on the CPU (see the GPU section of each page).

Computing time for 16 realisations of a 1000 × 1000 grid:

| | `sgs` | `fftma` | `spectral` | `tb` |
| --- | --- | --- | --- | --- |
| unconditional | 3.3 s | 0.07 s | 0.05 s | 0.21 s |
| 2000 hard data (`-km auto`, here `direct`) | 3.3 s | 1.8 s | 1.6 s | 1.6 s |

How the time grows with the number of realisations (1000 × 1000, unconditional):

| realisations | 1 | 16 | 64 |
| --- | --- | --- | --- |
| `sgs` | 2.9 s | 3.0 s | 3.6 s |
| `fftma` | 0.01 s | 0.05 s | 0.27 s |
| `spectral` | 0.01 s | 0.05 s | 0.24 s |
| `tb` | 0.01 s | 0.16 s | 0.67 s |

`sgs` spends almost all its time in the plan (neighbour search and kriging systems), which is shared by all realisations of a path; `tb` costs about the same for every realisation, but its line sums are evaluated with per-axis tables on a regular grid, so on this grid it is only a few times slower than the FFT methods; the FFT methods are cheap per realisation. With hard data, the FFT methods and `tb` add the conditioning, here one factorisation of the 2000 × 2000 data covariance and 2000 multiply-adds per cell and realisation, which then dominates.

Covariance for horizontal lags (along the rows) of 500 unconditional realisations of a 128 × 128 grid (the mean is known, 0), with twice the standard error of the mean over realisations:

| lag (cells) | 0 | 5 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| model | 1 | 0.752 | 0.519 | 0.148 | 0 |
| `sgs` | 0.997 | 0.750 | 0.519 | 0.160 | 0.006 |
| `fftma` | 0.999 | 0.752 | 0.518 | 0.151 | 0.005 |
| `spectral` | 0.989 | 0.742 | 0.510 | 0.144 | −0.003 |
| `tb` | 1.008 | 0.760 | 0.525 | 0.154 | 0.009 |
| ± (2 standard errors) | 0.015–0.016 | 0.014–0.015 | 0.013 | 0.011–0.012 | 0.012 |

At this precision all four reproduce the model; the largest deviation, `sgs` at a lag of 20 cells, is at the limit of the sampling error. The test does not resolve smaller differences.
