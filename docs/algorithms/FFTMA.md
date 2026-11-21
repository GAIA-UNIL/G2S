---
title: FFT-MA and spectral simulation
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /fftma
  - /spectral
---

# FFT-MA and spectral simulation (fftma, spectral)

`fftma` (also `FFTMA`, `FFT-MA`, `FFTMovingAverage`) and `spectral` (also `Spectral`, `SPECTRAL`) simulate a stationary Gaussian random field on a 1D, 2D or 3D grid with fast Fourier transforms. Both compute the spectrum of the covariance once, then produce each realisation with a few FFTs, so a realisation costs about as much as an FFT of the grid. They differ only in where the random numbers are drawn:

- **FFT moving average** (Le Ravalec et al., 2000) draws white noise in space and convolves it with the square root of the covariance. The noise can be returned and given back (`-on`, `-ni`), which is what local perturbation and gradual deformation need.
- **Spectral** draws the random numbers directly in frequency.

For a given seed the two methods give different realisations with the same covariance. Hard data in `-di` are honoured by conditioning each realisation by kriging (see [Conditioning to hard data](#conditioning-to-hard-data)).

The model must be stationary: spatially varying parameters and several models with `-ii` are refused; a [covariance map]({{ site.baseurl }}/algorithms/CovarianceModels.html#covariance-maps) is accepted. To compare with the other methods, see [Choosing a Gaussian simulation method]({{ site.baseurl }}/algorithms/GaussianSimulation.html).

## Periodic edge effect and the buffer

FFTs make the simulated field periodic: the two opposite edges of the FFT grid are correlated as if they were neighbours. To keep this out of the result, the simulation runs on an **extended grid**: the grid of `-di` plus a buffer along each axis (by default the extent of the largest practical range of the model along that axis), rounded up to a size whose prime factors are all 7 or less, and the output is cropped back to `-di`. When the buffer covers the reach of the model, the covariance inside the output grid is the model covariance. The extended grid is reported as the job metric `extended_grid` (sizes with the last array axis first: `420x315` for a numpy grid of shape `(256, 384)`).

The covariance sampled on the extended grid must have a non-negative spectrum. Negative values are set to zero; the fraction of the spectral energy removed this way is reported (`clipped_energy_fraction`), and above 0.1% a warning says that the simulated covariance deviates from the model and suggests a larger buffer (`-buf`).

With `-csim` the simulation is periodic on the `-di` grid itself, without buffer: the field tiles without seams. Its covariance is then the periodic version of the model on that grid (each lag taken at its shortest periodic distance), which differs from the model when the range is not small compared with the grid. A grid size with a prime factor larger than 7 is kept, with a warning that the FFTs are slower.

## Parameters

Usage: `result = g2s('-a', 'fftma', flag1, value1, flag2, value2, ...)` or `result = g2s('-a', 'spectral', ...)`

Outputs: `result["simulation"]` / `result.simulation` and, with `-on`, `result["noise"]` / `result.noise` (see [Outputs](#outputs)).

| Flag | Description | Default | Mandatory |
| ---- | ----------- | ------- | --------- |
| `-di` | Grid of the simulation, 1D, 2D or 3D. Non-NaN cells are hard data, honoured exactly; an all-NaN grid gives an unconditional simulation. Through the interfaces, give `-dt` as well (`'-dt', [0]`). | | &#x2714; |
| `-cov` | Stationary covariance model, see [Covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html). | | &#x2714; |
| `-cvar` | `name value`: binds a model variable (a number, or the table of a covariance map); repeat as needed. | | |
| `-method` | `fftma` or `spectral`. | the name used with `-a` | |
| `-nr` | Number of realisations. | 1 (with `-ni`: the number of noise variables) | |
| `-ri` | Index of the first realisation: realisation `r` of a run with `-ri 0` equals realisation 0 of a run with `-ri r`. | 0 | |
| `-m` | Mean (a number). | 0 | |
| `-s` | Seed. | from the clock | |
| `-j` | Number of threads; a fraction is a share of the cores. | 1 | |
| `-buf` | Buffer in cells, one value or one per axis. The extended grid is then rounded up to an FFT-friendly size. Ignored with `-csim` and with `-ni`. | the extent of the largest practical range along each axis | |
| `-csim` | Periodic simulation on the `-di` grid (no buffer); the conditioning then uses periodic distances. | off | |
| `-ni` | `fftma` only: white noise on the extended grid, one variable per realisation. Its size sets the extended grid; it must cover the `-di` grid (equal it with `-csim`). | drawn from the seed | |
| `-on` | `fftma` only: also output the white noise of the unconditional field (output 2, on the extended grid). | off | |
| `-kt` | Conditioning: kriging type of the residuals, `sk` or `ok`. | `sk` | |
| `-km` | Conditioning: kriging method, `direct`, `local` or `auto`. | `auto` | |
| `-n` | Conditioning, local: number of neighbours. | 50 | |
| `-ki` | Conditioning, local: kernel image (search window); cells with a zero or NaN weight are excluded. | unrestricted | |
| `-ks` | Conditioning, local: kernel size (search window), in cells. | unrestricted | |
| `-os` | Conditioning, local: at most `k` neighbours per quadrant (2D) or octant (3D). | off | |
| `-maxmem` | Conditioning, direct: largest *n* × *n* matrix allowed, in GiB. | 2 | |

## Outputs

- `simulation`: the grid of `-di` with one variable per realisation (Python shape `(ny, nx, nr)` in 2D).
- `noise` (`fftma` with `-on`): the white noise of each realisation on the extended grid, one variable per realisation. It is the noise of the unconditional field; conditioning does not change it. Giving it back with `-ni` (same model, `-di` and `-m`) reproduces the realisation.
- Job metrics: `extended_grid`, `clipped_energy_fraction`, `covariance_model` (canonical text), and with hard data `kriging_method` and `conditioning_ms`.

Through the Python and MATLAB interfaces, an image given with `-ni` is uploaded as a single variable: give the noise of one realisation per run (a 2D array for a 2D grid), as in the example below.

## Conditioning to hard data

The non-NaN cells of `-di` are hard data. Each unconditional realisation *z*<sub>u</sub> is conditioned by kriging of its residuals at the data:

*z*<sub>c</sub>(*x*) = *z*<sub>u</sub>(*x*) + *K*[*d* − *z*<sub>u</sub>](*x*),

where *K* is the kriging estimate from the residuals *d<sub>i</sub>* − *z*<sub>u</sub>(*x<sub>i</sub>*), with the same covariance model. The result honours the data. With simple kriging (`-kt sk`, the default) and the direct method, the conditioned field has the conditional distribution of the model given the data; far from the data it is the unconditional field. With ordinary kriging (`-kt ok`), the mean of the realisation is replaced by the mean estimated from the data (globally with `direct`, in each neighbourhood with `local`), and `-m` plays no role: use it when the mean is unknown or drifts slowly.

The kriging methods are those of the [kriging]({{ site.baseurl }}/algorithms/Kriging.html) executable: `direct` uses every datum (exact; one factorisation shared by all realisations), `local` the `-n` nearest data of each cell (approximate where the neighbourhood leaves out data that matter), and `auto` takes `direct` up to 10000 data while the matrix fits in `-maxmem`, `local` beyond. The flag is `-km` here because `-method` already selects `fftma` or `spectral`, and the default kriging type is `sk` (it is `ok` for the `kriging` executable). With `-csim` the kriging uses the periodic covariance of the simulated field.

## Reproducibility

For a given seed, model, grid and FFTW library on a given machine, the output is bit-identical for any `-j` and any split of the realisations over `-nr` and `-ri`. This does not hold across FFTW builds or machines. The seed defaults to the clock: give `-s` to reproduce a run.

## Examples

These examples need a running G2S server. They are in `example/python/fftma/` and `example/matlab/fftma/`.

### Unconditional simulation, spectral method and periodic field
{% include include_code.md examplePath="fftma/fftma_unconditional" %}

### Conditional simulation
{% include include_code.md examplePath="fftma/fftma_conditional" %}

### Noise input and output (gradual deformation)
{% include include_code.md examplePath="fftma/fftma_noise" %}

## References

- Le Ravalec, M., Noetinger, B., & Hu, L. Y. (2000). The FFT moving average (FFT-MA) generator: an efficient numerical method for generating and conditioning Gaussian simulations. *Mathematical Geology*.
- Pardo-Igúzquiza, E., & Chica-Olmo, M. (1993), spectral simulation.
- Dietrich, C. R., & Newsam, G. N. (1993, 1997), circulant embedding.
