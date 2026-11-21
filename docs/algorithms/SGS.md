---
title: Sequential Gaussian simulation
author:
  - Mathieu Gravey
date: 2026-11-21
toc: true
redirect_from:
  - /sgs
---

# Sequential Gaussian simulation (sgs)

`sgs` (also `SGS`, `SequentialGaussianSimulation`) simulates a Gaussian random field on a grid by visiting the empty cells along a path and drawing each one from its kriging distribution given the hard data and the cells simulated before it, in a limited neighbourhood. It honours hard data exactly and accepts every covariance model of the [covariance grammar]({{ site.baseurl }}/algorithms/CovarianceModels.html), including spatially varying parameters, several models selected by region, and covariance maps.

`sgs` works in two phases. The **plan** finds the neighbours of every node and solves its kriging system; neither depends on the values, only on the grid, the hard-data locations, the path and the model. Nodes whose neighbours lie at the same relative positions and that share the same resolved model share one solved system. The **replay** then applies the stored weights and adds the noise, for any number of realisations at once. Computing the plan ahead and in parallel follows Nunes & Almeida (2010) and Schoedl et al. (2023); replaying one path for many realisations is the constant-path approach of Nussbaumer et al. (2018). The plan can be written to a file and replayed later (`-wo`, `-wi`).

`sgs` is univariate and simulates the values as given: transform the data to normal scores beforehand if needed, and back-transform the result.

For an unconditional or a stationary conditional simulation, the FFT methods and turning bands are usually faster; see [Choosing a Gaussian simulation method]({{ site.baseurl }}/algorithms/GaussianSimulation.html). A Python prototype of SGS written with [pyDev]({{ site.baseurl }}/algorithms/PyDev.html) (`sgs`, `sgsCP`, `sgsTiCP`) is kept as an example of that framework; the native `sgs` described here is the faster route for production.

## Parameters

Usage: `result = g2s('-a', 'sgs', flag1, value1, flag2, value2, ...)`

Outputs: `result["simulation"]` / `result.simulation` and `result["indexmap"]` / `result.indexmap` (see [Outputs](#outputs)), plus the usual `time`, `progress` and `job_id`.

| Flag | Description | Default | Mandatory |
| ---- | ----------- | ------- | --------- |
| `-di` | Destination grid, one variable, 1 to 4 dimensions. NaN cells are simulated; the other cells are hard data and are honoured exactly. Through the interfaces, give `-dt` as well (`'-dt', [0]`). | | &#x2714; |
| `-cov` | Covariance model(s), see [Covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html). | | &#x2714; |
| `-cvar` | `name value`: binds a model variable (number, image on the grid, or covariance map); repeat as needed. | | |
| `-mapclip` | Project covariance-map tables onto a valid covariance, see [Covariance maps]({{ site.baseurl }}/algorithms/CovarianceModels.html#covariance-maps). | off | |
| `-ii` | Model index image (0-based) when several models are given. | model 0 | |
| `-kt` | Kriging type: `sk` (simple) or `ok` (ordinary). | `sk` | |
| `-m` | Simple-kriging mean: a number or an image on the grid. With `ok`, it is only used for nodes without any neighbour. | 0 | |
| `-n` | Maximum number of neighbours per node. | 50 | |
| `-ki` | Kernel image: its size is the neighbourhood window, and cells with a zero or NaN weight are excluded from it. | | |
| `-ks` | Kernel size, in cells, one value for every axis, when `-ki` is not given. The size (given or default) is limited to 2 × grid − 1 along each axis (to the grid size with `-csim`) and made odd (an even size is reduced by 1). | twice the largest practical range of the model along each axis, + 1 | |
| `-wd` | Order the kernel cells by weight instead of by Euclidean distance. | distance | |
| `-os` | Octant search: at most `k` neighbours per sector (quadrants in 2D, octants in 3D; up to 4D). | off | |
| `-csim` | Circular simulation: the grid wraps around on every axis. | off | |
| `-nr` | Number of realisations per path. | 1 | |
| `-np` | Number of paths. `-nr` × `-np` is at most 65535. | 1 | |
| `-ri` | Index of the first realisation: realisation `r` of a run with `-ri 0` equals realisation 0 of a run with `-ri r` (same seed and inputs). | 0 | |
| `-sp` | Simulation path: an image on the grid, visited from low to high values; NaN cells come last and −&infin; cells are not simulated. | random path | |
| `-pt` | Path type when `-sp` is not given: `random`, `multigrid` (coarse to fine levels of stride 2<sup>k</sup>, random order inside a level) or `regular` (the same levels, raster order inside a level). | `random` | |
| `-s` | Seed, an integer from 0 to 2<sup>64</sup> − 1 (read the same way by `sgs`, `fftma`, `spectral` and `tb`). | from the clock | |
| `-j` | Number of threads; a fraction is a share of the cores (`0.5` = half). | 1 | |
| `-wo` | File (on the server) where the plan of the path is written. Needs `-np 1`. | | |
| `-wi` | Plan file (on the server) to replay instead of planning. Needs `-np 1`; cannot be combined with `-wo`. | | |
| `-noshare` | Solve the system of every node, without sharing identical systems. | sharing on | |

## Outputs

- `simulation`: the grid of `-di` with `-np` × `-nr` variables, ordered by path then realisation: variable `p × nr + r` is realisation `r` of path `p`. With a 2D grid and several realisations, the Python result has shape `(ny, nx, np × nr)`.
- `indexmap`: the number of neighbours used at each node, one variable per path (0 for hard data and for cells that were not simulated).
- Job metrics: `plan_ms` and `replay_ms` (time of the two phases), `plan_nodes`, `plan_systems_solved`, `plan_sharing_ratio` (systems solved per simulated node, for the first path), `plan_levels`, and `plan_systems_jittered` / `plan_systems_failed`: how many kriging systems of the run needed a diagonal jitter or could not be solved (a node whose system could not be solved is drawn from the unconditional distribution). Both are also in the log, with a warning when they are not 0: they are the sign of a covariance that is not positive definite (typically a raw covariance map, see [Covariance maps]({{ site.baseurl }}/algorithms/CovarianceModels.html#covariance-maps)) or of a very smooth model with many close neighbours.

## Neighbourhood and kriging

The neighbours of a node are the `-n` nearest informed cells (hard data or cells earlier in the path) inside the kernel window, ordered by distance (by weight with `-wd`); with `-os k`, at most `k` per sector. The kernel weights only select and order the cells; they do not enter the kriging. The default window covers twice the largest practical range of the model, so that every cell within the range can be a neighbour.

With simple kriging, the node is drawn from a normal distribution with the SK estimate (around the mean `-m`) and the SK variance; a node without any neighbour is drawn from the unconditional distribution (mean `-m`, variance the sill). With ordinary kriging, the estimate uses the local mean of the neighbours; a node without any neighbour gets the mean `-m` and the sill as variance.

With a spatially varying model or several models, the system of each node is built with the model resolved at that node (see [Validity of spatially varying parameters]({{ site.baseurl }}/algorithms/CovarianceModels.html#validity-of-spatially-varying-parameters)).

## Realisations, paths and reproducibility

All realisations of one path share the plan, so `-nr` adds little cost beyond the first realisation. Realisations of the same path are distinct random fields, but they share the visiting order. To also vary the path, use `-np`: each path has its own plan, derived from the seed and the path index (path 0 of a run with `-np 3` is the path of a run with `-np 1`).

For a given seed and inputs, the result does not depend on the number of threads (`-j`), and realisation `r` of a run with `-ri 0` equals realisation 0 of a run with `-ri r`, so realisations can be computed in several runs. The seed defaults to the clock: give `-s` to reproduce a run.

## Plan files (`-wo`, `-wi`)

`-wo file` writes the plan of the path to a binary file; `-wi file` replays it, which skips the neighbour search and every kriging system: only the replay is computed. The file is read and written by the server, so `file` is a path on the server machine. On replay, the seed still selects the noise: with the same `-s` and `-ri`, the realisations of the first run come back identical; with other values, new realisations of the same path are produced.

A plan is valid only for the inputs it was computed with: the grid, the hard-data locations, the covariance model (and the values of its variables), `-kt`, `-n`, the kernel (`-ki`, `-ks`, `-wd`), `-os` and `-csim`. `sgs` checks this and refuses a file that does not match. The hard-data **values** and the mean `-m` may differ between the run that wrote the plan and the replay. With `-sp`, the path given must be the one of the file.

## Examples

These examples need a running G2S server. They are in `example/python/sgs/` and `example/matlab/sgs/`; examples with spatially varying models, regions and covariance maps are on the [Covariance models]({{ site.baseurl }}/algorithms/CovarianceModels.html#examples) page.

### Unconditional simulation
{% include include_code.md examplePath="sgs/sgs_unconditional" %}

### Conditional simulation
{% include include_code.md examplePath="sgs/sgs_conditional" %}

### Plan once, replay later
{% include include_code.md examplePath="sgs/sgs_plan_replay" %}

## References

- Nunes, R., & Almeida, J. A. (2010). Parallelization of sequential Gaussian, indicator and direct simulation algorithms. *Computers & Geosciences*.
- Nussbaumer, R., Mariethoz, G., Gravey, M., Gloaguen, E., & Holliger, K. (2018). Accelerating sequential Gaussian simulation with a constant path. *Computers & Geosciences*.
- Schoedl, P., MacKie, E. J., Field, M. J., Stubbs, E. A., Zhang, A., Hibbs, M., & Gravey, M. (2023). A Python multiprocessing approach for fast geostatistical simulations of subglacial topography. *Computing in Science & Engineering*. [doi:10.1109/MCSE.2023.3317773](https://doi.org/10.1109/MCSE.2023.3317773)
