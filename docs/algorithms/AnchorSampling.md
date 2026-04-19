---
title: Anchor Sampling
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
redirect_from:
  - /as
---

# Anchor Sampling AS

Pronunciation note: say **AS** like the French *as* 🂡, meaning the ace of a deck, or spell it **A, S**. Please do not pronounce it like “ass”.

Anchor Sampling (AS) is a location-anchored variant of QuickSampling. It preserves the same path-based simulation workflow, neighborhood template logic, continuous/categorical handling, multivariate support, kernels, and random or explicit simulation paths. The core conceptual change is the sampling constraint:

- In QS, once a neighborhood match is found, the sampled center can come from any admissible location in the training image(s).
- In AS, when simulating a cell at coordinate `(i,j,...)`, the sampled center value must come from that exact same coordinate across an aligned stack of training images.

This makes AS suitable when you have many realizations, scenarios, priors, or historical fields defined on the same grid geometry and you want neighborhood-based selection without losing spatial anchoring.

## Parameters for AS

Usage: `[sim,index,time,finalprogress,jobid] = g2s(flag1,value1, flag2,value2, ...)`
Outputs: `sim` = simulation, `index` = selected TI id for each simulated value, `time` = simulation time, `finalprogress` = final progression of the simulation (normally 100), `jobid` = job ID.

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-a` | Use `as`, `AS`, or `AnchorSampling`. | &#x2714; |
| `-ti` | Training images provided as an aligned stack. All TIs must have exactly the same spatial size as the simulation grid. In multivariate cases, the last dimension still represents the variables. NaN TI values are ignored. | &#x2714; |
| `-di` | Destination image (simulation grid). NaN values are simulated. Non-NaN values are conditioning data. Its spatial size must match every TI. | &#x2714; |
| `-dt` | Data type. `0` &rarr; continuous and `1` &rarr; categorical. Provide one value per variable. | &#x2714; |
| `-k` | Number of best anchored TI candidates retained before sampling. | &#x2714; |
| `-n` | Number of closest neighbors to consider. As in QS, this can be scalar or variable-specific. | &#x2714; |
| `-ki` | Optional kernel image defining the neighborhood footprint and weights. | |
| `-sp` | Optional simulation path. Default is a random path. As in QS, explicit path images are supported. | |
| `-j` | Parallel execution settings. Same meaning as in QS. | |
| `-s` | Random seed value. | |
| `-mi` | Optional prior or mask image with one value per TI at each spatial location. In Python and MATLAB this is typically a stack with trailing TI axis `(spatial..., n_ti)`. Mismatch is computed first, then `-mi` is applied as an admissibility mask before top-`k`: non-finite (`NaN`/`inf`) or non-positive mask values are excluded. The retained top-`k` candidates are finally sampled using `-mi` as relative weights. | |

### Advanced parameters

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-kii` | Per-pixel kernel-selection image, same semantics as QS. | |
| `-ni` | Per-pixel number of neighbors. | |
| `-kvi` | Per-pixel number of retained best candidates. | |
| `-cti` | Treat each TI as periodic over each dimension when evaluating TI-side neighborhoods. | |
| `-csim` | Create a periodic simulation over each dimension. | |
| `-fs` | Full simulation with distinct paths per variable, same semantics as QS. | |

### Important differences from QS

- The index image exported by AS stores the selected **TI id**, not a flattened TI pixel address.
- `-ii` is intentionally not supported in AS. Use `-mi` instead when you want to constrain or weight which TI can be selected at a given location.
- `-adsim` is not supported in AS. AS requires all TIs and the simulation grid to share the same geometry.
- AS is CPU-first. GPU flags are accepted for interface compatibility, but the anchored matcher currently runs on CPU.

## Examples

The following examples assume the G2S server is running.

### Minimal Python example
{% include include_code.md exampleName="AnchorSampling" %}

### Larger diagnostic example in Python

For a richer diagnostic script with a rectangular synthetic case, `-mi` edge cases, exported TI-id checks, and a live figure window, see:

- [example/python/AnchorSamplingSyntheticExperiment.py](https://github.com/GAIA-UNIL/G2S/blob/master/example/python/AnchorSamplingSyntheticExperiment.py)

## When to use AS instead of QS

Use AS when:

- your training data are available as a large stack of aligned realizations or scenarios
- the same coordinate across TIs has a stable physical meaning
- you want neighborhood matching to choose **which TI** to use at a location, rather than **which location** to sample from

Use QS when:

- stationarity in TI coordinates is acceptable
- the center value should be allowed to come from any TI location
- you need workflows such as TI-index maps through `-ii` or augmented-dimensional simulation

## Notes

- AS keeps the same general workflow as QS, including categorical, continuous, multivariate, kernel-based, and path-image driven simulations.
- In masked runs, AS computes mismatch first, applies `-mi` admissibility masking before top-`k` ranking (dropping non-finite and non-positive entries), then uses `-mi` as relative weights inside the retained set.
