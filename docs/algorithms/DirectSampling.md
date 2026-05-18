---
title: Direct Sampling
author:
  - Mathieu Gravey
date: 2026-05-16
toc: true
redirect_from:
  - /ds
---

# Direct Sampling DS

Native Direct Sampling is available as `ds`, `DS`, or `DirectSampling`. It is a CPU-only implementation integrated with the modern G2S simulation foundation used by QS and AS: random or explicit simulation paths, vector and full simulation, per-node control maps, interface upload plumbing, informed-neighbor filtering, kernel handling, and OpenMP path-level parallelism.

The old DS-like implementation remains available as `ds-l`, `dsl`, `DirectSamplingLike`, and `DS-L`. Those names continue to resolve to the legacy executable and are intentionally separate from native `ds`.

## Parameters

Usage: `[sim,index,time,finalprogress,jobid] = g2s(flag1,value1, flag2,value2, ...)`

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-a` | Use `ds`, `DS`, or `DirectSampling`. | &#x2714; |
| `-ti` | Training image. Multiple TIs are supported. | &#x2714; |
| `-di` | Destination image. `NaN` values are simulated; finite values are conditioning data unless `--forceSimulation` is set. | &#x2714; |
| `-dt` | Data type per variable: `0` continuous, `1` categorical. | &#x2714; |
| `-th` | Acceptance threshold. DS accepts the first scanned candidate at or below this mismatch. | &#x2714; |
| `-f` / `-mer` | Maximum exploration ratio. TI candidates are visited without replacement through a deterministic pseudo-random permutation. | &#x2714; |
| `-n` | Number of informed neighbors to use, scalar or per-variable. | &#x2714; unless `-ni` is provided |
| `-ki` | Optional kernel image. Kernel weights are honored using the true kernel flat index after neighborhood sorting. | |
| `-sp` | Optional simulation path. Equal path priorities are resolved by index for reproducible ordering. Default is a random path. | |
| `-j` | Number of OpenMP path-level workers. | |
| `-s` | Global seed. DS candidate permutations and stochastic transforms use stateless hash-based draws from this seed. | |
| `-ii` | Optional per-node TI-id map. Invalid values (`NaN`, non-finite, negative, non-integer, or out-of-range) mean unrestricted TI selection. | |
| `-ni` | Optional per-node neighbor-count map. | |
| `-kii` | Optional per-node kernel-index map. | |
| `-kvi` | Optional per-node exploration-ratio map. | |
| `-cti` | Treat each TI as circular during TI-side lookup. | |
| `-csim` | Treat the simulation domain as circular for neighbor lookup. | |
| `-wPO` | Enable vector simulation path optimization. Equal dependency depths are resolved by path index for same-seed reproducibility. This mirrors the QS flag and is ignored by full simulation. | |
| `-fs` | Full simulation: one path entry per variable. | |
| `--forceSimulation` | Simulate finite destination values too. | |
| `-cn` / `-cnorm` | Continuous rooted Lp mismatch norm. Provide one strictly positive value for all continuous variables or one per continuous variable. Default is `2`. | |

## Transforms

Native DS can apply stochastic local transforms to the search-pattern offsets before scanning TI candidates. The TI itself is never transformed. Transformed offsets use nearest-neighbor rounding for CPU lookup.

| Flag | Description |
| ---- | ----------- |
| `-rmi` | Rotation-center map. In 2D it has one angle channel in radians. In 3D it has four quaternion channels `(qx,qy,qz,qw)`. |
| `-rti` | Rotation-tolerance map with the same channel count as `-rmi`. Values are finite and non-negative. |
| `-smi` | Isotropic scale-center map, one strictly positive finite channel. |
| `-sti` | Isotropic scale tolerance, one finite non-negative channel. It is a half-width in log-scale space. |

For each simulated node, DS draws a local transform around the nominal maps. Zero tolerance is deterministic. Offsets are transformed in this order: scale, rotate, nearest-neighbor rounding. 3D quaternions are normalized before use; invalid or near-zero quaternions fall back to identity rotation.

Transform cache bins are deterministic: 1 degree for 2D angles and `0.05` for log-scale. Quantized values are used both for cache keys and transformed-offset construction.

## Mismatch

Categorical mismatch is a normalized weighted mismatch count. Continuous mismatch is a kernel-weighted rooted Lp mismatch using `-cn` / `-cnorm`. The common continuous powers `1` and `2` use direct arithmetic instead of generic `pow` calls. Mixed-variable DS uses a normalized sum across active variables and ignores non-finite neighbor or TI values without consuming support.

For each simulated node, candidate order is a deterministic pseudo-random permutation over the flattened allowed TI cells. The permutation is keyed by the global seed, local per-node seed, simulation path order, and variable, so a fixed explicit path remains reproducible while changing `-s` changes the TI visit order. In parallel runs, DS waits for earlier-path strict-informed neighbors before scoring candidates, so thread timing does not change the data event.

## Examples

The following examples assume the G2S server is running.
They are fully unconditional, create all-`NaN` destination images with the same shape as the loaded training image, and use `-j 1.00001` for path-level parallel execution.

- Continuous stone examples: `example/python/DirectSamplingContinuous.py`, `example/matlab/DirectSamplingContinuous.m`
- Categorical Strebelle examples: `example/python/DirectSamplingCategorical.py`, `example/matlab/DirectSamplingCategorical.m`
- Local transform examples: `example/python/DirectSamplingTransform.py`, `example/matlab/DirectSamplingTransform.m`
- Full mixed multivariate examples: `example/python/DirectSamplingMixedFull.py`, `example/matlab/DirectSamplingMixedFull.m`

The Python examples load remote TIFFs with `urllib` and Pillow; they do not require `requests`. The MATLAB examples use `imread` with a `websave` fallback for older MATLAB releases.
