---
title: QS Calibration
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
---

# QS Calibration

In most cases, it is enough to explore the sensitivity of QS directly instead of launching a full AutoQS calibration.

This approach is usually much cheaper than AutoQS and is often sufficient to identify a good region for `k` and `n`.

## Principle

A practical approach is to run a single QS simulation where the parameters vary across the simulation grid:

- use `-kvi` to vary `k` over the image
- use `-ni` to vary `n` over the image
- inspect the resulting map to identify stable ranges of parameters

The idea is to sample the parameter space non-linearly:

- for `n`, in 2D it is often useful to explore the range with a squared progression
- for `k`, an exponential progression is often more informative than a linear one
- in 3D, the same idea generally suggests a cubic progression for `n`

## Examples

{% include include_code.md examplePath="qs/qsCalibrationMap" %}
