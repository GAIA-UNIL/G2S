---
title: SNESIM
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
---

# SNESIM

SNESIM is a categorical multiple-point simulation algorithm based on a search-tree representation of training-image patterns. In G2S it is exposed as a separate algorithm intended for categorical simulation problems and multi-grid workflows.

## Parameters for SNESIM

Usage: `result = g2s(flag1, value1, flag2, value2, ...)`

Schema outputs include `result["simulation"]` / `result.simulation`, `result["time"]` / `result.time`, `result["progress"]` / `result.progress`, and `result["job_id"]` / `result.job_id`.

| Flag | Description | Mandatory |
| ---- | ----------- | --------- |
| `-ti` | Training image or list of training images. SNESIM currently expects categorical data. | &#x2714; |
| `-di` | Destination image (simulation grid). NaN values are simulated and non-NaN values are used as conditioning data. | &#x2714; |
| `-dt` | Data type. For SNESIM this should describe categorical variables, typically `1` for each variable. | &#x2714; |
| `-ii` | Training-image index image. Use this to select which training image is used at each simulated node. |  |
| `-j` | Number of worker threads. Decimal values in `]0,1[` represent a fraction of the available logical cores. |  |
| `-mg` | Maximum multigrid level. The execution proceeds from this level down to `0`. |  |
| `-tpl` | Template radius. A value of `3` means offsets in `[-3,+3]`. Provide one value per dimension if needed. |  |
| `--tree-strategy` | Tree-selection strategy. Supported values are `first`, `ii`, and `merged`. Default: `merged`. |  |
| `-tree-root` | Root directory used for the SNESIM tree cache. |  |
| `-force-tree` | Force a rebuild of the tree cache instead of reusing cached trees. |  |
| `-s` | Random seed. |  |

## Examples

{% include include_code.md examplePath="snesim/snesim_example" %}

## Notes

- SNESIM is intended for categorical variables.
- The multigrid controls are specific to this algorithm.
- Tree caching is managed internally by the executable and can be controlled with `-tree-root` and `-force-tree`.
