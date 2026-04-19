# G2S: The GeoStatistical Server

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![](https://github.com/GAIA-UNIL/G2S//workflows/C/C++%20CI/badge.svg)
![](https://github.com/GAIA-UNIL/G2S/actions/workflows/pythonPublish.yml/badge.svg)



## Brief overview

**G2S** is composed of 2 parts:
- the first one is a server that manages computations and can be compiled for each hardware to obtain optimal performance.
- the second part is composed of different interfaces that communicate with the server through ZeroMQ. Interfaces can be added for each software. Similarly, G2S can be extended for any other geostatistical simulation algorithm.

Currently the **G2S** interface is available for *MATLAB* and *Python*. **G2S** is provided with **QS** (QuickSampling), **AS** (Anchor Sampling), and **NDS** (Narrow Distribution Selection).

**G2S** is currently only available for *UNIX*-based systems, *Linux* and *macOS*. A solution for *Windows 10+* is provided using *WSL* (Windows Subsystem for Linux). However, for previous *Windows* versions, the only solution currently available is to install a *Linux* system manually inside a virtual machine. 


## Documentation

An interactive and complete documentation is available [here](https://gaia-unil.github.io/G2S/).
The `docs/algorithms/example/` folder is a generated docs mirror (from `docs/sync_examples.sh`) and is intentionally gitignored.

## Build note

`make` in `build/Makefile` checks whether `include/zmq.hpp` exists. If missing, it auto-downloads `zmq.hpp` from `cppzmq` using `curl` (preferred), then `wget`, then `python`.

## AS mask order (`-mi`)

In Anchor Sampling, candidate mismatch is computed first, `-mi` invalid entries (`NaN`, `inf`, and non-positive values) are excluded before top-`k` ranking, and `-mi` weights are then used for weighted draw within the retained candidates.

## AS continuous norm (`-cnorm` / `-cn`)

Anchor Sampling continuous mismatch now supports configurable Lp norms with the proper root:
`pow(sum(pow(abs(diff), p)), 1/p)` (kernel-weighted internally).
Use one value to apply the same `p` to all continuous variables, or pass one value per continuous variable to configure each independently.

## Python AS example note

`example/python/AnchorSampling.py` is now a minimal synthetic AS demo (based on the larger `AnchorSamplingSyntheticExperiment.py` flow): one masked AS run, concise metrics, and a compact 4-panel visualization. It also keeps the safer Python interface call pattern with repeated `-ti` arguments and explicit `float32` arrays.

## Online Demo (Back! but slow)

An interactive online version is available [here](https://www.mgravey.com/mps.online/), to experiment with small unconditional simulations.
