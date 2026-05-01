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

Documentation and packaging notes should use concise, current wording because they are reused across generated artifacts.

A repository-wide review snapshot is available in `CODE_REVIEW_REPORT.md`.

## Build note

`make` in `build/Makefile` checks whether `include/zmq.hpp` exists. If missing, it auto-downloads `zmq.hpp` from `cppzmq` using `curl` (preferred), then `wget`, then `python`.

For Python wheels, `zmq.h` must also be available. The Python build first tries `pyzmq` include paths (PEP 517 isolated builds), then system include paths. If not found, install ZeroMQ development headers (for example `libzmq3-dev` on Debian/Ubuntu or `zeromq-devel` on RHEL/Fedora).

## Server job launch policy

By default, the server only launches algorithms whose requested `Algorithm` name resolves through `algosName.config`. Unknown names are rejected instead of falling back to `./<Algorithm>`. Use `--allow-unregistered-algorithms` when starting `g2s_server` to explicitly restore the legacy fallback behavior for local development or custom deployments.

Remote job requests are bounded before launch: job JSON is limited to 1 MiB, algorithm names to 2048 bytes, individual argv entries to 64 KiB, and total argv entries to 4096. Requests exceeding those limits are rejected instead of being truncated.

## Server job control

`KILL` requests now fail with a nonzero reply when the requested job id is unknown, malformed, or no longer tracked by the server.

## Server data protocol hardening

Data request frames are validated before dispatch. Uploads require exactly 64 hex hash characters, download/existence names are limited to safe 64-byte identifiers, job-id operations require exactly one `jobIdType`, and upload/download payloads are bounded.

Stored `.bgrid` payloads are read using the actual file or decompressed byte count. The embedded serialized size must match the bytes read, dimensions and variable counts are bounded, and malformed files are rejected instead of being allocated, sent back to clients, or deserialized from a short reply frame.

## AutoQS calibration noise

AutoQS calibration noise (`-ln`) randomizes neighbor-offset swaps using indexes drawn across the full neighbor vector.

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
