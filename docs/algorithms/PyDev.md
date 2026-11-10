---
title: pyDev
author:
  - Mathieu Gravey
date: 2026-11-10
toc: true
redirect_from:
  - /pydev
---

# pyDev: prototype a simulation algorithm in Python

pyDev lets you try a new sequential-simulation idea in Python without writing C++. G2S does the work around it: the simulation path, the search for informed neighbours, waiting for nodes simulated by other threads, hard data, the outputs and the server integration. Your Python class only has to predict one node from its neighbours.

It is available as `pyDev`, `PyDev`, or `pydev`.

## Build

pyDev embeds Python, so it is not part of the default build. It needs **Python 3.14 or later** with **numpy**, built as a shared library (the usual case for python.org, Homebrew, conda and uv interpreters).

```bash
cd build
make pydev PYDEV_PYTHON=/path/to/python3.14t
```

`PYDEV_PYTHON` defaults to `python3`. Use a **free-threaded** interpreter (`python3.14t`) to run `sample()` on several threads at once. With a regular build the GIL lets only one thread run Python code at a time; results are the same, it is only slower (the job log then contains a warning). If `PYDEV_PYTHON` is the `python` of a virtual environment, that environment's packages are available to your algorithm.

At run time:

| Variable | Meaning |
|---|---|
| `G2S_PYDEV_PYTHON` | Interpreter (or venv `python`) to use instead of the one pyDev was built with. It must be the same Python version and build (with or without GIL). |
| `G2S_PYDEV_ALGO_DIR` | Folder searched for `-pa` names. Default: `build/PyDevAlgo`. |
| `G2S_PYDEV_CACHE` | Folder for the weight cache of `sgsCP` and `sgsTiCP`. Default: `<tmp>/g2s_pydev_cache`. |

These are read by the pyDev process, so set them in the environment of the G2S server.

## Parameters

| Flag | Description | Mandatory |
|---|---|---|
| `-pa` | Python algorithm: `file`, `file:Class`, `folder/file.py:Class` or an absolute path. Relative names are looked up in the PyDevAlgo folder; the class name defaults to the file name. | &#x2714; |
| `-di` | Destination image. NaN values are simulated, other values are hard data. | &#x2714; |
| `-n` | Number of informed neighbours per node; one value, or one per variable. | &#x2714; |
| `-dt` | Data type per variable (0 continuous, 1 categorical), passed to the Python class. | |
| `-ti` | Training image(s), passed to the Python class (for example covariance maps). | |
| `-ki` | Kernel image defining the neighbourhood. Cells with zero or NaN weight are ignored. | |
| `-ks` | Kernel size when `-ki` is omitted. Default: the destination size, or half the TI size when TIs are given. | |
| `-wd` | Order neighbours by kernel weight instead of Euclidean distance. | |
| `-sp` | Simulation path (low to high; -&infin; is not simulated). Default: random path. | |
| `-ii` | Per-node integer passed to `sample()` as `idTI4Sampling` (for example which TI or model to use). | |
| `-s` | Random seed. | |
| `-j` | Number of threads (integer, or fraction of the cores). | |
| `-csim` | Periodic (circular) simulation domain. | |

pyDev runs a vector simulation: all variables of a node are simulated together.

## Writing an algorithm

Put a Python file in `build/PyDevAlgo` (or anywhere, and pass its absolute path to `-pa`):

```python
import numpy

class myAlgorithm:
    def __init__(self, TIs, kernel, di, dt, path):
        # called once; arrays are copies in the same layout as on the client side
        # TIs: list of arrays, kernel: array, di: destination image,
        # dt: int32 array (0 continuous, 1 categorical), path: array or None
        pass

    def build(self, n):            # optional, n = list given with -n
        pass

    def sample(self, coordinates, values, randUniform, indexInThePath, threadId,
               fullStationary, variableOfInterest, idTI4Sampling):
        # coordinates: int32 (k, ndim) neighbour offsets, row 0 = the node itself
        # values:      float64 (k, nvar), NaN when unknown or not needed
        # randUniform: the node's random number in [0, 1)
        # idTI4Sampling: value of -ii at the node, -1 without -ii
        return value                # or (values, index) to fill the index map

    def finalize(self):            # optional, called once after a successful run
        pass
```

- Offsets are given in NumPy axis order, so `ti[tuple(center + offset)]` reads the TI at that lag.
- Row 0 is always the node being simulated. Its values are NaN except for variables already known at that node.
- The other rows are the closest informed cells, in kernel order, until every variable has `n` values or the kernel is exhausted. A cell is informed if it is hard data or comes earlier in the path. pyDev waits for earlier nodes being simulated by other threads, so the result for a given seed does not depend on the number of threads.
- `sample()` returns one value per variable (a number, a list or an array), or a `(values, index)` tuple; `index` (a non-negative integer) is written to the `indexmap` output.
- If your code raises, the job stops and the error is reported to the client with the Python traceback in the job log.
- With a free-threaded Python, `sample()` is called from several threads at the same time: do not modify shared state in it without a lock (writing to a row reserved for `indexInThePath` is safe).

## Examples

The PyDevAlgo folder contains three sequential Gaussian simulation (ordinary kriging) algorithms built on `pydev_kriging.py`:

| `-pa` | Description |
|---|---|
| `sgs` | Exponential covariance. The index map stores the number of informed neighbours. |
| `sgsCP` | Same model with a constant path (`-sp`). The kriging weights of each path index are stored on the first run and replayed afterwards (index map: 1 computed, 2 replayed). |
| `sgsTiCP` | Covariance maps given as `-ti` (lag 0 at the centre), optionally one per region with `-ii`. Weights are cached as in `sgsCP`. |

Client-side scripts are in `example/python/pydev/`:

```python
from g2s import g2s
import numpy

result = g2s('-a', 'pyDev', '-pa', 'sgs', '-di', numpy.full((200, 200), numpy.nan),
             '-dt', [0], '-n', 50, '-j', 0.5)
simulation = result["simulation"]
```
