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

It is available as `pyDev`, `PyDev`, or `pydev`. For machine-learning models, a batch mode predicts many nodes in one Python call.

New to pyDev? The [pyDev tutorial]({{ site.baseurl }}/algorithms/PyDevTutorial.html) builds and runs a complete algorithm in a few lines.

## Build and install

pyDev embeds Python, so it is not part of the default build, and it is not included in the Homebrew package. Build it from source, after the usual server build (see the [server installation guide]({{ site.baseurl }}/installation/server.html)):

```bash
cd build
make c++ -j
make pydev            # or: make pydev PYDEV_PYTHON=/path/to/python3.14t
make install          # optional: installs pyDev and the PyDevAlgo folder with the other algorithms
```

`make pydev` needs **Python 3.14 or later** with **numpy**, providing `libpython` (the usual case for python.org, Homebrew, conda and uv interpreters). Without `PYDEV_PYTHON` it takes the first of `python3.16t`, `python3.15t`, `python3.14t`, `python3.16`, `python3.15`, `python3.14`, `python3` on `PATH` that is recent enough and has numpy, and prints its choice. Prefer a **free-threaded** Python (`python3.14t`): it runs `sample()` on several threads at once. With a regular build the GIL lets only one thread run Python code at a time; results are the same, it is only slower (the job log then contains a warning).

If `PYDEV_PYTHON` is the `python` of a virtual environment, pyDev is built against the Python that environment was created from; the environment itself is not recorded, so removing it later does not affect pyDev. To use another Python version or build, run `make pydev` again (delete `build/c++-build/pyDev` first).

## Choosing the Python environment

Each algorithm chooses its environment with a **shebang line** at the top of its file:

```python
#!/path/to/myenv/bin/python
```

or, resolved through the `PATH` of the G2S server:

```python
#!/usr/bin/env python3.14t
```

The packages of that environment (PyTorch, JAX, scikit-learn, …) are then available to the algorithm. Without a shebang, pyDev uses the Python it was built against, which then needs numpy.

The environment must come from the **same Python version and build** (with or without GIL) as pyDev, for example a venv created with `python3.14t -m venv myenv` when pyDev was built against `python3.14t`. pyDev checks this before starting Python and stops the job with a clear message when the environment does not exist anymore or does not match. The job log records the environment used (`python_environment`).

Optional variables, read by the pyDev process (set them in the environment of the G2S server):

| Variable | Meaning |
|---|---|
| `G2S_PYDEV_ALGO_DIR` | Folder searched for `-pa` names. Default: the `PyDevAlgo` folder installed next to pyDev, or `build/PyDevAlgo` in a source tree. |
| `G2S_PYDEV_CACHE` | Folder for the caches of `sgsCP`, `sgsTiCP` and `mlpTemplate`. Default: `<tmp>/g2s_pydev_cache`. |

## Parameters

| Flag | Description | Mandatory |
|---|---|---|
| `-pa` | Python algorithm: `file`, `file:Class`, `folder/file.py:Class` or an absolute path. Relative names are looked up in the PyDevAlgo folder; the class name defaults to the file name. The file's shebang line selects the Python environment. | &#x2714; |
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
| `-j` | Number of threads for `sample()` (integer, or fraction of the cores). In batch mode, the default batch size. | |
| `-bs` | Batch size when the class defines `sample_batch()`. Default: `-j`. Batch mode runs 2 &times; batch-size threads; `-j` is then ignored. `-bs 1` uses `sample()` when the class defines both. | |
| `-csim` | Periodic (circular) simulation domain. | |

pyDev runs a vector simulation: all variables of a node are simulated together.

## Writing an algorithm

Put a Python file in the PyDevAlgo folder (or anywhere, and pass its absolute path to `-pa`):

```python
#!/path/to/myenv/bin/python      (optional: the environment to use)
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

    def sample_batch(self, coordinates, values, randUniform, indexInThePath, idTI4Sampling):
        # optional: predicts several nodes in one call (for ML models, GPUs, vectorised code)
        # coordinates: int32 (B, K, ndim), values: float64 (B, K, nvar), padded with
        #   offset 0 / NaN rows when nodes have fewer than K neighbours
        # randUniform: (B,), indexInThePath: (B,), idTI4Sampling: (B,)
        return values               # (B, nvar) or (B,), or (values, index) with index (B,)

    def finalize(self):            # optional, called once after a successful run
        pass
```

A class needs `sample()`, `sample_batch()`, or both.

- Offsets are given in NumPy axis order, so `ti[tuple(center + offset)]` reads the TI at that lag.
- Row 0 is always the node being simulated. Its values are NaN except for variables already known at that node.
- The other rows are the closest informed cells, in kernel order, until every variable has `n` values or the kernel is exhausted. A cell is informed if it is hard data or comes earlier in the path. pyDev waits for earlier nodes being simulated by other threads, so the result for a given seed does not depend on the number of threads.
- `sample()` returns one value per variable (a number, a list or an array), or a `(values, index)` tuple; `index` (a non-negative integer) is written to the `indexmap` output.
- If your code raises, the job stops and the error is reported to the client with the Python traceback in the job log.
- With a free-threaded Python, `sample()` is called from several threads at the same time: do not modify shared state in it without a lock (writing to a row reserved for `indexInThePath` is safe).

## Batch mode

When the class defines `sample_batch()`, pyDev predicts several nodes per Python call:

- threads search neighbourhoods as usual; a node whose neighbourhood is complete is queued in a shared batch, and the thread moves on to the next node of the path;
- the batch is sent when it holds `-bs` nodes, or when every thread is blocked waiting for a node of that batch (nothing more can join it), so batches are as full as the path allows;
- one batch is predicted at a time while the next one fills;
- pyDev runs 2 &times; batch-size threads so that enough nodes are in flight.

Each node still sees exactly the neighbours it would see in per-node mode, so for a given seed the result is the same as with `sample()` (up to floating-point differences between your two implementations). The job log reports `python_calls` and `batch_average_size`.

Batching pays off when each Python call is expensive (neural networks, GPUs). For a cheap model, per-node `sample()` calls running in parallel on a free-threaded Python can be as fast or faster.

How full batches can be depends on the path: early in a random path most nodes depend on recently simulated ones, so batches are small; later they fill up. Large grids allow larger batches. As an indication, on a 150 &times; 150 grid with `-n 30`, batches of 8, 32, 64 and 256 averaged 7.9, 28, 48 and 81 nodes.

## Examples

The PyDevAlgo folder contains three sequential Gaussian simulation (ordinary kriging) algorithms built on `pydev_kriging.py`:

| `-pa` | Description |
|---|---|
| `sgs` | Exponential covariance. The index map stores the number of informed neighbours. Defines both `sample()` and a batched `sample_batch()` (one linear solve per batch), a template for ML models. |
| `sgsCP` | Same model with a constant path (`-sp`). The kriging weights of each path index are stored on the first run and replayed afterwards (index map: 1 computed, 2 replayed). |
| `mlpTemplate` | Batch-mode template for machine learning: a small neural network (NumPy only) trained on a categorical 2D TI in `build()` (weights cached), predicting a whole batch per forward pass in `sample_batch()`. A starting point to plug in PyTorch or JAX, not a tuned simulator. |
| `sgsTiCP` | Covariance maps given as `-ti` (lag 0 at the centre), optionally one per region with `-ii`. Weights are cached as in `sgsCP`. |

Client-side scripts are in `example/python/pydev/`:

```python
from g2s import g2s
import numpy

result = g2s('-a', 'pyDev', '-pa', 'sgs', '-di', numpy.full((200, 200), numpy.nan),
             '-dt', [0], '-n', 50, '-j', 0.5)
simulation = result["simulation"]
```
