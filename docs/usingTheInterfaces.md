---
title: Using the interfaces
author:
  - Mathieu Gravey
date: 2023-03-13
toc-depth: 2
---


# Using the interfaces {#userInterface}

A call to **g2s** is needed to launch any computation. Each call to **g2s** is composed of parameters of G2S and of the name of the algorithm used. Flags do **NOT** have a specific order.

<!--Tab Buttons-->
<div class="tab">
  <button class="tablinks" onclick="openTab(event, 'matlab', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Matlab.png" alt="Matlab">
  </button>
  <button class="tablinks" onclick="openTab(event, 'python', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks" onclick="openTab(event, 'R', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Rlogo.svg" alt="R">
  </button>
</div>

<div class="langcontent interface python">

## Use g2s in Python {#python}

```python
from g2s import g2s
result = g2s(...)  # returns a schema dictionary by default
simulation = result["simulation"]
elapsed_seconds = result.get("time")
```
You can either write `'-flag',value` or `flag=value` or a mix of both.

</div>

<div class="langcontent interface matlab">

## Use g2s in MATLAB {#matlab}

```matlab
result = g2s(...)          % returns a schema struct by default
simulation = result.simulation
elapsed_seconds = result.time
```
A call to g2s is needed to launch any computation. Each call to g2s is composed of parameters of G2S and of the name of the algorithm used.
Flags do NOT have a specific order.
You can either write `'-flag',value` or `flag=value` or a mix of both.

</div>

<div class="langcontent interface R">

## Use g2s in R {#R-lang}

An R interface is available in the repository. Please refer to the installation page for the current build instructions.

</div>

## Inteface flags

| Flag          | Description                                                                                                 |
|---------------|-------------------------------------------------------------------------------------------------------------|
| `--version`     | Return the version and compilation date of the interface                                                   |
| `-a`            | The simulation algorithm to be used. Common values are `qs`, `as`, native `ds`, `nds`, and `snesim`. Legacy DS-like aliases remain available as `ds-l`, `dsl`, `DirectSamplingLike`, and `DS-L`. |
| `-sa`           | Server address (default: localhost (the server is local), otherwise provide IP address). Nice when we have a powerful machine dedicated for computation                     |
| `-p`            | Port where to look for the server (default: 8128). Should be passed as an integer.                          |
| `-silent`       | Don't display the progression, useful for scripts                                                          |
| `-serverStatus` | Query the server status without submitting a job. Typical return values are negative for communication errors, `0` for an undefined status, and positive values when the server is operational. |
| `-noTO`         | Deactivate the communication timeout on the client side. Useful for slow or unstable network connections. |
| `-TO`           | Specify a custom communication timeout in milliseconds for the client side ZMQ socket. |
| `-shutdown`     | Shutdown the server, useful at the end of scripts                                                          |
| `-legacy_output` | Force the old positional tuple/multi-output contract instead of the default schema object. |

For distributed QuickSampling, the Python and MATLAB interfaces also accept native matrices for JSON-grid parameters. `-jg` / `-job_grid` / `-job_grid_json` are normalized to `-jg`, `-eg` / `-endpoint_grid_json` are normalized to `-eg`, and `-di_grid_json` is kept under its canonical name. Matrix values are converted to inline JSON before the normal JSON upload path, so they should not be passed through the binary image upload path.

## Interface output syntax

The current interface returns one schema object by default. In Python this object is a dictionary; in MATLAB it is a struct; in R it is a named list. Common keys are:

- `simulation`: main simulation output when the algorithm writes one
- `indexmap`: selected training-image or index output when available
- `time`: elapsed computation time in seconds
- `job_id`: server job id
- `status`: `submitted`, `running`, `success`, or failure metadata when available
- `progress`: latest progress percentage
- `artifacts`: logical artifact names such as `im_1_<job>`, `log_<job>`, and `meta_<job>`

Algorithms can add named outputs. For example, AutoQS returns `mean_error`, `deviation_error`, and `sample_count`; NDS returns `narrowness` and `simulation_path`; SNESIM returns `simulation`.

Use `-legacy_output` for old examples or old code that still unpacks positional outputs:

```python
simulation, indexmap, *_ = g2s("-a", "qs", ..., "-legacy_output")
```

```matlab
[simulation, indexmap] = g2s('-a', 'qs', ..., '-legacy_output');
```

Current examples are under `example/python/<algorithm>/` and `example/matlab/<algorithm>/`. Old positional examples are under `legacy_example/`.

For Anchor Sampling continuous norms (`-cnorm` / `-cn`), Python and MATLAB can pass either a scalar (single `p` for all continuous variables) or a vector (one `p` per continuous variable).

### Asynchronous mode flags
The following options represent the **Asynchronous mode**, which allows you to submit multiple jobs simultaneously and retrieve the results of each of them later on (as opposed to synchronous communication with the server, where you need to wait until a job is finished before you can submit a new one). You launch the async mode by simply adding the `-submitOnly` flag to your g2s call. This returns a schema object containing the job id, so the Python pattern is `jobid = g2s(...)["job_id"]`; in MATLAB, assign the result first and then read `submitted.job_id`. Don't forget to always include the server address if it's not local. See `example/python/qs/async_status.py` and `example/matlab/qs/async_status.m`.

| Flag             | Description                                                 |
|------------------|-------------------------------------------------------------|
| `-submitOnly`      | Submit a job and return a schema object containing `job_id`. |
| `-statusOnly`      | Check progression <br>Usage: `status = g2s('-statusOnly',jobid)` |
| `-waitAndDownload` | Download the result <br>Usage: `result = g2s('-waitAndDownload',jobid)` |
| `-kill`            | Kill a given task <br>Usage: `g2s('-kill',jobid)`               |
| `-after`           | Add one or more job dependencies. The submitted job starts only after the specified job id(s) are finished. Use `'-after',0` to explicitly allow independent jobs to run in parallel through the G2S dependency manager. |
