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
data = g2s(...)  # it returns a tuple that contains all the output maps and the computing duration
```
You can either write `'-flag',value` or `flag=value` or a mix of both.

</div>

<div class="langcontent interface matlab">

## Use g2s in MATLAB {#matlab}

```matlab
data = g2s(...)          % principal output, the simulation
[data, t] = g2s(...)     % the simulation and the computation time
[data, ...,t] = g2s(...) % the simulation, other outputs map and the computation time
```
A call to g2s is needed to launch any computation. Each call to g2s is composed of parameters of G2S and of the name of the algorithm used.
Flags do NOT have a specific order.
You can either write `'-flag',value` or `flag=value` or a mix of both.

</div>

<div class="langcontent interface R">

## Use g2s in R {#R-lang}

The R interface is not available at this moment. Please contact g2s@mgravey.com if you need it.

</div>

## Inteface flags

| Flag          | Description                                                                                                 |
|---------------|-------------------------------------------------------------------------------------------------------------|
| `--version`     | Return the version and compilation date of the interface                                                   |
| `-a`            | The simulation algorithm to be used, it can be 'qs', 'nds', 'ds-l' (DS-like, not maintained)                |
| `-sa`           | Server address (default: localhost (the server is local), otherwise provide IP address)                    |
|               | Nice when we have a powerful machine dedicated for computation                                              |
| `-p`            | Port where to look for the server (default: 8128). Should be passed as an integer.                          |
| `-silent`       | Don't display the progression, useful for scripts                                                          |
| `-serverStatus` | Inform if the server is working properly <br><1 &#8594; error, such as comunication, server shutdown,...)<br>=0 &#8594; undefined <br>>1 &#8594; server is operational <br>1: standard server working normaly |
| `-noTO`         | Deactivate TimeOut on ZMQ communication, useful for slow network (e.g., through internet)                   |
| `-TO`           | Specify a custom TimeOut on ZMQ communication( uint32 in millisecond )                                      |
| `-shutdown`     | Shutdown the server, useful at the end of scripts                                                          |


### Asynchronous mode flags
The following options represent the **Asynchronous mode**, which allows you to submit multiple jobs simultaneously and retrieve the results of each of them later on (as opposed to synchronous communication with the server, where you need to wait until a job is finished before you can submit a new one). You launch the async mode by simply adding the `-submitOnly` flag to your g2s call. This will give only the job ID as an output, so the g2s call becomes `jobid = g2s(flag1,value1, flag2,value2, ...)`. Don't forget to always include the server address if it's not local! See the example section for a demonstration in *MATLAB* and *Python*.

| Flag             | Description                                                 |
|------------------|-------------------------------------------------------------|
| `-submitOnly`      | Submit a job                                                |
| `-statusOnly`      | Check progression <br>Usage: `status = g2s('-statusOnly',jobid)` |
| `-waitAndDownload` | Download the result <br>Usage: `sim,_ = g2s('-waitAndDownload',jobid)` |
| `-kill`            | Kill a given task <br>Usage: `g2s('-kill',jobid)`               |
| `-after`           | Execute the job after another one is finished (e.g. `-after,previousJobId` ) |

