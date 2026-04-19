---
title: Distributed QuickSampling
author:
  - Mathieu Gravey
date: 2026-04-19
toc: true
redirect_from:
  - /algorithms/distributed-qs
---

# Distributed QuickSampling

Distributed QuickSampling is the cluster-oriented execution mode of QS. From a documentation point of view it is best treated as a separate workflow: the local QS flags still control the simulation itself, while an additional set of distributed inputs describes how several jobs are organized across a grid of subdomains.

Use this mode when a simulation is split into multiple QS jobs that need to exchange information across neighboring subdomains. For standard local or single-node simulations, refer to [QuickSampling]({{ site.baseurl }}/algorithms/QuickSampling.html).

## Launchers

Distributed QS is exposed through three launcher names that are used with the `-a` flag:

| Entry | Role |
| ---- | ---- |
| `qs_prep` | Prepare per-job TI and DI names such as `input_ti_0_<jobId>` and `input_di_<jobId>`. |
| `qs_dm` | Generic distributed launcher. It is the easiest way to test the distributed workflow on a single machine because it launches the child `qs` jobs itself. |
| `qs_dm_slurm` | Slurm-oriented launcher for cluster runs. |

## Distributed inputs

| Flag | Description |
| ---- | ----------- |
| `-jg` | Preferred job-grid flag. This identifies the distributed layout of the QS jobs. The value can be an inline JSON payload or a path to a JSON file. Accepted aliases are `-job_grid_json` and `-job_grid`. |
| `-eg` | Preferred endpoint-grid flag. This maps the distributed job grid to the endpoints used to exchange neighboring updates. The value can be an inline JSON payload or a path to a JSON file. Accepted alias: `-endpoint_grid_json`. |
| `-di_grid_json` | Destination-image grid payload. This maps the job grid to the destination-image identifiers used for neighbor halo exchange. The value can be an inline JSON payload or a path to a JSON file. |

## Input format

The three distributed payloads are JSON arrays that share the same row-major grid structure.

- The job grid contains job identifiers.
- The endpoint grid contains endpoint names or addresses for the corresponding jobs.
- The destination-image grid contains the DI identifiers associated with each job.

The shapes of the endpoint grid and DI grid must match the job grid.

## Storage path

The distributed launchers resolve their data and log directories as follows:

1. If `G2S_DATA_DIR` is defined, it is used first.
2. If `G2S_DATA_DIR` points to `<root>/data`, logs are written to `<root>/logs`.
3. If `G2S_DATA_DIR` points to `<root>`, data is written to `<root>/data` and logs to `<root>/logs`.
4. If `G2S_DATA_DIR` is not defined, the distributed launchers use `/scratch/<user>/G2S` when available.
5. If `/scratch/<user>/G2S` is not available, they fall back to `/tmp/G2S`.

This storage logic applies to the distributed launchers documented on this page. It is intentionally documented here rather than on the generic server page.

## Runtime behavior

- In `qs_dm`, the endpoint grid is generated automatically as `localhost:8130 + row_major_index`.
- For the `qs_dm` self-test workflow, users do not need to pass `-eg`.
- `qs_dm` launches the child `qs` jobs itself, so GNU Parallel and multiple servers are not required for a one-machine test.
- In distributed mode, the effective seed for each job is `base_seed + row_major_position`.
- If `-di_grid_json` is omitted, neighboring DI names fall back to `input_di_<jobId>`.
- Neighbor exchange includes edge neighbors, corner neighbors, and diagonal neighbors.
- Halo regions are padded internally during the run, and final outputs are cropped back to the central domain before they are written.
- `qs_dm_slurm` generates endpoints from the Slurm allocation at runtime, so `-eg` is ignored there.

## Example workflow

The example below is built around `qs_dm` because it is the easiest way to validate the distributed workflow on one machine.

<div class="tab code">
  <button class="tablinks python" onclick="openTab(event, 'python', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Python.svg" alt="Python">
  </button>
  <button class="tablinks matlab" onclick="openTab(event, 'matlab', 'interface')">
    <img src="{{ site.baseurl }}/assets/images/Matlab.png" alt="Matlab">
  </button>
</div>

<div class="langcontent code interface python">

```python
#This code requires the G2S server to be running
import time
import numpy
from PIL import Image
import requests
from io import BytesIO
from g2s import g2s

# load one base training image
base_ti = numpy.array(Image.open(BytesIO(requests.get(
    'https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff').content)),
    dtype=float)

# build four related training images from the same base pattern
rows, cols = base_ti.shape
y = numpy.linspace(0.0, 1.0, rows)[:, None]
x = numpy.linspace(0.0, 1.0, cols)[None, :]
rng = numpy.random.default_rng(12)

ti_101 = base_ti
ti_102 = numpy.clip(base_ti + 15.0 * x, 0, 255)
ti_103 = numpy.clip(base_ti + 15.0 * y, 0, 255)
ti_104 = numpy.clip(base_ti + 8.0 * (x + y) + rng.normal(0.0, 2.0, base_ti.shape), 0, 255)

# prepare one destination image per distributed job
def make_di(source, mode):
    di = numpy.full(source.shape, numpy.nan)
    if mode == 'top':
        di[:20, :] = source[:20, :]
    elif mode == 'left':
        di[:, :20] = source[:, :20]
    elif mode == 'diag':
        mask = numpy.abs(numpy.arange(rows)[:, None] - numpy.arange(cols)[None, :]) < 6
        di[mask] = source[mask]
    elif mode == 'center':
        di[rows//2-12:rows//2+12, cols//2-12:cols//2+12] = \
            source[rows//2-12:rows//2+12, cols//2-12:cols//2+12]
    return di

job_grid = [[101, 102],
            [103, 104]]

job_data = {
    101: (ti_101, make_di(ti_101, 'top')),
    102: (ti_102, make_di(ti_102, 'left')),
    103: (ti_103, make_di(ti_103, 'diag')),
    104: (ti_104, make_di(ti_104, 'center')),
}

# register the per-job inputs
for job_id, (ti, di) in job_data.items():
    g2s('-a', 'qs_prep', '-id', job_id, '-ti', ti, '-di', di)

# launch the distributed workflow
distributed_job = g2s(
    '-a', 'qs_dm',
    '-jg', job_grid,
    '-dt', [0],
    '-k', 1.2,
    '-n', 50,
    '-j', 0.5,
    '-s', 100,
    '-submitOnly'
)

# monitor the aggregate progress
status = 0
while status < 100:
    status = g2s('-statusOnly', distributed_job)
    print(f'distributed progress: {status}')
    time.sleep(1.0)

# retrieve the chunk results one by one
chunk_results = {}
for row in job_grid:
    for job_id in row:
        chunk_results[job_id], *_ = g2s('-waitAndDownload', job_id)

# for very large runs, adapt this loop to process one chunk at a time
# instead of keeping all chunks in memory simultaneously
```

</div>

<div class="langcontent code interface matlab">

```matlab
%This code requires the G2S server to be running
% load one base training image
base_ti=double(imread('https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/stone.tiff'));

% build four related training images from the same base pattern
[rows,cols]=size(base_ti);
[x,y]=meshgrid(linspace(0,1,cols),linspace(0,1,rows));
rng(12);

ti_101=base_ti;
ti_102=min(max(base_ti+15*x,0),255);
ti_103=min(max(base_ti+15*y,0),255);
ti_104=min(max(base_ti+8*(x+y)+2*randn(size(base_ti)),0),255);

% prepare one destination image per distributed job
make_di=@(source,mode) local_make_di(source,mode);

di_101=make_di(ti_101,'top');
di_102=make_di(ti_102,'left');
di_103=make_di(ti_103,'diag');
di_104=make_di(ti_104,'center');

job_grid=[101,102;103,104];

g2s('-a','qs_prep','-id',101,'-ti',ti_101,'-di',di_101);
g2s('-a','qs_prep','-id',102,'-ti',ti_102,'-di',di_102);
g2s('-a','qs_prep','-id',103,'-ti',ti_103,'-di',di_103);
g2s('-a','qs_prep','-id',104,'-ti',ti_104,'-di',di_104);

% launch the distributed workflow
distributed_job=g2s('-a','qs_dm',...
                    '-jg',job_grid,...
                    '-dt',[0],...
                    '-k',1.2,...
                    '-n',50,...
                    '-j',0.5,...
                    '-s',100,...
                    '-submitOnly');

% monitor the aggregate progress
status=0;
while status<100
    status=g2s('-statusOnly',distributed_job);
    fprintf('distributed progress: %g\n',status);
    pause(1);
end

% retrieve the chunk results one by one
chunk_results=cell(size(job_grid));
for iy=1:size(job_grid,1)
    for ix=1:size(job_grid,2)
        chunk_results{iy,ix}=g2s('-waitAndDownload',job_grid(iy,ix));
    end
end

% for very large runs, adapt this loop to process one chunk at a time
% instead of keeping all chunks in memory simultaneously

function di=local_make_di(source,mode)
    [nRows,nCols]=size(source);
    di=nan(size(source));
    switch mode
        case 'top'
            di(1:20,:)=source(1:20,:);
        case 'left'
            di(:,1:20)=source(:,1:20);
        case 'diag'
            mask=abs((1:nRows)'-(1:nCols))<6;
            di(mask)=source(mask);
        case 'center'
            r0=floor(nRows/2);
            c0=floor(nCols/2);
            di(r0-11:r0+12,c0-11:c0+12)=source(r0-11:r0+12,c0-11:c0+12);
    end
end
```

</div>

## Manual payload usage

If you do not use the helper launchers, the manual distributed inputs are still:

1. Pass the job-grid payload with `-jg`.
2. Pass the endpoint-grid payload with `-eg` when you are driving the child jobs yourself.
3. Pass the destination-image grid with `-di_grid_json` when neighbor DI metadata must be shared.

For `qs_dm`, step 2 is handled automatically.

## Cluster usage

For cluster-oriented deployment and queue-submission guidance, see the [cluster installation page]({{ site.baseurl }}/installation/cluster.html).
