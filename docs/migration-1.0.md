---
title: Migrating to G2S 1.0
author:
  - Mathieu Gravey
date: 2026-11-10
toc: true
toc-depth: 2
---

# Migrating to G2S 1.0 {#migration}

G2S 1.0 changes what a call to **g2s** returns. Older versions returned several separate values (the simulation, the index map, the time, …) in a fixed order. From 1.0, a call returns **one result object** with named fields:

| Language | Result type     | Reading the simulation         |
|----------|-----------------|--------------------------------|
| Python   | `dict`          | `result["simulation"]`         |
| MATLAB   | `struct`        | `result.simulation`            |
| R        | named `list`    | `result$simulation`            |

Scripts that unpack several outputs (`sim, idx, t = g2s(...)` in Python, `[sim, idx, t] = g2s(...)` in MATLAB, `data[[1]]` in R) need a small change. The parameters you pass to **g2s** have not changed.

You have two options:

1. **Quick fix:** add `-legacy_output` to your existing calls and keep the old behaviour. See [Quick fix](#quick-fix-keep-the-old-outputs).
2. **Recommended:** read the named fields instead. See [Updating your scripts](#updating-your-scripts).

Both work with G2S 1.0, and you can mix them in the same script.

## Quick fix: keep the old outputs

Add the `-legacy_output` flag to each call. Everything else stays the same.

Python:

```python
# before
simulation, index, t = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',[0], '-k',1.2, '-n',50)
# G2S 1.0, unchanged behaviour
simulation, index, t = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',[0], '-k',1.2, '-n',50, '-legacy_output')
```

MATLAB:

```matlab
% before
[simulation, index, t] = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',0, '-k',1.2, '-n',50);
% G2S 1.0, unchanged behaviour
[simulation, index, t] = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',0, '-k',1.2, '-n',50, '-legacy_output');
```

R (flags are passed as named arguments):

```r
data <- g2s(a = "qs", ti = ti, di = di, dt = array(0, c(1, 1)), k = 1.2, n = 50, legacy_output = TRUE)
image(data[[1]])
```

If you already have a result object and need the old form, each interface also provides a converter:

| Language | Converter                                             |
|----------|-------------------------------------------------------|
| Python   | `from g2s import schema_to_legacy` then `schema_to_legacy(result)` |
| MATLAB   | `g2sSchemaToLegacy(result)`                           |
| R        | `g2s_schema_to_legacy(result)`                        |

The converter returns the outputs in the old order: the images first (simulation, index map, …), then the time, then the remaining metadata, the progress and the job id. Old unpacking such as `simulation, index, t, *_ = schema_to_legacy(result)` therefore still lines up.

## Updating your scripts

### Which field replaces which output

| Old positional output         | Field in the result object              |
|-------------------------------|-----------------------------------------|
| 1st output (simulation)       | `simulation`                            |
| 2nd output (index map)        | `indexmap`                              |
| time output, in seconds       | `time`                                  |
| `-returnMeta` output          | not needed: the fields are now at the top level (see [below](#returnmeta-is-no-longer-needed)) |
| job id (`-submitOnly`)        | `job_id`                                |
| progress (`-statusOnly`)      | `progress` (percentage)                 |

The result also contains:

- `status`: `submitted`, `running`, `success` or `failed`;
- `artifacts`: names of the files the server keeps for this job (log, warnings, errors, progress, metadata, and each output image), useful for debugging;
- job details reported by the algorithm, such as `algorithm`, `duration_ms`, `start_time_unix_ms`, `end_time_unix_ms`, `stage`. These values are **text**, for example `result["duration_ms"]` is `'1511'`, not `1511`. Use `time` (a number, in seconds) when you need the duration.

### Python

```python
# before
simulation, index, t = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',[0], '-k',1.2, '-n',50)

# after
result = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',[0], '-k',1.2, '-n',50)
simulation = result['simulation']
index      = result['indexmap']
t          = result['time']
```

Scripts that took only the first output also change:

```python
# before
simulation, *_ = g2s(...)
# after
simulation = g2s(...)['simulation']
```

### MATLAB

```matlab
% before
[simulation, index, t] = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',0, '-k',1.2, '-n',50);

% after
result = g2s('-a','qs', '-ti',ti, '-di',di, '-dt',0, '-k',1.2, '-n',50);
simulation = result.simulation;
index      = result.indexmap;
t          = result.time;
```

With an older compiled MATLAB interface, `result = g2s(...)` may only fetch the first image. If `result.indexmap` is missing, write `[result, ~] = g2s(...)`: the ignored second output only tells the interface to download the second image too.

### R

```r
# before
data <- g2s(a = "qs", ti = ti, di = di, dt = array(0, c(1, 1)), k = 1.2, n = 50)
image(data[[1]])

# after
result <- g2s(a = "qs", ti = ti, di = di, dt = array(0, c(1, 1)), k = 1.2, n = 50)
image(result$simulation)
```

`data[[1]]` still runs on a G2S 1.0 result, but it returns whichever field comes first in the list, which is not necessarily the simulation. Always use the field name.

### Field names per algorithm

Most algorithms return `simulation` and `indexmap`. Some return other images, named after what they contain:

| Algorithm                              | Fields                                                      |
|----------------------------------------|-------------------------------------------------------------|
| QuickSampling (`qs`)                   | `simulation`, `indexmap`                                    |
| Anchor Sampling (`as`)                 | `simulation`, `indexmap`                                    |
| Direct Sampling (`ds`, `ds-l`)         | `simulation`, `indexmap`                                    |
| Narrow Distribution Selection (`nds`)  | `simulation`, `indexmap`, `narrowness`, `simulation_path`   |
| SNESIM (`snesim`)                      | `simulation`                                                |
| AutoQS (`autoQS`)                      | `mean_error`, `deviation_error`, `sample_count`             |

With a server older than 1.0, which does not publish these names, the first image is always called `simulation` and the second `indexmap`.

## Asynchronous mode

The asynchronous calls return the same result object. Only the way you read the job id and the progress changes.

| Call                          | Before                                    | G2S 1.0                                              |
|-------------------------------|-------------------------------------------|------------------------------------------------------|
| `-submitOnly`                 | returned the job id                       | returns an object; read `job_id`                     |
| `-statusOnly`, jobId          | returned the progress (and the job id)    | returns an object; read `progress` and `status`     |
| `-waitAndDownload`, jobId     | returned simulation, index map, time, …   | returns the full result object, as a normal call    |
| `-kill`, jobId                | returned the job id                       | returns the job's last known state (see note)        |
| `-serverStatus`               | returned a number                         | unchanged: still a number                            |
| `-serverVersion`              | (new)                                     | returns the server version as text, e.g. `'1.0.0'`   |
| `-shutdown`                   | no output                                 | unchanged: no output                                 |

Python:

```python
# before
jobid = g2s('-a','qs', ..., '-submitOnly')
simulation, index, *_ = g2s('-waitAndDownload', jobid)

# after
jobid = g2s('-a','qs', ..., '-submitOnly')['job_id']
state = g2s('-statusOnly', jobid)
print(state['status'], state['progress'])
simulation = g2s('-waitAndDownload', jobid)['simulation']
```

MATLAB:

```matlab
% before
jobid = g2s('-a','qs', ..., '-submitOnly');
[simulation, index] = g2s('-waitAndDownload', jobid);

% after
submitted = g2s('-a','qs', ..., '-submitOnly');
jobid = submitted.job_id;
state = g2s('-statusOnly', jobid);
result = g2s('-waitAndDownload', jobid);
simulation = result.simulation;
```

> Note: the object returned by `-kill` is read just before the job stops, so its `status` can still say `running`. Do not use it to confirm that the job was stopped; call `-statusOnly` afterwards instead.

## `-returnMeta` is no longer needed

In earlier versions, `-returnMeta` added one extra output holding the job's final summary (`status`, `algorithm`, `duration_ms`, start and end times, …).

In G2S 1.0 these fields are always included at the top level of the result, so you can simply remove `-returnMeta`:

```python
# before
simulation, index, t, meta = g2s(..., '-returnMeta')
print(meta['status'], meta['duration_ms'])

# after
result = g2s(...)
print(result['status'], result['duration_ms'])
```

`-returnMeta` is still accepted. It only has an effect together with `-legacy_output`, where it keeps adding the summary as an extra output.

## Finding the calls to update

These searches find most calls that need attention:

| Pattern to search for                 | Why                                                     |
|---------------------------------------|---------------------------------------------------------|
| `, *_ = g2s(`, or several names before `= g2s(` | Python unpacking of several outputs  |
| `] = g2s(`                            | MATLAB calls with several outputs                       |
| `[[1]]` after a `g2s(` call           | R access by position                                    |
| `-submitOnly`                         | the job id is now `['job_id']` / `.job_id`              |
| `-statusOnly`                         | the progress is now `['progress']` / `.progress`        |
| `-returnMeta`                         | can be removed                                          |

## Examples

The examples shipped with G2S show both styles:

- `example/python/<algorithm>/` and `example/matlab/<algorithm>/` use the new result object;
- `legacy_example/python/` and `legacy_example/matlab/` keep the old positional style, with `-legacy_output` added to each call.

Comparing a file in `example/` with its counterpart in `legacy_example/` shows the exact change for that case.

See [Using the interfaces]({{ site.baseurl }}/usingTheInterfaces.html#interface-output-syntax) for the full description of the result object.
