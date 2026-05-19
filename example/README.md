# Examples

Current examples use the schema output interface. They are grouped by language, then by algorithm. The old flat example set has been converted into these folders, so the website and smoke runners can use schema-output examples without depending on `-legacy_output`.

Some MATLAB schema examples use `[result, ~] = g2s(...)` when they read secondary outputs such as `result.indexmap`. The first return value is still the schema struct; the ignored second output only keeps older mex builds downloading `im_2_<job>`.

- `python/qs` and `matlab/qs`: QuickSampling examples
- `python/as` and `matlab/as`: Anchor Sampling examples
- `python/ds` and `matlab/ds`: native Direct Sampling examples
- `python/auto_qs` and `matlab/auto_qs`: AutoQS calibration examples
- `python/snesim` and `matlab/snesim`: SNESIM examples
- `python/reporting` and `matlab/reporting`: reporting and error-propagation probes

Run every Python example, including legacy examples, with:

```sh
python3 -m pip install -U -e build/python-build
python3 example/python/run_all_examples.py
```

Run every MATLAB example, including legacy examples, from MATLAB with:

```matlab
addpath('example/matlab')
run_all_examples
```

The reporting probes intentionally let the fatal error path propagate. The run-all scripts count those scripts as expected failures. The MATLAB runner executes scripts in isolated workspaces so variables from one example cannot shadow built-in functions in the next one. The Python runner also accepts `--current-only`, `--legacy-only`, and `--only TEXT` for narrower smoke runs.

Legacy positional-output examples were moved to `legacy_example/`. Those scripts pass `-legacy_output` directly to keep the old output contract.
