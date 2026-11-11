"""Run the recording QS / DS / SNESIM algorithms (algos/seqsim.py) through pyDev.

    python run.py [--port 8128] [--python /path/to/env/bin/python] [name ...]

name is a key of RUNS, optionally with a grid-size suffix: qs_strebelle (80 x 80), qs_strebelle_50
(50 x 50). Without names, every configuration is run at 80 x 80. Each run writes to records/:
<name>.pkl (the log of every simulated node), <name>_sim.npy and <name>_path.npy.

--python selects the Python environment pyDev runs the algorithm in (it needs numpy and must match
the Python pyDev was built against); it is written as the shebang of a run copy of algos/seqsim.py.
Without it, pyDev uses the Python it was built against.
"""
import argparse
import json
import os
import time
import urllib.request

import numpy
import tifffile
from g2s import g2s

SP = os.path.dirname(os.path.abspath(__file__))
TI_URL = "https://raw.githubusercontent.com/GAIA-UNIL/TrainingImagesTIFF/master/{}.tiff"


def load_ti(name):
    """Full TI downscaled by 2 (2x2 block mean; Strebelle thresholded back to binary)."""
    path = os.path.join(SP, "data", f"{name}.tiff")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(TI_URL.format(name), path)
    a = tifffile.imread(path).astype(numpy.float64)
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    if name == "strebelle":
        a = (a >= 0.5).astype(numpy.float64)
    return a


def random_path(seed, size):
    return numpy.random.default_rng(seed).permutation(size[0] * size[1]).reshape(size).astype(float)


def multigrid_path(seed, levels, size):
    """Coarse grid first (every 2**levels node), then each finer level, random order within a level."""
    rng = numpy.random.default_rng(seed)
    yy, xx = numpy.indices(size)
    lev = numpy.zeros(size, int)
    for L in range(1, levels + 1):
        lev[(yy % 2**L == 0) & (xx % 2**L == 0)] = L
    path = numpy.empty(size)
    rank = 0
    for L in range(levels, -1, -1):
        cells = rng.permutation(numpy.flatnonzero(lev.ravel() == L))
        path.ravel()[cells] = numpy.arange(rank, rank + len(cells))
        rank += len(cells)
    return path


# Parameters of the G2S examples: example/python/qs/unconditionalSimulation.py (k 1.2, n 50;
# k 2.4 for Strebelle as in qs/qs_rotation_2d.py, less verbatim copy),
# ds/DirectSamplingCategorical.py (th 0.12, f 0.4, n 48), ds/DirectSamplingContinuous.py
# (th 0.08, f 0.35, n 40), snesim/snesim_example.py (-mg 4, -tpl 3).
RUNS = {
    "qs_strebelle":     dict(cls="QSRec", ti="strebelle", dt=1, n=50, k=2.4),
    "qs_stone":         dict(cls="QSRec", ti="stone", dt=0, n=50, k=1.2),
    "snesim_strebelle": dict(cls="SNESIMRec", ti="strebelle", dt=1, radius=3, levels=4),
    "ds_strebelle":     dict(cls="DSRec", ti="strebelle", dt=1, n=48, f=0.4, th=0.12),
    "ds_stone":         dict(cls="DSRec", ti="stone", dt=0, n=40, f=0.35, th=0.08),
}


def algorithm_file(python):
    """algos/seqsim.py, or a copy starting with a shebang that selects the pyDev environment."""
    source = os.path.join(SP, "algos", "seqsim.py")
    if not python:
        return source
    target = os.path.join(SP, "algos", "seqsim_run.py")
    with open(source) as f:
        code = f.read()
    with open(target, "w") as f:
        f.write(f"#!{python}\n" + code)
    return target


def run(name, port, python=None, seed=7):
    base, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        size = (int(suffix), int(suffix))
    else:
        base, size = name, (80, 80)
    cfg = dict(RUNS[base], output=f"{name}.pkl")
    ti = load_ti(cfg["ti"])
    os.makedirs(os.path.join(SP, "records"), exist_ok=True)
    with open(os.path.join(SP, "records", "next_run.json"), "w") as f:
        json.dump(cfg, f)
    di = numpy.full(size, numpy.nan)
    if cfg["cls"] == "SNESIMRec":
        r = cfg["radius"] * 2 ** cfg["levels"]
        kernel = numpy.ones((2 * r + 1, 2 * r + 1))      # large enough for the template of the coarsest level
        path = multigrid_path(seed, cfg["levels"], size)
        n = kernel.size
    else:
        kernel = numpy.ones(tuple(s // 2 * 2 + 1 for s in ti.shape))   # QS default: uniform, TI/2+1
        path = random_path(seed, size)
        n = cfg["n"]
    t = time.time()
    res = g2s('-sa', 'localhost', '-p', port, '-a', 'pyDev',
              '-pa', f"{algorithm_file(python)}:{cfg['cls']}",
              '-ti', ti, '-di', di, '-dt', [cfg["dt"]], '-ki', kernel, '-n', n,
              '-sp', path, '-s', seed, '-j', 4)
    sim = res["simulation"] if isinstance(res, dict) else res[0]
    numpy.save(os.path.join(SP, "records", f"{name}_sim.npy"), sim)
    numpy.save(os.path.join(SP, "records", f"{name}_path.npy"), path)
    print(f"{name}: {time.time() - t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="configurations to run (default: all at 80 x 80)")
    ap.add_argument("--port", type=int, default=8128, help="port of the G2S server (default 8128)")
    ap.add_argument("--python", help="Python environment for pyDev (shebang of the algorithm file)")
    args = ap.parse_args()
    for name in (args.names or list(RUNS)):
        run(name, args.port, args.python)


if __name__ == "__main__":
    main()
