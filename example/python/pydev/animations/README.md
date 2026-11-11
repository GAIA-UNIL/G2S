# Animations of sequential simulation (QS, DS, SNESIM)

Teaching animations that show how a sequential simulation works, node by node, for
QuickSampling, Direct Sampling and SNESIM. They come from real simulations: the algorithms run
through **pyDev** (G2S does the path and the neighbour search, Python predicts each node) and log
every step. The animations only replay that log: the path, each node's data event, the search in the
training image (or in the SNESIM search tree) and the value pasted.

Outputs, for each configuration:

- `out/<name>.mp4`: 1920 × 1080, 30 fps, H.264 (for slides and web pages);
- `out/<name>.svg`: animated SVG (SMIL), looping, about 1.2 MB at 50 × 50 (for web pages; it does
  not play in PowerPoint);
- with a dark background: `out/<name>_dark.mp4` and `out/<name>_dark.svg`.

## Requirements

- A G2S server with pyDev built (`make pydev`, see `docs/algorithms/PyDev.md`), running on this machine.
- A Python environment for pyDev with **numpy**, from the same Python version and build as pyDev
  (for example `python3.14t -m venv pydevenv && pydevenv/bin/pip install numpy`).
- A client environment with `g2s numpy matplotlib tifffile pillow`, and `ffmpeg` on the `PATH`.
- Optional, to check SVGs: Google Chrome (`snap.py`).

## Regenerate everything

```bash
cd example/python/pydev/animations

# 1. simulations (logs in records/); add _50 for a 50 x 50 grid instead of 80 x 80
python run.py --port 8128 --python /path/to/pydevenv/bin/python \
    qs_strebelle qs_stone snesim_strebelle ds_strebelle ds_stone \
    qs_strebelle_50 qs_stone_50 snesim_strebelle_50 ds_strebelle_50 ds_stone_50

# 2. check the logs against the simulations G2S returned
python check.py qs_strebelle qs_stone snesim_strebelle ds_strebelle ds_stone

# 3. videos (a few minutes each; they can run in parallel)
python render.py qs_strebelle
python render.py qs_strebelle --theme dark           # dark background
python render.py qs_strebelle --preview 400 1200     # only PNGs of some frames, to check the layout

# 4. animated SVGs
python svg.py qs_strebelle_50
python svg.py qs_strebelle_50 dark
python snap.py out/qs_strebelle_50.svg out/qs_check 13.4 40   # PNGs of the SVG at 13.4 s and 40 s
```

The training images (Strebelle and stone, from GAIA-UNIL/TrainingImagesTIFF) are downloaded to
`data/` on first use and downscaled by 2 (125 × 125 and 100 × 100).

## Files

| File | Role |
|---|---|
| `algos/seqsim.py` | pyDev algorithms `QSRec`, `DSRec`, `SNESIMRec`: predict each node and log what they did. |
| `run.py` | Runs a configuration through pyDev (fixed path given with `-sp`) and saves the log. |
| `check.py` | Checks a log: path index, value and neighbours of every node against the simulation. |
| `render.py` | Draws the video from a log (matplotlib frames piped to ffmpeg). Defines the timeline. |
| `svg.py` | Builds the animated SVG from the same log and timeline. |
| `snap.py` | Screenshots an SVG at given times with headless Chrome. |

## Algorithms and parameters

Parameters come from the G2S examples:

| Configuration | Parameters | Source |
|---|---|---|
| `qs_stone` | n 50, k 1.2 | `example/python/qs/unconditionalSimulation.py` |
| `qs_strebelle` | n 50, k 2.4 | k of `qs/qs_rotation_2d.py` (k 1.2 copies large pieces of the small TI) |
| `ds_strebelle` | n 48, th 0.12, f 0.4 | `ds/DirectSamplingCategorical.py` |
| `ds_stone` | n 40, th 0.08, f 0.35 | `ds/DirectSamplingContinuous.py` |
| `snesim_strebelle` | template 7 × 7, 5 grid levels | `snesim/snesim_example.py` (`-tpl 3 -mg 4`) |

The Python versions follow the G2S implementations:

- **QS**: mismatch at every TI position (share of differing neighbours for categorical variables,
  mean squared difference for continuous ones), uniform kernel; with a non-integer k the candidate is
  `floor(u·k)` among the `ceil(k)` best, as in `quantileSamplingModule.hpp`.
- **DS**: TI positions scanned in random order, distance = share of differing neighbours or RMSE
  (`-cnorm 2`), first candidate with distance ≤ th accepted, otherwise the best after a fraction f.
- **SNESIM**: one search tree per grid level, built once from the TI (template ordered by Manhattan
  distance; only positions where the whole scaled template fits; centre category counted at every
  node), as in `snesimTree.cpp`. Each node only descends the tree: unknown template nodes branch into
  every child, the counts are read at the deepest node reached (`snesimCPUThreadDevice.cpp`), and the
  category is drawn from them. Multigrid path: every 16th node, then 8th, 4th, 2nd, all.

## Notes

- Colours are set by the themes in `render.py` (`THEMES`, `set_theme`). On the dark background the
  colour map is flipped (low values medium blue, high values pale yellow) so that both categories of
  Strebelle stay visible.

- The run copy `algos/seqsim_run.py` (made by `--python`) and the folders `data/`, `records/` and
  `out/` are not tracked.
- In the SVG, the fast phases only animate the current node and the filling (drawing every data event
  would make the file about ten times larger); transitions are cuts instead of fades.
