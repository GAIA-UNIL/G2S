# SNESIM Development Notes

This file is the living technical note for the SNESIM implementation in this branch.
It should be updated every time SNESIM code or behavior changes.

## Goal

Implement a SNESIM algorithm in G2S:

- pixel-based MPS simulation
- multi-grid simulation flow
- categorical variables only
- precomputed search tree
- tree cache persisted per training image (TI) for reuse across runs
- shared immutable tree across worker threads

## Current Status (2026-02-22)

Current scaffold compiles and is wired around the default simulation engine.

Implemented:

- dedicated `snesim` executable
- SNESIM tree/cache module (`snesimTree`)
- SNESIM non-FFT CPU worker module (`snesimCPUThreadDevice`)
- tree cache load/save under `/tmp/G2S/data/snesim_trees/<ti_name>/tree_level_<L>.meta`
- categorical TI/DI validation and automatic category counting
- optional TI-index image (`-ii`) validation for per-node TI selection
- multi-grid plan precomputation inside `src/snesim.cpp`
- default `simulation(...)` call from `include/simulation.hpp` for each level
- shared immutable tree registry and global level switch for all worker threads
- per-level tree strategy selection: `first`, `ii`, `merged` (default `merged`)
- tree deep-copy API and branch-preserving merge API (`MergePolicy::SumCounts`)
- full per-level SNESIM tree construction from TI and `pathPositionArray`
  - contiguous node vector storage
  - fixed-size arrays per node: `categoryCounts[nClasses]` and `childNodeIndex[nClasses]`
  - branch creation by next free node index (`-1` means no branch)
  - count updates at each visited node
  - invalid templates (outside/NaN/unmapped category) skipped for now
- recursive tree-based pixel simulation in `SNESIMCPUThreadDevice::simulatePixel`
  - per-level `pathPositionArray` is shared to all workers
  - depth search follows template order
  - neighbor subset and template path advance together in lockstep order
  - NaN/missing/unknown neighbor values trigger branch-all exploration
  - branch stops on missing child for known neighbor
  - `totalStat` and `maxDepth` logic selects deepest compatible statistics
  - if multiple branches reach same depth, stats are summed bin-by-bin
- per-level internal progress reset removed for SNESIM by using callback-based overall progress aggregation
- path-position dump helper is kept in code but disabled by default
- temporary posterior CSV dump has been removed from SNESIM output
- computation timing now reports the simulation-loop duration (QS-style lines)
- tree build/load phase timing is logged separately (`[SNESIM] tree creation time`)
- simulation neighbor budget now follows full level template length (`numberNeighbor = pathPositionArray.size()`) instead of fixed 16
- conditioning decode now reads categorical class from encoded neighbor vectors (one-hot block), not from raw first scalar
- build integration for c++ and intel Makefiles
- algorithm registration in `build/algosName.config`

Not implemented yet:

- finalized posterior/path strategy for production behavior
- extended multi-variable conditioning/traversal behavior beyond current first-variable categorical path

## Code Map

- App entrypoint and multi-grid orchestration: `src/snesim.cpp`
- WSNESIM entrypoint and multi-grid orchestration: `src/wsnesim.cpp`
- Tree structures and cache: `include/snesimTree.hpp`, `src/snesimTree.cpp`
- SNESIM CPU worker and shared tree-level switch: `include/snesimCPUThreadDevice.hpp`, `src/snesimCPUThreadDevice.cpp`
- Python example script: `example/python/snesim_example.py`
- Python example script (wildcard): `example/python/wsnesim_example.py`
- Build wiring: `build/c++-build/Makefile`, `build/intel-build/Makefile`
- Algo registry: `build/algosName.config`

## CLI (Current Scaffold)

`snesim -ti <ti> -di <di> [options]`

Supported options currently include:

- `-ti` (repeatable): training image id/hash
- `-di`: destination image id/hash
- `-ii`: TI index image id/hash (forces `ii` strategy)
- `-o`: output id/hash alias
- `--tree-strategy`: `first|ii|merged` (default `merged`)
- `-j`, `--jobs`: worker count
- `-mg`, `--mg-level`: max level (execution is always from that max level down to `0`)
- `-tpl`, `--template-radius` (repeatable): template radius (`3` means offsets in `[-3,+3]`)
- `--template-size` (repeatable): deprecated alias, interpreted as radius
- `-tree-root`, `--tree-root`: tree cache root
- `-force-tree`, `--force-tree`, `-ft`: force tree rebuild
- `-s`: seed
- `-r`: report output

## Data Constraints

SNESIM currently enforces:

- TI must be categorical (`DataImage::Categorical` for all variables)
- TI values must be finite and integer-like category labels
- category count is inferred automatically from TI values
- DI must be categorical and compatible with TI variable count
- DI non-NaN known values must belong to TI category set
- if `-ii` is used: TI-index image must match DI grid, have one variable, and contain integer TI ids in `[0, nTI-1]`

## Tree Cache Contract (Current)

Location:

- root default: `/tmp/G2S/data/snesim_trees`
- per TI folder: `<root>/<sanitized_ti_name>/`
- metadata file per level: `tree_level_<L>.meta`

Metadata currently stores:

- source name and cache name
- dimensions and variable count
- category list and frequencies
- template radius config
- grid level
- max conditioning data placeholder
- complete node arrays (`node_<i>_counts`, `node_<i>_children`)

Cache reuse currently checks:

- TI dimensions
- variable count
- category set
- template radius
- grid level
- max conditioning data placeholder

Note:

- merged tree is recomputed each run from per-TI trees (no merged-tree cache yet)

## Threading Model (Current)

- SNESIM uses dedicated CPU worker objects that are not FFT-based.
- Worker class owns static shared tree memory keyed by grid level.
- Shared tree registry stores one tree per TI for each level and an optional merged tree per level.
- Tree handles are immutable (`std::shared_ptr<const SearchTree>`) and shared across threads.
- A global grid-level value is set once per level pass and read by all workers.

## Simulation Flow (Current Scaffold)

1. Load TI and DI, then validate categorical constraints.
2. Precompute multi-grid planning in `buildMultigridPlans(...)`:
   - level list from highest to `0` (`0` is finest)
   - deterministic `pathPositionArray` per level
   - random simulation path per level
   - one posterior path shared across all level runs
3. Build all tree handles per level and per TI:
   - for each level, load cached tree if compatible; otherwise build from that level `pathPositionArray` and cache
   - if strategy is `merged`, merge per-TI trees into one merged tree per level
4. Register shared trees in `SNESIMCPUThreadDevice`, then create one worker per CPU thread.
   - per-TI trees are always registered by level
   - merged tree is also registered by level when strategy is `merged`
   - level `pathPositionArray` is also registered so all workers traverse with same deterministic template order
5. Loop levels from highest to `0`:
   - set global worker level once
   - call default `simulation(...)` with that level path, that level `pathPositionArray`, shared posterior path, and optional `-ii` image
6. Write output image.

## Test Script

Run:

- `python3 example/python/snesim_example.py`
- `python3 example/python/wsnesim_example.py`

What it checks:

- downloads `strebelle.tiff` from the public TI repository
- discretizes TI to categories for SNESIM categorical mode
- runs SNESIM through `g2s(...)`
- shows TI and simulated result side by side

## Documentation Maintenance Rule

For every SNESIM-related code change:

- update this file in the same change set
- update `Current Status`, `Code Map`, and `Simulation Flow` if behavior changed
- add one bullet in `Change Log`
- keep unresolved items listed in `Open Tasks`

## Open Tasks

- finalize and validate production multigrid path/posterior strategy
- add reproducibility and regression tests for cache reuse and categorical validation
- optimize branch-all recursion cost for sparse conditioning cases

## Change Log

- 2026-02-24: Added Python example script `example/python/wsnesim_example.py` showing `-a wsnesim` and wildcard depth usage (`--wd 2`).
- 2026-02-24: Added `wsnesim` wildcard mode switch (`--wd-mode prefix|suffix`, default `suffix`) and propagated mode through tree build, worker traversal, and cache metadata compatibility (`wildcard_mode`).
- 2026-02-24: Added separate `wsnesim` executable/algorithm with wildcard depth (`--wd`), wildcard-aware tree branch `[nClasses]`, ws-specific tree cache root (`/tmp/G2S/data/wsnesim_trees`), and cache compatibility metadata (`branch_count`, `wildcard_enabled`, `wildcard_depth`) while keeping strict `snesim` behavior unchanged (`--wd 0` equivalent path).
- 2026-02-21: Created SNESIM scaffold (app, tree/cache module, non-FFT CPU worker module, build integration, algo registration).
- 2026-02-21: Added Python example script `example/python/snesim_example.py` using `strebelle.tiff` URL loading.
- 2026-02-23: Removed inactive SNESIM trace helper plumbing from `SNESIMCPUThreadDevice` and renamed the Python example to `snesim_example.py`; clarified `-mg` example commentary (`4` means levels `4..0`).
- 2026-02-21: Added global multigrid planning (descending levels, per-level paths, deterministic per-level `pathPositionArray`, posterior path).
- 2026-02-22: Removed `snesimSimulation` module dependency; `src/snesim.cpp` now calls default `simulation(...)` directly per grid level.
- 2026-02-22: Added explicit shared-tree memory and global-level-switch comments/contract in `SNESIMCPUThreadDevice` and tree scaffold comments in `snesimTree`.
- 2026-02-22: Added `--tree-strategy` (`first|ii|merged`, default `merged`) and `-ii` TI-index image routing (if `-ii` is provided, strategy is forced to `ii`).
- 2026-02-22: Refactored SNESIM tree manager to store per-TI trees per level plus optional merged trees; merged trees are recomputed each run (no merged cache).
- 2026-02-22: Added tree merge/deep-copy APIs (`SearchTree::deepCopy`, `SearchTree::addStatisticsFrom`, `mergeTrees`) with branch-preserving sum policy.
- 2026-02-22: Updated multigrid CLI semantics so `-mg`/`--mg-level` is treated as the max grid level; execution always runs levels `max..0`.
- 2026-02-22: Removed `-maxn` from active SNESIM CLI usage (now ignored if provided); current scaffold uses internal fixed neighbor limit for simulation.
- 2026-02-22: Updated template argument semantics to radius-based (`-tpl` / `--template-radius`), so `3` means `±3`; `--template-size` kept as deprecated alias.
- 2026-02-22: Implemented full per-level tree creation with fixed-size node arrays and level-specific tree cache files (`tree_level_<L>.meta`), with legacy `tree.meta` fallback only for level `0`.
- 2026-02-22: Implemented recursive tree simulation traversal with depth-based stat aggregation (`totalStat`, `maxDepth`), NaN branch-all behavior, and per-level shared `pathPositionArray` in CPU workers.
- 2026-02-22: Removed fixed 16-neighbor cap for SNESIM simulation passes (now uses full per-level template length) and fixed categorical conditioning decode from encoded neighbor vectors.
- 2026-02-22: Disabled verbose SNESIM trace/path-position dumps by default and added SNESIM overall-progress reporting across all grid levels using simulation callbacks.
- 2026-02-22: Progress prints inside `simulation.hpp` are now muted when a callback is provided; distributed QS (`dm_qs`) regains progress through callback-side reporting.
- 2026-02-22: Removed SNESIM posterior CSV output and added QS-style computation timing for the SNESIM simulation loop only.
- 2026-02-22: Added explicit tree creation timing logs with `[SNESIM] tree creation time` in seconds and milliseconds plus structured syntax `[SNESIM_TIMING] tree_creation_ms=... tree_creation_s=...`.
