# Changes

## 2026-05-12

- Removed the short-lived SNESIM path-optimization CLI support, logging, and documentation. SNESIM now always uses the default per-level execution order built by its multigrid planner.

## 2026-05-10

- Split server-side reporting into `/tmp/G2S/logs`, `/tmp/G2S/warnings`, `/tmp/G2S/errors`, `/tmp/G2S/progress`, and `/tmp/G2S/meta`, while keeping the server stateless and reusing the existing text-download protocol through conventional artifact names such as `progress_<job>` and `meta_<job>`.
- Added a shared reporting helper for server-launched jobs so the main algorithms can publish structured progress, final timings, warnings, and generic failure metadata alongside the human-readable log.
- Changed status polling to prefer structured progress and metadata files over regex scraping from the plain log, with log parsing kept only as a backward-compatible fallback.
- Updated the Python interface to print warnings in orange and errors in red, keep fatal errors as exceptions, support live log/warning streaming with `-showLogs`, and optionally return final key/value metadata with `-returnMeta`.
- Updated the MATLAB interface so warnings are non-fatal again, warnings use MATLAB's warning channel, fatal errors still abort the call, and `-returnMeta` can return the final key/value summary as a struct.
- Added the same metadata-return and live-log plumbing in the shared interface template and the R binding so the C++ interface layer stays consistent across bindings.
- Updated the main algorithm entrypoints to emit explicit `INPUT`, `PARAM`, and `OUTPUT` log lines so successful image loads, effective parameter values, resumed backups, and written result artifacts are visible in the chronological log instead of only appearing implicitly in raw argv or failure messages.
- Updated `qs` and `snesim` logging for `-wPO`: QS now reports the effective `path_optimization` value in its `PARAM` block, and SNESIM records when `-wPO` was requested instead of letting the flag fall through as an ignored argument.
- Expanded `README.md` and `DOCUMENTATION.md` to describe the stateless cursor-based polling model, `-showLogs` / `-returnMeta`, and the operator-facing meaning of `-wPO` log entries in QS and SNESIM.
- Expanded the MATLAB SNESIM example to use the public Strebelle training image more robustly, with a fallback download path for MATLAB releases that cannot `imread()` HTTP URLs directly, plus elapsed-time reporting.
- Corrected the MATLAB SNESIM example to follow the categorical MATLAB test pattern more closely by casting the Strebelle TIFF and destination grid to `single`, using a normal `-j 0.5` thread setting, and rendering categorical outputs with `imagesc`.
- Updated the SNESIM algorithm page and top-level README to point to the MATLAB/Python Strebelle examples explicitly.
- Fixed the SNESIM `simulation()` call to match the current callback-aware signature, preserving explicit `posteriorPath` tracking and progress callbacks when path optimization is disabled or enabled.


## 2026-05-01

- Added explicit `/tmp/G2S/data` and `/tmp/G2S/logs` writability diagnostics: server startup now probes both runtime directories, upload failures log the failing operation, and payload publication errors print the target path instead of failing silently.
- Added ARM NEON `float` fast paths for `fKst::findKSmallest` so Apple Silicon has smallest-side SIMD parity with `findKBiggest`.
- Hardened `fKst` top-k helpers (`fKb.hpp` / `fKs.hpp`) against `k == 1` boundary reads, fixed SIMD scans so every array value is visited exactly once, and kept randomized SIMD tie handling consistent with scalar threshold comparisons.
- Standardized typo-like internal names across C++ interfaces and examples, including conversion types, frequency memory-address types, cross-measurement helpers, and stale misspelled example filenames.
- Fixed obvious spelling mistakes in comments, status messages, and build helper output.
- Kept gzip-only `DataImage` helpers out of Windows Python extension builds, matching the existing no-zlib wheel configuration.
- Made the Python build preprocess step remove ignored generated source/header mirrors before copying fresh repository trees.
- Documented `build/python-build` generated trees as ephemeral packaging inputs and added server protocol schema/validation notes.
- Documented `.bgrid` / `DataImage` serialization as an internal local-only binary format, not a portable exchange or archival format.
- Clarified the `.gitignore` `*.bgrid` rule to reflect that generated local binary data files should not be tracked.
- Documented the `WITH_VERSION_CONTROL` startup version check as a trusted-network convenience feature and noted that untrusted deployments should build without it or harden it first.
- Clarified `.gitignore` local configuration entries.
- Documented unpinned third-party build downloads as an accepted project tradeoff for current build workflows.
- Restricted `DataImage::write` logical filenames to letters, digits, `_`, `.`, and `-` before creating `/tmp/G2S/data` symlinks.
- Clamped invalid or negative `-maxCJ` values to `1` and fixed the queue limit check to stop at the configured maximum.
- Fixed quantile and CPU-thread accelerator seed conversion to avoid `UINT_MAX` float rounding warnings.
- Documented full-payload `sendData` / `sendJson` transfers as an accepted current workload limitation.
- Documented seeded-run reproducibility as scoped to the same machine, version, build/runtime environment, and thread configuration.
- Recomputed server-side hashes for `.bgrid` and JSON uploads, moved `.bgrid` hashes to the full serialized payload, rejected mismatches, and switched data writes to temporary-file publication so ordinary uploads cannot overwrite existing content-addressed payloads.
- Hardened server startup parsing for `-p` and `-maxCJ` so missing, malformed, or out-of-range values fail cleanly.
- Made unsupported AutoQS full and augmented-dimensional simulation enum cases explicit no-ops during calibration to avoid compiler switch warnings.
- Restricted server job execution to algorithm names registered in `algosName.config` by default.
- Added the `--allow-unregistered-algorithms` server flag for deployments that intentionally need the legacy `./<Algorithm>` fallback.
- Hardened server `KILL` handling so unknown or malformed job ids are rejected instead of risking invalid queue access or process-group signalling.
- Replaced fixed-size stack buffers in remote job argv construction with owned strings and explicit request, algorithm, argument, and argv-count limits.
- Hardened data upload/download handling: request frames now validate hash and job-id lengths, `.bgrid` files must match their actual payload size, serialized dimensions/data sizes are bounded, and truncated or malformed files are rejected before server send or client deserialization.
- Changed server-created `/tmp/G2S` runtime directories from world-writable `0777` to shared service owner/group `0770` permissions.
- Fixed AutoQS calibration noise so random neighbor-offset swaps can select indexes across the full neighbor vector.

## 2026-04-30

- Added `CODE_REVIEW_REPORT.md`, a repository-wide code review covering correctness, data integrity, security, performance, tests, dead code, and maintainability risks.

## 2026-04-27

- Applied small code-quality cleanup: fixed obvious typos, one direct measurement syntax issue, and refreshed maintenance notes.

## 2026-04-19

- Fixed `build/Makefile` `zmq.hpp` bootstrap to work reliably in CI:
  - resolved `include/zmq.hpp` from the Makefile location (independent of current working directory),
  - removed compile-probe-based detection for `zmq.hpp`,
  - added downloader fallback order: `curl`, then `wget`, then `python`,
  - clarified failure message (`curl/wget/python` or manual `include/zmq.hpp`).
- Updated docs and ignore rules for `zmq.hpp` auto-download temp file handling.
