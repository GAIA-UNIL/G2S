# Changes

## 2026-05-01

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
