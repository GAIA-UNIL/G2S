# Changes

## 2026-07-23

- Added a documented 200×200 Python stone example for exercising the on-demand browser QS bridge, including configurable size, seed, timeout, interactive plotting, and optional figure export.
- Added a separate eight-worker OpenMP/pthread QS WebAssembly artifact while retaining the single-thread compatibility artifact; FFTW remains internally single-threaded.
- Preserved normal `-j` semantics across Python/MATLAB and JavaScript, added a browser-owned maximum-thread spinner defaulting to four, and report requested/effective limits and clamping warnings in result metadata.
- Added live browser progress and stage display driven by the existing QS reporting callback, plus a local COOP/COEP development server and threaded smoke coverage for cancellation and ten sequential runs.
- Documented the required Python wheel rebuild/reinstall step and made the browser example report the imported G2S location, preventing an older wheel from being mistaken for a browser-bridge failure.
- Added a copy-ready `browser/deploy/` static site with the supplied QS Lottie progress animation, local Lottie runtime, browser-only demo simulation, bundled Wasm assets, and deployment instructions.
- Clarified that the browser package is an experimental try-before-installing preview; the local G2S server remains the recommended high-performance path and is expected to be 5–10× faster for current workloads.
- Hardened the preview bridge against transient heartbeat failures, serialized progress uploads, and prevented duplicate preview tabs from claiming the same interface session.
- Made the local development server serve both IPv4 and IPv6 localhost addresses so generic HTTP servers cannot silently remove the required isolation headers.
- Added Cloudflare Pages `_headers` configuration to the browser deployment package for cross-origin-isolated WebAssembly threads.
- Replaced the synthetic browser demo with selectable Stone and Strebelle training images, randomized seeds, and simulations matching each source image's original dimensions.
- Documented direct Git deployment through Cloudflare Workers Builds using a static asset directory.
- Made browser transport origin handling permissive by default for hosted preview pages while retaining `-browserOrigin` as an exact-origin restriction.
- Added a contextual link from compatibility mode to the Cloudflare-hosted multithreaded browser preview.

## 2026-07-18

- Removed the obsolete Emscripten build target, WebSocket bridge, browser-only server and tasking branches, interface shims, and stale online-demo link. Native C++, Intel, MATLAB, Python, and R paths remain unchanged.
- Added a replacement QS-only browser architecture that compiles the reusable CPU QS path and float FFTW to WebAssembly, executes jobs in a Web Worker, and exposes a typed Promise-based JavaScript API with progress, cancellation, result retrieval, and worker recreation.
- Added one shared Python/MATLAB `-sa browser` transport using a pinned vendored cpp-httplib listener bound to `127.0.0.1`, exact-origin CORS, per-command nonce/session authentication, Local Network Access response headers, validated JSON/float32 protocol messages, a 30-second default connection/heartbeat timeout, and immediate port-conflict errors.
- Added pinned emsdk 6.0.3 and checksum-pinned FFTW 3.3.11 browser build flow, Chrome/Chromium and Firefox smoke fixtures, and native transport tests covering timeout, authentication, binary results, and port conflicts. Generated Wasm, FFTW, and test artifacts remain ignored.

## 2026-05-12

- Removed the short-lived SNESIM path-optimization CLI support, logging, and documentation. SNESIM now always uses the default per-level execution order built by its multigrid planner.

## 2026-05-16

- Added native CPU Direct Sampling as `ds`, `DS`, and `DirectSampling`, with legacy `ds-l` aliases preserved on the old implementation. Native DS reuses the shared simulation path, vector/full modes, per-node control maps, interface upload plumbing, and OpenMP path scheduling while providing DS-specific sequential TI scanning, threshold acceptance, kernel-weighted mixed mismatch, safe `-ii` handling, and stochastic local rotation/scale transforms.
- Fixed shared interface matrix serialization so native matrices passed to distributed QS JSON parameters are converted before job serialization, any remaining matrix-valued parameter is uploaded as a binary payload before the job JSON is built, residual serialization errors include the offending flag name, MATLAB scalar numeric parameters such as `-k`, `-n`, and `-j` are no longer mistaken for uploadable matrices, and fallback upload checks no longer cast already-normalized string values as native matrices.
- Changed native DS candidate scanning from a random-start contiguous TI segment to an array-free deterministic pseudo-random permutation keyed by global seed, local per-node seed, path order, and variable, and made the DS sample context thread-local for parallel runs.
- Fixed native DS reproducibility for repeated same-seed runs by adding deterministic tie-breakers to explicit simulation-path sorting and `-wPO` dependency sorting.
- Fixed the shared parallel simulation neighbor wait logic for strict-informed samplers such as native DS so a worker waits for earlier-path neighbors instead of dropping still-pending `NaN` values based on thread timing.
- Short-circuited native DS continuous mismatch scoring for `-cn` / `-cnorm` values `1` and `2` so the common norm powers avoid generic `pow` calls in the candidate loop.
- Fixed native DS edge cases where no-neighbor nodes ignored `-ii` TI-selection maps, full simulation indexed cell-level `-ii` maps by full variable slot, multi-kernel runs without `-kii` lost the default kernel weights, TI-border candidates could be accepted from partial data-event support, and categorical outputs could retain isolated one-cell islands.
- Added native DS support for the QS-style `-wPO` path-optimization flag in vector simulation, with explicit `path_optimization=true|false` reporting.
- Updated the native DS Python examples to be fully unconditional, to size destination images from the loaded training image instead of using sparse point conditioning on fixed 180x180 grids, and to pass `-j 1.00001`.
- Added matching MATLAB native DS examples for continuous stone, categorical Strebelle, transform-controlled DS, and full mixed multivariate DS.
- Added deterministic CPU-side QuickSampling search-pattern transforms through `-rmi` rotation maps and `-smi` isotropic scale maps. QS now transforms local neighborhood offsets before matching while leaving the training image unchanged, with support for 2D radians and 3D quaternions.
- Adjusted QS transform matching so transformed offsets are used for simulation-side neighbor lookup while TI scoring keeps the original template/kernel offsets, allowing constant rotations to produce visibly rotated structures.
- Expanded the 2D QS transform Python examples to 500x500 Strebelle simulations with same-seed baseline comparisons, stronger transform maps, `-j 1.0001` parallel execution, and saved comparison figures.
- Added a 2D constant-rotation diagnostic example comparing `-rmi +pi/2` with clockwise and counter-clockwise rotated training images.
- Hardened Python interface handling for `-rmi`/`-smi` so transform-map arrays are forced through the normal ZeroMQ upload path for local and remote servers, and added interface/server/QS validation to reject empty transform-map values.
- Reworked `errorTest` into a reusable `report_probe` utility algorithm that emits structured logs, warnings, progress, metadata, and fatal errors through the current reporting helpers instead of writing only the legacy ad hoc error file path.
- Added Python and MATLAB reporting-probe examples that exercise `-showLogs`, warning propagation, fatal error propagation, and `-returnMeta` through the real interface bindings.
- Made the Python reporting-probe example accept interface builds that return extra trailing values beyond elapsed time and metadata, avoiding fixed-length tuple unpack failures during smoke tests.
- Changed the Python and MATLAB reporting-probe examples so the fatal probe now propagates the native interface exception by default instead of catching it inside the demo script.

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
