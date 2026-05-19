# G2S: The GeoStatistical Server

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![](https://github.com/GAIA-UNIL/G2S//workflows/C/C++%20CI/badge.svg)
![](https://github.com/GAIA-UNIL/G2S/actions/workflows/pythonPublish.yml/badge.svg)



## Brief overview

**G2S** is composed of 2 parts:
- the first one is a server that manages computations and can be compiled for each hardware to obtain optimal performance.
- the second part is composed of different interfaces that communicate with the server through ZeroMQ. Interfaces can be added for each software. Similarly, G2S can be extended for any other geostatistical simulation algorithm.

Currently the **G2S** interface is available for *MATLAB* and *Python*. **G2S** is provided with **QS** (QuickSampling), **AS** (Anchor Sampling), native **DS** (Direct Sampling), and **NDS** (Narrow Distribution Selection).
Native QS supports deterministic CPU-side search-pattern transforms through `-rmi` rotation maps and `-smi` isotropic scale maps. These maps transform simulation-side neighborhood lookups while leaving the training image unchanged and keeping TI/kernel scoring on the original template offsets. The Python 2D transform examples use 500x500 Strebelle simulations with `-j 1.0001`, stronger transform maps, saved baseline comparison figures, and a constant-rotation diagnostic that compares `-rmi +pi/2` against rotated training images.
Native DS is available as `ds`, `DS`, or `DirectSampling`; the old DS-like implementation remains available as `ds-l`, `dsl`, `DirectSamplingLike`, and `DS-L`. Native DS honors `-ii` TI-selection maps in vector and full simulation, including initial nodes that have no informed neighbors yet. DS candidate scans now use an array-free deterministic pseudo-random permutation keyed by the global seed, local per-node seed, path order, and variable instead of a contiguous wrapped TI segment, path sorting and strict-informed parallel neighbor waits are deterministic for same-seed reproducibility, and the common continuous norms `1` and `2` avoid generic `pow` calls in the hot score loop.
The repository also includes a dedicated **SNESIM** executable for categorical multigrid simulation.
Current interface demos live under `example/<language>/<algorithm>/` and use the schema output interface by default. Legacy positional-output demos were moved to `legacy_example/<language>/` and pass `-legacy_output` directly in each interface call. Native DS examples are under `example/python/ds/` and `example/matlab/ds/`; SNESIM examples are under `example/python/snesim/` and `example/matlab/snesim/`.
Distributed QS interface calls accept native matrix values for JSON-grid parameters such as `-jg`, `-eg`, and `-di_grid_json`; the shared interface layer converts them to JSON before upload and job serialization.
For reporting-path checks, `example/python/reporting/reporting_probe.py` and `example/matlab/reporting/reporting_probe.m` call the server-side `report_probe` utility algorithm and exercise plain logs, warnings, fatal errors, `-showLogs`, and schema-first output through the real interface bindings. Both examples let the fatal error propagate by default so callers only suppress it when they explicitly wrap the call in `try`/`except` or `try`/`catch`.

**G2S** is currently only available for *UNIX*-based systems, *Linux* and *macOS*. A solution for *Windows 10+* is provided using *WSL* (Windows Subsystem for Linux). However, for previous *Windows* versions, the only solution currently available is to install a *Linux* system manually inside a virtual machine. 


## Documentation

An interactive and complete documentation is available [here](https://gaia-unil.github.io/G2S/).
The `docs/algorithms/example/` folder is a generated docs mirror (from `docs/sync_examples.sh`) and is intentionally gitignored.

Documentation and packaging notes should use concise, current wording because they are reused across generated artifacts.

A repository-wide review snapshot is available in `CODE_REVIEW_REPORT.md`.
Maintenance notes for generated build trees and the server protocol schema are available in `DOCUMENTATION.md`.

Internal C++ naming is kept in standard English spelling for shared types and virtual interfaces so generated package mirrors and downstream bindings stay easier to audit.
User-facing messages and build helper output should follow the same spelling conventions.

On Apple Silicon, the internal `fKst` top-k helpers provide ARM NEON float paths for both biggest and smallest candidate scans.

## Local binary data files

`DataImage` `.bgrid` files are an internal, local binary format. They are intended to be written and read by the same G2S build on the same machine/environment, for temporary or local reuse only. They are not a portable exchange format, not a long-term archival format, and are not expected to be transferred between machines, architectures, compiler configurations, or G2S versions.

Because this format uses native in-memory sizes and encodings, compatibility across 32/64-bit systems, endian differences, enum layout changes, or future binary layout changes is intentionally not guaranteed. If `.bgrid` files need to become shareable, durable, or cross-version data, the format should first be replaced or wrapped with an explicit versioned wire format using fixed-width fields, a magic/version header, declared endian policy, payload length validation, and integrity checks.

## Build note

`make` in `build/Makefile` checks whether `include/zmq.hpp` exists. If missing, it auto-downloads `zmq.hpp` from `cppzmq` using `curl` (preferred), then `wget`, then `python`.

For Python wheels, `zmq.h` must also be available. The Python build first tries `pyzmq` include paths (PEP 517 isolated builds), then system include paths. If not found, install ZeroMQ development headers (for example `libzmq3-dev` on Debian/Ubuntu or `zeromq-devel` on RHEL/Fedora).
Windows Python wheels build without zlib linkage, so gzip `.bgrid.gz` helpers are disabled there and plain `.bgrid` payloads remain the supported path.

Some build and packaging helpers fetch third-party source files from upstream default branches, including `cppzmq`'s `zmq.hpp` and JsonCpp for Python packaging. This is an accepted project tradeoff: if an upstream change breaks the build, the local build scripts or package inputs must be updated at that time. For fully reproducible or audited release builds, use pinned package-manager dependencies or a reviewed local dependency snapshot.

The Python packaging preprocess step creates ignored local copies of repository source and header trees under `build/python-build/`. These generated copies are removed before each preprocess run and should be treated as ephemeral build inputs, not source of truth.

## Server protocol schema

The ZeroMQ server protocol starts each request with `infoContainer` from `include/protocol.hpp`, then appends a task-specific payload. The current schema and validation expectations are documented in `DOCUMENTATION.md`.

## Startup version check

When G2S is built with `WITH_VERSION_CONTROL`, the server checks the configured Git remote at startup and prints a message if a newer version appears to be available. This is an optional convenience feature for trusted local or institutional networks, not a security boundary and not a required startup dependency.

For untrusted, restricted, offline, or privacy-sensitive deployments, build without `WITH_VERSION_CONTROL`. If this check ever needs to run safely across untrusted networks, it should first be made explicitly opt-in at runtime and hardened with HTTPS-only URL validation, timeouts, quiet failure handling, and remote host validation.

## Server job launch policy

By default, the server only launches algorithms whose requested `Algorithm` name resolves through `algosName.config`. Unknown names are rejected instead of falling back to `./<Algorithm>`. Use `--allow-unregistered-algorithms` when starting `g2s_server` to explicitly restore the legacy fallback behavior for local development or custom deployments.

Remote job requests are bounded before launch: job JSON is limited to 1 MiB, algorithm names to 2048 bytes, individual argv entries to 64 KiB, and total argv entries to 4096. Requests exceeding those limits are rejected instead of being truncated.

## Server job control

Server startup validates numeric values for `-p` and `-maxCJ`; missing, malformed, or out-of-range values are rejected before binding the socket.

`KILL` requests now fail with a nonzero reply when the requested job id is unknown, malformed, or no longer tracked by the server.

## Server runtime storage

The server stores runtime data under `/tmp/G2S/` by default. Binary payloads stay in `/tmp/G2S/data`, while per-job reporting is now split into:

- `/tmp/G2S/logs`: chronological human-readable job logs
- `/tmp/G2S/warnings`: warning event streams
- `/tmp/G2S/errors`: fatal error payloads
- `/tmp/G2S/progress`: machine-readable in-progress status snapshots
- `/tmp/G2S/meta`: final key/value summaries read at job completion

These directories are shared by jobs triggered through the server and use `0770` permissions, so operators should run the server with the service user and trusted group that are expected to access the shared job data.

At startup, the server probes these runtime directories with a create/remove write test. If any of `/tmp/G2S/data`, `/tmp/G2S/logs`, `/tmp/G2S/warnings`, `/tmp/G2S/errors`, `/tmp/G2S/progress`, or `/tmp/G2S/meta` is not writable, startup fails with an explicit error on `stderr` instead of continuing until a later upload or report update silently fails.

Progress and final duration no longer have to be inferred from the plain log. Interfaces can poll structured `progress_<job>` and `meta_<job>` text artifacts through the existing text-download protocol, while `-showLogs` can tail `log_<job>` and `warning_<job>` for live display without making the server stateful.

Interfaces now use the schema named-result mode by default. Callers receive one dictionary/struct/list object with stable keys such as `simulation`, `indexmap`, `time`, `job_id`, `status`, `progress`, flattened metadata fields, and an `artifacts` sub-object containing logical refs like `log_<job>` and `im_1_<job>`. Schema mode downloads all available result artifacts into that single object even when the language binding receives only one return value. For compatibility with older servers that do not publish output descriptors, output 1 defaults to `simulation` and output 2 defaults to `indexmap`. The old positional return contract remains available by passing `-legacy_output`; it also takes precedence if other return-format options are present.

Server-registered algorithms publish schema descriptors for their conventional runtime outputs where applicable: QS, AS, DS, legacy DS-L, NDS, AutoQS, SNESIM, and Echo name their `im_<n>_<job>` artifacts so schema callers can use stable result keys instead of generic positional labels.

Schema examples are organized by algorithm under `example/python/` and `example/matlab/`, including schema-output conversions of the old flat example set. The old positional examples remain available under `legacy_example/` and pass `-legacy_output` directly.
To smoke-test all current and legacy examples, install the Python example dependencies from the local package metadata, then run `python3 example/python/run_all_examples.py` for Python or call `run_all_examples` from MATLAB after adding `example/matlab` to the path. The MATLAB runner executes each script in an isolated workspace, including legacy filenames such as `3D.m`. The reporting probes' fatal paths are counted as expected failures by both runners.
MATLAB schema examples that read secondary outputs may use `[result, ~] = g2s(...)`; the first value is still the schema struct, and the ignored value keeps older mex builds fetching the second image artifact.

The server remains stateless for reporting delivery. It only exposes current artifact contents; each interface keeps its own per-job read cursors such as `log_offset` and `warning_offset` so only newly appended text is displayed on each poll.

Human-readable algorithm logs are also being standardized around explicit sections:

- `INPUT`: each successfully loaded image/grid and its resolved dimensions, variable count, encoding, and variable-type summary
- `PARAM`: effective parameter values actually used after parsing and defaulting, including path-optimization flags when the algorithm accepts them
- `OUTPUT`: each written result artifact and its final dimensions/encoding

That keeps the chronological log useful for operators while the structured sidecars remain the source of truth for progress and final metadata.

For quick end-to-end validation of that path, the built-in `report_probe` algorithm (served by `errorTest`) accepts `-mode log`, `-mode warning`, or `-mode error` and emits the corresponding structured artifacts through the same reporting helpers used by the main algorithms.

The current `-wPO` logging convention is intentionally explicit:

- `qs` and native `ds` log `path_optimization=true|false` because the flag is parsed and used by the vector simulation path logic
- `snesim` logs `path_optimization_requested=true|false` so operators can see the requested CLI flag even though the current scaffold does not expose an effective path-optimization mode in the same way as QS

## Server data protocol hardening

Data request frames are validated before dispatch. Uploads require exactly 64 hex hash characters, download/existence names are limited to safe 64-byte identifiers, job-id operations require exactly one `jobIdType`, and upload/download payloads are bounded.

The server recomputes upload hashes before storing `.bgrid` and JSON payloads. Mismatched names are rejected, `.bgrid` hashes cover the full serialized frame payload, serialized sizes must match the frame payload, and files are published through temporary files so normal uploads do not overwrite existing content-addressed objects.

Stored `.bgrid` payloads are read using the actual file or decompressed byte count. The embedded serialized size must match the bytes read, dimensions and variable counts are bounded, and malformed files are rejected instead of being allocated, sent back to clients, or deserialized from a short reply frame.

## Runtime data transfer limits

`sendData` and `sendJson` currently load the complete response file into memory and send it as one ZeroMQ message. This keeps the existing protocol simple, but large outputs can temporarily increase memory use and response latency. This is an accepted limitation for current workloads; if large simulations make this a practical issue, the protocol should be extended with explicit size limits, chunked transfer, or a zero-copy/memory-mapped path.

## AutoQS calibration noise

AutoQS calibration noise (`-ln`) randomizes neighbor-offset swaps using indexes drawn across the full neighbor vector.
AutoQS calibration intentionally only runs the vector calibration path; full and augmented-dimensional simulation modes are ignored by calibration.

## Reproducibility

Random seeds are intended to make runs reproducible within the same machine, build configuration, G2S version, compiler/runtime environment, thread configuration, and input data. Reproducibility is not guaranteed across different machines, architectures, compiler versions, library versions, G2S versions, or concurrency settings.

The internal `fKst` top-k helpers used by QS/NDS maintain sorted candidate buffers. Scalar and SIMD paths are expected to visit each array value exactly once, and randomized tie handling applies the same threshold inclusivity when equal values are sampled.

## AS mask order (`-mi`)

In Anchor Sampling, candidate mismatch is computed first, `-mi` invalid entries (`NaN`, `inf`, and non-positive values) are excluded before top-`k` ranking, and `-mi` weights are then used for weighted draw within the retained candidates.

## AS continuous norm (`-cnorm` / `-cn`)

Anchor Sampling continuous mismatch now supports configurable Lp norms with the proper root:
`pow(sum(pow(abs(diff), p)), 1/p)` (kernel-weighted internally).
Use one value to apply the same `p` to all continuous variables, or pass one value per continuous variable to configure each independently.

## Python AS example note

`example/python/as/anchor_sampling.py` is a minimal synthetic AS demo using schema output. The richer AS diagnostic is available as `example/python/as/AnchorSamplingSyntheticExperiment.py`, with the positional-output version kept under `legacy_example/python/AnchorSamplingSyntheticExperiment.py`.

## Online Demo (Back! but slow)

An interactive online version is available [here](https://www.mgravey.com/mps.online/), to experiment with small unconditional simulations.
