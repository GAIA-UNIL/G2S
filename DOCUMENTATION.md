# G2S Maintenance Documentation

## QS WebAssembly architecture

The browser target is intentionally a small QS-only layer rather than a second server implementation. `src/qs.cpp` keeps the existing native CLI and adapters; `include/qsCore.hpp` and `src/qsCore.cpp` provide the reusable in-memory request/result entry point; `src/qsWasmBindings.cpp` translates typed JavaScript values into that entry point. `G2S_BROWSER_BUILD` isolates MEMFS cleanup and single-thread FFTW planning changes from native builds.

`browser/src/g2s-api.js` exposes `G2S.create()`, jobs, typed array loading, typed configuration, Promise-based execution, progress callbacks, cancellation, and result access. `browser/src/g2s-worker.js` owns the Emscripten module and all simulation work. Cancellation terminates that worker; a later run creates a clean worker. `browser/src/g2s-bridge.js` keeps the page available between commands and translates interface manifests into the typed API.

The initial supported browser set is current Chrome/Chromium and Firefox. Safari is not a release gate. The Wasm target is CPU-only and excludes ZeroMQ, CUDA, OpenCL, distributed execution, autosave, and server tasking. It emits a single-thread compatibility artifact and a separate OpenMP/pthread artifact because Emscripten cannot provide runtime thread fallback in one binary.

The pthread artifact uses an eight-worker pool and the existing QS path-level OpenMP regions. FFTW planning and execution remain internally single-threaded to avoid nested parallelism. `src/wasmOpenMpMicrotask.cpp` is linked only into this artifact and extends the pinned Emscripten OpenMP dispatcher for QS regions that capture more than its stock 15-argument WebAssembly limit. Native builds do not compile this file.

### Reproducible browser build

The build requires emsdk 6.0.3. `build/wasm-build/Makefile` rejects another `emcc` version. `build/wasm-build/fetch_fftw.sh` downloads FFTW 3.3.11 and verifies SHA-256 `5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1` before building float/static FFTW without Fortran, OpenMP, or FFTW threads. The loopback listener vendors cpp-httplib v0.50.1 under `include_interfaces/third_party/cpp-httplib/` with its MIT license.

Run:

```sh
source /path/to/emsdk/emsdk_env.sh
make -C build wasm
make -C browser/test transport
python3 browser/serve.py
```

`make -C build wasm-single` and `make -C build wasm-threaded` build either artifact independently. `browser/serve.py` supplies `Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`, and `Cross-Origin-Resource-Policy: same-origin`; the pthread artifact cannot run without a cross-origin-isolated page and `SharedArrayBuffer`.

The Python interface must also be rebuilt and reinstalled from this checkout before using `-sa browser`:

```sh
make -C build python
python3 -m pip install --force-reinstall build/python-build/dist/g2s-*.whl
```

If an example reports that the ordinary G2S server is offline, first confirm that Python is not importing an older wheel.

Then open `/test/smoke.html?threaded=1` from that server to exercise four-thread execution, progress, cancellation, worker recreation, and ten sequential simulations. `browser/dist`, downloaded FFTW sources, intermediate libraries, and native test binaries are generated and ignored.

### JavaScript API

Inputs must be `Float32Array` objects with positive `shape` values and one variable type (`"continuous"` / `0` or `"categorical"` / `1`) per interleaved variable:

```js
const engine = await G2S.create({
  workerUrl: new URL("./g2s-worker.js?threaded=1", import.meta.url),
  maxThreads: 4
});
const job = engine.createJob();
job.loadArray("trainingImage", trainingData, {
  shape: [width, height], variableTypes: ["categorical"]
});
job.loadArray("destination", destinationData, {
  shape: [width, height], variableTypes: ["categorical"]
});
job.configure("qs", { candidates: 2, neighbors: [40], seed: 123, threads: 4 });
job.onProgress(({ percent, message }) => updateProgress(percent, message));
const result = await job.run();
const simulation = result.getArray("simulation").data;
const index = result.getArray("index").data;
```

Calling `job.cancel()` rejects the current Promise with `AbortError`. The engine remains reusable and creates a clean worker for the next run.

Direct consumers that omit `workerUrl` receive the single-thread compatibility worker. The supplied bridge page performs the cross-origin-isolation check and threaded worker selection automatically.

The bridge page chooses `g2s-qs-pthreads.mjs` when `crossOriginIsolated` is true and otherwise uses `g2s-qs.mjs`. Its spinner defaults to four, is bounded by `navigator.hardwareConcurrency` and the eight-worker build pool, and can be changed between jobs. Python/MATLAB `-j` values retain their native interpretation: an integer requests that count, a non-integer scales the browser capacity, and a non-positive value requests the maximum. Results include `requested_threads`, `effective_threads`, and `browser_max_threads`; a clamped request also includes `thread_warning`.

Progress comes from the existing QS simulation reporting callback. The browser shows it immediately and forwards updates to the loopback listener at a throttled rate, so the page and Python/MATLAB observe the same measured simulation progress without flooding HTTP.

### On-demand loopback protocol

`include_interfaces/interfaceTemplate.hpp` branches to `src_interfaces/browserTransport.cpp` only for `-sa browser` (or `web`). Python wheels and MATLAB MEX builds compile this same source. The listener binds only `127.0.0.1`, fails when its port is already occupied, and exists only for the duration of one synchronous command. The preloaded page continues polling after the listener disappears, so later commands require no page reload or click.

The default listener is `http://127.0.0.1:8129`; browser mode accepts any page origin by default and echoes the requesting origin in CORS responses. `-browserOrigin` can restrict an individual command to one exact origin. Each command creates a random session ID and 192-bit nonce. After session discovery, every request must provide protocol version 1, that session ID, and that nonce. The listener answers preflight and Chromium Local Network Access negotiation; the browser requests `targetAddressSpace: "loopback"` where supported and otherwise uses ordinary CORS for Firefox. Known public page origins include `https://www.mgravey.com` and `https://mps-online.mathieu-1cc.workers.dev`.

Protocol v1 uses these endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/v1/session` | discover protocol, session, nonce, and algorithm |
| `GET` | `/v1/job` | retrieve the typed job manifest |
| `GET` | `/v1/arrays/{id}` | retrieve one float32 input body |
| `GET` | `/v1/control` | heartbeat and cancellation state |
| `POST` | `/v1/progress` | deliver throttled progress |
| `POST` | `/v1/results/{id}` | upload one validated result body |
| `POST` | `/v1/complete` | finish with duration and metadata |
| `POST` | `/v1/error` | finish with a structured browser error |

Array dimensions, variable counts, element counts, encodings, and byte lengths are checked before allocation or conversion. Bodies are canonical cell-major, interleaved-variable, little-endian 32-bit values. Integer result maps retain their `uint32` or `int32` encoding while using the same four-byte transport body. `.bgrid` is not a bridge format because it contains native-width fields.

The default connection and heartbeat timeout is 30 seconds. `-TO` replaces it; `-noTO` cannot disable the finite browser timeout. A missing page, denied local-network request, wrong origin/session/nonce, browser error, or lost heartbeat returns an interface error rather than waiting indefinitely. A running simulation may exceed the timeout as long as the page continues its one-second control heartbeat.

For production HTTPS hosting, include `http://127.0.0.1:8129` in `connect-src` and configure the production host to return the same COOP/COEP headers as `browser/serve.py`. Retest in both Chrome and Firefox because local-network permission policies are browser-controlled and may change. Use `-browserOrigin https://host[:port]` only when an exact-origin restriction is desired; do not broaden the listener bind address.

## Static deployment package

`browser/deploy/` is intentionally copyable as a complete static preview site. It is a try-before-installing experience, not a replacement for the local G2S server: the local server remains recommended and is expected to be roughly 5–10× faster for current workloads. The package contains `index.html`, `app.js`, the local Lottie runtime and `assets/qs_logo.json`, Stone and Strebelle training images, the browser engine files, and both compiled Wasm variants. Its in-page demo runs full-size QS simulations with a new seed on every run, and the optional Python/MATLAB bridge does not use server-side simulation. Run `make -C build wasm` before copying the folder so the Wasm files match the current source. The deployment host must return COOP/COEP/CORP headers and allow `http://127.0.0.1:8129` in `connect-src`; see `browser/deploy/README.md`.

## Generated Python build tree

`build/python-build/preprocess` creates a local packaging mirror by copying the repository `src`, `include`, `src_interfaces`, and `include_interfaces` trees into `build/python-build/`.

Those copied directories are generated build inputs, not source of truth. They are ignored by git and may be deleted at any time. The `preprocess` target removes the generated mirror before copying fresh files so stale local headers or source files are not mixed with current repository code.

When reviewing, debugging, or changing G2S code, use the repository-level source and header trees. Do not treat ignored files under `build/python-build/src`, `build/python-build/include`, `build/python-build/src_interfaces`, `build/python-build/include_interfaces`, or the local `build/python-build/jsoncpp` clone as canonical project source.

## Native Direct Sampling

Native `ds` is implemented by `src/ds.cpp` and `include/directSampling.hpp`. It intentionally reuses the generic `simulation()` / `simulationFull()` orchestration and only opts into additional `SamplingModule` hooks for raw neighbor values, strict informed-neighbor filtering, safe TI-id resolution, per-node sample context, and kernel flat-index mapping. Legacy `ds-l` remains implemented by `src/ds-l.cpp` and should not be changed when working on native DS behavior.

Native DS uses the first configured kernel as its default mismatch kernel and switches to a per-node kernel only when a valid `-kii` map selects one. `-ii` maps are interpreted per simulation cell in both vector and full simulation; multi-channel `-ii` maps may provide per-variable TI selection for full simulation. Candidate patterns that cannot support the full observed data event inside the selected TI are rejected instead of being scored on a partial neighborhood. After simulation, native DS applies a conservative categorical singleton cleanup that preserves conditioning values and only flips one-cell islands fully surrounded by another categorical value.

Native DS visits TI candidates through a deterministic pseudo-random permutation over the allowed flattened TI candidates. The permutation is array-free, bounded by `-f` / `-mer`, and keyed by the global seed, local per-node seed, simulation path order, and variable. Per-node sample context is stored per thread so parallel path workers do not overwrite each other's candidate-order context.

Explicit `-sp` path ordering and `-wPO` dependency ordering use deterministic tie-breakers. This is required for repeated same-seed DS runs because equal path priorities or dependency depths otherwise leave the visit order unspecified, especially when path optimization uses a parallel sort implementation.

When a sampler requests strict informed neighbors, the parallel simulation loop waits until earlier-path neighbor values are written before adding them to the data event. Native DS depends on this behavior for reproducibility under `-j`; otherwise thread timing can decide whether a still-pending neighbor is skipped.

Continuous DS mismatch keeps the generic `pow` path for custom `-cn` / `-cnorm` values, but shortcuts the common `1` and `2` powers with direct absolute-difference and multiplication/square-root operations in the candidate scoring loop.

The Python and MATLAB native DS examples are intentionally fully unconditional: their destination images are all `NaN`, use the same spatial and variable shape as the loaded training image, and pass `-j 1.00001` to exercise path-level parallel execution. Do not add sparse demonstration conditioning points to those examples unless the example is explicitly renamed and documented as conditional.

The transform helper in `include/qsTransformUtils.hpp` is shared by QS and native DS. Rotation and scale tolerance maps are inert unless provided by a caller; existing QS deterministic transform behavior should remain unchanged.

## Server protocol schema and validation

G2S clients and the server communicate over ZeroMQ using one binary request frame. Every request starts with `infoContainer`, defined in `include/protocol.hpp`:

```cpp
struct infoContainer {
    int version;
    taskType task;
};
```

The remaining payload depends on `taskType`. Current task payload shapes are:

| Task | Payload |
| --- | --- |
| `EXIST` | 64-byte content identifier |
| `UPLOAD` | 64-byte content identifier followed by serialized `.bgrid` payload |
| `DOWNLOAD` | 64-byte content identifier |
| `JOB` | job JSON bytes |
| `PROGESSION` | one `jobIdType` |
| `DURATION` | one `jobIdType` |
| `KILL` | one `jobIdType` |
| `UPLOAD_JSON` | 64-byte content identifier followed by JSON bytes |
| `DOWNLOAD_JSON` | 64-byte content identifier |
| `SHUTDOWN` | no payload |
| `SERVER_STATUS` | no payload |
| `JOB_STATUS` | one `jobIdType` |
| `DOWNLOAD_TEXT` | 64-byte content identifier |

Validation must happen before a request payload is cast, copied, deserialized, used as a filename, or passed to task-specific handlers. At minimum, request handling must check:

- the frame is at least `sizeof(infoContainer)`;
- `version` is positive and understood by the receiver;
- `task` is one of the supported enum values;
- fixed-size payloads match exactly;
- variable-size payloads are bounded by explicit limits;
- 64-byte content identifiers contain only the expected safe hash or object-name format for that task;
- job JSON, data payloads, and text payloads are rejected when malformed or oversized;
- invalid requests produce a deterministic error reply.

The server currently performs part of this validation in `src/server.cpp` before dispatch and in data storage helpers. Future protocol changes should move toward one parser/serializer layer that returns typed validated requests, so new task cases do not repeat manual size checks and byte casts.

New or changed task types should include parser tests, including malformed and boundary-size frames. Fuzzing the request parser is the preferred way to cover truncated frames, oversized frames, unknown task ids, invalid content identifiers, and mismatched serialized payload sizes.

## Structured reporting artifacts

The server remains stateless with respect to client delivery state. Interfaces are responsible for remembering what they have already displayed.

Structured per-job reporting now uses sidecar files under `/tmp/G2S/`:

- `/tmp/G2S/logs/<job>.log`: chronological human-readable log
- `/tmp/G2S/warnings/<job>.txt`: warning event stream
- `/tmp/G2S/errors/<job>.txt`: fatal error payload
- `/tmp/G2S/progress/<job>.kv`: current machine-readable progress snapshot
- `/tmp/G2S/meta/<job>.kv`: final key/value summary

The existing `DOWNLOAD_TEXT` task is reused for these artifacts through conventional names:

- `log_<job>`
- `warning_<job>`
- `error_<job>`
- `progress_<job>`
- `meta_<job>`

This keeps the server stateless:

- the server only returns the current contents of the requested artifact;
- the interface keeps local cursors such as `log_offset` and `warning_offset`;
- structured progress is polled as current state, not tailed as an append-only stream.

`progress_<job>` is a line-based key/value file. Typical keys include:

- `status`
- `progress_percent`
- `stage`
- `stage_detail`
- `current_step`
- `total_steps`
- `last_update_unix_ms`

`meta_<job>` is also a line-based key/value file and is intended to be read at the end of the run. Typical keys include:

- `job_id`
- `algorithm`
- `status`
- `start_time_unix_ms`
- `end_time_unix_ms`
- `duration_ms`
- `warning_count`
- algorithm-specific timing keys such as `tree_creation_ms` or `simulation_ms`

Interfaces should prefer the structured `progress` and `meta` files for progress and duration. Plain logs should be treated as human-facing traces, not as the canonical machine-readable status source.

## Stateless interface polling

The reporting path is designed to keep the server stateless. The server does not remember what was already sent to any client or interface. Instead:

- the server exposes the current contents of `log_<job>`, `warning_<job>`, `error_<job>`, `progress_<job>`, and `meta_<job>`
- the interface keeps local per-job cursors for append-only streams such as `log_offset` and `warning_offset`
- each poll requests the current progress snapshot plus any newly appended log/warning bytes
- final metadata is read from `meta_<job>` once the run finishes

This split matters operationally:

- `progress_<job>` is current state, so interfaces overwrite their previous view on each poll
- `log_<job>` and `warning_<job>` are append-only streams, so interfaces display only the new suffix they have not shown yet
- `error_<job>` is a terminal payload, so interfaces can fetch it when job status changes to failure

This keeps delivery-side state out of the server while still allowing live `-showLogs` output in MATLAB, Python, and other bindings.

The human log should still be structured enough to follow setup and outputs without reading raw source. The current convention is:

- `INPUT`: successful data/image loads with resolved shape and encoding
- `PARAM`: effective parameter values after parsing, defaulting, and mode selection
- `OUTPUT`: emitted result artifacts with resolved shape and encoding

These log lines are for operators and debugging only. They should not be parsed as the authoritative machine-readable state channel.

## Interface display behavior

Bindings now separate transport/state from display:

- `-showLogs` enables live display of newly appended `log_<job>` and `warning_<job>` text while the job runs
- `-returnMeta` returns the final parsed `meta_<job>` key/value payload to the caller
- Python displays warnings in orange/yellow-ish ANSI text and errors in red before warning/exception propagation
- MATLAB warnings are non-fatal again, while fatal errors still abort the call

The repository includes a small reporting-only utility algorithm, `report_probe`, for exercising this stack without running a full simulation. It is implemented by `src/errorTest.cpp`, registered in `build/algosName.config`, and supports:

- `-mode log`: progress plus plain log lines, then success
- `-mode warning`: progress, plain log lines, one warning event, then success
- `-mode error`: progress, plain log lines, one warning event, a fatal error payload, then nonzero exit

The companion examples `example/python/reporting_probe.py` and `example/matlab/reporting_probe.m` are the intended smoke tests for interface-side warning/error/log rendering. Their error-path probe intentionally does not catch the fatal interface exception, so the default behavior matches ordinary caller expectations in Python and MATLAB. Users who want to recover from that failure path should add their own `try`/`except` or `try`/`catch` around the probe call.

## Algorithm logging conventions

The human-readable log is now expected to show both setup and effective behavior, not just raw argv. In practice this means:

- successful image/grid loads should produce `INPUT` lines with the resolved source name, shape, variable count, encoding, and variable-type summary
- resolved execution settings should produce `PARAM` lines after argument parsing and defaulting
- emitted outputs should produce `OUTPUT` lines with the resolved artifact id and resulting shape

For `-wPO`, the current conventions are:

- `qs` and native `ds`: log `path_optimization=true|false` because the flag is effective in vector simulation
- `snesim`: logs `path_optimization_requested=true|false` so the request is visible in the operator log, even though the current scaffold does not expose the same effective mode as QS

## QS deterministic search-pattern transforms

Native QS accepts deterministic CPU-only local search-pattern transforms:

- `-rmi` supplies rotation per simulated node;
- `-smi` supplies isotropic scale per simulated node.

Transforms map original QS template offsets into simulation-space lookup offsets before candidate matching. The training image is not transformed. QS reads already simulated values at transformed offsets, then passes those values to the existing matcher with the original TI/kernel offsets so kernel weights keep their original flat-index mapping. This preserves vector/full simulation behavior while allowing constant rotations or scales to turn or resize TI structures in the simulation.

Supported geometries are 2D and 3D only. 2D rotation maps use one channel containing radians in the XY plane. 3D rotation maps use four channels containing quaternion values in `(qx, qy, qz, qw)` order; invalid or near-zero node quaternions fall back to identity rotation. Scale maps use one channel; invalid node scale values fall back to identity scale.

The Python example `example/python/qs_rotation_equivalence_2d.py` checks the constant-rotation sign convention by comparing `-rmi +pi/2` with clockwise and counter-clockwise rotated training images.
The static package also includes `browser/deploy/_headers`. Keep this file at the deployment root when using Cloudflare Pages; it enables COOP/COEP for the threaded Wasm build.

For Cloudflare Workers Builds, deploy the directory that contains `index.html` and `_headers`, for example `npx wrangler deploy --assets ./public/ --compatibility-date 2026-07-24` when the package is under `public/`. Workers Static Assets parses the `_headers` file inside that asset directory.
