# G2S Maintenance Documentation

## Generated Python build tree

`build/python-build/preprocess` creates a local packaging mirror by copying the repository `src`, `include`, `src_interfaces`, and `include_interfaces` trees into `build/python-build/`.

Those copied directories are generated build inputs, not source of truth. They are ignored by git and may be deleted at any time. The `preprocess` target removes the generated mirror before copying fresh files so stale local headers or source files are not mixed with current repository code.

When reviewing, debugging, or changing G2S code, use the repository-level source and header trees. Do not treat ignored files under `build/python-build/src`, `build/python-build/include`, `build/python-build/src_interfaces`, `build/python-build/include_interfaces`, or the local `build/python-build/jsoncpp` clone as canonical project source.

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
- `-returnFormat schema` returns one named dictionary/struct/list envelope instead of the legacy positional tuple/multi-output contract
- Python displays warnings in orange/yellow-ish ANSI text and errors in red before warning/exception propagation
- MATLAB warnings are non-fatal again, while fatal errors still abort the call

The repository includes a small reporting-only utility algorithm, `report_probe`, for exercising this stack without running a full simulation. It is implemented by `src/errorTest.cpp`, registered in `build/algosName.config`, and supports:

- `-mode log`: progress plus plain log lines, then success
- `-mode warning`: progress, plain log lines, one warning event, then success
- `-mode error`: progress, plain log lines, one warning event, a fatal error payload, then nonzero exit

The companion examples `example/python/reporting_probe.py` and `example/matlab/reporting_probe.m` are the intended smoke tests for interface-side warning/error/log rendering and the new schema return mode. Their error-path probe intentionally does not catch the fatal interface exception, so the default behavior matches ordinary caller expectations in Python and MATLAB. Users who want to recover from that failure path should add their own `try`/`except` or `try`/`catch` around the probe call.

Schema mode is designed for forward-compatible callers:

- outputs are named semantically when the algorithm publishes `result_output_<n>_name`
- `artifacts` always contains logical refs for `log`, `warning`, `error`, `progress`, `meta`, and named output artifacts
- top-level reserved interface keys are `simulation`, named outputs, `time`, `job_id`, `status`, `progress`, `artifacts`, `error`, and `warnings`
- remaining progress/meta key-values are flattened at the top level unless they collide with reserved interface keys

During the transition, the legacy positional contract remains the default. Compatibility helpers are available in each binding layer:

- Python: `g2s.schema_to_legacy(result)`
- MATLAB: `g2sSchemaToLegacy(result)`
- R: `g2s_schema_to_legacy(result)`

## Algorithm logging conventions

The human-readable log is now expected to show both setup and effective behavior, not just raw argv. In practice this means:

- successful image/grid loads should produce `INPUT` lines with the resolved source name, shape, variable count, encoding, and variable-type summary
- resolved execution settings should produce `PARAM` lines after argument parsing and defaulting
- emitted outputs should produce `OUTPUT` lines with the resolved artifact id and resulting shape

For `-wPO`, the current conventions are:

- `qs`: logs `path_optimization=true|false` because the flag is effective in QS simulation
- `snesim`: logs `path_optimization_requested=true|false` so the request is visible in the operator log, even though the current scaffold does not expose the same effective mode as QS
