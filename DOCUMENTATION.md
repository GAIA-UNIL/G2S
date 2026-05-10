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

The human log should still be structured enough to follow setup and outputs without reading raw source. The current convention is:

- `INPUT`: successful data/image loads with resolved shape and encoding
- `PARAM`: effective parameter values after parsing, defaulting, and mode selection
- `OUTPUT`: emitted result artifacts with resolved shape and encoding

These log lines are for operators and debugging only. They should not be parsed as the authoritative machine-readable state channel.
