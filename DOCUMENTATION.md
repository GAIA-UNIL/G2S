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
