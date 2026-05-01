# Repository-Wide Code Review Report

Review date: 2026-04-30  
Repository: `/Users/mathieugravey/githubProject/G2S`

## Executive Summary

G2S is a C++ server plus command-line algorithms with Python, R, MATLAB, packaging, cluster, and documentation layers. The highest-risk issues are in the unauthenticated ZeroMQ server protocol, the job execution path, temporary-file/data handling under `/tmp/G2S`, and unchecked binary payload parsing. There are also correctness bugs in calibration noise injection, typed outputconversion for Python/MATLAB, and several untested or partially wired code paths.

Verification performed:

- `make c++ -j2` from `build/` completed successfully on macOS.
- `./c++-build/test -sampling 1D 2D 3D` from `build/` completed successfully.
- Build emitted warnings in `include/quantileSamplingModule.hpp` for `UINT_MAX` to `float`conversion, and an incomplete `switch` in `src/auto_qs.cpp`.
- No SQL, database migrations, or database schema files were found.

Highest-priority work:

1. Lock down or redesign server job submission before exposing the server beyond trusted local users.
2. Validate all protocol message lengths and serialized payloads before reading them.
3. Fix `KILL` handling to avoid undefined behavior and accidental process-group termination.
4. Replace `/tmp/G2S` world-writable shared storage with per-user/private storage and atomic validated writes.
5. Add tests for server protocol errors, typed outputs, calibration noise, Python/R/MATLAB bindings, packaging, and distributed launchers.

## Critical Issues

### C-1: Unauthenticated network job execution

- Severity: Critical
- File/location: `src/server.cpp:226-230`, `src/server.cpp:285-290`, `src/jobTasking.cpp:93-103`, `src/jobTasking.cpp:242-257`
- Description: The server binds a REP socket to `tcp://*:<port>` and accepts `JOB` messages that choose an algorithm name and parameters. The job runner maps the request to an executable and calls `execvp`/`execv` without authentication, authorization, transport security, or a strict server-side allowlist independent of request data.
- Evidence: `server.cpp` builds `tcp://*:%d` and calls `receiver.bind(address)`. `JOB` requests are passed into `receiveJob`. `jobTasking.cpp` reads `job["Algorithm"]`, builds `./%s`, and executes it.
- Impact: Any client that can reach the port can start compute jobs, pass arbitrary arguments to installed algorithms/scripts, consume CPU/memory/disk, trigger file operations under `/tmp/G2S`, and potentially execute scripts present in the server working directory.
- Suggested fix: Bind to `127.0.0.1` by default, require an explicit `--listen 0.0.0.0` style option for remote use, add authentication or mTLS/CurveZMQ, enforce a hardcoded allowlist of runnable algorithms, reject path separators and unknown names, and run jobs under a least-privilege service account with resource limits.
- Removal risk: Low for adding auth and local-only default; medium if existing cluster workflows depend on unauthenticated remote access.
- Confidence: High.

### C-2: Invalid kill requests can crash the server or signal the process group

- Severity: Critical
- File/location: `src/jobManager.cpp:22-31`, `src/server.cpp:320-325`
- Description: `receiveKill` dereferences `queue.end()` when the requested job is not queued. If that does not crash, `jobIds.look4pid[jobId]` inserts a missing key with PID `0`, then `kill(0, SIGTERM)` signals the server process group.
- Evidence: `std::get<0>(*it)` is called without checking `it != queue.end()`. `operator[]` is used on `look4pid` for an untrusted job ID.
- Impact: A malformed or stale `KILL` request can terminate unrelated jobs and possibly the server itself. It is also reachable through the network protocol.
- Suggested fix: Check the queue iterator before dereference; use `find` instead of `operator[]`; reject unknown job IDs; only signal positive PIDs that are currently tracked; return a nonzero status for missing jobs.
- Removal risk: Low.
- Confidence: High.

### C-3: Remote request strings can overflow fixed-size stack buffers

- Severity: Critical
- File/location: `src/jobTasking.cpp:93-103`, `src/jobTasking.cpp:181-209`
- Description: Job JSON strings are copied into fixed arrays with `strcpy`/`sprintf`. Algorithm names can exceed `algo[2048]` or `exeName[1024]`, and parameter values can exceed the `tempMemory[][100]` slots.
- Evidence: `strcpy(algo, algoStr.c_str())`, `sprintf(exeName,"./%s",algo)`, and `strcpy(tempMemory[tempMemIndex], ...)` are used on protocol-controlled strings.
- Impact: A remote client can corrupt the server stack, crash the server, or potentially gain code execution depending on compiler/runtime mitigations.
- Suggested fix: Replace fixed C arrays with `std::string`/`std::vector<std::string>` and construct `argv` from owned strings. Enforce maximum request and argument lengths at protocol boundaries.
- Removal risk: Low.
- Confidence: High.

### C-4: Server can leak memory contents or exhaust memory from crafted data files

- Severity: Critical
- File/location: `src/dataManagement.cpp:79-107`, `src/dataManagement.cpp:165-193`, `include/DataImage.hpp:154-176`
- Description: Downloads trust the serialized `fullSize` field read from the file, allocate that amount, and send `fullSize` bytes even if the file is shorter or decompression reads less. The deserializer also trusts all sizes and copies without bounds.
- Evidence: `gzread(dataFile, &fullSize, sizeof(fullSize))`, `malloc(fullSize)`, then `gzread(dataFile, buffer, fullSize)` and `memcpy(reply.data(), buffer, fullSize)` without checking read counts. `DataImage(char* raw)` resizes vectors and copies based only on the embedded `fullSize`.
- Impact: A malformed upload or local file under `/tmp/G2S/data` can cause out-of-memory denial of service or return uninitialized heap memory to a client. Malformed downloads can also crash clients that deserialize the payload.
- Suggested fix: Validate hash format, header size, full payload length, compression read counts, maximum allowed dimensions/data size, and `fullSize == actual file payload size`. Return an error on mismatch.
- Removal risk: Low.
- Confidence: High.

## Potential Bugs

### B-1: Protocol messages are read past the received frame size

- Severity: High
- File/location: `src/server.cpp:263-325`, `src/dataManagement.cpp:68-71`, `src/dataManagement.cpp:203-206`
- Description: Several request handlers read fixed-size payloads after only checking that the frame contains `infoContainer`. `EXIST`, `DOWNLOAD`, `KILL`, `JOB_STATUS`, `PROGESSION`, and `DURATION` need additional length checks.
- Evidence: `memcpy(&jobId, request.data()+sizeof(infoContainer), sizeof(jobId))` is performed without `requestSize >= sizeof(infoContainer)+sizeof(jobId)`. Data operations copy 64 bytes of name/hash from `dataName` without checking the incoming frame.
- Impact: Short frames can cause out-of-bounds reads, incorrect job IDs, crashes, or stale memory being interpreted as a path/hash.
- Suggested fix: Add per-task minimum and maximum frame-size validation before dispatch. Return protocol errors for invalid lengths.
- Removal risk: Low.
- Confidence: High.

### B-2: Invalid task/version can wedge the ZeroMQ REP socket

- Severity: High
- File/location: `src/server.cpp:257-262`
- Description: The server calls `continue` for `infoRequest.version <= 0` without sending a reply. The `switch` also has no default reply for unknown task values. REP sockets must send one reply for each received request.
- Evidence: `if(infoRequest.version<=0) continue;` is inside the receive path after a successful `recv`.
- Impact: A malformed request can put the REP socket into an invalid send/receive state, effectively causing a denial of service.
- Suggested fix: Always send an error response for malformed version/task values. Add a default case.
- Removal risk: Low.
- Confidence: High.

### B-3: Calibration noise path never selects random swap indexes

- Severity: Medium
- File/location: `include/calibration.hpp:200-205`
- Description: Random swap positions are computed as `floor(uniform / size)`. Because `uniform` is in `[0, 1)`, dividing by `size > 1` always yields `< 1`, so `floor(...)` is always `0`.
- Evidence: `neighborArrayVector.begin() + floor(uniformDistributionOverSource(randomGenerator)/neighborValueArrayVector.size())`.
- Impact: The requested calibration noise does not randomize neighbor order as intended; AutoQS calibration can report misleading parameter quality.
- Suggested fix: Multiply by the vector size, or use `std::uniform_int_distribution<size_t>(0, size - 1)`.
- Removal risk: Low.
- Confidence: High.

### B-4: Python integer/unsigned outputs copy float bytes into integer arrays

- Severity: High
- File/location: `include_interfaces/python3_interface.hpp:238-245`, `src/qs.cpp:1182-1187`, `src/as.cpp:796-799`, `src/nds.cpp:563-564`
- Description: `DataImage` always stores `float*`, but Python output allocates an `NPY_INT32` or `NPY_UINT32` array for integer encodings and then `memcpy`s raw float bytes into it.
- Evidence: `typ` is set to `NPY_UINT32`, then `float* data = (float*)PyArray_DATA_SAFE(arr); std::memcpy(data, img._data, img.dataSize() * sizeof(float));`. Algorithms mark ID/path images as `UInteger`.
- Impact: Python users receive bit-reinterpreted values instead of numeric integer conversions for encoded ID/path outputs.
- Suggested fix: For integer encodings, allocate the integer dtype and cast each `image._data[i]` to the target integer type. Add tests for QS/AS/NDS `UInteger` outputs.
- Removal risk: Low.
- Confidence: High.

### B-5: MATLAB integer/unsigned outputs are written through a float pointer

- Severity: High
- File/location: `include_interfaces/matlab_interface.hpp:300-315`
- Description: MATLAB output creates `mxINT32_CLASS`/`mxUINT32_CLASS` arrays, then casts `mxGetPr(array)` to `float*` and writes float values into the array storage.
- Evidence: Integer array classes are selected at lines 302-305, followed by `float* data=(float*)mxGetPr(array)` and float writes.
- Impact: MATLAB integer outputs can be corrupted or crash on newer MATLAB APIs where `mxGetPr` is for double data. This affects ID/path outputs.
- Suggested fix: Use `mxGetData`, branch on encoding, and write typed `int32_t*`/`uint32_t*` with explicit casts.
- Removal risk: Low.
- Confidence: High.

### B-6: DirectMeasure delta encoding is mathematically wrong and can address invalid memory

- Severity: Medium
- File/location: `src/DirectMeasureCPUThreadDevice.cpp:221-228`, `include/DirectMeasureCPUThreadDevice.hpp:67-78`
- Description: `candidateForPattern` encodes neighbor deltas with `encoded += encoded * _fftSize[j] + delta`, which double-counts the previous encoded value. It also stores signed deltas and later indexes `_srcCplx` with `index + _encodedDeltaPosition[i]`.
- Evidence: The helper `index(...)` uses `finalValue *= _fftSize[i]; finalValue += deltaVect[i]`, but the implementation uses `encoded += encoded * _fftSize[j] + ...`.
- Impact: If this device is wired in, it can compare the wrong positions or read outside valid memory for negative offsets.
- Suggested fix: Replace with a single tested coordinate-to-linear-offset helper and validate bounds before indexing.
- Removal risk: Medium; the class appears currently unused, so first verify whether any downstream extension loads it.
- Confidence: Medium.

### B-7: AutoQS has unhandled simulation enum values

- Severity: Medium
- File/location: `src/auto_qs.cpp:715-725`
- Description: The build warns that `fullSim` and `augmentedDimSim` are not handled in the `switch`.
- Evidence: `make c++ -j2` emitted `warning: enumeration values 'fullSim' and 'augmentedDimSim' not handled in switch`.
- Impact: AutoQS modes can silently do nothing or produce incomplete calibration results if those enum states are selected.
- Suggested fix: Implement cases or explicitly reject unsupported modes with an error and test coverage.
- Removal risk: Low.
- Confidence: High.

### B-8: Server command-line parsing reads missing values

- Severity: Medium
- File/location: `src/server.cpp:109-125`
- Description: `-maxCJ` and `-p` read `argv[i+1]` without checking that a value exists.
- Evidence: `maxNumberOfConcurrentJob=atoi(argv[i+1])` and `port=atoi(argv[i+1])` have no `(i+1 < argc)` guard.
- Impact: Starting the server with a missing option value can read out of bounds and crash.
- Suggested fix: Use a small option parser or add value guards and validation ranges for port/concurrency.
- Removal risk: Low.
- Confidence: High.

## Database & Data Integrity Review

No database models, SQL files, or migrations were found. The main data-integrity risks are the custom binary/image store and temporary JSON/text files.

### D-1: Server trusts client-supplied hashes and payload metadata

- Severity: High
- File/location: `src/dataManagement.cpp:30-65`, `src/dataManagement.cpp:116-151`
- Description: Upload handlers accept a client-supplied 64-byte hash/name and write the payload under that name without recomputing the digest or validating that the embedded serialized size matches the frame size.
- Evidence: `storeData` copies `hash` from the request, subtracts 64 bytes, and writes the rest directly.
- Impact: Clients can overwrite or spoof content-addressed names, poison another job's inputs/results, and create corrupted files that later crash readers.
- Suggested fix: Recompute SHA-256 server-side, reject mismatches, validate serialized payloads before storing, and use exclusive/atomic writes.
- Removal risk: Medium; legacy clients may rely on precomputed names, but the server can still verify and return the canonical hash.
- Confidence: High.

### D-2: Shared `/tmp/G2S` storage should be restricted to the server group

- Severity: Medium
- File/location: `src/server.cpp:195-198`, `src/DataImage.cpp:45-67`, `include/DataImage.hpp:202-214`
- Description: The server intentionally shares `/tmp/G2S`, `/tmp/G2S/data`, and `/tmp/G2S/logs` across jobs triggered by the server, but the directories should be writable only by the server owner/trusted group instead of every local user.
- Evidence: Runtime directories are created and chmodded with `0770`; hardcoded `/tmp/G2S/data/...` paths remain throughout the code.
- Impact: Deployments that intentionally put untrusted users in the server's runtime group may still allow those users to tamper with data/logs, delete or replace files, cause wrong simulation inputs/outputs, or stage malformed files for server/client crashes.
- Suggested fix: Ensure deployment creates or owns `/tmp/G2S` with the server service user and trusted group using `0770`; support `G2S_DATA_DIR` consistently in the server for deployments that need a different shared storage root; use `open(..., O_NOFOLLOW|O_CREAT|O_EXCL)` or equivalent safe writes.
- Removal risk: Low to medium for existing cluster workflows; existing directories may need one-time ownership or permission migration.
- Confidence: High.

### D-3: Serialized `DataImage` format is architecture-dependent and unversioned

- Severity: Medium
- File/location: `include/DataImage.hpp:125-176`
- Description: The binary format serializes `size_t`, enum values, and raw floats with native size, alignment, and endianness. There is no magic value, version, checksum, or bounds validation.
- Evidence: `*((size_t*)(raw+4*index))=fullSize` and enum casts are used directly into the byte buffer.
- Impact: Files may not be portable across 32/64-bit or endian differences, and corrupted files are hard to diagnose safely.
- Suggested fix: Define an explicit wire format with fixed-width integer types, version/magic, endian policy, payload length, and checksum.
- Removal risk: Medium; existing `.bgrid` files require migration or backward-compatible reader.
- Confidence: High.

## Security Review

### S-1: Version check performs unauthenticated network access during server startup

- Severity: Medium
- File/location: `src/server.cpp:140-169`, `build/Makefile:111-119`
- Description: When built with `WITH_VERSION_CONTROL`, startup fetches `GIT_URL/raw/master/version` using libcurl without TLS policy, timeout, or opt-in runtime control.
- Evidence: `curl_easy_setopt(curl, CURLOPT_URL, url); curl_easy_perform(curl);`.
- Impact: Startup can hang or leak network metadata; version messages can be spoofed if the remote URL is not HTTPS or if the configured origin is unexpected.
- Suggested fix: Make update checks opt-in, set timeouts, require HTTPS, validate host, and ignore failures quietly.
- Removal risk: Low.
- Confidence: Medium.

### S-2: Build and packaging download unpinned third-party code

- Severity: Medium
- File/location: `build/Makefile:66-95`, `build/python-build/Makefile:15-26`, `.github/workflows/ccpp.yml:28-36`
- Description: The build downloads `zmq.hpp` from the `master` branch and the Python build clones JsonCpp from its default branch. CI does the same.
- Evidence: `ZMQ_HPP_URL := https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp`, `git clone https://github.com/open-source-parsers/jsoncpp.git`.
- Impact: Builds are non-reproducible and can break or ingest compromised upstream changes.
- Suggested fix: Pin dependency versions or commit hashes, verify checksums, and prefer package-manager dependencies or vendored reviewed snapshots.
- Removal risk: Low to medium depending on release process.
- Confidence: High.

### S-3: `DataImage::write` can create symlinks from unsanitized names

- Severity: Medium
- File/location: `include/DataImage.hpp:202-214`, `src/DataImage.cpp:25-31`
- Description: The requested `filename` is interpolated into `/tmp/G2S/data/%s.bgrid...` and passed as the symlink path. Path separators are not rejected.
- Evidence: `snprintf(fullFilename,2048,"/tmp/G2S/data/%s.bgrid%s",filename.c_str(),extra); createLink(outputFullFilename, fullFilename);`.
- Impact: A local API caller can attempt path traversal or create unexpected symlinks where parent directories exist. In multi-user `/tmp/G2S`, this compounds tampering risk.
- Suggested fix: Restrict logical data names to a safe pattern such as `[A-Za-z0-9_.-]+`, or require explicit paths outside the shared data store and use safe file APIs.
- Removal risk: Low.
- Confidence: Medium.

## Performance Review

### P-1: Job launcher can start too many concurrent jobs

- Severity: Medium
- File/location: `src/jobTasking.cpp:293-317`, `src/server.cpp:104-119`
- Description: The queue runner checks `if(jobIds.look4pid.size()>maxNumberOfJob)` before launching more work. This allows one more than the configured maximum, and a negative `atoi` value assigned to `unsigned` can become a huge limit.
- Evidence: `maxNumberOfConcurrentJob=atoi(argv[i+1])`; later `if(jobIds.look4pid.size()>maxNumberOfJob) return false`.
- Impact: Production servers can oversubscribe CPU/memory and degrade or fail under load.
- Suggested fix: Validate `maxCJ >= 1`, use `>=`, and consider per-user/job memory and process limits.
- Removal risk: Low.
- Confidence: High.

### P-2: `sendData`/`sendJson` load entire payloads into memory

- Severity: Medium
- File/location: `src/dataManagement.cpp:79-107`, `src/dataManagement.cpp:165-193`
- Description: Downloads allocate the whole data/json file and copy it into one ZeroMQ message.
- Evidence: `buffer = malloc(fullSize)` followed by `zmq::message_t reply(fullSize)`.
- Impact: Large simulations can double memory usage during download and cause latency spikes or OOM.
- Suggested fix: Define size limits, stream/chunk large payloads, or use memory-mapped/zero-copy APIs where appropriate.
- Removal risk: Medium; protocol changes need client updates.
- Confidence: High.

### P-3: Random seedconversion loses entropy and emits warnings

- Severity: Low
- File/location: `include/quantileSamplingModule.hpp:455-456`, `include/quantileSamplingModule.hpp:524-525`
- Description: `UINT_MAX` is converted to `float`, and the compiler warns that `4294967295` changes to `4294967296`.
- Evidence: Build warning from `make c++ -j2` on lines 456 and 525.
- Impact: Sampling reproducibility and distribution can be subtly affected.
- Suggested fix: Use integer seed construction, `std::seed_seq`, or `uint32_t(std::numeric_limits<uint32_t>::max() * double(seed))` with explicit bounds.
- Removal risk: Low.
- Confidence: High.

## Testing Review

### T-1: Existing C++ test is narrow and mostly print-based

- Severity: Medium
- File/location: `src/test.cpp:189-459`, `.github/workflows/ccpp.yml:43-45`
- Description: CI runs only `./build/c++-build/test -sampling 1D 2D 3D`. The test checks one sampling property and prints comparisons, but does not cover the server protocol, malformed inputs, CLI parsing, bindings, packaging, distributed launchers, SNESIM, AutoQS edge cases, or security-sensitive paths.
- Evidence: `src/test.cpp` implements the `-sampling` flow and returns `allRight`; CI invokes only that command.
- Impact: Critical server and interface regressions can ship while CI stays green.
- Suggested fix: Add unit tests for serialization validation, job manager state transitions, protocol framing, typed outputs, Python wheel import plus real calls, R/MATLABconversion where feasible, distributed launcher dry-runs, and negative security tests.
- Removal risk: Low.
- Confidence: High.

### T-2: Python package CI only tests import

- Severity: Medium
- File/location: `.github/workflows/pythonPublish.yml:60-71`, `.github/workflows/pythonPublishTest.yml:68-77`
- Description: Wheel jobs install the built wheel and run only `import g2s`.
- Evidence: `python -c "import g2s; print('g2s import OK')"`.
- Impact: Broken runtime calls, typed outputconversion, missing DLLs, or server communication failures can pass release CI.
- Suggested fix: Add a minimal server-backed or mocked protocol test for `g2s.run('--version')`, matrix upload/download conversion, and UInteger outputs.
- Removal risk: Low.
- Confidence: High.

### T-3: No sanitizer, static-analysis, or script lint coverage

- Severity: Medium
- File/location: `.github/workflows/ccpp.yml`, build scripts under `build/`
- Description: There is no AddressSanitizer/UndefinedBehaviorSanitizer build, no `clang-tidy`, no shellcheck, and no Python lint/type check for distributed scripts.
- Evidence: Workflow contains dependency install, build, and one unit-test command only.
- Impact: The current buffer overflows, out-of-bounds reads, and shell-script robustness issues are unlikely to be caught automatically.
- Suggested fix: Add scheduled or PR-gated sanitizer jobs and lightweight lint jobs for C++/Python/shell.
- Removal risk: Low.
- Confidence: High.

## Dead Code & Unused Features Review

### U-1: DirectMeasure and FullMeasure devices appear unused

- Severity: Medium
- File/location: `src/DirectMeasureCPUThreadDevice.cpp`, `include/DirectMeasureCPUThreadDevice.hpp`, `src/FullMeasureCPUThreadDevice.cpp`, `include/FullMeasureCPUThreadDevice.hpp`, `build/c++-build/Makefile:24-68`
- Description: These device classes are compiled into dependency files but not linked into default targets, and `rg` found no constructor use outside their own definitions.
- Evidence: Default targets link `CPUThreadDevice`, `OpenCLGPUDevice`, and `AcceleratorDevice`; no target includes `DirectMeasureCPUThreadDevice.o` or `FullMeasureCPUThreadDevice.o`.
- Impact: Maintenance burden remains for code that is not exercised. DirectMeasure also contains correctness risks if later re-enabled.
- Suggested fix: Confirm whether these are planned features. If not, remove source/header files and references. If yes, wire them behind tests and build targets.
- Removal risk: Needs verification. External consumers may include these headers even if this repository does not.
- Confidence: Medium.

### U-2: `dsk` target is defined but not built, installed, or advertised

- Severity: Low
- File/location: `build/c++-build/Makefile:24`, `build/c++-build/Makefile:64-65`, `build/algosName.config:1-24`
- Description: `dsk` has a build rule but is absent from the default `all` target, install target, and algorithm config.
- Evidence: `all:` lists `ds-l` but not `dsk`; `algosName.config` has `ds-l`/`DirectSamplingLike` but no `dsk`.
- Impact: The feature is partially implemented and likely untested/stale.
- Suggested fix: Remove `dsk` or fully wire it into build/install/config/docs/tests.
- Removal risk: Needs verification with users who may build `make dsk` manually.
- Confidence: Medium.

### U-3: Generated Python build tree is ignored but can become stale between builds

- Severity: Low
- File/location: `build/python-build/Makefile:28-35`, ignored paths `build/python-build/src`, `build/python-build/include`, `build/python-build/jsoncpp`
- Description: The Python build preprocess step copies repository source/header trees into ignored local directories. Existing ignored copies can contain stale code until `preprocess` is rerun.
- Evidence: `preprocess` runs `cp -r ../../src .` and similar commands. Current ignored `build/python-build/src/DirectMeasureCPUThreadDevice.cpp` still showed older syntax errors before regeneration.
- Impact: Developers can inspect or package stale generated source by mistake, causing confusing build/debug behavior.
- Suggested fix: Clean generated directories before copying, make the generated tree clearly ephemeral, and avoid reviewing ignored generated copies as source of truth.
- Removal risk: Low.
- Confidence: High.

### U-4: Stale TODOs and incomplete docs remain

- Severity: Low
- File/location: `docs/installation/interfaces.md:112`, `docs/algorithms/QuickSampling.md:180`, `src/qs.cpp:780`, `src/nds.cpp:369`, `src/dsk.cpp:458`, `src/NvidiaGPUAcceleratorDevice.cu:638-647`
- Description: Several TODO markers remain in user-facing docs and production source.
- Evidence: `rg TODO` found incomplete documentation and implementation notes.
- Impact: Users encounter unfinished docs, and maintainers lack a tracked decision on whether code paths are intentionally incomplete.
- Suggested fix: Convert real work to issues, remove obsolete TODOs, and finish or delete incomplete docs.
- Removal risk: Low.
- Confidence: High.

## Maintainability & Architecture

### M-1: Custom protocol lacks schema and centralized validation

- Severity: High
- File/location: `include/protocol.hpp`, `src/server.cpp`, `include_interfaces/interfaceTemplate.hpp`, `src/dataManagement.cpp`
- Description: The protocol is a raw binary header plus task-specific payloads, but validation is scattered or absent.
- Evidence: Each switch case manually casts/copies bytes. There is no request parser returning validated typed objects.
- Impact: Every new task risks repeating out-of-bounds reads and reply-state bugs.
- Suggested fix: Introduce a single protocol parser/serializer with explicit sizes, max lengths, and error replies. Add fuzz tests for request parsing.
- Removal risk: Medium; needs coordinated server/client changes.
- Confidence: High.

### M-2: C-style memory management and raw ownership are widespread

- Severity: Medium
- File/location: `include/DataImage.hpp`, `src/*Device.cpp`, `include/simulation.hpp`, `include/anchorSimulation.hpp`
- Description: The code mixes `malloc/free`, `new/delete`, raw pointers, and manual ownership across algorithm paths.
- Evidence: Many source files allocate arrays manually; `DataImage::operator=(DataImage&&)` overwrites `_data` without freeing existing storage.
- Impact: Leaks, double frees, and exception-safety issues are hard to prevent, especially under error paths.
- Suggested fix: Incrementally move owned memory to `std::vector`, `std::unique_ptr`, and RAII wrappers for FFT/OpenCL resources. Delete or implement copy semantics explicitly.
- Removal risk: Medium due to performance-sensitive code; convert hot paths carefully.
- Confidence: High.

### M-3: Build configuration suppresses useful warnings

- Severity: Medium
- File/location: `build/Makefile:6-8`
- Description: The default flags include `-Wno-unused`, `-Wno-deprecated`, and `-Wno-deprecated-declarations`.
- Evidence: `CFLAGS`/`CXXFLAGS` suppress these warning classes globally.
- Impact: Dead code, stale APIs, and unused variables are harder to detect. This conflicts with the need to identify unused features.
- Suggested fix: Re-enable warnings in CI, optionally keep suppressions only for third-party headers or known legacy files.
- Removal risk: Medium; enabling warnings may produce a large cleanup backlog.
- Confidence: High.

### M-4: Cross-platform packaging assumptions are brittle

- Severity: Medium
- File/location: `build/python-build/setup.py:166-170`, `build/python-build/setup.py:172-197`, `build/python-build/g2s/__init__.py:15-24`
- Description: macOS wheels force `-arch arm64`, Windows DLL handling depends on local relative paths, and `data_files` is computed from a class attribute that is not populated at setup declaration time.
- Evidence: `ext.extra_compile_args += ["-arch", "arm64"]`; `getattr(build_ext, "copy_dlls", [])` is evaluated in `setup(...)`.
- Impact: Intel macOS users and Windows wheel consumers can receive unusable packages. Current CI import-only tests may miss runtime DLL resolution issues.
- Suggested fix: Build universal2 or architecture-matrix macOS wheels, package DLLs as package data, and test an actual `g2s.run` call on each platform.
- Removal risk: Medium.
- Confidence: Medium.

## Recommended Fix Priority

1. Critical security/stability: C-1, C-2, C-3, C-4.
2. Protocol and data integrity: B-1, B-2, D-1, D-2, D-3, M-1.
3. User-visible correctness: B-3, B-4, B-5, B-7, B-8.
4. Test infrastructure: T-1, T-2, T-3.
5. Packaging and maintainability: S-1, S-2, S-3, P-1, P-2, P-3, M-2, M-3, M-4.
6. Dead/stale code cleanup: U-1, U-2, U-3, U-4.

## Appendix / Manual Verification Needed

- Confirm intended server threat model. If remote unauthenticated use is intentionally supported for clusters, document it clearly and add compensating controls.
- Confirm whether external extensions or downstream users include `DirectMeasureCPUThreadDevice`, `FullMeasureCPUThreadDevice`, `CPUThreadAcceleratorDevice`, or `dsk`.
- Run sanitizer builds: ASan/UBSan for C++ algorithms and server protocol handlers.
- Add malformed protocol tests for every `taskType`.
- Test Python, R, and MATLAB typed outputs for `UInteger` result images.
- Test Python wheel runtime behavior on Windows and Intel macOS, not only import.
- Verify distributed QS on a real shared filesystem and Slurm cluster.
- Audit ignored generated artifacts before releases; ensure source distributions are regenerated from current tracked sources.
