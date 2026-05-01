# Changes

## 2026-05-01

- Restricted server job execution to algorithm names registered in `algosName.config` by default.
- Added the `--allow-unregistered-algorithms` server flag for deployments that intentionally need the legacy `./<Algorithm>` fallback.
- Hardened server `KILL` handling so unknown or malformed job ids are rejected instead of risking invalid queue access or process-group signalling.
- Replaced fixed-size stack buffers in remote job argv construction with owned strings and explicit request, algorithm, argument, and argv-count limits.
- Hardened data upload/download handling: request frames now validate hash and job-id lengths, `.bgrid` files must match their actual payload size, serialized dimensions/data sizes are bounded, and truncated or malformed files are rejected before server send or client deserialization.
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
