# Changes

## 2026-04-19

- Fixed `build/Makefile` `zmq.hpp` bootstrap to work reliably in CI:
  - resolved `include/zmq.hpp` from the Makefile location (independent of current working directory),
  - removed compile-probe-based detection for `zmq.hpp`,
  - added downloader fallback order: `curl`, then `wget`, then `python`,
  - clarified failure message (`curl/wget/python` or manual `include/zmq.hpp`).
- Updated docs and ignore rules for `zmq.hpp` auto-download temp file handling.
