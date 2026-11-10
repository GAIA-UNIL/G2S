# G2S Package description

This package contains the interfaces for communicating with the G2S server. G2S is a flexible MPS (Multiple Point Statistics) framework that provides QS (Quantile Sampling) and NDS (Narrow Distribution Selection).

For more information, please visit the [G2S website](https://gaia-unil.github.io/G2S/).

Version 1.0.1 supports simultaneous `g2s.run` calls from Python threads, including
free-threaded Python 3.14 (`python3.14t`). Each call owns its ZeroMQ connection and
releases the interpreter thread during network I/O and server waits. Use the
wheel matching your interpreter's ABI, and keep shared NumPy inputs unchanged
until all calls finish. The server schedules the submitted jobs.
