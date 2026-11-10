#!/usr/bin/env python3
"""Compiler/linker flags for embedding Python in the pyDev algorithm.

Usage: <python> pydev_config.py --cflags | --ldflags | --check

Run it with the interpreter pyDev should embed (Python >= 3.14, with numpy).
A free-threaded build (python3.14t) lets sample() run on several threads at once.
If that interpreter lives in a virtual environment, the venv is also used at run time.
"""

import glob
import os
import sys
import sysconfig


def fail(message):
    sys.stderr.write("pyDev: " + message + "\n")
    sys.exit(1)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if sys.version_info < (3, 14):
        fail(f"Python >= 3.14 is required, {sys.executable} is {sys.version.split()[0]} "
             "(set PYDEV_PYTHON=/path/to/python3.14 or python3.14t)")
    try:
        import numpy
    except ImportError:
        fail(f"numpy is not installed for {sys.executable}")

    config = sysconfig.get_config_var
    if mode == "--cflags":
        print(f"-I{sysconfig.get_path('include')} -I{numpy.get_include()} "
              f"-DG2S_PYDEV_DEFAULT_PYTHON=\\\"{sys.executable}\\\"")
    elif mode == "--ldflags":
        ldversion = config("LDVERSION")
        library_dir = None
        for candidate in (config("LIBDIR"), config("LIBPL")):
            if candidate and glob.glob(os.path.join(candidate, f"libpython{ldversion}.*")):
                library_dir = candidate
                break
        if library_dir is None:
            fail(f"libpython{ldversion} not found (Python must be built with --enable-shared or provide libpython)")
        flags = [f"-L{library_dir}", f"-lpython{ldversion}"]
        flags += (config("LIBS") or "").split() + (config("SYSLIBS") or "").split()
        if config("Py_ENABLE_SHARED"):
            flags.append(f"-Wl,-rpath,{library_dir}")
        print(" ".join(flags))
    elif mode == "--check":
        gil = "free-threaded" if config("Py_GIL_DISABLED") else "with GIL"
        print(f"{sys.executable}: Python {sys.version.split()[0]} ({gil}), numpy {numpy.__version__}")
    else:
        fail("unknown option " + mode)


if __name__ == "__main__":
    main()
