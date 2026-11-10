#!/bin/sh
# Prints the Python that pyDev should be built against, or fails.
# Used by `make pydev` when PYDEV_PYTHON is not given.
# Preference: free-threaded builds first, newest first; Python >= 3.14 with numpy.
for candidate in python3.16t python3.15t python3.14t python3.16 python3.15 python3.14 python3; do
	path=$(command -v "$candidate" 2>/dev/null) || continue
	if "$path" -c 'import sys, numpy; sys.exit(0 if sys.version_info >= (3, 14) else 1)' >/dev/null 2>&1; then
		echo "$path"
		exit 0
	fi
done
echo "pyDev: no Python >= 3.14 with numpy found on PATH." >&2
echo "pyDev: install one (python3.14t recommended) or run: make pydev PYDEV_PYTHON=/path/to/python3.14t" >&2
exit 1
