# -----------------------------------------------------------------------------
# G2S Python interface sub-Makefile
# -----------------------------------------------------------------------------

PYTHON   ?= python3
VERSION_FILE := ../../version

# Default target
all: build

# Ensure local directory tree exists
dirs:
	@mkdir -p $(LOCAL_SRC_DIRS)

include/zmq.hpp:
	mkdir -p include
	wget "https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp" -O include/zmq.hpp

jsoncpp:
	git clone https://github.com/open-source-parsers/jsoncpp.git
	cd jsoncpp && python3 amalgamate.py

preprocess: jsoncpp
	cp -r ../../src .
	cp -r ../../include .
	cp -r ../../src_interfaces .
	cp -r ../../include_interfaces .
	cp -r jsoncpp/dist/json include
	cp jsoncpp/dist/jsoncpp.cpp src/
	@echo "✅ Preprocessing complete — local tree reproduced."

# Version file for Python
version:
	echo "__version__ = \"$(shell cat $(VERSION_FILE))\"" > g2s/_version.py

# Build wheels and sdist
build:preprocess version include/zmq.hpp
	$(PYTHON) -m build

wheel:preprocess build
	@echo "✅ Wheel built."

sdist: version preprocess include/zmq.hpp
	$(PYTHON) -m build --sdist
	@echo "✅ Source distribution created."

# Clean
clean:
	rm -rf build dist *.egg-info src include src_interfaces include_interfaces jsoncpp g2s/_version.py
	@echo "🧹 Cleaned all generated files."

.PHONY: all preprocess build wheel sdist clean version dirs
