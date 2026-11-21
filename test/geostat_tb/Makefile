# Tests of the turning-bands engine (include/geostat/turningBands.hpp) and of tb.
#   make            build and run (tb is tested when build/c++-build/tb exists)
#   make TB=/path/to/tb
#   make timings    also time tb on 1000 x 1000 and 200 x 200 x 200 grids
#   make quick      a subset of the statistical cases (every structure type, fewer
#                   dimension / anisotropy combinations), for CI
#   make SIMD_FLAGS="-mavx2 -mfma"   the engine built for another instruction set
#                   (x86: default SSE2, -msse4.1, -mavx2 -mfma, -mavx512f; see
#                   include/geostat/simdLanes.hpp); `make clean` first
# Images are written to /tmp/G2S/data under names unique to the run, removed at exit
# (test/geostat_common/testData.hpp).
REPO_ROOT := $(abspath ../..)
CXX ?= c++
SIMD_FLAGS ?=
OUT := $(CURDIR)/out/tb_test
TB ?= $(REPO_ROOT)/build/c++-build/tb
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test timings quick clean

all: test

test: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(TB)

timings: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(TB) timings

quick: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(TB) quick

$(OUT): tb_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/test/geostat_common/testData.hpp $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused $(SIMD_FLAGS) -I$(REPO_ROOT)/include $(EXTRA_INC) tb_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@.$$$$ $(EXTRA_LIB) -lz -lpthread && mv -f $@.$$$$ $@

clean:
	rm -rf $(CURDIR)/out
