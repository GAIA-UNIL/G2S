# Tests of the conditioning of Gaussian simulations by kriging of the residuals
# (include/geostat/conditioning.hpp) and of its use in fftma / spectral and tb,
# with the flags shared by sgs, kriging, fftma and tb.
#   make                 build and run (the executables are tested when build/c++-build has them)
#   make BIN=/path/to/c++-build
#   make timings [THREADS=8]   fftma / spectral / tb on 1000 x 1000 with 1000 and 10000 data, R = 1 and 16
# Images are written to /tmp/G2S/data under names unique to the run, removed at exit
# (test/geostat_common/testData.hpp).
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/conditioning_test
BIN ?= $(REPO_ROOT)/build/c++-build
THREADS ?= 8
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test timings clean

all: test

test: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(BIN)

timings: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(BIN) timings $(THREADS)

$(OUT): conditioning_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/test/geostat_common/testData.hpp $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) conditioning_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@.$$$$ $(EXTRA_LIB) -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lz -lpthread && mv -f $@.$$$$ $@

clean:
	rm -rf $(CURDIR)/out
