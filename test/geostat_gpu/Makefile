# Tests of the GPU back ends (include/geostat/gpu), OpenCL only (the CUDA back
# end needs an Nvidia GPU and is not exercised here).
#   make            build and run; prints "skipped: no OpenCL device" and succeeds
#                   when the build has no OpenCL or the machine no GPU
#   make timings    also the device / CPU timings of the benchmark cases
# Needs OpenCL and clFFT as the executables do: build/Makefile's detection
# (`make clfft` in build/ builds clFFT locally). Executables are taken from
# build/c++-build (override with BIN=...).
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/gpu_test
BIN ?= $(REPO_ROOT)/build/c++-build
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include)
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)
UNAME_S := $(shell uname -s)

# FFTW: the SIMD build of build/fftw-simd when present (as the executables)
FFTW_SIMD_PREFIX := $(REPO_ROOT)/build/fftw-simd/install
ifneq ($(wildcard $(FFTW_SIMD_PREFIX)/lib/libfftw3f.a),)
	EXTRA_INC := -I$(FFTW_SIMD_PREFIX)/include $(EXTRA_INC)
	EXTRA_LIB := -L$(FFTW_SIMD_PREFIX)/lib $(EXTRA_LIB)
endif

# OpenCL and clFFT, detected as build/Makefile does
CLFFT_PREFIX := $(REPO_ROOT)/build/clfft/install
ifeq ($(UNAME_S),Darwin)
	OPENCL_LIB := -framework OpenCL -lclFFT
else
	OPENCL_LIB := -lclFFT -lOpenCL
endif
ifneq ($(wildcard $(CLFFT_PREFIX)/lib/libclFFT.a),)
	OPENCL_LIB := -L$(CLFFT_PREFIX)/lib $(OPENCL_LIB)
	EXTRA_INC := -I$(CLFFT_PREFIX)/include $(EXTRA_INC)
endif
HAS_OPENCL := $(shell echo 'int main(){}' | $(CXX) -xc - -o /dev/null $(OPENCL_LIB) 2>/dev/null && echo 1 || echo 0)
ifeq ($(HAS_OPENCL),1)
	OPENCL_FLAGS := -DWITH_OPENCL=1
else
	OPENCL_LIB :=
endif

SOURCES := gpu_test.cpp $(REPO_ROOT)/src/DataImage.cpp $(REPO_ROOT)/src/geostatOpenCL.cpp $(REPO_ROOT)/src/geostatGpu.cpp

.PHONY: all test timings clean

all: test

test: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(BIN)

timings: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(BIN) timings

$(OUT): $(SOURCES) $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(wildcard $(REPO_ROOT)/include/geostat/gpu/*) $(REPO_ROOT)/test/geostat_common/testData.hpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -Wno-deprecated-declarations $(OPENCL_FLAGS) -I$(REPO_ROOT)/include $(EXTRA_INC) $(SOURCES) -o $@.$$$$ $(EXTRA_LIB) $(OPENCL_LIB) -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lz -lpthread -ldl && mv -f $@.$$$$ $@

clean:
	rm -rf $(CURDIR)/out
