# Tests of the FFT simulation engine (include/geostat/spectralEngine.hpp) and
# of the fftma / spectral executable.
#   make              build and run (the executable is tested when build/c++-build/fftma exists)
#   make FFTMA=/path/to/fftma
#   make bench [THREADS=8]   timings for 1000x1000 and 200x200x200, R = 1 and 16
#   make FFTW=system  link the system / Homebrew FFTW even when the local SIMD build exists
#                     (the engine checks hold for any FFTW; the executable keeps its own)
REPO_ROOT := $(abspath ../..)
CXX ?= c++
FFTW ?= auto
OUT := $(CURDIR)/out/geostat_fft_test$(if $(filter system,$(FFTW)),_system)
FFTMA ?= $(REPO_ROOT)/build/c++-build/fftma
THREADS ?= 8
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
# Same FFTW as the executables: the local SIMD build (make fftw-simd) first when present, as build/Makefile does;
# the executable-versus-engine checks are bit for bit.
FFTW_SIMD_LIB := $(if $(filter system,$(FFTW)),,$(wildcard $(REPO_ROOT)/build/fftw-simd/install/lib/libfftw3f.a))
EXTRA_LIB := $(if $(FFTW_SIMD_LIB),-L$(dir $(FFTW_SIMD_LIB))) $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test bench clean

all: test

test: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(FFTMA)

bench: $(OUT)
	cd $(CURDIR)/out && $(OUT) bench $(THREADS)

$(OUT): geostat_fft_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/test/geostat_common/testData.hpp $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) geostat_fft_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@.$$$$ $(EXTRA_LIB) -lfftw3f_threads -lfftw3f -lfftw3_threads -lfftw3 -lz -lpthread && mv -f $@.$$$$ $@

clean:
	rm -rf $(CURDIR)/out
