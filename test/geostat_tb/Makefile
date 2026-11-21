# Tests of the turning-bands engine (include/geostat/turningBands.hpp) and of tb.
#   make            build and run (tb is tested when build/c++-build/tb exists)
#   make TB=/path/to/tb
#   make timings    also time tb on 1000 x 1000 and 200 x 200 x 200 grids
# Images are written by DataImage to /tmp/G2S/data (names tb_test_*, im_1_9877xx).
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/tb_test
TB ?= $(REPO_ROOT)/build/c++-build/tb
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test timings clean

all: test

test: $(OUT)
	$(OUT) $(TB)

timings: $(OUT)
	$(OUT) $(TB) timings

$(OUT): tb_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) tb_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@ $(EXTRA_LIB) -lz -lpthread

clean:
	rm -rf $(CURDIR)/out
