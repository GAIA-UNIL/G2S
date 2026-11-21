# Tests of the SGS engine (include/geostat/sgsEngine.hpp, weightTable.hpp,
# gridAddressing.hpp) and of the sgs executable.
#   make            build and run (the executable is tested when build/c++-build/sgs exists)
#   make SGS=/path/to/sgs
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/sgs_test
SGS ?= $(REPO_ROOT)/build/c++-build/sgs
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test clean

all: test

test: $(OUT)
	cd $(CURDIR)/out && $(OUT) $(SGS)

$(OUT): sgs_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) sgs_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@ $(EXTRA_LIB) -lz -lpthread

clean:
	rm -rf $(CURDIR)/out
