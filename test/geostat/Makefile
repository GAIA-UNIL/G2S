# Standalone tests of the geostat foundation (include/geostat) and of covEval.
#   make            build and run (covEval is tested when build/c++-build/covEval exists)
#   make COVEVAL=/path/to/covEval
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/geostat_test
COVEVAL ?= $(REPO_ROOT)/build/c++-build/covEval
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test clean

all: test

test: $(OUT)
	$(OUT) $(CURDIR)/cov_cases.txt $(COVEVAL)

$(OUT): geostat_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) geostat_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@ $(EXTRA_LIB) -lz -lpthread

clean:
	rm -rf $(CURDIR)/out
