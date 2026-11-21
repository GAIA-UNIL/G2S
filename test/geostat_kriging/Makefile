# Tests of the kriging engines (include/geostat/krigingGlobal.hpp, krigingLocal.hpp,
# kdTreeSearch.hpp, denseSolver.hpp) and of the kriging executable.
#   make            build and run (the executable is tested when build/c++-build/kriging exists)
#   make KRIGING=/path/to/kriging
REPO_ROOT := $(abspath ../..)
CXX ?= c++
OUT := $(CURDIR)/out/kriging_test
KRIGING ?= $(REPO_ROOT)/build/c++-build/kriging
BREW_PREFIX := $(shell brew --prefix 2>/dev/null)
EXTRA_INC := $(if $(BREW_PREFIX),-I$(BREW_PREFIX)/include) -I/usr/include/jsoncpp
EXTRA_LIB := $(if $(BREW_PREFIX),-L$(BREW_PREFIX)/lib)

.PHONY: all test clean

all: test

test: $(OUT)
	$(OUT) $(KRIGING)

$(OUT): kriging_test.cpp $(wildcard $(REPO_ROOT)/include/geostat/*.hpp) $(REPO_ROOT)/src/DataImage.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -Wall -Wno-unused -I$(REPO_ROOT)/include $(EXTRA_INC) kriging_test.cpp $(REPO_ROOT)/src/DataImage.cpp -o $@ $(EXTRA_LIB) -lz -lpthread

clean:
	rm -rf $(CURDIR)/out
