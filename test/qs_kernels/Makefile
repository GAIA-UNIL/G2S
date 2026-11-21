REPO_ROOT := $(abspath ../..)
CXX ?= c++
SIMD_FLAGS ?=
OUT := $(CURDIR)/out/qs_kernels_test

.PHONY: all test clean

all: test

test: $(OUT)
	$(OUT)

$(OUT): qs_kernels_test.cpp $(REPO_ROOT)/include/complexMulti.hpp $(REPO_ROOT)/include/fKb.hpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 $(SIMD_FLAGS) -I$(REPO_ROOT)/include $< -o $@

clean:
	rm -rf $(CURDIR)/out
