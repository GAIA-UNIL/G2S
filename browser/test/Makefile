REPO_ROOT := $(abspath ../..)
CXX ?= c++
JSONCPP_FLAGS := $(shell pkg-config --cflags --libs jsoncpp)
OUT := $(CURDIR)/out/browser_transport_test

.PHONY: all transport clean

all: transport

transport: $(OUT)
	$(OUT)

$(OUT): browser_transport_test.cpp $(REPO_ROOT)/src_interfaces/browserTransport.cpp
	@mkdir -p $(dir $@)
	$(CXX) -std=c++17 -O2 -pthread \
		-I$(REPO_ROOT)/include_interfaces \
		$^ $(JSONCPP_FLAGS) -o $@

clean:
	rm -rf $(CURDIR)/out
