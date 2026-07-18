REPO_ROOT := $(abspath ../..)
EMSDK_VERSION := 6.0.3
BUILD_DIR := $(CURDIR)/out
FFTW_PREFIX := $(BUILD_DIR)/fftw
DIST_DIR := $(REPO_ROOT)/browser/dist
FFTW_SOURCE := $(CURDIR)/cache/fftw-3.3.11

EMCC ?= emcc
EMXX ?= em++
EMCONFIGURE ?= emconfigure
EMMAKE ?= emmake

SOURCES := \
	$(REPO_ROOT)/src/qs.cpp \
	$(REPO_ROOT)/src/qsCore.cpp \
	$(REPO_ROOT)/src/qsWasmBindings.cpp \
	$(REPO_ROOT)/src/DataImage.cpp \
	$(REPO_ROOT)/src/sharedMemoryManager.cpp \
	$(REPO_ROOT)/src/CPUThreadDevice.cpp \
	$(REPO_ROOT)/src/AcceleratorDevice.cpp

CPPFLAGS := -I$(REPO_ROOT)/include -I$(FFTW_PREFIX)/include \
	-DG2S_BROWSER_BUILD=1 -DG2S_QS_NO_MAIN=1 -DG2S_PATH_INDEX_TYPE=unsigned
CXXFLAGS := -O3 -std=c++17 -Wno-vla-extension -Wno-unused-command-line-argument
EMFLAGS := -sUSE_ZLIB=1 -sFILESYSTEM=1 -sALLOW_MEMORY_GROWTH=1 \
	-sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createG2SModule \
	-sENVIRONMENT=worker -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 -sASSERTIONS=1 \
	-lembind

.PHONY: all clean check-toolchain

all: check-toolchain $(DIST_DIR)/g2s-qs.mjs

check-toolchain:
	@command -v $(EMCC) >/dev/null || (echo "emcc is required; activate the pinned emsdk toolchain first" >&2; exit 1)
	@command -v $(EMXX) >/dev/null || (echo "em++ is required; activate the pinned emsdk toolchain first" >&2; exit 1)
	@$(EMCC) --version | head -1 | grep -q " $(EMSDK_VERSION) " || (echo "emsdk $(EMSDK_VERSION) is required" >&2; exit 1)

$(FFTW_SOURCE)/configure: fetch_fftw.sh
	@sh ./fetch_fftw.sh >/dev/null

$(FFTW_PREFIX)/lib/libfftw3f.a: $(FFTW_SOURCE)/configure | check-toolchain
	@mkdir -p $(BUILD_DIR) $(FFTW_PREFIX)
	cd $(FFTW_SOURCE) && $(EMCONFIGURE) ./configure \
		--prefix=$(FFTW_PREFIX) --enable-float --disable-shared --enable-static \
		--disable-fortran --disable-threads --disable-openmp --disable-doc
	$(EMMAKE) $(MAKE) -C $(FFTW_SOURCE) -j2
	$(EMMAKE) $(MAKE) -C $(FFTW_SOURCE) install

$(DIST_DIR)/g2s-qs.mjs: $(SOURCES) $(FFTW_PREFIX)/lib/libfftw3f.a
	@mkdir -p $(DIST_DIR)
	$(EMXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) $(FFTW_PREFIX)/lib/libfftw3f.a \
		-o $@ $(EMFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)/g2s-qs.mjs $(DIST_DIR)/g2s-qs.wasm
