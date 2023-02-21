export CC=icc
export CXX=icpc
export AR=xiar
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS =linux
endif
ifeq ($(UNAME_S),Darwin)
    OS =mac
endif

ifeq ($(ICPX),1)
	export CC=icx
	export CXX=icpx
	export AR=ar
	CXXFLAGS+= -fp-model=precise
endif

export DEST_DIR_EXTENSION:=$(shell pwd)

ifeq ($(MKLROOT),)
	MKLROOT := $(/opt/intel/compilers_and_libraries/$(OS)/mkl)
endif

CXXFLAGS+= -axSSE4.2 -axAVX -axAVX2 -axcommon-avx512 -fopenmp -DXEON -DWITH_MKL -I${MKLROOT}/include/fftw -qmkl 
LDFLAGS+=-fopenmp -static-intel -qopenmp-link=static  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
LDFLAGS:=$(subst -lfftw3f,,$(subst -lfftw3f_threads,,$(LDFLAGS)))

.PHONY: all depend algoNames extension clean

all: g2s_server g2s_echo g2s_qs g2s_nds g2s_ds-l g2s_algoNames libg2s.a g2s_errorTest g2s_auto_qs

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cu 
	$(NVCC) -ccbin=$(CXX) -c -o $@ $< $(NVFLAGS) $(foreach option, $(CFLAGS) $(CXXFLAGS), --compiler-options $(option))  $(INC) $(LIBINC) -allow-unsupported-compiler

%.o: $(SRCS)/%.cpp 
	$(CXX) $(CFLAGS) $(CXXFLAGS) -c -o $@ $<  $(INC) $(LIBINC) $(OPENCL_INC)

g2s_server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)

g2s_echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

g2s_qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

g2s_auto_qs: auto_qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

g2s_nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

g2s_ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

g2s_errorTest: errorTest.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

libg2s.a: DataImage.o
	$(AR) $(ARFLAGS) $@ $^

g2s_algoNames:
	ln -sf ../g2s_algosName.config g2s_algosName.config

extension:libg2s.a algoNames
	$(MAKE) -C ../extension clean
	$(MAKE) -C ../extension
	$(MAKE) -C ../extension install

clean:
	rm -rf *.o *.so *.a
	rm -rf g2s_server g2s_echo g2s_test g2s_qs g2s_nds g2s_ds g2s_ds-l eg2s_rrorTest g2s_auto_qs
	rm -rf g2s_algosName.config
	rm -rf .depend