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

export DEST_DIR_EXTENSION:=$(shell pwd)

ifeq ($(MKLROOT),)
	MKLROOT := $(/opt/intel/compilers_and_libraries/$(OS)/mkl)
endif

CXXFLAGS+= -xhost -restrict -fopenmp -DXEON -DWITH_MKL -I${MKLROOT}/include/fftw -qmkl
LDFLAGS+=-fopenmp -qmkl -lirc -Wl,-rpath,${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib/
LDFLAGS:=$(subst -lfftw3f,,$(subst -lfftw3f_threads,,$(LDFLAGS)))

.PHONY: all depend algoNames extension clean

all: server echo qs nds ds-l algoNames libg2s.a errorTest auto_qs snesim

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cu 
	$(NVCC) -ccbin=$(CXX) -c -o $@ $< $(NVFLAGS) $(foreach option, $(CFLAGS) $(CXXFLAGS), --compiler-options $(option))  $(INC) $(LIBINC) -allow-unsupported-compiler

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(OPENCL_INC)

server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

auto_qs: auto_qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

snesim: snesim.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB) $(NVLIB)

errorTest: errorTest.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

libg2s.a: DataImage.o
	$(AR) $(ARFLAGS) $@ $^

algoNames:
	ln -sf ../algosName.config algosName.config

extension:libg2s.a algoNames
	$(MAKE) -C ../extension clean
	$(MAKE) -C ../extension
	$(MAKE) -C ../extension install

clean:
	rm -rf *.o *.so *.a
	rm -rf server echo test qs nds ds ds-l errorTest
	rm -rf algosName.config
	rm -rf .depend