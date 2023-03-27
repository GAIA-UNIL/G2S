ifeq ($(UNAME_S),Darwin)
	ifeq ($(brewIsAvailable),1)
    	export CFLAGS+= -Xpreprocessor -fopenmp  -I$(shell  brew --prefix libomp)/include
    	export LDFLAGS += -lomp -L$(shell brew --prefix libomp)/lib
	endif
endif

doseSupportCPP17:= $(shell echo "int main(){}" | $(CXX) -xc++ - -o /dev/null -std=c++17 2>/dev/null && echo 1 || echo 0)
ifeq ($(doseSupportCPP17),1)
	export CFLAGS += -std=c++17
	export NVFLAGS += -std=c++17
endif
doseNeedFS_Lib:= $(shell echo "int main(){}" | $(CXX) -xc++ - -o /dev/null -lstdc++fs 2>/dev/null && echo 1 || echo 0)
ifeq ($(doseNeedFS_Lib),1)
	export LDFLAGS += -lstdc++fs
endif


export DEST_DIR_EXTENSION:=$(shell pwd)

.PHONY: all depend algoNames extension clean

all: server echo qs nds ds-l algoNames libg2s.a errorTest test auto_qs $(WITH_CUDA)

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cu 
	$(NVCC) -ccbin=$(CXX) -c -o $@ $< $(NVFLAGS) $(foreach option, $(CFLAGS) $(CXXFLAGS), --compiler-options $(option))  $(INC) $(LIBINC)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(OPENCL_INC)

server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

g2s_cuda.so: NvidiaGPUAcceleratorDevice.o
	$(CXX) -shared -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(NVLIB)

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

auto_qs: auto_qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

dsk: dsk.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

errorTest: errorTest.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

test: test.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
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