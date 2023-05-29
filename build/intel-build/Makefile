export CC=icc
export CXX=icpc
export AR=xiar

ifeq ($(MKLROOT),)
	MKLROOT := $(/opt/intel/compilers_and_libraries/$(OS)/mkl)
endif

MKL_STAT_PATH= ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_intel_thread.a ${MKLROOT}/lib/libmkl_core.a


ifneq ($(shell icpx 2>/dev/null; echo $$? ),127)
	export CC=icx
	export CXX=icpx
	export AR=ar
	CXX_FLAGS= -fp-model=precise
	MKL_STAT_PATH=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
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


CXX_FLAGS+= -xhost -fopenmp -DXEON -DWITH_MKL -I${MKLROOT}/include/fftw -qmkl 
LDFLAGS+=-fopenmp -lpthread -lm -ldl

LDFLAGS_NO_SERVER_ONLY:=$(subst -lfftw3f,,$(subst -lfftw3f_threads,-qmkl,$(LDFLAGS_NO_SERVER_ONLY)))

ifneq ($(origin AS_STATIC), undefined)
	CXX_FLAGS:=$(subst -xhost,-axSSE4.2 -axAVX -axAVX2 -axcommon-avx512, $(CXX_FLAGS))
	NVLIB+= --compiler-options -static-intel  #$(subst -lcudart,-lcudart_static, $(subst -lcublas,-lcublas_static, $(subst -lcufft,-lcufft_static, $(NVLIB)))) 
	LDFLAGS_NO_SERVER_ONLY:=$(subst -qmkl,-static-intel -qopenmp-link=static $(MKL_STAT_PATH) ,$(LDFLAGS_NO_SERVER_ONLY))
endif

.PHONY: all depend algoNames extension clean

all: g2s_server echo qs nds ds-l algoNames libg2s.a errorTest auto_qs $(WITH_CUDA)

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(CXX_FLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cu 
	$(NVCC) -ccbin=$(CXX) -c -o $@ $< $(NVFLAGS) $(foreach option, $(subst -xhost,,$(CFLAGS) $(CXXFLAGS)), --compiler-options $(option))  $(INC) $(LIBINC) -allow-unsupported-compiler

%.o: $(SRCS)/%.cpp 
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(CXX_FLAGS) -c -o $@ $<  $(INC) $(LIBINC) $(OPENCL_INC)

g2s_server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

g2s_cuda.so: NvidiaGPUAcceleratorDevice.o
	$(NVCC) -ccbin=$(CXX) -shared -o $@ $^ $(NVLIB) --compiler-options -fuse-ld=lld

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

auto_qs: auto_qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

errorTest: errorTest.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

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
	rm -rf g2s_server echo test qs nds ds ds-l errorTest auto_qs
	rm -rf algosName.config
	rm -rf .depend