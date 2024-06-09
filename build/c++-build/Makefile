ifeq ($(UNAME_S),Darwin)
	ifeq ($(brewIsAvailable),1)
    	export CFLAGS+= -Xpreprocessor -fopenmp  -I$(shell  brew --prefix libomp)/include
    	export LDFLAGS_NO_SERVER_ONLY += -lomp -L$(shell brew --prefix libomp)/lib
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

all: g2s_server echo qs nds ds-l libg2s.a errorTest test auto_qs ds $(WITH_CUDA)

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cu 
	$(NVCC) -ccbin=$(CXX) -c -o $@ $< $(NVFLAGS) $(foreach option, $(CFLAGS) $(CXXFLAGS), --compiler-options $(option))  $(INC) $(LIBINC)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(OPENCL_INC)

g2s_server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_SERVER_ONLY)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

g2s_cuda.so:  NvidiaGPUAcceleratorDevice.o
	$(NVCC) -ccbin=$(CXX) -shared -o $@ $^ $(NVLIB) 

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

auto_qs: auto_qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

dsk: dsk.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB) 

ds: ds.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB) 

errorTest: errorTest.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

test: test.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o $(ExtraDevice)
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(LDFLAGS_NO_SERVER_ONLY) $(OPENCL_LIB)

libg2s.a: DataImage.o
	$(AR) $(ARFLAGS) $@ $^

algoNames:
	ln -sf ../algosName.config algosName.config

extension:libg2s.a algoNames
	$(MAKE) -C ../extension clean
	$(MAKE) -C ../extension
	$(MAKE) -C ../extension install

create_install_path:
	@printf "\e[32mCreation of $(INSTALL_DIR) if needed \e[0m\n"
	@mkdir -p $(INSTALL_DIR)/bin
	@mkdir -p $(INSTALL_DIR)/libexec
	@echo ${PATH} | tr ":" "\n" | grep -x $(INSTALL_DIR)/bin > /dev/null || printf "\e[31mPlease add $(INSTALL_DIR)/bin to your PATH to propoerly use g2s !! \e[0m\n"

install:create_install_path all
	@chmod +x ../g2s
	@cp -f ../g2s $(INSTALL_DIR)/bin
	@cp -f ./g2s_server $(INSTALL_DIR)/libexec
	@cp -f ./qs $(INSTALL_DIR)/libexec || printf "\e[31error: 'qs' is missing !!\e[0m\n"
	@cp -f ./nds $(INSTALL_DIR)/libexec || printf "\e[31error: 'nds' is missing !!\e[0m\n"
	@cp -f ./ds-l $(INSTALL_DIR)/libexec || printf "\e[31error: 'ds-l' is missing !!\e[0m\n"
	@cp -f ./auto_qs $(INSTALL_DIR)/libexec || printf "\e[31error: 'auto_qs' is missing !!\e[0m\n"
	@cp -f ../algosName.config $(INSTALL_DIR)/libexec
	@cp -f ./g2s_cuda.so $(INSTALL_DIR)/libexec || printf "\e[33mwarning: 'g2s_cuda.so' is missing, this is probably due to the absance of CUDA compiler.\e[0m\n"
	@printf "\e[32mInstallation c++ finished!\e[0m\n"


clean:
	rm -rf *.o *.so *.a
	rm -rf g2s_server echo test qs nds ds ds-l errorTest auto_qs
	rm -rf algosName.config
	rm -rf .depend