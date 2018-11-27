






#OPENCL_LIB= -framework OpenCL -lclFFT
#OPENCL_LIB= -lOpenCL -lclFFT
#OPENCL_INC=-DWITH_OPENCL=1


all: server echo qs nds ds ds-l algoNames

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2sRExtension.cpp,,$(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(OPENCL_INC)

server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

ds: ds.o DataImage.o sharedMemoryManager.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)
	mkdir -p ../3party_bin/ && which deesseOMP 2>/dev/null | xargs -I{} cp -f {}  ../3party_bin/
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS)

dsk: dsk.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o OpenCLGPUDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)
	$(CXX) -shared -o $@.so $^ $(LIB_PATH) $(LDFLAGS) $(OPENCL_LIB)

algoNames:
	ln -sf ../algosName.config algosName.config

clean:
	rm -rf *.o *.so
	rm -rf server echo test qs nds ds ds-l
	rm -rf algosName.config
	rm -rf .depend