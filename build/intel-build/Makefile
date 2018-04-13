CC=icc
CXX=icpc
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS =linux
endif
ifeq ($(UNAME_S),Darwin)
    OS =mac
endif

CXXFLAGS+= -xhost -restrict -fopenmp -DXEON -DWITH_MKL -I/opt/intel/compilers_and_libraries/$(OS)/mkl/include/fftw -mkl
LDFLAGS+=-fopenmp -mkl -lirc -Wl,-rpath,${MKLROOT}/lib -Wl,-rpath,/opt/intel/compilers_and_libraries/$(OS)/lib/
LDFLAGS:=$(subst -lfftw3f,,$(subst -lfftw3f_threads,,$(LDFLAGS)))

all: server echo qs nds ds ds-l

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/module/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC)

server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

echo: echo.o DataImage.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

qs: qs.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

nds: nds.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

ds: ds.o DataImage.o sharedMemoryManager.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

ds-l: ds-l.o DataImage.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf server echo test qs nds ds ds-l
	rm -rf .depend