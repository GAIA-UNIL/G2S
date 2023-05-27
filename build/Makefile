BUILD_DIR=../.

export GIT_URL=$(shell sh -c 'git config --get remote.origin.url || echo "https://github.com/GAIA-UNIL/G2S.git"')

export SRCS=$(BUILD_DIR)/../src
export CFLAGS=-O3 -std=c++11 -Wno-deprecated -Wno-unused -Wno-deprecated-declarations
export CXXFLAGS=
export INC=-I$(BUILD_DIR)/../include
export LIBINC=-I/usr/include -I/usr/include/jsoncpp -I/opt/local/include
export LDFLAGS=-lz -lpthread
export LDFLAGS_NO_SERVER_ONLY= -lfftw3f -lfftw3f_threads -ldl
export LDFLAGS_SERVER_ONLY= -lzmq -ljsoncpp
export LIB_PATH= -L/usr/lib -L/usr/lib/x86_64-linux-gnu/ -L/opt/local/lib -L/usr/local/lib/
export ARFLAGS= rcs
export ExtraDevice=AcceleratorDevice.o

export brewIsAvailable := $(shell brew -v > /dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(brewIsAvailable),1)
	export LIBINC+=-I$(shell  brew --prefix)/include
	export LIB_PATH+=-L$(shell  brew --prefix)/lib
endif

export NVCC=nvcc
export NVFLAGS +=
export NVLIB +=


export UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS =linux
    OPENCL_LIB+= -lOpenCL -lclFFT
endif
ifeq ($(UNAME_S),Darwin)
    OS =mac
    OPENCL_LIB+= -framework OpenCL -lclFFT
endif

export OS

dosentSupportOpenCL:= $(shell echo 'int main(){}' | c++ -xc - -o /dev/null ${OPENCL_LIB} 2>/dev/null && echo 1 || echo 0 )
ifeq ($(dosentSupportOpenCL),1)
	export OPENCL_INC+=-DWITH_OPENCL=1
	export OPENCL_LIB
else
	export OPENCL_LIB=
endif

dosentSupportOMP:= $(shell echo "int main(){}" | c++ -xc - -o /dev/null -fopenmp 2>/dev/null && echo 1 || echo 0)
ifeq ($(dosentSupportOMP),1)
	export CFLAGS += -fopenmp
	export LDFLAGS += -fopenmp
endif

dosentSupportCUDA:= $(shell echo "int main(){}" | ${NVCC} -x c - -o /dev/null 2>/dev/null && echo 1 || echo 0)
ifeq ($(dosentSupportCUDA),1)
	#export ExtraDevice+= NvidiaGPUAcceleratorDevice.o
	export CFLAGS+= -DWITH_CUDA
	export WITH_CUDA+=g2s_cuda.so
	export NVLIB +=-L$(dir $(shell sh -c "which ${NVCC}"))/../targets/x86_64-linux/lib/ -lcudart -lcufft -lcublas
	export LIBINC+=-I$(dir $(shell sh -c "which ${NVCC}"))/../targets/x86_64-linux/include/
endif


ifneq ("$(wildcard /usr/include/hbwmalloc.h)","")
	export CFLAGS+= -DHBW_MALLOC
	export LDFLAGS_SERVER_ONLY+= -lmemkind
endif

ifneq (,$(wildcard /usr/include/*/curl/curl.h /usr/include/curl/curl.h /opt/local/include/curl/curl.h))
	export CFLAGS+= -DWITH_VERSION_CONTROL -DCURRENT_VERSION=\"$(shell cat ../version)\" -DGIT_URL=\"$(GIT_URL)\"
	export LDFLAGS_SERVER_ONLY+= -lcurl
endif

DEFAULT_JOB=c++
ifneq ($(shell icpc 2>/dev/null; echo $$? ),127)
	DEFAULT_JOB+=intel
endif
ifneq ($(shell icpx 2>/dev/null; echo $$? ),127)
	DEFAULT_JOB+=intel
endif

DEFAULT_JOB := $(sort $(DEFAULT_JOB))

.DEFAULT_GOAL=default

.PHONY: c++ em intel update-extension c++-extension intel-extension extension python python-extension-module

.DEFAULT:
	$(foreach JOB,$(DEFAULT_JOB),$(MAKE) -C $(JOB)-build $(MAKECMDGOALS);)

c++:
	$(MAKE) -C c++-build

em:
	emmake make -C em-build

intel:
	$(MAKE) -C intel-build

c++-server:
	$(MAKE) g2s_server -C c++-build

update-extension:
	git submodule init
	git submodule update
	git submodule foreach git pull origin master

c++-extension:
	$(MAKE) -C c++-build extension

intel-extension:
	$(MAKE) -C intel-build extension

extension: c++-extension intel-extension

python:
	cd python-build && (python3 setup.py install --user || python setup.py install --user)

python-extension-module: python
	ln -sf  ../python-build/pyEcho.py c++-build/pyEcho.py
	ln -sf  $(shell which python3) c++-build/python3
	ln -sf  ../python-build/pyEcho.py intel-build/pyEcho.py
	ln -sf  $(shell which python3) intel-build/python3
	echo "pyEcho	./pyEcho.py	-ti" >> algosName.config