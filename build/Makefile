BUILD_DIR=../.

export GIT_URL=$(shell sh -c 'git config --get remote.origin.url || echo "https://github.com/GAIA-UNIL/G2S.git"')

export SRCS=$(BUILD_DIR)/../src
export CFLAGS=-O3 -std=c++11 -fPIC
export CXXFLAGS=
export INC=-I$(BUILD_DIR)/../include
export LIBINC=-I/usr/include -I/usr/include/jsoncpp -I/opt/local/include
export LDFLAGS=-lzmq -lz -ljsoncpp -lcurl -lfftw3f -lfftw3f_threads -ldl -lpthread
export LIB_PATH= -L/usr/lib -L/opt/local/lib

dosentSupportOMP:= $(shell sh -c 'echo "int main(){}" | c++ -xc - -o /dev/null -fopenmp 2>/dev/null && echo 0 || echo 1 ')
ifeq ("$(dosentSupportOMP)","0")
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

ifneq ("$(wildcard /usr/include/hbwmalloc.h)","")
export CFLAGS+= -DHBW_MALLOC
export LDFLAGS+= -lmemkind
endif

ifneq (,$(wildcard /usr/include/*/curl/curl.h /usr/include/curl/curl.h /opt/local/include/curl/curl.h))
	export CFLAGS+= -DWITH_VERSION_CONTROL -DCURRENT_VERSION=\"$(shell sh -c "cat ../version")\" -DGIT_URL=\"$(GIT_URL)\"
endif

.DEFAULT:
	$(MAKE) -C c++-build $@
	$(MAKE) -C intel-build $@

compile_all:all

c++:
	$(MAKE) -C c++-build

em:
	emmake make -C em-build

intel:
	$(MAKE) -C intel-build

python:
	cd python-build && python3 setup.py build_ext --inplace
