BUILD_DIR=../.
export SRCS=$(BUILD_DIR)/../src
export CFLAGS=-O3 -std=c++11 -fPIC
export CXXFLAGS=
export INC=-I$(BUILD_DIR)/../include
export LIBINC=-I/usr/include -I/usr/include/jsoncpp -I/opt/local/include
export LDFLAGS=-lzmq -lz -ljsoncpp -lfftw3f -lfftw3f_threads -ldl -lpthread
export LIB_PATH= -L/usr/lib -L/opt/local/lib

ifneq ("$(wildcard /usr/include/hbwmalloc.h)","")
export CFLAGS+= -DHBW_MALLOC
export LDFLAGS+= -lmemkind
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
	cd python-build && python3 compile.py build_ext --inplace
