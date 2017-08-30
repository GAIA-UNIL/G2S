
export SRCS=$(shell pwd)/../src
export CFLAGS=-O3 -std=c++11
export CXXFLAGS=
export INC=-I$(shell pwd)/../include
export LIBINC=-I/usr/include -I/opt/local/include
export LDFLAGS=-lzmq -lz -ljsoncpp
export LIB_PATH= -L/usr/lib -L/opt/local/lib

.DEFAULT:
	$(MAKE) -C c++-build $@
	emmake make -C em-build $@
	$(MAKE) -C intel-build $@

compile_all:all

c++:
	$(MAKE) -C c++-build

em:
	emmake make -C em-build

intel:
	$(MAKE) -C intel-build