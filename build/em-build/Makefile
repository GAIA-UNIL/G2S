LIB_PATH=-L$(EM_PATH)/lib/
LIBINC=-I$(EM_PATH)/include/

EMFLAGS=-s USE_ZLIB=1 -s ASYNCIFY=1 -s ASSERTIONS=1
# -s SIMD=1 -s ASM_JS=2
CFLAGS=-O3 -std=c++11
CXXFLAGS+=-msse2
LDFLAGS=-lzmq -ljsoncpp -lwszmq


ALGOS=echo qs test

all: $(ALGOS) server.html

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2smodule.cpp,,$(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp))))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/module/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS)

server.html: server.o dataManagement.o sharedMemoryManager.o jobManager.o jobTasking.o status.o CPUThreadDevice.o $(ALGOS)
	$(CXX) -o $@ $(subst $(ALGOS),,$^) $(CFLAGS) $(LIB_PATH) $(LDFLAGS) $(EMFLAGS) -s MAIN_MODULE=1 -s TOTAL_MEMORY=268435456 $(addprefix --preload-file , $(ALGOS)) -s LINKABLE=1 --bind -s DISABLE_EXCEPTION_CATCHING=0
	sed -ie "s/var Module = {/var Module = {arguments: ['-mT', '-fM'],/" server.html

echo.js: echo.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(LDFLAGS) $(EMFLAGS)

echo:echo.js
	cp $^ $@

test.js: test.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(EMFLAGS)

test:test.js
	cp $^ $@

qs.js: qs.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(LDFLAGS) $(EMFLAGS)
	sed -i -e 's/shiftRightArithmeticByScalar/shiftRightByScalar/g' qs.js
	sed -i -e 's/shiftRightLogicalByScalar/shiftRightByScalar/g' qs.js

qs: qs.js
	cp $^ $@

nds.js: nds.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(EMFLAGS)

nds: nds.js
	cp $^ $@

ds.js: ds.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(EMFLAGS)

ds: ds.js
	cp $^ $@

ds-l.js: ds-l.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(EMFLAGS)
	sed -i -e 's/shiftRightArithmeticByScalar/shiftRightByScalar/g' ds-l.js
	sed -i -e 's/shiftRightLogicalByScalar/shiftRightByScalar/g' ds-l.js

ds-l: ds-l.js
	cp $^ $@

clean:
	rm -rf *.o *.js
	rm -rf server.html echo test
	rm -rf .depend