LIB_PATH=-L$(EM_PATH)/lib/
LIBINC=-I$(EM_PATH)/include/

EMFLAGS=-s USE_ZLIB=1 -s ASYNCIFY=1 -s ASSERTIONS=1
CFLAGS=-O3 -std=c++11
LDFLAGS=-lzmq -ljsoncpp -lwszmq


ALGOS=echo

all: echo test server.html 

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/module/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) $(EMFLAGS)

server.html: server.o dataManagement.o jobManager.o jobTasking.o status.o $(ALGO)
	$(CXX) -o $@ $(subst $(ALGOS),,$^) $(CFLAGS) $(LIB_PATH) $(LDFLAGS) $(EMFLAGS) -s MAIN_MODULE=1 -s TOTAL_MEMORY=268435456 --preload-file $(ALGOS) -s LINKABLE=1 --bind -s DISABLE_EXCEPTION_CATCHING=0
	sed -ie "s/var Module = {/var Module = {arguments: ['-mT', '-fM'],/" server.html

echo.js: echo.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(LDFLAGS) $(EMFLAGS)

echo:echo.js
	cp echo.js echo

test.js: test.o
	$(CXX) -o $@ $^ -s SIDE_MODULE=1 $(CFLAGS) $(LIB_PATH) $(EMFLAGS)

test:test.js
	cp test.js test

clean:
	rm -rf *.o *.js
	rm -rf server.html echo test
	rm -rf .depend