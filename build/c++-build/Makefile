









all: server echo test qs nds ds ds-l

depend: .depend Makefile

.depend: $(subst $(SRCS)/g2s.cpp,,$(subst $(SRCS)/cvtZMQ2WS.cpp,,$(wildcard $(SRCS)/*.cpp)))
	rm -f ./.depend
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC) -MM $^ > ./.depend;

include .depend

%.o: $(SRCS)/module/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC)

%.o: $(SRCS)/%.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS) $(CXXFLAGS) $(INC) $(LIBINC)

server: server.o dataManagement.o jobManager.o jobTasking.o status.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

echo: echo.o dataManagement.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

test: test.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

qs: qs.o dataManagement.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

nds: nds.o dataManagement.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

ds: ds.o dataManagement.o sharedMemoryManager.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

ds-l: ds-l.o dataManagement.o sharedMemoryManager.o CPUThreadDevice.o
	$(CXX) -o $@ $^ $(LIB_PATH) $(LDFLAGS)

clean:
	rm -rf *.o
	rm -rf server echo test qs nds ds ds-l
	rm -rf .depend