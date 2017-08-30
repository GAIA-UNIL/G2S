









all: server echo test

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

clean:
	rm -rf *.o
	rm -rf server echo test
	rm -rf .depend