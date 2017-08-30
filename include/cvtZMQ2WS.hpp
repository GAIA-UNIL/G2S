#ifndef CVT_ZMQ2WS_HPP
#define CVT_ZMQ2WS_HPP
#include <atomic>
#include <string>
#define ASIO_STANDALONE

void cvtServer(char* from, char* to, std::atomic<bool> &serverRun, std::atomic<bool> &done);


#endif // CVT_ZMQ2WS_HPP