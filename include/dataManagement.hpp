#ifndef DATA_MANAGEMENT_HPP
#define DATA_MANAGEMENT_HPP

#include <iostream>
#include "zlib.h"
#include <zmq.hpp>
#include "picosha2.h"

int storeData(char* data, size_t sizeBuffer,bool force, bool compressed);
zmq::message_t sendData( char* dataName);

int dataIsPresent(char* data);

float* loadData(const char * hash, int &sizeX, int &sizeY, int &sizeZ, int &dim, int &nbVariable);
char* writeData(float* data, int sizeX, int sizeY, int sizeZ, int dim, int nbVariable);

#endif // DATA_MANAGEMENT_HPP