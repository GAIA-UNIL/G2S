#ifndef DATA_MANAGEMENT_HPP
#define DATA_MANAGEMENT_HPP

#include <iostream>
#include "zlib.h"
#include <zmq.hpp>
#include "picosha2.h"
#include "utils.hpp"
#include "DataImage.hpp"

int storeData(char* data, size_t sizeBuffer,bool force, bool compressed);
zmq::message_t sendData( char* dataName);

int dataIsPresent(char* data);

	


#endif // DATA_MANAGEMENT_HPP