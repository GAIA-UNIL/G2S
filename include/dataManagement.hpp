/*
 * G2S
 * Copyright (C) 2018, Mathieu Gravey (gravey.mathieu@gmail.com) and UNIL (University of Lausanne)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
int storeJson(char* data, size_t sizeBuffer,bool force, bool compressed);
zmq::message_t sendJson( char* dataName);

int dataIsPresent(char* data);

	


#endif // DATA_MANAGEMENT_HPP