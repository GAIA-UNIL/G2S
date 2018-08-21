/*
 * G2S (c) by Mathieu Gravey (gravey.mathieu@gmail.com)
 * 
 * G2S is licensed under a
 * Creative Commons Attribution-NonCommercial 4.0 International License.
 * 
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
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

int dataIsPresent(char* data);

	


#endif // DATA_MANAGEMENT_HPP