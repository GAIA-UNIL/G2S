#!/bin/sh
cd "$(dirname "$0")"
sudo apt update -y
sudo apt upgrade -y
sudo apt install build-essential libzmq3-dev libjsoncpp-dev zlib1g-dev libfftw3-dev libcurl4-openssl-dev -y
sudo wget "https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp" -O /usr/include/zmq.hpp
cd ..
make c++ -j
