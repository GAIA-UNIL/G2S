#!/bin/sh
cd "$(dirname "$0")"
cd ../..
mkdir -p external
cd external
dir=$(pwd)/selfCompiled;
mkdir -p $dir

if [[ ! -d libzmq ]]; then
      git clone https://github.com/zeromq/libzmq.git
fi
cd libzmq
git pull
git clean -f
if [[ $(command -v autoconf) ]]; then
      bash ./autogen.sh
      ./configure --prefix=$dir
      make -j
      make install
else
      mkdir build
      cd build
      cmake -DCMAKE_INSTALL_PREFIX=$dir  ..
      make -j
      make install
      cd ..
fi
cd ..

wget "https://raw.githubusercontent.com/zeromq/cppzmq/5999e5adc7eeca10bb469246ee3909da037c9036/zmq.hpp" -O ${dir}/include/zmq.hpp

if [[ ! -d jsoncpp ]]; then
      git clone https://github.com/open-source-parsers/jsoncpp.git
fi
cd jsoncpp
git pull
git clean -f
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=release -DCMAKE_INSTALL_PREFIX="$dir" -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=ON -G "Unix Makefiles" ../..
make -j
make install

cd ../../..

if [[ ! -d fftw-3.3.8 ]]; then
      wget "http://www.fftw.org/fftw-3.3.8.tar.gz"
      tar -xf fftw-3.3.8.tar.gz
fi
cd fftw-3.3.8
make clean

dosentSupportOMP= $(shell sh -c 'echo "int main(){}" | c++ -xc - -o /dev/null -fopenmp 2>/dev/null && echo 0 || echo 1 ')
if [[ dosentSupportOMP ]]; then
      ./configure --prefix=$dir    \
            --enable-shared  \
            --enable-threads \
            --enable-sse2    \
            --enable-avx     \
            --enable-avx2    \
            --enable-float   \
            --enable-threads \
            # --enable-openmp
else
            ./configure --prefix=$dir    \
            --enable-shared  \
            --enable-threads \
            --enable-sse2    \
            --enable-avx     \
            --enable-avx2    \
            --enable-float   \
            --enable-threads \
            --enable-openmp
fi


make -j
make install
cd ..

cd ../build

# git checkout Makefile
if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "/export LIBINC=/s/$/ -I..\/..\/external\/selfCompiled\/include/" Makefile
      sed -i '' "/export LIB_PATH=/s/$/ -L..\/..\/external\/selfCompiled\/lib/" Makefile
else
      sed -i "/export LIBINC=/s/$/ -I..\/..\/external\/selfCompiled\/include/" Makefile
      sed -i "/export LIB_PATH=/s/$/ -L..\/..\/external\/selfCompiled\/lib/" Makefile
fi
