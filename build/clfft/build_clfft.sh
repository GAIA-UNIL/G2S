#!/usr/bin/env sh
# Build clFFT (OpenCL FFT library, single and double precision) into ./install.
#
# The OpenCL back ends of G2S (QS's -W_GPU and the -W_GPU of tb, fftma/spectral
# and kriging) need clFFT next to the system OpenCL. Homebrew and most Linux
# distributions no longer package it; when ./install exists, `make c++` uses
# it (build/Makefile adds its include and lib directories and enables
# WITH_OPENCL when the OpenCL library links). Nothing is installed outside
# this directory.
#
# Needs cmake and an OpenCL library (macOS: the OpenCL framework; Linux:
# libOpenCL / an ICD loader and the OpenCL headers).
#
# clFFT 2.12.2 does not compile with recent clang (it takes the address of a
# temporary std::ostringstream in generator.transpose.cpp); the two lines are
# patched with std::to_string, which is what they compute.
set -eu

CLFFT_VERSION=2.12.2
CLFFT_SHA256=e7348c146ad48c6a3e6997b7702202ad3ee3b5df99edf7ef00bbacc21e897b12
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CACHE_DIR="$SCRIPT_DIR/cache"
ARCHIVE="$CACHE_DIR/clFFT-$CLFFT_VERSION.tar.gz"
SOURCE_DIR="$CACHE_DIR/clFFT-$CLFFT_VERSION"
BUILD_DIR="$CACHE_DIR/build"
PREFIX="$SCRIPT_DIR/install"

mkdir -p "$CACHE_DIR"
if [ ! -f "$ARCHIVE" ]; then
	curl -fL "https://github.com/clMathLibraries/clFFT/archive/refs/tags/v$CLFFT_VERSION.tar.gz" -o "$ARCHIVE"
fi

if command -v sha256sum >/dev/null 2>&1; then
	ACTUAL_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
else
	ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
fi
if [ "$ACTUAL_SHA256" != "$CLFFT_SHA256" ]; then
	echo "clFFT checksum mismatch: expected $CLFFT_SHA256, got $ACTUAL_SHA256" >&2
	exit 1
fi

rm -rf "$SOURCE_DIR"
tar -xzf "$ARCHIVE" -C "$CACHE_DIR"
TRANSPOSE="$SOURCE_DIR/src/library/generator.transpose.cpp"
sed 's/static_cast<std::ostringstream\*>(&(std::ostringstream() << smaller_dim))->str()/std::to_string(smaller_dim)/; s/static_cast<std::ostringstream\*>(&(std::ostringstream() << dim_ratio))->str()/std::to_string(dim_ratio)/' \
	"$TRANSPOSE" > "$TRANSPOSE.patched"
mv "$TRANSPOSE.patched" "$TRANSPOSE"

JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

# build/Makefile exports G2S's flags; they must not leak into clFFT's build
unset CFLAGS CXXFLAGS LDFLAGS CPPFLAGS INC LIBINC LIB_PATH MAKEFLAGS MFLAGS

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Configuring clFFT $CLFFT_VERSION (static library)"
cmake "$SOURCE_DIR/src" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_CLIENT=OFF -DBUILD_TEST=OFF \
	-DBUILD_EXAMPLES=OFF -DBUILD_CALLBACK_CLIENT=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_INSTALL_PREFIX="$PREFIX" > configure.log
make -j"$JOBS" > make.log
make install > install.log
# clFFT installs its static library under lib/import (or lib64/import); move it next to the headers' lib
mkdir -p "$PREFIX/lib"
for library in "$PREFIX"/lib*/import/libclFFT.a; do
	if [ -f "$library" ]; then mv "$library" "$PREFIX/lib/libclFFT.a"; fi
done
echo "clFFT installed in $PREFIX; rebuild with 'make c++' to enable the OpenCL back ends."
