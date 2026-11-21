#!/usr/bin/env sh
# Build a SIMD-enabled single-precision FFTW for QS/NDS into ./install.
#
# QS spends most of its time in FFTW. Some packaged FFTW builds have no SIMD kernels:
# Homebrew's on Apple silicon, for example, is built without NEON, which makes every
# transform about 1.6x slower. When ./install exists, `make c++` links it statically
# in place of the system FFTW.
#
# SIMD flags follow the target architecture:
#   arm64 / aarch64 : NEON
#   x86-64          : SSE2, AVX, AVX2 (FFTW checks the CPU at run time, so the
#                     library still runs on CPUs without AVX)
# Override with FFTW_SIMD_FLAGS="..." (for example to add --enable-avx512).
set -eu

FFTW_VERSION=3.3.11
FFTW_SHA256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CACHE_DIR="$SCRIPT_DIR/cache"
ARCHIVE="$CACHE_DIR/fftw-$FFTW_VERSION.tar.gz"
SOURCE_DIR="$CACHE_DIR/fftw-$FFTW_VERSION"
BUILD_DIR="$CACHE_DIR/build"
PREFIX="$SCRIPT_DIR/install"

mkdir -p "$CACHE_DIR"
if [ ! -f "$ARCHIVE" ]; then
	curl -fL "https://fftw.org/fftw-$FFTW_VERSION.tar.gz" -o "$ARCHIVE"
fi

if command -v sha256sum >/dev/null 2>&1; then
	ACTUAL_SHA256=$(sha256sum "$ARCHIVE" | awk '{print $1}')
else
	ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
fi
if [ "$ACTUAL_SHA256" != "$FFTW_SHA256" ]; then
	echo "FFTW checksum mismatch: expected $FFTW_SHA256, got $ACTUAL_SHA256" >&2
	exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
	tar -xzf "$ARCHIVE" -C "$CACHE_DIR"
fi

if [ -z "${FFTW_SIMD_FLAGS+x}" ]; then
	case "$(uname -m)" in
		arm64|aarch64) FFTW_SIMD_FLAGS="--enable-neon" ;;
		x86_64|amd64) FFTW_SIMD_FLAGS="--enable-sse2 --enable-avx --enable-avx2" ;;
		*) FFTW_SIMD_FLAGS="" ;;
	esac
fi

JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

# build/Makefile exports G2S's C++ flags; they must not leak into FFTW's C build
unset CXXFLAGS LDFLAGS CPPFLAGS INC LIBINC LIB_PATH MAKEFLAGS MFLAGS
CFLAGS="${FFTW_CFLAGS:--O3}"
export CFLAGS

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "Configuring FFTW $FFTW_VERSION (single precision) with: ${FFTW_SIMD_FLAGS:-no SIMD}"
# shellcheck disable=SC2086
"$SOURCE_DIR/configure" --prefix="$PREFIX" --enable-float --enable-threads \
	--enable-static --disable-shared --with-pic --disable-fortran --disable-doc \
	$FFTW_SIMD_FLAGS > configure.log
make -j"$JOBS" > make.log
make install > install.log
echo "SIMD FFTW installed in $PREFIX; rebuild with 'make c++' to use it."
