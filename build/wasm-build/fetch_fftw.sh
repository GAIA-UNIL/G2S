#!/usr/bin/env sh
set -eu

FFTW_VERSION=3.3.11
FFTW_SHA256=5630c24cdeb33b131612f7eb4b1a9934234754f9f388ff8617458d0be6f239a1
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CACHE_DIR="$SCRIPT_DIR/cache"
ARCHIVE="$CACHE_DIR/fftw-$FFTW_VERSION.tar.gz"
SOURCE_DIR="$CACHE_DIR/fftw-$FFTW_VERSION"

mkdir -p "$CACHE_DIR"
if [ ! -f "$ARCHIVE" ]; then
	curl -fL "https://fftw.org/fftw-$FFTW_VERSION.tar.gz" -o "$ARCHIVE"
fi

ACTUAL_SHA256=$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')
if [ "$ACTUAL_SHA256" != "$FFTW_SHA256" ]; then
	echo "FFTW checksum mismatch: expected $FFTW_SHA256, got $ACTUAL_SHA256" >&2
	exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
	tar -xzf "$ARCHIVE" -C "$CACHE_DIR"
fi

printf '%s\n' "$SOURCE_DIR"
