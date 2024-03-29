#!/bin/bash

sudo apt-get install build-essential devscripts debhelper -y
mkdir -p g2s-generic/tmp/g2s-generic-src/build

cp -r ../../src g2s-generic/tmp/g2s-generic-src/
cp -r ../../include g2s-generic/tmp/g2s-generic-src/
cp ../../version g2s-generic/tmp/g2s-generic-src/
wget "https://raw.githubusercontent.com/zeromq/cppzmq/master/zmq.hpp" -O g2s-generic/tmp/g2s-generic-src/include/zmq.hpp
mkdir -p g2s-generic/tmp/g2s-generic-src/build/c++-build
cp ../Makefile g2s-generic/tmp/g2s-generic-src/build/
cp ../c++-build/Makefile g2s-generic/tmp/g2s-generic-src/build/c++-build/
cp ../algosName.config g2s-generic/tmp/g2s-generic-src/build/

mkdir -p ./g2s-generic/usr/bin/g2s_bin/

# Read the version number from the version file
version_number=$(cat ../../version)

# Replace the version placeholder in the man page source, in-place
sed -i -e "s/Version x.y.z/Version ${version_number}/g" ./g2s-generic/usr/share/man/man1/g2s.1

gzip -v9 -n ./g2s-generic/usr/share/doc/g2s/changelog
gzip -v9 -n ./g2s-generic/usr/share/man/man1/g2s.1

sudo chown root:root -R ./g2s-generic/*
sudo chmod 755 ./g2s-generic/DEBIAN/postinst
sudo chmod 755 ./g2s-generic/DEBIAN/prerm
sudo chmod 755 -R ./g2s-generic/usr/bin/*
sudo chmod 755 -R ./g2s-generic/lib/systemd/system/*
sudo chmod 644 ./g2s-generic/lib/systemd/system/g2s.service
sed -i -e "s/Version: x.y.z/Version: ${version_number}/g" ./g2s-generic/DEBIAN/control
dpkg-deb --build ./g2s-generic "g2s_${version_number}_all.deb"