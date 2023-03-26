#!/bin/bash
make intel -j
mkdir -p ./g2s-amd64/usr/bin/g2s_bin/
cp ./algosName.config ./g2s-amd64/usr/bin/g2s_bin/
objcopy --strip-debug --strip-unneeded ./intel-build/g2s_server  ./g2s-amd64/usr/bin/g2s_bin/g2s_server 
objcopy --strip-debug --strip-unneeded ./intel-build/echo  ./g2s-amd64/usr/bin/g2s_bin/echo 
objcopy --strip-debug --strip-unneeded ./intel-build/qs  ./g2s-amd64/usr/bin/g2s_bin/qs 
objcopy --strip-debug --strip-unneeded ./intel-build/nds  ./g2s-amd64/usr/bin/g2s_bin/nds 
objcopy --strip-debug --strip-unneeded ./intel-build/ds-l ./g2s-amd64/usr/bin/g2s_bin/ds-l 
objcopy --strip-debug --strip-unneeded ./intel-build/errorTest  ./g2s-amd64/usr/bin/g2s_bin/errorTest 
objcopy --strip-debug --strip-unneeded ./intel-build/auto_qs  ./g2s-amd64/usr/bin/g2s_bin/auto_qs 

# Read the version number from the version file
version_number=$(cat ../../version)

# Replace the version placeholder in the man page source, in-place
sed -i -e "s/Version x.y.z/Version ${version_number}/g" ./g2s-amd64/usr/share/man/man1/g2s.1

gzip -v9 -n ./g2s-amd64/usr/share/doc/g2s/changelog
gzip -v9 -n ./g2s-amd64/usr/share/man/man1/g2s.1

sudo chown root:root -R ./g2s-amd64/*
sudo chmod 755 ./g2s-amd64/DEBIAN/postinst
sudo chmod 755 ./g2s-amd64/DEBIAN/prerm
sudo chmod 755 -R ./g2s-amd64/usr/bin/*
sudo chmod 755 -R ./g2s-amd64/lib/systemd/system/*
sudo chmod 644 ./g2s-amd64/lib/systemd/system/g2s.service
sudo chmod 644 ./g2s-amd64/usr/bin/g2s_bin/algosName.config
sed -i -e "s/Version: x.y.z/Version: ${version_number}/g" ./g2s-amd64/DEBIAN/control
dpkg-deb --build ./g2s-amd64