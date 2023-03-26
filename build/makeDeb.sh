#!/bin/bash
make intel -j
mkdir -p ./g2s-package/usr/bin/g2s_bin/
cp ./algosName.config ./g2s-package/usr/bin/g2s_bin/
objcopy --strip-debug --strip-unneeded ./intel-build/g2s_server  ./g2s-package/usr/bin/g2s_bin/g2s_server 
objcopy --strip-debug --strip-unneeded ./intel-build/echo  ./g2s-package/usr/bin/g2s_bin/echo 
objcopy --strip-debug --strip-unneeded ./intel-build/qs  ./g2s-package/usr/bin/g2s_bin/qs 
objcopy --strip-debug --strip-unneeded ./intel-build/nds  ./g2s-package/usr/bin/g2s_bin/nds 
objcopy --strip-debug --strip-unneeded ./intel-build/ds-l ./g2s-package/usr/bin/g2s_bin/ds-l 
objcopy --strip-debug --strip-unneeded ./intel-build/errorTest  ./g2s-package/usr/bin/g2s_bin/errorTest 
objcopy --strip-debug --strip-unneeded ./intel-build/auto_qs  ./g2s-package/usr/bin/g2s_bin/auto_qs 

# Read the version number from the version file
version_number=$(cat ../version)

# Replace the version placeholder in the man page source, in-place
sed -i -e "s/Version x.y.z/Version ${version_number}/g" ./g2s-package/share/man/man1/g2s.1

gzip -n ./g2s-package/share/doc/g2s/changelog.Debian
gzip -n ./g2s-package/share/man/man1/g2s.1


sudo chown root:root -R ./g2s-package/*
sudo chmod 755 ./g2s-package/DEBIAN/postinst
sudo chmod 755 ./g2s-package/DEBIAN/prerm
sudo chmod 755 -R ./g2s-package/usr/bin/*
sudo chmod 755 -R ./g2s-package/lib/systemd/system/*
sudo chmod 644 ./g2s-package/lib/systemd/system/g2s.service
sudo chmod 644 ./g2s-package/usr/bin/g2s_bin/algosName.config
sed -i -e "s/Version: x.y.z/Version: ${version_number}/g" ./g2s-package/DEBIAN/control
dpkg-deb --build ./g2s-package