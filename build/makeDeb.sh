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


sudo chown root:root -R ./g2s-package/*
sudo chmod 755 ./g2s-package/DEBIAN/postinst
sudo chmod 755 ./g2s-package/DEBIAN/prerm
sudo chmod 755 -R ./g2s-package/usr/bin/*
sudo chmod 755 -R ./g2s-package/lib/systemd/system/*
sudo chmod 644 ./g2s-package/lib/systemd/system/g2s.service
sudo chmod 644 ./g2s-package/usr/bin/g2s_bin/algosName.config
sed -i "8s/.*/& $(cat ../version)/" ./g2s-package/DEBIAN/control 
dpkg-deb --build ./g2s-package