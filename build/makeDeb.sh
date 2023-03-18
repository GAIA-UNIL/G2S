#!/bin/env bash
make intel -j
mkdir -p ./g2s-package/usr/bin/g2s_bin/
cp ./algosName.config ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/g2s_server ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/echo ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/qs ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/nds ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/ds-l ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/errorTest ./g2s-package/usr/bin/g2s_bin/
cp ./intel-build/auto_qs ./g2s-package/usr/bin/g2s_bin/

sudo chown root:root -R ./g2s-package/*
sudo chmod 555 ./g2s-package/DEBIAN/postinst
sudo chmod 555 ./g2s-package/DEBIAN/prerm
sudo chmod 555 -R ./g2s-package/usr/bin/*
sudo chmod 555 -R ./g2s-package/etc/systemd/system/*
sed -i "8s/.*/& $(cat ../version)/" ./g2s-package/DEBIAN/control 
dpkg-deb --build ./g2s-package