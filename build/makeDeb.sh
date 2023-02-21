#!/bin/env basj^h
make intel -j
cp ./g2s_algosName.config ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_server ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_echo ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_qs ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_nds ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_ds-l ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_algoNames ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_errorTest ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_auto_qs ./g2s-package/usr/bin/g2s_bin

sudo chown root:root -R ./g2s-package/*
dpkg-deb --build ./g2s-package