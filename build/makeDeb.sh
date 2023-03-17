#!/bin/env bash
make intel -j
cp ./algosName.config ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/g2s_server ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/echo ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/qs ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/nds ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/ds-l ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/algoNames ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/errorTest ./g2s-package/usr/bin/g2s_bin
cp ./intel-build/auto_qs ./g2s-package/usr/bin/g2s_bin

sudo chown root:root -R ./g2s-package/*
dpkg-deb --build ./g2s-package