#!/bin/bash

cd "$(dirname "$0")"

version_number=$(cat ../version)
sed -i -e "s/Version: x.y.z/Version: ${version_number}/g" ./g2s/DEBIAN/control

bash ./makeDeb-generic.sh
bash ./makeDeb-amd.sh