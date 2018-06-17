#!/bin/sh

cp -r ~/githubProject/G2S/build/intel-build $PBS_SCRATCHDIR
cd $PBS_SCRATCHDIR/intel-build
./server -To 60