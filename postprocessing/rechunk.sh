#!/bin/bash

# move to main foler of isi-cfact if executed from this folder

if [ -e rechunk.sh ]; then
    cd ..
fi
bash preprocessing/rechunk.sh 0 #  0 for executing rechunk to optimise for access of lonlat grid
