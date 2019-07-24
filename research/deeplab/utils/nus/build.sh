#!/bin/bash -x

if ! [ -e build ]; then 
	mkdir build
fi

cd build
cmake ..
make

