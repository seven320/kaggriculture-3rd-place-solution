#!/bin/bash
# C++ エンジン(kagsim)をビルドする。動いているプロセスが古い .so を使っている間は、上書きせず別名で作ってから mv で差し替える
cd "$(dirname "$0")"
INC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
g++ -O3 -shared -std=c++17 -fPIC -march=native $(python3 -c "import pybind11;print('-I'+pybind11.get_include())") -I"$INC" kagsim.cpp -o kagsim_build.so
cp kagsim_build.so kagsim_tmp.so && mv -f kagsim_tmp.so kagsim$(python3-config --extension-suffix)
