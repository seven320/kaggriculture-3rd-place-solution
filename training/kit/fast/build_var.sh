#!/bin/bash
# kagsim_var(ルール変種つきの写し)をビルドする。元の kagsim(.so)には一切触らない。
#   使い方: kagg/fast/build_var.sh [312|310|all]   (既定 all)
#   310 はコンテナ(nvcr pytorch 23.12, Python 3.10)用。ログインノードの /usr/include/python3.10 と
#   torch 同梱の pybind11 ヘッダで作る。計算ノードの CPU が違っても動くよう -march=native は付けない
#   (浮動小数の結果は変わらない: -std=c++17 では FMA への縮約が既定で無効)。
set -ue
cd "$(dirname "$0")"
WHICH=${1:-all}
PB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))")
if [ "$WHICH" = 312 ] || [ "$WHICH" = all ]; then
  INC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
  SUF=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
  g++ -O3 -shared -std=c++17 -fPIC -march=x86-64-v2 -I"$PB" -I"$INC" kagsim_var.cpp -o kagsim_var_tmp$SUF
  mv -f kagsim_var_tmp$SUF kagsim_var$SUF; echo "built kagsim_var$SUF"
fi
if [ "$WHICH" = 310 ] || [ "$WHICH" = all ]; then
  g++ -O3 -shared -std=c++17 -fPIC -march=x86-64-v2 -I"$PB" -I/usr/include/python3.10 kagsim_var.cpp -o kagsim_var_tmp.cpython-310-x86_64-linux-gnu.so
  mv -f kagsim_var_tmp.cpython-310-x86_64-linux-gnu.so kagsim_var.cpython-310-x86_64-linux-gnu.so; echo "built kagsim_var.cpython-310"
fi
