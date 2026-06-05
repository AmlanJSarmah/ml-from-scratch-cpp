#!/usr/bin/env bash
set -euo pipefail

if ! command -v cmake >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y cmake
  else
    echo "cmake not found and apt-get is unavailable" >&2
    exit 1
  fi
fi

python3 -m pip install --upgrade pip --break-system-packages
python3 -m pip install scikit-learn numpy pandas --break-system-packages

npm --prefix server ci
npm --prefix server run build

mkdir -p build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -S . -B build
cmake --build build -j
