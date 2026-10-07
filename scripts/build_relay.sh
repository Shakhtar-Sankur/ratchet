#!/bin/sh
# Builds relay's C interface (librelay_c.so) from the pinned submodule: with CUDA when
# nvcc is available (relay/build-cuda), otherwise CPU only (relay/build).
#   scripts/build_relay.sh [--cpu] [CUDA_ARCH]      e.g. scripts/build_relay.sh 75   (T4)
set -e
cd "$(dirname "$0")/.."
git submodule update --init relay
if [ "$1" != "--cpu" ] && command -v nvcc >/dev/null 2>&1; then
  ARCH=${1:-75}
  cmake -S relay -B relay/build-cuda -DRELAY_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DRELAY_NATIVE=OFF >/dev/null
  cmake --build relay/build-cuda -j "$(nproc)" --target relay_c
  echo "built relay/build-cuda/librelay_c.so (CUDA, sm_$ARCH)"
else
  cmake -S relay -B relay/build >/dev/null
  cmake --build relay/build -j "$(nproc)" --target relay_c
  echo "built relay/build/librelay_c.so (CPU)"
fi
