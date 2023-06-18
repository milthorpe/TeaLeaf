#!/usr/bin/env bash

set -eu

NVHPC_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/23.5"

VERBOSE="OFF"
function test() {
  rm -rf build
  "$@" # &>/dev/null
  cmake --build build # &>/dev/null
  echo "$@"
#  ./build/*-tealeaf
}

test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=serial -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=serial -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=omp -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS=--cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/

test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=cuda -DCMAKE_CUDA_COMPILER=$NVHPC_DIR/compilers/bin/nvcc -DCUDA_ARCH=sm_60

#(
#set +eu
#source  /opt/intel/oneapi/setvars.sh
#set -eu
#test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build  -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=sycl -DSYCL_COMPILER=ONEAPI-ICPX
#)
#
#(
#set +eu
#source  /opt/intel/oneapi/setvars.sh --include-intel-llvm
#set -eu
#test cmake -DCMAKE_BUILD_TYPE=Release -S. -B build  -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DMODEL=sycl \
#  -DSYCL_COMPILER=ONEAPI-Clang \
#  -DCXX_EXTRA_FLAGS="-fsycl-targets=nvptx64-nvidia-cuda;--cuda-path=$NVHPC_DIR/cuda/;-Xsycl-target-backend;--cuda-gpu-arch=sm_60" \
#  -DCXX_EXTRA_LINK_FLAGS="-fsycl-targets=nvptx64-nvidia-cuda;--cuda-path=$NVHPC_DIR/cuda/;-Xsycl-target-backend;--cuda-gpu-arch=sm_60"
#)
