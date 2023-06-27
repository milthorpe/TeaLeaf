#!/usr/bin/env bash

set -eu

NVHPC_DIR="/opt/nvidia/hpc_sdk/Linux_x86_64/23.5"

VERBOSE="ON"

function test() {
  rm -rf build
  echo "${@:3}"
  cmake -DCMAKE_BUILD_TYPE=Release -S. -B build -GNinja -DCMAKE_VERBOSE_MAKEFILE=$VERBOSE -DENABLE_MPI=ON "${@:3}" # &>/dev/null
  cmake --build build                                                                                              #  &>/dev/null
  #  export OMP_PLACES=cores
  #  export OMP_PROC_BIND=true
  export OMP_NUM_THREADS=1
  mpirun -np "$1" ./build/*-tealeaf --device "$2" # | grep -i -e "This run" -e "Timestep *" #--file Benchmarks/tea_bm_4.in
}

(
  module load mpi
  #  test 12 "0" -DMODEL=serial -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  #  test 12 "0" -DMODEL=serial -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  #  test 12 "0" -DMODEL=omp -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  #  test 12 "0" -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
  #  test 12 "0" -DMODEL=omp -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DOFFLOAD=NVIDIA:sm_60 -DCXX_EXTRA_FLAGS=--cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/
  #  test 3 "0" -DMODEL=cuda -DCMAKE_CUDA_COMPILER=$NVHPC_DIR/compilers/bin/nvcc -DCUDA_ARCH=sm_60
  #  test 3 "0" -DMODEL=hip -DCMAKE_CXX_COMPILER=/usr/lib/aomp_17.0-1/bin/hipcc

  #  test 12 "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DKOKKOS_IN_TREE=/home/tom/Downloads/kokkos-4.0.01/ -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON
  #  test 12 "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DKOKKOS_IN_TREE=/home/tom/Downloads/kokkos-4.0.01/ -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ZEN3=ON

  #  module use /opt/nvidia/hpc_sdk/modulefiles
  #  module load nvhpc-nompi

  #  export LD_LIBRARY_PATH=${NVHPC_DIR}/cuda/lib64:${LD_LIBRARY_PATH:-}

)

(
  module load mpi
  export PATH=/opt/rocm-5.4.3/bin:${PATH:-}
  test 3 "0" -DMODEL=kokkos -DCMAKE_CXX_COMPILER=hipcc \
    -DKOKKOS_IN_TREE=/home/tom/Downloads/kokkos-4.0.01/ -DKokkos_ENABLE_HIP=ON  -DKokkos_ARCH_NAVI1012=ON
)

#(
#  export PATH=$NVHPC_DIR/compilers/bin/:${PATH:-}
#  test 3 "0" -DMODEL=kokkos -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ \
#    -DKOKKOS_IN_TREE=/home/tom/Downloads/kokkos-4.0.01/ -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_PASCAL61=ON
#)
#
#(
#  set +eu
#  source /opt/intel/oneapi/setvars.sh
#  set -eu
#  export DPCPP_CPU_NUM_CUS=1
#  export DPCPP_CPU_SCHEDULE=static
#  test 12 "AMD" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-ICPX
#  test 12 "AMD" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-ICPX
#)

(
  set +eu
  source /opt/intel/oneapi/setvars.sh --include-intel-llvm
  set -eu
  #  cuda_sycl_flags="-fsycl-targets=nvptx64-nvidia-cuda;--cuda-path=$NVHPC_DIR/cuda/;-Xsycl-target-backend;--cuda-gpu-arch=sm_60"
  #  test 3 "NVIDIA" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"
  #  test 3 "NVIDIA" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$cuda_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$cuda_sycl_flags"

  #  export LD_LIBRARY_PATH=/opt/rocm-5.4.3/lib:${LD_LIBRARY_PATH:-}
  #  hip_sycl_flags="-fsycl;-fsycl-targets=amdgcn-amd-amdhsa;-Xsycl-target-backend;--offload-arch=gfx1012"
  #  test 3 "Radeon" -DMODEL=sycl-acc -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
  #  test 3 "Radeon" -DMODEL=sycl-usm -DSYCL_COMPILER=ONEAPI-Clang -DSYCL_COMPILER_DIR=/opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/ -DCXX_EXTRA_FLAGS="$hip_sycl_flags" -DCXX_EXTRA_LINK_FLAGS="$hip_sycl_flags"
)
