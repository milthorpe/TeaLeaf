KOKKOS_OPTIONS=KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest

all:
	make -j -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/unit_test/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest

test: all
	make -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/unit_test/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest test

clean:
	make -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/unit_test/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest clean
