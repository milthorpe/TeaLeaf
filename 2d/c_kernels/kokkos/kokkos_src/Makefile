KOKKOS_OPTIONS=KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest

lib:
	cd core; \
	make -j -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/src/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest

install: lib
	cd core; \
	make -j -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/src/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest install

build-test:
	make -C core/unit_test
	make -C core/perf_test
	make -C containers/unit_tests
	make -C containers/performance_tests
	make -C algorithms/unit_tests
	make -C example/fixture
	make -C example/feint
	make -C example/fenl

test: build-test
	make -C core/unit_test test
	make -C core/perf_test test
	make -C containers/unit_tests test
	make -C containers/performance_tests test
	make -C algorithms/unit_tests test
	make -C example/fixture test
	make -C example/feint test
	make -C example/fenl test

clean:
	make -C core/unit_test clean
	make -C core/perf_test clean
	make -C containers/unit_tests clean
	make -C containers/performance_tests clean
	make -C algorithms/unit_tests clean
	make -C example/fixture clean
	make -C example/feint clean
	make -C example/fenl clean
	cd core; \
	make -f /home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/core/src/Makefile KOKKOS_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src KOKKOS_DEVICES=,OpenMP,Cuda GTEST_PATH=/home/users/p02185/repos/TeaLeaf/2d/c_kernels/kokkos/kokkos_src/tpls/gtest clean
