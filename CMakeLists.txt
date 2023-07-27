cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(tealeaf VERSION 1.0 LANGUAGES C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

# some nicer defaults for standard C
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)


include(FetchContent)

######
#set(MODEL hip)
######
#set(MODEL std-indices)
#set(NVHPC_OFFLOAD cc60)
#set(CXX_EXTRA_FLAGS -stdpar=multicore)
#set(USE_TBB ON)


########
#########
#set(MODEL std-indices)
#set(NVHPC_OFFLOAD cc60)
#set(CXX_EXTRA_FLAGS -stdpar=multicore)
###########
#set(MODEL serial)
#########
#set(MODEL kokkos)
#set(KOKKOS_IN_TREE /home/tom/Downloads/kokkos-4.0.01/)
#set(Kokkos_ENABLE_OPENMP ON)
########
######
#set(MODEL cuda)
#set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/bin/nvcc)
#set(CUDA_ARCH sm_60)
#set(ENABLE_PROFILING ON)
#set(MPI_HOME /usr/lib64/openmpi/)
######
#set(MODEL sycl)
#set(SYCL_COMPILER ONEAPI-DPCPP)
#set(SYCL_COMPILER_DIR /opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/)
#set(CXX_EXTRA_FLAGS -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/ -Xsycl-target-backend --cuda-gpu-arch=sm_60)
#set(CXX_EXTRA_LINK_FLAGS -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/ -Xsycl-target-backend --cuda-gpu-arch=sm_60)
######
#set(MODEL sycl-acc)
#set(SYCL_COMPILER ONEAPI-ICPX)
#set(SYCL_COMPILER_DIR /opt/intel/oneapi/compiler/2023.1.0/linux/bin-llvm/)
#set(CXX_EXTRA_FLAGS -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/ -Xsycl-target-backend --cuda-gpu-arch=sm_60)
#set(CXX_EXTRA_LINK_FLAGS -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/ -Xsycl-target-backend --cuda-gpu-arch=sm_60)
#set(ENABLE_MPI ON)
#set(ENABLE_PROFILING OFF)


######
#set(CMAKE_CXX_COMPILER g++)
#set(CMAKE_C_COMPILER gcc)
#set(CMAKE_CUDA_HOST_COMPILER g++)
######
#set(MODEL omp)
#set(OFFLOAD NVIDIA:sm_60)
#set(CXX_EXTRA_FLAGS --cuda-path=/opt/nvidia/hpc_sdk/Linux_x86_64/2023/cuda/)


#set(MPI_HOME /usr/lib64/openmpi/)


## Flags for debugging only, enable for development (ASan only works on few models)
set(SANITIZE OFF)
if (SANITIZE)
    set(C_EXTRA_FLAGS -fsanitize=address -fsanitize=undefined)
    set(C_EXTRA_LINK_FLAGS -fsanitize=address -fsanitize=undefined)
    set(CXX_EXTRA_FLAGS -fsanitize=address -fsanitize=undefined)
    set(CXX_EXTRA_LINK_FLAGS -fsanitize=address -fsanitize=undefined)
endif ()


# the final executable name
set(EXE_NAME tealeaf)


## select default build type if not given
if (NOT CMAKE_BUILD_TYPE)
    message("No CMAKE_BUILD_TYPE specified, defaulting to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ((NOT BUILD_TYPE STREQUAL RELEASE) AND (NOT BUILD_TYPE STREQUAL DEBUG))
    message(FATAL_ERROR "Only Release or Debug is supported, got `${CMAKE_BUILD_TYPE}`")
endif ()

# setup some defaults flags for everything
set(DEFAULT_DEBUG_C_FLAGS -O2 ${DEBUG_FLAGS} -g)
set(DEFAULT_RELEASE_C_FLAGS -O3   -march=native -g)
set(DEFAULT_DEBUG_CXX_FLAGS -O2)
set(DEFAULT_RELEASE_CXX_FLAGS -O3   -march=native -g)

macro(hint_flag FLAG DESCRIPTION)
    if (NOT DEFINED ${FLAG})
        message(STATUS "${FLAG}: ${DESCRIPTION}")
    else ()
        # i.e. `-DFOO="-a -b"` becomes CMake's semicolon separated list `FOO=`-a;-b`
        separate_arguments(${FLAG})
    endif ()
endmacro()

# hint common extra flag options for all models if they are not set
hint_flag(CXX_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `CXX_EXTRA_LINK_FLAGS`")
hint_flag(CXX_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(CXX_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(CXX_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED CXX_EXTRA_FLAGS) AND (NOT DEFINED CXX_EXTRA_LINK_FLAGS))
    set(CXX_EXTRA_LINK_FLAGS ${CXX_EXTRA_FLAGS})
endif ()


hint_flag(C_EXTRA_FLAGS "
        Appends to common compile flags. These will be appended at link phase as well.
        To use separate flags at link phase, set `C_EXTRA_LINK_FLAGS`")
hint_flag(C_EXTRA_LINK_FLAGS "
        Appends to link flags which appear *before* the objects.
        Do not use this for linking libraries, as the link line is order-dependent")
hint_flag(C_EXTRA_LIBRARIES "
        Append to link flags which appear *after* the objects.
        Use this for linking extra libraries (e.g `-lmylib`, or simply `mylib`)")
hint_flag(C_EXTRA_LINKER_FLAGS "
        Append to linker flags (i.e GCC's `-Wl` or equivalent)")

# copy CXX_EXTRA_FLAGS <- CXX_EXTRA_LINK_FLAGS
if ((DEFINED C_EXTRA_FLAGS) AND (NOT DEFINED C_EXTRA_LINK_FLAGS))
    set(C_EXTRA_LINK_FLAGS ${C_EXTRA_FLAGS})
endif ()


option(USE_TBB "Enable oneTBB library for *supported* models. Enabling this on models that
                don't explicitly link against TBB is a no-op, see description of your selected
                model on how this is used." OFF)

if (USE_TBB)
    FetchContent_Declare(
            TBB
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
            GIT_TAG v2021.9.0
    )
    # Don't fail builds on waring (TBB has -Wall while not being free of warnings from unused symbols...)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(TBB_STRICT OFF)
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(TBB)
    if (NOT TBB_POPULATED)
        FetchContent_Populate(TBB)
        add_subdirectory(${tbb_SOURCE_DIR} ${tbb_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

option(USE_ONEDPL "Enable oneDPL library for *supported* models. Enabling this on models that
                   don't explicitly link against DPL is a no-op, see description of your selected
                   model on how this is used." OFF)

if (USE_ONEDPL)
    FetchContent_Declare(
            oneDPL
            GIT_REPOSITORY https://github.com/oneapi-src/oneDPL.git
            GIT_TAG oneDPL-2022.1.0-rc3
    )
    string(TOLOWER ${USE_ONEDPL} ONEDPL_BACKEND)
    # XXX oneDPL looks for omp instead of openmp, which mismatches(!) with ONEDPL_PAR_BACKEND if using find_package
    if (ONEDPL_BACKEND STREQUAL "openmp")
        set(ONEDPL_BACKEND omp)
    endif ()
    # Not using FetchContent_MakeAvailable (CMake>= 3.14) because we need EXCLUDE_FROM_ALL
    FetchContent_GetProperties(oneDPL)
    if (NOT oneDPL_POPULATED)
        FetchContent_Populate(oneDPL)
        if (USE_TBB)
            macro(find_package NAME)
                if ("${NAME}" STREQUAL "TBB")
                    message(STATUS "Discarding oneDPL's call to find_package(${NAME} ${ARGN})")
                else ()
                    _find_package(${NAME} ${ARGN})
                endif ()
            endmacro()
        endif ()
        add_subdirectory(${onedpl_SOURCE_DIR} ${onedpl_BINARY_DIR} EXCLUDE_FROM_ALL)

        # Fixup oneDPL's omission on setting DPCPP definitions.
        # We do this after the creation of the oneDPL target.
        if (ONEDPL_BACKEND MATCHES "^(dpcpp|dpcpp_only)$")
            target_compile_definitions(oneDPL INTERFACE ONEDPL_USE_DPCPP_BACKEND=1)
        endif ()

    endif ()
endif ()


# include our macros
include(cmake/register_models.cmake)

register_flag_optional(ENABLE_MPI "Enables MPI support at compile time, set MPI_HOME (e.g -DMPI_HOME=/usr/lib64/openmpi/) if not on PATH" OFF)
register_flag_optional(ENABLE_PROFILING "Enables kernel profiler, this may introduce synchronisation overhead for some models." OFF)

# register out models <model_name> <preprocessor_def_name> <source files...>
register_model(omp USE_OMP
        cg.c cheby.c field_summary.c diffuse_overload.c jacobi.c kernel_initialise.c
        kernel_interface.c local_halos.c pack_halos.c ppcg.c set_chunk_data.c set_chunk_state.c
        solver_methods.c store_energy.c)
register_model(serial USE_SERIAL
        cg.c cheby.c field_summary.c jacobi.c kernel_initialise.c kernel_interface.c
        local_halos.c pack_halos.c ppcg.c set_chunk_data.c set_chunk_state.c
        solver_methods.c store_energy.c)
register_model(cuda USE_CUDA
        cg.cu cheby.cu cuknl_shared.cu field_summary.cu jacobi.cu kernel_initialise.cu
        kernel_interface.cu local_halos.cu pack_halos.cu ppcg.cu set_chunk_data.cu
        set_chunk_state.cu set_field.cu solver_methods.cu store_energy.cu)
register_model(hip USE_HIP
        cg.cpp cheby.cpp cuknl_shared.cpp field_summary.cpp jacobi.cpp kernel_initialise.cpp
        kernel_interface.cpp local_halos.cpp pack_halos.cpp ppcg.cpp set_chunk_data.cpp
        set_chunk_state.cpp set_field.cpp solver_methods.cpp store_energy.cpp)
register_model(std-indices USE_STD
        cg.cpp cheby.cpp field_summary.cpp jacobi.cpp kernel_initialise.cpp kernel_interface.cpp
        local_halos.cpp pack_halos.cpp ppcg.cpp set_chunk_data.cpp set_chunk_state.cpp
        solver_methods.cpp store_energy.cpp)
register_model(kokkos USE_KOKKOS
        cg.cpp cheby.cpp jacobi.cpp kernel_initialise.cpp kernel_interface.cpp local_halos.cpp
        pack_halos.cpp ppcg.cpp set_chunk_data.cpp set_chunk_state.cpp
        solver_methods.cpp store_energy.cpp)

register_model(sycl-acc USE_SYCL_ACC
        cg.cpp cheby.cpp field_summary.cpp jacobi.cpp kernel_initialise.cpp kernel_interface.cpp
        local_halos.cpp pack_halos.cpp ppcg.cpp set_chunk_data.cpp set_chunk_state.cpp
        solver_methods.cpp store_energy.cpp)

register_model(sycl-usm USE_SYCL_USM
        cg.cpp cheby.cpp field_summary.cpp jacobi.cpp kernel_initialise.cpp kernel_interface.cpp
        local_halos.cpp pack_halos.cpp ppcg.cpp set_chunk_data.cpp set_chunk_state.cpp
        solver_methods.cpp store_energy.cpp)

#register_model(sycl SYCL fasten.hpp)
#register_model(acc ACC fasten.hpp)
# defining RAJA collides with the RAJA namespace so USE_RAJA
#register_model(raja USE_RAJA fasten.hpp)
#register_model(tbb TBB fasten.hpp)
#register_model(thrust THRUST fasten.hpp) # TODO

set(USAGE ON CACHE BOOL "Whether to print all custom flags for the selected model")

message(STATUS "Available models:  ${REGISTERED_MODELS}")
if (NOT DEFINED MODEL)
    message(FATAL_ERROR "MODEL is unspecified, pick one from the available models")
else ()
    string(TOLOWER ${MODEL} MODEL)
    message(STATUS "Selected model  :  ${MODEL}")
endif ()

# load the $MODEL.cmake file and setup the correct IMPL_* based on $MODEL
load_model(${MODEL})

if (USAGE)
    # print the usage of the registered flag options
    registered_flags_action(print RESULT)
    message(STATUS "${RESULT}")
endif ()
# check required/set default for all registered flag options
registered_flags_action(check RESULT)
message(STATUS "${RESULT}")

list(APPEND IMPL_SOURCES
        driver/main.c
        driver/comms.c
        driver/chunk.c
        driver/shared.c
        driver/diffuse.c
        driver/profiler.c
        driver/settings.c
        driver/initialise.c
        driver/parse_config.c

        driver/cg_driver.c
        driver/ppcg_driver.c
        driver/cheby_driver.c
        driver/jacobi_driver.c
        driver/eigenvalue_driver.c
        driver/halo_update_driver.c
        driver/remote_halo_driver.c
        driver/store_energy_driver.c
        driver/field_summary_driver.c
        driver/set_chunk_data_driver.c
        driver/solve_finished_driver.c
        driver/set_chunk_state_driver.c
        driver/kernel_initialise_driver.c
        #
        )

# run model specific setup, i.e append build flags, etc
setup()

# CMake insists that -O2 (or equivalent) is the universally accepted optimisation level
# we remove that here and use our own <BUILD_TYPE>_FLAGS
if (CMAKE_CXX_FLAGS_${BUILD_TYPE})
    wipe_gcc_style_optimisation_flags(CMAKE_CXX_FLAGS_${BUILD_TYPE})
endif ()

message(STATUS "Default ${CMAKE_BUILD_TYPE} flags are `${DEFAULT_${BUILD_TYPE}_CXX_FLAGS}`, set ${BUILD_TYPE}_CXX_FLAGS to override (CXX_EXTRA_* flags are not affected)")


# setup common build flag defaults if there are no overrides
if (NOT DEFINED ${BUILD_TYPE}_CXX_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_CXX_FLAGS ${DEFAULT_${BUILD_TYPE}_CXX_FLAGS})
elseif ()
    set(ACTUAL_${BUILD_TYPE}_CXX_FLAGS ${${BUILD_TYPE}_CXX_FLAGS})
endif ()

if (NOT DEFINED ${BUILD_TYPE}_C_FLAGS)
    set(ACTUAL_${BUILD_TYPE}_C_FLAGS ${DEFAULT_${BUILD_TYPE}_C_FLAGS})
elseif ()
    set(ACTUAL_${BUILD_TYPE}_C_FLAGS ${${BUILD_TYPE}_C_FLAGS})
endif ()

if (NOT ENABLE_MPI)
    list(APPEND IMPL_DEFINITIONS NO_MPI)
else ()
    find_package(MPI REQUIRED)
    list(APPEND LINK_LIBRARIES MPI::MPI_C)
endif ()
if (ENABLE_PROFILING)
    list(APPEND IMPL_DEFINITIONS ENABLE_PROFILING)
endif ()

message(STATUS "CXX vendor  : ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "CC   vendor  :${CMAKE_C_COMPILER_ID} (${CMAKE_C_COMPILER})")
message(STATUS "Platform    : ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Sources     : ${IMPL_SOURCES}")
message(STATUS "Libraries   : ${LINK_LIBRARIES} ${CXX_EXTRA_LIBRARIES}")

message(STATUS "CXX Flags   : ${CMAKE_CXX_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_CXX_FLAGS} ${CXX_EXTRA_FLAGS}
    CXX flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_CXX_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_CXX_FLAGS}`
        Override (${BUILD_TYPE}_CXX_FLAGS) = `${${BUILD_TYPE}_CXX_FLAGS}`
        Extras (CXX_EXTRA_FLAGS) = `${CXX_EXTRA_FLAGS}`")
message(STATUS "CXX Link Flags  : ${LINK_FLAGS} ${CXX_EXTRA_LINK_FLAGS}")
message(STATUS "CXX Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${CXX_EXTRA_LINKER_FLAGS} ")

message(STATUS "C Flags   : ${CMAKE_C_FLAGS_${BUILD_TYPE}} ${ACTUAL_${BUILD_TYPE}_C_FLAGS} ${C_EXTRA_FLAGS}
    C flags derived from (CMake + (Override ? Override : Default) + Extras), where:
        CMake                    = `${CMAKE_C_FLAGS_${BUILD_TYPE}}`
        Default                  = `${DEFAULT_${BUILD_TYPE}_C_FLAGS}`
        Override (${BUILD_TYPE}_C_FLAGS) = `${${BUILD_TYPE}_C_FLAGS}`
        Extras (C_EXTRA_FLAGS) = `${C_EXTRA_FLAGS}`")
message(STATUS "C Link Flags  : ${LINK_FLAGS} ${C_EXTRA_LINK_FLAGS}")
message(STATUS "C Linker Flags: ${CMAKE_EXE_LINKER_FLAGS} ${C_EXTRA_LINKER_FLAGS} ")

message(STATUS "Defs        : ${IMPL_DEFINITIONS}")
message(STATUS "Executable  : ${BIN_NAME}")

# below we have all the usual CMake target setup steps


include_directories(${CMAKE_BINARY_DIR}/generated)


add_executable(${EXE_NAME} ${IMPL_SOURCES})
target_link_libraries(${EXE_NAME} PUBLIC ${LINK_LIBRARIES} m)
target_compile_definitions(${EXE_NAME} PUBLIC ${IMPL_DEFINITIONS} _POSIX_C_SOURCE=200809L)
target_include_directories(${EXE_NAME} PUBLIC driver)


if (CXX_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${CXX_EXTRA_LIBRARIES})
endif ()

if (C_EXTRA_LIBRARIES)
    target_link_libraries(${EXE_NAME} PUBLIC ${C_EXTRA_LIBRARIES})
endif ()


target_compile_options(${EXE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:${ACTUAL_RELEASE_CXX_FLAGS};${CXX_EXTRA_FLAGS}>>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:${ACTUAL_DEBUG_CXX_FLAGS};${CXX_EXTRA_FLAGS}>>")

target_compile_options(${EXE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:C>:$<$<CONFIG:Release>:${ACTUAL_RELEASE_C_FLAGS};${C_EXTRA_FLAGS}>>")
target_compile_options(${EXE_NAME} PUBLIC "$<$<COMPILE_LANGUAGE:C>:$<$<CONFIG:Debug>:${ACTUAL_DEBUG_C_FLAGS};${C_EXTRA_FLAGS}>>")

target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:LINKER:${CXX_EXTRA_LINKER_FLAGS}>)
target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${LINK_FLAGS};${CXX_EXTRA_LINK_FLAGS}>)

target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:C>:LINKER:${C_EXTRA_LINKER_FLAGS}>)
target_link_options(${EXE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:C>:${LINK_FLAGS};${C_EXTRA_LINK_FLAGS}>)


# some models require the target to be already specified so they can finish their setup here
# this only happens if the model.cmake definition contains the `setup_target` macro
if (COMMAND setup_target)
    setup_target(${EXE_NAME})
endif ()

target_compile_definitions(${EXE_NAME} PRIVATE)

#if ((CMAKE_GENERATOR MATCHES "Unix Makefiles") OR (CMAKE_GENERATOR MATCHES "Ninja"))
#    add_custom_target(extract_compile_commands ALL
#            COMMAND ${CMAKE_COMMAND}
#            -DCOMMAND_FILE=${CMAKE_BINARY_DIR}/compile_commands.json
#            -DPRE_CONFIGURE_FILE= "${CMAKE_SOURCE_DIR}/src/meta_build.h.in"
#            -DPOST_CONFIGURE_FILE= "${CMAKE_BINARY_DIR}/generated/meta_build.h"
#            -DSRC_DIR_PREFIX= "${CMAKE_SOURCE_DIR}/src"
#            -DOUT_DIR_PREFIX= "CMakeFiles/${EXE_NAME}.dir"
#            -P ${CMAKE_SOURCE_DIR}/cmake/extract_compile_command.cmake
#            )
#    add_dependencies(${EXE_NAME} extract_compile_commands)
#else ()
#    set(COMPILE_COMMANDS COMPILE_FLAGS)
#endif ()

set_target_properties(${EXE_NAME} PROPERTIES OUTPUT_NAME "${BIN_NAME}")

install(TARGETS ${EXE_NAME} DESTINATION bin)