
register_flag_required(CMAKE_CXX_COMPILER
        "Absolute path to the AMD HIP C++ compiler")

macro(setup)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endmacro()