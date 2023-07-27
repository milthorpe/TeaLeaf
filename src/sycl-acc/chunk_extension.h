#pragma once
// #include "sycl_shared.hpp"

#ifdef SYCL_LANGUAGE_VERSION
  #include <CL/sycl.hpp>
using namespace cl::sycl;

typedef buffer<double, 1> *FieldBufferType;

// Empty extension point
typedef struct ChunkExtension {
  FieldBufferType comms_buffer;

} ChunkExtension;
#else
typedef struct ChunkExtension {
  void *comms_buffer;
} ChunkExtension;
typedef void *FieldBufferType;
#endif
