#pragma once

#include <CL/sycl.hpp>

using namespace cl;

using FieldBufferType = double *;
using StagingBufferType = double *;

struct ChunkExtension {
  sycl::queue *device_queue;
};
