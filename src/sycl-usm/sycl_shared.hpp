#pragma once

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

typedef double *SyclBuffer;

namespace utils {

// Used to manage copying the raw pointers
template <class T> static void packMirror(SyclBuffer &buffToPack, const T *buffer, int len, queue *device_queue) {
  device_queue->copy(buffer, buffToPack, len).wait_and_throw();
}

template <class T> static void unpackMirror(T *buffer, SyclBuffer &buffToUnpack, int len, queue *device_queue) {
  device_queue->copy(buffToUnpack, buffer, len).wait_and_throw();
}

} // namespace utils
