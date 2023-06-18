#pragma once

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

typedef buffer<double, 1> SyclBuffer;

namespace utils {

// Used to manage copying the raw pointers
template <class T> static void packMirror(SyclBuffer &buffToPack, const T *buffer, int len, queue *device_queue) {
  device_queue
      ->submit([&](handler &h) { //
        h.copy(buffer, buffToPack.template get_access<access::mode::write>(h));
      })
      .wait_and_throw();
}

template <class T> static void unpackMirror(T *buffer, SyclBuffer &buffToUnpack, int len, queue *device_queue) {
  device_queue
      ->submit([&](handler &h) { //
        h.copy(buffToUnpack.template get_access<access::mode::read>(h), buffer);
      })
      .wait_and_throw();
}

} // namespace utils
