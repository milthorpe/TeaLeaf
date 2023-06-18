#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Updates the local left halo region(s)
void update_left(const int x,            //
                 const int y,            //
                 const int halo_depth,   //
                 SyclBuffer &bufferBuff, //
                 const int face,         //
                 const int depth,        //
                 queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_left>(range<1>(y * depth), [=](id<1> idx) {
      const auto flip = idx[0] % depth;
      const auto lines = idx[0] / depth;
      const auto offset = lines * (x - depth);
      const auto to_index = offset + halo_depth - depth + idx[0];
      const auto from_index = to_index + 2 * (depth - flip) - 1;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local right halo region(s)
void update_right(const int x,            //
                  const int y,            //
                  const int halo_depth,   //
                  SyclBuffer &bufferBuff, //
                  const int face,         //
                  const int depth,        //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_right>(range<1>(y * depth), [=](id<1> idx) {
      const auto flip = idx[0] % depth;
      const auto lines = idx[0] / depth;
      const auto offset = x - halo_depth + lines * (x - depth);
      const auto to_index = offset + idx[0];
      const auto from_index = to_index - (1 + flip * 2);
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local top halo region(s)
void update_top(const int x,            //
                const int y,            //
                const int halo_depth,   //
                SyclBuffer &bufferBuff, //
                const int face,         //
                const int depth,        //
                queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_top>(range<1>(x * depth), [=](id<1> idx) {
      const auto lines = idx[0] / x;
      const auto offset = x * (y - halo_depth);
      const auto to_index = offset + idx[0];
      const auto from_index = to_index - (1 + lines * 2) * x;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Updates the local bottom halo region(s)
void update_bottom(const int x,            //
                   const int y,            //
                   const int halo_depth,   //
                   SyclBuffer &bufferBuff, //
                   const int face,         //
                   const int depth,        //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto buffer = bufferBuff.get_access<access::mode::read_write>(h);
    h.parallel_for<class update_bottom>(range<1>(x * depth), [=](id<1> idx) {
      const auto lines = idx[0] / x;
      const auto offset = x * halo_depth;
      const auto from_index = offset + idx[0];
      const auto to_index = from_index - (1 + lines * 2) * x;
      buffer[to_index] = buffer[from_index];
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
