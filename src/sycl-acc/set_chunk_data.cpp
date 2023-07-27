#include "settings.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises the vertices
void set_chunk_data_vertices(const int x,              //
                             const int y,              //
                             const int halo_depth,     //
                             SyclBuffer &vertex_xBuff, //
                             SyclBuffer &vertex_yBuff, //
                             const double x_min,       //
                             const double y_min,
                             const double dx, //
                             const double dy, //
                             queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto vertex_x = vertex_xBuff.get_access<access::mode::discard_write>(h);
    auto vertex_y = vertex_yBuff.get_access<access::mode::discard_write>(h);
    h.parallel_for<class set_chunk_data_vertices>(range<1>(tealeaf_MAX(x, y) + 1), [=](id<1> idx) {
      if (idx[0] < x + 1) {
        vertex_x[idx[0]] = x_min + dx * double(idx[0] - halo_depth);
      }
      if (idx[0] < y + 1) {
        vertex_y[idx[0]] = y_min + dy * double(idx[0] - halo_depth);
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Sets all of the cell data for a chunk
void set_chunk_data(const int x,              //
                    const int y,              //
                    const int halo_depth,     //
                    SyclBuffer &vertex_xBuff, //
                    SyclBuffer &vertex_yBuff, //
                    SyclBuffer &cell_xBuff,   //
                    SyclBuffer &cell_yBuff,   //
                    SyclBuffer &volumeBuff,   //
                    SyclBuffer &x_areaBuff,   //
                    SyclBuffer &y_areaBuff,   //
                    const double x_min,       //
                    const double y_min,       //
                    const double dx,          //
                    const double dy,          //
                    queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto vertex_x = vertex_xBuff.get_access<access::mode::read>(h);
    auto vertex_y = vertex_yBuff.get_access<access::mode::read>(h);
    auto volume = volumeBuff.get_access<access::mode::write>(h);
    auto x_area = x_areaBuff.get_access<access::mode::write>(h);
    auto y_area = y_areaBuff.get_access<access::mode::write>(h);
    auto cell_y = cell_yBuff.get_access<access::mode::write>(h);
    auto cell_x = cell_xBuff.get_access<access::mode::write>(h);

    h.parallel_for<class set_chunk_data>(range<1>(x * y), [=](id<1> idx) {
      if (idx[0] < x) {
        cell_x[idx[0]] = 0.5 * (vertex_x[idx[0]] + vertex_x[idx[0] + 1]);
      }
      if (idx[0] < y) {
        cell_y[idx[0]] = 0.5 * (vertex_y[idx[0]] + vertex_y[idx[0] + 1]);
      }

      if (idx[0] < x * y) {
        volume[idx[0]] = dx * dy;
        x_area[idx[0]] = dy;
        y_area[idx[0]] = dx;
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
