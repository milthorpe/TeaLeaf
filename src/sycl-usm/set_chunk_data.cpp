#include "settings.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises the vertices
void set_chunk_data_vertices(const int x,          //
                             const int y,          //
                             const int halo_depth, //
                             SyclBuffer &vertex_x, //
                             SyclBuffer &vertex_y, //
                             const double x_min,   //
                             const double y_min,
                             const double dx, //
                             const double dy, //
                             queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class set_chunk_data_vertices>(range<1>(tealeaf_MAX(x, y) + 1), [=](id<1> idx) {
          if (idx[0] < x + 1) {
            vertex_x[idx[0]] = x_min + dx * double(idx[0] - halo_depth);
          }
          if (idx[0] < y + 1) {
            vertex_y[idx[0]] = y_min + dy * double(idx[0] - halo_depth);
          }
        });
      })
      .wait_and_throw();
}

// Sets all of the cell data for a chunk
void set_chunk_data(const int x,          //
                    const int y,          //
                    const int halo_depth, //
                    SyclBuffer &vertex_x, //
                    SyclBuffer &vertex_y, //
                    SyclBuffer &cell_x,   //
                    SyclBuffer &cell_y,   //
                    SyclBuffer &volume,   //
                    SyclBuffer &x_area,   //
                    SyclBuffer &y_area,   //
                    const double x_min,   //
                    const double y_min,   //
                    const double dx,      //
                    const double dy,      //
                    queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
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
      })
      .wait_and_throw();
}
