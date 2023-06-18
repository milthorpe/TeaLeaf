#include "../../settings.h"
#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Sets the initial state for the chunk
void set_chunk_initial_state(const int x,            //
                             const int y,            //
                             double default_energy,  //
                             double default_density, //
                             SyclBuffer &energy0,    //
                             SyclBuffer &density,    //
                             queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class set_chunk_initial_state>(range<1>(x * y), [=](id<1> idx) {
          energy0[idx[0]] = default_energy;
          density[idx[0]] = default_density;
        });
      })
      .wait_and_throw();
}

// Sets all of the additional states in order
void set_chunk_state(const int x,          //
                     const int y,          //
                     const int halo_depth, //
                     State state,          //
                     SyclBuffer &energy0,  //
                     SyclBuffer &density,  //
                     SyclBuffer &u,        //
                     SyclBuffer &cell_x,   //
                     SyclBuffer &cell_y,   //
                     SyclBuffer &vertex_x, //
                     SyclBuffer &vertex_y, //
                     queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class set_chunk_state>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          bool applyState = false;

          if (state.geometry == RECTANGULAR) // Rectangular state
          {
            applyState = (vertex_x[kk + 1] >= state.x_min && vertex_x[kk] < state.x_max && vertex_y[jj + 1] >= state.y_min &&
                          vertex_y[jj] < state.y_max);
          } else if (state.geometry == CIRCULAR) // Circular state
          {
            double radius = cl::sycl::sqrt((cell_x[kk] - state.x_min) * (cell_x[kk] - state.x_min) +
                                           (cell_y[jj] - state.y_min) * (cell_y[jj] - state.y_min));

            applyState = (radius <= state.radius);
          } else if (state.geometry == POINT) // Point state
          {
            applyState = (vertex_x[kk] == state.x_min && vertex_y[jj] == state.y_min);
          }

          // Check if state applies at this vertex, and apply
          if (applyState) {
            energy0[idx[0]] = state.energy;
            density[idx[0]] = state.density;
          }

          if (kk > 0 && kk < x - 1 && jj > 0 && jj < y - 1) {
            u[idx[0]] = energy0[idx[0]] * density[idx[0]];
          }
        });
      })
      .wait_and_throw();
}
