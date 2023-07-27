#include "settings.h"
#include "dpl_shim.h"
#include "ranged.h"
#include <algorithm>
#include <cmath>

/*
 *      SET CHUNK STATE KERNEL
 *		Sets up the chunk geometry.
 */

// Entry point for set chunk state kernel
void set_chunk_state(int x,                //
                     int y,                //
                     double *vertex_x,     //
                     double *vertex_y,     //
                     double *cell_x,       //
                     double *cell_y,       //
                     double *density,      //
                     double *energy0,      //
                     double *u,            //
                     const int num_states, //
                     State *states) {
  double default_energy = states[0].energy;
  double default_density = states[0].density;
  // Set the initial state
  ranged<int> it(0, x * y);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int index) {
    energy0[index] = default_energy;
    density[index] = default_density;
  });

  // Apply all of the states in turn
  for (int ss = 1; ss < num_states; ++ss) {
    State state = states[ss];
    ranged<int> it(0, x * y);
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int index) {
      const size_t kk = index % x;
      const size_t jj = index / x;

      bool applyState = false;

      if (state.geometry == RECTANGULAR) { // Rectangular state

        applyState = (vertex_x[kk + 1] >= state.x_min && vertex_x[kk] < state.x_max && //
                      vertex_y[jj + 1] >= state.y_min && vertex_y[jj] < state.y_max);
      } else if (state.geometry == CIRCULAR) { // Circular state

        double radius = std::sqrt((cell_x[kk] - state.x_min) * (cell_x[kk] - state.x_min) + //
                                  (cell_y[jj] - state.y_min) * (cell_y[jj] - state.y_min));

        applyState = (radius <= state.radius);
      } else if (state.geometry == POINT) // Point state
      {
        applyState = (vertex_x[kk] == state.x_min && vertex_y[jj] == state.y_min);
      }

      // Check if state applies at this vertex, and apply
      if (applyState) {
        energy0[index] = state.energy;
        density[index] = state.density;
      }

      if (kk > 0 && kk < x - 1 && jj > 0 && jj < y - 1) {
        u[index] = energy0[index] * density[index];
      }
    });
  }
}
