#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include <algorithm>
#include <cstdlib>

/*
 *		SHARED SOLVER METHODS
 */

// Copies the current u into u0
void copy_u(const int x,          //
            const int y,          //
            const int halo_depth, //
            double *u0,           //
            double *u) {
  ranged<int> it(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      u0[index] = u[index];
    }
  });
}

// Calculates the current value of r
void calculate_residual(const int x,          //
                        const int y,          //
                        const int halo_depth, //
                        double *u,            //
                        double *u0,           //
                        double *r,            //
                        double *kx,           //
                        double *ky) {
  ranged<int> it(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      const double smvp = tealeaf_SMVP(u);
      r[index] = u0[index] - smvp;
    }
  });
}

// Calculates the 2 norm of a given buffer
void calculate_2norm(const int x,          //
                     const int y,          //
                     const int halo_depth, //
                     double *buffer,       //
                     double *norm) {
  ranged<int> it(halo_depth, y - halo_depth);
  *norm += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int jj) {
    double norm_temp = 0.0;
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      norm_temp += buffer[index] * buffer[index];
    }
    return norm_temp;
  });
}

// Finalises the solution
void finalise(const int x,          //
              const int y,          //
              const int halo_depth, //
              double *energy,       //
              double *density,      //
              double *u) {
  ranged<int> it(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      energy[index] = u[index] / density[index];
    }
  });
}
