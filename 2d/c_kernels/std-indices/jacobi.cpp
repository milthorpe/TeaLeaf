#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include <cmath>

/*
 *		JACOBI SOLVER KERNEL
 */

// Initialises the Jacobi solver
void jacobi_init(const int x,           //
                 const int y,           //
                 const int halo_depth,  //
                 const int coefficient, //
                 double rx,             //
                 double ry,             //
                 double *density,       //
                 double *energy,        //
                 double *u0,            //
                 double *u,             //
                 double *kx,            //
                 double *ky) {
  if (coefficient < CONDUCTIVITY && coefficient < RECIP_CONDUCTIVITY) {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }
  ranged<int> it1(1, y - 1);
  std::for_each(EXEC_POLICY, it1.begin(), it1.end(), [=](int jj) {
    for (int kk = 1; kk < x - 1; ++kk) {
      const int index = kk + jj * x;
      double temp = energy[index] * density[index];
      u0[index] = temp;
      u[index] = temp;
    }
  });

  ranged<int> it2(halo_depth, y - 1);
  std::for_each(EXEC_POLICY, it2.begin(), it2.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - 1; ++kk) {
      const int index = kk + jj * x;
      double densityCentre = (coefficient == CONDUCTIVITY) ? density[index] : 1.0 / density[index];
      double densityLeft = (coefficient == CONDUCTIVITY) ? density[index - 1] : 1.0 / density[index - 1];
      double densityDown = (coefficient == CONDUCTIVITY) ? density[index - x] : 1.0 / density[index - x];

      kx[index] = rx * (densityLeft + densityCentre) / (2.0 * densityLeft * densityCentre);
      ky[index] = ry * (densityDown + densityCentre) / (2.0 * densityDown * densityCentre);
    }
  });
}

// The main Jacobi solve step
void jacobi_iterate(const int x,          //
                    const int y,          //
                    const int halo_depth, //
                    double *error,        //
                    double *kx,           //
                    double *ky,           //
                    double *u0,           //
                    double *u,            //
                    double *r) {

  ranged<int> it1(0, y);
  std::for_each(EXEC_POLICY, it1.begin(), it1.end(), [=](int jj) {
    for (int kk = 0; kk < x; ++kk) {
      const int index = kk + jj * x;
      r[index] = u[index];
    }
  });

  ranged<int> it(halo_depth, y - halo_depth);
  *error = std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int jj) {
    double err = 0.0;
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      u[index] = (u0[index] + (kx[index + 1] * r[index + 1] + kx[index] * r[index - 1]) +
                  (ky[index + x] * r[index + x] + ky[index] * r[index - x])) /
                 (1.0 + (kx[index] + kx[index + 1]) + (ky[index] + ky[index + x]));

      err += fabs(u[index] - r[index]);
    }
    return err;
  });
}
