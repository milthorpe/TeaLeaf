#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include "std_shared.h"
/*
 *		PPCG SOLVER KERNEL
 */

// Initialises the PPCG solver
void ppcg_init(const int x,          //
               const int y,          //
               const int halo_depth, //
               double theta,         //
               double *r,            //
               double *sd) {
  ranged<int> it(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      sd[index] = r[index] / theta;
    }
  });
}

// The PPCG inner iteration
void ppcg_inner_iteration(const int x,          //
                          const int y,          //
                          const int halo_depth, //
                          double alpha,         //
                          double beta,          //
                          double *u,            //
                          double *r,            //
                          double *kx,           //
                          double *ky,           //
                          double *sd) {
  ranged<int> it1(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it1.begin(), it1.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      const double smvp = tealeaf_SMVP(sd);
      r[index] -= smvp;
      u[index] += sd[index];
    }
  });

  ranged<int> it2(halo_depth, y - halo_depth);
  std::for_each(EXEC_POLICY, it2.begin(), it2.end(), [=](int jj) {
    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
      const int index = kk + jj * x;
      sd[index] = alpha * sd[index] + beta * r[index];
    }
  });
}
