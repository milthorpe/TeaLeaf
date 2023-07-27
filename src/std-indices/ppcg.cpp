#include "shared.h"
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
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
    const int index = range.restore(i, x);
    sd[index] = r[index] / theta;
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

  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());

  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
    const int index = range.restore(i, x);
    const double smvp = tealeaf_SMVP(sd);
    r[index] -= smvp;
    u[index] += sd[index];
  });

  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
    const int index = range.restore(i, x);
    sd[index] = alpha * sd[index] + beta * r[index];
  });
}
