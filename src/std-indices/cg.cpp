#include <iostream>
#include <numeric>

#include "shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include "std_shared.h"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises the CG solver
void cg_init(const int x,           //
             const int y,           //
             const int halo_depth,  //
             const int coefficient, //
             double rx,             //
             double ry,             //
             double *rro,           //
             double *density,       //
             double *energy,        //
             double *u,             //
             double *p,             //
             double *r,             //
             double *w,             //
             double *kx,            //
             double *ky) {
  if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY) {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }

  {
    Range2d range(0, 0, x, y);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const int index = range.restore(i, x);
      p[index] = 0.0;
      r[index] = 0.0;
      u[index] = energy[index] * density[index];
    });
    //    ranged<int> it(0, y);
    //    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    //      for (int kk = 0; kk < x; ++kk) {
    //        const int index = kk + jj * x;
    //        p[index] = 0.0;
    //        r[index] = 0.0;
    //        u[index] = energy[index] * density[index];
    //      }
    //    });
  }

  {
    Range2d range(1, 1, x - 1, y - 1);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const int index = range.restore(i, x);
      w[index] = (coefficient == CONDUCTIVITY) ? density[index] : 1.0 / density[index];
    });
    //    ranged<int> it(1, y - 1);
    //    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    //      for (int kk = 1; kk < x - 1; ++kk) {
    //        const int index = kk + jj * x;
    //        w[index] = (coefficient == CONDUCTIVITY) ? density[index] : 1.0 / density[index];
    //      }
    //    });
  }

  {
    Range2d range(halo_depth, halo_depth, x - 1, y - 1);
    ranged<int> it(0, range.sizeXY());
    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
      const int index = range.restore(i, x);
      kx[index] = rx * (w[index - 1] + w[index]) / (2.0 * w[index - 1] * w[index]);
      ky[index] = ry * (w[index - x] + w[index]) / (2.0 * w[index - x] * w[index]);
    });
    //    ranged<int> it(halo_depth, y - 1);
    //    std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
    //      for (int kk = halo_depth; kk < x - 1; ++kk) {
    //        const int index = kk + jj * x;
    //        kx[index] = rx * (w[index - 1] + w[index]) / (2.0 * w[index - 1] * w[index]);
    //        ky[index] = ry * (w[index - x] + w[index]) / (2.0 * w[index - x] * w[index]);
    //      }
    //    });
  }

  {
    Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
    ranged<int> it(0, range.sizeXY());
    *rro += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int i) {
      const int index = range.restore(i, x);
      const double smvp = tealeaf_SMVP(u);
      w[index] = smvp;
      r[index] = u[index] - w[index];
      p[index] = r[index];
      return r[index] * p[index];
    });
    //    Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
    //    ranged<int> it(0, range.sizeXY());
    //    *rro += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int i) {
    //      const int jj = (i / range.sizeX()) + range.fromX;
    //      const int kk = (i % range.sizeX()) + range.fromY;
    //      const int index = kk + jj * x;
    //      const double smvp = tealeaf_SMVP(u);
    //      w[index] = smvp;
    //      r[index] = u[index] - w[index];
    //      p[index] = r[index];
    //      return r[index] * p[index];
    //    });
  }
}

// Calculates w
void cg_calc_w(const int x,          //
               const int y,          //
               const int halo_depth, //
               double *pw,           //
               double *p,            //
               double *w,            //
               double *kx,           //
               double *ky) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());
  *pw += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int i) {
    const int index = range.restore(i, x);
    const double smvp = tealeaf_SMVP(p);
    w[index] = smvp;
    return w[index] * p[index];
  });
  //  ranged<int> it(halo_depth, y - halo_depth);
  //  *pw += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int jj) {
  //    double pw_temp = 0.0;
  //    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
  //      const int index = kk + jj * x;
  //      const double smvp = tealeaf_SMVP(p);
  //      w[index] = smvp;
  //      pw_temp += w[index] * p[index];
  //    }
  //    return pw_temp;
  //  });
}

// Calculates u and r
void cg_calc_ur(const int x,          //
                const int y,          //
                const int halo_depth, //
                const double alpha,   //
                double *rrn,          //
                double *u,            //
                double *p,            //
                double *r,            //
                double *w) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());
  *rrn += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int i) {
    const int index = range.restore(i, x);
    u[index] += alpha * p[index];
    r[index] -= alpha * w[index];
    return r[index] * r[index];
  });
  //  ranged<int> it(halo_depth, y - halo_depth);
  //  *rrn += std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), 0.0, std::plus<>(), [=](int jj) {
  //    double rrn_temp = 0.0;
  //    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
  //      const int index = kk + jj * x;
  //      u[index] += alpha * p[index];
  //      r[index] -= alpha * w[index];
  //      rrn_temp += r[index] * r[index];
  //    }
  //    return rrn_temp;
  //  });
}

// Calculates p
void cg_calc_p(const int x,          //
               const int y,          //
               const int halo_depth, //
               const double beta,    //
               double *p,            //
               double *r) {
  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int i) {
    const int index = range.restore(i, x);
    p[index] = beta * p[index] + r[index];
  });
  //  ranged<int> it(halo_depth, y - halo_depth);
  //  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int jj) {
  //    for (int kk = halo_depth; kk < x - halo_depth; ++kk) {
  //      const int index = kk + jj * x;
  //      p[index] = beta * p[index] + r[index];
  //    }
  //  });
}
