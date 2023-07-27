#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises the Chebyshev solver
void cheby_init(const int x,          //
                const int y,          //
                const int halo_depth, //
                const double theta,   //
                SyclBuffer &p,        //
                SyclBuffer &r,        //
                SyclBuffer &u,        //
                SyclBuffer &u0,       //
                SyclBuffer &w,        //
                SyclBuffer &kx,       //
                SyclBuffer &ky,       //
                queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cheby_init>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            // smvp uses kx and ky and index
            int index = idx[0];
            const double smvp = tealeaf_SMVP(u);
            w[idx[0]] = smvp;
            // could make w write only and then use smvp here
            r[idx[0]] = u0[idx[0]] - w[idx[0]];
            p[idx[0]] = r[idx[0]] / theta;
          }
        });
      })
      .wait_and_throw();
}

// Calculates U
void cheby_calc_u(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  SyclBuffer &p,        //
                  SyclBuffer &u,        //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class cheby_calc_u>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        u[idx[0]] += p[idx[0]];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// The main Cheby iteration step
void cheby_iterate(const int x,          //
                   const int y,          //
                   const int halo_depth, //
                   const double alpha,   //
                   const double beta,    //
                   SyclBuffer &p,        //
                   SyclBuffer &r,        //
                   SyclBuffer &u,        //
                   SyclBuffer &u0,       //
                   SyclBuffer &w,        //
                   SyclBuffer &kx,       //
                   SyclBuffer &ky,       //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class cheby_iterate>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        // smvp uses kx and ky and index
        int index = idx[0];
        const double smvp = tealeaf_SMVP(u);
        w[index] = smvp;
        // could make w write only and then use smvp here
        r[index] = u0[index] - w[index];
        p[index] = alpha * p[index] + beta * r[index];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
