#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises Sd
void ppcg_init(const int x,          //
               const int y,          //
               const int halo_depth, //
               const double theta,   //
               SyclBuffer &sd,   //
               SyclBuffer &r,    //
               queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class ppcg_init>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        sd[idx[0]] = r[idx[0]] / theta;
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates U and R
void ppcg_calc_ur(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  SyclBuffer &sd,   //
                  SyclBuffer &r,    //
                  SyclBuffer &u,    //
                  SyclBuffer &kx,   //
                  SyclBuffer &ky,   //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class ppcg_calc_ur>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        // smvp uses kx and ky and index
        int index = idx[0];
        const double smvp = tealeaf_SMVP(sd);
        r[idx[0]] -= smvp;
        u[idx[0]] += sd[idx[0]];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates Sd
void ppcg_calc_sd(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  const double theta,   //
                  const double alpha,   //
                  const double beta,    //
                  SyclBuffer &sd,   //
                  SyclBuffer &r,    //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class ppcg_calc_sd>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        sd[idx[0]] = alpha * sd[idx[0]] + beta * r[idx[0]];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
