#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises the Chebyshev solver
void cheby_init(const int x,          //
                const int y,          //
                const int halo_depth, //
                const double theta,   //
                SyclBuffer &pBuff,    //
                SyclBuffer &rBuff,    //
                SyclBuffer &uBuff,    //
                SyclBuffer &u0Buff,   //
                SyclBuffer &wBuff,    //
                SyclBuffer &kxBuff,   //
                SyclBuffer &kyBuff,   //
                queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto p = pBuff.get_access<access::mode::write>(h);
    auto r = rBuff.get_access<access::mode::read_write>(h);
    auto u = uBuff.get_access<access::mode::read>(h);
    auto w = wBuff.get_access<access::mode::read_write>(h);
    auto u0 = u0Buff.get_access<access::mode::read>(h);
    auto kx = kxBuff.get_access<access::mode::read>(h);
    auto ky = kyBuff.get_access<access::mode::read>(h);
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
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}

// Calculates U
void cheby_calc_u(const int x,          //
                  const int y,          //
                  const int halo_depth, //
                  SyclBuffer &pBuff,    //
                  SyclBuffer &uBuff,    //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto p = pBuff.get_access<access::mode::read>(h);
    auto u = uBuff.get_access<access::mode::read_write>(h);
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
                   SyclBuffer &pBuff,    //
                   SyclBuffer &rBuff,    //
                   SyclBuffer &uBuff,    //
                   SyclBuffer &u0Buff,   //
                   SyclBuffer &wBuff,    //
                   SyclBuffer &kxBuff,   //
                   SyclBuffer &kyBuff,   //
                   queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto p = pBuff.get_access<access::mode::read_write>(h);
    auto r = rBuff.get_access<access::mode::read_write>(h);
    auto u = uBuff.get_access<access::mode::read>(h);
    auto u0 = u0Buff.get_access<access::mode::read>(h);
    auto w = wBuff.get_access<access::mode::read_write>(h);
    auto kx = kxBuff.get_access<access::mode::read>(h);
    auto ky = kyBuff.get_access<access::mode::read>(h);
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
