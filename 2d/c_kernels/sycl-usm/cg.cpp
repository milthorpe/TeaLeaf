#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Initialises p,r,u,w
void cg_init_u(const int x,           //
               const int y,           //
               const int coefficient, //
               SyclBuffer &p,         //
               SyclBuffer &r,         //
               SyclBuffer &u,         //
               SyclBuffer &w,         //
               SyclBuffer &density,   //
               SyclBuffer &energy, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_init_u>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          p[idx[0]] = 0.0;
          r[idx[0]] = 0.0;
          u[idx[0]] = energy[idx[0]] * density[idx[0]];
          if (jj > 0 && jj < y - 1 && kk > 0 & kk < x - 1) {
            w[idx[0]] = (coefficient == CONDUCTIVITY) ? density[idx[0]] : 1.0 / density[idx[0]];
          }
        });
      })
      .wait_and_throw();
}

// Initialises kx,ky
void cg_init_k(const int x,          //
               const int y,          //
               const int halo_depth, //
               SyclBuffer &w,        //
               SyclBuffer &kx,       //
               SyclBuffer &ky,       //
               const double rx,      //
               const double ry, queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_init_k>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (jj >= halo_depth && jj < y - 1 && kk >= halo_depth && kk < x - 1) {
            kx[idx[0]] = rx * (w[idx[0] - 1] + w[idx[0]]) / (2.0 * w[idx[0] - 1] * w[idx[0]]);
            ky[idx[0]] = ry * (w[idx[0] - x] + w[idx[0]]) / (2.0 * w[idx[0] - x] * w[idx[0]]);
          }
        });
      })
      .wait_and_throw();
}

// Initialises w,r,p and calculates rro
void cg_init_others(const int x,          //
                    const int y,          //
                    const int halo_depth, //
                    SyclBuffer &kx,       //
                    SyclBuffer &ky,       //
                    SyclBuffer &p,        //
                    SyclBuffer &r,        //
                    SyclBuffer &u,        //
                    SyclBuffer &w,        //
                    double *rro,          //
                    queue &device_queue) {

  buffer<double, 1> rro_temp{range<1>{1}};

  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class cg_init_others>(
            range<1>(x * y),                                                                                       //
            sycl::reduction(rro_temp, h, {}, sycl::plus<>(), sycl::property::reduction::initialize_to_identity()), //
            [=](item<1> item, auto &acc) {
              const auto kk = item[0] % x;
              const auto jj = item[0] / x;
              if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
                auto index = item[0]; // smvp uses kx and ky and index
                const double smvp = tealeaf_SMVP(u);
                w[item[0]] = smvp;
                r[item[0]] = u[item[0]] - w[item[0]];
                p[item[0]] = r[item[0]];
                acc += r[item[0]] * p[item[0]];
              }
            });
      })
      .wait_and_throw();
  *rro += rro_temp.get_host_access()[0];
}

// Calculates the value for w
void cg_calc_w(const int x,          //
               const int y,          //
               const int halo_depth, //
               SyclBuffer &w,        //
               SyclBuffer &p,        //
               SyclBuffer &kx,       //
               SyclBuffer &ky,       //
               double *pw,           //
               queue &device_queue) {
  buffer<double, 1> pw_temp{range<1>{1}};
  device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_calc_w>(range<1>(x * y),                                                                                      //
                                    sycl::reduction(pw_temp, h, {}, sycl::plus<>(), sycl::property::reduction::initialize_to_identity()), //
                                    [=](item<1> item, auto &acc) {
                                      const auto kk = item[0] % x;
                                      const auto jj = item[0] / x;
                                      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
                                        // smvp uses kx and ky and index
                                        int index = item[0];
                                        const double smvp = tealeaf_SMVP(p);
                                        w[item[0]] = smvp;
                                        acc += w[item[0]] * p[item[0]];
                                      }
                                    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
  *pw += pw_temp.get_host_access()[0];
}

// Calculates the value of u and r
void cg_calc_ur(const int x,          //
                const int y,          //
                const int halo_depth, //
                SyclBuffer &u,        //
                SyclBuffer &r,        //
                SyclBuffer &p,        //
                SyclBuffer &w,        //
                const double alpha,   //
                double *rrn,          //
                queue &device_queue) {

  buffer<double, 1> rrn_temp{range<1>{1}};
  device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_calc_ur>(
        range<1>(x * y),                                                                                       //
        sycl::reduction(rrn_temp, h, {}, sycl::plus<>(), sycl::property::reduction::initialize_to_identity()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            u[item[0]] += alpha * p[item[0]];
            r[item[0]] -= alpha * w[item[0]];
            acc += r[item[0]] * r[item[0]];
          }
        });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
  *rrn += rrn_temp.get_host_access()[0];
}

// Calculates a value for p
void cg_calc_p(const int x,          //
               const int y,          //
               const int halo_depth, //
               const double beta,    //
               SyclBuffer &p,        //
               SyclBuffer &r,        //
               queue &device_queue) {
  device_queue.submit([&](handler &h) {
    h.parallel_for<class cg_calc_p>(range<1>(x * y), [=](id<1> idx) {
      const auto kk = idx[0] % x;
      const auto jj = idx[0] / x;
      if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
        p[idx[0]] = beta * p[idx[0]] + r[idx[0]];
      }
    });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
