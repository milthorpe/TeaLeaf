#include "../../shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

// Copies the inner u into u0.
void copy_u(const int x,          //
            const int y,          //
            const int halo_depth, //
            SyclBuffer &u,        //
            SyclBuffer &u0,       //
            queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class copy_u>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            u0[idx[0]] = u[idx[0]];
          }
        });
      })
      .wait_and_throw();
}

// Calculates the residual r.
void calculate_residual(const int x,          //
                        const int y,          //
                        const int halo_depth, //
                        SyclBuffer &u,        //
                        SyclBuffer &u0,       //
                        SyclBuffer &r,        //
                        SyclBuffer &kx,       //
                        SyclBuffer &ky,       //
                        queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class calculate_residual>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            // smvp uses kx and ky and index
            int index = idx[0];
            const double smvp = tealeaf_SMVP(u);
            r[idx[0]] = u0[idx[0]] - smvp;
          }
        });
      })
      .wait_and_throw();
}

// Calculates the 2 norm of the provided buffer.
void calculate_2norm(const int x,          //
                     const int y,          //
                     const int halo_depth, //
                     SyclBuffer &b,        //
                     double *norm,         //
                     queue &device_queue) {
  buffer<double, 1> norm_temp{range<1>{1}};
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class calculate_2norm>(
            range<1>(x * y), sycl::reduction(norm_temp, h, {}, sycl::plus<>(), sycl::property::reduction::initialize_to_identity()), //
            [=](item<1> item, auto &acc) {
              const auto kk = item[0] % x;
              const auto jj = item[0] / x;
              if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
                acc += b[item[0]] * b[item[0]];
              }
            });
      })
      .wait_and_throw();
  *norm = norm_temp.get_host_access()[0];
}

// Finalises the energy field.
void finalise(const int x,          //
              const int y,          //
              const int halo_depth, //
              SyclBuffer &u,        //
              SyclBuffer &density,  //
              SyclBuffer &energy,   //
              queue &device_queue) {
  device_queue
      .submit([&](handler &h) {
        h.parallel_for<class finalise>(range<1>(x * y), [=](id<1> idx) {
          const auto kk = idx[0] % x;
          const auto jj = idx[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            energy[idx[0]] = u[idx[0]] / density[idx[0]];
          }
        });
      })
      .wait_and_throw();
}
