#include "chunk.h"
#include "shared.h"
#include "sycl_shared.hpp"

using namespace cl::sycl;

struct Summary {
  double vol;
  double mass;
  double ie;
  double temp;
  [[nodiscard]] constexpr Summary operator+(const Summary &that) const { //
    return {vol + that.vol, mass + that.mass, ie + that.ie, temp + that.temp};
  }
};

void field_summary_func(const int x,          //
                        const int y,          //
                        const int halo_depth, //
                        SyclBuffer &u,        //
                        SyclBuffer &density,  //
                        SyclBuffer &energy0,  //
                        SyclBuffer &volume,   //
                        double *vol,          //
                        double *mass,         //
                        double *ie,           //
                        double *temp,         //
                        queue &device_queue) {
  buffer<Summary, 1> summary_temp{range<1>{1}};
  device_queue.submit([&](handler &h) {
    h.parallel_for<class field_summary_func>(
        range<1>(x * y),                                                                                           //
        sycl::reduction(summary_temp, h, {}, sycl::plus<>(), sycl::property::reduction::initialize_to_identity()), //
        [=](item<1> item, auto &acc) {
          const auto kk = item[0] % x;
          const auto jj = item[0] / x;
          if (kk >= halo_depth && kk < x - halo_depth && jj >= halo_depth && jj < y - halo_depth) {
            const double cellVol = volume[item[0]];
            const double cellMass = cellVol * density[item[0]];
            acc += Summary{
                cellVol,
                cellMass,
                cellMass * energy0[item[0]],
                cellMass * u[item[0]],
            };
          }
        });
  });
  device_queue.wait_and_throw();
  auto s = summary_temp.get_host_access()[0];
  *vol = s.vol;
  *mass = s.mass;
  *ie = s.ie;
  *temp = s.temp;
}

// Copies energy0 into energy1.
void store_energy(const int x,         //
                  const int y,         //
                  SyclBuffer &energy,  //
                  SyclBuffer &energy0, //
                  queue &device_queue) {
  device_queue
      .submit(
          [&](handler &h) { h.parallel_for<class store_energy>(range<1>(x * y), [=](id<1> idx) { energy[idx[0]] = energy0[idx[0]]; }); })
      .wait_and_throw();
}

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

void run_store_energy(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);

  store_energy(chunk->x, chunk->y, (chunk->energy), (chunk->energy0), *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_field_summary(Chunk *chunk, Settings &settings, double *vol, double *mass, double *ie, double *temp) {
  START_PROFILING(settings.kernel_profile);
  field_summary_func(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->density), (chunk->energy0), (chunk->volume), vol, mass,
                     ie, temp, *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

// Shared solver kernels
void run_copy_u(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  copy_u(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->u0), *(chunk->ext->device_queue));
  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_residual(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  calculate_residual(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->u0), (chunk->r), (chunk->kx), (chunk->ky),
                     *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_calculate_2norm(Chunk *chunk, Settings &settings, SyclBuffer buffer, double *norm) {
  START_PROFILING(settings.kernel_profile);
  calculate_2norm(chunk->x, chunk->y, settings.halo_depth, (buffer), norm, *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}

void run_finalise(Chunk *chunk, Settings &settings) {
  START_PROFILING(settings.kernel_profile);
  finalise(chunk->x, chunk->y, settings.halo_depth, (chunk->u), (chunk->density), (chunk->energy), *(chunk->ext->device_queue));

  STOP_PROFILING(settings.kernel_profile, __func__);
}
