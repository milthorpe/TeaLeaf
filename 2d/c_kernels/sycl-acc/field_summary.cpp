#include "../../shared.h"
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
// buffer
// queue

void field_summary_func(const int x,             //
                        const int y,             //
                        const int halo_depth,    //
                        SyclBuffer &uBuff,       //
                        SyclBuffer &densityBuff, //
                        SyclBuffer &energy0Buff, //
                        SyclBuffer &volumeBuff,  //
                        double *vol,             //
                        double *mass,            //
                        double *ie,              //
                        double *temp,            //
                        queue &device_queue) {
  buffer<Summary, 1> summary_temp{range<1>{1}};
  device_queue.submit([&](handler &h) {
    auto u = uBuff.get_access<access::mode::read>(h);
    auto density = densityBuff.get_access<access::mode::read>(h);
    auto energy0 = energy0Buff.get_access<access::mode::read>(h);
    auto volume = volumeBuff.get_access<access::mode::read>(h);
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
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
  auto s = summary_temp.get_host_access()[0];
  *vol = s.vol;
  *mass = s.mass;
  *ie = s.ie;
  *temp = s.temp;
}
