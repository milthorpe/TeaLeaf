#include "sycl_shared.hpp"

using namespace cl::sycl;

// Copies energy0 into energy1.
void store_energy(const int x,             //
                  const int y,             //
                  SyclBuffer &energyBuff,  //
                  SyclBuffer &energy0Buff, //
                  queue &device_queue) {
  device_queue.submit([&](handler &h) {
    auto energy = energyBuff.get_access<access::mode::write>(h);
    auto energy0 = energy0Buff.get_access<access::mode::read>(h);
    h.parallel_for<class store_energy>(range<1>(x * y), [=](id<1> idx) { energy[idx[0]] = energy0[idx[0]]; });
  });
#ifdef ENABLE_PROFILING
  device_queue.wait_and_throw();
#endif
}
