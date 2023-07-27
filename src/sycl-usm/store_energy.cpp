#include "sycl_shared.hpp"

using namespace cl::sycl;

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
