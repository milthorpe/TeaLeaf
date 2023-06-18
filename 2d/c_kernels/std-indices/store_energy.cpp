#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"

// Store original energy state
void store_energy(int x, int y, double *energy0, double *energy) {
  ranged<int> it(0, x * y);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](int ii) { energy[ii] = energy0[ii]; });
}
