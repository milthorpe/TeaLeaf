#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include "std_shared.h"
#include <numeric>

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */

struct Summary {
  double vol;
  double mass;
  double ie;
  double temp;
  [[nodiscard]] constexpr Summary operator+(const Summary &that) const { //
    return {vol + that.vol, mass + that.mass, ie + that.ie, temp + that.temp};
  }
};

// The field summary kernel
void field_summary(const int x,          //
                   const int y,          //
                   const int halo_depth, //
                   double *volume,       //
                   double *density,      //
                   double *energy0,      //
                   double *u,            //
                   double *volOut,       //
                   double *massOut,      //
                   double *ieOut,        //
                   double *tempOut) {

  Range2d range(halo_depth, halo_depth, x - halo_depth, y - halo_depth);
  ranged<int> it(0, range.sizeXY());
  auto summary = std::transform_reduce(EXEC_POLICY, it.begin(), it.end(), Summary{}, std::plus<>(), [=](int i) {
    const int index = range.restore(i, x);
    const double cellVol = volume[index];
    const double cellMass = cellVol * density[index];
    return Summary{.vol = cellVol, .mass = cellMass, .ie = cellMass * energy0[index], .temp = cellMass * u[index]};
  });

  *volOut += summary.vol;
  *ieOut += summary.ie;
  *tempOut += summary.temp;
  *massOut += summary.mass;
}
