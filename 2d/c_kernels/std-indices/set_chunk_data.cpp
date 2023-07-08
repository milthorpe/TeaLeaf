#include "../../settings.h"
#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"

/*
 * 		SET CHUNK DATA KERNEL
 * 		Initialises the chunk's mesh data.
 */
// Initialises the vertices
void set_chunk_data_vertices(const int x,          //
                             const int y,          //
                             const int halo_depth, //
                             double *vertex_x,     //
                             double *vertex_y,     //
                             const double x_min,   //
                             const double y_min,   //
                             const double dx,      //
                             const double dy) {

  ranged<int> it(0, tealeaf_MAX(x, y) + 1);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    if (index < x + 1) {
      vertex_x[index] = x_min + dx * (index - halo_depth);
    }

    if (index < y + 1) {
      vertex_y[index] = y_min + dy * (index - halo_depth);
    }
  });
}

// Extended kernel for the chunk initialisation
void set_chunk_data(Settings *settings, //
                    int x,              //
                    int y,              //
                    int left,           //
                    int bottom,         //
                    double *cell_x,     //
                    double *cell_y,     //
                    double *vertex_x,   //
                    double *vertex_y,   //
                    double *volume,     //
                    double *x_area,     //
                    double *y_area) {
  ranged<int> it(0, x * y);
  double dx = settings->dx;
  double dy = settings->dy;
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    if (index < x) {
      cell_x[index] = 0.5 * (vertex_x[index] + vertex_x[index + 1]);
    }

    if (index < y) {
      cell_y[index] = 0.5 * (vertex_y[index] + vertex_y[index + 1]);
    }
    if (index < x * y) {
      volume[index] = dx * dy;
      x_area[index] = dy;
      y_area[index] = dx;
    }
  });
}
