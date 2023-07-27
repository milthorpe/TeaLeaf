#include "settings.h"
#include "shared.h"
#include "kokkos_shared.hpp"
#include <stdlib.h>

// Allocates, and zeroes an individual buffer
void allocate_buffer(double **a, int x, int y) {
  *a = (double *)malloc(sizeof(double) * x * y);

  if (*a == nullptr) {
    die(__LINE__, __FILE__, "Error allocating buffer %s\n");
  }

#pragma omp parallel for
  for (int jj = 0; jj < y; ++jj) {
    for (int kk = 0; kk < x; ++kk) {
      const int index = kk + jj * x;
      (*a)[index] = 0.0;
    }
  }
}

// Allocates all of the field buffers
void kernel_initialise(Settings *settings, int x, int y, KView **density0, KView **density, KView **energy0, KView **energy, KView **u,
                       KView **u0, KView **p, KView **r, KView **mi, KView **w, KView **kx, KView **ky, KView **sd, KView **volume,
                       KView **x_area, KView **y_area, KView **cell_x, KView **cell_y, KView **cell_dx, KView **cell_dy, KView **vertex_dx,
                       KView **vertex_dy, KView **vertex_x, KView **vertex_y, KView **comms_buffer,
                       Kokkos::View<double *>::HostMirror **host_comms_mirror, double **cg_alphas, double **cg_betas, double **cheby_alphas,
                       double **cheby_betas) {
  print_and_log(settings, "Performing this solve with the Kokkos %s solver\n", settings->solver_name);

  Kokkos::initialize();

  *density0 = new KView("density0", x * y);
  *density = new KView("density", x * y);
  *energy0 = new KView("energy0", x * y);
  *energy = new KView("energy", x * y);
  *u = new KView("u", x * y);
  *u0 = new KView("u0", x * y);
  *p = new KView("p", x * y);
  *r = new KView("r", x * y);
  *mi = new KView("mi", x * y);
  *w = new KView("w", x * y);
  *kx = new KView("kx", x * y);
  *ky = new KView("ky", x * y);
  *sd = new KView("sd", x * y);
  *volume = new KView("volume", x * y);
  *x_area = new KView("x_area", (x + 1) * y);
  *y_area = new KView("y_area", x * (y + 1));
  *cell_x = new KView("cell_x", x);
  *cell_y = new KView("cell_y", y);
  *cell_dx = new KView("cell_dx", x);
  *cell_dy = new KView("cell_dy", y);
  *vertex_dx = new KView("vertex_dx", (x + 1));
  *vertex_dy = new KView("vertex_dy", (y + 1));
  *vertex_x = new KView("vertex_x", (x + 1));
  *vertex_y = new KView("vertex_y", (y + 1));

  *comms_buffer = new KView("comms_buffer", tealeaf_MAX(x, y) * settings->halo_depth);
  *host_comms_mirror = new KView::HostMirror();
  **host_comms_mirror = Kokkos::create_mirror_view(**comms_buffer);

  allocate_buffer(cg_alphas, settings->max_iters, 1);
  allocate_buffer(cg_betas, settings->max_iters, 1);
  allocate_buffer(cheby_alphas, settings->max_iters, 1);
  allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(double *cg_alphas, double *cg_betas, double *cheby_alphas, double *cheby_betas) {
  free(cg_alphas);
  free(cg_betas);
  free(cheby_alphas);
  free(cheby_betas);

  // TODO: Actually shouldn't be called on a per chunk basis, only by rank
  Kokkos::finalize();
}
