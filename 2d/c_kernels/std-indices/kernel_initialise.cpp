#include "../../settings.h"
#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include "std_shared.h"
#include <cstdlib>

// Allocates, and zeroes and individual buffer
static void allocate_buffer(double **a, int x, int y) {
  *a = alloc_raw<double>(x * y);

  if (*a == NULL) {
    die(__LINE__, __FILE__, "Error allocating buffer %s\n");
  }
  std::fill(EXEC_POLICY, *a, *a + (x * y), 0.0);
}

// Allocates all of the field buffers
void kernel_initialise(Settings *settings,    //
                       int x,                 //
                       int y,                 //
                       double **density0,     //
                       double **density,      //
                       double **energy0,      //
                       double **energy,       //
                       double **u,            //
                       double **u0,           //
                       double **p,            //
                       double **r,            //
                       double **mi,           //
                       double **w,            //
                       double **kx,           //
                       double **ky,           //
                       double **sd,           //
                       double **volume,       //
                       double **x_area,       //
                       double **y_area,       //
                       double **cell_x,       //
                       double **cell_y,       //
                       double **cell_dx,      //
                       double **cell_dy,      //
                       double **vertex_dx,    //
                       double **vertex_dy,    //
                       double **vertex_x,     //
                       double **vertex_y,     //
                       double **cg_alphas,    //
                       double **cg_betas,     //
                       double **cheby_alphas, //
                       double **cheby_betas) {

  print_and_log(settings, "Performing this solve with the C++ std::par_unseq %s solver\n", settings->solver_name);

  if (settings->device_selector) {
    print_and_log(settings, "Device selection is unsupported for this model, ignoring selector `%s`\n", settings->device_selector);
  }

  allocate_buffer(density0, x, y);
  allocate_buffer(density, x, y);
  allocate_buffer(energy0, x, y);
  allocate_buffer(energy, x, y);
  allocate_buffer(u, x, y);
  allocate_buffer(u0, x, y);
  allocate_buffer(p, x, y);
  allocate_buffer(r, x, y);
  allocate_buffer(mi, x, y);
  allocate_buffer(w, x, y);
  allocate_buffer(kx, x, y);
  allocate_buffer(ky, x, y);
  allocate_buffer(sd, x, y);
  allocate_buffer(volume, x, y);
  allocate_buffer(x_area, x + 1, y);
  allocate_buffer(y_area, x, y + 1);
  allocate_buffer(cell_x, x, 1);
  allocate_buffer(cell_y, 1, y);
  allocate_buffer(cell_dx, x, 1);
  allocate_buffer(cell_dy, 1, y);
  allocate_buffer(vertex_dx, x + 1, 1);
  allocate_buffer(vertex_dy, 1, y + 1);
  allocate_buffer(vertex_x, x + 1, 1);
  allocate_buffer(vertex_y, 1, y + 1);

  *cg_alphas = static_cast<double *>(std::malloc(sizeof(double) * settings->max_iters));
  *cg_betas = static_cast<double *>(std::malloc(sizeof(double) * settings->max_iters));
  *cheby_alphas = static_cast<double *>(std::malloc(sizeof(double) * settings->max_iters));
  *cheby_betas = static_cast<double *>(std::malloc(sizeof(double) * settings->max_iters));
  std::fill(*cg_alphas, *cg_alphas + settings->max_iters, 0);
  std::fill(*cg_betas, *cg_betas + settings->max_iters, 0);
  std::fill(*cheby_alphas, *cheby_alphas + settings->max_iters, 0);
  std::fill(*cheby_betas, *cheby_betas + settings->max_iters, 0);
}

void kernel_finalise(double *density0,     //
                     double *density,      //
                     double *energy0,      //
                     double *energy,       //
                     double *u,            //
                     double *u0,           //
                     double *p,            //
                     double *r,            //
                     double *mi,           //
                     double *w,            //
                     double *kx,           //
                     double *ky,           //
                     double *sd,           //
                     double *volume,       //
                     double *x_area,       //
                     double *y_area,       //
                     double *cell_x,       //
                     double *cell_y,       //
                     double *cell_dx,      //
                     double *cell_dy,      //
                     double *vertex_dx,    //
                     double *vertex_dy,    //
                     double *vertex_x,     //
                     double *vertex_y,     //
                     double *cg_alphas,    //
                     double *cg_betas,     //
                     double *cheby_alphas, //
                     double *cheby_betas) {
  dealloc_raw(density0);
  dealloc_raw(density);
  dealloc_raw(energy0);
  dealloc_raw(energy);
  dealloc_raw(u);
  dealloc_raw(u0);
  dealloc_raw(p);
  dealloc_raw(r);
  dealloc_raw(mi);
  dealloc_raw(w);
  dealloc_raw(kx);
  dealloc_raw(ky);
  dealloc_raw(sd);
  dealloc_raw(volume);
  dealloc_raw(x_area);
  dealloc_raw(y_area);
  dealloc_raw(cell_x);
  dealloc_raw(cell_y);
  dealloc_raw(cell_dx);
  dealloc_raw(cell_dy);
  dealloc_raw(vertex_dx);
  dealloc_raw(vertex_dy);
  dealloc_raw(vertex_x);
  dealloc_raw(vertex_y);
  dealloc_raw(cg_alphas);
  dealloc_raw(cg_betas);
  dealloc_raw(cheby_alphas);
  dealloc_raw(cheby_betas);
}
