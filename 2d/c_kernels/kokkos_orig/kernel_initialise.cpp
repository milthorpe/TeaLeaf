#include <stdlib.h>
#include "kokkos_shared.hpp"
#include "../../settings.h"
#include "../../shared.h"

// Allocates, and zeroes an individual buffer
void allocate_buffer(double** a, int x, int y)
{
    *a = (double*)malloc(sizeof(double)*x*y);

    if(*a == NULL) 
    {
        die(__LINE__, __FILE__, "Error allocating buffer %s\n");
    }

#pragma omp parallel for
    for(int jj = 0; jj < y; ++jj)
    {
        for(int kk = 0; kk < x; ++kk)
        {
            const int index = kk + jj*x;
            (*a)[index] = 0.0;
        }
    }
}

// Allocates all of the field buffers
void kernel_initialise(
        Settings* settings, int x, int y, KView* density0, 
        KView* density, KView* energy0, KView* energy, KView* u, 
        KView* u0, KView* p, KView* r, KView* mi, 
        KView* w, KView* kx, KView* ky, KView* sd, 
        KView* volume, KView* x_area, KView* y_area, KView* cell_x, 
        KView* cell_y, KView* cell_dx, KView* cell_dy, KView* vertex_dx, 
        KView* vertex_dy, KView* vertex_x, KView* vertex_y, KView* comms_buffer,
        Kokkos::View<double*>::HostMirror* host_comms_mirror, 
        double** cg_alphas, double** cg_betas, double** cheby_alphas, 
        double** cheby_betas)
{
    Kokkos::initialize();

    *density0 = KView("density0", x*y);
    *density = KView("density", x*y);
    *energy0 = KView("energy0", x*y);
    *energy = KView("energy", x*y);
    *u = KView("u", x*y);
    *u0 = KView("u0", x*y);
    *p = KView("p", x*y);
    *r = KView("r", x*y);
    *mi = KView("mi", x*y);
    *w = KView("w", x*y);
    *kx = KView("kx", x*y);
    *ky = KView("ky", x*y);
    *sd = KView("sd", x*y);
    *volume = KView("volume", x*y);
    *x_area = KView("x_area", (x+1)*y);
    *y_area = KView("y_area", x*(y+1));
    *cell_x = KView("cell_x", x);
    *cell_y = KView("cell_y", y);
    *cell_dx = KView("cell_dx", x);
    *cell_dy = KView("cell_dy", y);
    *vertex_dx = KView("vertex_dx", (x+1));
    *vertex_dy = KView("vertex_dy", (y+1));
    *vertex_x = KView("vertex_x", (x+1));
    *vertex_y = KView("vertex_y", (y+1));

    *comms_buffer = KView("comms_buffer", MAX(x, y)*settings->halo_depth);
    *host_comms_mirror = Kokkos::create_mirror_view(*comms_buffer);

    allocate_buffer(cg_alphas, settings->max_iters, 1);
    allocate_buffer(cg_betas, settings->max_iters, 1);
    allocate_buffer(cheby_alphas, settings->max_iters, 1);
    allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(
        double* cg_alphas, double* cg_betas, double* cheby_alphas,
        double* cheby_betas)
{
    free(cg_alphas);
    free(cg_betas);
    free(cheby_alphas);
    free(cheby_betas);

    // TODO: Actually shouldn't be called on a per chunk basis, only by rank
    Kokkos::finalize();
}

