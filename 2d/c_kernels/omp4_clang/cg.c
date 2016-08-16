#include <stdlib.h>
#include <omp.h>
#include "../../shared.h"

/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */

// Initialises the CG solver
void cg_init(
    const int x,
    const int y,
    const int halo_depth,
    const int coefficient,
    double rx,
    double ry,
    double* rro,
    double* density,
    double* energy,
    double* u,
    double* p,
    double* r,
    double* w,
    double* kx,
    double* ky)
{
  if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
  {
    die(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
  }

#pragma omp target teams distribute parallel for
  for(int jj = 0; jj < y; ++jj)
  {
    for(int kk = 0; kk < x; ++kk)
    {
      const int index = kk + jj*x;
      p[index] = 0.0;
      r[index] = 0.0;
      u[index] = energy[index]*density[index];
    }
  }

#pragma omp target teams distribute parallel for
  for(int jj = 1; jj < y-1; ++jj)
  {
    for(int kk = 1; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      w[index] = (coefficient == CONDUCTIVITY) 
        ? density[index] : 1.0/density[index];
    }
  }

#pragma omp target teams distribute parallel for
  for(int jj = halo_depth; jj < y-1; ++jj)
  {
    for(int kk = halo_depth; kk < x-1; ++kk)
    {
      const int index = kk + jj*x;
      kx[index] = rx*(w[index-1]+w[index]) /
        (2.0*w[index-1]*w[index]);
      ky[index] = ry*(w[index-x]+w[index]) /
        (2.0*w[index-x]*w[index]);
    }
  }

  const int nb = 512;
  double* reduce_temp = (double*)calloc(sizeof(double),nb);
#pragma omp target teams distribute parallel for map(tofrom: reduce_temp[:nb]) //reduction(+:rro_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      const double smvp = SMVP(u);
      w[index] = smvp;
      r[index] = u[index]-w[index];
      p[index] = r[index];
#pragma omp critical
      reduce_temp[omp_get_team_num()] += r[index]*p[index];
    }
  }

  for(int ii = 0; ii < nb; ++ii) {
    *rro += reduce_temp[ii];
  }
}

// Calculates w
void cg_calc_w(
    const int x,
    const int y,
    const int halo_depth,
    double* pw,
    double* p,
    double* w,
    double* kx,
    double* ky)
{
  const int nb = 512;
  double* reduce_temp = (double*)calloc(sizeof(double),nb);
#pragma omp target teams distribute parallel for map(tofrom: reduce_temp[:nb]) //reduction(+:pw_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;
      const double smvp = SMVP(p);
      w[index] = smvp;

#pragma omp critical
      reduce_temp[omp_get_team_num()] += w[index]*p[index];
    }
  }

  for(int ii = 0; ii < nb; ++ii) {
    *pw += reduce_temp[ii];
  }
}

// Calculates u and r
void cg_calc_ur(
    const int x,
    const int y,
    const int halo_depth,
    const double alpha,
    double* rrn,
    double* u,
    double* p,
    double* r,
    double* w)
{
  const int nb = 512;
  double* reduce_temp = (double*)calloc(sizeof(double),nb);
#pragma omp target teams distribute parallel for map(tofrom: reduce_temp[:nb]) //reduction(+:rrn_temp)
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      u[index] += alpha*p[index];
      r[index] -= alpha*w[index];

#pragma omp critical
      reduce_temp[omp_get_team_num()] += r[index]*r[index];
    }
  }

  for(int ii = 0; ii < nb; ++ii) {
    *rrn += reduce_temp[ii];
  }
}

// Calculates p
void cg_calc_p(
    const int x,
    const int y,
    const int halo_depth,
    const double beta,
    double* p,
    double* r)
{
#pragma omp target teams distribute parallel for
  for(int jj = halo_depth; jj < y-halo_depth; ++jj)
  {
    for(int kk = halo_depth; kk < x-halo_depth; ++kk)
    {
      const int index = kk + jj*x;

      p[index] = beta*p[index] + r[index];
    }
  }
}

