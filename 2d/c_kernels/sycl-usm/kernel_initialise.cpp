#include "../../settings.h"
#include "../../shared.h"
#include "sycl_shared.hpp"
#include <stdlib.h>

using namespace cl::sycl;

// Allocates, and zeroes an individual buffer
void allocate_buffer(double **a, int x, int y) {
  //*a = (double*)malloc(sizeof(double)*x*y);
  *a = new double[x * y];

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
void kernel_initialise(Settings *settings, int x, int y, SyclBuffer *density0Buff, SyclBuffer *densityBuff, SyclBuffer *energy0Buff,
                       SyclBuffer *energyBuff, SyclBuffer *uBuff, SyclBuffer *u0Buff, SyclBuffer *pBuff, SyclBuffer *rBuff,
                       SyclBuffer *miBuff, SyclBuffer *wBuff, SyclBuffer *kxBuff, SyclBuffer *kyBuff, SyclBuffer *sdBuff,
                       SyclBuffer *volumeBuff, SyclBuffer *x_areaBuff, SyclBuffer *y_areaBuff, SyclBuffer *cell_xBuff,
                       SyclBuffer *cell_yBuff, SyclBuffer *cell_dxBuff, SyclBuffer *cell_dyBuff, SyclBuffer *vertex_dxBuff,
                       SyclBuffer *vertex_dyBuff, SyclBuffer *vertex_xBuff, SyclBuffer *vertex_yBuff, SyclBuffer *comms_bufferBuff,
                       double **cg_alphas, double **cg_betas, double **cheby_alphas, double **cheby_betas, queue **device_queue) {
  print_and_log(settings, "Performing this solve with the SYCL %s solver\n", settings->solver_name);

  auto selector = !settings->device_selector ? "0" : std::string(settings->device_selector);
  auto devices = sycl::device::get_devices();

  print_and_log(settings, "Available devices = %d\n", devices.size());
  if (devices.empty()) {
    die(__LINE__, __FILE__, "sycl::device::get_devices() returned 0 devices.");
  }
  for (int i = 0; i < devices.size(); ++i) {
    print_and_log(settings, "\t[%d] %s\n", i, devices[i].get_info<sycl::info::device::name>().c_str());
  }

  device selected;
  try {
    selected = devices.at(std::stoul(selector));
  } catch (const std::exception &e) {
    print_and_log(settings, "Unable to parse/select device index `%s`:%s\n", selector.c_str(), e.what());
    print_and_log(settings, "Attempting to match device with substring `%s`\n", selector.c_str());

    auto matching = std::find_if(devices.begin(), devices.end(), [selector](const sycl::device &device) {
      return device.get_info<sycl::info::device::name>().find(selector) != std::string::npos;
    });
    if (matching != devices.end()) {
      selected = *matching;
      print_and_log(settings, "Using first device matching substring `%s`\n", selector.c_str());
    } else if (devices.size() == 1) {
      print_and_log(settings, "No matching device but there's only one device, will be using that anyway\n");
    } else {
      die(__LINE__, __FILE__, "No matching devices for `%s`\n", selector.c_str());
    }
  }

  (*device_queue) = new queue(selected);
  print_and_log(settings, "Running on = %s\n", (**device_queue).get_device().get_info<cl::sycl::info::device::name>().c_str());

  (*density0Buff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*densityBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*energy0Buff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*energyBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*uBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*u0Buff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*pBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*rBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*miBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*wBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*kxBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*kyBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*sdBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*volumeBuff) = sycl::malloc_device<double>(x * y, **device_queue);
  (*x_areaBuff) = sycl::malloc_device<double>((x + 1) * y, **device_queue);
  (*y_areaBuff) = sycl::malloc_device<double>(x * (y + 1), **device_queue);
  (*cell_xBuff) = sycl::malloc_device<double>(x, **device_queue);
  (*cell_yBuff) = sycl::malloc_device<double>(y, **device_queue);
  (*cell_dxBuff) = sycl::malloc_device<double>(x, **device_queue);
  (*cell_dyBuff) = sycl::malloc_device<double>(y, **device_queue);
  (*vertex_dxBuff) = sycl::malloc_device<double>((x + 1), **device_queue);
  (*vertex_dyBuff) = sycl::malloc_device<double>((y + 1), **device_queue);
  (*vertex_xBuff) = sycl::malloc_device<double>((x + 1), **device_queue);
  (*vertex_yBuff) = sycl::malloc_device<double>((y + 1), **device_queue);
  (*comms_bufferBuff) = sycl::malloc_device<double>((tealeaf_MAX(x, y) * settings->halo_depth), **device_queue);

  allocate_buffer(cg_alphas, settings->max_iters, 1);
  allocate_buffer(cg_betas, settings->max_iters, 1);
  allocate_buffer(cheby_alphas, settings->max_iters, 1);
  allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(SyclBuffer *density0Buff, SyclBuffer *densityBuff, SyclBuffer *energy0Buff, SyclBuffer *energyBuff, SyclBuffer *uBuff,
                     SyclBuffer *u0Buff, SyclBuffer *pBuff, SyclBuffer *rBuff, SyclBuffer *miBuff, SyclBuffer *wBuff, SyclBuffer *kxBuff,
                     SyclBuffer *kyBuff, SyclBuffer *sdBuff, SyclBuffer *volumeBuff, SyclBuffer *x_areaBuff, SyclBuffer *y_areaBuff,
                     SyclBuffer *cell_xBuff, SyclBuffer *cell_yBuff, SyclBuffer *cell_dxBuff, SyclBuffer *cell_dyBuff,
                     SyclBuffer *vertex_dxBuff, SyclBuffer *vertex_dyBuff, SyclBuffer *vertex_xBuff, SyclBuffer *vertex_yBuff,
                     SyclBuffer *comms_bufferBuff, double *cg_alphas, double *cg_betas, double *cheby_alphas, double *cheby_betas,
                     cl::sycl::queue **device_queue) {

  delete (cg_alphas);
  delete (cg_betas);
  delete (cheby_alphas);
  delete (cheby_betas);

  sycl::free(*density0Buff, **device_queue);
  sycl::free(*densityBuff, **device_queue);
  sycl::free(*energy0Buff, **device_queue);
  sycl::free(*energyBuff, **device_queue);
  sycl::free(*uBuff, **device_queue);
  sycl::free(*u0Buff, **device_queue);
  sycl::free(*pBuff, **device_queue);
  sycl::free(*rBuff, **device_queue);
  sycl::free(*miBuff, **device_queue);
  sycl::free(*wBuff, **device_queue);
  sycl::free(*kxBuff, **device_queue);
  sycl::free(*kyBuff, **device_queue);
  sycl::free(*sdBuff, **device_queue);
  sycl::free(*volumeBuff, **device_queue);
  sycl::free(*x_areaBuff, **device_queue);
  sycl::free(*y_areaBuff, **device_queue);
  sycl::free(*cell_xBuff, **device_queue);
  sycl::free(*cell_yBuff, **device_queue);
  sycl::free(*cell_dxBuff, **device_queue);
  sycl::free(*cell_dyBuff, **device_queue);
  sycl::free(*vertex_dxBuff, **device_queue);
  sycl::free(*vertex_dyBuff, **device_queue);
  sycl::free(*vertex_xBuff, **device_queue);
  sycl::free(*vertex_yBuff, **device_queue);
  sycl::free(*comms_bufferBuff, **device_queue);

  delete (*device_queue);
}
