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
void kernel_initialise(Settings *settings, int x, int y, SyclBuffer **density0Buff, SyclBuffer **densityBuff, SyclBuffer **energy0Buff,
                       SyclBuffer **energyBuff, SyclBuffer **uBuff, SyclBuffer **u0Buff, SyclBuffer **pBuff, SyclBuffer **rBuff,
                       SyclBuffer **miBuff, SyclBuffer **wBuff, SyclBuffer **kxBuff, SyclBuffer **kyBuff, SyclBuffer **sdBuff,
                       SyclBuffer **volumeBuff, SyclBuffer **x_areaBuff, SyclBuffer **y_areaBuff, SyclBuffer **cell_xBuff,
                       SyclBuffer **cell_yBuff, SyclBuffer **cell_dxBuff, SyclBuffer **cell_dyBuff, SyclBuffer **vertex_dxBuff,
                       SyclBuffer **vertex_dyBuff, SyclBuffer **vertex_xBuff, SyclBuffer **vertex_yBuff, SyclBuffer **comms_bufferBuff,
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

  (*density0Buff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*densityBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*energy0Buff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*energyBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*uBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*u0Buff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*pBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*rBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*miBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*wBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*kxBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*kyBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*sdBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*volumeBuff) = new SyclBuffer{range<1>{(size_t)x * y}};
  (*x_areaBuff) = new SyclBuffer{range<1>{(size_t)(x + 1) * y}};
  (*y_areaBuff) = new SyclBuffer{range<1>{(size_t)x * (y + 1)}};
  (*cell_xBuff) = new SyclBuffer{range<1>{(size_t)x}};
  (*cell_yBuff) = new SyclBuffer{range<1>{(size_t)y}};
  (*cell_dxBuff) = new SyclBuffer{range<1>{(size_t)x}};
  (*cell_dyBuff) = new SyclBuffer{range<1>{(size_t)y}};
  (*vertex_dxBuff) = new SyclBuffer{range<1>{(size_t)(x + 1)}};
  (*vertex_dyBuff) = new SyclBuffer{range<1>{(size_t)(y + 1)}};
  (*vertex_xBuff) = new SyclBuffer{range<1>{(size_t)(x + 1)}};
  (*vertex_yBuff) = new SyclBuffer{range<1>{(size_t)(y + 1)}};
  (*comms_bufferBuff) = new SyclBuffer{range<1>{(size_t)(tealeaf_MAX(x, y) * settings->halo_depth)}};

  allocate_buffer(cg_alphas, settings->max_iters, 1);
  allocate_buffer(cg_betas, settings->max_iters, 1);
  allocate_buffer(cheby_alphas, settings->max_iters, 1);
  allocate_buffer(cheby_betas, settings->max_iters, 1);
}

void kernel_finalise(SyclBuffer **density0Buff, SyclBuffer **densityBuff, SyclBuffer **energy0Buff, SyclBuffer **energyBuff,
                     SyclBuffer **uBuff, SyclBuffer **u0Buff, SyclBuffer **pBuff, SyclBuffer **rBuff, SyclBuffer **miBuff,
                     SyclBuffer **wBuff, SyclBuffer **kxBuff, SyclBuffer **kyBuff, SyclBuffer **sdBuff, SyclBuffer **volumeBuff,
                     SyclBuffer **x_areaBuff, SyclBuffer **y_areaBuff, SyclBuffer **cell_xBuff, SyclBuffer **cell_yBuff,
                     SyclBuffer **cell_dxBuff, SyclBuffer **cell_dyBuff, SyclBuffer **vertex_dxBuff, SyclBuffer **vertex_dyBuff,
                     SyclBuffer **vertex_xBuff, SyclBuffer **vertex_yBuff, SyclBuffer **comms_bufferBuff, double *cg_alphas,
                     double *cg_betas, double *cheby_alphas, double *cheby_betas, cl::sycl::queue **device_queue) {
  delete (cg_alphas);
  delete (cg_betas);
  delete (cheby_alphas);
  delete (cheby_betas);

  delete (*device_queue);

  delete (*density0Buff);
  delete (*densityBuff);
  delete (*energy0Buff);
  delete (*energyBuff);
  delete (*uBuff);
  delete (*u0Buff);
  delete (*pBuff);
  delete (*rBuff);
  delete (*miBuff);
  delete (*wBuff);
  delete (*kxBuff);
  delete (*kyBuff);
  delete (*sdBuff);
  delete (*volumeBuff);
  delete (*x_areaBuff);
  delete (*y_areaBuff);
  delete (*cell_xBuff);
  delete (*cell_yBuff);
  delete (*cell_dxBuff);
  delete (*cell_dyBuff);
  delete (*vertex_dxBuff);
  delete (*vertex_dyBuff);
  delete (*vertex_xBuff);
  delete (*vertex_yBuff);
  delete (*comms_bufferBuff);
}
