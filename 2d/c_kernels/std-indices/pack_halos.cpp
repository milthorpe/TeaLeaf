#include "../../shared.h"
#include "dpl_shim.h"
#include "ranged.h"
#include <algorithm>

// Packs left data into buffer.
void pack_left(const int x,          //
               const int y,          //
               const int depth,      //
               const int halo_depth, //
               double *field,        //
               double *buffer) {
  ranged<int> it(0, y * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int lines = index / depth;
    const int offset = halo_depth + lines * (x - depth);
    buffer[index] = field[offset + index];
  });
}

// Packs right data into buffer.
void pack_right(const int x,          //
                const int y,          //
                const int depth,      //
                const int halo_depth, //
                double *field,        //
                double *buffer) {
  ranged<int> it(0, y * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int lines = index / depth;
    const int offset = x - halo_depth - depth + lines * (x - depth);
    buffer[index] = field[offset + index];
  });
}

// Packs top data into buffer.
void pack_top(const int x,          //
              const int y,          //
              const int depth,      //
              const int halo_depth, //
              double *field,        //
              double *buffer) {
  ranged<int> it(0, x * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int offset = x * (y - halo_depth - depth);
    buffer[index] = field[offset + index];
  });
}

// Packs bottom data into buffer.
void pack_bottom(const int x,          //
                 const int y,          //
                 const int depth,      //
                 const int halo_depth, //
                 double *field,        //
                 double *buffer) {
  ranged<int> it(0, x * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int offset = x * halo_depth;
    buffer[index] = field[offset + index];
  });
}

// Unpacks left data from buffer.
void unpack_left(const int x,          //
                 const int y,          //
                 const int depth,      //
                 const int halo_depth, //
                 double *field,        //
                 double *buffer) {
  ranged<int> it(0, y * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int lines = index / depth;
    const int offset = halo_depth - depth + lines * (x - depth);
    field[offset + index] = buffer[index];
  });
}

// Unpacks right data from buffer.
void unpack_right(const int x,          //
                  const int y,          //
                  const int depth,      //
                  const int halo_depth, //
                  double *field,        //
                  double *buffer) {
  ranged<int> it(0, y * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int lines = index / depth;
    const int offset = x - halo_depth + lines * (x - depth);
    field[offset + index] = buffer[index];
  });
}

// Unpacks top data from buffer.
void unpack_top(const int x,          //
                const int y,          //
                const int depth,      //
                const int halo_depth, //
                double *field,        //
                double *buffer) {
  ranged<int> it(0, x * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int offset = x * (y - halo_depth);
    field[offset + index] = buffer[index];
  });
}

// Unpacks bottom data from buffer.
void unpack_bottom(const int x,          //
                   const int y,          //
                   const int depth,      //
                   const int halo_depth, //
                   double *field,        //
                   double *buffer) {
  ranged<int> it(0, x * depth);
  std::for_each(EXEC_POLICY, it.begin(), it.end(), [=](const int index) {
    const int offset = x * (halo_depth - depth);
    field[offset + index] = buffer[index];
  });
}

typedef void (*pack_kernel_f)(int, int, int, int, double *, double *);

// Either packs or unpacks data from/to buffers.
void pack_or_unpack(const int x,          //
                    const int y,          //
                    const int depth,      //
                    const int halo_depth, //
                    const int face,       //
                    bool pack,            //
                    double *field,        //
                    double *buffer) {
  pack_kernel_f kernel = NULL;

  switch (face) {
    case CHUNK_LEFT: kernel = pack ? pack_left : unpack_left; break;
    case CHUNK_RIGHT: kernel = pack ? pack_right : unpack_right; break;
    case CHUNK_TOP: kernel = pack ? pack_top : unpack_top; break;
    case CHUNK_BOTTOM: kernel = pack ? pack_bottom : unpack_bottom; break;
    default: die(__LINE__, __FILE__, "Incorrect face provided: %d.\n", face);
  }

  kernel(x, y, depth, halo_depth, field, buffer);
}
