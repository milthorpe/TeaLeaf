#pragma once

#ifdef __cplusplus
  #include "kokkos_shared.hpp"

typedef Kokkos::View<double *> *FieldBufferType;

typedef struct ChunkExtension {
  FieldBufferType comms_buffer;
  Kokkos::View<double *>::HostMirror *host_comms_mirror;
} ChunkExtension;

#else

typedef void *FieldBufferType;

typedef struct ChunkExtension {
  void *comms_buffer;
  void *host_comms_mirror;
} ChunkExtension;

#endif
