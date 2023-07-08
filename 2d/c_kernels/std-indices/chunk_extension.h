#pragma once

typedef double* FieldBufferType;

// Empty extension point
typedef struct ChunkExtension
{
  FieldBufferType comm_buffer;
} ChunkExtension;
