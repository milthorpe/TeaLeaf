#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunk.h"

static void dump_data(FILE *out, const char *name, double *data, int size) {
  double *host = (double *)malloc(size * sizeof(double));
  memcpy(host, data, size * sizeof(double));

  bool all_zero = true;
  for (int i = 0; i < size; i++) {
    if (host[i] != 0.0) {
      all_zero = false;
      break;
    }
  }

  fprintf(out, "[%s,+0]", name);
  if (all_zero) {
    fprintf(out, "(0.0 * %d)", size);
  } else {
    for (int i = 0; i < size; i++) {
      fprintf(out, "%.5f,", host[i]);
      if (i % 20 == 0) {
        fprintf(out, "\n[%s,+%d]", name, i);
      }
    }
  }
  fprintf(out, "\n");
  free(host);
}

void dump_chunk(const char *prefix, const char *suffix, Chunk *chunk, Settings *settings) {
  char name[256] = {};
  sprintf(name, "%s_rank=%d+%s.txt", prefix, settings->rank, suffix);
  FILE *out = fopen(name, "w");

  fprintf(out, "x=%d\n", chunk->x);
  fprintf(out, "y=%d\n", chunk->y);
  fprintf(out, "dt_init=%f\n", chunk->dt_init);

  fprintf(out, "left=%d\n", chunk->left);
  fprintf(out, "right=%d\n", chunk->right);
  fprintf(out, "bottom=%d\n", chunk->bottom);
  fprintf(out, "top=%d\n", chunk->top);

  dump_data(out, "density", chunk->density, chunk->x * chunk->y);
  dump_data(out, "energy", chunk->energy, chunk->x * chunk->y);
  dump_data(out, "u", chunk->u, chunk->x * chunk->y);
  dump_data(out, "p", chunk->p, chunk->x * chunk->y);
  dump_data(out, "r", chunk->r, chunk->x * chunk->y);
  dump_data(out, "w", chunk->w, chunk->x * chunk->y);
  dump_data(out, "kx", chunk->kx, chunk->x * chunk->y);
  dump_data(out, "ky", chunk->ky, chunk->x * chunk->y);

  fclose(out);
}

// Initialise the chunk
void initialise_chunk(Chunk *chunk, Settings *settings, int x, int y) {
  // Initialise the key variables
  chunk->x = x + settings->halo_depth * 2;
  chunk->y = y + settings->halo_depth * 2;
  chunk->dt_init = settings->dt_init;

  // Allocate the neighbour list
  chunk->neighbours = (int *)malloc(sizeof(int) * NUM_FACES);

  // Allocate the MPI comm buffers
  int lr_len = chunk->y * settings->halo_depth * NUM_FIELDS;
  chunk->left_send = (double *)malloc(sizeof(double) * lr_len);
  chunk->left_recv = (double *)malloc(sizeof(double) * lr_len);
  chunk->right_send = (double *)malloc(sizeof(double) * lr_len);
  chunk->right_recv = (double *)malloc(sizeof(double) * lr_len);

  int tb_len = chunk->x * settings->halo_depth * NUM_FIELDS;
  chunk->top_send = (double *)malloc(sizeof(double) * tb_len);
  chunk->top_recv = (double *)malloc(sizeof(double) * tb_len);
  chunk->bottom_send = (double *)malloc(sizeof(double) * tb_len);
  chunk->bottom_recv = (double *)malloc(sizeof(double) * tb_len);

  // Initialise the ChunkExtension, which allows composition of extended
  // fields specific to individual implementations
  chunk->ext = (ChunkExtension *)malloc(sizeof(ChunkExtension));
}

// Finalise the chunk
void finalise_chunk(Chunk *chunk) {
  free(chunk->neighbours);
  free(chunk->ext);
  free(chunk->left_send);
  free(chunk->left_recv);
  free(chunk->right_send);
  free(chunk->right_recv);
  free(chunk->top_send);
  free(chunk->top_recv);
  free(chunk->bottom_send);
  free(chunk->bottom_recv);
}
