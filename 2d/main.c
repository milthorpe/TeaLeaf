#include "comms.h"
#include "application.h"
#include "chunk.h"
#include "shared.h"
#include "drivers/drivers.h"

void settings_overload(Settings* settings, int argc, char** argv);

int main(int argc, char** argv)
{
  // Immediately initialise MPI
  initialise_comms(argc, argv);

  barrier();

  // Create the settings wrapper
  Settings* settings = (Settings*)malloc(sizeof(Settings));
  set_default_settings(settings);
  settings_overload(settings, argc, argv);

  // Fill in rank information
  initialise_ranks(settings);

  barrier();

  // Perform initialisation steps
  Chunk* chunks;
  initialise_application(&chunks, settings);
  print_and_log(settings, "Using input:    %s\n", settings->tea_in_filename);
  print_and_log(settings, "Using output:   %s\n", settings->tea_out_filename);
  print_and_log(settings, "Using problems: %s\n", settings->test_problem_filename);

#ifndef NO_MPI

  long chunk_comms_total_x = 0, chunk_comms_total_y = 0;
  for(int i = 0; i < settings->num_chunks_per_rank; ++i){
    chunk_comms_total_x += chunks[i].x * settings->halo_depth * NUM_FIELDS;
    chunk_comms_total_y += chunks[i].y * settings->halo_depth * NUM_FIELDS;
  }
  long global_chunks_total_x = 0, global_chunks_total_y = 0;
  MPI_Reduce(&chunk_comms_total_x, &global_chunks_total_x, 1, MPI_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&chunk_comms_total_y, &global_chunks_total_y, 1, MPI_LONG, MPI_SUM, MASTER, MPI_COMM_WORLD);
  print_and_log(settings, "MPI total x buffer elements: %ld\n", global_chunks_total_x);
  print_and_log(settings, "MPI total y buffer elements: %ld\n", global_chunks_total_y);
  print_and_log(settings, "MPI total x buffer size:     %ld KB\n", chunk_comms_total_x * sizeof(double) / 1000);
  print_and_log(settings, "MPI total y buffer size:     %ld KB\n", chunk_comms_total_y * sizeof(double) / 1000);
  const char *mpi_cuda_aware_ct =
  #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
      "true";
  #elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
      "false";
  #else
      "unknown";
  #endif
  const char *mpi_cuda_aware_rt =
  #if defined(MPIX_CUDA_AWARE_SUPPORT)
      MPIX_Query_cuda_support() ? "true" : "false";
  #else
      "unknown";
  #endif
  print_and_log(settings, "MPI built with CUDA-awareness: %s\n", mpi_cuda_aware_ct);
  print_and_log(settings, "MPI runtime CUDA-awareness:    %s\n", mpi_cuda_aware_rt);

#endif

  // Perform the solve using default or overloaded diffuse
#ifndef DIFFUSE_OVERLOAD
  diffuse(chunks, settings);
#else
  diffuse_overload(chunks, settings);
#endif

  // Print the kernel-level profiling results
  if(settings->rank == MASTER)
  {
    PRINT_PROFILING_RESULTS(settings->kernel_profile);
  }

  // Finalise the kernel
  kernel_finalise_driver(chunks, settings);

  // Finalise each individual chunk
  for(int cc = 0; cc < settings->num_chunks_per_rank; ++cc)
  {
    finalise_chunk(&(chunks[cc]));
    free(&(chunks[cc]));
  }

  profiler_finalise(&settings->kernel_profile);
  profiler_finalise(&settings->application_profile);
  profiler_finalise(&settings->wallclock_profile);

  // Finalise the application
  free(settings);
  finalise_comms();

  return EXIT_SUCCESS;
}

void settings_overload(Settings* settings, int argc, char** argv)
{
  for(int aa = 1; aa < argc; ++aa)
  {
    // Overload the solver
    if(tealeaf_strmatch(argv[aa], "-solver") || tealeaf_strmatch(argv[aa], "--solver") || tealeaf_strmatch(argv[aa], "-s"))
    {
      if(aa+1 == argc) break;
      if(tealeaf_strmatch(argv[aa+1], "cg")) settings->solver = CG_SOLVER;
      if(tealeaf_strmatch(argv[aa+1], "cheby")) settings->solver = CHEBY_SOLVER;
      if(tealeaf_strmatch(argv[aa+1], "ppcg")) settings->solver = PPCG_SOLVER;
      if(tealeaf_strmatch(argv[aa+1], "jacobi")) settings->solver = JACOBI_SOLVER;
    }
    else if(tealeaf_strmatch(argv[aa], "-x"))
    {
      if(aa+1 == argc) break;
      settings->grid_x_cells = atoi(argv[aa]);
    }
    else if(tealeaf_strmatch(argv[aa], "-y"))
    {
      if(aa+1 == argc) break;
      settings->grid_y_cells = atoi(argv[aa]);
    }
    else if(tealeaf_strmatch(argv[aa], "-d") || tealeaf_strmatch(argv[aa], "--device"))
    {
      if(aa+1 == argc) break;
      settings->device_selector = argv[aa+1];
    }
    else if(tealeaf_strmatch(argv[aa], "--problems") || tealeaf_strmatch(argv[aa], "-p"))
    {
      if(aa+1 == argc) break;
      settings->test_problem_filename = argv[aa+1];
    }
    else if(tealeaf_strmatch(argv[aa], "--in") || tealeaf_strmatch(argv[aa], "-i") || tealeaf_strmatch(argv[aa], "--file") || tealeaf_strmatch(argv[aa], "-f"))
    {
      if(aa+1 == argc) break;
      settings->tea_in_filename = argv[aa+1];
    }
    else if(tealeaf_strmatch(argv[aa], "--out") || tealeaf_strmatch(argv[aa], "-o"))
    {
      if(aa+1 == argc) break;
      settings->tea_out_filename = argv[aa+1];
    }
    else if(tealeaf_strmatch(argv[aa], "-help") || tealeaf_strmatch(argv[aa], "--help") || tealeaf_strmatch(argv[aa], "-h"))
    {
      print_and_log(settings, "tealeaf <options>\n");
      print_and_log(settings, "options:\n");
      print_and_log(settings, "\t-solver, --solver, -s:\n");
      print_and_log(settings, "\t\tCan be 'cg', 'cheby', 'ppcg', or 'jacobi'\n");
      print_and_log(settings, "\t-p, --problems:\n");
      print_and_log(settings, "\t\tProblems file path'\n");
      print_and_log(settings, "\t-i, --in, -f, --file:\n");
      print_and_log(settings, "\t\tInput deck file path'\n");
      print_and_log(settings, "\t-o, --out:\n");
      print_and_log(settings, "\t\tOutput file path'\n");
      finalise_comms();
      exit(EXIT_SUCCESS);
    } 
  }
}
