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
