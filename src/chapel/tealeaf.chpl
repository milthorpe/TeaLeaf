/* 
    This is a Chapel implementation of the TeaLeaf mini-app from the 
    University of Bristol, originally translated from the C version
    https://github.com/UoB-HPC/TeaLeaf 
    Copyright 2023-2024 Ahmad Azizi
    Copyright 2024 Oak Ridge National Laboratory
*/
module main {
    use Time;
    use settings;
    use chunks;
    use diffuse;
    use parse_config;
    use initialise;
    use profile;
    use GpuDiagnostics;
    use IO;
    
    param TEALEAF_VERSION = "2.0";

    proc main (args: [] string){
        // Time full program elapsed time
        var wallclock = new stopwatch();
        wallclock.start();

        const runLocale: locale = if useGPU then here.gpus[0] else here;
        on runLocale {
            // Create the settings wrapper
            var setting_var = new setting();
            writeln("Device : ", setting_var.locale);
            
            set_default_settings(setting_var);

            writeln("TeaLeaf:");
            writef(" - Ver.:     %s\n", TEALEAF_VERSION);
            writef(" - Deck:     %s\n", setting_var.tea_in_filename);
            writef(" - Out:      %s\n", setting_var.tea_out_filename);
            writef(" - Problem:  %s\n", setting_var.test_problem_filename);
            writef(" - Solver:   %s\n", setting_var.solver);
            writef(" - Profiler: %s\n", enableProfiling: string);
            writeln("Model:");
            writef(" - Name:      %s\n", "Chapel");
            writef(" - Execution: %s\n", if useGPU then "Offload" else "Host");
            try! stdout.flush();

            initProfiling();

            // Initialise states
            find_num_states(setting_var); 
            const states_domain = {0..<setting_var.num_states};
            var states: [states_domain] settings.state;

            // Read input files for state and setting information
            read_config(setting_var, states);
            
            // Create array of records of chunks and initialise
            var chunk_var = new Chunk(setting_var);

            initialise_application(chunk_var, setting_var, states);

            diffuse(chunk_var, setting_var);
            // Print the verbose profile summary
            reportProfiling();
        }
        wallclock.stop();
        writeln("\nTotal time elapsed: ", wallclock.elapsed(), " seconds");   
    }
}