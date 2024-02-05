module diffuse {
    use Time;
    use chunks;
    use settings;
    use solve_finish_driver;
    use local_halos;
    use field_summary;
    use ppcg_driver;
    use cg_driver;
    use jacobi_driver;
    use cheby_driver;
    use profile;
    use IO;

    var wallclock = new stopwatch();

    // The main timestep loop
    proc diffuse(ref chunk_var : chunks.Chunk, ref setting_var : settings.setting) {
        const end_step = setting_var.end_step : int;

        for tt in 0..<end_step do{
            solve(chunk_var, setting_var, tt);
        } 
        field_summary_driver(chunk_var, setting_var, true);
    }

    // Performs a solve for a single timestep
    proc solve(ref chunk_var : chunks.Chunk, ref setting_var : settings.setting, const ref tt : int) {
        
        wallclock.start();

        // Calculate minimum timestep information
        const dt : real = setting_var.dt_init;

        // Pick the smallest timestep across all ranks
        var rx : real = dt / (setting_var.dx * setting_var.dx);
        var ry : real = dt / (setting_var.dy * setting_var.dy);
       
        // Prepare halo regions for solve
        reset_fields_to_exchange(setting_var);
        setting_var.fields_to_exchange[FIELD_ENERGY1] = true;
        setting_var.fields_to_exchange[FIELD_DENSITY] = true;
        halo_update_driver(chunk_var, setting_var, 2);

        var error : real = 0;
        var iterations_count : int = 0;
        var iterations_prime : int = 0;
        var inner_steps : int = 0;

        writeln();
        writeln(" Timestep ", tt);

        // Perform the solve with one of the integrated solvers
        select (setting_var.solver) {
            when Solver.Jacobi{
                jacobi_driver(chunk_var, setting_var, rx, ry, error, iterations_count);
            }
            when Solver.CG{
                cg_driver(chunk_var, setting_var, rx, ry, error, iterations_count);
            }
            when Solver.Chebyshev{
                cheby_driver(chunk_var, setting_var, rx, ry, error, iterations_count, 
                            iterations_prime, inner_steps);
            }
            when Solver.PPCG{
                ppcg_driver(chunk_var, setting_var, rx, ry, error, iterations_count,
                            iterations_prime, inner_steps);
            }
        }

        // Perform solve finalisation tasks
        solve_finished_driver(chunk_var, setting_var);
        
        if tt % setting_var.summary_frequency == 0 {
            field_summary_driver(chunk_var, setting_var, false);
        }

        wallclock.stop();
        
        writef(" Wallclock:             %.3dr s\n", wallclock.elapsed());
        const average : real = wallclock.elapsed() / (setting_var.grid_x_cells * setting_var.grid_y_cells);
        writeln(" Avg. time per cell:    ", average, " s");
        writeln(" Error:                 ", error);
        try! stdout.flush();
    }
}