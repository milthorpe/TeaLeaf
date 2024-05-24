module parse_config {
    use settings;
    use chunks;
    use IO;

    

    proc find_num_states (ref setting_var : setting){
        var counter : int;

        try {
            var tea_in = open (setting_var.tea_in_filename, ioMode.r); // open file
            var tea_in_reader = tea_in.reader(locking=false); // read file

            
            var line: string;
            
                for line in tea_in_reader.lines(){ // as long as not last line
                    if line.find("state") != -1 {
                        counter += 1;
                    }
                    
                }
            tea_in.close();

            setting_var.num_states = counter;
        } catch {
            writeln("Warning: unrecognized line error ", counter);
        }
        
    }

    // Read configuration file
    proc read_config(ref setting_var : setting, ref states : [0..<setting_var.num_states] state){

        // Read all of the settings from the config
        read_file(setting_var, states);

        // Set the cell widths now
        setting_var.dx = (setting_var.grid_x_max - setting_var.grid_x_min) / (setting_var.grid_x_cells : real);
        setting_var.dy = (setting_var.grid_y_max - setting_var.grid_y_min) / (setting_var.grid_y_cells : real);
        
        
    }   

    proc read_file(ref setting_var : setting, ref states : [0..<setting_var.num_states] state){ 
        // Open the configuration file
        try {
            var tea_in = open (setting_var.tea_in_filename, ioMode.r);
            var tea_in_reader = tea_in.reader(locking=false);
            var line: string;
            var counter : int; // fine line number
            // state variables
            var stateNum: int;
            var density: real;
            var energy: real;
            var geomType: string;
            var xmin: real;
            var xmax: real;
            var ymin: real;
            var ymax: real;
            var radius: real;  // Radius in text file goes last

            while tea_in_reader.readLine(line) { //TODO maybe improve this implementation
                counter += 1;
                // Check what the next line is equivalent to
                if line.find("*tea") != -1 {
                    continue;
                } else if line.find("*endtea", 0..) != -1{
                    break;  // End of file
                } else if line.find("state", 0..) != -1{
                    var tokens = line.split(); // break line into pieces

                    var temp = tokens[1] : int; // get state number

                    // Read all of the states from the configuration file
                    if temp == 1 {
                        states[temp-1].defined = true;
                        var energy_val = tokens[3].split('=')[0..];
                        var density_val = tokens[2].split('=')[0..];
                        states[temp-1].energy = energy_val[1] : real;
                        states[temp-1].density = density_val[1] : real;
                    } else {
                        // get value after equals
                        var energy_val = tokens[3].split('=')[0..];
                        var density_val = tokens[2].split('=')[0..];
                        var xmin_val = tokens[5].split('=')[0..];
                        var xmax_val = tokens[6].split('=')[0..];
                        var ymin_val = tokens[7].split('=')[0..];
                        var ymax_val = tokens[8].split('=')[0..];
                        var geomType_val = tokens[4].split('=')[0..];

                        states[temp-1].defined = true;
                        states[temp-1].energy = energy_val[1] : real;
                        states[temp-1].density = density_val[1] : real;
                        states[temp-1].x_min = xmin_val[1] : real;
                        states[temp-1].x_max = xmax_val[1] : real;
                        states[temp-1].y_min = ymin_val[1] : real;
                        states[temp-1].y_max = ymax_val[1] : real;

                        if geomType_val[1] == "rectangle"  
                            {states[temp-1].geometry = Geometry.RECTANGULAR;}
                        else if geomType_val[1] == "circular" 
                        {
                            // Only use radius var if geometry is set to circular
                            var radius_val = tokens[9].split('=')[0..];
                            states[temp-1].geometry = Geometry.CIRCULAR;
                            states[temp-1].radius = radius_val[1]: real;
                        }  
                        else if geomType_val[1] == "point" 
                            {states[temp-1].geometry = Geometry.POINT;}
                    }
                    continue;
                // Parse the switches
                } else if line.find("use_cg", 0..) != -1{
                    setting_var.solver = Solver.CG;
                    continue;
                } else if line.find("use_jacobi", 0..) != -1 {
                    setting_var.solver = Solver.Jacobi;
                    continue;
                } else if line.find("use_chebyshev", 0..) != -1 {
                    setting_var.solver = Solver.Chebyshev;
                    continue;
                } else if line.find("use_ppcg", 0..) != -1 {
                    setting_var.solver = Solver.PPCG;
                    continue;
                } else if line.find("use_c_kernels", 0..) != -1 {
                    // Do nothing
                    continue;
                } else if line.find("check_result", 0..) != -1 {
                    setting_var.check_result = true;
                    continue;
                } else if line.find("errswitch", 0..) != -1 {
                    setting_var.error_switch = true;
                    continue;
                } else if line.find("preconditioner_on", 0..) != -1 {
                    setting_var.preconditioner = true;
                    continue;
                } else if line.find("coefficient_density", 0..) != -1 {
                    setting_var.coefficient = CONDUCTIVITY;
                    continue;
                } else if line.find("coefficient_inverse_density", 0..) != -1 {
                    setting_var.coefficient = RECIP_CONDUCTIVITY;
                    continue;
                } 

                // Parse the key-value pairs
                else {
                    var (key, sep, value) = line.partition('=');
                    if key.find("xmin") >= 0 {
                        setting_var.grid_x_min = value : real;
                        continue;
                    } else if key.find("ymin") >= 0 {
                        setting_var.grid_y_min = value : real;
                        continue;
                    } else if key.find("xmax") >= 0 {
                        setting_var.grid_x_max = value : real;
                        continue;
                    } else if key.find("ymax") >= 0 {
                        setting_var.grid_y_max = value : real;
                        continue;
                    } else if key.find("x_cells") >= 0 {
                        setting_var.grid_x_cells = value : int(32);
                        continue;
                    } else if key.find("y_cells") >= 0 {
                        setting_var.grid_y_cells = value : int(32);
                        continue;
                    } else if key.find("initial_timestep") >= 0 {
                        setting_var.dt_init = value : real;
                        continue;
                    } else if key.find("end_time") >= 0 {
                        setting_var.end_time = value : real;
                        continue;
                    } else if key.find("end_step") >= 0 {
                        setting_var.end_step = value : real;
                        continue;
                    } else if key.find("summary_frequency") >= 0 {
                        setting_var.summary_frequency = value : int;
                        continue;
                    } else if key.find("presteps") >= 0 {
                        setting_var.presteps = value : int;
                        continue;
                    } else if key.find("ppcg_inner_steps") >= 0 {
                        setting_var.ppcg_inner_steps = value : int;
                        continue;
                    } else if key.find("epslim") >= 0 {
                        setting_var.eps_lim = value : real;
                        continue;
                    } else if key.find("max_iters") >= 0 {
                        setting_var.max_iters = value : int;
                        continue;
                    } else if key.find("eps") >= 0 {
                        setting_var.eps = value : real;
                        continue;
                    } else if key.find("halo_depth") >= 0 {
                        setting_var.halo_depth = value : int(32);
                        continue;
                    } else {  // If file is not formatted properly
                        // writeln("Warning: unrecognized line ", counter, ": ", line);
                    }
                }
            }
            // close the file
            tea_in.close();
        } catch {
            // writeln("Warning: unrecognized line error ");
        }
    }

    proc write_file(filename: string, ref chunk_var : chunks.Chunk, ref settings : setting, ref states : [0..<settings.num_states] state){
        try {
            // Open the file for writing
            var file = open(filename, ioMode.rw);

            // Create a writer for the file
            var writer = file.writer();

            // Write settings data
            writer.writeln("Solution Parameters:");
            writer.writeln("\tdt_init = ", settings.dt_init);
            writer.writeln("\tend_time = ", settings.end_time);
            writer.writeln("\tend_step = ", settings.end_step);
            writer.writeln("\tgrid_x_min = ", settings.grid_x_min);
            writer.writeln("\tgrid_y_min = ", settings.grid_y_min);
            writer.writeln("\tgrid_x_max = ", settings.grid_x_max);
            writer.writeln("\tgrid_y_max = ", settings.grid_y_max);
            writer.writeln("\tgrid_x_cells = ", settings.grid_x_cells);
            writer.writeln("\tgrid_y_cells = ", settings.grid_y_cells);
            writer.writeln("\tpresteps = ", settings.presteps);
            writer.writeln("\tppcg_inner_steps = ", settings.ppcg_inner_steps);
            writer.writeln("\teps_lim = ", settings.eps_lim);
            writer.writeln("\tmax_iters = ", settings.max_iters);
            writer.writeln("\teps = ", settings.eps);
            writer.writeln("\thalo_depth = ", settings.halo_depth);
            writer.writeln("\tcheck_result = ", settings.check_result);
            writer.writeln("\tcoefficient = ", settings.coefficient);
            writer.writeln("\tnum_chunks_per_rank = ", Locales);
            writer.writeln("\tsummary_frequency = ", settings.summary_frequency);

            // Write state data
            for ss in 0..settings.num_states-1 {
                writer.writeln("\t\nstate ", ss);
                writer.writeln("\tdensity = ", states[ss].density);
                writer.writeln("\tenergy= ", states[ss].energy);
                if ss > 0 {
                    writer.writeln("\tx_min = ", states[ss].x_min);
                    writer.writeln("\ty_min = ", states[ss].y_min);
                    writer.writeln("\tx_max = ", states[ss].x_max);
                    writer.writeln("\ty_max = ", states[ss].y_max);
                    writer.writeln("\tradius = ", states[ss].radius);
                    writer.writeln("\tgeometry = ", states[ss].geometry);
                }
            }

            writer.close();
            file.close();
        } catch e: Error {
            writeln("  Error :", e);
        }
    }

    proc write_timestep(filename: string, ref chunk_var : chunks.Chunk, iteration, iteration_prime, 
                        inner, solver, timestep, err, wallclock, average){
        try{
            // Open the file for writing
            var file = open(filename, ioMode.rw);

            // Create a writer for the file
            var writer = file.writer();

            writer.writeln("\tSolver = ", solver);

            writer.writeln("\tTimestep ", timestep);
            writer.writeln("\tIterations = ", iteration);
            
            if solver == Solver.Chebyshev {
                writer.writeln("\tCHEBY iterations = ", iteration_prime);
                writer.writeln("\t", iteration_prime, " estimated" );
            } else if solver == Solver.PPCG {
                writer.writeln("\tPPCG iterations = ", iteration_prime);
                writer.writeln("\t", iteration_prime, " PPCG inner iterations " );
            }

            writer.writeln("\tConduction error = ", err);
            writer.writeln("\tTime elapsed for current timestep = ", wallclock);
            writer.writeln("\tAvg. time per cell for current timestep = ", average);
            
            writer.close();
            file.close();
        } catch e: Error {
            writeln("  Error :", e);
        }

    }
}
