module cg_driver {
    use cg;
    use settings;
    use chunks;
    use local_halos;
    use solver_methods;
    use profile;

    // Performs a full solve with the CG solver kernels
    proc cg_driver (ref chunk_var : chunks.Chunk, ref setting_var : settings.setting, rx: real,
                    ry: real, ref error: real, out interation_count : int) {
        var rro : real;

        // Perform CG initialisation
        cg_init_driver(chunk_var, setting_var, rx, ry, rro);

        var tt_prime : int;
        // Iterate till convergence
        for tt in 0..<setting_var.max_iters {
            cg_main_step_driver(chunk_var, tt, rro, error);

            halo_update_driver (chunk_var, setting_var, 1);

            if (sqrt(abs(error)) < setting_var.eps) then break;

            tt_prime += 1;
        }
        interation_count = tt_prime;
        writeln(" CG:                    ", tt_prime, " iterations");
    }

    // Invokes the CG initialisation kernels
    proc cg_init_driver(ref chunk_var: chunks.Chunk, ref setting_var: settings.setting,
                        rx: real, ry: real, out rro: real) {
        cg_init(chunk_var.x, chunk_var.y, setting_var.halo_depth, setting_var.coefficient, rx, ry, rro, chunk_var.density, 
                chunk_var.energy, chunk_var.u, chunk_var.p, chunk_var.r, chunk_var.w, chunk_var.kx, chunk_var.ky,
                chunk_var.reduced_local_domain, chunk_var.reduced_OneD, chunk_var.local_Domain, chunk_var.OneD);

        reset_fields_to_exchange(setting_var);
        setting_var.fields_to_exchange[FIELD_U] = true;
        setting_var.fields_to_exchange[FIELD_P] = true;
        halo_update_driver(chunk_var, setting_var, 1);

        copy_u(setting_var.halo_depth, chunk_var.u, chunk_var.u0);
    }

    // Invokes the main CG solve kernels
    proc cg_main_step_driver(ref chunk_var: chunks.Chunk, tt: int,
                                ref rro: real, ref error: real) {
        var pw: real;
        
        cg_calc_w(pw, chunk_var.p, chunk_var.w, chunk_var.kx, chunk_var.ky,
            chunk_var.reduced_local_domain, chunk_var.reduced_OneD);

        const alpha = rro / pw;
        
        var rrn: real;
    
        chunk_var.cg_alphas[tt] = alpha;

        cg_calc_ur(alpha, rrn, chunk_var.u, chunk_var.p, chunk_var.r, chunk_var.w,
            chunk_var.reduced_local_domain, chunk_var.reduced_OneD);

        const beta = rrn / rro;
        
        chunk_var.cg_betas[tt] = beta;
        cg_calc_p(beta, chunk_var.p, chunk_var.r, chunk_var.reduced_local_domain, chunk_var.reduced_OneD);
        error = rrn;
        rro = rrn;
    }
}
