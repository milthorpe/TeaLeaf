/*
 *		SHARED SOLVER METHODS
 */
module solver_methods {
    use profile;
    use chunks;
    use GPU;
    
    // Copies the current u into u0
    proc copy_u (const in halo_depth: int, const ref u: [?u_domain] real, ref u0: [u_domain] real) {
        startProfiling("copy_u");

        forall ij in u_domain.expand(-halo_depth) {
            u0[ij] = u[ij];
        } 

        stopProfiling("copy_u");
    }

    // Calculates the current value of r
    proc calculate_residual(const in halo_depth: int(32), const ref u: [?Domain] real, const ref u0: [Domain] real, 
                            ref r: [Domain] real, const ref kx: [Domain] real, const ref ky: [Domain] real,
                        const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32))) {
        startProfiling("calculate_residual");

        //forall (i, j) in Domain.expand(-halo_depth) {
        forall oneDIdx in reduced_OneD {
            const (i,j) = reduced_local_domain.orderToIndex(oneDIdx);
            const smvp: real = (1.0 + ((kx[i+1, j]+kx[i, j])
                + (ky[i, j+1]+ky[i, j])))*u[i, j]
                - ((kx[i+1, j]*u[i+1, j])+(kx[i, j]*u[i-1, j]))
                - ((ky[i, j+1]*u[i, j+1])+(ky[i, j]*u[i, j-1]));
            
            r[i, j] = u0[i, j] - smvp;
        }
    
        stopProfiling("calculate_residual");
    }

    // Calculates the 2 norm of a given buffer
    proc calculate_2norm (const in halo_depth: int(32), ref buffer: [?buffer_domain] real, ref norm: real) {
        startProfiling("calculate_2norm");

        var norm_temp: real;

        const innerDomain = buffer_domain.expand(-halo_depth);

        if useGPU {
            forall ij in innerDomain {
                buffer[ij] = buffer[ij] ** 2;
            } 
        
            norm_temp = gpuSumReduce(buffer[innerDomain]); // This causes a lot of transfers between host and device
        
        } else {
            norm_temp = + reduce (buffer[innerDomain] ** 2);
        }
        norm = norm_temp;

        stopProfiling("calculate_2norm");
    }

    // Finalises the solution
    proc finalise(const in x: int(32), const in y: int(32), const in halo_depth: int(32), ref energy: [?Domain] real, 
                    const ref density: [Domain] real, const ref u: [Domain] real) {
        startProfiling("finalise");

        const halo_domain = Domain[halo_depth-1..< y - halo_depth, halo_depth-1..<x-halo_depth];
        forall ij in halo_domain {
            energy[ij] = u[ij] / density[ij];
        }
        stopProfiling("finalise");
    }
}
