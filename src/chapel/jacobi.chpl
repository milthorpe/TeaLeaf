/*
 *		JACOBI SOLVER KERNEL
 */
module jacobi {
    use settings;
    use chunks;
    use Math;
    use profile;
    use GPU;

    // Initialises the Jacobi solver
    proc jacobi_init(x: int(32), y: int(32), halo_depth: int(32), coefficient: real, 
                    rx: real, ry: real, ref u: [?Domain] real, ref u0: [Domain] real, 
                    const ref energy: [Domain] real, const ref density: [Domain] real,
                    ref kx: [Domain] real, ref ky: [Domain] real) {
        
        startProfiling("jacobi_init");
        const Inner = Domain[halo_depth..<y - 1, halo_depth..<x - 1];

        // if coefficient < 1 && coefficient < RECIP_CONDUCTIVITY
        // {
        //     writeln("Coefficient ", coefficient, " is not valid.\n");
        //      // stopProfiling("jacobi_init");
        //     exit(-1);
        // }
        
        // u = energy * density;
        forall ij in Domain {
            u[ij] = energy[ij] * density[ij];
        }

        forall (i, j) in Inner { 
            var densityCentre: real;
            var densityLeft: real;
            var densityDown: real;

            if coefficient == CONDUCTIVITY {  
                densityCentre = density[i, j];
                densityLeft = density[i, j-1];
                densityDown = density[i-1, j ];
            } else {
                densityCentre = 1.0/density[i, j];
                densityLeft =  1.0/density[i, j - 1];
                densityDown = 1.0/density[i - 1, j];
            }

            kx[i, j] = rx*(densityLeft+densityCentre)/(2.0*densityLeft*densityCentre);
            ky[i, j] = ry*(densityDown+densityCentre)/(2.0*densityDown*densityCentre);
        }

        stopProfiling("jacobi_init");
    }

    // The main Jacobi solve step
    proc jacobi_iterate(ref u: [?Domain] real, const ref u0: [Domain] real, 
                        ref r: [Domain] real, out error: real, const ref kx: [Domain] real, 
                        const ref ky: [Domain] real,
                        const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32)), const ref local_domain: subdomain(Domain), const ref OneD: domain(1,int(32))) {

        //forall (i, j) in Domain {
        forall oneDIdx in OneD {
            const (i,j) = local_domain.orderToIndex(oneDIdx);
            r[i,j] = u[i,j];
        }

        param ONE = 1.0:int(32);
        param ZERO = 0.0:int(32);
        const north = (ONE,ZERO), south = (-ONE,ZERO), east = (ZERO,ONE), west = (ZERO,-ONE);

        var err: real = 0.0;
        if useGPU {
            //forall ij in Domain.expand(-halo_depth) {
            forall oneDIdx in reduced_OneD with (+ reduce err) {
                const ij = reduced_local_domain.orderToIndex(oneDIdx);
                const stencil : real = (u0[ij] 
                                            + kx[ij + east] * r[ij + east] 
                                            + kx[ij] * r[ij + west]
                                            + ky[ij + north] * r[ij + north] 
                                            + ky[ij] * r[ij + south])
                                        / (1.0 + kx[ij] + kx[ij + east] 
                                            + ky[ij] + ky[ij + north]);
                u[ij] = stencil;

                err += abs(u[ij] - r[ij]);
            }

        } else {
            forall ij in reduced_local_domain with (+ reduce err) {
                const stencil : real = (u0[ij] 
                                            + kx[ij + east] * r[ij + east] 
                                            + kx[ij] * r[ij + west]
                                            + ky[ij + north] * r[ij + north] 
                                            + ky[ij] * r[ij + south])
                                        / (1.0 + kx[ij] + kx[ij + east] 
                                            + ky[ij] + ky[ij + north]);
                u[ij] = stencil;

                err += abs(stencil - r[ij]);
            }
        }
        error = err;
    }
}
