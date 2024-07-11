/*
 *		PPCG SOLVER KERNEL
 */
module ppcg {
    use profile;
    use chunks;

    // Initialises the PPCG solver
    proc ppcg_init(halo_depth: int, theta: real, const ref r: [?Domain] real, ref sd: [Domain] real) {
        startProfiling("ppcg_init");
        
        forall ij in Domain.expand(-halo_depth) {
            sd[ij] = r[ij] / theta;
        }

        stopProfiling("ppcg_init");
    }

    proc ppcg_inner_iteration(halo_depth: int, alpha: real, beta: real, ref u: [?Domain] real, 
                                ref r: [Domain] real, ref sd: [Domain] real, const ref kx: [Domain] real, 
                                const ref ky: [Domain] real) {
        startProfiling("ppcg_inner_iteration");

        forall (i, j) in Domain.expand(-halo_depth) {
            const smvp : real = (1.0 + (kx[i+1, j]+kx[i, j])
                        + (ky[i, j+1]+ky[i, j]))*sd[i, j]
                        - (kx[i+1, j]*sd[i+1, j]+kx[i, j]*sd[i-1, j])
                        - (ky[i, j+1]*sd[i, j+1]+ky[i, j]*sd[i, j-1]);

            r[i, j] -= smvp;
            u[i, j] += sd[i, j];
        }
        
        forall ij in Domain.expand(-halo_depth) {
            sd[ij] = alpha * sd[ij] + beta * r[ij]; // TODO check implicit version
        }

        stopProfiling("ppcg_inner_iteration");
    }
}
