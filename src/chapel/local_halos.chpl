module local_halos {
    use chunks;
    use settings;
    use profile;

    // Invoke the halo update kernels using driver
    proc halo_update_driver(ref chunk_var: chunks.Chunk, const setting_var: settings.setting, depth: int(32)) {
        startProfiling("halo_update_driver");

        if is_fields_to_exchange(setting_var) {
            local_halos (chunk_var.x, chunk_var.y, depth, setting_var.halo_depth, setting_var.fields_to_exchange,
            chunk_var.density, chunk_var.energy0, chunk_var.energy, chunk_var.u, chunk_var.p, chunk_var.sd);
        }

        stopProfiling("halo_update_driver");
    }

    // The kernel for updating halos locally
    proc local_halos(x: int(32), y: int(32), depth: int(32), halo_depth: int(32),
        const fields_to_exchange: NUM_FIELDS*bool, ref density: [?D] real, ref energy0: [D] real,
        ref energy: [D] real, ref u: [D] real, ref p: [D] real, ref sd: [D] real) {
        
        if fields_to_exchange[FIELD_DENSITY] then update_face(x, y, halo_depth, depth, density);

        if fields_to_exchange[FIELD_P] then update_face(x, y, halo_depth, depth, p);

        if fields_to_exchange[FIELD_ENERGY0] then update_face(x, y, halo_depth, depth, energy0);

        if fields_to_exchange[FIELD_ENERGY1] then update_face(x, y, halo_depth, depth, energy);

        if fields_to_exchange[FIELD_U] then update_face(x, y, halo_depth, depth, u);

        if fields_to_exchange[FIELD_SD] then update_face(x, y, halo_depth, depth, sd);
    }

    // Updates faces in turn.
    proc update_face(x: int(32), y: int(32), halo_depth: int(32), depth: int(32), ref buffer: [?D] real) {
        if useGPU {
            const west_domain = D[halo_depth..<y-halo_depth, 0..<depth]; // west side of global halo
            foreach oneDIdx in 0..#west_domain.size {
                const (i,j) = west_domain.orderToIndex(oneDIdx);
                buffer[i, halo_depth-j-1] = buffer[i, j + halo_depth];
            }

            const east_domain = D[halo_depth..<y-halo_depth, 0..<depth]; // east side of global halo
            foreach oneDIdx in 0..#east_domain.size {
                const (i,j) = east_domain.orderToIndex(oneDIdx);
                buffer[i, x-halo_depth+j] = buffer[i, x-halo_depth-j-1];
            }
            
            const south_domain = D[0..<depth, halo_depth..<x-halo_depth]; // south side of global halo
            foreach oneDIdx in 0..#south_domain.size {
                const (i,j) = south_domain.orderToIndex(oneDIdx);
                buffer[y-halo_depth+i, j] = buffer[y-halo_depth-i-1, j];
            }

            const north_domain = D[0..<depth, halo_depth..<x-halo_depth];  //  north side of global halo
            foreach oneDIdx in 0..#north_domain.size {
                const (i,j) = north_domain.orderToIndex(oneDIdx);
                buffer[halo_depth-i-1, j] = buffer[halo_depth+i, j];
            }
        } else {
            const west_domain = D[halo_depth..<y-halo_depth, 0..<depth]; // west side of global halo
            forall (i, j) in west_domain { 
                buffer[i, halo_depth-j-1] = buffer[i, j + halo_depth];
            }

            const east_domain = D[halo_depth..<y-halo_depth, 0..<depth]; // east side of global halo
            forall (i, j) in east_domain { 
                buffer[i, x-halo_depth+j] = buffer[i, x-halo_depth-j-1];
            }
            
            const south_domain = D[0..<depth, halo_depth..<x-halo_depth]; // south side of global halo
            forall (i, j) in south_domain { 
                buffer[y-halo_depth+i, j] = buffer[y-halo_depth-i-1, j];
            }

            const north_domain = D[0..<depth, halo_depth..<x-halo_depth];  //  north side of global halo
            forall (i, j) in north_domain {
                buffer[halo_depth-i-1, j] = buffer[halo_depth+i, j];
            }
        }
    }
}
