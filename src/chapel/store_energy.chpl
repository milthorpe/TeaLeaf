module store_energy {
    use profile;
    
    // Store original energy state
    proc store_energy (const ref energy0: [?E_Dom] real, ref energy: [E_Dom] real) {
        startProfiling("store_energy");

        [ij in E_Dom] energy[ij] = energy0[ij];

        stopProfiling("store_energy");
    }
}
