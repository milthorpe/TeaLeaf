// Initialise the chunkBlock
module chunks {
  use settings;
  use StencilDist;
  use BlockDist;

  // Set as True if using multilocale
  config param useStencilDist = false;
  config param useBlockDist = false;
  config param useGPU = false;

  var global_x: int(32);
  var global_y: int(32);
  var global_halo_depth: int(32);
  var global_dt_init: real;

  record Chunk {
    proc init(const ref setting_var: settings.setting) {
      global_halo_depth = setting_var.halo_depth;
      global_x = setting_var.grid_x_cells;
      global_y = setting_var.grid_y_cells;
      global_dt_init = setting_var.dt_init;
    }

    var halo_depth = global_halo_depth;
    var x_inner = global_x;
    var y_inner = global_y;

    var x: int(32) = x_inner + halo_depth * 2;
    var y: int(32) = y_inner + halo_depth * 2;
    
    // Domains
    const local_Domain = {0:int(32)..<y, 0:int(32)..<x};

    // Define the bounds of the arrays
    const Domain = if useStencilDist then local_Domain dmapped stencilDist(local_Domain, fluff=(1, 1))
                else if useBlockDist then local_Domain dmapped blockDist(local_Domain)
                else local_Domain;
    const reduced_local_domain = Domain.expand(-halo_depth);

    const x_domain = {0:int(32)..<x};
    const y_domain = {0:int(32)..<y};
    const x1_domain = {0:int(32)..<x+1};
    const y1_domain = {0:int(32)..<y+1};
    const max_iter_domain = {0:int(32)..<settings.max_iters};
  
    //TODO set up condition to make sure number of locales is only so big compared to grid size
    // if numLocales > (x * y) 
    // {
    //   writeln("Too few locales for grid size :", x,"x", y);
    //   exit(-1);
    // }

    var left: int;
    var right: int;
    var bottom: int;
    var top: int;
    
    var dt_init: real = global_dt_init;
    var density: [Domain] real = noinit; 
    var density0: [Domain] real = noinit;
    var energy: [Domain] real = noinit;
    var energy0: [Domain] real = noinit;

    var u: [Domain] real = noinit;
    var u0: [Domain] real = noinit;
    var p: [Domain] real = noinit;
    var r: [Domain] real = noinit;
    // var mi: [Domain] real = noinit;
    var w: [Domain] real = noinit;
    var kx: [Domain] real = noinit;
    var ky: [Domain] real = noinit;
    var sd: [Domain] real = noinit;
    var temp: [reduced_local_domain] real = noinit;
    

    var cell_x: [x_domain] real = noinit;
    var cell_dx: [x_domain] real = noinit;
    var cell_y: [y_domain] real = noinit;
    var cell_dy: [y_domain] real = noinit;

    var vertex_x: [x1_domain] real = noinit;
    var vertex_dx: [x1_domain] real = noinit;
    var vertex_y: [y1_domain] real = noinit;
    var vertex_dy: [y1_domain] real = noinit;

    var volume: [Domain] real = noinit;

    // Cheby and PPCG arrays
    var theta: real;
    var eigmin: real;
    var eigmax: real;

    var cg_alphas: [max_iter_domain] real = noinit;
    var cg_betas: [max_iter_domain] real = noinit;
    var cheby_alphas: [max_iter_domain] real = noinit;
    var cheby_betas: [max_iter_domain] real = noinit;

  }
}