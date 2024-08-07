module settings{
  // Global constants
  param NUM_FIELDS: int = 6;
  param NUM_FACES: int = 4;

  param FIELD_DENSITY: int = 0;
  param FIELD_ENERGY0: int = 1;
  param FIELD_ENERGY1: int = 2;
  param FIELD_U: int = 3;
  param FIELD_P: int = 4;
  param FIELD_SD: int = 5;
  param CONDUCTIVITY: int = 1;
  param RECIP_CONDUCTIVITY: int = 2;
  param CG_ITERS_FOR_EIGENVALUES: int = 20;
  param ERROR_SWITCH_MAX: real = 1.0;

  enum Solver {Jacobi, CG, Chebyshev, PPCG}
  enum Geometry {RECTANGULAR, CIRCULAR, POINT}

  record setting {   
    var test_problem_filename: string;
    var tea_in_filename: string;
    var tea_out_filename: string;
    var grid_x_min: real;
    var grid_y_min: real;
    var grid_x_max: real;
    var grid_y_max: real;
    var grid_x_cells: int(32);
    var grid_y_cells: int(32);
    var dt_init: real;
    var max_iters: int;
    var eps: real;
    var end_time: real;
    var rank: int;
    var end_step: real;
    var summary_frequency: int;
    var coefficient: int;
    var error_switch: bool;
    var presteps: int;
    var eps_lim: real;
    var check_result: bool;
    var ppcg_inner_steps: int;
    var preconditioner: bool;
    var num_states: int;
    var halo_depth: int(32);
    var is_offload: bool;
    var fields_to_exchange: NUM_FIELDS * bool;
    var solver: Solver;
    var dx: real;
    var dy: real;
  }

  record state {
    var defined: bool;
    var density: real;
    var energy: real;
    var x_min: real;
    var y_min: real;
    var x_max: real;
    var y_max: real;
    var radius: real;
    var geometry: Geometry;
  }

  // Used for output to file
  record TimestepInfo {
        var step: int;
        var iterations: int;
        var iterations_prime: int;
        var inner_steps: int;
        var conductionError: real;
        var elapsedTime: real;
        var avgTimePerCell: real;
  }

  var max_iters: int;

  proc set_default_settings(ref setting_var : setting)
  {
    setting_var.test_problem_filename = "tea.problems";
    setting_var.tea_in_filename = "tea.in";
    setting_var.tea_out_filename = "tea.out";
    setting_var.grid_x_min = 0.0;
    setting_var.grid_y_min = 0.0;
    setting_var.grid_x_max = 100.0;
    setting_var.grid_y_max = 100.0;
    setting_var.grid_x_cells = 10;
    setting_var.grid_y_cells = 10;
    setting_var.dt_init = 0.1;
    setting_var.max_iters = 10000;
    setting_var.eps = 0.000000000000001;
    setting_var.end_time = 10.0;
    setting_var.end_step = 2147483647;
    setting_var.summary_frequency = 10;
    setting_var.solver = Solver.CG;
    setting_var.coefficient = 1;
    setting_var.error_switch = false;
    setting_var.presteps = 30;
    setting_var.eps_lim = 0.00001;
    setting_var.check_result = true;
    setting_var.ppcg_inner_steps = 10;
    setting_var.preconditioner = false;
    setting_var.num_states = 0;
    setting_var.halo_depth = 2;
    setting_var.is_offload = false;

    max_iters = setting_var.max_iters;

  }

  
  // Resets all of the fields to be exchanged
  proc reset_fields_to_exchange(ref setting_var : setting)
  {
    for i in 0..<NUM_FIELDS do setting_var.fields_to_exchange[i] = false;
  }

  // Checks if any of the fields are to be exchanged
  proc is_fields_to_exchange(const setting_var: setting)
  {
    var flag : bool = false;
    for ii in 0..<NUM_FIELDS do 
    {
      if setting_var.fields_to_exchange[ii] then
        flag = true;
    }
    return flag;
  }
}


