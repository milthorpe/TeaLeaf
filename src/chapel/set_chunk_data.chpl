/*
 * 		SET CHUNK DATA KERNEL
 * 		Initialises the chunk's mesh data.
 */

// Extended kernel for the chunk initialisation
module set_chunk_data{
	use settings;
	use chunks;
	
	proc set_chunk_data_driver(ref chunk_var: chunks.Chunk,  const ref setting_var : settings.setting){ 
		const x_min: real = setting_var.grid_x_min + setting_var.dx * (chunk_var.left:real);
		const y_min: real = setting_var.grid_y_min + setting_var.dy * (chunk_var.bottom:real);

		ref vertex_x = chunk_var.vertex_x;
		[ii in vertex_x.domain] vertex_x[ii] = x_min + setting_var.dx * (ii - setting_var.halo_depth); 

		ref vertex_y = chunk_var.vertex_y;
		[ii in vertex_y.domain] vertex_y[ii] = y_min + setting_var.dy * (ii - setting_var.halo_depth);  

		ref cell_x = chunk_var.cell_x;
		[ii in cell_x.domain] cell_x[ii] = 0.5 * (vertex_x[ii] + vertex_x[ii+1]);

		ref cell_y = chunk_var.cell_y;
		[ii in cell_y.domain] cell_y[ii] = 0.5 * (vertex_y[ii] + vertex_y[ii+1]);

		ref volume = chunk_var.volume;
		[ii in volume.domain] volume[ii] = setting_var.dx * setting_var.dy;
	}
}
