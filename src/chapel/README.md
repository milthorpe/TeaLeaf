# TeaLeafChapel

TeaLeaf mini-app originally created by University of Bristol, translated into the Chapel Language with support for single and multi-locale execution
The TeaLeaf mini-app is an iterative sparse linear solver on a structured grid, using a diffusion problem to simulate heat conduction over
a number of timesteps.

This application is based on the C implementation of TeaLeaf, made by the University of Bristol, with the git commit 'c66cafb' (https://github.com/UoB-HPC/TeaLeaf).

Copyright 2023-2024 Ahmad Azizi

Copyright 2024 Oak Ridge National Laboratory

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation
This version of TeaLeafChapel has been tested to work on Chapel version 1.32.0.
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/TeaLeafChapel.git
2. Run the Makefile
   There are three versions of TeaLeafChapel you can build.
   To make the local single-node version: 
   ```bash
   make
   ```
   To make the distributed multi-node version using the block distribution Chapel module: 
   ```bash
   make block
   ```
   To make the distributed multi-node version using the stencil distribution Chapel module: 
   ```bash
   make stencil
   ```
   This will produce an object file with the names respectively:
   ```
   chapel-tealeaf
   chapel-tealeaf-block
   chapel-tealeaf-stencil
   ```
To set additional Chapel compilation flags, set the environment variable `CHPL_FLAGS` when running `make`.

## Usage
There are a number of test problems with known solutions within the ```tea.problem``` file.
Each line pertains to a problem, with the ordering being:
  ```
  'x-axis dimension' 'y-axis dimension' 'timesteps' 'error difference to expected solution'
  ```
These values can be put into the ```tea.in``` file into the following rows:
  ```
  x_cells
  y_cells
  end_step
  ```
If a known problem is being used, then TeaLeafChapel will automatically compare the output error to the corresponding known error.

To run TeaLeafChapel, after having built it, use the following:
  ```
  ./chapel-tealeaf
  ```

To run the benchmarks provided:
  ```
  ./tests/benchmark.sh
  ```

To manually set the maximum number of threads, open the file in your editor of choice and change the line:
```
max_threads=X
```
where X is the maximum number of threads you will use.

