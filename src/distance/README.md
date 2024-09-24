# Info to produce distances

## The scripts in order perform:

1. _make_grid.sbatch:_  
Runs _make_grid.py_. The scripts runs a class implementing a quad tree defined as  
"_...often used to partition a two-dimensional space by recursively subdividing it into four quadrants or regions..._" 

The scripts takes one argument N_UL_SQUARE (int) indicating the number of local unit inside each square of the grid.

On the HPC with 1 node, 3 tasks, and 20 cpus-per-taks:  
- takes around 8min for 100k each cell (i.e., pretty coarse grid),10k each cell ().,1k each cell (i.e., pretty fine grid).

The scripts return a CSV with the bounds of ALL square created (i.e., also the square inside each other. In grid_csv_to_gpkg.py the surplus squares are removed.)

2. _make_gpkg.sbatch:_  
Runs _grid_csv_to_gpkg.py_. The scripts takes the csv obtained from make_grid.sbatch and produces a geopackage file of all the squares.  
  
On the HPC with 1 node, 1 tasks, and 48 cpus-per-taks:  
- takes around 10sec.


3. _int_and_get_distmat.sbatch:_
Runs _get_distance_matrix.py_. The scripts takes the grid and by intersecting with the position of the local units, attach the index of the grid to each local unit. Then, computes the distance matrix for each grid cell.
  
On the HPC with 1 node, 3 tasks, and 20 cpus-per-task:  
- takes around 40min.

4. _get_counts.sbatch:_
Runs _get_num_grid_cell.py_. The scripts takes counts the number of local unit inside each cell.
  
On the HPC with 1 node, 3 tasks, and 20 cpus-per-task:  
- takes around 3min.

5. _check_distances.sbatch:_
Runs _control_distances.py_. The scripts takes counts the number of local unit inside each cell.
  
On the HPC with 1 node, 3 tasks, and 20 cpus-per-task:  
- takes around 3min.

6. _create_ensemble.sbatch:_