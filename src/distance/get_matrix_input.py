#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from glob import glob
from tqdm import tqdm

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.sparse import csr_array,csc_array,save_npz,load_npz
from scipy.optimize import least_squares, fsolve

sys.path.append("/mnt/beegfs/lcesarini/graph-ensembles/src")


parser = argparse.ArgumentParser(
    description="Parsing the arguments to return the distance matrix"
)
# parser.add_argument(
#     "-pm","--plot_maps", action="store_true", help="Visualize the graph on a map."
# )
# parser.add_argument(
#     "--speed", action="store_true", help="load with geopandas or JSON"
# )
# parser.add_argument(
#    "--index_grid",default=1153, type=int,
#     help="Index of the grid for which we want to compute the distances."    
#    )
parser.add_argument(
    "--n_grid", default=0, type=int,
    help="Number of grid cells."
   )

# parser.add_argument(
#     "--n_iter", default=100, type=int,
#     help="""
#         Number of sample distances we want to get (i.e., how big do we want our ensemble?). 
#         For the number equal to the number of UL we get the complete distance matrix.
#         """
#    )
args = parser.parse_args()

n_grid=args.n_grid

os.chdir("/mnt/beegfs/lcesarini/")

"""
Creation of a np.array with shape (n_ul,3) where the 3 columns are: 

- ID_UL: np.int unique id representing the local unit
- SECTOR: np.int representing the sector
- ID_GRID: np.int representing the grid to which the local unit belongs
"""

with open(f'ge-distance/src/distance/out/UL_italy_poly_assigned_{n_grid}.json', 'r', encoding='utf-8') as f:
    r=json.load(f) 

gdf=json.loads(r)

row_id=[]
sectors=[]
index_grid=[]
employees=[]

for f in (gdf['features']):
    row_id.append(int(f['properties']['row_id']))
    index_grid.append(int(f['properties']['index']))
    sectors.append(f['properties']['Sectors'])
    employees.append(int(f['properties']['addetti_ul']))

# Step 1: Create a mapping of unique strings to integers
unique_strings = list(set(sectors))
string_to_int = {string: idx for idx, string in enumerate(unique_strings)}

# Step 2: Convert the list of strings to a list of integers using the mapping
sector_integers = [string_to_int[string] for string in sectors]

arr_ul=np.concatenate([
    np.array(row_id).reshape(-1,1),
    np.array(sector_integers).reshape(-1,1),
    np.array(index_grid).reshape(-1,1),
    np.array(employees).reshape(-1,1)
    ],axis=1)


arr_ul=csc_array(arr_ul,dtype=np.int32)
save_npz(f'ge-distance/src/distance/out/arr_input_ul_{n_grid}.npz', arr_ul)



