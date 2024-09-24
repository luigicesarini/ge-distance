#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import json
import fiona
import numpy as np
import scipy as sp
import pandas as pd
from numba import jit
from time import time
from tqdm import tqdm
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import gridspec
from haversine import haversine, Unit 
from scipy.spatial import distance_matrix
# from quadtrees import Point, Rect, QuadTree
from scipy.sparse import csr_array,csc_array
from shapely.geometry import Polygon,LineString

import argparse

parser = argparse.ArgumentParser(
    description="Parsing the arguments to return the distance matrix"
)
# parser.add_argument(
#     "-pm","--plot_maps", action="store_true", help="Visualize the graph on a map."
# )
# parser.add_argument(
#     "--speed", action="store_true", help="load with geopandas or JSON"
# )
parser.add_argument(
   "--index_grid",default=1153, type=int,
    help="Index of the grid for which we want to compute the distances."    
   )
parser.add_argument(
    "--n_grid", default=0, type=int,
    help="Number of grid cells."
   )

parser.add_argument(
    "--n_iter", default=100, type=int,
    help="""
        Number of sample distances we want to get (i.e., how big do we want our ensemble?). 
        For the number equal to the number of UL we get the complete distance matrix.
        """
   )
args = parser.parse_args()

n_grid=args.n_grid
index_grid=args.index_grid
n_iter=args.n_iter

os.chdir("/mnt/beegfs/lcesarini/")

grid=gpd.read_file(f'ge-distance/src/distance/out/grid_{n_grid}.gpkg')

dist_mat=sp.sparse.load_npz(f'ge-distance/src/distance/out/dist_mat_{n_grid}.npz')

with open(f'ge-distance/src/distance/out/UL_italy_poly_assigned_{n_grid}.json', 'r', encoding='utf-8') as f:
    r=json.load(f) 

gdf=json.loads(r)

lon=[]
lat=[]
# sectors=[]
id_grid=[]
row_id=[]
for f in gdf['features']:
    lon.append(f['properties']['lng'])
    lat.append(f['properties']['lat'])
    row_id.append(f['properties']['row_id'])
    id_grid.append(f['properties']['index'])
    # sectors.append(f['properties']['Sectors'])


id_grid=np.array(id_grid).reshape(-1,1)
lon=np.array(lon).reshape(-1,1)
lat=np.array(lat).reshape(-1,1)
# sectors=np.array(sectors)
row_id=np.array(row_id).reshape(-1,1)

unique_id_grid=np.unique(id_grid)
dict_distance_matrix={}
dict_dist_true={}
# for i in range(0,len(unique_id_grid)):

i=index_grid

for k in tqdm(range(n_iter)):
# for each grid id, sample a UL inside that grid
    SAMPLE_I=np.random.choice(np.arange(0,lat[id_grid==unique_id_grid[i]].shape[0]), 1)[0]
    COORDS_I=lon[id_grid==unique_id_grid[i]][SAMPLE_I],lat[id_grid==unique_id_grid[i]][SAMPLE_I]# then, sample a UL inside any other grid

    distance_matrix=[]
    dist_true=[]
    for j in range(0,len(unique_id_grid)):

    # for j in range(0,3):
        if j!=i:
            SAMPLE_J=np.random.choice(np.arange(0,lat[id_grid==unique_id_grid[j]].shape[0]), 1)[0]
            COORDS_J=lon[id_grid==unique_id_grid[j]][SAMPLE_J],lat[id_grid==unique_id_grid[j]][SAMPLE_J]# then, sample a UL inside any other grid
            # then, get the vector of distances according to the distance matrix
            # print(f'{i},{j}: {dist_mat[i,j] / 1000:.2f}')
            distance_matrix.append(dist_mat[i,j] / 1000)
            distance = haversine(COORDS_I, COORDS_J, unit=Unit.KILOMETERS)
            # print(f'{i},{j}: {distance:.2f}')
            dist_true.append(distance)
        else:
            next
    
    dict_distance_matrix.update({k:distance_matrix})
    dict_dist_true.update({k:dist_true})
# then, get the vector of distances according to the distance matrix

# finally the vector of REAL distances accordign to lon & lat

# in the end we will have two matrix with the distances, one according to the distance matrix and the other according to the real distances
df_1=pd.DataFrame.from_records(dict_distance_matrix).melt()
df_2=pd.DataFrame.from_records(dict_dist_true).melt()

df_1['dist_true']=df_2['value']
df_1.rename(columns={'variable':'index_grid','value':'dist_matrix'},inplace=True)

df_1.to_csv(f'ge-distance/src/distance/tmp/distances_{index_grid}_{n_grid}.csv',index=False)

