#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import json
import fiona
import numpy as np
import scipy as sp
import pandas as pd
from numba import jit
from time import time
import seaborn as sns
import geopandas as gpd
import cartopy.crs as ccrs
from datetime import datetime
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from quadtrees import Point, Rect, QuadTree
from scipy.sparse import csr_array,csc_array
from shapely.geometry import Polygon,LineString
from scipy.spatial import distance_matrix

import argparse

parser = argparse.ArgumentParser(
    description="Parsing the arguments to return the distance matrix"
)
parser.add_argument(
    "-load_json", action="store_true", help="Load local unit with JSON"
)
# parser.add_argument(
#     "--speed", action="store_true", help="load with geopandas or JSON"
# )
# parser.add_argument(
#    "--plot", action="store_true", help="Make plots?"
#    )
# parser.add_argument(
#     "--n_ul", default=10000, type=int,
#     help="Number of edges as a multiplier of the nodes (i.e., How many edges are expected per node)."
#    )
parser.add_argument(
    "--n_grid", default=95, type=int, choices=[95,925,9174],
    help="Number of squares in the grid"
   )

args = parser.parse_args()
N_GRID=args.n_grid
os.chdir("/mnt/beegfs/lcesarini")

LOAD_JSON=args.load_json
if LOAD_JSON:
    print(f'Inizio: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    import json
    with open(f'ge-distance/src/distance/out/UL_italy_poly_assigned_{N_GRID}.json', 'r', encoding='utf-8') as f:
        r=json.load(f) 

    gdf=json.loads(r)

    lon=[]
    lat=[]
    sectors=[]
    id_grid=[]
    row_id=[]
    for f in gdf['features']:
        lon.append(f['properties']['lng'])
        lat.append(f['properties']['lat'])
        row_id.append(f['properties']['row_id'])
        id_grid.append(f['properties']['index'])
        sectors.append(f['properties']['Sectors'])

    print(f'Loaded the UL: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    # load the distance matrix
    dist_mat = sp.sparse.load_npz(f'ge-distance/src/distance/out/dist_mat_{N_GRID}.npz')
    # dist_mat = sp.sparse.load_npz('2024_dis_io/out/dist_mat_100k.npz')
    print(f'Loaded the distance matrix: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    #count the UL inside each grid box
    id,counts=np.unique(id_grid,return_counts=True)
    print(f'Make the count inside each grid box: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    #write the output of value counts to file
    df_count=pd.DataFrame({'index':id,'counts':counts})
    df_count.to_csv(f'ge-distance/src/distance/out/UL_count_{N_GRID}.csv')
    print(f'Write to disk: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    df_count.sort_values(by='counts',ascending=False,inplace=True)

    sns.barplot(data=df_count,
                x='id',y='counts')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('Number of ULs')
    plt.xlabel('ID of the area')
    #remove xticks
    plt.xticks([])
    plt.title('Number of ULs per polygon')
    plt.savefig(f'ge-distance/src/distance/tmp/UL_count_hist_{N_GRID}.png')
    plt.close()
else:

    print(f'Inizio: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # load local units
    INT_ul=gpd.read_file('ge-distance/src/distance/out/UL_italy_poly_assigned.gpkg')
    print(f'Loaded the UL: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    INT_ul.merge(df_count,on='index').plot(column='counts',legend=True)

    # load the distance matrix
    dist_mat = sp.sparse.load_npz('ge-distance/src/distance/out/dist_mat_100k.npz')
    print(f'Loaded the distance matrix: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


    #count the UL inside each grid box
    COUNT=INT_ul.groupby('index').size().reset_index(name='counts')
    print(f'Make the count inside each grid box: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    #write the output of value counts to file
    COUNT.to_csv('ge-distance/src/distance/out/UL_count.csv')
    print(f'Write to disk: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    import seaborn as sns
    # sns.histplot(INT_ul_min['index'],binwidth=1)
    sns.histplot(INT_ul['index'],binwidth=1)
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Number of ULs')
    plt.ylabel('ID of the area')
    plt.title('Number of ULs per polygon')
    plt.savefig('ge-distance/src/distance/out/UL_count_hist.png')
    plt.close()



#plot the grid
