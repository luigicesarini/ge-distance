#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import json
import fiona
import numpy as np
import scipy as sp
from numba import jit
from time import time
import geopandas as gpd
import cartopy.crs as ccrs
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
# parser.add_argument(
#     "-pm","--plot_maps", action="store_true", help="Visualize the graph on a map."
# )
# parser.add_argument(
#     "--speed", action="store_true", help="load with geopandas or JSON"
# )
parser.add_argument(
   "--plot", action="store_true", help="Make plots?"
   )
parser.add_argument(
    "--n_ul", default=10000, type=int,
    help="Number of edges as a multiplier of the nodes (i.e., How many edges are expected per node)."
   )
parser.add_argument(
    "--n_grid", default=95, type=int, choices=[95,925,9174],
    help="Number of squares in the grid"
   )

args = parser.parse_args()
N_GRID=args.n_grid
os.chdir("/mnt/beegfs/lcesarini/")
"""
Possible approach to speed up
1. introduce numba
2. sparse matrix csc, csr

"""

all_geocoded=gpd.read_file('2024_dis_io/out/UL_italy_poly_assigned.gpkg')
all_geocoded.to_crs(3857,inplace=True)

ita=gpd.read_file('ge-distance/res/gadm36_ITA.gpkg',layer='gadm36_ITA_1')
ita.to_crs(3857,inplace=True)
# ita.boundary.plot(color='red')
# plt.savefig("Italy.png")
# plt.close()
# n)k*+2ubHy;NE,a
def compute_centroids(grid):
    """
    Compute centroids of the polygons in the grid.
    
    Parameters:
    grid: GeoDataFrame containing the grid of polygons
    
    Returns:
    numpy array containing the centroids
    """
    centroids = grid.geometry.centroid
    centroids_coords = np.array([(point.x, point.y) for point in centroids])
    return (centroids_coords)

def compute_distance_matrix(centroids):
    """
    Compute the distance matrix for the centroids.
    
    Parameters:
    centroids: numpy array containing the centroids
    
    Returns:
    numpy array representing the distance matrix
    """
    dist_matrix = distance_matrix(centroids, centroids)
    return csr_array(dist_matrix)

"""
AIM OF ALL OF THESE
1. compute the distance matrix between all the polygons
2. intersects the polygons with the local units. 
    That is, each local units has an id assigning it to a cell grid
3. use this indices as a node attribute that we can use to retrieve 
    the distance among local units, and use it.
"""

#read the quadtree
grid=gpd.read_file(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/grid_{N_GRID}.gpkg')
grid.to_crs(3857,inplace=True)

if 'index' not in grid.columns:
    grid['index']=np.arange(grid.shape[0])
grid['area']=grid.area.values

if os.path.exists(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/UL_italy_poly_assigned_{grid.shape[0]}.json'):
    with open(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/UL_italy_poly_assigned_{grid.shape[0]}.json', 'r', encoding='utf-8') as f:
        r=json.load(f) 

    gdf=json.loads(r)
    lon=[]
    lat=[]
    index=[]

    for f in (gdf['features']):
        lon.append(f['properties']['lng'])
        lat.append(f['properties']['lat'])
        index.append(f['properties']['index'])

    centroids_grid= compute_centroids(grid[grid.index.isin(index)])
    # centroids_grid=compute_centroids(grid)
    dist_mat=compute_distance_matrix(centroids_grid)

    sp.sparse.save_npz(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/dist_mat_{grid.shape[0]}.npz',dist_mat)
else:
    if 'index' in all_geocoded.columns:
        all_geocoded.drop(columns=['index'],inplace=True)

    if 'area' in all_geocoded.columns:
        all_geocoded.drop(columns=['area'],inplace=True)

    INT_ul=gpd.overlay(all_geocoded,grid, how='intersection',keep_geom_type=True)

    # group df by row_id and select the row with the minimum area
    INT_ul_min=INT_ul.loc[INT_ul.groupby('row_id')['area'].idxmin()]
    #find all_geocoded.row_id not in INT_ul.row_id
    INT_ul_min.to_file(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/UL_italy_poly_assigned_{grid.shape[0]}.gpkg',driver='GPKG')


    with open(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/UL_italy_poly_assigned_{grid.shape[0]}.json', 'w') as f:
        json.dump(INT_ul_min.to_json(), f)

    with open(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/UL_italy_poly_assigned_{grid.shape[0]}.json', 'r', encoding='utf-8') as f:
        r=json.load(f) 

    gdf=json.loads(r)
    lon=[]
    lat=[]
    index=[]

    for f in (gdf['features']):
        lon.append(f['properties']['lng'])
        lat.append(f['properties']['lat'])
        index.append(f['properties']['index'])

    centroids_grid= compute_centroids(grid[grid.index.isin(index)])
    # centroids_grid=compute_centroids(grid)
    dist_mat=compute_distance_matrix(centroids_grid)

    sp.sparse.save_npz(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/dist_mat_{grid.shape[0]}.npz',dist_mat)



# dist_points = INT_ul_min.iloc[[0,1111]].geometry.apply(lambda geom: INT_ul_min.iloc[[0,1111]].distance(geom))


# print(dist_mat.shape)
# print(f"Max distance: {dist_mat.max()/1000:.2f}km")
# print(f"Max distance, where: {np.argmax(dist_mat)}")

if args.plot:
    Im,Jm=np.argwhere(dist_mat==dist_mat.max())[:,0]
    I,J=np.random.choice(a=np.arange(dist_mat.shape[0]),size=2)
    # dist_mat.toarray()[I,J]
    fig,ax=plt.subplots(1,1,figsize=(16,12),
                        subplot_kw={'projection':ccrs.epsg(3857)})
    #layout

    ita.boundary.plot(edgecolor='red', facecolor='none',ax=ax)
    # grid.iloc[[I,J]].plot(edgecolor='black', facecolor='none',ax=ax)
    grid.plot(edgecolor='black', facecolor='none',ax=ax)
    INT.iloc[[I,J]].plot(edgecolor='black', facecolor='none',ax=ax)
    gpd.GeoDataFrame([LineString(INT.iloc[[I,J]].centroid)],crs=3857,geometry=0).plot(color='blue',ax=ax)
    INT.iloc[[Im,Jm]].plot(edgecolor='green', facecolor='none',ax=ax)
    gpd.GeoDataFrame([LineString(INT.iloc[[Im,Jm]].centroid)],crs=3857,geometry=0).plot(color='green',ax=ax)
    ax.set_title(f"N° cell: {grid.shape[0]}")

    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    plt.savefig("Italy.png",bbox_inches='tight')
    plt.close()

    fig,ax=plt.subplots(1,1,figsize=(16,12),
                        subplot_kw={'projection':ccrs.epsg(3857)})
    INT_ul_min[INT_ul_min.row_id==0.0].plot(edgecolor='black', facecolor='none',ax=ax)
    INT_ul_min[INT_ul_min.row_id==1111.0].plot(edgecolor='black', facecolor='none',ax=ax)
    grid[grid.index.isin(INT_ul_min[INT_ul_min.row_id==0.0]['index'].values)].plot(edgecolor='green',linewidth=2, facecolor='none',ax=ax)
    grid[grid.index.isin(INT_ul_min[INT_ul_min.row_id==1111.0]['index'].values)].plot(edgecolor='green',linewidth=2, facecolor='none',ax=ax)
    ita.boundary.plot(edgecolor='red', facecolor='none',ax=ax)
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    plt.savefig("2024_dis_io/out/fig/two_ul.png",bbox_inches='tight')
    plt.close()



    fig,ax=plt.subplots(1,1,figsize=(16,12),
                        subplot_kw={'projection':ccrs.epsg(3857)})
    INT_ul_min[INT_ul_min.row_id==3594032.0].plot(edgecolor='black', facecolor='none',ax=ax)
    grid[grid.index.isin(INT_ul_min['index'].values)].plot(edgecolor='blue', facecolor='none',ax=ax)
    grid[grid.index.isin(INT_ul_min[INT_ul_min.row_id==3594032.0]['index'].values)].plot(edgecolor='green',linewidth=2, facecolor='none',ax=ax)
    ita.boundary.plot(edgecolor='red', facecolor='none',ax=ax)
    ax.coastlines()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    plt.savefig("2024_dis_io/out/fig/ul_int.png",bbox_inches='tight')
    plt.close()