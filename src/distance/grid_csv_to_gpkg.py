#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import json
import numpy as np
import pandas as pd
from glob import glob
import geopandas as gpd
import cartopy.crs as ccrs
from shapely.geometry import Polygon
import matplotlib.patches as patches

from datetime import datetime
from matplotlib import gridspec
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from quadtrees import Point, Rect, QuadTree


import argparse

parser = argparse.ArgumentParser(
    description="Creation of real network and introducing the distance."
)
# parser.add_argument(
#     "-pm","--plot_maps", action="store_true", help="Visualize the graph on a map."
# )
# parser.add_argument(
#     "--speed", action="store_true", help="load with geopandas or JSON"
# )
# parser.add_argument(
#    "-nodes", default=100, type=int, help="Number of nodes"
#    )
parser.add_argument(
    "--n_ul", default=10000, type=int, choices=[1000,10000,100000],
    help="Number of edges as a multiplier of the nodes (i.e., How many edges are expected per node)."
   )

args = parser.parse_args()
N_UL_SQUARE=args.n_ul
filename=glob(f'/mnt/beegfs/lcesarini/ge-distance/res/*coords_rect_{N_UL_SQUARE}_*')
x=pd.read_csv(filename[0],header=None)

rectangle=[]
for row in x.itertuples():
    bottom_left = ( row[1], row[2])
    top_right = ( row[3], row[4] )

    # Derive the other two corners
    top_left = ( row[1], row[4] )
    bottom_right = ( row[3], row[2] )

    # Create the rectangle polygon
    rectangle.append(Polygon([bottom_left, bottom_right, top_right, top_left, bottom_left]))

gdf = gpd.GeoDataFrame({'geometry': rectangle},crs=4326)
gdf.reset_index(inplace=True)

containment_matrix = gdf.apply(lambda row: gdf.geometry.contains(row.geometry), axis=1)
indices_to_keep=np.array([ containment_matrix.loc[:,i].sum() for i in range(containment_matrix.shape[1]) ]) == 1
gdf=gdf[indices_to_keep].reset_index(drop=True).drop(columns='index').reset_index()
gdf.to_file(f'/mnt/beegfs/lcesarini/ge-distance/src/distance/out/grid_{gdf.shape[0]}.gpkg',driver='GPKG')
print(gdf.shape)

#plot gdf 
fig,ax=plt.subplots(1,1,figsize=(16,12),subplot_kw={'projection':ccrs.PlateCarree()})
gdf.boundary.plot(edgecolor='red', facecolor='none',ax=ax)
# grid.iloc[[I,J]].plot(edgecolor='black', facecolor='none',ax=ax)
ax.set_title(f"N° cell: {gdf.shape[0]}")
ax.coastlines()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
plt.savefig(f"/mnt/beegfs/lcesarini/ge-distance/src/distance/tmp/grid_{gdf.shape[0]}.png",bbox_inches='tight')
plt.close()
