#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from quadtrees import Point, Rect, QuadTree
from matplotlib import gridspec
from datetime import datetime

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
    "--n_ul", default=10000, type=int,
    help="Number of edges as a multiplier of the nodes (i.e., How many edges are expected per node)."
   )

args = parser.parse_args()



with open('/mnt/beegfs/lcesarini/2024_dis_io/out/UL_italy_poly_assigned.json', 'r', encoding='utf-8') as f:
    r=json.load(f) 

gdf=json.loads(r)

# with open('/mnt/beegfs/lcesarini/2024_dis_io/res/Lombardia_geocoded.json', 'r', encoding='utf-8') as f:
#     r2=json.load(f) 

# gdf2=json.loads(r2)

if os.path.exists(f'/mnt/beegfs/lcesarini/2024_dis_io/res/test.csv'):
    os.remove(f'/mnt/beegfs/lcesarini/2024_dis_io/res/test.csv')
    with open(f'/mnt/beegfs/lcesarini/2024_dis_io/res/test.csv', 'w') as f:
        pass

prop_keys=gdf['features'][0]['properties'].keys()
if ("lng" in prop_keys) & ("lat" in prop_keys):
    name_lon="lng"
    name_lat="lat"
elif ("lon" in prop_keys) & ("lat" in prop_keys):
    name_lon="lon"
    name_lat="lat"
else:
    raise ValueError("No lon/lat keys found in properties")


lon=[]
lat=[]

for f in gdf['features']:
    lon.append(f['properties'][name_lon])
    lat.append(f['properties'][name_lat])

# for f in gdf2['features']:
#     lon.append(f['properties'][name_lon])
#     lat.append(f['properties'][name_lat])

POINTS=np.transpose(np.array([lon,lat]))
print(POINTS.shape)
N_UL_SQUARE=args.n_ul
coords = POINTS#np.random.randn(N, 2) * height/3 + (width/2, height/2)
points = [Point(*coord) for coord in coords]

width, height = 11.842856472197115, 11.588787199999999
center_x,center_y=12.588848036098558, 41.2889397

domain = Rect(center_x,center_y, width, height)


if __name__ == '__main__':
    TIMESTAMP=datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print('1')
    qtree = QuadTree(domain, N_UL_SQUARE)
    print('2')
    for point in points:
        qtree.insert(point)
    print('3')
    ax = plt.subplot()

    qtree.draw(ax,n_ul=N_UL_SQUARE,timestamp=TIMESTAMP)
    plt.savefig(f"/mnt/beegfs/lcesarini/ge-distance/src/distance/tmp/qdtree_{TIMESTAMP}.png")

    print(f'With {N_UL_SQUARE} UL per square')
