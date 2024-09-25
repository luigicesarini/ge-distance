#!/home/luigi.cesarini/.conda/envs/my_xclim_env/bin/python
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from glob import glob
from tqdm import tqdm

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.sparse import csr_array,csc_array,save_npz,load_npz

sys.path.append("/mnt/beegfs/lcesarini/ge-distance/src")

os.chdir(f"/mnt/beegfs/lcesarini/ge-distance/")
from graph_ensembles.sparse import MultiFitnessModel as MFM

import argparse


np.random.seed(111)
parser = argparse.ArgumentParser(
    description="Creation of real network and introducing the distance."
)
parser.add_argument(
    "-pm","--plot_maps", action="store_true", help="Visualize the graph on a map."
)
parser.add_argument(
    "--speed", default='json', type=str, 
    help="load with geopandas, JSON, or sparse_matrix",choices=['gpkg','json','sparse']
)
parser.add_argument(
   "-nodes", default=100, type=int, help="Number of nodes"
   )
parser.add_argument(
   "-ce","--coeff_edges", default=100, type=int,
    help="Number of edges as a multiplier of the nodes (i.e., How many edges are expected per node)."
   )
parser.add_argument(
   "-jn","--job_name", default='example', type=str,
    help="The slurm job name running the script"
   )
parser.add_argument(
    "-ng","--n_grid", default=95, type=int, choices=[95,925,9174],
    help="Number of squares in the grid"
)
# parser.add_argument("-xtol", default=1e-6, type=float)

args = parser.parse_args()

PLOT=args.plot_maps
NODES=args.nodes
COEFF_EDGES=args.coeff_edges
SLOW=args.speed
DEBUG=False
JOB_NAME=args.job_name
n_grid=args.n_grid

if DEBUG:
    PLOT=True
    NODES=100
    COEFF_EDGES=4
    SLOW='sparse'
# import graph_ensembles.methods as ge
# import graph_ensembles.iterative_models as gei
print("\n-------------------\n")
print(NODES,COEFF_EDGES,SLOW,PLOT)
# Load the data for vertices
if SLOW=="gpkg":
    print("We're going slow")

    gdf=gpd.read_file(f'/src/distance/out/UL_italy_poly_assigned_{n_grid}.gpkg')
    gdf['lng']=gdf.geometry.x
    gdf['lat']=gdf.geometry.y

    sub_g=gdf[:NODES]
    longitude=sub_g.lng.values
    latitude=sub_g.lat.values

    N_VERT=sub_g.Sectors.shape[0]
    N_EDGE=sub_g.Sectors.shape[0] * COEFF_EDGES
    N_SECT=sub_g.Sectors.unique().shape[0]
elif SLOW=="json":
    print("We're going faster")
    with open(f'/src/distance/out/UL_italy_poly_assigned_{n_grid}.json', 'r', encoding='utf-8') as f:
        r=json.load(f) 

    gdf=json.loads(r)
    lon=[]
    lat=[]
    sectors=[]
    for f in (gdf['features']):
        lon.append(f['properties']['lon'])
        lat.append(f['properties']['lat'])
        sectors.append(f['properties']['Sectors'])

    N_VERT=len(lon[:NODES])
    N_EDGE=len(lat[:NODES])  * COEFF_EDGES
    N_SECT=np.unique(sectors[:NODES]).shape[0]
elif SLOW=="sparse":
    print("We're going super fast")
    arr_ul=load_npz(f'src/distance/out/arr_input_ul_{n_grid}.npz')

    sample_nodes=np.random.randint(0,arr_ul.shape[0],NODES)
    arr_ul=arr_ul[sample_nodes]

    N_VERT=arr_ul.shape[0]
    N_EDGE=arr_ul.shape[0] * COEFF_EDGES
    N_SECT= np.unique(arr_ul.toarray()[:,1]).shape[0]

print("\n-------------------\n")
print(N_VERT,N_EDGE,N_SECT)
print("\n-------------------\n")

PROP_IN=np.random.randint(1,6,N_VERT * N_SECT).reshape(-1,N_SECT)
PROP_OUT=np.random.randint(0,5,N_VERT * N_SECT).reshape(-1,N_SECT)

gmfm=MFM(
        num_vertices=N_VERT,
        num_edges=N_EDGE,
        num_labels=N_SECT,
        prop_out=PROP_OUT,
        prop_in=PROP_IN,
        param=1,
        id_grid=arr_ul[:,2].toarray(),
        n_ul=95,
        # param=np.random.randint(1,10,N_SECT),
        # num_edges_label=np.random.randint(2,5,N_SECT),
        # selfloops=1,
        # per_label=1,
    )

#get the distance matrix
gmfm.get_dist_matrix()
# print(f"Number of local unit:{gmfm.get_dist_matrix()}")
# print(gmfm.dist_mat)


test_i=np.random.randint(0,N_VERT)
test_j=np.random.randint(0,N_VERT)

if test_i==test_j:
    test_j+=1

# print("pre computation distance")
# print(f"\n---------------------\n")



print("\n-------------------\n")
print(f"Distance {test_i},{test_j}: {gmfm.prop_dyad(test_i,test_j,gmfm.id_grid,gmfm.dist_mat):.2f}")
print(f"\n---------------------\n")
# print("post computation distance")

# gmfm.fit(verbose=False)

# dir(gmfm.solver_output)

# print(gmfm.solver_output.x)
# gmfm.solver_output.target
# gmfm.solver_output.f_seq

# print(f"exp_av_nn_prop:{gmfm.exp_av_nn_prop()}")
# print(f"expected_out_degree:{gmfm.expected_out_degree()}")
# print(f"\n---------------------\n")
# print(f"expected_out_degree_by_label:{gmfm.expected_out_degree_by_label()}")
# print(f"expected_in_degree:{gmfm.expected_in_degree()}")
# print(f"expected_in_degree_by_label:{gmfm.expected_in_degree_by_label()}")
# print(f"expected_num_edges:{gmfm.expected_num_edges()}")
# print(f"\n---------------------\n")
# print(f"expected_num_edges_label:{gmfm.expected_num_edges_label()}")

# for ele in dir(gmfm):
#     if "__" not in ele:
#         print(ele) 

# x=gmfm.sample()
# sampled_nx=x.to_networkx()

# print(f"N° vertices: {N_VERT}")
# print(f"N° edges: {N_EDGE}")
# print(f"N° sectors: {N_SECT}")
# print(f"S_in: {PROP_IN.max():.2f}")
# print(f"S_out: {PROP_OUT.max():.2f}")


if PLOT:
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap as Basemap
    gif_filename = f'src/distance/tmp/graph_sampled_{N_VERT}_{N_EDGE}_{JOB_NAME}.gif'

    with open(f'./src/distance/out/UL_italy_poly_assigned_{n_grid}.json', 'r', encoding='utf-8') as f:
        r=json.load(f) 

    gdf=json.loads(r)
    lon=[]
    lat=[]
    for f in (gdf['features']):
        lon.append(f['properties']['lng'])
        lat.append(f['properties']['lat'])

    # lon=lon[:NODES]
    # lat=lat[:NODES]
    lon=np.array(lon)[sample_nodes]
    lat=np.array(lat)[sample_nodes]

    for f in glob(f"src/distance/tmp/*_{N_VERT}_{N_EDGE}_{JOB_NAME}.png"):
        os.remove(f)

    if os.path.exists(gif_filename):
        os.remove(gif_filename)
    else:
        print("The file does not exist")
    
    # dict_pos={x:[np.random.uniform(10,15),np.random.uniform(40,45)] for i,x in enumerate(sampled_nx.nodes)}
    for _ in range(10):

        SAMPLED_NX=gmfm.sample().to_networkx()
        fig,ax=plt.subplots(1,1,figsize=(16,16))
        m = Basemap(
                projection='merc',
                llcrnrlon=5,#np.min(lon)*0.995,
                llcrnrlat=35,#np.min(lat)*0.9995,
                urcrnrlon=20,#np.max(lon)*1.005,
                urcrnrlat=47.2,#np.max(lat)*1.0005,
                lat_ts=0,
                resolution='l',
                ax=ax,
                suppress_ticks=False)
        # m.etopo()
        m.drawcountries(linewidth = 3);
        m.drawstates(linewidth = 0.2)
        m.drawcoastlines(linewidth=3)
        m.fillcontinents(color='coral',lake_color='aqua')
        m.drawmapboundary(fill_color='aqua') 
        dict_pos={x:list(m(lon[i],lat[i])) for i,x in enumerate(SAMPLED_NX.nodes)}

        # nx.draw(SAMPLED_NX,
        #         pos=dict_pos,
        #         alpha=0.5,
        #         edge_color='green',
        #         node_color='red',node_size=30,
        #         ax=ax,with_labels=True);
        
        nx.draw_networkx_nodes(G = SAMPLED_NX, pos = dict_pos, nodelist = SAMPLED_NX.nodes(), 
                        node_color = 'r', alpha = 0.8, node_size = 100, label=True)
        nx.draw_networkx_edges(G = SAMPLED_NX, pos = dict_pos, edge_color='g',
                                alpha=0.2, arrows = True)
        plt.tight_layout()
        plt.savefig(f"src/distance/tmp/sample_{_}_{N_VERT}_{N_EDGE}_{JOB_NAME}.png", dpi = 300)
        plt.close()


    #CREATE the GIF
    # Directory containing PNG images
    png_dir = 'src/distance/tmp/'
    list_png=glob(f"{png_dir}/sample*_{N_VERT}_{N_EDGE}_{JOB_NAME}.png")
    # List PNG files in the directory
    list_png.sort()
    # Create GIF filename

    frames=[]
    for img in list_png[:10]:
        frames.append(imageio.imread(img))
    imageio.mimsave(gif_filename, frames, 'GIF', duration=250)

    print("GIF created successfully.")