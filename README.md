# GE-Distance



The ge-distance package builds upon the graph ensemble package, a project by a project by [Leonardo Niccolò Ialongo](https://datasciencephd.eu/students/leonardo-niccol%C3%B2-ialongo/) and [Emiliano Marchese](https://www.imtlucca.it/en/emiliano.marchese/), under the supervision of [Diego Garlaschelli](https://networks.imtlucca.it/people/diego), which contains a set of methods to build fitness based graph ensembles from marginal information.

The graph-ensemble repository can be found at:
[https://github.com/LeonardoIalongo/graph-ensembles](https://github.com/LeonardoIalongo/graph-ensembles)


<img src="https://github.com/luigicesarini/ge-distance/blob/master/src/distance/tmp/graph_sampled_100_300_ensemble_test.gif" alt="Network GIF" style="height: 50%; width:40%;border:solid 1px #555; float: left; margin-right: 15px;"/>


## Project Description
This project aims at reproducing a firm-to-firm network for the entire italian country, introducing the distance as a property that influences the construction of such network.



## TODO List
List of the taks envisioned for bringing to completion the project:

[  ] _Introducing the distance as dyadic properties_  
[  ] _Scale the creation of ensemble to the 5mln nodes in the firm-to-firm network._  
[  ]

## Features
The repository contains a module, __distance__, where starting from the locations of the firms, compute the distance matrix among firms, in two steps:

1. Discretize the interested area, dividing it into square with similar firm density inside based on a [quadtree](https://en.wikipedia.org/wiki/Quadtree) approach. [Example grid](https://github.com/luigicesarini/ge-distance/blob/master/src/distance/tmp/grid_925.png)
2. Compute the distance (geodetic for now, street distance when psosible) between each cell pair, and attach to each firm, the corresponding _cell_id_.

The _cell_id_ is later used to retrieve the distance for each i,j pair of firms from the distance matrix, when creating the ensemble.

As of now, in src/distance/out/ can be found 3  __arr_input_ul_*.npz__ and __dist_mat_*.npz__ depending on the resolution (i.e., 95 for 95 cells discretizing the area, 925 higher, and etc.,).

In the input array we found the univocal id of the firm, the cell_id belonging to the firm, and the sector of the firm.

The distance matrix is a symmetric matrix 95*95 (or 925x925, etc.,). 

MISSING THE NUMBER OF EMPLOY PER FIRM that I will add to the input.


## Installation
TO COMPLETE ONCE FINISHED

**Clone the repository**

    git clone https://github.com/lcesarini/ge-distance.git
    cd ge-distance


## Usage
Some examples to put

    python your_script.py --arg1 value --arg2 value

## Contributing
All the commit,pull, merge stuff that are unnecessary now.

## License
This project is licensed under the GNU GENERAL PUBLIC LICENSE License. See the [LICENSE](https://github.com/luigicesarini/ge-distance/blob/master/LICENSE.txt) file for more details. 

## Contact
Research center emails, personal, and what not
