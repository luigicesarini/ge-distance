""" This module defines the classes of graphs that support the creation of
ensembles and the exploration of their properties.

This version stores all properties of the graphs as numpy arrays and is 
recommended for small or dense graphs. For large sparse graphs consider the 
sparse module or the spark one. 
"""

import numpy as np
import pandas as pd
from . import lib
import warnings
import networkx as nx
from numba import jit


class Graph():
    """ General class for undirected graphs. 

    Note that edges can be weighted or not. If they are not, the strengths 
    will be equal to the degrees. The class does not check for the uniqueness 
    of the links definitions. If a link is provided multiple times with 
    weights, they will be summed. This also applies to the case where both the
    (i, j, weight) and symmetric (j, i, weight) are provided.

    Attributes
    ----------
    num_vertices: int
        number of vertices in the graph
    num_edges: int
        number of distinct directed edges in the graph
    adj: numpy.array
        the adjacency matrix of the graph
    id_dict: dict
        dictionary to convert original identifiers to new position id
    id_dtype: numpy.dtype
        type of the id (e.g. np.uint16)
    num_groups: int  (or None)
        number of vertex groups
    group_dict: dict (or none)
        dictionary to convert v_group columns into numeric ids
    group_dtype: numpy.dtype
        type of the group id
    groups: numpy.array
        array with the group each node belongs to
    total_weight: numpy.float64
        sum of all the weights of the edges

    Methods
    -------
    degree:
        compute the undirected degree sequence
    degree_by_group:
        compute the undirected degree sequence by group as a 2D array
    strength:
        compute the total strength sequence
    strength_by_group:
        compute the total strength sequence by group as a 2D array
    adjacency_matrix:
        return the adjacency matrix of the graph
    to_networkx:
        return a Networkx equivalent
    """
    def __init__(self, v, e, v_id, src, dst, weight=None, v_group=None):
        """Return a Graph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str or list of str
            identifier column for the source vertex
        dst: str or list of str
            identifier column for the destination vertex
        weight: str or None
            identifier column for the weight of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        Graph
            the graph object
        """
        assert isinstance(v, pd.DataFrame), 'Only dataframe input supported.'
        assert isinstance(e, pd.DataFrame), 'Only dataframe input supported.'

        # If column names are passed as lists with one elements extract str
        if isinstance(v_id, list) and len(v_id) == 1:
            v_id = v_id[0]
        if isinstance(src, list) and len(src) == 1:
            src = src[0]
        if isinstance(dst, list) and len(dst) == 1:
            dst = dst[0]

        # Determine size of indices
        self.num_vertices = len(v.index)
        num_bytes = self.get_num_bytes(self.num_vertices)
        self.id_dtype = np.dtype('u' + str(num_bytes))

        # Get dictionary of id to internal id (_id)
        # also checks that no id in v is repeated
        try:
            if isinstance(v_id, list):
                v['_node'] = list(zip(*[v[x] for x in v_id]))
            else:
                v['_node'] = v[v_id]

            self.id_dict = self.generate_id_dict(v, '_node', check_unique=True)

            # Create index with new id value and sort
            v = v.set_index(v['_node'].apply(
                lambda x: self.id_dict.get(x)).values)
            v = v.sort_index()

        except ValueError as err:
            raise err
        except Exception:
            rep_msg = ('There is at least one repeated id in the vertex '
                       'dataframe.')
            raise Exception(rep_msg)

        # If v_group is given then create dict and array
        if v_group is not None:
            if isinstance(v_group, list) and len(v_group) == 1:
                v_group = v_group[0]

            if isinstance(v_group, list):
                v['_group'] = list(zip(*[v[x] for x in v_group]))
            else:
                v['_group'] = v[v_group]

            self.group_dict = self.generate_id_dict(v, '_group')
            self.num_groups = len(self.group_dict)
            num_bytes = self.get_num_bytes(self.num_groups)
            self.group_dtype = np.dtype('u' + str(num_bytes))
            self.groups = v['_group'].apply(
                lambda x: self.group_dict.get(x)
                ).values.astype(self.group_dtype)

        # Check that no vertex id in e is not present in v
        # and generate optimized edge list
        if isinstance(src, list) and isinstance(dst, list):
            e['_src'] = list(zip(*[e[x] for x in src]))
            e['_dst'] = list(zip(*[e[x] for x in dst]))
        elif isinstance(src, str) and isinstance(dst, str):
            e['_src'] = e[src]
            e['_dst'] = e[dst]
        else:
            raise ValueError('src and dst can be either both lists or str.')

        msg = 'Some vertices in e are not in v.'
        try:
            e['_src'] = e['_src'].apply(lambda x: self.id_dict.get(x))
            src_array = e['_src'].values.astype(self.id_dtype)
            e['_dst'] = e['_dst'].apply(lambda x: self.id_dict.get(x))
            dst_array = e['_dst'].values.astype(self.id_dtype)
        except KeyError:
            raise Exception(msg)
        except Exception:
            raise Exception()

        # Construct adjacency matrix
        if weight is not None:            
            val_array = e[weight].values
            self.adj = np.zeros((self.num_vertices, self.num_vertices), 
                                dtype=val_array.dtype)
            self.adj[src_array, dst_array] = val_array
        else:
            self.adj = np.zeros((self.num_vertices, self.num_vertices), 
                                dtype=bool)
            self.adj[src_array, dst_array] = True

        # Compute number of edges considering self loop
        self.num_edges = self.get_num_edges()

        if weight is not None:
            # Compute total weight
            self.total_weight = self.get_total_weight

    def adjacency_matrix(self, directed=False, weighted=False):
        """ Return the adjacency matrix of the graph.
        """
        # Ensure matrix is symmetric as this is undirected
        if weighted:
            # Symmetrize
            adj = self.adj + self.adj.T

            # Remove double count diagonal
            adj.ravel()[::adj.shape[1]+1] = np.diag(self.adj)

        else:
            adj = self.adj != 0
            adj = adj | adj.T

        return adj

    def get_num_edges(self, recompute=False):
        """ Compute the number of edges.
        """
        if not hasattr(self, 'num_edges') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self.num_edges = (adj.sum() + np.diag(adj).sum()) / 2

        return self.num_edges

    def get_total_weight(self, recompute=False):
        """ Compute the sum of all the weights.
        """
        if not hasattr(self, 'total_weight') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=True)
            self.total_weight = (adj.sum() + np.diag(adj).sum()) / 2

        return self.total_weight

    def degree(self, recompute=False):
        """ Compute the undirected degree sequence.
        """
        if not hasattr(self, '_degree') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self._degree = adj.sum(axis=0)

        return self._degree

    def degree_by_group(self, recompute=False):
        """ Compute the undirected degree sequence to and from each group.
        """
        if not hasattr(self, '_degree_by_group') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self._degree_by_group = self.sum_by_group(adj, self.groups)

        return self._degree_by_group

    def strength(self, recompute=False):
        """ Compute the total strength sequence.
        """
        if not hasattr(self, '_strength') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=False)
            self._strength = adj.sum(axis=1)

        return self._strength

    def strength_by_group(self, recompute=False):
        """ Compute the total strength sequence to and from each group.
        """
        if not hasattr(self, '_strength_by_group') or recompute:
            adj = self.adjacency_matrix(directed=False, weighted=True)
            self._strength_by_group = self.sum_by_group(adj, self.groups)

        return self._strength_by_group

    def to_networkx(self):
        """ Return a networkx Graph object for this graph.
        """
        # Initialize Graph object
        G = nx.Graph(self.adj)

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, 'groups'):
            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr)

        return G

    @staticmethod
    def get_num_bytes(num_items):
        """ Determine the number of bytes needed for storing ids for num_items.
        """
        return int(max(2**np.ceil(np.log2(np.log2(num_items + 1)/8)), 1))

    @staticmethod
    def generate_id_dict(df, id_col, check_unique=False):
        """ Return id dictionary for given dataframe columns.
        """
        id_dict = {}

        if check_unique:
            ids, counts = np.unique(df[id_col], return_counts=check_unique)
            assert np.all(counts == 1)
        else:
            ids = np.unique(df[id_col])

        for i, x in enumerate(ids):
            id_dict[x] = i

        return id_dict

    @staticmethod
    @jit(nopython=True)
    def sum_by_group(mat, g_arr):
        """ Sums the values of the matrix along the second axis using the 
        provided grouping.
        """
        # Initialize empty result
        M = g_arr.max() + 1
        res = np.zeros((mat.shape[0], M), dtype=np.int64)
        
        # Count group occurrences
        for i in range(M):
            res[:, i] = mat[:, np.where(g_arr == i)[0]].sum(axis=1)
            
        return res













class DiGraph(Graph):
    """ General class for directed graphs.

    Methods
    -------
    out_degree:
        compute the out degree sequence
    in_degree:
        compute the in degree sequence
    out_degree_by_group:
        compute the out degree sequence by group as a 2D array
    in_degree_by_group:
        compute the in degree sequence by group as a 2D array
    adjacency_matrix:
        return the directed adjacency matrix of the graph
    to_networkx:
        return a networkx DiGraph equivalent

    out_strength:
        compute the out strength sequence
    in_strength:
        compute the in strength sequence
    out_strength_by_group:
        compute the out strength sequence by group as a 2D array
    in_strength_by_group:
        compute the in strength sequence by group as a 2D array
    """

    def __init__(self, v, e, v_id, src, dst, v_group=None):
        """Return a DirectedGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str or list of str
            identifier column for the source vertex
        dst: str or list of str
            identifier column for the destination vertex
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        DirectedGraph
            the graph object
        """
        super().__init__(v, e, v_id=v_id, src=src, dst=dst, v_group=v_group)

        # Compute degree (undirected)
        d = self.degree()

        # Warn if vertices have no edges
        zero_idx = np.nonzero(d == 0)[0]
        if len(zero_idx) == 1:
            warnings.warn(str(list(self.id_dict.keys())[zero_idx[0]]) +
                          " vertex has no edges.", UserWarning)

        if len(zero_idx) > 1:
            names = []
            for idx in zero_idx:
                names.append(list(self.id_dict.keys())[idx])
            warnings.warn(str(names) + " vertices have no edges.",
                          UserWarning)

    def adjacency_matrix(self, directed=True, weighted=False):
        """ Return the adjacency matrix of the graph.
        """
        if directed:
            return self.adj

    def degree(self, recompute=False):
        """ Compute the undirected degree sequence.
        """
        if not hasattr(self, '_degree') or recompute:
            adj = self.adjacency_matrix(undirected=True, weighted=False)
            self._degree = adj.sum(axis=0)

        return self._degree

    def out_degree(self, recompute=False):
        """ Compute the out degree sequence.
        """
        if not hasattr(self, '_out_degree') or recompute:
            adj = (self.adj != 0).astype(np.uint8)
            self._out_degree = adj.sum(axis=1)
            self._in_degree = adj.sum(axis=0)

        return self._out_degree

    def in_degree(self, recompute=False):
        """ Compute the in degree sequence.
        """
        if not hasattr(self, '_in_degree') or recompute:
            adj = (self.adj != 0).astype(np.uint8)
            self._out_degree = adj.sum(axis=1)
            self._in_degree = adj.sum(axis=0)

        return self._in_degree

    def out_degree_by_group(self, recompute=False):
        """ Compute the out degree sequence to each group.
        """
        if not hasattr(self, '_out_degree_by_group') or recompute:
            adj = self.adj.copy()
            adj[adj != 0] = 1
            self._out_degree_by_group = self.sum_by_group(adj, self.groups)
            self._in_degree_by_group = self.sum_by_group(adj.T, self.groups)

        return self._out_degree_by_group

    def in_degree_by_group(self, recompute=False):
        """ Compute the in degree sequence from each group.
        """
        if not hasattr(self, '_in_degree_by_group') or recompute:
            adj = self.adj.copy()
            adj[adj != 0] = 1
            self._out_degree_by_group = self.sum_by_group(adj, self.groups)
            self._in_degree_by_group = self.sum_by_group(adj.T, self.groups)

        return self._in_degree_by_group

    def adjacency_matrix(self):
        """ Return the adjacency matrix of the graph.
        """
        adj = self.adj.copy()
        adj[adj != 0] = 1
        return adj

    def to_networkx(self, original=False):
        """ Return a networkx DiGraph object for this graph.
        """
        # Initialize DiGraph object
        G = nx.DiGraph(self.adj)

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, 'groups'):
            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr)

        return G



    def out_strength(self, recompute=False):
        """ Compute the out strength sequence.
        """
        if not hasattr(self, '_out_strength') or recompute:
            self._out_strength = self.adj.sum(axis=1)

        return self._out_strength

    def in_strength(self, recompute=False):
        """ Compute the in strength sequence.
        """
        if not hasattr(self, '_in_strength') or recompute:
            self._in_strength = self.adj.sum(axis=0)

        return self._in_strength

    def out_strength_by_group(self, recompute=False):
        """ Compute the out strength sequence to each group.
        """
        if not hasattr(self, '_out_strength_by_group') or recompute:
            self._out_strength_by_group = self.sum_by_group(
                self.adj, self.groups)

        return self._out_strength_by_group

    def in_strength_by_group(self, recompute=False):
        """ Compute the in strength sequence from each group.
        """
        if not hasattr(self, '_in_strength_by_group') or recompute:
            self._in_strength_by_group = self.sum_by_group(
                self.adj.T, self.groups)

        return self._in_strength_by_group

    def weighted_adjacency_matrix(self):
        """ Return the weighted adjacency matrix.
        """
        return self.adj



class LabelGraph(DirectedGraph):
    """ General class for directed graphs with labelled edges.

    Attributes
    ----------
    num_labels: int
        number of distinct edge labels
    num_edges_label: numpy.array
        number of edges by label (in order)
    label_dtype: numpy.dtype
        the data type of the label internal id

    Methods
    -------
    degree_by_label:
        compute the degree of each vertex by label
    out_degree_by_label:
        compute the out degree of each vertex by label
    in_degree_by_label:
        compute the in degree of each vertex by label
    """
    def __init__(self, v, e, v_id, src, dst, edge_label, v_group=None):
        """Return a LabelGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str (list of str not yet supported)
            identifier column for the source vertex
        dst: str (list of str not yet supported)
            identifier column for the destination vertex
        edge_label:
            identifier column for label of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        LabelGraph
            the graph object
        """
        super(DirectedGraph, self).__init__(v, e, v_id=v_id, src=src, dst=dst,
                                            v_group=v_group)

        # If column names are passed as lists with one elements extract str
        if isinstance(edge_label, list) and len(edge_label) == 1:
            edge_label = edge_label[0]

        # Get dictionary of label to numeric internal label
        if isinstance(edge_label, list):
            e['_label'] = list(zip(*[e[x] for x in edge_label]))
        else:
            e['_label'] = e[edge_label]

        self.label_dict = self.generate_id_dict(e, '_label')
        self.num_labels = len(self.label_dict.keys())
        num_bytes = self.get_num_bytes(self.num_labels)
        self.label_dtype = np.dtype('u' + str(num_bytes))

        # Convert labels to construct adjacency matrix by layer
        e['_label'] = e['_label'].apply(lambda x: self.label_dict.get(x))
        lbl_array = e['_label'].values.astype(self.label_dtype)
        src_array = e['_src'].values.astype(self.id_dtype)
        dst_array = e['_dst'].values.astype(self.id_dtype)

        # Define adjaceny tensor as 3D array with dimensions (label, src, dst)
        self.adj_tensor = np.zeros(
            (self.num_labels, self.num_vertices, self.num_vertices), 
            dtype=np.uint8)
        self.adj_tensor[lbl_array, src_array, dst_array] = 1

        # Compute number of edges by label
        self.num_edges_label = self.adj_tensor.sum(axis=[1, 2])

    def degree_by_label(self, recompute=False):
        """ Compute the degree sequence by label.
        """
        if not hasattr(self, '_degree_by_label') or recompute:
            sym_adj = self.adj_tensor + self.adj_tensor.T
            sym_adj[sym_adj != 0] = 1
            self._degree_by_label = sym_adj.sum(axis=2)

        return self._degree_by_label

    def out_degree_by_label(self, recompute=False):
        """ Compute the out degree sequence by label.
        """
        if not hasattr(self, '_out_degree_by_label') or recompute:
            adj = self.adj_tensor.copy()
            adj[adj != 0] = 1
            self._out_degree_by_label = adj.sum(axis=2)
            self._in_degree_by_label = adj.sum(axis=1)

        return self._out_degree_by_label

    def in_degree_by_label(self, recompute=False):
        """ Compute the in degree sequence by label.
        """
        if not hasattr(self, '_in_degree_by_label') or recompute:
            adj = self.adj_tensor.copy()
            adj[adj != 0] = 1
            self._out_degree_by_label = adj.sum(axis=2)
            self._in_degree_by_label = adj.sum(axis=1)

        return self._in_degree_by_label

    def adjacency_matrix(self):
        """ Return the adjacency matrix.
        """
        return self.adj

    def adjacency_tensor(self):
        """ Return the adjacency tensor.
        """
        return self.adj_tensor

    def to_networkx(self):
        """ Return a networkx MultiDiGraph equivalent object for this graph.
        """
        # Initialize object
        G = nx.MultiDiGraph()

        # Add original node ids
        for node_id, i in self.id_dict.items():
            G.add_node(i, node_id=node_id)

        # If present add group info
        if hasattr(self, 'groups'):
            for i, gr in enumerate(self.groups):
                G.add_node(i, group=gr)

        # Add edges per layer
        for lbl in range(self.num_labels):
            G.add_edges_from(zip(row, col), label=lbl)

        return G


class WeightedLabelGraph(WeightedGraph, LabelGraph):
    """ General class for directed graphs with labelled and weighted edges.

    Attributes
    ----------
    total_weight_label: numpy.array
        sum of all the weights of the edges by label

    Methods
    -------
    strength_by_label:
        compute the strength of each vertex by label
    out_strength_by_label:
        compute the out strength of each vertex by label
    in_strength_by_label:
        compute the in strength of each vertex by label
    """
    def __init__(self, v, e, v_id, src, dst, weight, edge_label, v_group=None):
        """Return a WeightedLabelGraph object given vertices and edges.

        Parameters
        ----------
        v: pandas.dataframe
            list of vertices and their properties
        e: pandas.dataframe
            list of edges and their properties
        v_id: str or list of str
            specifies which column uniquely identifies a vertex
        src: str (list of str not yet supported)
            identifier column for the source vertex
        dst: str (list of str not yet supported)
            identifier column for the destination vertex
        weight:
            identifier column for the weight of the edges
        edge_label:
            identifier column for label of the edges
        v_group: str or list of str or None
            identifier of the group id of the vertex

        Returns
        -------
        sGraph
            the graph object
        """
        LabelGraph.__init__(self, v, e, v_id=v_id, src=src, dst=dst,
                            edge_label=edge_label, v_group=v_group)

        if isinstance(weight, list) and len(weight) == 1:
            weight = weight[0]

        # Convert weights to float64 for computations
        self.e = append_fields(
            self.e,
            'weight',
            e[weight].to_numpy().astype(np.float64)[self.sort_ind],
            dtypes=np.float64)

        # Ensure all weights are positive
        msg = 'Zero or negative edge weights are not supported.'
        assert np.all(self.e.weight > 0), msg

        # Compute total weight
        self.total_weight = np.sum(self.e.weight)

        # Compute total weight by label
        self.total_weight_label = mt.compute_tot_weight_by_label(
            self.e, self.num_labels)

    def strength_by_label(self, get=False):
        """ Compute the out strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'strength'):
            s, s_dict = mt.compute_strength_by_label(self.e)

            self.lv.strength = s.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )

            self.lv.strength.sort()
            self.lv.strength_dict = s_dict

        if get:
            return self.lv.strength

    def out_strength_by_label(self, get=False):
        """ Compute the out strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'out_strength'):
            s_out, s_in, sout_dict, sin_dict = \
                mt.compute_in_out_strength_by_label(self.e)

            self.lv.out_strength = s_out.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s_out.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )
            self.lv.in_strength = s_in.view(
                type=np.recarray,
                dtype=[('label', 'f8'), ('id', 'f8'), ('value', 'f8')]
                ).reshape((s_in.shape[0],)).astype(
                [('label', self.label_dtype),
                 ('id', self.id_dtype),
                 ('value', 'f8')]
                )

            self.lv.out_strength.sort()
            self.lv.in_strength.sort()
            self.lv.out_strength_dict = sout_dict
            self.lv.in_strength_dict = sin_dict

        if get:
            return self.lv.out_strength

    def in_strength_by_label(self, get=False):
        """ Compute the in strength sequence by label.

        If get is true it returns the array. Result is added to lv.
        """
        if not hasattr(self.lv, 'in_strength'):
            self.out_strength_by_label()

        if get:
            return self.lv.in_strength

    def adjacency_matrix(self, kind='csr', compressed=False):
        """ Return the adjacency matrices for each label as a list of scipy
        sparse matrices.
        """
        if compressed:
            adj = lib.to_sparse(
                    self.e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col='weight')
        else:
            adj = []
            for i in range(self.num_labels):
                e = self.e[self.e.label == i]
                adj.append(lib.to_sparse(
                    e, (self.num_vertices, self.num_vertices), kind=kind,
                    i_col='src', j_col='dst', data_col='weight'))
        return adj

    def to_networkx(self, original=False):
        G = nx.MultiDiGraph()
        if original:
            id_conv = list(self.id_dict.keys())
            label_conv = list(self.label_dict.keys())

            if hasattr(self, 'gv'):
                group_conv = list(self.group_dict.keys())
                v_num = mt.id_attr_dict(
                    self.v, id_col='id', attr_cols=['group'])
                v = []
                for row in v_num:
                    v.append((id_conv[row[0]],
                             {'group': group_conv[row[1]['group']]}))
            else:
                v_num = self.v.id
                v = []
                for row in v_num:
                    v.append(id_conv[row])

            e = []
            for row in self.e:
                e.append((id_conv[row.src], id_conv[row.dst],
                         {'weight': row.weight,
                          'label': label_conv[row.label]}))
        else:
            if hasattr(self, 'gv'):
                v = mt.id_attr_dict(self.v, id_col='id', attr_cols=['group'])
            else:
                v = self.v.id

            e = []
            for row in self.e:
                e.append((row.src, row.dst,
                         {'weight': row.weight,
                          'label': row.label}))

        G.add_nodes_from(v)
        G.add_edges_from(e)

        return G




class Graph():
    """ Generator function for graphs. Returns correct object depending on inputs
    passed.
    """
    def __new__(cls, v, e, **kwargs):
        # Ensure passed arguments are accepted
        allowed_arguments = ['v_id', 'src', 'dst', 'weight',
                             'edge_label', 'v_group']
        for name in kwargs:
            if name not in allowed_arguments:
                raise ValueError('Illegal argument passed: ' + name)

        if 'v_id' not in kwargs:
            id_col = 'id'
        else:
            id_col = kwargs['v_id']

        if 'src' not in kwargs:
            src_col = 'src'
        else:
            src_col = kwargs['src']

        if 'dst' not in kwargs:
            dst_col = 'dst'
        else:
            dst_col = kwargs['dst']

        if 'v_group' in kwargs:
            v_group = kwargs['v_group']
        else:
            v_group = None

        if 'weight' in kwargs:
            weight_col = kwargs['weight']
            if 'edge_label' in kwargs:
                label_col = kwargs['edge_label']
                return WeightedLabelGraph(v,
                                          e,
                                          v_id=id_col,
                                          src=src_col,
                                          dst=dst_col,
                                          weight=weight_col,
                                          edge_label=label_col,
                                          v_group=v_group)
            else:
                return WeightedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col,
                                     weight=weight_col,
                                     v_group=v_group)

        else:
            if 'edge_label' in kwargs:
                label_col = kwargs['edge_label']
                return LabelGraph(v,
                                  e,
                                  v_id=id_col,
                                  src=src_col,
                                  dst=dst_col,
                                  edge_label=label_col,
                                  v_group=v_group)
            else:
                return DirectedGraph(v,
                                     e,
                                     v_id=id_col,
                                     src=src_col,
                                     dst=dst_col,
                                     v_group=v_group)




