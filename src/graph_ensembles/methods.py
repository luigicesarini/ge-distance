import numpy as np
from random import random
from numba import jit
from numba import prange
from math import ceil
from math import sqrt


# --------------- GRAPH METHODS ---------------
def _get_num_bytes(num_items):
    """ Determine the number of bytes needed for storing ids for num_items."""
    return int(max(2**np.ceil(np.log2(np.log2(num_items + 1)/8)), 1))


@jit(nopython=True)
def _check_unique_edges(e):
    """ Check that the edges are not repeated in the sorted edge list."""
    for i in np.arange(len(e)-1):
        if (e[i].src == e[i+1].src) and (e[i].dst == e[i+1].dst):
            assert False, 'There are repeated edges.'


@jit(nopython=True)
def _check_unique_labelled_edges(e):
    """ Check that the edges are not repeated in the sorted edge list."""
    for i in np.arange(len(e)-1):
        if ((e[i].label == e[i+1].label) and
           (e[i].src == e[i+1].src) and
           (e[i].dst == e[i+1].dst)):
            assert False, 'There are repeated edges.'


def _generate_id_dict(v, id_col):
    """ Return id dictionary. """
    id_dict = {}
    rep_msg = 'There is at least one repeated id in the vertex dataframe.'

    if isinstance(id_col, list):
        if len(id_col) > 1:
            # Id is a tuple
            i = 0
            for x in v[id_col].itertuples(index=False):
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        elif len(id_col) == 1:
            # Extract series
            i = 0
            for x in v[id_col[0]]:
                if x in id_dict:
                    raise Exception(rep_msg)
                else:
                    id_dict[x] = i
                    i += 1

        else:
            # No column passed
            raise ValueError('At least one id column must be given.')

    elif isinstance(id_col, str):
        # Extract series
        i = 0
        for x in v[id_col]:
            if x in id_dict:
                raise Exception(rep_msg)
            else:
                id_dict[x] = i
                i += 1

    else:
        raise ValueError('id_col must be string or list of strings.')

    return id_dict


def _generate_label_dict(e, label):
    """ Return id dictionary. """
    label_dict = {}

    if isinstance(label, list):
        if len(label) > 1:
            # Id is a tuple
            i = 0
            for x in e[label].itertuples(index=False):
                if x in label_dict:
                    pass
                else:
                    label_dict[x] = i
                    i += 1

        elif len(label) == 1:
            # Extract series
            i = 0
            for x in e[label[0]]:
                if x in label_dict:
                    pass
                else:
                    label_dict[x] = i
                    i += 1

        else:
            # No column passed
            raise ValueError('At least one label column must be given.')

    elif isinstance(label, str):
        # Extract series
        i = 0
        for x in e[label]:
            if x in label_dict:
                pass
            else:
                label_dict[x] = i
                i += 1

    else:
        raise ValueError('edge_label must be string or list of strings.')

    return label_dict


@jit(nopython=True)
def _compute_num_edges_by_label(e, num_lbl):
    edges = np.zeros(num_lbl, dtype=np.int64)
    for n in range(len(e)):
        edges[e[n].label] += 1

    return edges


@jit(nopython=True)
def _compute_tot_weight_by_label(e, num_lbl):
    weight = np.zeros(num_lbl, dtype=np.float64)
    for n in range(len(e)):
        weight[e[n].label] += e[n].weight

    return weight


@jit(nopython=True)
def _compute_degree(e, num_v):
    d = np.zeros(num_v, dtype=np.int64)
    s = set()
    for n in range(len(e)):
        i = e[n].src
        j = e[n].dst
        if i <= j:
            pair = (i, j)
        else:
            pair = (j, i)

        if pair not in s:
            s.add(pair)
            d[i] += 1
            d[j] += 1

    return d


@jit(nopython=True)
def _compute_degree_by_label(e):
    lv_dict = {}
    i = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in lv_dict:
            lv_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in lv_dict:
            lv_dict[pair_d] = i
            i += 1

    d = np.zeros((len(lv_dict), 3), dtype=np.uint64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = lv_dict[tmp]
        d[ind, 0:2] = tmp
        d[ind, 2] += 1

        tmp = (e[n].label, e[n].dst)
        ind = lv_dict[tmp]
        d[ind, 0:2] = tmp
        d[ind, 2] += 1

    return d, lv_dict


@jit(nopython=True)
def _compute_in_out_degree(e, num_v):
    d_out = np.zeros(num_v, dtype=np.int64)
    d_in = np.zeros(num_v, dtype=np.int64)
    for n in range(len(e)):
        d_out[e[n].src] += 1
        d_in[e[n].dst] += 1

    return d_out, d_in


@jit(nopython=True)
def _compute_in_out_degree_labelled(e, num_v):
    d_out = np.zeros(num_v, dtype=np.int64)
    d_in = np.zeros(num_v, dtype=np.int64)
    s = set()
    for n in range(len(e)):
        i = e[n].src
        j = e[n].dst
        pair = (i, j)
        if pair not in s:
            s.add(pair)
            d_out[i] += 1
            d_in[j] += 1

    return d_out, d_in


@jit(nopython=True)
def _compute_in_out_degree_by_label(e):
    out_dict = {}
    in_dict = {}
    i = 0
    j = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in out_dict:
            out_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in in_dict:
            in_dict[pair_d] = j
            j += 1

    d_out = np.zeros((len(out_dict), 3), dtype=np.uint64)
    d_in = np.zeros((len(in_dict), 3), dtype=np.uint64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = out_dict[tmp]
        d_out[ind, 0:2] = tmp
        d_out[ind, 2] += 1

        tmp = (e[n].label, e[n].dst)
        ind = in_dict[tmp]
        d_in[ind, 0:2] = tmp
        d_in[ind, 2] += 1

    return d_out, d_in, out_dict, in_dict


@jit(nopython=True)
def _compute_strength(e, num_v):
    s = np.zeros(num_v, dtype=np.float64)

    for n in range(len(e)):
        s[e[n].src] += e[n].weight
        s[e[n].dst] += e[n].weight

    return s


@jit(nopython=True)
def _compute_strength_by_label(e):
    lv_dict = {}
    i = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in lv_dict:
            lv_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in lv_dict:
            lv_dict[pair_d] = i
            i += 1

    s = np.zeros((len(lv_dict), 3), dtype=np.float64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = lv_dict[tmp]
        s[ind, 0:2] = tmp
        s[ind, 2] += e[n].weight

        tmp = (e[n].label, e[n].dst)
        ind = lv_dict[tmp]
        s[ind, 0:2] = tmp
        s[ind, 2] += e[n].weight

    return s, lv_dict


@jit(nopython=True)
def _compute_in_out_strength(e, num_v):
    s_out = np.zeros(num_v, dtype=np.float64)
    s_in = np.zeros(num_v, dtype=np.float64)
    for n in range(len(e)):
        s_out[e[n].src] += e[n].weight
        s_in[e[n].dst] += e[n].weight

    return s_out, s_in


@jit(nopython=True)
def _compute_in_out_strength_by_label(e):
    out_dict = {}
    in_dict = {}
    i = 0
    j = 0
    for n in range(len(e)):
        pair_s = (e[n].label, e[n].src)
        if pair_s not in out_dict:
            out_dict[pair_s] = i
            i += 1

        pair_d = (e[n].label, e[n].dst)
        if pair_d not in in_dict:
            in_dict[pair_d] = j
            j += 1

    s_out = np.zeros((len(out_dict), 3), dtype=np.float64)
    s_in = np.zeros((len(in_dict), 3), dtype=np.float64)
    for n in range(len(e)):
        tmp = (e[n].label, e[n].src)
        ind = out_dict[tmp]
        s_out[ind, 0:2] = tmp
        s_out[ind, 2] += e[n].weight

        tmp = (e[n].label, e[n].dst)
        ind = in_dict[tmp]
        s_in[ind, 0:2] = tmp
        s_in[ind, 2] += e[n].weight

    return s_out, s_in, out_dict, in_dict


# --------------- RANDOM GRAPH METHODS ---------------
@jit(nopython=True)
def _random_graph(n, p):
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """

    if n > 10:
        max_len = int(ceil(n*(n-1)*p + 3*sqrt(n*(n-1)*p*(1-p))))
        a = np.empty((max_len, 2), dtype=np.uint64)
    else:
        a = np.empty((n*(n-1), 2), dtype=np.uint64)
    count = 0
    for i in range(n):
        for j in range(n):
            if random() < p:
                a[count, 0] = i
                a[count, 1] = j
                count += 1
    assert a.shape[0] > count, 'Miscalculated bounds of max successful draws.'
    return a[0:count, :].copy()


@jit(nopython=True)
def _random_labelgraph(n, l, p):  # noqa: E741
    """ Generates a edge list given the number of vertices and the probability
    p of observing a link.
    """

    if n > 10:
        max_len = int(np.ceil(np.sum(n*(n-1)*p + 3*np.sqrt(n*(n-1)*p*(1-p)))))
        a = np.empty((max_len, 3), dtype=np.uint64)
    else:
        a = np.empty((n*(n-1)*l, 3), dtype=np.uint64)
    count = 0
    for i in range(l):
        for j in range(n):
            for k in range(n):
                if random() < p[i]:
                    a[count, 0] = i
                    a[count, 1] = j
                    a[count, 2] = k
                    count += 1
    assert a.shape[0] > count, 'Miscalculated bounds of max successful draws.'
    return a[0:count, :].copy()


# --------------- STRIPE METHODS ---------------
# @jit(nopython=True, parallel=True)
def prob_matrix_stripe_one_z(out_strength, in_strength, z, N):
    """ Compute the probability matrix of the stripe fitness model given the
    single parameter z controlling for the density.
    """
    p = np.ones((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] *= 1 - tmp2 / (1 + tmp2)

    return 1 - p


# @jit(nopython=True, parallel=True)
def prob_array_stripe_one_z(out_strength, in_strength, z, N, G):
    """ Compute the probability array of the stripe fitness model given the
    single parameter z controlling for the density.
    """
    p = np.zeros((N, N, G), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in, sect_out] = tmp2 / (1 + tmp2)

    return p


# @jit(forceobj=True, parallel=False)
def expected_links_stripe_one_z(out_strength, in_strength, z):
    """ Compute the expected number of links of the stripe fitness model
    given the single parameter z controlling for the density.
    """
    N = max(np.max(out_strength[:, 0]), np.max(in_strength[:, 0])) + 1
    return prob_matrix_stripe_one_z(out_strength, in_strength, z, int(N)).sum()


@jit(nopython=True)
def assign_weights_cimi_stripe_one_z(p, out_strength, in_strength,
                                     N, strengths_stripe, expected=True):
    """ Return the weighted adjacency matrix of the stripe fitness model
    with just one global parameter z controlling for the density.
    Depending on the value of "expected" the weighted adjacency matrix is the
    expceted one or an ensemble realisation.

    Parameters
    ----------
    p: scipy.sparse.matrix or np.ndarray or list of lists
        the binary probability matrix
    out_strength: np.ndarray
        the out strength sequence of graph
    in_strength: np.ndarray
        the in strength sequence of graph
    strengths_stripe: np.ndarray
        strengths for stripe
    expected: bool
        If True the strength of each link is the expected one

    Returns
    -------
    np.ndarray
        Depending on the value of expected, returns the expected weighted
        matrix or an ensemble realisation

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider allowing for sparse matrices in case of groups and
    to avoid iteration over all indices.
    """

    W = np.zeros((N, N), dtype=np.float64)
    if expected:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_out = int(out_strength[i, 1])
            s_out = out_strength[i, 2]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_in = int(in_strength[j, 1])
                s_in = in_strength[j, 2]
                if (ind_out != ind_in) & (sect_out == sect_in):
                    tot_w = strengths_stripe[sect_out]
                    tmp = s_out*s_in
                    W[ind_out, ind_in] = (tmp)/(tot_w)
    else:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_out = int(out_strength[i, 1])
            s_out = out_strength[i, 2]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_in = int(in_strength[j, 1])
                s_in = in_strength[j, 2]
                if (ind_out != ind_in) & (sect_out == sect_in):
                    if p[ind_out, ind_in] > np.random.random():
                        tot_w = strengths_stripe[sect_out]
                        tmp = s_out*s_in
                        W[ind_out, ind_in] = (tmp)/(tot_w* p[ind_out, ind_in])
    return W


@jit(nopython=True, parallel=True)
def prob_matrix_stripe_mult_z(out_strength, in_strength, z, N):
    """ Compute the probability matrix of the stripe fitness model with one
    parameter controlling for the density for each label.
    """
    p = np.ones((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = z[sect_out]*s_out*s_in
                p[ind_out, ind_in] *= 1 - tmp / (1 + tmp)
    return 1 - p


@jit(nopython=True, parallel=True)
def prob_array_stripe_mult_z(out_strength, in_strength, z, N, G):
    """ Compute the probability array of the stripe fitness model with one
    parameter controlling for the density for each label.
    """
    p = np.zeros((N, N, G), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out]*tmp
                p[ind_out, ind_in, sect_out] = tmp2 / (1 + tmp2)

    return p


# @jit(forceobj=True, parallel=True)
def expected_links_stripe_mult_z(out_strength, in_strength, z):
    """ Compute the expected number of links for the stripe fitness model
    with one parameter controlling for the density for each label.
    """
    exp_links = np.zeros(len(z))

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in) & (sect_out == sect_in):
                tmp = z[sect_out]*s_out*s_in
                exp_links[sect_out] += tmp / (1 + tmp)

    return exp_links


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_block_one_z(out_strength, in_strength,
                                          z, N):
    """
    Function computing the Probability Matrix of the Cimi block model
    with just one parameter controlling for the density.
    """

    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_node_i = int(out_strength[i, 1])
        sect_out = int(out_strength[i, 2])
        s_out = out_strength[i, 3]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_node_j = int(in_strength[j, 1])
            sect_in = int(in_strength[j, 2])
            s_in = in_strength[j, 3]
            if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


# @jit(forceobj=True, parallel=True)
def expected_links_block_one_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    p = 0.0

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_node_i = int(out_strength[i, 1])
        sect_out = int(out_strength[i, 2])
        s_out = out_strength[i, 3]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_node_j = int(in_strength[j, 1])
            sect_in = int(in_strength[j, 2])
            s_in = in_strength[j, 3]
            if ((ind_out != ind_in) & (sect_out == sect_node_j) &
               (sect_in == sect_node_i)):
                tmp = s_out*s_in
                tmp2 = z*tmp
                p += tmp2 / (1 + tmp2)
    return p


@jit(nopython=True)
def assign_weights_cimi_block_one_z(p, out_strength, in_strength,
                                    N, strengths_block, expected=True):
    """Function returning the weighted adjacency matrix of the Cimi block
    model with just one global parameter z controlling for the density.
    Depending on the value of "expected" the weighted adjacency matrix can be
    the expceted one or just an ensemble realisation.

    Parameters
    ----------
    p: scipy.sparse.matrix or np.ndarray or list of lists
        the binary probability matrix
    out_strength: np.ndarray
        the out strength sequence of graph organised by sector
    in_strength: np.ndarray
        the in strength sequence of graph organised by sector
    strengths_block: np.ndarray
        total strengths between every couple of groups
    expected: bool
        If True the strength of each link is the expected one otherwise
        it is just a single realisation
    Returns
    -------
    np.ndarray
        Depending on the value of expected, returns the expected weighted
        matrix or an ensemble realisation

    TODO: Currently implemented with numpy arrays and standard iteration over
    all indices. Consider allowing for sparse matrices in case of groups and
    to avoid iteration over all indices.
    """

    W = np.zeros((N, N), dtype=np.float64)
    if expected:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_node_i = int(out_strength[i, 1])
            sect_out = int(out_strength[i, 2])
            s_out = out_strength[i, 3]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_node_j = int(in_strength[j, 1])
                sect_in = int(in_strength[j, 2])
                s_in = in_strength[j, 3]
                if ((ind_out != ind_in) & (sect_out == sect_node_j) &
                    (sect_in == sect_node_i)):
                    tot_w = strengths_block[sect_node_i, sect_node_j]
                    tmp = s_out*s_in
                    W[ind_out, ind_in] = (tmp)/(tot_w)
    else:
        for i in np.arange(out_strength.shape[0]):
            ind_out = int(out_strength[i, 0])
            sect_node_i = int(out_strength[i, 1])
            sect_out = int(out_strength[i, 2])
            s_out = out_strength[i, 3]
            for j in np.arange(in_strength.shape[0]):
                ind_in = int(in_strength[j, 0])
                sect_node_j = int(in_strength[j, 1])
                sect_in = int(in_strength[j, 2])
                s_in = in_strength[j, 3]
                if ((ind_out != ind_in) & (sect_out == sect_node_j) &
                    (sect_in == sect_node_i)):
                    if p[ind_out, ind_in] > np.random.random():
                        tot_w = strengths_block[sect_out, sect_in]
                        tmp = s_out*s_in
                        W[ind_out, ind_in] = (tmp)/(tot_w * p[ind_out, ind_in])

    return W


@jit(nopython=True, parallel=True)
def vector_fitness_prob_array_block_mult_z(out_strength, in_strength, z, N):
    """
    Function computing the Probability Matrix of the Cimi block model
    with just one parameter controlling for the density.
    """
    p = np.zeros((N, N), dtype=np.float64)

    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[ind_out, ind_in] = tmp2 / (1 + tmp2)
    return p


# @jit(forceobj=True, parallel=True)
def expected_links_block_mult_z(out_strength, in_strength, z):
    """
    Function computing the expeceted number of links, under the Cimi
    stripe model, given the parameter z controlling for the density.
    """
    n_sector = int(np.sqrt(max(z.shape)))
    p = np.zeros(shape=(n_sector, n_sector), dtype=np.float)
    z = np.reshape(z, newshape=(n_sector, n_sector))
    for i in np.arange(out_strength.shape[0]):
        ind_out = int(out_strength[i, 0])
        sect_out = int(out_strength[i, 1])
        s_out = out_strength[i, 2]
        for j in prange(in_strength.shape[0]):
            ind_in = int(in_strength[j, 0])
            sect_in = int(in_strength[j, 1])
            s_in = in_strength[j, 2]
            if (ind_out != ind_in):
                tmp = s_out*s_in
                tmp2 = z[sect_out, sect_in]*tmp
                p[sect_out, sect_in] += tmp2 / (1 + tmp2)
    return np.reshape(p, -1)
