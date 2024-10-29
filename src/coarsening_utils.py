import numpy as np
import pygsp as gsp
from pygsp import graphs, filters, reduction
import scipy as sp
from scipy import sparse
import torch

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from sortedcontainers import SortedList
import graph_utils

import time


def coarsen(
    G_orig,
    G,
    replay_nodes,
    feature,
    K=10,
    r=0.5,
    max_levels=1,
    method="repro",
    Uk=None,
    lk=None,
    max_level_r=0.99,
):
    """
    This function provides a common interface for coarsening algorithms that contract subgraphs

    Parameters
    ----------
    G : pygsp Graph
    K : int
        The size of the subspace we are interested in preserving.
    r : float between (0,1)
        The desired reduction defined as 1 - n/N.
    method : String
        ['variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge', 'algebraic_JC', 'affinity_GS', 'kron']

    Returns
    -------
    C : np.array of size n x N
        The coarsening matrix.
    Gc : pygsp Graph
        The smaller graph.
    Call : list of np.arrays
        Coarsening matrices for each level
    Gall : list of (n_levels+1) pygsp Graphs
        All graphs involved in the multilevel coarsening

    Example
    -------
    C, Gc, Call, Gall = coarsen(G, K=10, r=0.8)
    """
    r = np.clip(r, 0, 0.999)
    G0 = G
    N = G.N

    # current and target graph sizes
    n, n_target = N, np.ceil((1 - r) * N)

    C = sp.sparse.eye(N, format="csc")
    Gc = G

    Call, Gall = [], []
    Gall.append(G_orig)

    for level in range(1, max_levels + 1):
        # print ('level:', level)

        G = Gc
        # G_orig = Gc

        # how much more we need to reduce the current graph
        r_cur = np.clip(1 - n_target / n, 0.0, max_level_r)

        if level == 1:
            if (Uk is not None) and (lk is not None) and (len(lk) >= K):
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk[:, :K] @ np.diag(lsinv[:K])
            else:
                offset = 2 * max(G.dw)
                T = offset * sp.sparse.eye(G.N, format="csc") - G.L
                lk, Uk = sp.sparse.linalg.eigsh(T, k=K, which="LM", tol=1e-5)
                lk = (offset - lk)[::-1]
                Uk = Uk[:, ::-1]
                mask = lk < 1e-10
                lk[mask] = 1
                lsinv = lk ** (-0.5)
                lsinv[mask] = 0
                B = Uk @ np.diag(lsinv)
            A = B
        else:
            B = iC.dot(B)
            d, V = np.linalg.eig(B.T @ (G.L).dot(B))
            mask = d == 0
            d[mask] = 1
            dinvsqrt = d ** (-1 / 2)
            dinvsqrt[mask] = 0
            A = B @ np.diag(dinvsqrt) @ V

        coarsening_list = repro(
            G, replay_nodes, feature, K=K, A=A, r=r_cur)

        iC = get_coarsening_matrix(G, coarsening_list)

        C = iC.dot(C)
        Call.append(iC)

        Wc = graph_utils.zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))

    Gc_ = G_orig
    for i in range(len(Call)):
        iC = Call[i]
        Wc = graph_utils.zero_diag(coarsen_matrix(Gc_.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc_ = gsp.graphs.Graph(Wc)
        else:
            Gc = gsp.graphs.Graph(Wc, coords=coarsen_vector(G_orig.coords, iC))
        Gall.append(Gc_)

    return C, Gc_, Call, Gall


################################################################################
# General coarsening utility functions
################################################################################


def coarsen_vector(x, C):
    return (C.power(2)).dot(x)


def lift_vector(x, C):
    # Pinv = C.T; Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return Pinv.dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))


def lift_matrix(W, C):
    P = C.power(2)
    return (P.T).dot(W.dot(P))


def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    degree = G.d
    C = sp.sparse.eye(G.N, format="lil")
    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)
        # print (subgraph)

        # add v_j's to v_i's row
        # print (degree[subgraph])
        total_degee = sum(degree[subgraph])
        # print (total_degee)
        # print (degree[subgraph]/total_degee)
        # quit()
        C[subgraph[0], subgraph] = np.sqrt(degree[subgraph]/total_degee)

        # C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)
        # C[subgraph[0], subgraph[0]] = 1 / np.sqrt(nc)
        # C[subgraph[0], subgraph[1]] = 1 / np.sqrt(nc)
        # print (C[subgraph[0], subgraph])
        # quit()
        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csc_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def coarsening_quality(G, C, kmax=30, Uk=None, lk=None):
    """
    Measures how good is a coarsening.

    Parameters
    ----------
    G : pygsp Graph
    C : np.array(n,N)
        The coarsening matrix
    kmax : int
        Until which eigenvalue we are interested in.

    Returns
    -------
    metric : dictionary
        Contains all relevant metrics for coarsening quality:
        * error_eigenvalue : np.array(kmax)
        * error_subspace : np.array(kmax)
        * error_sintheta : np.array(kmax)
        * angle_matrix : np.array(kmax)
        * rss constants : np.array(kmax)
        as well as some general properties of Gc:
        * r : int
            reduction ratio
        * m : int
            number of edges
    """
    N = G.N
    I = np.eye(N)

    if (Uk is not None) and (lk is not None) and (len(lk) >= kmax):
        U, l = Uk, lk
    elif hasattr(G, "U"):
        U, l = G.U, G.e
    else:
        l, U = sp.sparse.linalg.eigsh(G.L, k=kmax, which="SM", tol=1e-3)

    l[0] = 1
    linv = l ** (-0.5)
    linv[0] = 0
    # l[0] = 0 # avoids divide by 0

    # below here, everything is C specific
    n = C.shape[0]
    Pi = C.T @ C
    S = graph_utils.get_S(G).T
    Lc = C.dot((G.L).dot(C.T))
    Lp = Pi @ G.L @ Pi

    if kmax > n / 2:
        [Uc, lc] = graph_utils.eig(Lc.toarray())
    else:
        lc, Uc = sp.sparse.linalg.eigsh(Lc, k=kmax, which="SM", tol=1e-3)

    if not sp.sparse.issparse(Lc):
        print("warning: Lc should be sparse.")

    metrics = {"r": 1 - n / N, "m": int((Lc.nnz - n) / 2)}

    kmax = np.clip(kmax, 1, n)

    # eigenvalue relative error
    metrics["error_eigenvalue"] = np.abs(l[:kmax] - lc[:kmax]) / l[:kmax]
    metrics["error_eigenvalue"][0] = 0

    # angles between eigenspaces
    metrics["angle_matrix"] = U.T @ C.T @ Uc

    # rss constants
    #    metrics['rss'] = np.diag(U.T @ Lp @ U)/l - 1
    #    metrics['rss'][0] = 0

    # other errors
    kmax = np.clip(kmax, 2, n)

    error_subspace = np.zeros(kmax)
    error_subspace_bound = np.zeros(kmax)
    error_sintheta = np.zeros(kmax)

    M = S @ Pi @ U @ np.diag(linv)
    #    M_bound = S @ (I - Pi) @ U @ np.diag(linv)

    for kIdx in range(1, kmax):
        error_subspace[kIdx] = np.abs(np.linalg.norm(M[:, : kIdx + 1], ord=2) - 1)
        #        error_subspace_bound[kIdx] = np.linalg.norm( M_bound[:,:kIdx + 1], ord=2)
        error_sintheta[kIdx] = (
            np.linalg.norm(metrics["angle_matrix"][0 : kIdx + 1, kIdx + 1 :], ord="fro")
            ** 2
        )

    metrics["error_subspace"] = error_subspace
    # metrics['error_subspace_bound'] = error_subspace_bound
    metrics["error_sintheta"] = error_sintheta

    return metrics


def plot_coarsening(
    Gall, Call, size=3, edge_width=0.8, node_size=20, alpha=0.55, title=""
):
    """
    Plot a (hierarchical) coarsening

    Parameters
    ----------
    G_all : list of pygsp Graphs
    Call  : list of np.arrays

    Returns
    -------
    fig : matplotlib figure
    """
    # colors signify the size of a coarsened subgraph ('k' is 1, 'g' is 2, 'b' is 3, and so on)
    colors = ["k", "g", "b", "r", "y"]

    n_levels = len(Gall) - 1
    if n_levels == 0:
        return None
    fig = plt.figure(figsize=(n_levels * size * 3, size * 2))

    for level in range(n_levels):

        G = Gall[level]
        edges = np.array(G.get_edge_list()[0:2])

        Gc = Gall[level + 1]
        #         Lc = C.dot(G.L.dot(C.T))
        #         Wc = sp.sparse.diags(Lc.diagonal(), 0) - Lc;
        #         Wc = (Wc + Wc.T) / 2
        #         Gc = gsp.graphs.Graph(Wc, coords=(C.power(2)).dot(G.coords))
        edges_c = np.array(Gc.get_edge_list()[0:2])
        C = Call[level]
        C = C.toarray()

        if G.coords.shape[1] == 2:
            ax = fig.add_subplot(1, n_levels + 1, level + 1)
            ax.axis("off")
            ax.set_title(f"{title} | level = {level}, N = {G.N}")

            [x, y] = G.coords.T
            for eIdx in range(0, edges.shape[1]):
                ax.plot(
                    x[edges[:, eIdx]],
                    y[edges[:, eIdx]],
                    color="k",
                    alpha=alpha,
                    lineWidth=edge_width,
                )
            for i in range(Gc.N):
                subgraph = np.arange(G.N)[C[i, :] > 0]
                ax.scatter(
                    x[subgraph],
                    y[subgraph],
                    c=colors[np.clip(len(subgraph) - 1, 0, 4)],
                    s=node_size * len(subgraph),
                    alpha=alpha,
                )

        elif G.coords.shape[1] == 3:
            ax = fig.add_subplot(1, n_levels + 1, level + 1, projection="3d")
            ax.axis("off")

            [x, y, z] = G.coords.T
            for eIdx in range(0, edges.shape[1]):
                ax.plot(
                    x[edges[:, eIdx]],
                    y[edges[:, eIdx]],
                    zs=z[edges[:, eIdx]],
                    color="k",
                    alpha=alpha,
                    lineWidth=edge_width,
                )
            for i in range(Gc.N):
                subgraph = np.arange(G.N)[C[i, :] > 0]
                ax.scatter(
                    x[subgraph],
                    y[subgraph],
                    z[subgraph],
                    c=colors[np.clip(len(subgraph) - 1, 0, 4)],
                    s=node_size * len(subgraph),
                    alpha=alpha,
                )

    # the final graph
    Gc = Gall[-1]
    edges_c = np.array(Gc.get_edge_list()[0:2])

    if G.coords.shape[1] == 2:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1)
        ax.axis("off")
        [x, y] = Gc.coords.T
        ax.scatter(x, y, c="k", s=node_size, alpha=alpha)
        for eIdx in range(0, edges_c.shape[1]):
            ax.plot(
                x[edges_c[:, eIdx]],
                y[edges_c[:, eIdx]],
                color="k",
                alpha=alpha,
                lineWidth=edge_width,
            )

    elif G.coords.shape[1] == 3:
        ax = fig.add_subplot(1, n_levels + 1, n_levels + 1, projection="3d")
        ax.axis("off")
        [x, y, z] = Gc.coords.T
        ax.scatter(x, y, z, c="k", s=node_size, alpha=alpha)
        for eIdx in range(0, edges_c.shape[1]):
            ax.plot(
                x[edges_c[:, eIdx]],
                y[edges_c[:, eIdx]],
                z[edges_c[:, eIdx]],
                color="k",
                alpha=alpha,
                lineWidth=edge_width,
            )

    ax.set_title(f"{title} | level = {n_levels}, n = {Gc.N}")
    fig.tight_layout()
    return fig


################################################################################
# Variation-based contraction algorithms
################################################################################
def cosine_sim(x, y, eps=1e-08):
    div = np.linalg.norm(x)*np.linalg.norm(y)
    if abs(div) < eps:
        return np.dot(x, y)/eps
    return np.dot(x, y)/div

def repro(G, replay_nodes, feature, A=None, K=10, r=0.5):
    # Proposed graph coarsening method: Node Representation Proximity
    N, deg, M = G.N, G.dw, G.Ne
    ones = np.ones(2)
    Pibot = np.eye(2) - np.outer(ones, ones) / 2

    # cost function for the edge
    def subgraph_cost(G, A, edge):
        edge, w = edge[:2].astype(np.int), edge[2]
        source, target = edge[0], edge[1]
        if source == target: return 0
        d = cosine_sim(feature[source].cpu().numpy(), feature[target].cpu().numpy())
        if edge[0] in replay_nodes or edge[1] in replay_nodes:
            return 100-d
        return -d

    edges = np.array(G.get_edge_list())
    weights = np.array([subgraph_cost(G, A, edges[:, e]) for e in range(M)])
    coarsening_list = matching(G, weights=-weights, r=r)

    return coarsening_list


def find(data, i):
    # print (data[i], i)
    if i != data[i]:
        data[i] = find(data, data[i])
    return data[i]

def union(data, i, j):
    # print (i,j)
    pi, pj = find(data, i), find(data, j)
    if pi != pj:
        data[pi] = pj
        return False
    return True

# def connected(data, i, j):
#     return find(data, i) == find(data, j)

def matching(G, weights, r):
    """
    Generates a matching greedily by selecting at each iteration the edge
    with the largest weight and then removing all adjacent edges from the
    candidate set.

    Parameters
    ----------
    G : pygsp graph
    weights : np.array(M)
        a weight for each edge
    r : float
        The desired dimensionality reduction (r = 1 - n/N)

    Notes:
    * The complexity of this is O(M)
    * Depending on G, the algorithm might fail to return ratios>0.3
    """

    N = G.N

    # the edge set
    edges = np.array(G.get_edge_list()[0:2])
    M = edges.shape[1]

    idx = np.argsort(-weights)
    # idx = np.argsort(weights)[::-1]
    edges = edges[:, idx]

    # the candidate edge set
    candidate_edges = edges.T.tolist()

    # the matching edge set (this is a list of arrays)
    matching = []

    # which vertices have been selected
    # marked = np.zeros(N, dtype=np.int32)
    marked = np.array([i for i in range(N)], dtype=np.int32)

    n, n_target = N, (1 - r) * N
    while len(candidate_edges) > 0:
        # pop a candidate edge
        [i, j] = candidate_edges.pop(0)
        # check if marked
        if i == j:
            continue
        if not union(marked, i, j):
            n -= 1
        # termination condition
        if n <= n_target:
            break

    matching = [[] for i in range(N)]
    for i in range(N):
        matching[find (marked, i)].append(i)
    matching = list(filter(lambda x: len(x) >= 2, matching))
    matching = [np.array(matching[i]) for i in range(len(matching))]
    return matching

def coarsen_vector(x, C):
    return (C.power(2)).dot(x)

def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return (Pinv.T).dot(W.dot(Pinv))
