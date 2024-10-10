import functools
import math
from copy import deepcopy

import networkx as nx
from networkx.utils.decorators import not_implemented_for


def degree_centrality(G):
    """
    Copied from networkx without default normalization
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    centrality = {n: d for n, d in G.degree()}
    return centrality


@not_implemented_for("undirected")
def in_degree_centrality(G):
    """
    Copied from networkx without default normalization
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    centrality = {n: d for n, d in G.in_degree()}
    return centrality


@not_implemented_for("undirected")
def out_degree_centrality(G):
    """
    Copied from networkx without default normalization
    """
    if len(G) <= 1:
        return {n: 1 for n in G}

    return dict(G.out_degree())


def closeness_centrality(G, u=None, distance=None):
    """
    Copied from networkx without default normalization
    """
    if G.is_directed():
        G = G.reverse()  # create a reversed graph view

    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(
            nx.single_source_dijkstra_path_length, weight=distance
        )
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes
    else:
        nodes = [u]
    closeness_centrality = {}
    for n in nodes:
        sp = path_length(G, n)
        totsp = sum(sp.values())
        len_G = len(G)
        _closeness_centrality = 0.0
        if totsp > 0.0 and len_G > 1:
            _closeness_centrality = (len(sp) - 1.0) / totsp
            # normalize to number of nodes-1 in connected part
        closeness_centrality[n] = _closeness_centrality
    if u is not None:
        return closeness_centrality[u]
    else:
        return closeness_centrality


def voterank(G, number_of_nodes=None):
    """
    Copied from networkx without default normalization
    """
    influential_nodes = []
    voterank = {}
    if len(G) == 0:
        return influential_nodes
    if number_of_nodes is None or number_of_nodes > len(G):
        number_of_nodes = len(G)
    if G.is_directed():
        # For directed graphs compute average out-degree
        avgDegree = sum(deg for _, deg in G.out_degree()) / len(G)
    else:
        # For undirected graphs compute average degree
        avgDegree = sum(deg for _, deg in G.degree()) / len(G)
    # step 1 - initiate all nodes to (0,1) (score, voting ability)
    for n in G.nodes():
        voterank[n] = [0, 1]
    # Repeat steps 1b to 4 until num_seeds are elected.
    for _ in range(number_of_nodes):
        # step 1b - reset rank
        for n in G.nodes():
            voterank[n][0] = 0
        # step 2 - vote
        for n, nbr in G.edges():
            # In directed graphs nodes only vote for their in-neighbors
            voterank[n][0] += voterank[nbr][1]
            if not G.is_directed():
                voterank[nbr][0] += voterank[n][1]
        for n, _ in influential_nodes:
            voterank[n][0] = 0
        # step 3 - select top node
        n = max(G.nodes, key=lambda x: voterank[x][0])
        # if voterank[n][0] == 0:
        #    return influential_nodes
        influential_nodes.append((n, voterank[n][0]))
        # weaken the selected node
        voterank[n] = [0, 0]
        # step 4 - update voterank properties
        for _, nbr in G.edges(n):
            voterank[nbr][1] -= 1 / avgDegree
            voterank[nbr][1] = max(voterank[nbr][1], 0)

    centrality = {n: d for n, d in influential_nodes}
    none_vote_nodes = list(set(G.nodes) - set(centrality.keys()))
    for node in none_vote_nodes:
        centrality[node] = 0

    return dict(sorted(centrality.items()))


def avarage_neighbors_degree(G) -> dict:
    out_degrees_centrality = out_degree_centrality(G)
    avarage_degrees = []

    for node in G.nodes:
        node_neighbors = nx.neighbors(G, node)
        neighbors_copy = deepcopy(node_neighbors)
        number_of_neighbors = len(list(neighbors_copy))

        neighbors_degree = 0
        for neighbor in node_neighbors:
            neighbors_degree += out_degrees_centrality[neighbor]

        if number_of_neighbors == 0:
            neighbors_avarage_degree = 0.0
        else:
            neighbors_avarage_degree = round(neighbors_degree / number_of_neighbors, 3)
        avarage_degrees.append((node, neighbors_avarage_degree))

    centrality = {n: d for n, d in avarage_degrees}
    return centrality


def avarage_undirected_neighbors_degree(G) -> dict:
    out_degrees_centrality = degree_centrality(G)
    avarage_degrees = []

    for node in G.nodes:
        node_neighbors = nx.neighbors(G, node)
        neighbors_copy = deepcopy(node_neighbors)
        number_of_neighbors = len(list(neighbors_copy))

        neighbors_degree = 0
        for neighbor in node_neighbors:
            neighbors_degree += out_degrees_centrality[neighbor]

        if number_of_neighbors == 0:
            neighbors_avarage_degree = 0.0
        else:
            neighbors_avarage_degree = round(neighbors_degree / number_of_neighbors, 3)
        avarage_degrees.append((node, neighbors_avarage_degree))

    centrality = {n: d for n, d in avarage_degrees}
    return centrality


def local_reaching_centrality(G) -> dict:
    """
    Copied from networkx without default normalization
    """
    local_reaching = []

    for node in G.nodes:
        centrality = nx.local_reaching_centrality(G, node, normalized=False)
        local_reaching.append((node, centrality))

    centrality = {n: d for n, d in local_reaching}
    return centrality


@not_implemented_for("multigraph")
def eigenvector_centrality(G, max_iter=10000, tol=1.0e-6, nstart=None, weight=None):
    r"""
    Copied from networkx without default normalization
    Compute the eigenvector centrality for the graph `G`.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node $i$ is
    the $i$-th element of the vector $x$ defined by the equation

    .. math::

        Ax = \lambda x

    where $A$ is the adjacency matrix of the graph `G` with eigenvalue
    $\lambda$. By virtue of the Perron–Frobenius theorem, there is a unique
    solution $x$, all of whose entries are positive, if $\lambda$ is the
    largest eigenvalue of the adjacency matrix $A$ ([2]_).

    Parameters
    ----------
    G : graph
      A networkx graph

    max_iter : integer, optional (default=100)
      Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional (default=None)
      Starting value of eigenvector iteration for each node.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality(G)
    >>> sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

    Raises
    ------
    NetworkXPointlessConcept
        If the graph `G` is the null graph.

    NetworkXError
        If each value in `nstart` is zero.

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    See Also
    --------
    eigenvector_centrality_numpy
    pagerank
    hits

    Notes
    -----
    The measure was introduced by [1]_ and is discussed in [2]_.

    The power iteration method is used to compute the eigenvector and
    convergence is **not** guaranteed. Our method stops after ``max_iter``
    iterations or when the change in the computed vector between two
    iterations is smaller than an error tolerance of
    ``G.number_of_nodes() * tol``. This implementation uses ($A + I$)
    rather than the adjacency matrix $A$ because it shifts the spectrum
    to enable discerning the correct eigenvector even for networks with
    multiple dominant eigenvalues.

    For directed graphs this is "left" eigenvector centrality which corresponds
    to the in-edges in the graph. For out-edges eigenvector centrality
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Phillip Bonacich.
       "Power and Centrality: A Family of Measures."
       *American Journal of Sociology* 92(5):1170–1182, 1986
       <http://www.leonidzhukov.net/hse/2014/socialnetworks/papers/Bonacich-Centrality.pdf>
    .. [2] Mark E. J. Newman.
       *Networks: An Introduction.*
       Oxford University Press, USA, 2010, pp. 169.

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    # If no initial vector is provided, start with the all-ones vector.
    if nstart is None:
        nstart = {v: 1 for v in G}
    if all(v == 0 for v in nstart.values()):
        raise nx.NetworkXError("initial vector cannot have all zero values")
    # Normalize the initial vector so that each entry is in [0, 1]. This is
    # guaranteed to never have a divide-by-zero error by the previous line.
    nstart_sum = sum(nstart.values())
    x = {k: v / nstart_sum for k, v in nstart.items()}
    nnodes = G.number_of_nodes()
    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        for n in x:
            for nbr in G[n]:
                w = G[n][nbr].get(weight, 1) if weight else 1
                x[nbr] += xlast[n] * w
        # Normalize the vector. The normalization denominator `norm`
        # should never be zero by the Perron--Frobenius
        # theorem. However, in case it is due to numerical error, we
        # assume the norm to be one instead.
        norm = math.hypot(*x.values()) or 1
        x = {k: v / norm for k, v in x.items()}
        # Check for convergence (in the L_1 norm).
        if sum(abs(x[n] - xlast[n]) for n in x) < nnodes * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)
