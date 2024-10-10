import networkx as nx
import numpy as np

from .custom_centralities import (avarage_neighbors_degree,
                                  avarage_undirected_neighbors_degree,
                                  closeness_centrality, degree_centrality,
                                  eigenvector_centrality, in_degree_centrality,
                                  local_reaching_centrality,
                                  out_degree_centrality, voterank)

HEADERS = [
    "node",
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "avarage_neighbors_degree",
    "closeness_centrality",
    "local_reaching_centrality",
    "voterank",
    "eigenvector_centrality",
    "pagerank",
    "load_centrality",
    "clustering",
    "core_number",
    "harmonic_centrality",
    "betweenness_centrality",
]


def preprocess(data):
    if isinstance(data, int):
        return np.arange(data)

    return list(data.values())


def normalize(data):
    if isinstance(data, int):
        return np.arange(data)

    data2 = list(data.values())
    data2.sort(reverse=True)
    result = [data2.index(number) for number in data.values()]
    return np.array(result) + 1 / len(data)


def normalize_numpy(data):
    if isinstance(data, int):
        return np.arange(data)

    data = list(data.values())
    indicies = np.argsort(np.argsort(data))
    result = indicies + 1 / len(data)

    return result


def get_features(G) -> np.ndarray:
    metrics = [
        len,
        nx.degree_centrality,
        nx.in_degree_centrality,
        nx.out_degree_centrality,
        nx.closeness_centrality,
        nx.eigenvector_centrality,
        nx.pagerank,
        nx.load_centrality,
        nx.betweenness_centrality,
        nx.clustering,
        nx.core_number,
        nx.harmonic_centrality,
    ]

    result = np.stack([normalize(func(G)) for func in metrics])
    result = np.swapaxes(result, 0, 1)

    return result


def get_raw_features(G) -> np.ndarray:
    metrics = [
        len,
        degree_centrality,
        in_degree_centrality,
        out_degree_centrality,
        avarage_neighbors_degree,
        closeness_centrality,
        local_reaching_centrality,
        voterank,
        eigenvector_centrality,
        nx.pagerank,
        nx.load_centrality,
        nx.clustering,
        nx.core_number,
        nx.harmonic_centrality,
    ]

    result = [preprocess(func(G)) for func in metrics]
    result.append(preprocess(nx.betweenness_centrality(G, normalized=False)))
    result = np.stack(result)
    result = np.swapaxes(result, 0, 1)

    return result


def get_raw_undirected_features(G) -> np.ndarray:
    metrics = [
        len,
        degree_centrality,
        degree_centrality,
        degree_centrality,
        avarage_undirected_neighbors_degree,
        closeness_centrality,
        local_reaching_centrality,
        voterank,
        eigenvector_centrality,
        nx.pagerank,
        nx.load_centrality,
        nx.clustering,
        nx.core_number,
        nx.harmonic_centrality,
    ]

    result = [preprocess(func(G)) for func in metrics]
    result.append(preprocess(nx.betweenness_centrality(G, normalized=False)))
    result = np.stack(result)
    result = np.swapaxes(result, 0, 1)

    return result
