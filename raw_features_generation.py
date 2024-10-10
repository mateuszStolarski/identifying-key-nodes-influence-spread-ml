import argparse
from time import perf_counter

from networkx import read_graphml, selfloop_edges

from feature_extraction.centrality_features import (
    HEADERS, get_raw_features, get_raw_undirected_features)
from feature_extraction.ic_influence import create_directory, to_csv
from logging_utils import get_logger, get_timestamp

logger = get_logger(__name__, level="info", logdir="data/logs", console=True)


def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("graph_name", help="Graph name", type=str)
    my_parser.add_argument("direction", help="Directed/undirected", type=str)

    return my_parser.parse_args()


def main(args):
    result_dir = f"./data/processed/{args.direction}/{args.graph_name}/"

    create_directory(result_dir)
    logger.info(f"Process started: {get_timestamp()}")

    logger.info(f"Processing: {args.graph_name}")
    graph = read_graphml(f"./data/raw/{args.graph_name}.gml", node_type=int)
    graph.remove_edges_from(selfloop_edges(graph))
    if "undirected" in args.direction:
        graph = graph.to_undirected()
    logger.info("Graph loaded")

    start = perf_counter()
    if "undirected" in args.direction:
        centralities = get_raw_undirected_features(graph)
    else:
        centralities = get_raw_features(graph)
    stop = perf_counter()
    logger.info(f"Process complete: {get_timestamp()}")
    logger.info(f"Centralities done. Elapsed: {stop - start}s")

    to_csv(result_dir + "raw_centrality_features.csv", centralities, header=HEADERS)


if __name__ == "__main__":
    main(parse_args())
