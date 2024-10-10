import argparse
from time import perf_counter

from networkx import read_graphml, selfloop_edges

from feature_extraction.ic_influence import (create_directory,
                                             generate_single_labels)
from logging_utils import get_logger, get_timestamp

logger = get_logger(__name__, level="info", logdir="data/logs", console=True)


def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("graph_name", help="Graph name", type=str)
    my_parser.add_argument(
        "-t", "--treshold", help="Treshold value", nargs="?", default=0.1, type=float
    )

    return my_parser.parse_args()


def main(args):
    result_dir = f"./data/processed/directed/{args.graph_name}/"
    treshold = args.treshold

    create_directory(result_dir)
    logger.info(f"Process started: {get_timestamp()}")

    logger.info(f"Processing: {args.graph_name}")
    graph = read_graphml(f"./data/raw/{args.graph_name}.gml", node_type=int)
    graph.remove_edges_from(selfloop_edges(graph))
    logger.info("Graph loaded")

    start = perf_counter()
    generate_single_labels(graph, result_dir, treshold)
    stop = perf_counter()

    logger.info(f"Process complete: {get_timestamp()}")
    logger.info(f"Elapsed: {stop - start}s")


if __name__ == "__main__":
    main(parse_args())
