import argparse
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tqdm.auto import tqdm

from logging_utils import get_logger, get_timestamp
from model_tunning import load_graph_data, model_tunning_params, process_graph

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
logger = get_logger(__name__, level="info", logdir="data/logs", console=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_type",
        type=str,
        help="directed or undirected",
        choices=["directed", "undirected"],
    )
    parser.add_argument("-g", "--graphs", nargs="*", help="List of graphs to process")

    return parser.parse_args()


def main(args):
    data_dir = Path(f"data/processed/{args.graph_type}/")
    params_dir = Path(f"data/params/{args.graph_type}/")

    params_dir.mkdir(parents=True, exist_ok=True)

    models = [
        LogisticRegression,
        KNeighborsClassifier,
        SVC,
        RandomForestClassifier,
        LGBMClassifier,
    ]

    if args.graphs:
        graph_folders = [data_dir / item for item in args.graphs]
    else:
        graph_folders = list(data_dir.glob("*"))

    logger.info(f"Process started: {get_timestamp()}")
    start = perf_counter()

    for graph_folder in tqdm(graph_folders, desc="Graphs"):
        logger.debug(f"Graph loaded: {graph_folder.stem}")

        graph_subfolder = params_dir / graph_folder.stem
        graph_subfolder.mkdir(parents=True, exist_ok=True)

        X, df = load_graph_data(graph_folder)

        for model in tqdm(models, desc="Models", leave=False):
            params = getattr(model_tunning_params, f"{model.__name__}_params")
            logger.debug(f"    Model loaded: {model.__name__}")
            process_graph(X, df, graph_subfolder, model, params)

    stop = perf_counter()
    logger.info(f"Process complete: {get_timestamp()}")
    logger.info(f"Elapsed: {stop - start}s")


if __name__ == "__main__":
    main(parse_args())
