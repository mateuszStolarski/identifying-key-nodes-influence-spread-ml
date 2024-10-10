import argparse
import csv
import json
import os
import sys
import warnings
from inspect import signature
from pathlib import Path
from time import perf_counter

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from logging_utils import get_logger, get_timestamp

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

MODELS = [
    LogisticRegression,
    KNeighborsClassifier,
    SVC,
    RandomForestClassifier,
    LGBMClassifier,
]

logger = get_logger(__name__, level="info", logdir="data/logs/", console=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "graph_type", type=str, choices=["directed", "undirected", "undirected2"]
    )
    parser.add_argument("-n", "--n-jobs", type=int, default=8)

    parser.add_argument(
        "-g",
        "--graphs",
        nargs="*",
        help="List of graphs to process (as training graphs)",
    )
    # ALL graphs are used as test graphs
    return parser.parse_args()


def to_csv(data, filename):
    with open(filename, "w", encoding="utf-8") as handle:
        csvwriter = csv.writer(handle)
        csvwriter.writerows(data)


def read_data(graph):
    features = pd.read_csv(graph / "features.csv").drop(["node"], axis=1)
    labels = pd.read_csv(graph / "labels.csv").drop(["node"], axis=1)

    features = StandardScaler().fit_transform(features)

    return features, labels


def read_params(graph, model):
    params_dir = ["params" if item == "processed" else item for item in graph.parts]
    params_dir = Path(*params_dir)

    with open(params_dir / (model + ".json"), "r", encoding="utf-8") as handle:
        params = json.load(handle)

    return params


def process_graph(train_graph, test_graphs, n_jobs):
    graph_result = []
    train_features, train_labels = read_data(train_graph)

    train_features, _, train_labels, _ = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=2021
    )

    columns = train_labels.columns

    for model_class in tqdm(MODELS, desc="Models", leave=False):
        logger.debug(f"    Fitting model: {model_class.__name__}")

        params = read_params(train_graph, model_class.__name__)
        model_signature = signature(model_class.__init__).parameters

        for column in tqdm(columns, desc="Columns", leave=False):
            model_params = params[column]
            if "n_jobs" in model_signature:
                model_params["n_jobs"] = n_jobs

            model = model_class(**model_params)
            model = model.fit(train_features, train_labels[column].values)

            for test_graph in tqdm(test_graphs, desc="Test graphs", leave=False):
                if train_graph == test_graph:
                    _, test_features, _, test_labels = train_test_split(
                        train_features,
                        train_labels,
                        test_size=0.2,
                        random_state=2021,
                    )
                else:
                    test_features, test_labels = read_data(test_graph)

                row = [
                    train_graph.stem,
                    test_graph.stem,
                    model_class.__name__,
                    column[:-2],
                    column[-1],
                ]
                if column in test_labels:
                    row += parse_report(
                        classification_report(
                            model.predict(test_features),
                            test_labels[column],
                            output_dict=True,
                        )
                    )
                graph_result.append(row)

    return graph_result


def parse_report(report):
    # accuracy, macro, topcls
    result = []

    result.append(report["accuracy"])

    macro = report["macro avg"]
    del macro["support"]
    result += list(macro.values())

    topcls = report[str(len(report) - 4)]
    del topcls["support"]
    result += list(topcls.values())

    return result


def main(args):
    data_dir = Path(f"data/processed/{args.graph_type}/")
    result_dir = Path(f"data/results/{args.graph_type}/")
    result_dir.mkdir(parents=True, exist_ok=True)

    test_graphs = list(data_dir.glob("*"))
    # ALL graphs are used as test graphs

    if args.graphs:
        train_graphs = [data_dir / item for item in args.graphs]
    else:
        train_graphs = test_graphs.copy()

    header = ["train_graph", "test_graph", "model", "label_name", "n_classes"]
    header += [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1-score",
        "topcls_precision",
        "topcls_recall",
        "topcls_f1-score",
    ]
    header = [header]
    logger.info(f"Process started: {get_timestamp()}")
    start = perf_counter()

    for train_graph in tqdm(train_graphs, desc="Train graphs"):
        logger.debug(f"Processing graph: {train_graph.stem}")
        result = process_graph(train_graph, test_graphs, args.n_jobs)

        to_csv(
            header + result,
            result_dir / f"result_{train_graph.stem}@{get_timestamp()}.csv",
        )

    stop = perf_counter()
    logger.info(f"Process complete: {get_timestamp()}")
    logger.info(f"Elapsed: {(stop - start):.2f}s")


if __name__ == "__main__":
    main(parse_args())
