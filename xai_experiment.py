import argparse
import os
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helpers.poster_experiment_helper import SEED
from logging_utils import get_logger, get_timestamp
from model_tunning.model_tunning import load_graph_data, standard_data
from xai.model_helper import fit_model, get_model_predict, load_model
from xai.shap import explain_model

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

logger = get_logger(__name__, level="info", logdir="data/logs", console=True)
np.random.seed(SEED)


def read_data(graph):
    features = pd.read_csv(graph / "features.csv").drop(["node"], axis=1)
    labels = pd.read_csv(graph / "labels.csv").drop(["node"], axis=1)

    features = StandardScaler().fit_transform(features)

    return features, labels


def parse_args():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("graph_name", help="Graph name", type=str)
    my_parser.add_argument(
        "model_name",
        help="Model name",
        type=str,
        choices=[
            "RandomForestClassifier",
            "SVC",
            "LGBMClassifier",
            "KNeighborsClassifier",
            "LogisticRegression",
        ],
    )
    my_parser.add_argument(
        "graph_type",
        help="directed or undirected",
        type=str,
        choices=["directed", "undirected"],
    )
    my_parser.add_argument("label_name", help="Output label name for model", type=str)
    my_parser.add_argument("n_labels", help="Number of labels for output", type=str)
    my_parser.add_argument("test_graph", help="test_graph", type=str)
    my_parser.add_argument(
        "n_samples",
        help="number of data samples for shap",
        type=int,
        nargs="?",
        default=1000,
    )

    return my_parser.parse_args()


def main(args):
    logger.info(f"Process started: {get_timestamp()}")
    logger.info(
        f"Processing: {args.graph_name}, N_labels: {args.label_name}_{args.n_labels}, Model: {args.model_name}, Train_graph: {args.graph_name}, Test_graph: {args.test_graph}"
    )

    result_dir = Path(
        f"./data/results/{args.graph_type}/xai/{args.label_name}_{args.n_labels}/{args.model_name}/{args.graph_name}/{args.test_graph}"
    )
    # result_dir = Path(f'./data/results/{args.graph_type}/{args.graph_name}/xai/{args.model_name}/{args.test_graph}')
    graph_path = Path(f"data/processed/{args.graph_type}/{args.graph_name}")
    test_path = Path(f"data/processed/{args.graph_type}/{args.test_graph}")
    params_path = Path(f"data/params/{args.graph_type}/{args.graph_name}")

    result_dir.mkdir(parents=True, exist_ok=True)
    x_columns_df, Y = load_graph_data(graph_path)
    # X, Y = load_graph_data(graph_path)
    # X = standard_data(X)

    train_graph = args.graph_name
    test_graph = args.test_graph
    train_features, train_labels = read_data(graph_path)

    X, _, Y, _ = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=2021
    )
    if train_graph == test_graph:
        _, X_test, _, _ = train_test_split(
            train_features,
            train_labels,
            test_size=0.2,
            random_state=2021,
        )
    else:
        X_test, _ = read_data(test_path)

    # X_test, _ = load_graph_data(test_path)
    # X_test = standard_data(X_test)
    X_test = pd.DataFrame(data=X_test, columns=x_columns_df.columns)
    # print(X_test.head())

    start = perf_counter()
    model = load_model(
        args.model_name, str(params_path), f"{args.label_name}_{args.n_labels}"
    )
    model = fit_model(model, X, Y[f"{args.label_name}_{args.n_labels}"], SEED)
    predict = get_model_predict(args.model_name, model)

    n_samples = args.n_samples if args.n_samples else len(X_test)
    n_samples = n_samples if args.n_samples < len(X_test) else len(X_test)
    explain_model(
        X_test.sample(n=n_samples, random_state=SEED), result_dir, predict, SEED
    )
    stop = perf_counter()

    logger.info(f"Process complete: {get_timestamp()}")
    logger.info(
        f"Processed: {args.graph_name}, N_labels: {args.label_name}_{args.n_labels}, Model: {args.model_name}, Train_graph: {args.graph_name}, Test_graph: {args.test_graph}"
    )
    logger.info(f"Elapsed: {stop - start}s")


if __name__ == "__main__":
    main(parse_args())
