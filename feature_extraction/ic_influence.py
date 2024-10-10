import csv
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from feature_extraction.ic_features import prepare_and_execute_ic

HEADER = ["node", "treshold"]


def to_csv(filename, data, header) -> None:
    with open(filename, "w", encoding="utf-8") as handle:
        csvwriter = csv.writer(handle)
        csvwriter.writerow(header)
        csvwriter.writerows(data)


def concat_to_csv(filename, data, header) -> None:
    df = pd.read_csv(filename)
    current_headers = list(df.columns)
    header = current_headers + header
    df = df.to_numpy()

    data = np.c_[df, data]
    with open(filename, "w", encoding="utf-8") as handle:
        csvwriter = csv.writer(handle)
        csvwriter.writerow(header)
        csvwriter.writerows(data)


def create_directory(directory) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def nodes_influence(graph, treshold) -> np.ndarray:
    si_values = []
    iter_values = []
    recovered_values = []

    for node in graph:
        max_si, max_iter, max_recovered = prepare_and_execute_ic(node, graph, treshold)
        si_values.append(max_si)
        iter_values.append(max_iter)
        recovered_values.append(max_recovered)

    return si_values, iter_values, recovered_values


def prepare_data(si_values, iter_values, recovered_values) -> np.ndarray:
    return np.dstack((si_values, iter_values, recovered_values)).squeeze()


def prepare_info_data(graph, treshold) -> np.ndarray:
    tresholds = np.full_like(graph.nodes, treshold, dtype=float)
    return np.dstack((graph.nodes, tresholds)).squeeze()


def calculate_mean_influence(result_dir, treshold, filename, n_simulations):
    df = pd.read_csv(filename)

    si_headers = []
    iter_headers = []
    recovered_headers = []
    for iteration in range(n_simulations):
        si_headers.append(f"max_si_{iteration}")
        iter_headers.append(f"max_iter_{iteration}")
        recovered_headers.append(f"max_recovered_{iteration}")

    si_data = df[si_headers].to_numpy()
    iter_data = df[iter_headers].to_numpy()
    recovered_data = df[recovered_headers].to_numpy()

    si_mean = np.mean(si_data, axis=1)
    iter_mean = np.mean(iter_data, axis=1)
    recovered_mean = np.mean(recovered_data, axis=1)
    data_header = ["max_si_mean", "max_iter_mean", "max_recovered_mean"]

    data = prepare_data(si_mean, iter_mean, recovered_mean)
    concat_to_csv(result_dir + f"labels_{treshold}.csv", data, data_header)


def generate_labels_flat(graph, result_dir, tresholds, n_simulations) -> None:
    """
    Batch labels generation for all params
    """
    for treshold in tqdm(tresholds):
        data = prepare_info_data(graph, treshold)
        to_csv(result_dir + f"labels_{treshold}.csv", data, HEADER)

        for iteration in tqdm(range(n_simulations)):
            data_header = [
                f"max_si_{iteration}",
                f"max_iter_{iteration}",
                f"max_recovered_{iteration}",
            ]
            si_values, iter_values, recovered_values = nodes_influence(graph, treshold)

            data = prepare_data(si_values, iter_values, recovered_values)
            concat_to_csv(result_dir + f"labels_{treshold}.csv", data, data_header)

        calculate_mean_influence(
            result_dir, treshold, result_dir + f"labels_{treshold}.csv", n_simulations
        )


def generate_single_labels_flat(graph, filepath, treshold, iteration) -> None:
    """
    Batch labels generation for single set of params
    """
    data_header = [
        f"max_si_{iteration}",
        f"max_iter_{iteration}",
        f"max_recovered_{iteration}",
    ]
    si_values, iter_values, recovered_values = nodes_influence(graph, treshold)

    data = prepare_data(si_values, iter_values, recovered_values)
    concat_to_csv(filepath, data, data_header)


def inner_func_concurrent(treshold, graph, result_dir, n_simulations):
    data = prepare_info_data(graph, treshold)
    to_csv(result_dir + f"labels_{treshold}.csv", data, HEADER)

    for iteration in tqdm(range(n_simulations)):
        data_header = [
            f"max_si_{iteration}",
            f"max_iter_{iteration}",
            f"max_recovered_{iteration}",
        ]
        si_values, iter_values, recovered_values = nodes_influence(graph, treshold)

        data = prepare_data(si_values, iter_values, recovered_values)
        concat_to_csv(result_dir + f"labels_{treshold}.csv", data, data_header)

    calculate_mean_influence(
        result_dir, treshold, result_dir + f"labels_{treshold}.csv", n_simulations
    )


def generate_labels(graph, result_dir, tresholds, n_simulations, n_jobs=1) -> None:
    create_directory(result_dir)

    if n_jobs == 1:
        return generate_labels_flat(graph, result_dir, tresholds, n_simulations)

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        list(
            executor.map(
                inner_func_concurrent,
                tresholds,
                repeat(graph),
                repeat(result_dir),
                repeat(n_simulations),
            )
        )


def get_iteration(filepath) -> int:
    with open(filepath, "r") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

    iteration = 1
    if len(fieldnames) > 2:
        last_max_recovered = fieldnames[::-1][0]
        idx = last_max_recovered.index("_", 4)
        last_iteration = last_max_recovered[idx + 1 : len(last_max_recovered)]
        iteration += int(last_iteration)

    return iteration


def generate_single_labels(graph, result_dir, treshold) -> None:
    create_directory(result_dir)

    filepath = result_dir + f"bem_labels_{treshold}.csv"
    if not os.path.exists(filepath):
        data = prepare_info_data(graph, treshold)
        to_csv(filepath, data, HEADER)

    iteration = get_iteration(filepath)

    generate_single_labels_flat(graph, filepath, treshold, iteration)
