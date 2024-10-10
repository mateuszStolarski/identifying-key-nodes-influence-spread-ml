import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from tqdm.auto import tqdm

from logging_utils import get_logger

logger = get_logger(__name__, level="info", logdir="data/logs", console=True)
warnings.filterwarnings("ignore")

BINS = list(range(2, 8))

COLUMNS = ["max_si_mean", "max_iter_mean", "max_recovered_mean"]

columns_processed, columns_rejected = 0, 0


acceptable_tresholds = {
    "Pubmed": [0.1, 0.15, 0.2],
    "Citeseer": [0.4, 0.3, 0.2],
    "Facebook": [0.1, 0.15, 0.2],
    "Github": [0.1, 0.15, 0.2],
}


def get_files(path):
    folder = Path(path)
    pattern = re.compile(r".*\d.csv")  # ends with *digit*.csv
    labels = [item for item in folder.glob("*") if pattern.match(str(item))]

    return labels


def discretize(df, n_bins):
    model = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")

    new_columns = []
    result = []

    for col in COLUMNS:
        data = df[col].values.reshape(-1, 1)
        result.append(model.fit_transform(data).astype(int))

        new_col = "_".join(col.split("_")[:2])
        new_columns.append(new_col + f"_{n_bins}")

        assert len(np.unique(result[-1]) == n_bins)

    return pd.DataFrame(np.hstack(result), columns=new_columns)


def clear(df, treshold=5):
    global columns_processed, columns_rejected

    good_cols = []
    for col in df.columns:
        sizes = df[col].value_counts().values

        if sizes[-1] >= treshold:
            good_cols.append(col)
        else:
            logger.debug(f"    Rejected: {col}")
            columns_rejected += 1
        columns_processed += 1

    if not good_cols:
        return None
    return df[good_cols]


def process_folder(file_names, bins):
    for file in file_names:
        logger.debug(f"Graph: {file.stem}")
        result = []
        df = pd.read_csv(str(file))
        # original = df.copy(deep=True)
        for n in bins:
            processed = discretize(df.copy(), n)
            if processed is not None:
                processed = clear(processed)
                result.append(processed)

        # result.insert(0, original)  # add original data to the result
        result = pd.concat(result, axis=1)
        new_name = str(file)[:-4] + "_discretized.csv"
        result.to_csv(new_name, index=False)


def clear_discretized():
    data_folder = Path("data/processed")
    for file in data_folder.rglob("*"):
        if file.stem.endswith("discretized"):
            file.unlink()


def get_no_classes(folder: Path):
    files = [item for item in folder.glob("*.csv") if item.stem.endswith("discretized")]
    logger.debug(f"Classes for {folder.stem}")

    for file in files:
        temp = pd.read_csv(file)
        tresh = file.stem.split("_")[1]
        logger.debug(f"Treshold: {tresh} - {len(temp.columns)}")


def merge_files(folder: Path, delete=False):
    stack = []
    files = [item for item in folder.glob("*.csv") if item.stem.endswith("discretized")]
    tresholds = acceptable_tresholds.get(folder.stem)

    for file in files:
        tresh = file.stem.split("_")[1]

        if float(tresh) in tresholds:
            df = pd.read_csv(file)
            df.insert(0, "treshold", tresh)
            df.insert(0, "node", df.index)
            stack.append(df)

        if delete:
            file.unlink()

    if stack:
        result = pd.concat(stack, axis=0, ignore_index=True)
        result = result.dropna(axis=1)
        result.to_csv(folder / "labels.csv", index=False)


def generate_features(folder: Path):
    centralities = pd.read_csv(folder / "raw_centrality_features.csv")
    centralities.node = centralities.node.astype(int)

    labels = pd.read_csv(folder / "labels.csv")

    cut = labels[["node", "treshold"]]
    result = cut.merge(centralities, how="inner", on="node")

    assert len(result) == len(labels)

    result = result.sort_values(by=["node", "treshold"])
    result.to_csv(folder / "features.csv", index=False)

    labels = labels.sort_values(by=["node", "treshold"])
    labels = labels.drop(["treshold"], axis=1)
    labels.to_csv(folder / "labels.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_type", type=str, help="directed or undirected")

    return parser.parse_args()


def main(args):
    data_folder = Path(f"data/processed/{args.graph_type}")
    datasets = [item for item in data_folder.glob("*") if item.is_dir()]
    for file in data_folder.rglob("*.csv"):
        if file.stem.startswith("directed"):
            file.rename(str(file).replace("directed_", ""))
        if file.stem.startswith("undirected"):
            file.rename(str(file).replace("undirected_", ""))

    for folder in tqdm(datasets):
        file_names = get_files(str(folder))
        process_folder(file_names, BINS)

        get_no_classes(folder)
        merge_files(folder, delete=True)
        generate_features(folder)

    logger.info(f"Column names: {COLUMNS}")
    logger.info(f"Columns processed: {columns_processed}")
    logger.info(f"Columns rejected: {columns_rejected}")
    logger.info(f"Columns accepted: {columns_processed - columns_rejected}")
    logger.info(
        f"Acceptance ratio: {(columns_processed - columns_rejected) / columns_processed}"
    )


if __name__ == "__main__":
    main(parse_args())
