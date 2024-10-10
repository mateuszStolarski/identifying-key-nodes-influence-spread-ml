import json
from inspect import signature
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

SEED = 2021
STANDARIZE_COLUMNS = ["core_number", "harmonic_centrality", "treshold"]


def get_kwargs(model, param_grid: dict) -> dict:
    kwargs = {}
    kwargs["param_distributions"] = param_grid
    kwargs["estimator"] = model
    kwargs["n_jobs"] = 8
    kwargs["cv"] = 5
    kwargs["refit"] = False
    kwargs["scoring"] = "f1_macro"
    kwargs["random_state"] = SEED
    kwargs["n_iter"] = 30

    return kwargs


def load_params(filename):
    params = Path(filename)

    if params.is_file():
        with params.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
    else:
        result = {}

    return result


def save_params(params, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(params, file)


def halving_search(model_class, param_grid: dict, X_train, y_train):
    model_signature = signature(model_class.__init__).parameters
    model = model_class()

    kwargs = get_kwargs(model=model, param_grid=param_grid)
    search = RandomizedSearchCV(**kwargs).fit(X_train, y_train)
    best_params = search.best_params_

    if "random_state" in model_signature:
        best_params["random_state"] = SEED

    return best_params


def standard_data(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns

    scaler = StandardScaler()

    # scaler.fit(df[STANDARIZE_COLUMNS])

    # standarized_columns = scaler.transform(df[STANDARIZE_COLUMNS])
    # df = df.drop(STANDARIZE_COLUMNS, axis=1).to_numpy()
    # df = np.c_[df, standarized_columns]

    df = scaler.fit_transform(df)

    return pd.DataFrame(data=df, columns=columns)


def load_graph_data(graph_dir):
    X = pd.read_csv(graph_dir / "features.csv")
    X = standard_data(X)
    X = X.drop(["node"], axis=1)

    df = pd.read_csv(graph_dir / "labels.csv")
    df = df.drop(["node"], axis=1)

    return X, df


def process_graph(X, df, result_subfolder, model, param):
    params_filename = result_subfolder / f"{model.__name__}.json"
    model_params = load_params(params_filename)

    for col in df.columns:
        y = df[col].values
        best_params = halving_search(model, param, X, y)
        model_params[col] = best_params

    save_params(model_params, params_filename)
