import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap
from shap._explanation import Explanation


def explain_model(X: pd.DataFrame, result_dir: str, predict, seed) -> None:
    shap_values = get_shap_values(X, predict, seed)
    pickle.dump(shap_values, open(f"{result_dir}/shap_values.sav", "wb"))
    create_xai_plots(X, shap_values, result_dir)


def get_shap_values(X: pd.DataFrame, predict, seed) -> Explanation:
    X1000 = shap.maskers.Independent(X, max_samples=1000)
    explainer = shap.Explainer(predict, X1000, seed=seed)

    return explainer(X)


def create_xai_plots(
    X: pd.DataFrame, shap_values: Explanation, result_dir: str
) -> None:
    summary_plot(X, shap_values, result_dir)
    max_bar_plot(shap_values, result_dir)
    mean_bar_plot(shap_values, result_dir)


def summary_plot(X: pd.DataFrame, shap_values: Explanation, result_dir: str) -> None:
    fig, _ = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, X, color="coolwarm")
    fig.set_size_inches(20, 10)
    fig.savefig(f"{result_dir}/summary_plot.png", dpi=1200)
    plt.close(fig)


def max_bar_plot(shap_values: Explanation, result_dir: str) -> None:
    fig, _ = plt.subplots(nrows=1, ncols=1)
    shap.plots.bar(shap_values.abs.max(0))
    fig.set_size_inches(20, 10)
    fig.savefig(f"{result_dir}/max_bar_plot.png", dpi=1200)
    plt.close(fig)


def mean_bar_plot(shap_values: Explanation, result_dir: str) -> None:
    fig, _ = plt.subplots(nrows=1, ncols=1)
    shap.plots.bar(shap_values)
    fig.set_size_inches(20, 10)
    fig.savefig(f"{result_dir}/mean_bar_plot.png", dpi=1200)
    plt.close(fig)
