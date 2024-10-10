import math

import numpy as np


def calculate_range(max_scale, min_scale, n_labels) -> float:
    return (max_scale - min_scale) / n_labels


def calculate_label(si, min_scale, range, n_labels) -> float:
    label = math.floor((si - min_scale) / range) + 1
    if label == n_labels + 1:
        label -= 1

    return label


def discretize(influence_vals: np.ndarray, n_labels: int) -> np.ndarray:
    max_scale = max(influence_vals)
    min_scale = min(influence_vals)
    si_range = calculate_range(max_scale, min_scale, n_labels)

    label_values = []
    for si_val in influence_vals:
        label = calculate_label(si_val, min_scale, si_range, n_labels)
        label_values.append(label)

    return label_values
