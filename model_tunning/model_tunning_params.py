LGBMClassifier_params = {
    "learning_rate": [0.001, 0.1, 0.25],
    "max_depth": [-1, 2, 4, 6],
    "num_leaves": [20, 30, 40],
    "min_child_samples": [10, 20, 30],
    "n_estimators": [50, 100, 200, 250, 300],
    "colsample_bytree": [1.0, 0.8, 0.5],
}

RandomForestClassifier_params = {
    "max_depth": [-1, 4, 6, 8, 10],
    "min_samples_split": [2, 3, 5, 7],
    "min_samples_leaf": [1, 2, 5, 10],
    "n_estimators": [50, 100, 200, 250, 300],
    "max_features": ["sqrt", None],
}

SVC_params = {
    "C": [0.5, 0.7, 1.0, 1.25, 1.5],
    "kernel": ["poly", "rbf"],
    "degree": [2, 3, 5],
    "max_iter": [1000],
}

KNeighborsClassifier_params = {
    "n_neighbors": list(range(5, 20)),
    "leaf_size": [20, 30, 40],
    "metric": ["cosine", "euclidean", "minkowski"],
}

LogisticRegression_params = {
    "penalty": ["l2", "none"],
    "C": [0.5, 0.7, 1.0, 1.25, 1.5],
    "max_iter": [1000],
}
