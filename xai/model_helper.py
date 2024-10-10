from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from model_tunning.model_tunning import load_params


def get_model_predict(model_name: str, model):
    return {
        RandomForestClassifier.__name__: lambda x: x.predict,
        SVC.__name__: lambda x: x.predict,
        LGBMClassifier.__name__: lambda x: x.predict,
        LogisticRegression.__name__: lambda x: x.predict,
        KNeighborsClassifier.__name__: lambda x: x.predict,
    }[model_name](model)


def get_model(model_name: str):
    return {
        RandomForestClassifier.__name__: RandomForestClassifier,
        SVC.__name__: SVC,
        LGBMClassifier.__name__: LGBMClassifier,
        LogisticRegression.__name__: LogisticRegression,
        KNeighborsClassifier.__name__: KNeighborsClassifier,
    }[model_name]


def load_model(model_name: str, graph_path: str, label: str):
    params = load_params(f"{graph_path}/{model_name}.json")
    model = get_model(model_name)

    return model(**params[label])


def fit_model(model, X, Y, seed):
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=seed)
    model.fit(X_train, y_train)

    return model
