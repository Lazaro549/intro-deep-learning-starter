import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 0
N_SPLITS = 5


def evaluate_regression():
    data = load_diabetes()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                ),
            ),
        ]
    )

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        model,
        data.data,
        data.target,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )

    return {
        "mae_mean": float(-np.mean(scores["test_mae"])),
        "mae_std": float(np.std(scores["test_mae"])),
        "rmse_mean": float(-np.mean(scores["test_rmse"])),
        "rmse_std": float(np.std(scores["test_rmse"])),
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"])),
    }


def evaluate_classification():
    data = load_breast_cancer()

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        model,
        data.data,
        data.target,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )

    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "precision_mean": float(np.mean(scores["test_precision"])),
        "precision_std": float(np.std(scores["test_precision"])),
        "recall_mean": float(np.mean(scores["test_recall"])),
        "recall_std": float(np.std(scores["test_recall"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
    }


if __name__ == "__main__":
    print("5-Fold Cross-Validation")
    print("=" * 30)

    print("\nRegression:")
    print(evaluate_regression())

    print("\nClassification:")
    print(evaluate_classification())
