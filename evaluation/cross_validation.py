"""Five-fold cross-validation using the same TensorFlow/Keras models as held-out testing.

Each fold fits its own scaler on the fold's training partition. Early stopping
uses a small validation split carved out of that training partition, never the
fold's evaluation partition.
"""

import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from evaluation.models import (
    BATCH_SIZE,
    EPOCHS,
    RANDOM_STATE,
    VALIDATION_SPLIT,
    build_classification_model,
    build_regression_model,
    make_early_stopping,
    reset_seed,
)


N_SPLITS = 5


def _mean_std(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def evaluate_regression():
    data = load_diabetes()
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for train_idx, test_idx in cv.split(data.data, data.target):
        X_train, X_test = data.data[train_idx], data.data[test_idx]
        y_train, y_test = data.target[train_idx], data.target[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        reset_seed()
        model = build_regression_model(X_train.shape[1])
        model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[make_early_stopping()],
            verbose=0,
        )
        predictions = model(X_test, training=False).numpy().reshape(-1)

        fold_metrics.append(
            {
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
            }
        )

    return {
        metric: _mean_std([fold[metric] for fold in fold_metrics])
        for metric in ("mae", "rmse", "r2")
    }


def evaluate_classification():
    data = load_breast_cancer()
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics = []

    for train_idx, test_idx in cv.split(data.data, data.target):
        X_train, X_test = data.data[train_idx], data.data[test_idx]
        y_train, y_test = data.target[train_idx], data.target[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        reset_seed()
        model = build_classification_model(X_train.shape[1])
        model.fit(
            X_train,
            y_train,
            validation_split=VALIDATION_SPLIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[make_early_stopping()],
            verbose=0,
        )
        probabilities = model(X_test, training=False).numpy().reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)

        fold_metrics.append(
            {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

    return {
        metric: _mean_std([fold[metric] for fold in fold_metrics])
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }


if __name__ == "__main__":
    print(f"{N_SPLITS}-Fold TensorFlow/Keras Cross-Validation")
    print("=" * 42)
    print("\nRegression:")
    print(evaluate_regression())
    print("\nClassification:")
    print(evaluate_classification())
