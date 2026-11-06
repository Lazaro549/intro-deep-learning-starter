"""Independent 80/20 held-out evaluation using the shared TensorFlow/Keras models.

The held-out test partition is used only for final metrics. Scaling is fitted
on the development partition, and early stopping uses an internal split of
that development data.
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
from sklearn.model_selection import train_test_split
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


TEST_SIZE = 0.20


def evaluate_regression():
    data = load_diabetes()
    X_dev, X_test, y_dev, y_test = train_test_split(
        data.data,
        data.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)

    reset_seed()
    model = build_regression_model(X_dev.shape[1])
    model.fit(
        X_dev,
        y_dev,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[make_early_stopping()],
        verbose=0,
    )

    predictions = model(X_test, training=False).numpy().reshape(-1)
    return {
        "test_samples": len(y_test),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }


def evaluate_classification():
    data = load_breast_cancer()
    X_dev, X_test, y_dev, y_test = train_test_split(
        data.data,
        data.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )

    scaler = StandardScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)

    reset_seed()
    model = build_classification_model(X_dev.shape[1])
    model.fit(
        X_dev,
        y_dev,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[make_early_stopping()],
        verbose=0,
    )

    probabilities = model(X_test, training=False).numpy().reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


if __name__ == "__main__":
    print("Held-Out Test Evaluation (TensorFlow/Keras)")
    print("=" * 46)
    print("\nRegression:")
    print(evaluate_regression())
    print("\nClassification:")
    print(evaluate_classification())
