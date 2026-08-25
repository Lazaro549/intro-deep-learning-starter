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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 0
TEST_SIZE = 0.20


def evaluate_regression():
    data = load_diabetes()

    X_dev, X_test, y_dev, y_test = train_test_split(
        data.data,
        data.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

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
                    early_stopping=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_dev, y_dev)

    predictions = model.predict(X_test)

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

    model.fit(X_dev, y_dev)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "test_samples": len(y_test),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions)
        ),
        "recall": float(
            recall_score(y_test, predictions)
        ),
        "f1": float(
            f1_score(y_test, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
    }


if __name__ == "__main__":
    print("Held-Out Test Evaluation")
    print("=" * 30)

    print("\nRegression:")
    print(evaluate_regression())

    print("\nClassification:")
    print(evaluate_classification())
