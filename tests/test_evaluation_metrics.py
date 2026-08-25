import numpy as np

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def test_regression_metrics_are_finite():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    assert np.isfinite(mae)
    assert np.isfinite(rmse)
    assert np.isfinite(r2)

    assert mae >= 0
    assert rmse >= 0


def test_perfect_regression_has_zero_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true.copy()

    assert mean_absolute_error(y_true, y_pred) == 0
    assert mean_squared_error(y_true, y_pred) == 0
    assert r2_score(y_true, y_pred) == 1


def test_classification_metrics_are_valid():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.9])

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    assert 0 <= accuracy <= 1
    assert 0 <= auc <= 1


def test_perfect_classification():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.1, 0.9, 0.99])

    assert accuracy_score(y_true, y_pred) == 1.0
    assert roc_auc_score(y_true, y_prob) == 1.0
