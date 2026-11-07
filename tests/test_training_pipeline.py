"""Integration tests for the shared end-to-end model training workflow."""

import numpy as np
import pytest

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.models import (
    build_classification_model,
    build_regression_model,
    reset_seed,
)


@pytest.mark.parametrize(
    ("dataset_loader", "model_builder", "expected_features", "task"),
    [
        (load_diabetes, build_regression_model, 10, "regression"),
        (load_breast_cancer, build_classification_model, 30, "classification"),
    ],
)
def test_complete_training_and_prediction_pipeline(
    dataset_loader, model_builder, expected_features, task
):
    """Exercise data split, scaling, model construction, training, and prediction."""
    data = dataset_loader()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=0,
        stratify=data.target if task == "classification" else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    assert X_train_scaled.shape == (len(y_train), expected_features)
    assert X_test_scaled.shape == (len(y_test), expected_features)

    reset_seed()
    model = model_builder(input_dim=X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    predictions = model.predict(X_test_scaled, verbose=0)

    assert "loss" in history.history
    assert len(history.history["loss"]) == 2
    assert predictions.shape == (len(y_test), 1)
    assert np.isfinite(predictions).all()

    if task == "classification":
        assert np.all((predictions >= 0.0) & (predictions <= 1.0))


@pytest.mark.parametrize(
    ("model_builder", "expected_features"),
    [
        (build_regression_model, 10),
        (build_classification_model, 30),
    ],
)
def test_shared_models_reject_wrong_input_dimension(model_builder, expected_features):
    """A model must reject feature matrices with an unexpected column count."""
    import tensorflow as tf

    model = model_builder(input_dim=expected_features)

    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        model.predict(np.zeros((2, expected_features + 1), dtype=np.float32), verbose=0)


def test_scaler_is_fitted_only_on_training_data():
    """Extreme held-out values must not affect the scaler's learned statistics."""
    X_train = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    X_test = np.array([[1_000_000.0, -1_000_000.0]])

    scaler = StandardScaler()
    scaler.fit(X_train)

    expected_mean = X_train.mean(axis=0)
    expected_scale = X_train.std(axis=0)

    X_test_scaled = scaler.transform(X_test)

    np.testing.assert_allclose(scaler.mean_, expected_mean)
    np.testing.assert_allclose(scaler.scale_, expected_scale)
    assert not np.allclose(scaler.mean_, np.vstack([X_train, X_test]).mean(axis=0))
    assert X_test_scaled.shape == (1, 2)
