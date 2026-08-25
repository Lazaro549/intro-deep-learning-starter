import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 0


def test_diabetes_preprocessing():
    data = load_diabetes()

    X_train, X_valid, _, _ = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    assert X_train_scaled.shape[1] == 10
    assert X_valid_scaled.shape[1] == 10
    assert np.isfinite(X_train_scaled).all()
    assert np.isfinite(X_valid_scaled).all()

    # Training data should be approximately standardized.
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)


def test_breast_cancer_preprocessing():
    data = load_breast_cancer()

    X_train, X_valid, y_train, y_valid = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    assert X_train_scaled.shape[1] == 30
    assert X_valid_scaled.shape[1] == 30
    assert len(y_train) == X_train_scaled.shape[0]
    assert len(y_valid) == X_valid_scaled.shape[0]
    assert np.isfinite(X_train_scaled).all()
    assert np.isfinite(X_valid_scaled).all()
