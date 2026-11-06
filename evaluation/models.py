"""Shared TensorFlow/Keras model definitions for evaluation scripts."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


RANDOM_STATE = 0
EPOCHS = 200
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15
PATIENCE = 15


def reset_seed():
    """Reset TensorFlow's random state before each independent model fit."""
    tf.keras.utils.set_random_seed(RANDOM_STATE)


def build_regression_model(input_dim: int) -> keras.Model:
    """Build the regression network used by both CV and held-out evaluation."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_classification_model(input_dim: int) -> keras.Model:
    """Build the binary classifier used by both CV and held-out evaluation."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def make_early_stopping():
    """Return a fresh callback so each fold gets independent callback state."""
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
