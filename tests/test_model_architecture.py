import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def test_single_neuron_output_shape():
    model = keras.Sequential(
        [
            layers.Input(shape=(10,)),
            layers.Dense(1),
        ]
    )

    output = model(tf.zeros((4, 10)))

    assert output.shape == (4, 1)


def test_deep_network_output_shape():
    model = keras.Sequential(
        [
            layers.Input(shape=(10,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    output = model(tf.zeros((4, 10)))

    assert output.shape == (4, 1)


def test_classification_model_output_range():
    model = keras.Sequential(
        [
            layers.Input(shape=(30,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    output = model(tf.zeros((8, 30)), training=False)

    assert output.shape == (8, 1)
    assert tf.reduce_all(output >= 0)
    assert tf.reduce_all(output <= 1)


def test_classification_model_contains_dropout():
    model = keras.Sequential(
        [
            layers.Input(shape=(30,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    assert any(
        isinstance(layer, layers.Dropout)
        for layer in model.layers
    )
