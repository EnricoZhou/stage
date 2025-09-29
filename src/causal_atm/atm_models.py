# atm_models.py
# ==============================================================================
# Causal Amortized Topic Model (C-ATM) in TensorFlow 2.x
# ==============================================================================

import tensorflow as tf
from tensorflow.keras import layers, Model


class Sampling(tf.keras.layers.Layer):
    """Reparameterization trick by sampling from N(mu, sigma^2)."""

    def call(self, inputs):
        mu, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps


def causal_atm_model(vocab_size=5000,
                     num_topics=50,
                     hidden_dim=200,
                     binary_outcome=True):
    """
    Causal Amortized Topic Model (C-ATM).

    Args:
        vocab_size: dimensione del vocabolario (bag-of-words).
        num_topics: numero di topics latenti.
        hidden_dim: dimensione layer hidden dell'encoder.
        binary_outcome: True se outcome binario.

    Returns:
        model: tf.keras.Model con output [g, q, recon].
    """

    # ----- Input: bag-of-words -----
    inputs = layers.Input(shape=(vocab_size,), name="bow_input")

    # Encoder
    h = layers.Dense(hidden_dim, activation="relu")(inputs)
    mu = layers.Dense(num_topics, name="mu")(h)
    logvar = layers.Dense(num_topics, name="logvar")(h)

    # Reparam trick (usiamo Sampling layer, NON tf.random.normal fuori dal grafo)
    z = Sampling()([mu, logvar])
    theta = layers.Softmax(name="theta")(z)

    # Decoder (ricostruzione BOW)
    recon = layers.Dense(vocab_size, activation="softmax", name="recon")(theta)

    # Propensity head
    g = layers.Dense(1, activation="sigmoid", name="g")(theta)

    # Outcome heads (q0 e q1 separati)
    if binary_outcome:
        q0 = layers.Dense(1, activation="sigmoid", name="q0")(theta)
        q1 = layers.Dense(1, activation="sigmoid", name="q1")(theta)
    else:
        q0 = layers.Dense(1, activation=None, name="q0")(theta)
        q1 = layers.Dense(1, activation=None, name="q1")(theta)

    model = Model(inputs=inputs, outputs=[g, q0, q1, recon], name="causal_atm")
    return model