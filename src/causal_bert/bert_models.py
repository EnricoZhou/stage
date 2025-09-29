# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT models that are compatible with TF 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
# from tensorflow.keras.layers import Lambda

# AGGOIUNTO
# from keras import ops
import keras

from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert_models import pretrain_model


def get_dragon_heads(binary_outcome: bool):
    """
    Simplest (and most common) variant of heads of dragonnet

    Returns: a keras layer with signature
    [float vector] -> {g: float, Q0: float, Q1: float}
    """

    def dragon_heads(z: tf.Tensor):
        g = tf.keras.layers.Dense(1, activation='sigmoid', name='g')(z)

        if binary_outcome:
            activation = 'sigmoid'
        else:
            activation = None

        q0 = tf.keras.layers.Dense(200, activation='relu')(z)
        q0 = tf.keras.layers.Dense(200, activation='relu')(q0)
        q0 = tf.keras.layers.Dense(1, activation=activation, name='q0')(q0)

        q1 = tf.keras.layers.Dense(200, activation='relu')(z)
        q1 = tf.keras.layers.Dense(200, activation='relu')(q1)
        q1 = tf.keras.layers.Dense(1, activation=activation, name='q1')(q1)

        return g, q0, q1

    return dragon_heads


def get_hydra_heads(binary_outcome: bool, num_treatments: int, missing_outcomes: bool):
    """
    A variant of dragonnet allowing possibly many treatment levels

    Returns: a keras layer with signature
    [float vector] -> {g: float, Q0: float, Q1: float}
    """

    def hydra_heads(z: tf.Tensor):
        if binary_outcome:
            activation = 'sigmoid'
        else:
            activation = None

        qz = tf.keras.layers.Dense(200, activation='relu')(z)
        qz = tf.keras.layers.Dense(200, activation='relu')(qz)

        q = []
        for treat in range(num_treatments):
            q.append(tf.keras.layers.Dense(1, activation=activation, name=f"q{treat}")(qz))

        if not missing_outcomes:
            g = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g')(z)
            return g, q
        else:
            g0 = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g0')(z)
            g1 = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g1')(z)
            m = tf.keras.layers.Dense(1, activation='sigmoid', name='y_is_obs')(z)
            return g0, g1, m, q

    return hydra_heads


def get_hydra_heads2(binary_outcome: bool, num_treatments: int, missing_outcomes: bool):
    """
    A variant of dragonnet allowing possibly many treatment levels

    Returns: a keras layer with signature
    [float vector] -> {g: float, Q0: float, Q1: float}
    """

    def hydra_heads(z: tf.Tensor):
        if binary_outcome:
            activation = 'sigmoid'
        else:
            activation = None

        qz = tf.keras.layers.Dense(200, activation='relu')(z)
        qz = tf.keras.layers.Dense(200, activation='relu')(qz)
        qs = tf.keras.layers.Dense(num_treatments, activation=activation, name='all_qs')(qz)

        q = []
        for treat, q_out in enumerate(tf.split(qs, num_or_size_splits=num_treatments, axis=-1)):
            renamer = Lambda(tf.identity, name=f"q{treat}")
            q += [renamer(q_out)]

        if not missing_outcomes:
            g = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g')(z)
            return g, q
        else:
            g0 = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g0')(z)
            g1 = tf.keras.layers.Dense(num_treatments, activation='softmax', name='g1')(z)
            m = tf.keras.layers.Dense(1, activation='sigmoid', name='y_is_obs')(z)
            return g0, g1, m, q

    return hydra_heads


def dragon_model_simple(bert_config,
                        max_seq_length: int,
                        binary_outcome: bool,
                        hub_module_url=None):
    """BERT dragon model in functional API style.

    Args:
      bert_config: BertConfig, the config defines the core BERT model.
      max_seq_length: integer, the maximum input sequence length.
      binary_outcome: bool, whether outcome is binary
      hub_module_url: (Experimental) TF-Hub path/url to Bert module.

    Returns:
      Combined prediction model (words, sample_weight, type) -> (one-hot labels)
      BERT sub-model (words, sample_weight, type) -> {g: float, Q0: float, Q1: float}
    """
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    if hub_module_url:
        bert_model = hub.KerasLayer(hub_module_url, trainable=True)
        pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
    else:
        bert_model = modeling.get_bert_model(
            input_word_ids,
            input_mask,
            input_type_ids,
            config=bert_config)
        pooled_output = bert_model.outputs[0]

    head_model = get_dragon_heads(binary_outcome)
    g, q0, q1 = head_model(pooled_output)

    # output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
    #     pooled_output)

    return tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=[g, q0, q1]), bert_model


def derpy_dragon_baseline(bert_config,
                          max_seq_length: int,
                          binary_outcome: bool):
    """A baseline for causal-BERT where we do not use finetuning

    Args:
      bert_config: BertConfig, the config defines the core BERT model.
      max_seq_length: integer, the maximum input sequence length.
      binary_outcome: bool, whether outcome is binary
      hub_module_url: (Experimental) TF-Hub path/url to Bert module.

    Returns:
      Combined prediction model (words, sample_weight, type) -> {g: float, Q0: float, Q1: float}
      BERT sub-model
    """
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    bert_model = modeling.get_bert_model(
        input_word_ids,
        input_mask,
        input_type_ids,
        config=bert_config)
    bert_model.trainable = False
    sequence_output = bert_model.outputs[1]
    # sum_sequence_output = tf.reduce_sum(sequence_output, axis=1)
    sum_sequence_output = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(sequence_output)
    sum_sequence_output = tf.keras.layers.Dense(500, activation='relu')(sum_sequence_output)
    sum_sequence_output = tf.keras.layers.Dense(200, activation='relu')(sum_sequence_output)

    head_model = get_dragon_heads(binary_outcome)
    g, q0, q1 = head_model(sum_sequence_output)

    # output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
    #     pooled_output)

    return tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=[g, q0, q1]), bert_model


def cross_ent(y_true, y_pred, sample_weight):
    y_true = tf.cast(y_true, tf.float32)
    example_error = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1. - y_pred)) * sample_weight
    return tf.reduce_sum(example_error)

# AGGIUNTO
'''def dragon_model(bert_config,
                 max_seq_length: int,
                 binary_outcome: bool):
    """BERT dragon model supervisionato (senza unsupervised loss)."""

    # Input classici di BERT
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    # BERT backbone
    bert_model = modeling.get_bert_model(
        input_word_ids,
        input_mask,
        input_type_ids,
        config=bert_config
    )

    pooled_output = bert_model.outputs[0]  # embedding [CLS]

    # Teste di Dragonnet (g, q0, q1)
    head_model = get_dragon_heads(binary_outcome)
    g, q0, q1 = head_model(pooled_output)

    # Modello finale
    dragon_model = tf.keras.Model(
        inputs=inputs,
        outputs=[g, q0, q1],
        name="dragon_model"
    )

    return dragon_model, bert_model'''

class MeanLossLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, x):
        return self.scale * tf.reduce_mean(x)
    
# AGGIUNTO
class UnsupervisedLossLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, y_pred):
        # aggiunge la loss senza richiedere y_true
        self.add_loss(self.scale * tf.reduce_mean(y_pred))
        return y_pred


def dragon_model(bert_config,
                 max_seq_length: int,
                 binary_outcome: bool,
                 use_unsup=False,
                 max_predictions_per_seq=20,
                 unsup_scale=1.):
    """BERT dragon model in functional API style.

    Args:
      bert_config: BertConfig, la configurazione di BERT
      max_seq_length: int, lunghezza massima delle sequenze
      binary_outcome: bool, True se outcome binario
      use_unsup: bool, se includere la loss non supervisionata stile pretraining
      max_predictions_per_seq: int, numero massimo di token mascherati
      unsup_scale: fattore di scala per la unsupervised loss

    Returns:
      dragon_model: modello Keras con output {'g','q0','q1'}
      bert_model: backbone BERT
    """

    if use_unsup:
        # Modello di pretraining (masked language modeling)
        pt_model, bert_model = pretrain_model(
            bert_config,
            max_seq_length,
            max_predictions_per_seq,
            initializer=None
        )
        inputs = pt_model.input

        # MODIFICATO
        # unsup_loss = unsup_scale * tf.reduce_mean(pt_model.output)

        # AGGIUNTO
        unsup_loss = MeanLossLayer(scale=unsup_scale, name="unsup_loss")(pt_model.output)


        # Ottieni pooled_output dal backbone BERT
        pooled_output, sequence_output = bert_model.outputs


    else:
        # Input classici di BERT
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        inputs = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        }

        # Backbone BERT
        bert_model = modeling.get_bert_model(
            input_word_ids,
            input_mask,
            input_type_ids,
            config=bert_config
        )

        pooled_output, sequence_output = bert_model.outputs

        # unsup_loss = lambda: 0  # placeholder

    # Teste finali di DragonNet (tutte su pooled_output → shape (batch,1))
    out_g = tf.keras.layers.Dense(1, activation='sigmoid', name='g')(pooled_output)
    if binary_outcome:
        activation = 'sigmoid'
    else:
        activation = None

    out_q0 = tf.keras.layers.Dense(200, activation='relu')(pooled_output)
    out_q0 = tf.keras.layers.Dense(200, activation='relu')(out_q0)
    out_q0 = tf.keras.layers.Dense(1, activation=activation, name='q0')(out_q0)

    out_q1 = tf.keras.layers.Dense(200, activation='relu')(pooled_output)
    out_q1 = tf.keras.layers.Dense(200, activation='relu')(out_q1)
    out_q1 = tf.keras.layers.Dense(1, activation=activation, name='q1')(out_q1)

    ''' print("pooled_output:", pooled_output.shape)
    print("out_g:", out_g.shape)
    print("out_q0:", out_q0.shape)
    print("out_q1:", out_q1.shape)'''
    # Modello finale
    '''ragon_model = tf.keras.Model(
        inputs=inputs,
        outputs={'g': out_g, 'q0': out_q0, 'q1': out_q1},
        name='dragon_model'
    )'''

    '''# MODIFICATO MASKING=FALSE
    dragon_model = tf.keras.Model(
        inputs=inputs,
        # outputs={'g': out_g, 'q0': out_q0, 'q1': out_q1, 'unsup': unsup_loss},
        outputs=[out_g, out_q0, out_q1],
        name='dragon_model'
    )'''

    # AGGIUNTO PER MASKING TRUE O FALSE
    if use_unsup:
        unsup_output = UnsupervisedLossLayer(scale=unsup_scale, name="unsup")(pt_model.output)

        dragon_model = tf.keras.Model(
            inputs=inputs,
            outputs={'g': out_g, 'q0': out_q0, 'q1': out_q1, 'unsup': unsup_output},
            name='dragon_model'
        )
    else:
        dragon_model = tf.keras.Model(
            inputs=inputs,
            outputs=[out_g, out_q0, out_q1],
            name='dragon_model'
        )



    # Aggiungi la unsupervised loss se attiva
    '''if use_unsup and unsup_loss is not None:
        dragon_model.add_loss(unsup_loss)'''



    return dragon_model, bert_model


def hydra_model(bert_config,
                max_seq_length: int,
                binary_outcome: bool,
                num_treatments: int,
                missing_outcomes: bool,
                use_unsup=False,
                max_predictions_per_seq=20,
                unsup_scale=1.):
    """BERT hydra model in functional API style.

    Args:
      bert_config: BertConfig, the config defines the core BERT model.
      max_seq_length: integer, the maximum input sequence length.
      binary_outcome: bool, whether outcome is binary
      num_treatments: int, number of treatment categories
      missing_outcomes: bool, whether missing outcomes are possible. In this case, we'll output prediction heads for
        P(T | missing = True, X), P(T | missing = False, X) and P(Missing | X) ('g0', 'g1', and 'missing')
      use_unsup: bool, whether to predict censored input words (requires same input features as bert pre-training)
      max_predictions_per_seq: integer, maximum number of input words that are censored
      unsup_scale: factor by which to scale unsupervised loss


    Returns:
      Combined prediction model (words, sample_weight, type) -> {g: float, Q0: float, Q1: float}
        if use_unsup=True this model has a loss term reflecting the unsupervised training objective
      BERT model
    """

    if use_unsup:
        pt_model, bert_model = pretrain_model(bert_config,
                                              max_seq_length,
                                              max_predictions_per_seq,
                                              initializer=None)

        inputs = pt_model.input
        unsup_loss = pt_model.outputs
        unsup_loss = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x), name="bert_pretrain_loss_and_metric_layer")(unsup_loss)
        # unsup_loss = unsup_scale * tf.reduce_mean(unsup_loss)

    else:
        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        inputs = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        }

        bert_model = modeling.get_bert_model(
            input_word_ids,
            input_mask,
            input_type_ids,
            config=bert_config)

        # unsup_loss = lambda: 0  # tf.convert_to_tensor(0.) doesn't work...
        unsup_loss = tf.keras.layers.Lambda(lambda x: 0.0, name="bert_pretrain_loss_and_metric_layer")(input_word_ids)


    pooled_output = bert_model.outputs[0]

    head_model = get_hydra_heads2(binary_outcome, num_treatments, missing_outcomes)
    outputs = head_model(pooled_output)  # note: number of outputs changes depending on missingness or not

    hydra_model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs)

    hydra_model.add_loss(unsup_loss)

    return hydra_model, bert_model


def classifier_model(bert_config,
                     float_type,
                     num_labels,
                     max_seq_length,
                     final_layer_initializer=None,
                     hub_module_url=None):
    """BERT classifier model in functional API style.

    Construct a Keras model for predicting `num_labels` outputs from an input with
    maximum sequence length `max_seq_length`.

    Args:
      bert_config: BertConfig, the config defines the core BERT model.
      float_type: dtype, tf.float32 or tf.bfloat16.
      num_labels: integer, the number of classes.
      max_seq_length: integer, the maximum input sequence length.
      final_layer_initializer: Initializer for final dense layer. Defaulted
        TruncatedNormal initializer.
      hub_module_url: (Experimental) TF-Hub path/url to Bert module.

    Returns:
      Combined prediction model (words, sample_weight, type) -> (one-hot labels)
      BERT sub-model (words, sample_weight, type) -> (bert_outputs)
    """
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

    if hub_module_url:
        bert_model = hub.KerasLayer(hub_module_url, trainable=True)
        pooled_output, _ = bert_model([input_word_ids, input_mask, input_type_ids])
    else:
        bert_model = modeling.get_bert_model(
            input_word_ids,
            input_mask,
            input_type_ids,
            config=bert_config,
            float_type=float_type)
        pooled_output = bert_model.outputs[0]

    if final_layer_initializer is not None:
        initializer = final_layer_initializer
    else:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range)

    output = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(
        pooled_output)
    output = tf.keras.layers.Dense(
        num_labels,
        kernel_initializer=initializer,
        name='output',
        dtype=float_type)(
        output)
    return tf.keras.Model(
        inputs={
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids
        },
        outputs=output), bert_model


if __name__ == '__main__':
    bert_config_file = "../pre-trained/uncased_L-12_H-768_A-12/bert_config.json"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    max_seq_length = 250
    binary_outcome = True
    hub_module_url = None

    derp = derpy_dragon_baseline(bert_config,
                                 max_seq_length,
                                 binary_outcome)


# AGGIUNTO

def no_dragon_model(input_dim=5000, binary_outcome=True):
    """
    Baseline 'no-dragon': semplice MLP senza BERT.
    Usa come input un vettore di dimensione input_dim (es. bag-of-words).
    """
    inputs = tf.keras.layers.Input(shape=(input_dim,), name="bow_input")

    # rete feedforward semplice
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # teste
    g = tf.keras.layers.Dense(1, activation="sigmoid", name="g")(x)
    if binary_outcome:
        q0 = tf.keras.layers.Dense(1, activation="sigmoid", name="q0")(x)
        q1 = tf.keras.layers.Dense(1, activation="sigmoid", name="q1")(x)
    else:
        q0 = tf.keras.layers.Dense(1, activation=None, name="q0")(x)
        q1 = tf.keras.layers.Dense(1, activation=None, name="q1")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[g, q0, q1], name="no_dragon_model")
    return model
