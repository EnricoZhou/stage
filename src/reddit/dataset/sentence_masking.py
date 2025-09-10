# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(token_ids, masked_lm_prob, max_predictions_per_seq, vocab, seed):
    """Creates the predictions for the masked LM objective.

    This should be essentially equivalent to the bits that Bert loads from pre-processed tfrecords

    Except: we just include masks instead of randomly letting the words through or randomly replacing
    """
    """
    Crea gli input per il compito di Masked Language Modeling (MLM) di BERT.
    Prende in input una sequenza di ID tokenizzati e restituisce:
      - output_ids: la sequenza con alcuni token mascherati con [MASK]
      - masked_lm_positions: posizioni dei token mascherati
      - masked_lm_ids: i token originali nascosti (ground truth)
      - masked_lm_weights: maschera binaria che indica quali posizioni contano
    """

    # Crea una maschera casuale: per ogni token genera un numero tra 0 e 1,
    # e lo confronta con la probabilità di mascheramento (es. 0.15 = 15%).
    basic_mask = tf.less(
        tf.random.uniform(token_ids.shape, minval=0, maxval=1, dtype=tf.float32, seed=seed),
        masked_lm_prob)

    # Definisce quali token possono essere candidati al mascheramento:
    # - non devono essere [CLS]
    # - non devono essere [SEP]
    cand_indexes = tf.logical_and(tf.not_equal(token_ids, vocab["[CLS]"]),
                                  tf.not_equal(token_ids, vocab["[SEP]"]))
    # Inoltre, non devono essere padding (0)
    cand_indexes = tf.logical_and(cand_indexes, tf.not_equal(token_ids, 0))

    # Combina la maschera casuale con i candidati: otteniamo i token effettivamente mascherati
    mask = tf.logical_and(cand_indexes, basic_mask)

    # Caso limite: se nessun token viene mascherato, allora obbliga a mascherare almeno il primo valido.
    masked_lm_positions = tf.cond(pred=tf.reduce_any(mask),
                                  true_fn=lambda: tf.where(mask),
                                  false_fn=lambda: tf.where(cand_indexes)[0:2])

    # Converti le posizioni da lista di [i,0] a lista semplice [i]
    masked_lm_positions = masked_lm_positions[:, 0]

    # Mischia l’ordine delle posizioni mascherate (per non avere sempre le stesse prime)
    masked_lm_positions = tf.random.shuffle(masked_lm_positions, seed=seed)

    # Limita il numero di token mascherati a max_predictions_per_seq
    masked_lm_positions = masked_lm_positions[0:max_predictions_per_seq]

    # Cast delle posizioni a int32 (compatibile con TensorFlow su TPU/GPU)
    masked_lm_positions = tf.cast(masked_lm_positions, dtype=tf.int32)

    # Ottiene i token originali corrispondenti alle posizioni mascherate (ground truth per la loss)
    masked_lm_ids = tf.gather(token_ids, masked_lm_positions)

    # Costruisce una maschera booleana finale con 1 nei punti da mascherare
    mask = tf.cast(
        tf.scatter_nd(tf.expand_dims(masked_lm_positions, 1), tf.ones_like(masked_lm_positions), token_ids.shape),
        bool)

    # Crea la sequenza finale sostituendo i token mascherati con l’ID di [MASK]
    output_ids = tf.where(mask, vocab["[MASK]"] * tf.ones_like(token_ids), token_ids)

    # Inizializza un array di pesi = 1 (serve per calcolare la loss solo sulle posizioni mascherate)
    masked_lm_weights = tf.ones_like(masked_lm_ids, dtype=tf.float32)

    # Calcola di quanto bisogna "paddare" per arrivare a max_predictions_per_seq
    add_pad = [[0, max_predictions_per_seq - tf.shape(input=masked_lm_positions)[0]]]

    # Padding di pesi, posizioni e token mascherati, così hanno lunghezza fissa
    masked_lm_weights = tf.pad(tensor=masked_lm_weights, paddings=add_pad, mode='constant')
    masked_lm_positions = tf.pad(tensor=masked_lm_positions, paddings=add_pad, mode='constant')
    masked_lm_ids = tf.pad(tensor=masked_lm_ids, paddings=add_pad, mode='constant')

    # Restituisce:
    # - la sequenza con [MASK] inseriti
    # - le posizioni dei token mascherati
    # - i token originali nascosti
    # - i pesi (1=valido, 0=padding)
    return output_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


def main(_):
    pass


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
