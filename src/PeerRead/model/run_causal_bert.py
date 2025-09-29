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
"""BERT classification finetuning runner in tf2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import pandas as pd

from absl import app
from absl import flags
import tensorflow as tf

# AGGIUNTO
'''import multiprocessing
num_cpu = multiprocessing.cpu_count()
print("Using", num_cpu, "CPU threads")
tf.config.threading.set_intra_op_parallelism_threads(num_cpu)
tf.config.threading.set_inter_op_parallelism_threads(num_cpu)'''


from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert import tokenization, common_flags
from tf_official.utils.misc import tpu_lib
from causal_bert import bert_models
from causal_bert.data_utils import dataset_to_pandas_df, filter_training

from PeerRead.dataset.dataset import make_dataset_fn_from_file, make_real_labeler, make_buzzy_based_simulated_labeler, \
    make_propensity_based_simulated_labeler

common_flags.define_common_bert_flags()

flags.DEFINE_enum(
    'mode', 'train_and_predict', ['train_and_predict', 'predict_only'],
    'One of {"train_and_predict", "predict_only"}. `train_and_predict`: '
    'trains the model and make predictions. '
    '`predict_only`: loads a trained model and makes predictions.')

flags.DEFINE_string('saved_path', None,
                    'Relevant only if mode is predict_only. Path to pre-trained model')

flags.DEFINE_bool(
    "do_masking", True,
    "Whether to randomly mask input words during training (serves as a sort of regularization)")

flags.DEFINE_float("treatment_loss_weight", 1.0, "how to weight the treatment prediction term in the loss")


flags.DEFINE_bool(
    "fixed_feature_baseline", False,
    "Whether to use BERT to produced fixed features (no finetuning)")


flags.DEFINE_string('input_files', None,
                    'File path to retrieve training data for pre-training.')

# Model training specific flags.
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer(
    'max_seq_length', 250,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')

flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')
flags.DEFINE_string(
    'hub_module_url', None, 'TF-Hub path/url to Bert module. '
                            'If specified, init_checkpoint flag should not be used.')

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("seed", 0, "Seed for rng.")

# Data splitting details
flags.DEFINE_integer("num_splits", 10,
                     "number of splits")
flags.DEFINE_string("dev_splits", '9', "indices of development splits")
flags.DEFINE_string("test_splits", '9', "indices of test splits")

# Flags specifically related to PeerRead experiment

flags.DEFINE_string(
    "treatment", "theorem_referenced",
    "Covariate used as treatment."
)

flags.DEFINE_string("simulated", 'real', "whether to use real data ('real'), attribute based ('attribute'), "
                                         "or propensity score-based ('propensity') simulation"),
flags.DEFINE_float("beta0", 0.25, "param passed to simulated labeler, treatment strength")
flags.DEFINE_float("beta1", 0.0, "param passed to simulated labeler, confounding strength")
flags.DEFINE_float("gamma", 0.0, "param passed to simulated labeler, noise level")
flags.DEFINE_float("exogenous_confounding", 0.0, "amount of exogenous confounding in propensity based simulation")
flags.DEFINE_string("base_propensities_path", '', "path to .tsv file containing a 'propensity score' for each unit,"
                                                  "used for propensity score-based simulation")

flags.DEFINE_string("simulation_mode", 'simple', "simple, multiplicative, or interaction")

flags.DEFINE_string("prediction_file", "../output/predictions.tsv", "path where predictions (tsv) will be written")

# AGGIUNTO
flags.DEFINE_bool("no_dragon", False, "Use simple MLP baseline without BERT")

FLAGS = flags.FLAGS

'''# AGGIUNTO
def _keras_format(features, labels):
    y = labels['outcome']
    t = tf.cast(labels['treatment'], tf.float32)

    # match shape (batch_size, 1)
    y = tf.expand_dims(y, -1)
    t = tf.expand_dims(t, -1)

    # labels come tuple nello stesso ordine degli output [g, q0, q1]
    labels = (t, y, y)

    # sample_weights come tuple nello stesso ordine
    sample_weights = (
        tf.ones_like(t, dtype=tf.float32),  # per g
        1 - t,                             # per q0
        t                                  # per q1
    )

    return features, labels, sample_weights'''

def _keras_format(features, labels):
    y = labels['outcome']
    t = tf.cast(labels['treatment'], tf.float32)

    y = tf.expand_dims(y, -1)
    t = tf.expand_dims(t, -1)

    if FLAGS.do_masking:
        # Usa le features per costruire un dummy della forma giusta
        # input_word_ids ha shape (batch, seq_len)
        seq_len = tf.shape(features['input_word_ids'])[1]
        vocab_size = 30522  # default BERT vocab size, puoi parametrizzarlo se diverso

        dummy_unsup = tf.zeros((tf.shape(t)[0], seq_len, vocab_size), dtype=tf.float32)

        labels = {
            'g': t,
            'q0': y,
            'q1': y,
            'unsup': dummy_unsup
        }
        sample_weights = {
            'g': tf.ones_like(t, dtype=tf.float32),
            'q0': 1 - t,
            'q1': t,
            'unsup': 1.0
        }
    else:
        labels = (t, y, y)
        sample_weights = (tf.ones_like(t, dtype=tf.float32), 1 - t, t)

    return features, labels, sample_weights



# AGGIUNTO
"""def _keras_format(features, labels):
    y = labels['outcome']
    t = tf.cast(labels['treatment'], tf.float32)

    # labels come lista, stesso ordine degli outputs [g, q0, q1]
    labels = [labels['treatment'], y, y]

    # sample_weights come lista nello stesso ordine
    sample_weights = [
        tf.ones_like(t, dtype=tf.float32),  # peso costante per g
        1 - t,                             # per q0
        t                                  # per q1
    ]

    return features, labels, sample_weights"""

# MODIFICATO
"""def _keras_format(features, labels):
    # features, labels = sample
    y = labels['outcome']
    t = tf.cast(labels['treatment'], tf.float32)
    labels = {'g': labels['treatment'], 'q0': y, 'q1': y}
    # MODIFICATO: 'g': tf.ones_like(t, dtype=tf.float32),  # peso costante per g

    sample_weights = {
        'g': tf.ones_like(t, dtype=tf.float32),  # peso costante per g
        'q0': 1 - t, 
        'q1': t
    }
    return features, labels, sample_weights
"""

def make_dataset(is_training: bool, do_masking=False):
    if FLAGS.simulated == 'real':
        labeler = make_real_labeler(FLAGS.treatment, 'accepted')

    elif FLAGS.simulated == 'attribute':
        labeler = make_buzzy_based_simulated_labeler(FLAGS.beta0, FLAGS.beta1, FLAGS.gamma, FLAGS.simulation_mode,
                                                     seed=0)
    elif FLAGS.simulated == 'propensity':
        model_predictions = pd.read_csv(FLAGS.base_propensities_path, '\t')

        base_propensity_scores = model_predictions['g']
        example_indices = model_predictions['index']

        labeler = make_propensity_based_simulated_labeler(treat_strength=FLAGS.beta0,
                                                          con_strength=FLAGS.beta1,
                                                          noise_level=FLAGS.gamma,
                                                          base_propensity_scores=base_propensity_scores,
                                                          example_indices=example_indices,
                                                          exogeneous_con=FLAGS.exogenous_confounding,
                                                          setting=FLAGS.simulation_mode,
                                                          seed=0)

    else:
        Exception("simulated flag not recognized")

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    dev_splits = [int(s) for s in str.split(FLAGS.dev_splits)]
    test_splits = [int(s) for s in str.split(FLAGS.test_splits)]

    train_input_fn = make_dataset_fn_from_file(
        input_files_or_glob=FLAGS.input_files,
        seq_length=FLAGS.max_seq_length,
        num_splits=FLAGS.num_splits,
        dev_splits=dev_splits,
        test_splits=test_splits,
        tokenizer=tokenizer,
        do_masking=do_masking,
        is_training=is_training,
        shuffle_buffer_size=25000,  # note: bert hardcoded this, and I'm following suit
        seed=FLAGS.seed,
        labeler=labeler)
    
    # AGGIUNTO: stampa quanti esempi ci sono in ogni split per capire se davvero i folds vengono vegerati corretamente
    '''count = 0
    for _ in train_input_fn(params={'batch_size': 1}):
        count += 1
    print("Total examples in this split:", count)'''


    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size

    dataset = train_input_fn(params={'batch_size': batch_size})

    # format expected by Keras for training
    if is_training:
        dataset = filter_training(dataset)
        dataset = dataset.map(_keras_format, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # AGGIUNTO: così i dati restano in RAM e non rallentano la GPU.
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


    return dataset


def make_dragonnet_metrics():
    METRICS = [
        tf.keras.metrics.BinaryAccuracy,
        tf.keras.metrics.Precision,
        tf.keras.metrics.Recall,
        tf.keras.metrics.AUC
    ]

    NAMES = ['binary_accuracy', 'precision', 'recall', 'auc']

    g_metrics = [m(name='metrics/' + n) for m, n in zip(METRICS, NAMES)]
    q0_metrics = [m(name='metrics/' + n) for m, n in zip(METRICS, NAMES)]
    q1_metrics = [m(name='metrics/' + n) for m, n in zip(METRICS, NAMES)]

    return {'g': g_metrics, 'q0': q0_metrics, 'q1': q1_metrics}


def main(_):
    # Users should always run this script under TF 2.x
    assert tf.version.VERSION.startswith('2.1')
    tf.random.set_seed(FLAGS.seed)

    # AGGIUNTO: forzo l'uso della GPU se disponibile
    print("Dispositivi disponibili:", tf.config.list_physical_devices())

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limita TensorFlow a usare SOLO la prima GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("✅ Uso GPU:", gpus[0])
        except RuntimeError as e:
            print("Errore nella configurazione GPU:", e)
    else:
        print("⚠️ Nessuna GPU trovata, userò solo CPU")


    # AGGIUNTO per verificare se utilizza la GPU
    '''print("Dispositivi fisici disponibili:", tf.config.list_physical_devices())
    print("Dispositivi logici disponibili:", tf.config.list_logical_devices())

    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU rilevata: le operazioni useranno la GPU tramite Metal")
    else:
        print("⚠️ Nessuna GPU rilevata: si userà solo la CPU")'''


    # with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    #     input_meta_data = json.loads(reader.read().decode('utf-8'))

    if not FLAGS.model_dir:
        FLAGS.model_dir = '/tmp/bert20/'
    #
    # Configuration stuff
    #
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = FLAGS.num_train_epochs
    train_data_size = 11778
    # train_data_size = 100
    steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)  # 368
    warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
    initial_lr = FLAGS.learning_rate

    strategy = None
    if FLAGS.strategy_type == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    elif FLAGS.strategy_type == 'tpu':
        cluster_resolver = tpu_lib.tpu_initialize(FLAGS.tpu)
        strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
    else:
        raise ValueError('The distribution strategy type is not supported: %s' %
                         FLAGS.strategy_type)

    #
    # Modeling and training
    #

    # the model
    def _get_dragon_model(do_masking):
        # AGGIUNTO
        if FLAGS.no_dragon:
            dragon_model = bert_models.no_dragon_model(input_dim=5000, binary_outcome=True)
            core_model = None
        
            
        if not FLAGS.fixed_feature_baseline:
                dragon_model, core_model = (
                bert_models.dragon_model(
                    bert_config,
                    max_seq_length=FLAGS.max_seq_length,
                    binary_outcome=True,
                    use_unsup=do_masking,
                    max_predictions_per_seq=20,
                    unsup_scale=1.))
        else:
            dragon_model, core_model = bert_models.derpy_dragon_baseline(
                bert_config,
                max_seq_length=FLAGS.max_seq_length,
                binary_outcome=True)

        # WARNING: the original optimizer causes a bug where loss increases after first epoch
        # dragon_model.optimizer = optimization.create_optimizer(
        #     FLAGS.train_batch_size * initial_lr, steps_per_epoch * epochs, warmup_steps)
        dragon_model.optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.train_batch_size * initial_lr)
        return dragon_model, core_model

    if FLAGS.mode == 'train_and_predict':
        # training. strategy.scope context allows use of multiple devices
        with strategy.scope():
            keras_train_data = make_dataset(is_training=True, do_masking=FLAGS.do_masking)

            dragon_model, core_model = _get_dragon_model(FLAGS.do_masking)
            optimizer = dragon_model.optimizer

            if FLAGS.init_checkpoint:
                checkpoint = tf.train.Checkpoint(model=core_model)
                # AGGIUNTO: Ignora i pesi mancanti
                checkpoint.restore(FLAGS.init_checkpoint).expect_partial()
                # MODIFICATO: Per evitare errori sui pesi mancanti
                # checkpoint.restore(FLAGS.init_checkpoint).assert_existing_objects_matched()

            latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
            if latest_checkpoint:
                dragon_model.load_weights(latest_checkpoint)

            # AGGIUNTO
            if FLAGS.do_masking:
            # versione con masked language modeling (unsup loss attiva)
                dragon_model.compile(
                    optimizer=optimizer,
                    loss={
                        'g': 'binary_crossentropy',
                        'q0': 'binary_crossentropy',
                        'q1': 'binary_crossentropy',
                        # loss identity per la parte unsupervised (masked LM)
                        'unsup': lambda y_true, y_pred: tf.reduce_mean(y_pred)
                    },
                    loss_weights={
                        'g': FLAGS.treatment_loss_weight,
                        'q0': 0.1,
                        'q1': 0.1,
                        'unsup': 1.0   # peso della loss unsup (puoi regolarlo)
                    },
                    metrics={
                        'g': ['accuracy'],
                        'q0': ['accuracy'],
                        'q1': ['accuracy']
                    }
                )
            else:
                # versione standard senza masking
                dragon_model.compile(
                    optimizer=optimizer,
                    loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                    loss_weights=[FLAGS.treatment_loss_weight, 0.1, 0.1],
                    metrics=['accuracy', 'accuracy', 'accuracy']
                )

            '''dragon_model.compile(
                optimizer=optimizer,
                loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                loss_weights=[FLAGS.treatment_loss_weight, 0.1, 0.1],
                metrics=['accuracy', 'accuracy', 'accuracy']
            )'''

            # AGGIUNTO
            '''dragon_model.compile(
                optimizer=optimizer,
                loss={
                    'g': 'binary_crossentropy',
                    'q0': 'binary_crossentropy',
                    'q1': 'binary_crossentropy',
                    'unsup': lambda y_true, y_pred: tf.reduce_mean(y_pred)  # loss identity
                },
                loss_weights={
                    'g': FLAGS.treatment_loss_weight,
                    'q0': 0.1,
                    'q1': 0.1,
                    'unsup': unsup_scale
                },
                metrics={
                    'g': ['accuracy'],
                    'q0': ['accuracy'],
                    'q1': ['accuracy']
                }
            )'''

            # MODIFICATO
            '''dragon_model.compile(optimizer=optimizer,
                                 loss={'g': 'binary_crossentropy', 'q0': 'binary_crossentropy',
                                       'q1': 'binary_crossentropy'},
                                 loss_weights={'g': FLAGS.treatment_loss_weight, 'q0': 0.1, 'q1': 0.1},
                                 weighted_metrics=make_dragonnet_metrics())'''

            summary_callback = tf.keras.callbacks.TensorBoard(FLAGS.model_dir, update_freq=128)
            # MODIFICATO: aggiunto .weights.h5
            checkpoint_dir = os.path.join(FLAGS.model_dir, 'model_checkpoint.{epoch:02d}.weights.h5')
            # MODIFICATO da period=10 -> save_freq='epoch'
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, save_freq='epoch')

            callbacks = [summary_callback, checkpoint_callback]

            # forza l'uso della GPU
            with tf.device('/GPU:0'):
                dragon_model.fit(
                    x=keras_train_data,
                    # validation_data=evaluation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    # vailidation_steps=eval_steps,
                    callbacks=callbacks)
                
        # save a final model checkpoint (so we can restore weights into model w/o training idiosyncracies)
        if FLAGS.model_export_path:
            model_export_path = FLAGS.model_export_path
        else:
            model_export_path = os.path.join(FLAGS.model_dir, 'trained/dragon.ckpt')

        checkpoint = tf.train.Checkpoint(model=dragon_model)
        saved_path = checkpoint.save(model_export_path)
    else:
        saved_path = FLAGS.saved_path

    # make predictions and write to file

    # create data and model w/o masking
    eval_data = make_dataset(is_training=False, do_masking=False)
    dragon_model, core_model = _get_dragon_model(do_masking=False)
    # reload the model weights (necessary because we've obliterated the masking)
    checkpoint = tf.train.Checkpoint(model=dragon_model)
    # AGGIUNTO
    checkpoint.restore(FLAGS.init_checkpoint).expect_partial()
    # MODIFICATO: Per evitare errori sui pesi mancanti
    # checkpoint.restore(saved_path).assert_existing_objects_matched()
    # loss added as simple hack to bizzarre keras bug that requires compile for predict, and a loss for compile
    
    # MODIFICATO: per evitare warning
    # dragon_model.add_loss(tf.constant(0.0))
    # loss added as simple hack to bizzarre keras bug that requires compile for predict, and a loss for compile
    dragon_model.compile()

    outputs = dragon_model.predict(x=eval_data)

    # AGGIUNTO
    # outputs è una lista: [g, q0, q1]
    out_dict = {
        'g':  outputs[0].reshape(-1),
        'q0': outputs[1].reshape(-1),
        'q1': outputs[2].reshape(-1),
    }

    # Ora puoi creare il DataFrame
    predictions = pd.DataFrame(out_dict)

    # AGGIUNTO
    ''' out_dict = {}
    out_dict['g']  = outputs['g']
    out_dict['q0'] = outputs['q0']
    out_dict['q1'] = outputs['q1']'''

    # MODIFICATO: da lista di array a dizionario di array
    """out_dict = {}
    out_dict['g'] = outputs[0].squeeze()
    out_dict['q0'] = outputs[1].squeeze()
    out_dict['q1'] = outputs[2].squeeze()

    predictions = pd.DataFrame(out_dict)"""

    label_dataset = eval_data.map(lambda f, l: l)
    data_df = dataset_to_pandas_df(label_dataset)

    outs = data_df.join(predictions)
    with tf.io.gfile.GFile(FLAGS.prediction_file, "w") as writer:
        writer.write(outs.to_csv(sep="\t"))


if __name__ == '__main__':
    # flags.mark_flag_as_required('bert_config_file')
    # flags.mark_flag_as_required('input_meta_data_path')
    # flags.mark_flag_as_required('model_dir')
    app.run(main)