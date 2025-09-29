# ==============================================================================
# Runner per BERT (full, con masked LM) baseline (Tabella 1: Language Modeling Helps)
# ==============================================================================

from absl import app, flags
import tensorflow as tf
import pandas as pd
from tf_official.nlp import bert_modeling as modeling
from tf_official.nlp.bert import tokenization
from causal_bert import bert_models
from causal_bert.data_utils import dataset_to_pandas_df, filter_training
from PeerRead.dataset.dataset import make_dataset_fn_from_file, make_real_labeler

# ---------------- FLAGS ---------------- #
flags.DEFINE_string("mode", "train_and_predict", "train_and_predict or predict_only")
flags.DEFINE_string("saved_path", None, "Path modello salvato (predict_only)")
flags.DEFINE_string("prediction_file", "../out/PeerRead/bert_full/predictions.tsv", "file TSV output predizioni")

flags.DEFINE_string("input_files", "../dat/PeerRead/proc/arxiv-all.tf_record", "TFRecord input")
flags.DEFINE_integer("train_batch_size", 16, "batch size train")
flags.DEFINE_integer("eval_batch_size", 16, "batch size eval")
flags.DEFINE_integer("num_train_epochs", 3, "numero di epoche")
flags.DEFINE_integer("max_seq_length", 250, "lunghezza massima sequenza BERT")
flags.DEFINE_float("learning_rate", 2e-5, "learning rate")
flags.DEFINE_string("vocab_file", "../pre-trained/uncased_L-12_H-768_A-12/vocab.txt", "BERT vocab file")
flags.DEFINE_string("bert_config_file", "../pre-trained/uncased_L-12_H-768_A-12/bert_config.json", "BERT config file")

FLAGS = flags.FLAGS


# ---------------- FORMATTER ---------------- #
def _keras_format(features, labels):
    """Allinea labels al formato del modello [g, q0, q1]."""
    y = labels["outcome"]
    t = tf.cast(labels["treatment"], tf.float32)

    y = tf.expand_dims(y, -1)
    t = tf.expand_dims(t, -1)

    # labels nello stesso ordine degli output
    labels = {
        "g": t,
        "q0": y,
        "q1": y,
        # dummy per unsup loss (masked LM)
        "unsup": tf.zeros((tf.shape(t)[0], FLAGS.max_seq_length, 30522), dtype=tf.float32)
    }

    sample_weights = {
        "g": tf.ones_like(t, dtype=tf.float32),
        "q0": 1 - t,
        "q1": t,
        "unsup": 1.0
    }
    return features, labels, sample_weights


# ---------------- DATASET ---------------- #
def make_dataset(is_training=True):
    labeler = make_real_labeler("theorem_referenced", "accepted")
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)

    train_input_fn = make_dataset_fn_from_file(
        input_files_or_glob=FLAGS.input_files,
        seq_length=FLAGS.max_seq_length,
        num_splits=10,
        dev_splits=[9],
        test_splits=[9],
        tokenizer=tokenizer,
        do_masking=True,   # 🔑 masked LM attivo
        is_training=is_training,
        shuffle_buffer_size=25000,
        seed=0,
        labeler=labeler
    )

    dataset = train_input_fn(params={"batch_size": FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size})

    if is_training:
        dataset = filter_training(dataset)
        dataset = dataset.map(_keras_format, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


# ---------------- MAIN ---------------- #
def main(_):
    tf.random.set_seed(42)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # modello DragonNet completo con unsupervised loss
    dragon_model, _ = bert_models.dragon_model(
        bert_config,
        max_seq_length=FLAGS.max_seq_length,
        binary_outcome=True,
        use_unsup=True,
        max_predictions_per_seq=20,
        unsup_scale=1.0
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    dragon_model.compile(
        optimizer=optimizer,
        loss={
            "g": "binary_crossentropy",
            "q0": "binary_crossentropy",
            "q1": "binary_crossentropy",
            "unsup": lambda y_true, y_pred: tf.reduce_mean(y_pred)  # dummy loss per masked LM
        },
        loss_weights={"g": 1.0, "q0": 0.1, "q1": 0.1, "unsup": 1.0},
        metrics={"g": ["accuracy"], "q0": ["accuracy"], "q1": ["accuracy"]}
    )

    if FLAGS.mode == "train_and_predict":
        train_ds = make_dataset(is_training=True)
        num_examples = 11778
        steps_per_epoch = num_examples // FLAGS.train_batch_size

        dragon_model.fit(
            train_ds,
            epochs=FLAGS.num_train_epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1
        )
        dragon_model.save_weights("bert_full_model.weights.h5")

    if FLAGS.mode == "predict_only":
        dragon_model.load_weights(FLAGS.saved_path or "bert_full_model.weights.h5")

    # eval
    eval_ds = make_dataset(is_training=False)
    outputs = dragon_model.predict(eval_ds)

    # outputs: [g, q0, q1, unsup]
    out_dict = {
        "g": outputs[0].reshape(-1),
        "q0": outputs[1].reshape(-1),
        "q1": outputs[2].reshape(-1),
    }
    predictions = pd.DataFrame(out_dict)

    label_dataset = eval_ds.map(lambda f, l: l)
    data_df = dataset_to_pandas_df(label_dataset)

    outs = data_df.join(predictions)
    outs.to_csv(FLAGS.prediction_file, sep="\t", index=False)
    print(f"✅ Predictions salvate in {FLAGS.prediction_file}")


if __name__ == "__main__":
    app.run(main)
