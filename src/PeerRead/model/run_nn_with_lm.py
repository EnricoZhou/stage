# run_nn_with_lm.py
# ==============================================================================
# Runner per NN con Language Modeling (baseline LM) - per Tabella 1
# ==============================================================================

from absl import app, flags
import tensorflow as tf
import pandas as pd

# ---------------- FLAGS ---------------- #
flags.DEFINE_string("mode", "train_and_predict", "train_and_predict or predict_only")
flags.DEFINE_string("saved_path", None, "Path modello salvato (predict_only)")
flags.DEFINE_string("prediction_file", "../out/PeerRead/nn_lm/predictions.tsv", "file TSV output predizioni")

flags.DEFINE_string("input_files", "../dat/PeerRead/proc/peerread_all_betas_bow.tfrecord", "TFRecord con bag-of-words")
flags.DEFINE_integer("train_batch_size", 32, "batch size train")
flags.DEFINE_integer("eval_batch_size", 32, "batch size eval")
flags.DEFINE_integer("num_epochs", 50, "numero di epoche")
flags.DEFINE_integer("vocab_size", 5000, "dimensione vocabolario")
flags.DEFINE_integer("hidden_dim", 200, "dimensione hidden encoder")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate per Adam")
flags.DEFINE_float("beta_filter", 5.0, "Se specificato, filtra i record con questo beta1")

FLAGS = flags.FLAGS


# ---------------- MODEL ---------------- #
def _get_nn_lm_model(vocab_size, hidden_dim, binary_outcome=True):
    inputs = tf.keras.layers.Input(shape=(vocab_size,), name="bow_input")

    # Encoder
    h = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    h = tf.keras.layers.Dense(hidden_dim, activation="relu")(h)

    # Outputs supervised
    g = tf.keras.layers.Dense(1, activation="sigmoid", name="g")(h)  # propensity
    q0 = tf.keras.layers.Dense(1, activation="sigmoid", name="q0")(h) if binary_outcome else tf.keras.layers.Dense(1)(h)
    q1 = tf.keras.layers.Dense(1, activation="sigmoid", name="q1")(h) if binary_outcome else tf.keras.layers.Dense(1)(h)

    # Decoder per ricostruzione bag-of-words
    recon = tf.keras.layers.Dense(vocab_size, activation="sigmoid", name="recon")(h)

    model = tf.keras.Model(inputs=inputs, outputs=[g, q0, q1, recon], name="nn_model_with_lm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss={
            "g": "binary_crossentropy",
            "q0": "binary_crossentropy",
            "q1": "binary_crossentropy",
            "recon": "binary_crossentropy",
        },
        loss_weights={
            "g": 1.0,
            "q0": 0.1,
            "q1": 0.1,
            "recon": 0.1,  # peso minore per la loss di LM (regolarizzante)
        },
        metrics={"g": ["accuracy"], "q0": ["mse"], "q1": ["mse"]},
    )
    return model


# ---------------- DATASET ---------------- #
def parse_bow_example(example_proto):
    feature_description = {
        "bow": tf.io.VarLenFeature(tf.float32),
        "index": tf.io.FixedLenFeature([], tf.int64),
        "treatment": tf.io.FixedLenFeature([], tf.int64),
        "outcome": tf.io.FixedLenFeature([], tf.int64),
        "confounding_level": tf.io.FixedLenFeature([], tf.string),
        "beta1": tf.io.FixedLenFeature([], tf.float32),
        "y0": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
        "y1": tf.io.FixedLenFeature([], tf.float32, default_value=-1.0),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    bow = tf.sparse.to_dense(parsed["bow"])

    features = {
        "bow": bow,
        "index": tf.cast(parsed["index"], tf.int32),
        "treatment": tf.cast(parsed["treatment"], tf.int32),
        "outcome": tf.cast(parsed["outcome"], tf.int32),
        "confounding_level": parsed["confounding_level"],
        "beta1": parsed["beta1"],
        "y0": parsed["y0"],
        "y1": parsed["y1"],
    }
    return features


def make_dataset(is_training=True, return_extras=False):
    dataset = tf.data.TFRecordDataset(FLAGS.input_files)
    dataset = dataset.map(parse_bow_example, num_parallel_calls=tf.data.AUTOTUNE)

    if FLAGS.beta_filter is not None:
        dataset = dataset.filter(lambda ex: tf.equal(ex["beta1"], tf.cast(FLAGS.beta_filter, tf.float32)))

    def add_labels(example):
        t = example["treatment"]
        y = example["outcome"]
        bow = example["bow"]

        # q0 riceve y solo se t=0, altrimenti dummy (0)
        q0_label = tf.where(tf.equal(t, 0), tf.expand_dims(y, -1), tf.zeros_like(tf.expand_dims(y, -1)))

        # q1 riceve y solo se t=1, altrimenti dummy (0)
        q1_label = tf.where(tf.equal(t, 1), tf.expand_dims(y, -1), tf.zeros_like(tf.expand_dims(y, -1)))

        labels_model = {
            "g": tf.expand_dims(t, -1),
            "q0": q0_label,
            "q1": q1_label,
            "recon": bow,  # supervisione non supervisionata: ricostruzione BoW
        }
        labels_extra = {
            "outcome": tf.expand_dims(y, -1),
            "treatment": tf.expand_dims(t, -1),
            "y0": tf.expand_dims(example["y0"], -1),
            "y1": tf.expand_dims(example["y1"], -1),
        }

        if return_extras:
            return bow, {**labels_model, **labels_extra}
        else:
            return bow, labels_model

    dataset = dataset.map(add_labels, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(5000)

    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------- MAIN ---------------- #
def main(_):
    tf.random.set_seed(0)

    model = _get_nn_lm_model(vocab_size=FLAGS.vocab_size, hidden_dim=FLAGS.hidden_dim, binary_outcome=True)

    if FLAGS.mode == "train_and_predict":
        train_ds = make_dataset(is_training=True)
        model.fit(train_ds, epochs=FLAGS.num_epochs, verbose=1)
        model.save_weights("nn_model_with_lm.weights.h5")

    if FLAGS.mode in ["train_and_predict", "predict_only"]:
        if FLAGS.mode == "predict_only":
            model.load_weights(FLAGS.saved_path)

    eval_ds_full = make_dataset(is_training=False, return_extras=True)
    eval_ds = eval_ds_full.map(lambda f, l: (f, {k: l[k] for k in ["g", "q0", "q1", "recon"]}))
    outputs = model.predict(eval_ds)
    g_pred, q0_pred, q1_pred, _ = outputs  # scarta ricostruzione

    true_outcomes, true_treatments, y0_list, y1_list = [], [], [], []
    for _, labels in eval_ds_full.unbatch():
        true_outcomes.append(int(labels["outcome"].numpy()[0]))
        true_treatments.append(int(labels["treatment"].numpy()[0]))
        y0_list.append(float(labels["y0"].numpy()[0]))
        y1_list.append(float(labels["y1"].numpy()[0]))

    df = pd.DataFrame({
        "outcome": true_outcomes,
        "treatment": true_treatments,
        "g": g_pred.reshape(-1),
        "q0": q0_pred.reshape(-1),
        "q1": q1_pred.reshape(-1),
    })
    if not all(v == -1.0 for v in y0_list):
        df["y0"] = y0_list
    if not all(v == -1.0 for v in y1_list):
        df["y1"] = y1_list

    df.to_csv(FLAGS.prediction_file, sep="\t", index=False)
    print(f"✅ Predictions salvate in {FLAGS.prediction_file}")


if __name__ == "__main__":
    app.run(main)
