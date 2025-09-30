# run_causal_atm.py
# ==============================================================================
# Runner per Causal Amortized Topic Model (C-ATM)
# ==============================================================================

from absl import app, flags
import tensorflow as tf
import pandas as pd

from causal_atm.atm_models import causal_atm_model
from PeerRead.dataset.dataset import make_buzzy_based_simulated_labeler

# ---------------- FLAGS ---------------- #
flags.DEFINE_string("mode", "train_and_predict", "train_and_predict or predict_only")
flags.DEFINE_string("saved_path", None, "Path modello salvato (predict_only)")
flags.DEFINE_string("prediction_file", "../out/PeerRead/c_atm/beta1/predictions.tsv", "file TSV output predizioni")

# Dataset
flags.DEFINE_string("input_files", "../dat/PeerRead/proc/peerread_all_betas_bow.tfrecord", "TFRecord con bag-of-words")
flags.DEFINE_integer("train_batch_size", 32, "batch size train")
flags.DEFINE_integer("eval_batch_size", 32, "batch size eval")
flags.DEFINE_integer("num_epochs", 20, "numero di epoche")
flags.DEFINE_integer("vocab_size", 5000, "dimensione vocabolario")
flags.DEFINE_integer("num_topics", 50, "numero di topics")
flags.DEFINE_integer("hidden_dim", 200, "dimensione hidden encoder")

# Parametri per simulazioni
flags.DEFINE_string("treatment", "theorem_referenced", "attributo usato come trattamento (es. theorem_referenced, buzzy_title)")
flags.DEFINE_string("simulated", "real", "real | attribute | propensity")
flags.DEFINE_float("beta0", 0.25, "forza del trattamento simulato")
# non serve più, uso direttamente beta_filter per filtrare
# flags.DEFINE_float("beta1", None, "forza del confondimento simulato (1.0=low, 5.0=med, 25.0=high)")
flags.DEFINE_float("gamma", 1.0, "rumore sull’outcome simulato")
flags.DEFINE_string("simulation_mode", "simple", "modalità di simulazione: simple | multiplicative | interaction")
flags.DEFINE_string("base_propensities_path", "", "per simulazione basata su propensità")
flags.DEFINE_float("exogenous_confounding", 0.0, "quota di confondimento esogeno")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate per Adam")

flags.DEFINE_float("beta_filter", 1.0, "Se specificato, filtra i record con questo beta1")

FLAGS = flags.FLAGS


# ---------------- MODEL ---------------- #
def _get_causal_atm_model(vocab_size, num_topics, hidden_dim, binary_outcome=True):
    model = causal_atm_model(vocab_size=vocab_size,
                             num_topics=num_topics,
                             hidden_dim=hidden_dim,
                             binary_outcome=binary_outcome)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss={
            "g": "binary_crossentropy",
            "q0": "binary_crossentropy",
            "q1": "binary_crossentropy",
            "recon": "categorical_crossentropy",
        },
        loss_weights={"g": 1.0, "q0": 0.1, "q1": 0.1, "recon": 0.01},
        metrics={"g": ["accuracy"], "q0": ["mse"], "q1": ["mse"]},
    )
    return model


# ---------------- DATASET ---------------- #
def parse_bow_example(example_proto):
    feature_description = {
        "bow": tf.io.VarLenFeature(tf.float32),  # 👈 CAMBIA in float32
        "index": tf.io.FixedLenFeature([], tf.int64),
        "treatment": tf.io.FixedLenFeature([], tf.int64),
        "outcome": tf.io.FixedLenFeature([], tf.int64),
        "confounding_level": tf.io.FixedLenFeature([], tf.string),
        "beta1": tf.io.FixedLenFeature([], tf.float32),
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
    }
    return features


def make_dataset(is_training=True, return_extras=False):

    dataset = tf.data.TFRecordDataset(FLAGS.input_files)
    dataset = dataset.map(parse_bow_example, num_parallel_calls=tf.data.AUTOTUNE)

    if FLAGS.beta_filter is not None:
        dataset = dataset.filter(
            lambda ex: tf.equal(ex["beta1"], tf.cast(FLAGS.beta_filter, tf.float32))
        )
    
    def add_labels(example):
        # usa direttamente treatment/outcome dal TFRecord
        t = example["treatment"]
        y = example["outcome"]
        bow = example["bow"]

        labels_model = {
            "g": tf.expand_dims(t, -1),
            "q0": tf.expand_dims(y * (1 - t), -1),  # solo se t=0
            "q1": tf.expand_dims(y * t, -1),        # solo se t=1
            "recon": bow,
        }

        labels_extra = {
            "outcome": tf.expand_dims(y, -1),
            "treatment": tf.expand_dims(t, -1),
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

    model = _get_causal_atm_model(
        vocab_size=FLAGS.vocab_size,
        num_topics=FLAGS.num_topics,
        hidden_dim=FLAGS.hidden_dim,
        binary_outcome=True,
    )

    if FLAGS.mode == "train_and_predict":
        train_ds = make_dataset(is_training=True)
        model.fit(train_ds, epochs=FLAGS.num_epochs)
        model.save_weights("causal_atm.weights.h5")

    if FLAGS.mode in ["train_and_predict", "predict_only"]:
        if FLAGS.mode == "predict_only":
            model.load_weights(FLAGS.saved_path)

    eval_ds_full = make_dataset(is_training=False, return_extras=True)
    eval_ds = eval_ds_full.map(lambda f, l: (f, {k: l[k] for k in ["g", "q0", "q1", "recon"]}))
    outputs = model.predict(eval_ds)
    g_pred, q0_pred, q1_pred, _ = outputs

    true_outcomes, true_treatments = [], []
    for _, labels in eval_ds_full.unbatch():
        true_outcomes.append(int(labels["outcome"].numpy()[0]))
        true_treatments.append(int(labels["treatment"].numpy()[0]))

    df = pd.DataFrame({
        "outcome": true_outcomes,
        "treatment": true_treatments,
        "g": g_pred.reshape(-1),
        "q0": q0_pred.reshape(-1),
        "q1": q1_pred.reshape(-1),
    })
    df.to_csv(FLAGS.prediction_file, sep="\t", index=False)
    print(f"✅ Predictions salvate in {FLAGS.prediction_file}")


if __name__ == "__main__":
    app.run(main)