# ==============================================================================
# Runner per ATM (Amortized Topic Model, unsupervised)
# Estrae topic proportions θ, poi usa regressori supervisionati per g, q0, q1
# ==============================================================================

from absl import app, flags
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

FLAGS = flags.FLAGS

# ---------------- FLAGS ---------------- #
flags.DEFINE_string("mode", "train_and_predict", "train_and_predict or predict_only")
flags.DEFINE_string("saved_path", None, "Path modello salvato (predict_only)")
flags.DEFINE_string("input_files", "../dat/PeerRead/proc/peerread_all_betas_bow.tfrecord", "TFRecord input")
flags.DEFINE_string("prediction_file", "../out/PeerRead/atm/predictions.tsv", "File TSV output")
flags.DEFINE_integer("train_batch_size", 32, "batch size train")
flags.DEFINE_integer("eval_batch_size", 32, "batch size eval")
flags.DEFINE_integer("num_epochs", 50, "Numero di epoche")
flags.DEFINE_integer("vocab_size", 5000, "Dimensione BoW")
flags.DEFINE_integer("hidden_dim", 100, "Dimensione hidden/embedding")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("beta_filter", None, "Se specificato, filtra i record con questo beta1")
flags.DEFINE_integer("num_topics", 50, "Numero di topic")

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
    return {
        "bow": bow,
        "treatment": tf.cast(parsed["treatment"], tf.int32),
        "outcome": tf.cast(parsed["outcome"], tf.int32),
        "beta1": parsed["beta1"],
        "y0": parsed["y0"],
        "y1": parsed["y1"],
    }

def make_dataset(is_training=True):
    dataset = tf.data.TFRecordDataset(FLAGS.input_files)
    dataset = dataset.map(parse_bow_example, num_parallel_calls=tf.data.AUTOTUNE)

    if FLAGS.beta_filter is not None:
        dataset = dataset.filter(lambda ex: tf.equal(ex["beta1"], tf.cast(FLAGS.beta_filter, tf.float32)))

    dataset = dataset.map(lambda ex: (ex["bow"], (ex["treatment"], ex["outcome"], ex["y0"], ex["y1"])))
    if is_training:
        dataset = dataset.shuffle(5000)

    batch_size = FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ---------------- MODEL ---------------- #
class ATM(tf.keras.Model):
    def __init__(self, vocab_size, hidden_dim, num_topics):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(vocab_size,)),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(num_topics)  # logit dei topic
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(num_topics,)),
            tf.keras.layers.Dense(vocab_size, activation="softmax")
        ])

    def call(self, x):
        z_logits = self.encoder(x)
        theta = tf.nn.softmax(z_logits)  # distribuzione sui topic
        bow_recon = self.decoder(theta)
        return bow_recon, theta

def reconstruction_loss(x, x_recon):
    return tf.reduce_mean(tf.reduce_sum(-x * tf.math.log(x_recon + 1e-10), axis=1))

# ---------------- MAIN ---------------- #
def main(_):
    train_ds = make_dataset(is_training=True)
    eval_ds = make_dataset(is_training=False)

    model = ATM(vocab_size=FLAGS.vocab_size, hidden_dim=FLAGS.hidden_dim, num_topics=FLAGS.num_topics)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    # training loop
    for epoch in range(FLAGS.num_epochs):
        for x, _ in train_ds:
            with tf.GradientTape() as tape:
                x_recon, _ = model(x)
                loss = reconstruction_loss(x, x_recon)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Epoch {epoch+1}, loss={loss.numpy():.4f}")

    # eval → estrai θ
    all_theta, all_treat, all_out, all_y0, all_y1 = [], [], [], [], []
    for x, (t, y, y0, y1) in eval_ds:
        _, theta = model(x)
        all_theta.extend(theta.numpy())
        all_treat.extend(t.numpy())
        all_out.extend(y.numpy())
        all_y0.extend(y0.numpy())
        all_y1.extend(y1.numpy())

    all_theta = np.array(all_theta)
    all_out = np.array(all_out)
    all_treat = np.array(all_treat)

    # ---------------- SUPERVISED REGRESSORS ---------------- #
    # g: propensity (Logistic regression)
    g_model = LogisticRegression(max_iter=1000)
    g_model.fit(all_theta, all_treat)
    g_pred = g_model.predict_proba(all_theta)[:, 1]

    # q0: outcome model for control
    q0_model = LinearRegression()
    q0_model.fit(all_theta[all_treat == 0], all_out[all_treat == 0])
    q0_pred = q0_model.predict(all_theta)

    # q1: outcome model for treated
    q1_model = LinearRegression()
    q1_model.fit(all_theta[all_treat == 1], all_out[all_treat == 1])
    q1_pred = q1_model.predict(all_theta)

    # ---------------- SAVE PREDICTIONS ---------------- #
    df = pd.DataFrame({
        "outcome": all_out,
        "treatment": all_treat,
        "g": g_pred,
        "q0": q0_pred,
        "q1": q1_pred,
        "y0": all_y0,
        "y1": all_y1,
    })
    df.to_csv(FLAGS.prediction_file, sep="\t", index=False)
    print(f"✅ Predictions salvate in {FLAGS.prediction_file}")


if __name__ == "__main__":
    app.run(main)
