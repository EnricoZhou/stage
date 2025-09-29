# run_lda_supervised.py
# ==============================================================================
# Runner per LDA + Logistic Regression (Tabella 1: Supervision Helps)
# ==============================================================================

from absl import app, flags
import tensorflow as tf
import pandas as pd
import numpy as np
from gensim import corpora, models
from sklearn.linear_model import LogisticRegression

# ---------------- FLAGS ---------------- #
flags.DEFINE_string("input_files", "../dat/PeerRead/proc/peerread_all_betas_bow.tfrecord", "TFRecord input")
flags.DEFINE_string("prediction_file", "../out/PeerRead/lda/predictions.tsv", "File TSV output predizioni")
flags.DEFINE_integer("num_topics", 50, "Numero di topic LDA")
flags.DEFINE_float("beta_filter", 5.0, "Se specificato, filtra i record con questo beta1")

FLAGS = flags.FLAGS

# ---------------- PARSER ---------------- #
def parse_bow_example(example_proto):
    feature_description = {
        "bow": tf.io.VarLenFeature(tf.float32),
        "index": tf.io.FixedLenFeature([], tf.int64),
        "treatment": tf.io.FixedLenFeature([], tf.int64),
        "outcome": tf.io.FixedLenFeature([], tf.int64),
        "beta1": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    bow = tf.sparse.to_dense(parsed["bow"])
    return {
        "bow": bow,
        "index": parsed["index"],
        "treatment": parsed["treatment"],
        "outcome": parsed["outcome"],
        "beta1": parsed["beta1"],
    }

# ---------------- MAIN ---------------- #
def main(_):
    dataset = tf.data.TFRecordDataset(FLAGS.input_files)
    dataset = dataset.map(parse_bow_example, num_parallel_calls=tf.data.AUTOTUNE)

    if FLAGS.beta_filter is not None:
        dataset = dataset.filter(lambda ex: tf.equal(ex["beta1"], tf.cast(FLAGS.beta_filter, tf.float32)))

    bows, treatments, outcomes = [], [], []
    for ex in dataset.as_numpy_iterator():
        bows.append(ex["bow"])
        treatments.append(int(ex["treatment"]))
        outcomes.append(int(ex["outcome"]))

    # ---------------- LDA features ---------------- #
    bows = [list(np.where(bow > 0)[0]) for bow in bows]  # indici parole non zero
    dictionary = corpora.Dictionary([[str(tok) for tok in bow] for bow in bows])
    corpus = [dictionary.doc2bow([str(tok) for tok in bow]) for bow in bows]

    lda = models.LdaModel(corpus=corpus, id2word=dictionary,
                          num_topics=FLAGS.num_topics, passes=5, random_state=0)
    lda_features = [lda.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    X = np.array([[p for _, p in doc] for doc in lda_features])  # shape (n_docs, num_topics)

    y = np.array(outcomes)
    t = np.array(treatments)

    # ---------------- Train logistic models ---------------- #
    # g: propensity (predict treatment)
    g_clf = LogisticRegression(max_iter=1000).fit(X, t)
    g_pred = g_clf.predict_proba(X)[:, 1]

    # q0: outcome model for control
    q0_clf = LogisticRegression(max_iter=1000).fit(X[t == 0], y[t == 0])
    q0_pred = q0_clf.predict_proba(X)[:, 1]

    # q1: outcome model for treated
    q1_clf = LogisticRegression(max_iter=1000).fit(X[t == 1], y[t == 1])
    q1_pred = q1_clf.predict_proba(X)[:, 1]

    # ---------------- Save predictions ---------------- #
    df = pd.DataFrame({
        "outcome": y,
        "treatment": t,
        "g": g_pred,
        "q0": q0_pred,
        "q1": q1_pred,
    })

    df.to_csv(FLAGS.prediction_file, sep="\t", index=False)
    print(f"✅ Predictions salvate in {FLAGS.prediction_file}")

if __name__ == "__main__":
    app.run(main)
