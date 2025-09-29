import os
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.special import expit  # sigmoid
from PeerRead.dataset.dataset import make_parser, compose, make_extra_feature_cleaning

# ==========================
# CONFIG
# ==========================
tf_record_file = "../dat/PeerRead/proc/arxiv-all.tf_record"
abs_len = 250
vocab_size = 5000
output_prefix = "../dat/PeerRead/proc/peerread_beta"

np.random.seed(0)  # fisso seed per stabilità

# ==========================
# STEP 1: leggi TFRecord
# ==========================
parser = make_parser(abs_len)
parser = compose(parser, make_extra_feature_cleaning())
raw_ds = tf.data.TFRecordDataset(tf_record_file).map(parser)

# ==========================
# STEP 2: costruisci vocabolario globale
# ==========================
token_counts = np.zeros(30522, dtype=np.int64)  # dimensione tipica vocab BERT
for ex in raw_ds:
    ids = ex["token_ids"].numpy()
    ids = ids[ids > 0]
    np.add.at(token_counts, ids, 1)

top_vocab = np.argsort(token_counts)[::-1][:vocab_size]
id2newid = {int(old): i for i, old in enumerate(top_vocab)}

print(f"[OK] Vocabolario costruito: {len(id2newid)} tokens")

def tokens_to_bow(ids):
    ids = [id2newid[x] for x in ids if x in id2newid]
    bow = np.bincount(ids, minlength=vocab_size).astype(np.float32)
    bow_sum = bow.sum()
    if bow_sum > 0:
        bow = bow / bow_sum
    return bow

# ==========================
# STEP 3: stima propensities P(T=1|z)
# ==========================
# reset raw_ds iterator
raw_ds = tf.data.TFRecordDataset(tf_record_file).map(parser)

records = []
for ex in raw_ds:
    idx = int(ex["index"].numpy())
    t = int(ex["theorem_referenced"].numpy())
    buzzy = int(ex["buzzy_title"].numpy())
    records.append((idx, t, buzzy, ex["token_ids"].numpy()))

df = pd.DataFrame(records, columns=["index", "treatment", "buzzy", "token_ids"])

# propensities empiriche
pi_buzzy = df.groupby("buzzy")["treatment"].mean().to_dict()
print(f"[OK] Propensities empiriche: {pi_buzzy}")

# ==========================
# STEP 4: funzione scrittura
# ==========================
def write_tfrecord(beta1_value):
    out_path = f"{output_prefix}{int(beta1_value)}_bow.tfrecord"
    writer = tf.io.TFRecordWriter(out_path)

    for _, row in df.iterrows():
        idx = int(row["index"])
        t = int(row["treatment"])
        buzzy = int(row["buzzy"])
        bow = tokens_to_bow(row["token_ids"])

        # simulazione outcome potenziali
        pi_z = pi_buzzy[buzzy]
        p0 = expit(beta1_value * (pi_z - 0.2))           # outcome se T=0
        p1 = expit(0.25 + beta1_value * (pi_z - 0.2))    # outcome se T=1
        y0 = np.random.binomial(1, p0)
        y1 = np.random.binomial(1, p1)

        # osservato dipende dal trattamento reale
        y = y1 if t == 1 else y0

        feature = {
            "bow": tf.train.Feature(float_list=tf.train.FloatList(value=bow)),
            "index": tf.train.Feature(int64_list=tf.train.Int64List(value=[idx])),
            "treatment": tf.train.Feature(int64_list=tf.train.Int64List(value=[t])),
            "outcome": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
            "y0": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y0)])),
            "y1": tf.train.Feature(float_list=tf.train.FloatList(value=[float(y1)])),
            "beta1": tf.train.Feature(float_list=tf.train.FloatList(value=[beta1_value])),
            "confounding_level": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(beta1_value).encode()])
            ),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    print(f"[OK] Salvato {out_path}")

# ==========================
# STEP 5: genera file
# ==========================
for b in [1.0, 5.0, 25.0]:
    write_tfrecord(b)