import tensorflow as tf
import pandas as pd
# from dataset import make_parser, compose, make_extra_feature_cleaning, make_buzzy_based_simulated_labeler
from PeerRead.dataset.dataset import (
    make_parser,
    compose,
    make_extra_feature_cleaning,
    make_buzzy_based_simulated_labeler,
)


# === Config ===
tf_record_file = "../dat/PeerRead/proc/arxiv-all.tf_record"
abs_len = 250

# parser (legge il tf_record grezzo + aggiunge campi theorem_referenced, buzzy_title ecc.)
parser = make_parser(abs_len)
parser = compose(parser, make_extra_feature_cleaning())

def make_dataset(beta0, beta1, gamma, out_path):
    print(f"[INFO] Creating simulated dataset with beta1={beta1}")

    # labeler simulato
    labeler = make_buzzy_based_simulated_labeler(
        treat_strength=beta0,
        con_strength=beta1,
        noise_level=gamma,
        setting="simple",
        seed=0
    )

    raw_ds = tf.data.TFRecordDataset(tf_record_file)
    ds = raw_ds.map(parser).map(labeler)

    rows = []
    for ex in ds:
        rows.append({
            "index": int(ex["index"].numpy()),
            "outcome": int(ex["outcome"].numpy()),
            "treatment": int(ex["treatment"].numpy()),
            "y0": float(ex["y0"].numpy()),
            "y1": float(ex["y1"].numpy()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved {len(df)} rows to {out_path}")

# === Genera Low / Med / High confounding ===
make_dataset(beta0=0.25, beta1=1.0, gamma=0.0, out_path="../dat/PeerRead/proc/peerread_beta1.tsv")
make_dataset(beta0=0.25, beta1=5.0, gamma=0.0, out_path="../dat/PeerRead/proc/peerread_beta5.tsv")
make_dataset(beta0=0.25, beta1=25.0, gamma=0.0, out_path="../dat/PeerRead/proc/peerread_beta25.tsv")