# compute_estimates_nn.py
# ==============================================================================
# Compute ATT estimates per Neural Network baseline (Tabella 3)
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from semi_parametric_estimation.att import att_estimates

pd.set_option('display.max_colwidth', None)


def att_from_nn_tsv(tsv_path, trim=0.0):
    """
    Carica predizioni da NN (g, q0, q1) e calcola stimatori ATT.
    """
    df = pd.read_csv(tsv_path, sep="\t")

    # Colonne attese: outcome, treatment, g, q0, q1
    y = df["outcome"].values.astype(float)
    t = df["treatment"].values.astype(float)
    g = df["g"].values.astype(float)
    q0 = df["q0"].values.astype(float)
    q1 = df["q1"].values.astype(float)

    # Stima ground truth: differenza media tra y1 e y0 tra i trattati (se disponibile)
    if "y1" in df.columns and "y0" in df.columns:
        gt = df[df.treatment == 1].y1.mean() - df[df.treatment == 1].y0.mean()
    else:
        gt = np.nan  # non disponibile per NN reale

    naive = df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()

    # trimming opzionale sui propensity scores
    inc_samp = np.logical_and(g > 0.03, g < 0.97)
    df = df[inc_samp]

    nuisance_dict = {
        "y": df["outcome"].values.astype(float),
        "t": df["treatment"].values.astype(float),
        "q0": df["q0"].values.astype(float),
        "q1": df["q1"].values.astype(float),
        "g": df["g"].values.astype(float),
    }
    nuisance_dict["prob_t"] = nuisance_dict["t"].mean()

    estimates = att_estimates(**nuisance_dict, deps=0.0005)

    estimates["ground_truth"] = gt
    estimates["unadjusted_est"] = naive  # sovrascrive per coerenza con trimming

    return estimates


def nn_confounding_level():
    """
    Confronto tra low, med, high confounding per NN.
    """
    estimates = {}
    estimates["low"] = att_from_nn_tsv("../out/PeerRead/nn/beta1/predictions.tsv")
    estimates["med"] = att_from_nn_tsv("../out/PeerRead/nn/beta5/predictions.tsv")
    # estimates["med"] = att_from_nn_tsv("../out/PeerRead/nn_lm/predictions.tsv")
    estimates["high"] = att_from_nn_tsv("../out/PeerRead/nn/beta25/predictions.tsv")

    estimate_df = pd.DataFrame(estimates)
    with tf.io.gfile.GFile("../out/PeerRead/nn/estimates.tsv", "w") as writer:
        writer.write(estimate_df.to_csv(sep="\t"))

    print(estimate_df.round(2))
    return estimate_df


if __name__ == "__main__":
    nn_confounding_level()
