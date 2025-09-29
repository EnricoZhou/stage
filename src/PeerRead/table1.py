import os
import pandas as pd
import numpy as np
from semi_parametric_estimation.att import att_estimates

def att_from_tsv(tsv_path, trim=0.03):
    df = pd.read_csv(tsv_path, sep="\t")

    y = df["outcome"].values.astype(float)
    t = df["treatment"].values.astype(float)
    g = df["g"].values.astype(float)
    q0 = df["q0"].values.astype(float)
    q1 = df["q1"].values.astype(float)

    if "y0" in df.columns and "y1" in df.columns:
        gt = df[df.treatment == 1].y1.mean() - df[df.treatment == 1].y0.mean()
    else:
        gt = np.nan

    naive = df[df.treatment == 1].outcome.mean() - df[df.treatment == 0].outcome.mean()

    inc_samp = np.logical_and(g > trim, g < 1 - trim)
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
    estimates["unadjusted_est"] = naive

    return estimates


def build_table1():
    # ----------- (a) Language Modeling Helps -----------
    rows_a = [
        {"method": "Ground truth", "PeerRead (ATT)": 0.06},
        {"method": "Unadjusted", "PeerRead (ATT)": 0.14},
    ]

    models_a = {
        "NN ψ^Q": ("../out/PeerRead/nn/beta5/predictions.tsv", "q_only"),
        "NN ψ^plugin": ("../out/PeerRead/nn/beta5/predictions.tsv", "plugin"),
        "BERT (sup. only) ψ^Q": ("../out/PeerRead/bert_sup/predictions.tsv", "q_only"),
        "BERT (sup. only) ψ^plugin": ("../out/PeerRead/bert_sup/predictions.tsv", "plugin"),
        "C-ATM ψ^Q": ("../out/PeerRead/c_atm/beta5/predictions.tsv", "q_only"),
        "C-ATM ψ^plugin": ("../out/PeerRead/c_atm/beta5/predictions.tsv", "plugin"),
        # C-BERT gestito separatamente: prende i dati da estimates.tsv
        # "C-BERT ψ^Q": ("../out/PeerRead/c_bert/predictions.tsv", "q_only"),
        # "C-BERT ψ^plugin": ("../out/PeerRead/c_bert/predictions.tsv", "plugin"),
    }

    for method, (path, metric) in models_a.items():
        if os.path.exists(path):
            est = att_from_tsv(path, trim=0.03)
            rows_a.append({"method": method, "PeerRead (ATT)": round(est[metric], 2)})
        else:
            rows_a.append({"method": method, "PeerRead (ATT)": None})

    # --- C-BERT da estimates.tsv ---
    cbert_path = "../out/PeerRead/buzzy-based-sim/estimates.tsv"
    if os.path.exists(cbert_path):
        df_cbert = pd.read_csv(cbert_path, sep="\t", index_col=0)
        for metric in ["q_only", "plugin"]:
            rows_a.append({
                "method": f"C-BERT ψ^{metric.replace('_','')}",
                "PeerRead (ATT)": round(df_cbert.loc[metric, "med"], 2)
            })

    df_a = pd.DataFrame(rows_a).set_index("method")
    print("\n=== (a) Language Modeling Helps ===")
    print(df_a)
    df_a.to_csv("../out/PeerRead/table1a_peerread.tsv", sep="\t")

    # ----------- (b) Supervision Helps -----------
    rows_b = [
        {"method": "Ground truth", "PeerRead (ATT)": 0.06},
        {"method": "Unadjusted", "PeerRead (ATT)": 0.14},
    ]

    models_b = {
        "BOW ψ^Q": ("../out/PeerRead/bow/predictions.tsv", "q_only"),
        "BOW ψ^plugin": ("../out/PeerRead/bow/predictions.tsv", "plugin"),
        "BERT ψ^Q": ("../out/PeerRead/bert/predictions.tsv", "q_only"),
        "BERT ψ^plugin": ("../out/PeerRead/bert/predictions.tsv", "plugin"),
        "LDA ψ^Q": ("../out/PeerRead/lda/predictions.tsv", "q_only"),
        "LDA ψ^plugin": ("../out/PeerRead/lda/predictions.tsv", "plugin"),
        "ATM ψ^Q": ("../out/PeerRead/atm/predictions.tsv", "q_only"),
        "ATM ψ^plugin": ("../out/PeerRead/atm/predictions.tsv", "plugin"),
    }

    for method, (path, metric) in models_b.items():
        if os.path.exists(path):
            est = att_from_tsv(path, trim=0.03)
            rows_b.append({"method": method, "PeerRead (ATT)": round(est[metric], 2)})
        else:
            rows_b.append({"method": method, "PeerRead (ATT)": None})

    df_b = pd.DataFrame(rows_b).set_index("method")
    print("\n=== (b) Supervision Helps ===")
    print(df_b)
    df_b.to_csv("../out/PeerRead/table1b_peerread.tsv", sep="\t")


if __name__ == "__main__":
    build_table1()
