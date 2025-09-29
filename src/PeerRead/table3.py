import os
import pandas as pd
import numpy as np
import tensorflow as tf
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

def build_table3():
    rows = []

    # === Ground truth dal file estimates.tsv di C-BERT ===
    cbert_path = "../out/PeerRead/nn/estimates.tsv"
    if os.path.exists(cbert_path):
        df_gt = pd.read_csv(cbert_path, sep="\t", index_col=0)
        rows.append({
            "method": "Ground truth",
            "low": round(df_gt.loc["ground_truth", "low"], 2),
            "med": round(df_gt.loc["ground_truth", "med"], 2),
            "high": round(df_gt.loc["ground_truth", "high"], 2),
        })

    # === Definizione metodi ===
    models = {
        "Unadjusted": "unadjusted_est",
        "NN ψ^Q": "q_only",
        "NN ψ^plugin": "plugin",
        "C-ATM ψ^Q": "q_only",
        "C-ATM ψ^plugin": "plugin",
        "C-BERT ψ^Q": "q_only",
        "C-BERT ψ^plugin": "plugin",
    }

    base_paths = {
        "NN ψ^Q": "../out/PeerRead/nn",
        "NN ψ^plugin": "../out/PeerRead/nn",
        "C-ATM ψ^Q": "../out/PeerRead/c_atm",
        "C-ATM ψ^plugin": "../out/PeerRead/c_atm",
        "Unadjusted": "../out/PeerRead/nn",
    }

    betas = {"low": 1.0, "med": 5.0, "high": 25.0}

    for method, metric in models.items():
        row = {"method": method}
        if "C-BERT" in method:
            if not os.path.exists('../out/PeerRead/buzzy-based-sim/estimates.tsv'):
                print(f"⚠️ File non trovato: {'../out/PeerRead/buzzy-based-sim/estimates.tsv'}")
                row.update({"low": None, "med": None, "high": None})
            else:
                df = pd.read_csv('../out/PeerRead/buzzy-based-sim/estimates.tsv', sep="\t", index_col=0)
                row["low"] = round(df.loc[metric, "low"], 2)
                row["med"] = round(df.loc[metric, "med"], 2)
                row["high"] = round(df.loc[metric, "high"], 2)
        else:
            for col, beta in betas.items():
                path = os.path.join(base_paths[method], f"beta{int(beta)}/predictions.tsv")
                if not os.path.exists(path):
                    print(f"⚠️ File non trovato: {path}")
                    row[col] = None
                    continue
                est = att_from_tsv(path)
                row[col] = round(est[metric], 2)
        rows.append(row)

    # === DataFrame finale ===
    df = pd.DataFrame(rows).set_index("method")
    print(df)
    df.to_csv("../out/PeerRead/table3_peerread.tsv", sep="\t")

if __name__ == "__main__":
    build_table3()
