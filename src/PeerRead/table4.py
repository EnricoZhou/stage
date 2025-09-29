import os
import glob
import pandas as pd
import numpy as np
from semi_parametric_estimation.att import att_estimates

def att_from_predictions(path, trim=0.03):
    df = pd.read_csv(path, sep="\t")
    y = df["outcome"].astype(float).values
    t = df["treatment"].astype(float).values
    g = df["g"].astype(float).values
    q0 = df["q0"].astype(float).values
    q1 = df["q1"].astype(float).values

    inc = np.logical_and(g > trim, g < 1 - trim)
    df = df[inc]

    nuisance_dict = {
        "y": df["outcome"].values.astype(float),
        "t": df["treatment"].values.astype(float),
        "q0": df["q0"].values.astype(float),
        "q1": df["q1"].values.astype(float),
        "g": df["g"].values.astype(float),
    }
    nuisance_dict["prob_t"] = nuisance_dict["t"].mean()

    return att_estimates(**nuisance_dict, deps=0.0005)

def aggregate_estimates(folder):
    all_est = {"unadjusted_est": [], "q_only": [], "plugin": []}
    for file in glob.glob(os.path.join(folder, "split*/predictions.tsv")):
        est = att_from_predictions(file)
        for k in all_est:
            all_est[k].append(est[k])

    summary = {}
    for k, vals in all_est.items():
        mean = np.mean(vals)
        std = np.std(vals)
        summary[k] = f"{mean:.2f} ± {std:.2f}"
    return summary

def build_table4():
    buzzy_folder = "../out/PeerRead/real/o_accepted_t_buzzy_title"
    theorem_folder = "../out/PeerRead/real/o_accepted_t_theorem_referenced"

    buzzy = aggregate_estimates(buzzy_folder)
    theorem = aggregate_estimates(theorem_folder)

    rows = [
        {"method": "Unadjusted", "buzzy": buzzy["unadjusted_est"], "theorem": theorem["unadjusted_est"]},
        {"method": "C-BERT ψ^Q", "buzzy": buzzy["q_only"], "theorem": theorem["q_only"]},
        {"method": "C-BERT ψ^plugin", "buzzy": buzzy["plugin"], "theorem": theorem["plugin"]},
    ]

    df = pd.DataFrame(rows).set_index("method")
    print(df)
    df.to_csv("../out/PeerRead/table4_peerread.tsv", sep="\t")

if __name__ == "__main__":
    build_table4()