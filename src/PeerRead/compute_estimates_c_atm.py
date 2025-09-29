import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

import tensorflow as tf

from semi_parametric_estimation.att import att_estimates

# AGGIUNTO
pd.set_option('display.max_colwidth', None)
# MODIFICATO
# pd.set_option('display.max_colwidth', -1)


def att_from_bert_tsv(tsv_path, test_split=True, trim=0.0):
    predictions = pd.read_csv(tsv_path, sep='\t')

    if test_split:
        reduced_df = predictions[predictions.in_test == 1]
    else:
        reduced_df = predictions[predictions.in_train == 1]

    gt = reduced_df[reduced_df.treatment == 1].y1.mean() - reduced_df[reduced_df.treatment == 1].y0.mean()
    # print(f"Ground truth: {gt}")

    naive = reduced_df[reduced_df.treatment == 1].outcome.mean() - reduced_df[reduced_df.treatment == 0].outcome.mean()
    # print(f"Naive: {naive}")

    selections = {'y': 'outcome',
                  't': 'treatment',
                  'q0': 'q0',
                  'q1': 'q1',
                  'g': 'g'}

    reduced_df = reduced_df[selections.values()]
    rename_dict = {v: k for k, v in selections.items()}
    reduced_df = reduced_df.rename(columns=rename_dict)

    inc_samp = np.logical_and(reduced_df['g'] > trim, reduced_df['g'] < 1 - trim)
    reduced_df = reduced_df[inc_samp]

    # get rid of any sample w/ less than 1% chance of receiving treatment
    # include_sample = reduced_df['g'] > 0.01
    # reduced_df = reduced_df[include_sample]

    nuisance_dict = reduced_df.to_dict('series')
    nuisance_dict['prob_t'] = nuisance_dict['t'].mean()
    estimates = att_estimates(**nuisance_dict, deps=0.0005)

    estimates['ground_truth'] = gt
    estimates['unadjusted_est'] = naive  # overwrite because trimming will screw this up
    # estimates['naive'] = naive

    return estimates

def att_from_atm_tsv(path, deps=0.001, trim=0.03):
    """
    Carica un file predictions_atm.tsv (colonne: outcome, treatment, g, q0, q1)
    e calcola le stime ATT con trimming sui propensity scores (g ∈ [0.03, 0.97]).
    """
    if not os.path.exists(path):
        print(f"⚠️ File non trovato: {path}")
        return None

    df = pd.read_csv(path, sep="\t")
    if df.empty:
        print(f"⚠️ File vuoto: {path}")
        return None

    required_cols = {"outcome", "treatment", "g", "q0", "q1"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ Colonne mancanti in {path}. Trovate: {df.columns}")
        return None

    y = df["outcome"].values.astype(float)
    t = df["treatment"].values.astype(int)
    g = df["g"].values.astype(float)
    q0 = df["q0"].values.astype(float)
    q1 = df["q1"].values.astype(float)

    # ✅ trimming come nel paper
    mask = (g >= trim) & (g <= 1 - trim)
    if mask.sum() == 0:
        print(f"⚠️ Nessun esempio valido dopo trimming in {path}")
        return None

    y, t, g, q0, q1 = y[mask], t[mask], g[mask], q0[mask], q1[mask]

    prob_t = t.mean()
    return att_estimates(q0, q1, g, t, y, prob_t, deps=deps)


def dragon_att(output_dir, deps=0.001, trim=0.03, test_split=True, trim_test=False):
    """
    Media delle stime ATT per tutti i file TSV in una cartella,
    con trimming integrato (g ∈ [0.03, 0.97]).
    """
    data_files = sorted(glob.glob(f'{output_dir}/*.tsv', recursive=True))
    estimates = []
    all_estimates = None

    for data_file in data_files:
        all_estimates = att_from_atm_tsv(data_file, deps=deps, trim=trim)
        if all_estimates is None:
            print(f"❌ Skipping file: {data_file}")
            continue
        estimates.append(all_estimates)

    if not estimates:
        print("⚠️ Nessuna stima valida trovata")
        return None

    avg_estimates = {}
    for k in estimates[0].keys():
        k_estimates = [est[k] for est in estimates if est is not None]
        if not k_estimates:
            continue
        if trim_test and len(k_estimates) > 2:
            k_estimates = np.sort(k_estimates)[1:-1]  # scarta estremi
        avg_estimates[k] = np.mean(k_estimates)
        avg_estimates[(k, 'std')] = np.std(k_estimates)
        if test_split:
            avg_estimates[(k, 'std')] /= np.sqrt(len(k_estimates))

    return avg_estimates


def confounding_level():
    # Comparison over compounding strength
    estimates = {}
    estimates['low'] = dragon_att('../out/PeerRead/c_atm/beta1')
    estimates['med'] = dragon_att('../out/PeerRead/c_atm/beta5')
    estimates['high'] = dragon_att('../out/PeerRead/c_atm/beta25')

    print("DEBUG estimates:", estimates)  # 👈 per vedere cosa c’è dentro


    estimate_df = pd.DataFrame(estimates)
    with tf.io.gfile.GFile('../out/PeerRead/c_atm/estimates.tsv', "w") as writer:
        writer.write(estimate_df.to_csv(sep="\t"))

    print(estimate_df.round(2))


def buzzy_baselines():
    base_dir = '../out/PeerRead/buzzy-baselines/'
    out_file = 'modesimple/beta00.25.beta15.0.gamma0.0'

    estimates = {}
    estimates['baseline'] = dragon_att(os.path.join(base_dir, 'buzzy', out_file))
    estimates['fixed_features'] = dragon_att(os.path.join(base_dir, 'fixed-features', out_file))
    estimates['no_pretrain'] = dragon_att(os.path.join(base_dir, 'no-init', out_file))
    estimates['no_masking'] = dragon_att(os.path.join(base_dir, 'no-masking', out_file))
    estimates['no_dragon'] = dragon_att(os.path.join(base_dir, 'no-dragon', out_file))

    estimate_df = pd.DataFrame(estimates)
    print(estimate_df.round(2))

    return estimate_df


def real():
    estimates = {}
    estimates['buzzy'] = dragon_att('../out/PeerRead/real/o_accepted_t_buzzy_title')
    estimates['theorem_referenced'] = dragon_att('../out/PeerRead/real/o_accepted_t_theorem_referenced')
    estimate_df = pd.DataFrame(estimates)
    print(estimate_df.round(2))

    return estimate_df


if __name__ == '__main__':
    estimates = confounding_level()
    # estimates = real()
    # estimates = buzzy_baselines()
