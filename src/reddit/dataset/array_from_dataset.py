"""
helpers to take samples from the dataset and turn them into numpy arrays
(for ease of inspection and use with baselines)

la Tabella 2 che hai postato viene proprio da esperimenti con i parametri di simulazione:
β1 (beta1) = livello di confounding (quanto un “confondente” nascosto influenza sia trattamento che outcome).
Low, Med, High nella tabella corrispondono a β1 = 1.0, 10.0, 100.0.
γ (gamma) = livello di noise (quanto rumore casuale viene aggiunto).
La tabella mostra due blocchi: γ = 1.0 e γ = 4.0.
β0 (beta0) = forza del trattamento, che nei tuoi script compare nei parametri di simulazione.
In questa tabella sembra fissato (ad esempio a β0 = 10.0), mentre vengono variati β1 e γ.

🔹 Come vengono usati

Lo script array_from_dataset.py (funzione subreddit_based_sim_dfs) prende come input --beta0, --beta1, --gamma.
Questi valori controllano il labeler simulato → cioè come vengono generati i valori di treatment, y0, y1.
Cambiando questi parametri ottieni dataset con diversi livelli di difficoltà (più o meno confondimento, più o meno rumore).
Addestri i modelli (NN, C-ATM, C-BERT…) su questi dataset.
Calcoli gli stimatori (ψ^Q e ψ^plugin) → riempi le celle della tabella.

Per ricreare i valori della Tabella 2 devi:
fissare β0 (ad es. 10.0),
variare β1 = 1.0, 10.0, 100.0,
variare γ = 1.0, 4.0,
eseguire i modelli e raccogliere i risultati.
"""
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
try:
    # sono se il computer supporta MKL, il mac preferisce utilizzare numpy
    import mkl_random as random
except ImportError:
    # migliore per mac M2
    import numpy.random as random

from bert import tokenization
from bert.tokenization import FullTokenizer
from reddit.dataset.dataset import make_input_fn_from_file, make_subreddit_based_simulated_labeler


def dataset_fn_to_df(dataset_fn):

    params = {'batch_size': 1}
    dataset = dataset_fn(params)

    itr = dataset.make_one_shot_iterator()

    samples = []

    for i in range(250000):
        try:
            sample = itr.get_next()
            for k in sample:
                sample[k] = sample[k].numpy()[0]
            samples += [sample]
            # print("year: {}".format(sample['year']))
        except:
            print(i)
            break

    df = pd.DataFrame(samples)

    return df


def subreddit_based_sim_dfs(subreddits, treat_strength, con_strength, noise_level, setting="simple", seed=0,
                            base_output_dir='../dat/sim/reddit_subreddit_based/'):

    labeler = make_subreddit_based_simulated_labeler(treat_strength, con_strength, noise_level, setting=setting, seed=seed)

    num_splits = 10
    dev_splits = [0]
    test_splits = [0]

    # data_file = '../dat/reddit/proc.tf_record'
    # vocab_file = "../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    input_dataset_from_filenames = make_input_fn_from_file(data_file,
                                                           250,
                                                           num_splits,
                                                           dev_splits,
                                                           test_splits,
                                                           tokenizer,
                                                           subreddits=subreddits,
                                                           is_training=False,
                                                           filter_test=False,
                                                           shuffle_buffer_size=25000,
                                                           seed=seed,
                                                           labeler=labeler)

    all_data = dataset_fn_to_df(input_dataset_from_filenames)
    output_df = all_data[['index', 'gender','outcome', 'y0', 'y1']]
    output_df = output_df.rename(index=str, columns={'gender': 'treatment'})

    output_dir = os.path.join(base_output_dir, "subreddits{}".format(subreddits), "mode{}".format(setting))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "beta0{}.beta1{}.gamma{}.tsv".format(treat_strength, con_strength, noise_level))

    output_df.to_csv(output_path, '\t')


def main():
    tf.enable_eager_execution()


    subreddit_based_sim_dfs(subreddits=subs, treat_strength=beta0, con_strength=beta1, noise_level=gamma, setting=mode, seed=0,
                            base_output_dir=base_output_dir)



    # print(itr.get_next()["token_ids"].name)
    # for i in range(1000):
    #     sample = itr.get_next()

    #
    # print(np.unique(df['year']))
    # print(df.groupby(['year'])['buzzy_title'].agg(np.mean))
    # print(df.groupby(['year'])['theorem_referenced'].agg(np.mean))
    # print(df.groupby(['year'])['accepted'].agg(np.mean))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", action="store", default='../dat/reddit/proc.tf_record')
    parser.add_argument("--vocab-file", action="store", default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument("--base-output-dir", action="store", default='../dat/sim/reddit_subreddit_based/')
    parser.add_argument("--subs", action="store", default='13,8,6')
    parser.add_argument("--mode", action="store", default="simple")
    parser.add_argument("--beta0", action="store", default='1.0')
    parser.add_argument("--beta1", action="store", default='1.0')
    parser.add_argument("--gamma", action="store", default='1.0')
    args = parser.parse_args()

    data_file = args.data_file
    vocab_file = args.vocab_file
    base_output_dir = args.base_output_dir
    subs = [int(s) for s in args.subs.split(',')]
    mode = args.mode
    beta0 = float(args.beta0)
    beta1 = float(args.beta1)
    gamma = float(args.gamma)

    # pass
    main()