'''from PeerRead.compute_estimates import att_from_bert_tsv

if __name__ == "__main__":
    estimates = att_from_bert_tsv("../output/predictions.tsv", test_split=False)
    print("ATT estimates:")
    for k, v in estimates.items():
        print(f"{k}: {v}")'''

import numpy as np
from PeerRead.compute_estimates import att_from_bert_tsv

def bootstrap_att(tsv_path, B=100, test_split=False, trim=0.0):
    estimates_list = []

    for b in range(B):
        est = att_from_bert_tsv(tsv_path, test_split=test_split, trim=trim)
        estimates_list.append(est)

    # Raggruppa per chiave
    keys = estimates_list[0].keys()
    results = {}
    for k in keys:
        vals = [e[k] for e in estimates_list]
        results[k] = (np.mean(vals), np.std(vals))  # media e deviazione standard

    return results

if __name__ == "__main__":
    res = bootstrap_att("../output/predictions_with_masking.tsv", B=100, test_split=False)
    print("ATT bootstrap estimates (mean ± std):")
    for k, (mean, std) in res.items():
        print(f"{k}: {mean:.3f} ± {std:.3f}")
